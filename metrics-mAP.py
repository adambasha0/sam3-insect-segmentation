import os
import json
import torch
import numpy as np
import cv2
import contextlib
from collections import defaultdict

# ==========================
# CONFIGURATION
# ==========================
ROOT_DATASET = "./flatbug-dataset"
PRED_FILENAME = "sam3_flatbug_strategy.json"
GT_FILENAME = "instances_default.json"
DEVICE = torch.device("cuda:1") 

# Exclude OOM datasets
ALLOWED_FOLDERS = {
    # "NHM-beetles-crops", "cao2022", "gernat2018",
    # "amarathunga2022", "biodiscover-arm", "Mothitor", "DIRT", "Diopsis",
    # "AMT", "anTraX", "ALUS", "BIOSCAN", "DiversityScanner", "ArTaxOr",
    "ubc-pitfall-traps", "ubc-scanned-sticky-cards",
    "sittinger2023", "pinoy2023", "sticky-pi", 
}

IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10).to(DEVICE)

# ==========================
# HELPER FUNCTIONS
# ==========================
def seg_to_mask(segmentation, height, width):
    if not segmentation: return None
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, list):
        for poly in segmentation:
            if not poly: continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
    elif isinstance(segmentation, dict):
        from pycocotools import mask as mask_utils
        rle = mask_utils.frPyObjects(segmentation, height, width)
        mask = mask_utils.decode(rle)
        if len(mask.shape) == 3: mask = mask[:, :, 0]
    return mask

def box_iou_gpu(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union

def mask_iou_gpu(masks1, masks2):
    m1 = masks1.view(masks1.shape[0], -1)
    m2 = masks2.view(masks2.shape[0], -1)
    inter = torch.mm(m1.float(), m2.float().t())
    area1 = m1.sum(dim=1).unsqueeze(1)
    area2 = m2.sum(dim=1).unsqueeze(0)
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

def compute_metrics_from_matches(tp, num_gt):
    """Calculates AP and Recall."""
    if num_gt == 0: return 0.0, 0.0
    
    tp_cumsum = torch.cumsum(tp, dim=0).float()
    fp_cumsum = torch.cumsum(~tp, dim=0).float()
    
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # 1. Calculate Max Recall (AR proxy)
    # The highest recall value achieved at any confidence level
    max_recall = torch.max(recalls).item() if len(recalls) > 0 else 0.0
    
    # 2. Calculate AP (Interpolated)
    precisions = torch.cat((torch.tensor([0.0], device=DEVICE), precisions, torch.tensor([0.0], device=DEVICE)))
    recalls = torch.cat((torch.tensor([0.0], device=DEVICE), recalls, torch.tensor([1.0], device=DEVICE)))
    
    for i in range(precisions.shape[0] - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])
        
    indices = torch.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = torch.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap.item(), max_recall

def evaluate_dataset_gpu(gt_path, pred_path):
    # 1. Load & Standardize
    with open(gt_path) as f: gt_data = json.load(f)
    with open(pred_path) as f: pred_data = json.load(f)
    
    gt_by_file = defaultdict(list)
    img_map = {i['id']: i['file_name'] for i in gt_data['images']}
    img_dims = {i['file_name']: (i['height'], i['width']) for i in gt_data['images']}
    
    ann_list = gt_data.get('annotations', [])
    if isinstance(ann_list, dict): ann_list = list(ann_list.values())
    for ann in ann_list:
        fname = img_map.get(ann['image_id'])
        if fname: gt_by_file[fname].append(ann)
            
    pred_by_file = defaultdict(list)
    p_anns = pred_data.get('annotations', [])
    if isinstance(p_anns, dict): p_anns = list(p_anns.values())
    p_img_map = {i['id']: i['file_name'] for i in pred_data.get('images', [])}
    for ann in p_anns:
        fname = p_img_map.get(ann['image_id'])
        if not fname: fname = img_map.get(ann['image_id'])
        if fname: pred_by_file[fname].append(ann)

    # 2. Process
    all_preds_box = [] 
    all_preds_mask = []
    total_gt = 0
    
    for fname, gt_anns in gt_by_file.items():
        total_gt += len(gt_anns)
        pred_anns = pred_by_file.get(fname, [])
        if not pred_anns: continue
        
        h, w = img_dims[fname]
        pred_anns.sort(key=lambda x: x['score'], reverse=True)
        scores = torch.tensor([p['score'] for p in pred_anns], device=DEVICE)
        
        p_boxes = torch.tensor([p['bbox'] for p in pred_anns], device=DEVICE)
        p_boxes[:, 2:] += p_boxes[:, :2] 
        g_boxes = torch.tensor([g['bbox'] for g in gt_anns], device=DEVICE)
        g_boxes[:, 2:] += g_boxes[:, :2]

        p_masks_np = np.array([seg_to_mask(p['segmentation'], h, w) for p in pred_anns])
        g_masks_np = np.array([seg_to_mask(g['segmentation'], h, w) for g in gt_anns])
        p_masks = torch.from_numpy(p_masks_np).to(DEVICE, dtype=torch.bool)
        g_masks = torch.from_numpy(g_masks_np).to(DEVICE, dtype=torch.bool)

        iou_box = box_iou_gpu(p_boxes, g_boxes)
        iou_mask = mask_iou_gpu(p_masks, g_masks)

        def match_for_thresholds(iou_matrix):
            num_p, num_g = iou_matrix.shape
            tp_matrix = torch.zeros((num_p, 10), dtype=torch.bool, device=DEVICE)
            for t_idx, thresh in enumerate(IOU_THRESHOLDS):
                iou_t = iou_matrix.clone()
                for p_idx in range(num_p):
                    max_val, max_gt_idx = torch.max(iou_t[p_idx], dim=0)
                    if max_val >= thresh:
                        tp_matrix[p_idx, t_idx] = True
                        iou_t[:, max_gt_idx] = -1.0
            return tp_matrix

        all_preds_box.append((scores, match_for_thresholds(iou_box)))
        all_preds_mask.append((scores, match_for_thresholds(iou_mask)))
        
        del p_masks, g_masks, iou_box, iou_mask
        torch.cuda.empty_cache()

    if total_gt == 0 or not all_preds_box: return [0.0]*6

    # 3. Global Stats
    cat_scores = torch.cat([x[0] for x in all_preds_box])
    cat_tp_box = torch.cat([x[1] for x in all_preds_box], dim=0)
    cat_tp_mask = torch.cat([x[1] for x in all_preds_mask], dim=0)
    
    sort_idx = torch.argsort(cat_scores, descending=True)
    sorted_tp_box = cat_tp_box[sort_idx]
    sorted_tp_mask = cat_tp_mask[sort_idx]
    
    # Calculate Metrics per Threshold
    box_metrics = [compute_metrics_from_matches(sorted_tp_box[:, i], total_gt) for i in range(10)]
    mask_metrics = [compute_metrics_from_matches(sorted_tp_mask[:, i], total_gt) for i in range(10)]
    
    # Extract Specifics
    # AP50 (Index 0)
    b_ap50, b_recall50 = box_metrics[0]
    m_ap50, m_recall50 = mask_metrics[0]
    
    # mAP (Average AP across 10 thresholds)
    b_map = sum([x[0] for x in box_metrics]) / 10.0
    m_map = sum([x[0] for x in mask_metrics]) / 10.0
    
    # Average Recall (Average of max recall across 10 thresholds - Standard COCO definition)
    b_ar = sum([x[1] for x in box_metrics]) / 10.0
    m_ar = sum([x[1] for x in mask_metrics]) / 10.0
    
    return b_map, b_ap50, b_ar, m_map, m_ap50, m_ar

# ==========================
# MAIN LOOP
# ==========================
print(f"{'DATASET':<20} | {'mAP(Box)':<8} | {'AP50(Box)':<9} | {'AR(Box)':<8} || {'mAP(Seg)':<8} | {'AP50(Seg)':<9} | {'AR(Seg)':<8}")
print("-" * 100)

for dataset_name in sorted(os.listdir(ROOT_DATASET)):
    if dataset_name not in ALLOWED_FOLDERS: continue
    
    dataset_path = os.path.join(ROOT_DATASET, dataset_name)
    gt_path = os.path.join(dataset_path, GT_FILENAME)
    if not os.path.exists(gt_path): gt_path = os.path.join(dataset_path, "annotations", GT_FILENAME)
    pred_path = os.path.join(dataset_path, PRED_FILENAME)

    if not os.path.exists(gt_path) or not os.path.exists(pred_path): continue

    try:
        metrics = evaluate_dataset_gpu(gt_path, pred_path)
        # Unpack: b_map, b_ap50, b_ar, m_map, m_ap50, m_ar
        print(f"{dataset_name:<20} | {metrics[0]:.3f}    | {metrics[1]:.3f}     | {metrics[2]:.3f}    || {metrics[3]:.3f}    | {metrics[4]:.3f}     | {metrics[5]:.3f}")
    except Exception as e:
        print(f"{dataset_name:<20} | ERROR: {str(e)[:30]}")
        torch.cuda.empty_cache()