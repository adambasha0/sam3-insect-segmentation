import os
import json
import numpy as np
import cv2
import gc
from collections import defaultdict

# ==========================
# CONFIGURATION
# ==========================
ROOT_DATASET = "./flatbug-dataset"
# No DEVICE needed for CPU script

# Filenames
PRED_FILENAME = "sam3_flatbug_strategy.json"
GT_FILENAME = "instances_default.json"

# Datasets to analyze (The OOM ones)
ALLOWED_FOLDERS = {
    "sticky-pi",
    "abram2023", 
    "pinoy2023", 
    "CollembolAI",
    "ubc-scanned-sticky-cards",
    "ubc-pitfall-traps",
    #"PeMaToEuroPep",
    "Mothitor", 
}

# Thresholds to sweep
THRESHOLDS = np.arange(0.35, 0.96, 0.05)
IOU_THRESHOLD = 0.5

# ==========================
# HELPER FUNCTIONS (CPU)
# ==========================
def seg_to_mask(segmentation, height, width):
    """Robustly converts COCO segmentation to binary mask."""
    if not segmentation: return None
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Handle Polygon format (list of lists)
    if isinstance(segmentation, list):
        for poly in segmentation:
            if not poly: continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
    # Handle RLE format (dict)
    elif isinstance(segmentation, dict):
        from pycocotools import mask as mask_utils
        rle = mask_utils.frPyObjects(segmentation, height, width)
        mask = mask_utils.decode(rle)
        if len(mask.shape) == 3: mask = mask[:, :, 0] # flattening if needed
        
    return mask

def mask_iou_cpu(mask1_np, mask2_np):
    """Calculates IoU on CPU using NumPy (No Torch/CUDA)."""
    # Use boolean arrays for speed and lower memory
    m1 = mask1_np.astype(bool)
    m2 = mask2_np.astype(bool)
    
    # Bitwise operations are fast on CPU
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    iou = (intersection / union) if union > 0 else 0.0
    return iou

def load_json_by_filename(json_path):
    """
    Loads COCO JSON and re-maps annotations to act like:
    dict[filename] -> list_of_annotations
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # 1. Map ID -> Filename
    id_to_file = {}
    img_list = data.get('images', [])
    if isinstance(img_list, dict):
        img_list = img_list.values()
        
    for img in img_list:
        id_to_file[img['id']] = (img['file_name'], img['height'], img['width'])
        
    # 2. Group Annotations by Filename
    anns_by_file = defaultdict(list)
    
    ann_list = data.get('annotations', [])
    if isinstance(ann_list, dict):
        ann_list = ann_list.values()
        
    for ann in ann_list:
        if ann['image_id'] not in id_to_file: continue
        fname, h, w = id_to_file[ann['image_id']]
        ann['__height'] = h
        ann['__width'] = w
        anns_by_file[fname].append(ann)
        
    return anns_by_file

# ==========================
# MAIN TUNER LOGIC
# ==========================
print(f"{'DATASET (CPU MODE)':<25} | {'BEST THRESH':<11} | {'F1':<6} | {'PREC':<6} | {'RECALL':<6}")
print("-" * 75)

global_best_thresh_accum = []

for dataset_name in sorted(os.listdir(ROOT_DATASET)):
    if dataset_name not in ALLOWED_FOLDERS:
        continue

    dataset_path = os.path.join(ROOT_DATASET, dataset_name)
    pred_path = os.path.join(dataset_path, PRED_FILENAME)
    
    # Check GT in root or annotations subdir
    gt_path = os.path.join(dataset_path, GT_FILENAME)
    if not os.path.exists(gt_path):
        gt_path = os.path.join(dataset_path, "annotations", GT_FILENAME)
    
    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        continue

    try:
        # 1. Load Data
        gt_by_file = load_json_by_filename(gt_path)
        pred_by_file = load_json_by_filename(pred_path)
        
        # 2. MATCHING PHASE
        all_pred_results = [] # Stores {'score': float, 'is_tp': bool}
        total_gt_count = 0
        
        for filename, gt_anns in gt_by_file.items():
            # Get preds for this file
            pred_anns = pred_by_file.get(filename, [])
            total_gt_count += len(gt_anns)
            
            if not pred_anns: continue
            
            # Prepare Masks
            h, w = gt_anns[0]['__height'], gt_anns[0]['__width']
            
            gt_masks = [seg_to_mask(a['segmentation'], h, w) for a in gt_anns]
            pred_masks = []
            valid_preds = []
            
            # Filter invalid masks early
            for p in pred_anns:
                m = seg_to_mask(p['segmentation'], h, w)
                if m is not None:
                    pred_masks.append(m)
                    valid_preds.append(p)
            
            if not gt_masks or not pred_masks:
                for p in valid_preds:
                    all_pred_results.append({'score': p['score'], 'is_tp': False})
                continue

            # Greedy Matching Strategy
            sorted_indices = sorted(range(len(valid_preds)), 
                                  key=lambda i: valid_preds[i]['score'], 
                                  reverse=True)
            
            matched_gt_indices = set()
            
            for idx in sorted_indices:
                p_mask = pred_masks[idx]
                p_score = valid_preds[idx]['score']
                
                best_iou = 0
                best_gt_idx = -1
                
                # Try to find a match among unmatched GTs
                for gt_idx, g_mask in enumerate(gt_masks):
                    if gt_idx in matched_gt_indices: continue
                    
                    # UPDATED: Use CPU IoU function
                    iou = mask_iou_cpu(p_mask, g_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Verdict
                if best_iou >= IOU_THRESHOLD:
                    matched_gt_indices.add(best_gt_idx)
                    all_pred_results.append({'score': p_score, 'is_tp': True})
                else:
                    all_pred_results.append({'score': p_score, 'is_tp': False})
            
            # Clear memory explicitly
            del gt_masks, pred_masks
            gc.collect()

        # 3. TUNING PHASE
        best_f1 = -1
        best_t = 0
        best_p = 0
        best_r = 0
        
        for t in THRESHOLDS:
            active_preds = [p for p in all_pred_results if p['score'] >= t]
            
            tp = sum(1 for p in active_preds if p['is_tp'])
            fp = sum(1 for p in active_preds if not p['is_tp'])
            fn = total_gt_count - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                best_p = precision
                best_r = recall

        print(f"{dataset_name:<25} | {best_t:.2f}        | {best_f1:.2f}   | {best_p:.2f}   | {best_r:.2f}")
        
        if best_f1 > 0:
            global_best_thresh_accum.append(best_t)

    except Exception as e:
        print(f"{dataset_name:<25} | ERROR: {str(e)[:30]}")
        continue

print("-" * 75)
if global_best_thresh_accum:
    avg_thresh = sum(global_best_thresh_accum) / len(global_best_thresh_accum)
    print(f"\nRECOMMENDED GLOBAL THRESHOLD: {avg_thresh:.2f}")
else:
    print("\nNo valid datasets found.")