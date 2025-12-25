import os
import json
import numpy as np
import torch
import cv2
import gc
from collections import defaultdict

# ==========================
# CONFIGURATION
# ==========================
ROOT_DATASET = "./flatbug-dataset"
DEVICE = torch.device("cuda:0") # Adjust index if needed

# Filenames
PRED_FILENAME = "sam3_flatbug_strategy.json"
GT_FILENAME = "instances_default.json"

# Datasets to analyze
ALLOWED_FOLDERS = {
    #"NHM-beetles-crops",
    #"cao2022", "gernat2018", "sittinger2023",
    #"amarathunga2022", "biodiscover-arm", "Mothitor", "DIRT", "Diopsis",
    #"AMI-traps", "AMT", "PeMaToEuroPep", "anTraX", "ALUS", "BIOSCAN",
    #"DiversityScanner", "ArTaxOr",
    "sticky-pi",
    "abram2023", "pinoy2023", "CollembolAI",
    "ubc-pitfall-traps", "ubc-scanned-sticky-cards",
}

# Thresholds to sweep
THRESHOLDS = np.arange(0.35, 0.96, 0.05)
IOU_THRESHOLD = 0.5

# ==========================
# HELPER FUNCTIONS
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

def mask_iou_gpu(mask1_np, mask2_np):
    """Calculates IoU on GPU."""
    m1 = torch.from_numpy(mask1_np).to(DEVICE, dtype=torch.bool)
    m2 = torch.from_numpy(mask2_np).to(DEVICE, dtype=torch.bool)
    
    inter = torch.logical_and(m1, m2).sum()
    union = torch.logical_or(m1, m2).sum()
    
    iou = (inter / union).item() if union > 0 else 0.0
    
    # Clean up immediately
    del m1, m2
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
    img_list = data['images']
    # Handle if images is a dict (user specific case) or list (standard)
    if isinstance(img_list, dict):
        img_list = img_list.values()
        
    for img in img_list:
        id_to_file[img['id']] = (img['file_name'], img['height'], img['width'])
        
    # 2. Group Annotations by Filename
    anns_by_file = defaultdict(list)
    
    # Handle if annotations is dict or list
    ann_list = data['annotations']
    if isinstance(ann_list, dict):
        ann_list = ann_list.values()
        
    for ann in ann_list:
        if ann['image_id'] not in id_to_file: continue
        fname, h, w = id_to_file[ann['image_id']]
        # Inject dimensions into annotation for convenience
        ann['__height'] = h
        ann['__width'] = w
        anns_by_file[fname].append(ann)
        
    return anns_by_file

# ==========================
# MAIN TUNER LOGIC
# ==========================
print(f"{'DATASET':<25} | {'BEST THRESH':<11} | {'F1':<6} | {'PREC':<6} | {'RECALL':<6}")
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
        # print(f"Skipping {dataset_name} (Missing files)")
        continue

    try:
        # 1. Load Data
        gt_by_file = load_json_by_filename(gt_path)
        pred_by_file = load_json_by_filename(pred_path)
        
        # 2. MATCHING PHASE (The Heavy Lifting)
        # We calculate matches ONCE using the most permissive threshold (0.0)
        # We store the result: "This prediction (Score 0.98) matched a GT? Yes/No"
        
        all_pred_results = [] # Stores {'score': float, 'is_tp': bool}
        total_gt_count = 0
        
        for filename, gt_anns in gt_by_file.items():
            # Get preds for this file
            pred_anns = pred_by_file.get(filename, [])
            
            # Count GTs
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
                # If no GTs but we have preds -> All FPs
                for p in valid_preds:
                    all_pred_results.append({'score': p['score'], 'is_tp': False})
                continue

            # Greedy Matching Strategy
            # 1. Sort predictions by score (High confidence gets first dibs)
            # 2. Compute IoU with all GTs
            # 3. Assign match
            
            # Combine score and index for sorting
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
                    
                    iou = mask_iou_gpu(p_mask, g_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Verdict
                if best_iou >= IOU_THRESHOLD:
                    matched_gt_indices.add(best_gt_idx)
                    all_pred_results.append({'score': p_score, 'is_tp': True})
                else:
                    all_pred_results.append({'score': p_score, 'is_tp': False})
            
            # Clean GPU memory per image
            torch.cuda.empty_cache()

        # 3. TUNING PHASE (The Math Sweep)
        # Now we just iterate thresholds and count True/False based on the list we built
        
        best_f1 = -1
        best_t = 0
        best_p = 0
        best_r = 0
        
        for t in THRESHOLDS:
            # Filter: Keep only predictions above threshold t
            active_preds = [p for p in all_pred_results if p['score'] >= t]
            
            tp = sum(1 for p in active_preds if p['is_tp'])
            fp = sum(1 for p in active_preds if not p['is_tp'])
            
            # FN = Total GTs - TPs found
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