import os
import json
import contextlib
import io
import copy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==========================
# CONFIGURATION
# ==========================
ROOT_DATASET = "./flatbug-dataset"
PRED_FILENAME = "sam3_flatbug_strategy.json"
GT_FILENAME = "instances_default.json"

# ONLY the datasets that crashed the GPU script
ALLOWED_FOLDERS = {
    "CollembolAI", "AMI-traps", "abram2023", "PeMaToEuroPep", 
    # "ubc-scanned-sticky-cards", "sittinger2023", "pinoy2023",
    # "sticky-pi", "ubc-pitfall-traps",
}

# ==========================
# HELPER: ID ALIGNMENT
# ==========================
def align_coco_ids(gt_data, pred_data):
    """Rewrites image_ids to match based on filename."""
    filename_to_id = {}
    next_id = 1
    
    # 1. Map Filenames to IDs
    gt_images = gt_data.get('images', [])
    if isinstance(gt_images, dict): gt_images = list(gt_images.values())
    
    for img in gt_images:
        fname = os.path.basename(img['file_name'])
        if fname not in filename_to_id:
            filename_to_id[fname] = next_id
            next_id += 1
            
    # 2. Rebuild GT
    new_gt = copy.deepcopy(gt_data)
    clean_gt_images = []
    old_id_to_new = {}
    
    for img in gt_images:
        fname = os.path.basename(img['file_name'])
        new_id = filename_to_id[fname]
        old_id_to_new[img['id']] = new_id
        img['id'] = new_id
        clean_gt_images.append(img)
    new_gt['images'] = clean_gt_images
    
    gt_anns = gt_data.get('annotations', [])
    if isinstance(gt_anns, dict): gt_anns = list(gt_anns.values())
    clean_gt_anns = []
    for ann in gt_anns:
        if ann['image_id'] in old_id_to_new:
            ann['image_id'] = old_id_to_new[ann['image_id']]
            clean_gt_anns.append(ann)
    new_gt['annotations'] = clean_gt_anns

    # 3. Rebuild Preds
    pred_anns = pred_data.get('annotations', []) if isinstance(pred_data, dict) else pred_data
    if isinstance(pred_anns, dict): pred_anns = list(pred_anns.values())
    
    pred_images = pred_data.get('images', [])
    pred_id_to_fname = {img['id']: os.path.basename(img['file_name']) for img in pred_images}
    
    clean_pred_anns = []
    for ann in pred_anns:
        fname = pred_id_to_fname.get(ann['image_id'])
        if fname and fname in filename_to_id:
            ann['image_id'] = filename_to_id[fname]
            clean_pred_anns.append(ann)

    # 4. Load COCO
    coco_gt = COCO()
    coco_gt.dataset = new_gt
    coco_gt.createIndex()
    coco_pred = coco_gt.loadRes(clean_pred_anns)
    
    return coco_gt, coco_pred

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
        with open(gt_path) as f: gt_data = json.load(f)
        with open(pred_path) as f: pred_data = json.load(f)
        
        # Silence COCO prints
        with contextlib.redirect_stdout(io.StringIO()):
            cocoGt, cocoDt = align_coco_ids(gt_data, pred_data)
            
            # BOX METRICS
            cocoEvalB = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEvalB.evaluate()
            cocoEvalB.accumulate()
            cocoEvalB.summarize()
            b_map = cocoEvalB.stats[0] # mAP 0.5:0.95
            b_ap50 = cocoEvalB.stats[1] # AP50
            b_ar = cocoEvalB.stats[8]  # AR maxDets=100

            # SEG METRICS
            cocoEvalS = COCOeval(cocoGt, cocoDt, 'segm')
            cocoEvalS.evaluate()
            cocoEvalS.accumulate()
            cocoEvalS.summarize()
            s_map = cocoEvalS.stats[0]
            s_ap50 = cocoEvalS.stats[1]
            s_ar = cocoEvalS.stats[8]

        print(f"{dataset_name:<20} | {b_map:.3f}    | {b_ap50:.3f}     | {b_ar:.3f}    || {s_map:.3f}    | {s_ap50:.3f}     | {s_ar:.3f}")

    except Exception as e:
        print(f"{dataset_name:<20} | ERROR: {str(e)[:30]}")