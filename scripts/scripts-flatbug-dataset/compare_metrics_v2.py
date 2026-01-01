"""
Quick Metrics Comparison Script

Compare original predictions vs V2 post-processed predictions for a single dataset.
"""

import os
import json
import contextlib
import io
import copy
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==========================
# CONFIGURATION
# ==========================
ROOT_DATASET = "./flatbug-dataset"
DATASET_NAME = "cao2022"
GT_FILENAME = "instances_default.json"

# Predictions to compare
PREDICTIONS = {
    "Original": "sam3_flatbug_strategy-1.json",
    "V2 PostProc (dilation=7, pad=5)": "sam3_flatbug_strategy-1_v2_postproc.json",
}

# ==========================
# HELPER: ID ALIGNMENT
# ==========================
def align_coco_ids(gt_data, pred_data):
    """Rewrites image_ids to match based on filename."""
    filename_to_id = {}
    next_id = 1
    
    gt_images = gt_data.get('images', [])
    if isinstance(gt_images, dict): gt_images = list(gt_images.values())
    
    for img in gt_images:
        fname = os.path.basename(img['file_name'])
        if fname not in filename_to_id:
            filename_to_id[fname] = next_id
            next_id += 1
            
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

    coco_gt = COCO()
    coco_gt.dataset = new_gt
    coco_gt.createIndex()
    coco_pred = coco_gt.loadRes(clean_pred_anns)
    
    return coco_gt, coco_pred


def evaluate_predictions(gt_path, pred_path):
    """Evaluate predictions and return metrics."""
    with open(gt_path) as f:
        gt_data = json.load(f)
    with open(pred_path) as f:
        pred_data = json.load(f)
    
    # Silence COCO prints
    with contextlib.redirect_stdout(io.StringIO()):
        cocoGt, cocoDt = align_coco_ids(gt_data, pred_data)
        
        # Box metrics
        cocoEvalB = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEvalB.evaluate()
        cocoEvalB.accumulate()
        cocoEvalB.summarize()
        
        # Segmentation metrics
        cocoEvalS = COCOeval(cocoGt, cocoDt, 'segm')
        cocoEvalS.evaluate()
        cocoEvalS.accumulate()
        cocoEvalS.summarize()
    
    return {
        'bbox': {
            'mAP': cocoEvalB.stats[0],
            'AP50': cocoEvalB.stats[1],
            'AP75': cocoEvalB.stats[2],
            'AR': cocoEvalB.stats[8],
        },
        'segm': {
            'mAP': cocoEvalS.stats[0],
            'AP50': cocoEvalS.stats[1],
            'AP75': cocoEvalS.stats[2],
            'AR': cocoEvalS.stats[8],
        }
    }


def main():
    print("\n" + "=" * 70)
    print(f"Metrics Comparison: {DATASET_NAME}")
    print("=" * 70)
    
    dataset_path = os.path.join(ROOT_DATASET, DATASET_NAME)
    gt_path = os.path.join(dataset_path, GT_FILENAME)
    
    if not os.path.exists(gt_path):
        gt_path = os.path.join(dataset_path, "annotations", GT_FILENAME)
    
    results = {}
    
    for name, pred_file in PREDICTIONS.items():
        pred_path = os.path.join(dataset_path, pred_file)
        
        if not os.path.exists(pred_path):
            print(f"\n⚠️  {name}: File not found - {pred_file}")
            continue
        
        print(f"\nEvaluating: {name}...")
        metrics = evaluate_predictions(gt_path, pred_path)
        results[name] = metrics
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    # Header
    print(f"\n{'Metric':<20}", end="")
    for name in results:
        print(f" | {name[:25]:<25}", end="")
    print()
    print("-" * (20 + 28 * len(results)))
    
    # Bbox metrics
    print("\nBOUNDING BOX METRICS:")
    for metric in ['mAP', 'AP50', 'AP75', 'AR']:
        print(f"  {metric:<18}", end="")
        for name in results:
            val = results[name]['bbox'][metric]
            print(f" | {val:>23.4f}", end="")
        print()
    
    # Segmentation metrics
    print("\nSEGMENTATION METRICS:")
    for metric in ['mAP', 'AP50', 'AP75', 'AR']:
        print(f"  {metric:<18}", end="")
        for name in results:
            val = results[name]['segm'][metric]
            print(f" | {val:>23.4f}", end="")
        print()
    
    # Calculate improvements
    if len(results) >= 2:
        names = list(results.keys())
        orig = results[names[0]]
        v2 = results[names[1]]
        
        print("\n" + "=" * 70)
        print("IMPROVEMENT (V2 - Original):")
        print("=" * 70)
        
        print("\nBounding Box:")
        for metric in ['mAP', 'AP50', 'AP75', 'AR']:
            diff = v2['bbox'][metric] - orig['bbox'][metric]
            pct = (diff / orig['bbox'][metric] * 100) if orig['bbox'][metric] != 0 else 0
            sign = "+" if diff >= 0 else ""
            print(f"  {metric}: {sign}{diff:.4f} ({sign}{pct:.1f}%)")
        
        print("\nSegmentation:")
        for metric in ['mAP', 'AP50', 'AP75', 'AR']:
            diff = v2['segm'][metric] - orig['segm'][metric]
            pct = (diff / orig['segm'][metric] * 100) if orig['segm'][metric] != 0 else 0
            sign = "+" if diff >= 0 else ""
            print(f"  {metric}: {sign}{diff:.4f} ({sign}{pct:.1f}%)")
    
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
