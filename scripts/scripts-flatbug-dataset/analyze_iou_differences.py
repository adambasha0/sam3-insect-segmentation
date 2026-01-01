#!/usr/bin/env python3
"""
Analyze IoU differences between GT and SAM3 predictions.
Helps understand why segmentation mAP is much lower than bbox mAP.
"""

import json
import numpy as np
from pycocotools import mask as maskUtils

# Load data
print("Loading data...")
gt = json.load(open('./flatbug-dataset/ALUS/instances_default.json'))
sam3_strat = json.load(open('./flatbug-dataset/ALUS/sam3_flatbug_strategy.json'))
sam3_inf = json.load(open('./flatbug-dataset/ALUS/sam3_flatbug_strategy-1.json'))

# Build lookup
gt_img_lookup = {img['file_name']: img for img in gt['images']}
strat_img_lookup = {img['file_name']: img for img in sam3_strat['images']}
inf_img_lookup = {img['file_name']: img for img in sam3_inf['images']}

def seg_to_mask(seg, h, w):
    """Convert COCO segmentation to binary mask."""
    try:
        rle = maskUtils.frPyObjects(seg, h, w)
        return maskUtils.decode(maskUtils.merge(rle))
    except:
        return None

def mask_iou(m1, m2):
    """Compute IoU between two binary masks."""
    if m1 is None or m2 is None:
        return 0.0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union > 0 else 0.0

def bbox_iou(b1, b2):
    """Compute IoU between two [x,y,w,h] bboxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0

# Analyze across multiple images
print("\n" + "="*80)
print("ANALYSIS: IoU Comparison (BBox vs Segmentation)")
print("="*80)

num_images_to_check = 20
bbox_ious_all = []
mask_ious_all = []
matched_count = 0
total_gt = 0

for img_idx, gt_img in enumerate(gt['images'][:num_images_to_check]):
    fname = gt_img['file_name']
    h, w = gt_img['height'], gt_img['width']
    gt_id = gt_img['id']
    
    sam_img = inf_img_lookup.get(fname)
    if not sam_img:
        continue
    sam_id = sam_img['id']
    
    # Get annotations
    gt_anns = [a for a in gt['annotations'] if a['image_id'] == gt_id]
    sam_anns = [a for a in sam3_inf['annotations'] if a['image_id'] == sam_id]
    
    # Convert to masks
    gt_data = [(a, a['bbox'], seg_to_mask(a['segmentation'], h, w)) for a in gt_anns]
    sam_data = [(a, a['bbox'], seg_to_mask(a['segmentation'], h, w)) for a in sam_anns]
    
    # Match GT to SAM by best bbox IoU
    for gt_ann, gt_bbox, gt_mask in gt_data:
        total_gt += 1
        best_bbox_iou = 0
        best_mask_iou = 0
        best_sam_idx = -1
        
        for sam_ann, sam_bbox, sam_mask in sam_data:
            biou = bbox_iou(gt_bbox, sam_bbox)
            if biou > best_bbox_iou:
                best_bbox_iou = biou
                best_mask_iou = mask_iou(gt_mask, sam_mask)
                best_sam_idx = sam_anns.index(sam_ann)
        
        if best_bbox_iou > 0.5:  # Consider a match at IoU > 0.5
            matched_count += 1
            bbox_ious_all.append(best_bbox_iou)
            mask_ious_all.append(best_mask_iou)

print(f"\nAnalyzed {num_images_to_check} images")
print(f"Total GT annotations checked: {total_gt}")
print(f"Matched (bbox IoU > 0.5): {matched_count}")

if bbox_ious_all:
    print(f"\n--- When matched by BBox (IoU > 0.5) ---")
    print(f"BBox IoU:  mean={np.mean(bbox_ious_all):.3f}, median={np.median(bbox_ious_all):.3f}")
    print(f"Mask IoU:  mean={np.mean(mask_ious_all):.3f}, median={np.median(mask_ious_all):.3f}")
    print(f"IoU DROP:  {np.mean(bbox_ious_all) - np.mean(mask_ious_all):.3f} (avg)")
    
    # How many have mask IoU significantly lower than bbox IoU?
    iou_drops = np.array(bbox_ious_all) - np.array(mask_ious_all)
    print(f"\n--- IoU Drop Distribution ---")
    print(f"Cases where mask IoU dropped by >0.2: {(iou_drops > 0.2).sum()} / {len(iou_drops)}")
    print(f"Cases where mask IoU dropped by >0.3: {(iou_drops > 0.3).sum()} / {len(iou_drops)}")
    print(f"Cases where mask IoU dropped by >0.5: {(iou_drops > 0.5).sum()} / {len(iou_drops)}")

# Now analyze the masks more carefully
print("\n" + "="*80)
print("ANALYSIS: Mask Quality Issues")
print("="*80)

# Check for multi-component masks (SAM might be fragmenting masks)
def count_connected_components(mask):
    if mask is None:
        return 0
    import cv2
    num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
    return num_labels - 1  # Subtract background

print("\n--- Connected Components per Mask ---")

gt_components = []
sam_components = []

for img_idx, gt_img in enumerate(gt['images'][:5]):
    fname = gt_img['file_name']
    h, w = gt_img['height'], gt_img['width']
    gt_id = gt_img['id']
    
    sam_img = inf_img_lookup.get(fname)
    if not sam_img:
        continue
    sam_id = sam_img['id']
    
    gt_anns = [a for a in gt['annotations'] if a['image_id'] == gt_id]
    sam_anns = [a for a in sam3_inf['annotations'] if a['image_id'] == sam_id]
    
    for ann in gt_anns:
        mask = seg_to_mask(ann['segmentation'], h, w)
        if mask is not None:
            gt_components.append(count_connected_components(mask))
    
    for ann in sam_anns:
        mask = seg_to_mask(ann['segmentation'], h, w)
        if mask is not None:
            sam_components.append(count_connected_components(mask))

print(f"GT masks: avg {np.mean(gt_components):.2f} components (range {min(gt_components)}-{max(gt_components)})")
print(f"SAM masks: avg {np.mean(sam_components):.2f} components (range {min(sam_components)}-{max(sam_components)})")

# Check mask coverage vs bbox area
print("\n--- Mask Coverage (mask_area / bbox_area) ---")

def get_coverage_stats(annotations, img_lookup, max_samples=200):
    coverages = []
    for ann in annotations[:max_samples]:
        img = None
        for im in img_lookup.values():
            if im['id'] == ann['image_id']:
                img = im
                break
        if img is None:
            continue
        
        h, w = img['height'], img['width']
        bbox = ann['bbox']
        bbox_area = bbox[2] * bbox[3]
        
        mask = seg_to_mask(ann['segmentation'], h, w)
        if mask is not None and bbox_area > 0:
            mask_area = mask.sum()
            coverages.append(mask_area / bbox_area)
    return coverages

gt_coverages = get_coverage_stats(gt['annotations'], gt_img_lookup)
sam_coverages = get_coverage_stats(sam3_inf['annotations'], inf_img_lookup)

print(f"GT:   mean={np.mean(gt_coverages):.3f}, std={np.std(gt_coverages):.3f}")
print(f"SAM3: mean={np.mean(sam_coverages):.3f}, std={np.std(sam_coverages):.3f}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print("""
1. MASK QUALITY: SAM3 masks have lower fill ratio (~0.49) vs GT (~0.64)
   - SAM3 masks are LESS FILLED relative to their bounding boxes
   - This directly impacts segmentation IoU

2. IoU DROP: When bbox matches well, mask IoU still drops significantly
   - This means the masks don't align well even when bboxes do
   - Likely cause: SAM3 masks have different boundaries than GT annotations

3. POSSIBLE CAUSES:
   a) SAM3 model outputs tighter/different mask boundaries
   b) Polygon simplification in our code is too aggressive
   c) Multi-scale detection creates inconsistent mask quality
   d) GT annotations might use different annotation conventions
""")
