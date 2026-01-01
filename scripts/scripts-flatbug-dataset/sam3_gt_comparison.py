#!/usr/bin/env python3
"""
SAM3 vs Ground Truth Comparison Script

This script iterates over dataset folders and compares SAM3 predictions with ground truth,
producing:
1. Overlay images with both SAM3 (blue) and GT (red) segmentation masks and bounding boxes
2. False Positive (FP) analysis metrics per dataset
3. Summary statistics

Output:
    DATASET                   | FP COUNT   | FN COUNT   | TP COUNT   | MEAN FP SCORE
    -------------------------------------------------------------------------------------
    ...
    Global Mean FP Score: X.XXXX
    Total FPs: XXXX
"""

import os
import json
import cv2
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import argparse

# ============================================================
# CONFIGURATION
# ============================================================
ROOT_DATASET = "./flatbug-dataset"
GT_FILENAME = "instances_default.json"
SAM3_FILENAME = "sam3_flatbug_strategy-1_v2_postproc.json" 
OUTPUT_FOLDER_NAME = "sam3_gt_comparison"

# Datasets to evaluate (set to None to process all)
DATASETS_TO_EVAL = None  # Will process all folders found
# Or specify specific datasets:
# DATASETS_TO_EVAL = ["Mothitor", "cao2022", "BIOSCAN"]

# Matching threshold
IOU_THRESHOLD = 0.5

# Visualization settings (BGR colors)
GT_COLOR = (0, 0, 200)       # Red for Ground Truth
SAM3_COLOR = (200, 100, 0)   # Blue for SAM3
FP_COLOR = (0, 165, 255)     # Orange for False Positives
FN_COLOR = (128, 0, 128)     # Purple for False Negatives

MASK_ALPHA = 0.4
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICK = 1


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_filename(fname: str) -> str:
    """Extract base filename for matching between GT and predictions."""
    return os.path.basename(fname) if fname else ""


def decode_segmentation_to_mask(segmentation: Any, height: int, width: int) -> Optional[np.ndarray]:
    """
    Convert COCO segmentation (polygon list or RLE) to binary mask.
    
    Args:
        segmentation: COCO format segmentation (list of polygons or RLE dict)
        height, width: Image dimensions
        
    Returns:
        Binary mask (H, W) with dtype uint8, or None if decoding fails
    """
    if segmentation is None or not segmentation:
        return None
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Handle polygon format (list of coordinate lists)
    if isinstance(segmentation, list):
        for poly in segmentation:
            if not poly or len(poly) < 6:  # Need at least 3 points
                continue
            try:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                pts = pts.astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
            except Exception:
                continue
        return mask
    
    # Handle RLE format (dict with 'counts' and 'size')
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as maskUtils
            decoded = maskUtils.decode(segmentation)
            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]
            return (decoded > 0).astype(np.uint8)
        except Exception:
            return None
    
    return None


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    if mask1 is None or mask2 is None:
        return 0.0
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return float(intersection) / float(union) if union > 0 else 0.0


def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two bounding boxes in [x, y, w, h] format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x1, y1, x2, y2]
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    
    # Intersection
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def blend_mask_on_image(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], 
                        alpha: float = 0.4) -> np.ndarray:
    """Blend a colored mask onto an image."""
    if mask is None or mask.sum() == 0:
        return image
    
    overlay = image.copy().astype(np.float32)
    colored_mask = np.zeros_like(overlay)
    colored_mask[:, :] = color
    
    mask_3ch = np.stack([mask] * 3, axis=-1).astype(bool)
    overlay[mask_3ch] = overlay[mask_3ch] * (1 - alpha) + colored_mask[mask_3ch] * alpha
    
    return overlay.astype(np.uint8)


def draw_bbox_with_label(image: np.ndarray, bbox: List[float], color: Tuple[int, int, int], 
                         label: str, thickness: int = 2) -> None:
    """Draw bounding box with label on image (in-place)."""
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=thickness)
    
    # Draw label background
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
    label_y1 = y1 - th - 6
    label_y2 = y1
    
    if label_y1 < 0:
        label_y1 = y1
        label_y2 = y1 + th + 6
    
    cv2.rectangle(image, (x1, label_y1), (x1 + tw + 8, label_y2), color, thickness=-1)
    
    # Draw text
    text_y = label_y2 - 4 if label_y1 < label_y2 else label_y1 + th + 2
    cv2.putText(image, label, (x1 + 4, text_y), FONT, FONT_SCALE, (255, 255, 255), 
                FONT_THICK, lineType=cv2.LINE_AA)


# ============================================================
# MATCHING LOGIC
# ============================================================

def match_predictions_to_gt(
    pred_annotations: List[Dict],
    gt_annotations: List[Dict],
    height: int,
    width: int,
    iou_threshold: float = 0.5,
    use_mask_iou: bool = True
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Match predictions to ground truth using IoU.
    
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
        Each is a list of dicts with annotation info and match details
    """
    # Prepare predictions with masks
    preds = []
    for ann in pred_annotations:
        mask = decode_segmentation_to_mask(ann.get("segmentation"), height, width)
        bbox = ann.get("bbox", [0, 0, 0, 0])
        score = ann.get("score", 0.0)
        preds.append({
            "annotation": ann,
            "mask": mask,
            "bbox": bbox,
            "score": score
        })
    
    # Sort by score descending (greedy matching)
    preds.sort(key=lambda x: x["score"], reverse=True)
    
    # Prepare GT with masks
    gts = []
    for ann in gt_annotations:
        mask = decode_segmentation_to_mask(ann.get("segmentation"), height, width)
        bbox = ann.get("bbox", [0, 0, 0, 0])
        gts.append({
            "annotation": ann,
            "mask": mask,
            "bbox": bbox,
            "matched": False
        })
    
    true_positives = []
    false_positives = []
    
    # Match predictions to GT
    for pred in preds:
        best_iou = 0.0
        best_gt_idx = None
        
        for i, gt in enumerate(gts):
            if gt["matched"]:
                continue
            
            # Compute IoU
            if use_mask_iou and pred["mask"] is not None and gt["mask"] is not None:
                iou = compute_mask_iou(pred["mask"], gt["mask"])
            else:
                iou = compute_box_iou(pred["bbox"], gt["bbox"])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold and best_gt_idx is not None:
            gts[best_gt_idx]["matched"] = True
            true_positives.append({
                "pred": pred,
                "gt": gts[best_gt_idx],
                "iou": best_iou
            })
        else:
            false_positives.append({
                "pred": pred,
                "best_iou": best_iou
            })
    
    # Collect false negatives (unmatched GT)
    false_negatives = [{"gt": gt} for gt in gts if not gt["matched"]]
    
    return true_positives, false_positives, false_negatives


# ============================================================
# VISUALIZATION
# ============================================================

def create_comparison_image(
    image_path: str,
    gt_annotations: List[Dict],
    pred_annotations: List[Dict],
    true_positives: List[Dict],
    false_positives: List[Dict],
    false_negatives: List[Dict],
    height: int,
    width: int
) -> Optional[np.ndarray]:
    """
    Create visualization image with GT (red) and SAM3 (blue) overlays.
    FPs are marked in orange, FNs in purple.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    vis = image.copy()
    H, W = vis.shape[:2]
    
    # Draw GT annotations (red) - all of them
    for ann in gt_annotations:
        mask = decode_segmentation_to_mask(ann.get("segmentation"), H, W)
        bbox = ann.get("bbox")
        
        if mask is not None:
            vis = blend_mask_on_image(vis, mask, GT_COLOR, alpha=MASK_ALPHA)
        if bbox:
            draw_bbox_with_label(vis, bbox, GT_COLOR, "GT")
    
    # Draw SAM3 predictions (blue for TP, orange for FP)
    fp_pred_ids = {id(fp["pred"]["annotation"]) for fp in false_positives}
    
    for ann in pred_annotations:
        mask = decode_segmentation_to_mask(ann.get("segmentation"), H, W)
        bbox = ann.get("bbox")
        score = ann.get("score", 0.0)
        
        # Determine color and label based on TP/FP status
        is_fp = id(ann) in fp_pred_ids or any(
            fp["pred"]["annotation"].get("id") == ann.get("id") and 
            fp["pred"]["annotation"].get("bbox") == ann.get("bbox")
            for fp in false_positives
        )
        
        if is_fp:
            color = FP_COLOR
            label = f"FP:{score:.2f}"
        else:
            color = SAM3_COLOR
            label = f"SAM3:{score:.2f}"
        
        if mask is not None:
            vis = blend_mask_on_image(vis, mask, color, alpha=MASK_ALPHA)
        if bbox:
            draw_bbox_with_label(vis, bbox, color, label)
    
    # Mark FNs (unmatched GT) with purple border
    for fn in false_negatives:
        bbox = fn["gt"]["bbox"]
        if bbox:
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), FN_COLOR, thickness=3)
            cv2.putText(vis, "FN", (x1 + 5, y1 + 20), FONT, FONT_SCALE, FN_COLOR, 
                       FONT_THICK, lineType=cv2.LINE_AA)
    
    return vis


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_dataset(
    dataset_path: str,
    dataset_name: str,
    gt_filename: str,
    sam3_filename: str,
    output_folder: str,
    iou_threshold: float = 0.5,
    save_images: bool = True
) -> Dict[str, Any]:
    """
    Process a single dataset and compute metrics.
    
    Returns:
        Dict with metrics: fp_count, fn_count, tp_count, fp_scores, etc.
    """
    gt_path = os.path.join(dataset_path, gt_filename)
    pred_path = os.path.join(dataset_path, sam3_filename)
    
    # Check for annotations subfolder
    if not os.path.exists(gt_path):
        gt_path = os.path.join(dataset_path, "annotations", gt_filename)
    
    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        return {"error": "Missing files", "gt_exists": os.path.exists(gt_path), 
                "pred_exists": os.path.exists(pred_path)}
    
    # Load JSONs
    try:
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        with open(pred_path, 'r') as f:
            pred = json.load(f)
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {str(e)}"}
    
    # Build image info lookups
    gt_images = {img["id"]: img for img in gt.get("images", [])}
    pred_images = {img["id"]: img for img in pred.get("images", [])}
    
    # Build filename-based lookups
    gt_id_to_info = {}
    for img in gt.get("images", []):
        norm_name = normalize_filename(img.get("file_name", ""))
        gt_id_to_info[img["id"]] = {
            "file_name": norm_name,
            "height": img.get("height", 0),
            "width": img.get("width", 0)
        }
    
    pred_id_to_filename = {}
    for img in pred.get("images", []):
        pred_id_to_filename[img["id"]] = normalize_filename(img.get("file_name", ""))
    
    # Group annotations by filename
    gt_by_file = defaultdict(list)
    for ann in gt.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id in gt_id_to_info:
            fname = gt_id_to_info[img_id]["file_name"]
            gt_by_file[fname].append(ann)
    
    pred_by_file = defaultdict(list)
    for ann in pred.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id in pred_id_to_filename:
            fname = pred_id_to_filename[img_id]
            pred_by_file[fname].append(ann)
    
    # Find common files
    gt_files = set(gt_by_file.keys())
    pred_files = set(pred_by_file.keys())
    common_files = gt_files.intersection(pred_files)
    
    if not common_files:
        return {
            "error": "No matching files",
            "gt_sample": list(gt_files)[:3],
            "pred_sample": list(pred_files)[:3]
        }
    
    # Create output folder
    if save_images:
        dataset_output = os.path.join(output_folder, dataset_name)
        os.makedirs(dataset_output, exist_ok=True)
    
    # Process each image
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_fp_scores = []
    per_image_stats = []  # Track per-image stats for finding worst cases
    
    for file_name in common_files:
        gt_anns = gt_by_file[file_name]
        pred_anns = pred_by_file[file_name]
        
        # Get image dimensions
        img_info = None
        for img_id, info in gt_id_to_info.items():
            if info["file_name"] == file_name:
                img_info = info
                break
        
        if img_info is None:
            continue
        
        H, W = img_info["height"], img_info["width"]
        
        # Match predictions to GT
        tps, fps, fns = match_predictions_to_gt(
            pred_anns, gt_anns, H, W, iou_threshold
        )
        
        total_tp += len(tps)
        total_fp += len(fps)
        total_fn += len(fns)
        
        # Track per-image stats
        per_image_stats.append({
            "file_name": file_name,
            "tp": len(tps),
            "fp": len(fps),
            "fn": len(fns),
            "fp_scores": [fp["pred"]["score"] for fp in fps]
        })
        
        # Collect FP scores
        for fp in fps:
            all_fp_scores.append(fp["pred"]["score"])
        
        # Create and save visualization
        if save_images:
            img_path = os.path.join(dataset_path, file_name)
            if os.path.exists(img_path):
                vis = create_comparison_image(
                    img_path, gt_anns, pred_anns, tps, fps, fns, H, W
                )
                if vis is not None:
                    out_path = os.path.join(dataset_output, file_name)
                    cv2.imwrite(out_path, vis)
    
    # Find images with most FPs and FNs
    top_fp_images = sorted(per_image_stats, key=lambda x: x["fp"], reverse=True)[:5]
    top_fn_images = sorted(per_image_stats, key=lambda x: x["fn"], reverse=True)[:5]
    
    return {
        "tp_count": total_tp,
        "fp_count": total_fp,
        "fn_count": total_fn,
        "fp_scores": all_fp_scores,
        "mean_fp_score": float(np.mean(all_fp_scores)) if all_fp_scores else 0.0,
        "num_images": len(common_files),
        "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0,
        "recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0,
        "top_fp_images": top_fp_images,
        "top_fn_images": top_fn_images
    }


def main():
    parser = argparse.ArgumentParser(description="Compare SAM3 predictions with Ground Truth")
    parser.add_argument("--root", type=str, default=ROOT_DATASET, help="Root dataset folder")
    parser.add_argument("--gt", type=str, default=GT_FILENAME, help="Ground truth JSON filename")
    parser.add_argument("--pred", type=str, default=SAM3_FILENAME, help="SAM3 predictions JSON filename")
    parser.add_argument("--output", type=str, default=OUTPUT_FOLDER_NAME, help="Output folder name")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="IoU threshold for matching")
    parser.add_argument("--no-images", action="store_true", help="Skip saving visualization images")
    parser.add_argument("--datasets", type=str, nargs="+", help="Specific datasets to process")
    args = parser.parse_args()
    
    root_dataset = args.root
    output_folder = os.path.join(root_dataset, args.output)
    os.makedirs(output_folder, exist_ok=True)
    
    # Determine which datasets to process
    if args.datasets:
        datasets = args.datasets
    elif DATASETS_TO_EVAL:
        datasets = DATASETS_TO_EVAL
    else:
        datasets = sorted([d for d in os.listdir(root_dataset) 
                          if os.path.isdir(os.path.join(root_dataset, d)) 
                          and not d.startswith(".")])
    
    # Print header
    print("=" * 100)
    print("SAM3 vs Ground Truth Comparison")
    print(f"IoU Threshold: {args.iou}")
    print(f"GT File: {args.gt}")
    print(f"Pred File: {args.pred}")
    print("=" * 100)
    print()
    print(f"{'DATASET':<25} | {'FP COUNT':<10} | {'FN COUNT':<10} | {'TP COUNT':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'MEAN FP SCORE':<12}")
    print("-" * 105)
    
    # Process datasets
    global_fp_scores = []
    global_stats = {
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0
    }
    
    results = {}
    
    for dataset_name in datasets:
        dataset_path = os.path.join(root_dataset, dataset_name)
        
        if not os.path.isdir(dataset_path):
            continue
        
        result = process_dataset(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            gt_filename=args.gt,
            sam3_filename=args.pred,
            output_folder=output_folder,
            iou_threshold=args.iou,
            save_images=not args.no_images
        )
        
        results[dataset_name] = result
        
        if "error" in result:
            print(f"{dataset_name:<25} | ERROR: {result['error']}")
            continue
        
        # Accumulate global stats
        global_fp_scores.extend(result["fp_scores"])
        global_stats["total_tp"] += result["tp_count"]
        global_stats["total_fp"] += result["fp_count"]
        global_stats["total_fn"] += result["fn_count"]
        
        # Print row
        print(f"{dataset_name:<25} | {result['fp_count']:<10} | {result['fn_count']:<10} | "
              f"{result['tp_count']:<10} | {result['precision']:<10.4f} | {result['recall']:<10.4f} | "
              f"{result['mean_fp_score']:<12.4f}")
    
    # Print summary
    print("-" * 105)
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if global_fp_scores:
        global_mean_fp = np.mean(global_fp_scores)
        print(f"Global Mean FP Score: {global_mean_fp:.4f}")
    else:
        print("Global Mean FP Score: N/A (no false positives)")
    
    print(f"Total True Positives:  {global_stats['total_tp']}")
    print(f"Total False Positives: {global_stats['total_fp']}")
    print(f"Total False Negatives: {global_stats['total_fn']}")
    
    total_pred = global_stats['total_tp'] + global_stats['total_fp']
    total_gt = global_stats['total_tp'] + global_stats['total_fn']
    
    if total_pred > 0:
        global_precision = global_stats['total_tp'] / total_pred
        print(f"Global Precision: {global_precision:.4f}")
    
    if total_gt > 0:
        global_recall = global_stats['total_tp'] / total_gt
        print(f"Global Recall: {global_recall:.4f}")
    
    if total_pred > 0 and total_gt > 0:
        f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
        print(f"Global F1 Score: {f1:.4f}")
    
    # Print worst-case images per dataset
    print()
    print("=" * 80)
    print("WORST-CASE IMAGES (Top 5 by FP and FN per dataset)")
    print("=" * 80)
    
    for dataset_name, result in results.items():
        if "error" in result:
            continue
        
        top_fp = result.get("top_fp_images", [])
        top_fn = result.get("top_fn_images", [])
        
        if not top_fp and not top_fn:
            continue
        
        print(f"\n--- {dataset_name} ---")
        
        if top_fp and top_fp[0]["fp"] > 0:
            print("  Top FP images:")
            for img in top_fp:
                if img["fp"] > 0:
                    print(f"    {img['file_name']}: {img['fp']} FPs (scores: {[f'{s:.2f}' for s in img['fp_scores'][:3]]}{'...' if len(img['fp_scores']) > 3 else ''})")
        
        if top_fn and top_fn[0]["fn"] > 0:
            print("  Top FN images:")
            for img in top_fn:
                if img["fn"] > 0:
                    print(f"    {img['file_name']}: {img['fn']} FNs")
    
    print()
    if not args.no_images:
        print(f"Visualization images saved to: {output_folder}")
    
    # Save results JSON
    results_json_path = os.path.join(output_folder, "comparison_results.json")
    with open(results_json_path, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable_results[k] = {
                    kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else 
                         ([float(x) for x in vv] if isinstance(vv, list) and vv and isinstance(vv[0], (np.floating, np.integer)) else vv))
                    for kk, vv in v.items()
                }
            else:
                serializable_results[k] = v
        
        json.dump({
            "config": {
                "iou_threshold": args.iou,
                "gt_file": args.gt,
                "pred_file": args.pred
            },
            "global_stats": global_stats,
            "global_mean_fp_score": float(np.mean(global_fp_scores)) if global_fp_scores else None,
            "per_dataset": serializable_results
        }, f, indent=2)
    
    print(f"Results saved to: {results_json_path}")


if __name__ == "__main__":
    main()
