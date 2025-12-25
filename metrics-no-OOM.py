# =====================================================
# SAFE GPU METRICS FOR SAM3 (NO OOM)
# =====================================================

import os
import json
import numpy as np
from collections import defaultdict
import cv2
import torch
import gc

# ==========================
# CONFIG
# ==========================
ROOT_DATASET = "./flatbug-dataset"
DEVICE = torch.device("cuda:1")   # <-- use CUDA 1

DATASETS_TO_EVAL = {
    #"pinoy2023",
    #"sticky-pi",
    #"ubc-pitfall-traps",
    #"diversityscanner",
    "CollembolAI",
    #"ubc-scanned-sticky-cards",
    #"abram2023"
}

IOU_THRESHOLD = 0.5

# ==========================
# MASK UTILITIES (CPU)
# ==========================
def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        if not poly: continue
        try:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        except Exception:
            continue
    return mask

def seg_to_mask(segmentation, height, width):
    if segmentation is None:
        return None
    if isinstance(segmentation, list):
        if len(segmentation) == 0: return None
        return polygons_to_mask(segmentation, height, width)
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
            return mask_utils.decode(segmentation).astype(np.uint8)
        except Exception:
            return None
    return None

# ==========================
# GPU IOU (single pair)
# ==========================
def mask_iou_gpu(mask1_np, mask2_np):
    m1 = torch.from_numpy(mask1_np).to(DEVICE, dtype=torch.bool)
    m2 = torch.from_numpy(mask2_np).to(DEVICE, dtype=torch.bool)
    inter = torch.logical_and(m1, m2).sum()
    union = torch.logical_or(m1, m2).sum()
    iou = (inter / union).item() if union > 0 else 0.0
    del m1, m2
    torch.cuda.empty_cache()
    return iou

# ==========================
# GLOBAL COUNTERS
# ==========================
TP_GLOBAL = 0
FP_GLOBAL = 0
FN_GLOBAL = 0

# ==========================
# DATASET LOOP
# ==========================
for dataset_name in sorted(os.listdir(ROOT_DATASET)):
    if dataset_name not in DATASETS_TO_EVAL:
        print(f"Skipping folder: {dataset_name}")
        continue

    print(f"\n======================")
    print(f"EVALUATING DATASET: {dataset_name}")
    print(f"======================")

    dataset_path = os.path.join(ROOT_DATASET, dataset_name)
    gt_file = os.path.join(dataset_path, "instances_default.json")
    pred_file = os.path.join(dataset_path, "sam3_results_pyramid_v2.json")

    if not os.path.isfile(gt_file) or not os.path.isfile(pred_file):
        print("❌ Missing GT or predictions, skipping.")
        continue

    # Load JSON
    gt = json.load(open(gt_file))
    pred = json.load(open(pred_file))

    # Map image sizes
    image_sizes = {im["file_name"]: (im["height"], im["width"]) for im in gt["images"]}

    # Group GT by file
    gt_by_file = defaultdict(list)
    for ann in gt["annotations"]:
        img = next(i for i in gt["images"] if i["id"] == ann["image_id"])
        gt_by_file[img["file_name"]].append(ann)

    # Group predictions by file
    pred_by_file = defaultdict(list)
    for ann in pred["annotations"]:
        if "file_name" in ann:
            pred_by_file[ann["file_name"]].append(ann)

    # Dataset counters
    TP = FP = FN = 0

    # ==========================
    # IMAGE LOOP
    # ==========================
    for idx, (file_name, gt_objs) in enumerate(gt_by_file.items(), 1):
        print(f"[{dataset_name}] Image {idx}/{len(gt_by_file)} → {file_name}", flush=True)

        if file_name not in image_sizes:
            continue

        H, W = image_sizes[file_name]
        pred_objs = pred_by_file.get(file_name, [])

        # Convert segmentations to masks
        gt_masks = [seg_to_mask(g.get("segmentation"), H, W) for g in gt_objs]
        pred_masks = [seg_to_mask(p.get("segmentation"), H, W) for p in pred_objs]
        pred_scores = [p.get("score", 1.0) for p in pred_objs]

        matched_gt = set()
        # Sort predictions by score descending
        order = sorted(range(len(pred_masks)),
                       key=lambda i: pred_scores[i] if pred_masks[i] is not None else 0.0,
                       reverse=True)

        for pi in order:
            pmask = pred_masks[pi]
            if pmask is None: continue

            best_iou = 0.0
            best_gi = None

            for gi, gmask in enumerate(gt_masks):
                if gi in matched_gt or gmask is None:
                    continue
                iou = mask_iou_gpu(pmask, gmask)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= IOU_THRESHOLD and best_gi is not None:
                TP += 1
                matched_gt.add(best_gi)
            else:
                FP += 1

        FN += sum(1 for gi in range(len(gt_masks)) if gi not in matched_gt)
        torch.cuda.empty_cache()
        gc.collect()

    # ==========================
    # DATASET METRICS
    # ==========================
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f"\nRESULTS [{dataset_name}]")
    print(f"TP={TP} FP={FP} FN={FN}")
    print(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")

    TP_GLOBAL += TP
    FP_GLOBAL += FP
    FN_GLOBAL += FN

# ==========================
# OVERALL METRICS
# ==========================
precision_g = TP_GLOBAL / (TP_GLOBAL + FP_GLOBAL) if TP_GLOBAL + FP_GLOBAL > 0 else 0
recall_g = TP_GLOBAL / (TP_GLOBAL + FN_GLOBAL) if TP_GLOBAL + FN_GLOBAL > 0 else 0
f1_g = 2 * precision_g * recall_g / (precision_g + recall_g) if precision_g + recall_g > 0 else 0

print("\n======================")
print("OVERALL SEGMENTATION METRICS")
print(f"TP={TP_GLOBAL} FP={FP_GLOBAL} FN={FN_GLOBAL}")
print(f"Precision={precision_g:.4f} Recall={recall_g:.4f} F1={f1_g:.4f}")
print("======================")
