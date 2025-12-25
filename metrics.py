# ==========================
# Segmentation metrics using IoU (GPU accelerated)
# CUDA DEVICE: 1
# ==========================

import os
import json
import numpy as np
from collections import defaultdict
import cv2
import torch

# ==========================
# CONFIG
# ==========================
root_dataset = "./flatbug-dataset"

datasets_to_eval = {
    "pinoy2023",
    "sticky-pi",
    "ubc-pitfall-traps",
    #"alus",
    #"bioscan",
    "diversityscanner",
    #"artaxor",
    #"collembolai",
    "ubc-scanned-sticky-cards",
    "abram2023"
}

IOU_THRESHOLD = 0.5
DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda")

print(f"Using device: {DEVICE}")

# ==========================
# UTILITIES
# ==========================

def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        if not poly:
            continue
        try:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        except Exception:
            pass
    return mask


def seg_to_mask(segmentation, height, width):
    if segmentation is None:
        return None

    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return np.zeros((height, width), dtype=np.uint8)
        return polygons_to_mask(segmentation, height, width)

    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
            return mask_utils.decode(segmentation).astype(np.uint8)
        except Exception:
            return None

    return None


def masks_to_tensor(masks):
    valid = [m for m in masks if m is not None]
    if len(valid) == 0:
        return None
    return torch.from_numpy(np.stack(valid)).to(DEVICE, dtype=torch.bool)


# ==========================
# GLOBAL COUNTERS
# ==========================
TP_global = FP_global = FN_global = 0

# ==========================
# DATASET LOOP
# ==========================
for dataset_name in sorted(os.listdir(root_dataset)):
    if dataset_name.lower() not in datasets_to_eval:
        print(f"Skipping folder: {dataset_name}")
        continue

    dataset_path = os.path.join(root_dataset, dataset_name)
    gt_file = os.path.join(dataset_path, "instances_default.json")
    pred_file = os.path.join(dataset_path, "sam3_results_pyramid_v2.json")

    if not os.path.isfile(gt_file) or not os.path.isfile(pred_file):
        print(f"❌ Missing files in {dataset_name}, skipping")
        continue

    print(f"\n======================")
    print(f"EVALUATING DATASET: {dataset_name}")
    print(f"======================")

    gt = json.load(open(gt_file))
    pred = json.load(open(pred_file))

    # --- FIXED: build mappings ONCE ---
    image_id_to_file = {im["id"]: im["file_name"] for im in gt["images"]}
    image_sizes = {im["file_name"]: (im["height"], im["width"]) for im in gt["images"]}

    gt_by_file = defaultdict(list)
    for ann in gt["annotations"]:
        fn = image_id_to_file.get(ann["image_id"])
        if fn:
            gt_by_file[fn].append(ann)

    pred_by_file = defaultdict(list)
    for ann in pred["annotations"]:
        fn = ann.get("file_name")
        if fn:
            pred_by_file[fn].append(ann)

    TP = FP = FN = 0
    skipped_pred = 0

    # ==========================
    # IMAGE LOOP
    # ==========================
    for idx, (file_name, gt_objs) in enumerate(gt_by_file.items(), 1):
        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(gt_by_file)} images")

        if file_name not in image_sizes:
            continue

        H, W = image_sizes[file_name]
        pred_objs = pred_by_file.get(file_name, [])

        gt_masks = [seg_to_mask(g["segmentation"], H, W) for g in gt_objs]
        pred_masks = []
        pred_scores = []

        for p in pred_objs:
            m = seg_to_mask(p["segmentation"], H, W)
            if m is None:
                skipped_pred += 1
            pred_masks.append(m)
            pred_scores.append(p.get("score", 1.0))

        gt_tensor = masks_to_tensor(gt_masks)
        pred_tensor = masks_to_tensor(pred_masks)

        if gt_tensor is None:
            FP += len(pred_masks)
            continue

        if pred_tensor is None:
            FN += gt_tensor.shape[0]
            continue

        # ==========================
        # GPU IoU MATRIX
        # ==========================
        P, G = pred_tensor.shape[0], gt_tensor.shape[0]

        pred_flat = pred_tensor.view(P, -1)
        gt_flat = gt_tensor.view(G, -1)

        inter = torch.matmul(pred_flat.float(), gt_flat.float().T)
        union = pred_flat.sum(1, keepdim=True) + gt_flat.sum(1) - inter
        iou = inter / (union + 1e-6)

        iou = iou.cpu().numpy()

        matched_gt = set()
        order = np.argsort(-np.array(pred_scores))

        for pi in order:
            best_gi = -1
            best_iou = 0.0
            for gi in range(G):
                if gi in matched_gt:
                    continue
                if iou[pi, gi] > best_iou:
                    best_iou = iou[pi, gi]
                    best_gi = gi

            if best_iou >= IOU_THRESHOLD:
                TP += 1
                matched_gt.add(best_gi)
            else:
                FP += 1

        FN += G - len(matched_gt)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f"TP={TP} FP={FP} FN={FN}")
    print(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")
    print(f"Skipped pred masks: {skipped_pred}")

    TP_global += TP
    FP_global += FP
    FN_global += FN

# ==========================
# OVERALL METRICS
# ==========================
precision = TP_global / (TP_global + FP_global) if TP_global + FP_global > 0 else 0
recall = TP_global / (TP_global + FN_global) if TP_global + FN_global > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

print("\n======================")
print("OVERALL SEGMENTATION EVALUATION")
print(f"TP={TP_global} FP={FP_global} FN={FN_global}")
print(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")
print("======================")
