# ==========================
# BEST DILATION FINDER
# ==========================
import os
import json
import numpy as np
from collections import defaultdict
import cv2

# ==========================
# CONFIG
# ==========================
root_dataset = "./flatbug-dataset"
datasets_to_eval = {
    "nhm-beetles-crops",
    "cao2022",
    "gernat2018",
    "sittinger2023",
    "amarathunga2022",
    "biodiscover-arm",
}

IOU_THRESHOLD = 0.5
MAX_DILATION = 50  # maximum dilation to try
OVERLAY_SAVE = True  # save overlay images for 2 examples

# Output folder
overlay_root = "dilated_metrics"
os.makedirs(overlay_root, exist_ok=True)

# ==========================
# UTILITIES
# ==========================
def seg_to_mask(segmentation, height, width):
    """Convert COCO segmentation (polygons or RLE) to binary mask."""
    if segmentation is None:
        return None
    if isinstance(segmentation, list):
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in segmentation:
            if not poly:
                continue
            try:
                pts = np.array(poly, dtype=np.int32).reshape(-1,2)
                cv2.fillPoly(mask, [pts], 1)
            except Exception:
                continue
        return mask
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
            return mask_utils.decode(segmentation).astype(np.uint8)
        except Exception:
            return None
    return None

def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def dilate_mask(mask, pixels):
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

def overlay_masks(img_path, gt_masks, pred_masks, save_path):
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((gt_masks[0].shape[0], gt_masks[0].shape[1], 3), dtype=np.uint8)
    overlay = img.copy()
    for m in gt_masks:
        overlay[m.astype(bool)] = (0,255,0)  # GT green
    for m in pred_masks:
        overlay[m.astype(bool)] = (0,0,255)  # prediction red
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

# ==========================
# MAIN LOOP
# ==========================
best_dilations = []

for dataset_name in sorted(datasets_to_eval):
    dataset_path = os.path.join(root_dataset, dataset_name)
    gt_file = os.path.join(dataset_path, "instances_default.json")
    pred_file = os.path.join(dataset_path, "sam3_results.json")

    if not os.path.isfile(gt_file) or not os.path.isfile(pred_file):
        print(f"❌ Missing GT or predictions for {dataset_name}, skipping.")
        continue

    print(f"\n======================\nEvaluating dataset: {dataset_name}\n======================")

    gt_data = json.load(open(gt_file))
    pred_data = json.load(open(pred_file))

    # map annotations by file_name
    gt_by_file = defaultdict(list)
    gt_image_sizes = {}
    for im in gt_data.get("images", []):
        gt_image_sizes[im["file_name"]] = (im["height"], im["width"])
    for ann in gt_data.get("annotations", []):
        file_name = next((im["file_name"] for im in gt_data["images"] if im["id"] == ann["image_id"]), None)
        if file_name:
            gt_by_file[file_name].append(ann)

    pred_by_file = defaultdict(list)
    for ann in pred_data.get("annotations", []):
        file_name = ann.get("file_name")
        if file_name:
            pred_by_file[file_name].append(ann)

    best_f1 = 0
    best_dil = 0
    downtrend_count = 0

    for d in range(MAX_DILATION):
        TP = FP = FN = 0

        for file_name, gt_objs in gt_by_file.items():
            if file_name not in gt_image_sizes:
                continue
            H, W = gt_image_sizes[file_name]
            pred_objs = pred_by_file.get(file_name, [])

            gt_masks = [seg_to_mask(g.get("segmentation"), H, W) for g in gt_objs]
            pred_masks = [seg_to_mask(p.get("segmentation"), H, W) for p in pred_objs]

            # apply dilation
            pred_masks = [dilate_mask(m, d) if m is not None else None for m in pred_masks]

            gt_cats = [g.get("category_id") for g in gt_objs]
            pred_cats = [p.get("category_id") for p in pred_objs]

            matched_gt = set()
            for pi, pmask in enumerate(pred_masks):
                if pmask is None:
                    continue
                pcat = pred_cats[pi]
                best_iou = 0
                best_gi = None
                for gi, (gmask, gcat) in enumerate(zip(gt_masks, gt_cats)):
                    if gi in matched_gt or gmask is None:
                        continue
                    if pcat != gcat:
                        continue
                    iou = mask_iou(pmask, gmask)
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                if best_iou >= IOU_THRESHOLD and best_gi is not None:
                    TP += 1
                    matched_gt.add(best_gi)
                else:
                    FP += 1
            FN += sum(1 for gi in range(len(gt_objs)) if gi not in matched_gt and gt_masks[gi] is not None)

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        print(f"Dilation {d}: TP={TP}, FP={FP}, FN={FN}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_dil = d
            downtrend_count = 0
        elif f1 < best_f1:
            downtrend_count += 1
            if downtrend_count >= 2:
                print(f"Stopping early at dilation {d} due to F1 downtrend.")
                break

    print(f"✅ Best dilation for {dataset_name}: {best_dil}, F1={best_f1:.4f}")
    best_dilations.append(best_dil)

    # save overlay images for 2 examples
    if OVERLAY_SAVE:
        example_files = list(gt_by_file.keys())[:2]
        for file_name in example_files:
            H, W = gt_image_sizes[file_name]
            gt_masks = [seg_to_mask(g.get("segmentation"), H, W) for g in gt_by_file[file_name]]
            pred_masks = [seg_to_mask(p.get("segmentation"), H, W) for p in pred_by_file.get(file_name, [])]
            pred_masks = [dilate_mask(m, best_dil) if m is not None else None for m in pred_masks]
            save_path = os.path.join(overlay_root, dataset_name, f"{file_name}")
            overlay_masks(os.path.join(root_dataset, dataset_name, file_name), gt_masks, pred_masks, save_path)

# ==========================
# Mean best dilation
# ==========================
mean_best_dil = np.mean(best_dilations)
print(f"\n======================\nMean best dilation across datasets: {mean_best_dil:.2f}\n======================")
