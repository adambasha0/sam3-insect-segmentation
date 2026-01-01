import os
import json
import contextlib
import io
import copy
import numpy as np
from datetime import datetime
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==========================
# CONFIGURATION
# ==========================
ROOT_DATASET = "./flatbug-dataset"
PRED_FILENAME = "sam3_flatbug_strategy_v2.json"
GT_FILENAME = "instances_default.json"
METRICS_OUTPUT_FILENAME = "sam3_metrics_results_v2.json"  # Must be different from PRED_FILENAME!
LOG_FILENAME = "sam3_metrics_log_v2.txt"

# SAFETY CHECK: Prevent overwriting prediction files
if METRICS_OUTPUT_FILENAME == PRED_FILENAME:
    raise ValueError(f"METRICS_OUTPUT_FILENAME cannot be the same as PRED_FILENAME! "
                     f"This would overwrite your predictions. Change METRICS_OUTPUT_FILENAME.")

# ONLY the datasets that crashed the GPU script
""" ALLOWED_FOLDERS = {
    "CollembolAI", "AMI-traps", "abram2023", "PeMaToEuroPep", 
    # "ubc-scanned-sticky-cards", "sittinger2023", "pinoy2023",
    # "sticky-pi", "ubc-pitfall-traps",
} """

ALLOWED_FOLDERS = {
    #"NHM-beetles-crops",
    #"cao2022",
    #"gernat2018",
    #"sittinger2023",
    #"amarathunga2022",
    #"biodiscover-arm",
    "Mothitor",
    #"DIRT",
    #"Diopsis",
    #"AMI-traps",
    #"AMT",
    #"PeMaToEuroPep",
    #"abram2023",
    #"anTraX",
    #"pinoy2023",
    #"sticky-pi",
    #"ubc-pitfall-traps",
    #"ALUS",
    #"BIOSCAN",
    #"DiversityScanner",
    #"ArTaxOr",
    #"CollembolAI",
    #"ubc-scanned-sticky-cards",
}

# ==========================
# HELPER: EXTRACT DETAILED METRICS
# ==========================
def extract_detailed_metrics(coco_eval, eval_type='bbox'):
    """
    Extract detailed metrics from COCOeval object for plotting.
    
    Returns a dictionary containing:
    - Precision-Recall curve data at different IoU thresholds
    - Per-category metrics
    - Score thresholds
    - All COCO standard metrics
    """
    metrics = {
        'eval_type': eval_type,
        'iou_thresholds': coco_eval.params.iouThrs.tolist(),
        'recall_thresholds': coco_eval.params.recThrs.tolist(),
        'max_detections': coco_eval.params.maxDets,
        'area_ranges': ['all', 'small', 'medium', 'large'],
        'area_range_values': coco_eval.params.areaRng,
    }
    
    # Standard COCO metrics
    stats = coco_eval.stats
    metrics['summary'] = {
        'mAP': float(stats[0]),           # AP @ IoU=0.50:0.95
        'AP50': float(stats[1]),          # AP @ IoU=0.50
        'AP75': float(stats[2]),          # AP @ IoU=0.75
        'AP_small': float(stats[3]),      # AP for small objects
        'AP_medium': float(stats[4]),     # AP for medium objects
        'AP_large': float(stats[5]),      # AP for large objects
        'AR_maxDet1': float(stats[6]),    # AR with max 1 detection
        'AR_maxDet10': float(stats[7]),   # AR with max 10 detections
        'AR_maxDet100': float(stats[8]),  # AR with max 100 detections
        'AR_small': float(stats[9]),      # AR for small objects
        'AR_medium': float(stats[10]),    # AR for medium objects
        'AR_large': float(stats[11]),     # AR for large objects
    }
    
    # Precision array: [TxRxKxAxM]
    # T: IoU thresholds, R: recall thresholds, K: categories, A: area ranges, M: max detections
    precision = coco_eval.eval['precision']  # Shape: (T, R, K, A, M)
    recall = coco_eval.eval['recall']        # Shape: (T, K, A, M)
    scores = coco_eval.eval['scores']        # Shape: (T, R, K, A, M)
    
    # Extract Precision-Recall curves for plotting
    # We'll store curves at different IoU thresholds
    pr_curves = {}
    
    # Key IoU thresholds to store (0.5, 0.75, and average)
    iou_indices = {
        'IoU_0.50': 0,
        'IoU_0.55': 1,
        'IoU_0.60': 2,
        'IoU_0.65': 3,
        'IoU_0.70': 4,
        'IoU_0.75': 5,
        'IoU_0.80': 6,
        'IoU_0.85': 7,
        'IoU_0.90': 8,
        'IoU_0.95': 9,
    }
    
    recall_thresholds = coco_eval.params.recThrs.tolist()
    
    for iou_name, iou_idx in iou_indices.items():
        # Average over all categories (K) and use area='all' (A=0), maxDet=100 (M=2)
        prec_at_iou = precision[iou_idx, :, :, 0, 2]  # Shape: (R, K)
        
        # Handle case with no valid precision values
        valid_prec = prec_at_iou[prec_at_iou > -1]
        if len(valid_prec) > 0:
            # Average precision across categories for each recall threshold
            mean_prec = np.mean(prec_at_iou[prec_at_iou > -1].reshape(-1, prec_at_iou.shape[1]), axis=1) if prec_at_iou.shape[1] > 0 else []
            
            # Store the curve data
            pr_curves[iou_name] = {
                'recall': recall_thresholds,
                'precision': np.mean(prec_at_iou, axis=1).tolist() if prec_at_iou.size > 0 else [],
                'AP': float(np.mean(prec_at_iou[prec_at_iou > -1])) if len(valid_prec) > 0 else 0.0,
            }
    
    metrics['precision_recall_curves'] = pr_curves
    
    # Per-category metrics (if categories exist)
    cat_ids = coco_eval.params.catIds
    if len(cat_ids) > 0:
        per_category = {}
        for k_idx, cat_id in enumerate(cat_ids):
            cat_prec = precision[:, :, k_idx, 0, 2]  # All IoUs, all recalls, this category
            cat_rec = recall[:, k_idx, 0, 2]  # All IoUs, this category
            
            valid_cat_prec = cat_prec[cat_prec > -1]
            per_category[str(cat_id)] = {
                'AP': float(np.mean(valid_cat_prec)) if len(valid_cat_prec) > 0 else 0.0,
                'AP50': float(np.mean(cat_prec[0][cat_prec[0] > -1])) if len(cat_prec[0][cat_prec[0] > -1]) > 0 else 0.0,
                'max_recall': float(np.max(cat_rec[cat_rec > -1])) if len(cat_rec[cat_rec > -1]) > 0 else 0.0,
            }
        metrics['per_category'] = per_category
    
    # AP at each IoU threshold (for IoU threshold analysis plot)
    ap_per_iou = []
    for t_idx in range(len(coco_eval.params.iouThrs)):
        prec_at_t = precision[t_idx, :, :, 0, 2]
        valid = prec_at_t[prec_at_t > -1]
        ap_per_iou.append(float(np.mean(valid)) if len(valid) > 0 else 0.0)
    metrics['ap_per_iou_threshold'] = {
        'iou_thresholds': coco_eval.params.iouThrs.tolist(),
        'AP_values': ap_per_iou,
    }
    
    # Size-based analysis (for object size distribution plot)
    metrics['size_analysis'] = {
        'small': {'AP': float(stats[3]), 'AR': float(stats[9])},
        'medium': {'AP': float(stats[4]), 'AR': float(stats[10])},
        'large': {'AP': float(stats[5]), 'AR': float(stats[11])},
    }
    
    return metrics


def save_metrics_to_file(dataset_path, dataset_name, bbox_metrics, segm_metrics, 
                         num_gt_images, num_gt_annotations, num_pred_annotations):
    """Save comprehensive metrics to JSON file in dataset folder."""
    
    output_data = {
        'dataset_name': dataset_name,
        'evaluation_timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'num_images': num_gt_images,
            'num_gt_annotations': num_gt_annotations,
            'num_pred_annotations': num_pred_annotations,
        },
        'bbox_metrics': bbox_metrics,
        'segm_metrics': segm_metrics,
        'plotting_guide': {
            'precision_recall_curve': 'Use precision_recall_curves dict with recall as x-axis and precision as y-axis',
            'iou_threshold_analysis': 'Use ap_per_iou_threshold to plot AP vs IoU threshold',
            'size_analysis': 'Use size_analysis to create bar chart comparing small/medium/large object performance',
            'bbox_vs_segm': 'Compare bbox_metrics.summary vs segm_metrics.summary for detection vs segmentation comparison',
        }
    }
    
    output_path = os.path.join(dataset_path, METRICS_OUTPUT_FILENAME)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return output_path


def write_log(log_file, message, also_print=True):
    """Write message to log file and optionally print to console."""
    log_file.write(message + '\n')
    if also_print:
        print(message)


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

# Create global log file
global_log_path = os.path.join(ROOT_DATASET, LOG_FILENAME)
global_log = open(global_log_path, 'w')

# Write header
header = f"SAM3 Metrics Evaluation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
write_log(global_log, "=" * 100)
write_log(global_log, header)
write_log(global_log, "=" * 100)
write_log(global_log, "")

table_header = f"{'DATASET':<25} | {'mAP(Box)':<8} | {'AP50(Box)':<9} | {'AR(Box)':<8} || {'mAP(Seg)':<8} | {'AP50(Seg)':<9} | {'AR(Seg)':<8}"
write_log(global_log, table_header)
write_log(global_log, "-" * 105)

# Summary statistics
all_results = []
successful_datasets = 0
failed_datasets = 0

for dataset_name in sorted(os.listdir(ROOT_DATASET)):
    if dataset_name not in ALLOWED_FOLDERS: continue

    dataset_path = os.path.join(ROOT_DATASET, dataset_name)
    gt_path = os.path.join(dataset_path, GT_FILENAME)
    if not os.path.exists(gt_path): gt_path = os.path.join(dataset_path, "annotations", GT_FILENAME)
    pred_path = os.path.join(dataset_path, PRED_FILENAME)

    if not os.path.exists(gt_path) or not os.path.exists(pred_path): 
        write_log(global_log, f"{dataset_name:<25} | SKIPPED: Missing GT or predictions file")
        continue

    # Create dataset-specific log file
    dataset_log_path = os.path.join(dataset_path, LOG_FILENAME)
    dataset_log = open(dataset_log_path, 'w')
    
    try:
        with open(gt_path) as f: gt_data = json.load(f)
        with open(pred_path) as f: pred_data = json.load(f)
        
        # Get dataset statistics
        num_gt_images = len(gt_data.get('images', []))
        num_gt_annotations = len(gt_data.get('annotations', []))
        pred_anns = pred_data.get('annotations', []) if isinstance(pred_data, dict) else pred_data
        num_pred_annotations = len(pred_anns) if isinstance(pred_anns, list) else 0
        
        # Write dataset info to local log
        write_log(dataset_log, f"Dataset: {dataset_name}", also_print=False)
        write_log(dataset_log, f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", also_print=False)
        write_log(dataset_log, f"GT Path: {gt_path}", also_print=False)
        write_log(dataset_log, f"Pred Path: {pred_path}", also_print=False)
        write_log(dataset_log, f"Number of Images: {num_gt_images}", also_print=False)
        write_log(dataset_log, f"Number of GT Annotations: {num_gt_annotations}", also_print=False)
        write_log(dataset_log, f"Number of Predictions: {num_pred_annotations}", also_print=False)
        write_log(dataset_log, "-" * 60, also_print=False)
        
        # Silence COCO prints during evaluation
        with contextlib.redirect_stdout(io.StringIO()):
            cocoGt, cocoDt = align_coco_ids(gt_data, pred_data)
            
            # BOX METRICS
            cocoEvalB = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEvalB.evaluate()
            cocoEvalB.accumulate()
            cocoEvalB.summarize()
            
            # SEG METRICS
            cocoEvalS = COCOeval(cocoGt, cocoDt, 'segm')
            cocoEvalS.evaluate()
            cocoEvalS.accumulate()
            cocoEvalS.summarize()
        
        # Extract detailed metrics for plotting
        bbox_metrics = extract_detailed_metrics(cocoEvalB, 'bbox')
        segm_metrics = extract_detailed_metrics(cocoEvalS, 'segm')
        
        # Get summary values
        b_map = bbox_metrics['summary']['mAP']
        b_ap50 = bbox_metrics['summary']['AP50']
        b_ar = bbox_metrics['summary']['AR_maxDet100']
        s_map = segm_metrics['summary']['mAP']
        s_ap50 = segm_metrics['summary']['AP50']
        s_ar = segm_metrics['summary']['AR_maxDet100']
        
        # Save metrics to JSON file
        metrics_path = save_metrics_to_file(
            dataset_path, dataset_name, bbox_metrics, segm_metrics,
            num_gt_images, num_gt_annotations, num_pred_annotations
        )
        
        # Write detailed results to dataset log
        write_log(dataset_log, "\n=== BOUNDING BOX METRICS ===", also_print=False)
        write_log(dataset_log, f"  mAP (IoU=0.50:0.95): {b_map:.4f}", also_print=False)
        write_log(dataset_log, f"  AP50 (IoU=0.50):     {b_ap50:.4f}", also_print=False)
        write_log(dataset_log, f"  AP75 (IoU=0.75):     {bbox_metrics['summary']['AP75']:.4f}", also_print=False)
        write_log(dataset_log, f"  AR (maxDet=100):     {b_ar:.4f}", also_print=False)
        write_log(dataset_log, f"  AP (small):          {bbox_metrics['summary']['AP_small']:.4f}", also_print=False)
        write_log(dataset_log, f"  AP (medium):         {bbox_metrics['summary']['AP_medium']:.4f}", also_print=False)
        write_log(dataset_log, f"  AP (large):          {bbox_metrics['summary']['AP_large']:.4f}", also_print=False)
        
        write_log(dataset_log, "\n=== SEGMENTATION METRICS ===", also_print=False)
        write_log(dataset_log, f"  mAP (IoU=0.50:0.95): {s_map:.4f}", also_print=False)
        write_log(dataset_log, f"  AP50 (IoU=0.50):     {s_ap50:.4f}", also_print=False)
        write_log(dataset_log, f"  AP75 (IoU=0.75):     {segm_metrics['summary']['AP75']:.4f}", also_print=False)
        write_log(dataset_log, f"  AR (maxDet=100):     {s_ar:.4f}", also_print=False)
        write_log(dataset_log, f"  AP (small):          {segm_metrics['summary']['AP_small']:.4f}", also_print=False)
        write_log(dataset_log, f"  AP (medium):         {segm_metrics['summary']['AP_medium']:.4f}", also_print=False)
        write_log(dataset_log, f"  AP (large):          {segm_metrics['summary']['AP_large']:.4f}", also_print=False)
        
        write_log(dataset_log, f"\nMetrics saved to: {metrics_path}", also_print=False)
        
        # Print summary to console and global log
        result_line = f"{dataset_name:<25} | {b_map:.3f}    | {b_ap50:.3f}     | {b_ar:.3f}    || {s_map:.3f}    | {s_ap50:.3f}     | {s_ar:.3f}"
        write_log(global_log, result_line)
        
        # Store for summary
        all_results.append({
            'dataset': dataset_name,
            'bbox': {'mAP': b_map, 'AP50': b_ap50, 'AR': b_ar},
            'segm': {'mAP': s_map, 'AP50': s_ap50, 'AR': s_ar},
            'num_images': num_gt_images,
            'num_annotations': num_gt_annotations,
        })
        successful_datasets += 1

    except Exception as e:
        error_msg = f"{dataset_name:<25} | ERROR: {str(e)[:60]}"
        write_log(global_log, error_msg)
        write_log(dataset_log, f"ERROR: {str(e)}", also_print=False)
        failed_datasets += 1
    
    finally:
        dataset_log.close()

# Write summary statistics
write_log(global_log, "")
write_log(global_log, "=" * 105)
write_log(global_log, "SUMMARY STATISTICS")
write_log(global_log, "=" * 105)
write_log(global_log, f"Total datasets processed: {successful_datasets + failed_datasets}")
write_log(global_log, f"Successful: {successful_datasets}")
write_log(global_log, f"Failed: {failed_datasets}")

if all_results:
    # Calculate averages
    avg_bbox_map = np.mean([r['bbox']['mAP'] for r in all_results])
    avg_bbox_ap50 = np.mean([r['bbox']['AP50'] for r in all_results])
    avg_segm_map = np.mean([r['segm']['mAP'] for r in all_results])
    avg_segm_ap50 = np.mean([r['segm']['AP50'] for r in all_results])
    
    write_log(global_log, "")
    write_log(global_log, "AVERAGE METRICS ACROSS ALL DATASETS:")
    write_log(global_log, f"  Bbox mAP:  {avg_bbox_map:.4f}")
    write_log(global_log, f"  Bbox AP50: {avg_bbox_ap50:.4f}")
    write_log(global_log, f"  Segm mAP:  {avg_segm_map:.4f}")
    write_log(global_log, f"  Segm AP50: {avg_segm_ap50:.4f}")
    
    # Save global summary JSON
    global_summary = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'num_datasets': len(all_results),
        'average_metrics': {
            'bbox': {'mAP': avg_bbox_map, 'AP50': avg_bbox_ap50},
            'segm': {'mAP': avg_segm_map, 'AP50': avg_segm_ap50},
        },
        'per_dataset_summary': all_results,
    }
    
    global_summary_path = os.path.join(ROOT_DATASET, 'sam3_metrics_summary.json')
    with open(global_summary_path, 'w') as f:
        json.dump(global_summary, f, indent=2)
    
    write_log(global_log, f"\nGlobal summary saved to: {global_summary_path}")

write_log(global_log, f"Log saved to: {global_log_path}")
global_log.close()

print(f"\n✓ Evaluation complete. Detailed metrics saved to each dataset folder as '{METRICS_OUTPUT_FILENAME}'")
print(f"✓ Logs saved to '{LOG_FILENAME}' in each dataset folder and root folder")