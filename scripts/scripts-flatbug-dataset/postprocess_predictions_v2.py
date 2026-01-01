"""
Post-Process Predictions Script - Apply V2 Enhancements

This script applies V2 enhancements (bbox padding, mask dilation) to existing
SAM3 prediction files without re-running inference.

Since we have the polygon segmentations stored in the predictions, we can:
1. Apply bbox padding (expand bounding boxes by N pixels)
2. Apply mask dilation by:
   - Converting polygon to mask
   - Dilating the mask
   - Converting back to polygon

This allows quick experimentation with V2 parameters on existing predictions.

Usage:
    python postprocess_predictions_v2.py
"""

import os
import sys
import json
import copy
import numpy as np
import cv2
from datetime import datetime
from typing import List, Dict, Any, Tuple

# ==========================
# CONFIGURATION
# ==========================

# Input/Output paths
ROOT_DATASET = "./flatbug-dataset"
DATASET_NAME = "cao2022"
INPUT_PRED_FILE = "sam3_flatbug_strategy-1.json"
OUTPUT_PRED_FILE = "sam3_flatbug_strategy-1_v2_postproc.json"

# V2 Enhancement Parameters
V2_ENABLE_BOX_PADDING = True
V2_BOX_PADDING_PX = 7               # Pixels to pad bounding boxes (like FlatBug's pad=7)
    
V2_ENABLE_MASK_DILATION = True
V2_MASK_DILATION_KERNEL = 12         # Dilation kernel size (odd number: 3,5,7,9,11)

# Additional V2 options (optional)
V2_ENABLE_POLYGON_EXPANSION = False  # Expand polygon outward from centroid
V2_POLYGON_EXPANSION_PX = 4          # Pixels to expand

# ==========================
# HELPER FUNCTIONS
# ==========================

def pad_bbox(bbox: List[float], padding: int, img_w: int, img_h: int) -> List[float]:
    """
    Pad a bounding box by the specified amount, clamping to image bounds.
    
    Args:
        bbox: [x, y, width, height] format bounding box (COCO format)
        padding: Pixels to pad on each side
        img_w, img_h: Image dimensions for clamping
        
    Returns:
        Padded [x, y, width, height] bbox
    """
    x, y, w, h = bbox
    
    if padding <= 0:
        return bbox
    
    # Convert to corner format
    x0, y0 = x, y
    x1, y1 = x + w, y + h
    
    # Expand box
    new_x0 = max(0, x0 - padding)
    new_y0 = max(0, y0 - padding)
    new_x1 = min(img_w, x1 + padding)
    new_y1 = min(img_h, y1 + padding)
    
    # Convert back to COCO format
    return [new_x0, new_y0, new_x1 - new_x0, new_y1 - new_y0]


def polygon_to_mask(polygon: List[float], width: int, height: int) -> np.ndarray:
    """
    Convert a polygon (flat list) to a binary mask.
    
    Args:
        polygon: Flat list [x0, y0, x1, y1, ...]
        width, height: Mask dimensions
        
    Returns:
        Binary mask (0/255)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if len(polygon) < 6:  # Need at least 3 points
        return mask
    
    # Convert to Nx2 array
    pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
    
    # Fill polygon
    cv2.fillPoly(mask, [pts], 255)
    
    return mask


def mask_to_polygon(mask: np.ndarray) -> List[float]:
    """
    Convert a binary mask to a polygon (flat list).
    
    Args:
        mask: Binary mask (0/255)
        
    Returns:
        Flat list [x0, y0, x1, y1, ...]
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return []
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    
    if len(largest) < 3:
        return []
    
    # Simplify slightly to reduce point count
    epsilon = 1.0
    simplified = cv2.approxPolyDP(largest, epsilon, closed=True)
    
    if len(simplified) < 3:
        return []
    
    return simplified.reshape(-1).tolist()


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply morphological dilation to expand a binary mask.
    
    Args:
        mask: Binary mask (0/255)
        kernel_size: Size of the dilation kernel (should be odd: 3, 5, 7, 9, 11)
        
    Returns:
        Dilated binary mask
    """
    if kernel_size <= 0:
        return mask
    
    # Use elliptical kernel for smoother dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    
    return dilated


def expand_polygon(polygon: List[float], expansion_px: float) -> List[float]:
    """
    Expand a polygon outward from its centroid.
    
    Args:
        polygon: Flat list of coordinates [x0, y0, x1, y1, ...]
        expansion_px: Number of pixels to expand outward
        
    Returns:
        Expanded polygon as flat list [x0, y0, x1, y1, ...]
    """
    if expansion_px <= 0 or len(polygon) < 6:
        return polygon
    
    # Convert to Nx2 array
    pts = np.array(polygon).reshape(-1, 2)
    
    # Calculate centroid
    centroid = pts.mean(axis=0)
    
    # Calculate direction from centroid to each point
    directions = pts - centroid
    distances = np.linalg.norm(directions, axis=1, keepdims=True)
    distances = np.maximum(distances, 1e-6)  # Avoid division by zero
    unit_dirs = directions / distances
    
    # Expand outward
    expanded_pts = pts + unit_dirs * expansion_px
    
    return expanded_pts.flatten().tolist()


def clamp_polygon(polygon: List[float], img_w: int, img_h: int) -> List[float]:
    """Clamp polygon coordinates to image bounds."""
    clamped = []
    for i in range(0, len(polygon), 2):
        x = max(0, min(img_w, polygon[i]))
        y = max(0, min(img_h, polygon[i + 1]))
        clamped.extend([x, y])
    return clamped


def compute_area_from_polygon(polygon: List[float]) -> float:
    """Compute area of a polygon using shoelace formula."""
    if len(polygon) < 6:
        return 0.0
    
    pts = np.array(polygon).reshape(-1, 2)
    n = len(pts)
    
    # Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i, 0] * pts[j, 1]
        area -= pts[j, 0] * pts[i, 1]
    
    return abs(area) / 2.0


def compute_bbox_from_polygon(polygon: List[float]) -> List[float]:
    """Compute bounding box from polygon."""
    if len(polygon) < 6:
        return [0, 0, 0, 0]
    
    pts = np.array(polygon).reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


# ==========================
# MAIN POST-PROCESSING
# ==========================

def postprocess_predictions(
    pred_data: Dict[str, Any],
    image_dims: Dict[int, Tuple[int, int]],
    bbox_padding: int = 0,
    mask_dilation: int = 0,
    polygon_expansion: float = 0
) -> Dict[str, Any]:
    """
    Apply V2 enhancements to prediction data.
    
    Args:
        pred_data: Original prediction data (COCO format)
        image_dims: Dict mapping image_id -> (width, height)
        bbox_padding: Pixels to pad bounding boxes
        mask_dilation: Dilation kernel size for masks
        polygon_expansion: Pixels to expand polygons outward
        
    Returns:
        Modified prediction data with V2 enhancements applied
    """
    # Deep copy to avoid modifying original
    new_data = copy.deepcopy(pred_data)
    
    annotations = new_data.get("annotations", [])
    
    processed_count = 0
    dilation_applied = 0
    padding_applied = 0
    expansion_applied = 0
    
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id not in image_dims:
            continue
        
        img_w, img_h = image_dims[image_id]
        
        # Get segmentation polygons
        segmentation = ann.get("segmentation", [])
        
        if not segmentation:
            continue
        
        new_segmentation = []
        
        for poly in segmentation:
            if len(poly) < 6:
                new_segmentation.append(poly)
                continue
            
            processed_poly = poly
            
            # Apply mask dilation if enabled
            if mask_dilation > 0:
                # Convert polygon to mask
                mask = polygon_to_mask(processed_poly, img_w, img_h)
                
                if mask.sum() > 0:
                    # Dilate the mask
                    dilated_mask = dilate_mask(mask, mask_dilation)
                    
                    # Convert back to polygon
                    new_poly = mask_to_polygon(dilated_mask)
                    
                    if len(new_poly) >= 6:
                        processed_poly = new_poly
                        dilation_applied += 1
            
            # Apply polygon expansion if enabled
            if polygon_expansion > 0:
                processed_poly = expand_polygon(processed_poly, polygon_expansion)
                processed_poly = clamp_polygon(processed_poly, img_w, img_h)
                expansion_applied += 1
            
            new_segmentation.append(processed_poly)
        
        ann["segmentation"] = new_segmentation
        
        # Recompute bbox from polygon if dilation/expansion was applied
        if (mask_dilation > 0 or polygon_expansion > 0) and new_segmentation:
            # Use first polygon to compute new bbox
            new_bbox = compute_bbox_from_polygon(new_segmentation[0])
            
            # Apply bbox padding
            if bbox_padding > 0:
                new_bbox = pad_bbox(new_bbox, bbox_padding, img_w, img_h)
                padding_applied += 1
            
            ann["bbox"] = new_bbox
            
            # Recompute area
            ann["area"] = compute_area_from_polygon(new_segmentation[0])
        
        elif bbox_padding > 0:
            # Just apply bbox padding without polygon changes
            old_bbox = ann.get("bbox", [0, 0, 0, 0])
            new_bbox = pad_bbox(old_bbox, bbox_padding, img_w, img_h)
            ann["bbox"] = new_bbox
            padding_applied += 1
            
            # Update area based on new bbox
            ann["area"] = new_bbox[2] * new_bbox[3]
        
        processed_count += 1
    
    print(f"  Processed {processed_count} annotations")
    print(f"  - Dilation applied: {dilation_applied}")
    print(f"  - Bbox padding applied: {padding_applied}")
    print(f"  - Polygon expansion applied: {expansion_applied}")
    
    return new_data


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("Post-Process Predictions - Apply V2 Enhancements")
    print("=" * 60)
    
    # Build paths
    dataset_path = os.path.join(ROOT_DATASET, DATASET_NAME)
    input_path = os.path.join(dataset_path, INPUT_PRED_FILE)
    output_path = os.path.join(dataset_path, OUTPUT_PRED_FILE)
    
    print(f"\nDataset: {DATASET_NAME}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    print(f"\nV2 Parameters:")
    print(f"  - BBOX_PADDING: {V2_BOX_PADDING_PX if V2_ENABLE_BOX_PADDING else 'disabled'}")
    print(f"  - MASK_DILATION: {V2_MASK_DILATION_KERNEL if V2_ENABLE_MASK_DILATION else 'disabled'}")
    print(f"  - POLYGON_EXPANSION: {V2_POLYGON_EXPANSION_PX if V2_ENABLE_POLYGON_EXPANSION else 'disabled'}")
    
    # Load prediction file
    if not os.path.exists(input_path):
        print(f"\n❌ Error: Input file not found: {input_path}")
        return
    
    print(f"\nLoading predictions...")
    with open(input_path, 'r') as f:
        pred_data = json.load(f)
    
    # Extract image dimensions from images list
    images = pred_data.get("images", [])
    image_dims = {}
    for img in images:
        image_dims[img["id"]] = (img["width"], img["height"])
    
    print(f"  Found {len(images)} images")
    print(f"  Found {len(pred_data.get('annotations', []))} annotations")
    
    # Apply post-processing
    print(f"\nApplying V2 enhancements...")
    new_pred_data = postprocess_predictions(
        pred_data,
        image_dims,
        bbox_padding=V2_BOX_PADDING_PX if V2_ENABLE_BOX_PADDING else 0,
        mask_dilation=V2_MASK_DILATION_KERNEL if V2_ENABLE_MASK_DILATION else 0,
        polygon_expansion=V2_POLYGON_EXPANSION_PX if V2_ENABLE_POLYGON_EXPANSION else 0
    )
    
    # Add metadata about post-processing
    new_pred_data["postprocessing"] = {
        "timestamp": datetime.now().isoformat(),
        "source_file": INPUT_PRED_FILE,
        "v2_parameters": {
            "bbox_padding_px": V2_BOX_PADDING_PX if V2_ENABLE_BOX_PADDING else 0,
            "mask_dilation_kernel": V2_MASK_DILATION_KERNEL if V2_ENABLE_MASK_DILATION else 0,
            "polygon_expansion_px": V2_POLYGON_EXPANSION_PX if V2_ENABLE_POLYGON_EXPANSION else 0,
        }
    }
    
    # Save output
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(new_pred_data, f, indent=2)
    
    print(f"\n✓ Done! Post-processed predictions saved to: {output_path}")
    print(f"\nTo evaluate metrics, update metrics-mAP-cpu.py to use:")
    print(f'  PRED_FILENAME = "{OUTPUT_PRED_FILE}"')
    print(f'  ALLOWED_FOLDERS = {{"{DATASET_NAME}"}}')


if __name__ == "__main__":
    main()
