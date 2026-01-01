"""
SAM3 FlatBug Inference Script - Version 2

This script builds on sam3_flatbug_inference.py with refinements discovered from
detailed analysis of FlatBug's source code:

Key Improvements over v1:
1. CHAIN_APPROX_NONE: Use all contour points before simplification (like FlatBug)
   instead of CHAIN_APPROX_SIMPLE which compresses horizontal/vertical segments
   
2. Dynamic polygon simplification tolerance: FlatBug uses a dynamic tolerance
   based on mask-to-image scale: tolerance = (mask_to_image_scale / 2).mean()
   For typical 256->1024 scaling, this gives tolerance ~2.0 instead of fixed 1.0
   
3. Largest contour only: FlatBug uses largest_only=True for contour extraction,
   keeping only the largest contour per mask to avoid fragmented detections

4. Mask dilation: SAM3 produces tighter masks than GT annotations (~50% vs ~64% 
   fill ratio). Optional dilation can expand masks to better match GT style.

5. Polygon expansion: Like FlatBug's expand_by_one, optionally expand polygon 
   outward to increase mask coverage.

6. Box padding: Like FlatBug's pad=5, optionally pad bounding boxes to ensure
   they fully encapsulate the masks.

Configuration parameters (SCORE_THRESHOLD, IOU_THRESHOLD, etc.) remain identical
to ensure fair comparison between FlatBug and SAM3.

Reference: /home/dolma/repo/flat-bug/src/flat_bug/geometric.py
Reference: /home/dolma/repo/flat-bug/src/flat_bug/predictor.py
"""

import sys
import os
import json
import math
from itertools import accumulate
from typing import List, Tuple, Dict, Any, Optional, Union
import torch
import torchvision
import cv2
import numpy as np
import gc
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path for SAM3 imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3_flatbug_config import DEFAULT_CFG, get_cfg, print_cfg

from dotenv import load_dotenv
from huggingface_hub import login

# Load HuggingFace token
load_dotenv()
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)
else:
    print("Warning: No HF_TOKEN found in .env file")

# ==========================
# 1. CONFIGURATION
# ==========================
DEVICE = "cuda:0"
BPE_PATH = "./assets/bpe_simple_vocab_16e6.txt.gz"
ROOT_DATASET = "./flatbug-dataset"

# Datasets to process
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

# Visualization settings
MASK_FILL_COLOR_RGBA = (135, 206, 250, 60)
BBOX_COLOR = "#0051FF"
LABEL_TEXT_COLOR = "black"
# Mask border (visualization)
MASK_BORDER_COLOR_RGBA = (0, 81, 255, 220)  # Blue, mostly opaque
MASK_BORDER_WIDTH = 6                      # Bold border width in pixels

# FlatBug mask size (YOLOv8 architecture uses 256x256 masks)
# SAM3 uses higher resolution masks, but we compute dynamic tolerance similarly
FLATBUG_MASK_SIZE = 256

# ==========================
# V2 ENHANCEMENT OPTIONS
# ==========================
# These options can be toggled on/off and tuned to improve segmentation metrics.
# Based on analysis: SAM3 masks are ~13% smaller than GT (50% vs 64% fill ratio).

# Contour extraction options
V2_USE_CHAIN_APPROX_NONE = True     # Use all contour points (like FlatBug) vs CHAIN_APPROX_SIMPLE
V2_USE_DYNAMIC_TOLERANCE = True     # Dynamic polygon simplification based on scale
V2_LARGEST_CONTOUR_ONLY = True      # Keep only largest contour per mask

# Mask dilation options (to expand SAM3's tighter masks)
V2_ENABLE_MASK_DILATION = True      # Enable/disable mask dilation
V2_MASK_DILATION_KERNEL = 5         # Dilation kernel size (odd number: 3,5,7,9,11)
                                    # Experiments show 7-9px optimal for ALUS dataset

# Polygon expansion options (like FlatBug's expand_by_one)
V2_ENABLE_POLYGON_EXPANSION = False # Enable/disable polygon expansion
V2_POLYGON_EXPANSION_PX = 4         # Pixels to expand polygon outward (2-6 optimal)

# Bounding box padding (like FlatBug's pad=5)
V2_ENABLE_BOX_PADDING = True        # Enable/disable box padding
V2_BOX_PADDING_PX = 5               # Pixels to pad bounding boxes

# Min mask area filter (like FlatBug's masks.sum() < 3 check)
# Filters out tiny noise masks before contour extraction
V2_ENABLE_MIN_MASK_AREA = True      # Enable/disable minimum mask area filter
V2_MIN_MASK_AREA_PIXELS = 3         # Minimum mask area in pixels (FlatBug uses 3)

# Linear interpolation before contour scaling (like FlatBug's linear_interpolate)
# Adds more points along polygon edges before scaling for smoother boundaries
V2_ENABLE_LINEAR_INTERPOLATION = False  # Enable/disable linear interpolation
V2_LINEAR_INTERP_POINTS = 10            # Number of interpolation points per edge

# IoS (Intersection over Smaller) NMS (like FlatBug's ios_masks in nms.py)
# More aggressive at suppressing smaller overlapping masks than standard IoU
V2_USE_IOS_NMS = False              # Use IoS instead of IoU for NMS
                                    # IoS = intersection / min(area1, area2)

# Collect all V2 options into a dict for easy config loading
V2_OPTIONS = {
    "USE_CHAIN_APPROX_NONE": V2_USE_CHAIN_APPROX_NONE,
    "USE_DYNAMIC_TOLERANCE": V2_USE_DYNAMIC_TOLERANCE,
    "LARGEST_CONTOUR_ONLY": V2_LARGEST_CONTOUR_ONLY,
    "MASK_DILATION_PIXELS": V2_MASK_DILATION_KERNEL if V2_ENABLE_MASK_DILATION else 0,
    "POLYGON_EXPANSION_PIXELS": V2_POLYGON_EXPANSION_PX if V2_ENABLE_POLYGON_EXPANSION else 0,
    "BBOX_PADDING_PIXELS": V2_BOX_PADDING_PX if V2_ENABLE_BOX_PADDING else 0,
    "MIN_MASK_AREA_PIXELS": V2_MIN_MASK_AREA_PIXELS if V2_ENABLE_MIN_MASK_AREA else 0,
    "LINEAR_INTERP_POINTS": V2_LINEAR_INTERP_POINTS if V2_ENABLE_LINEAR_INTERPOLATION else 0,
    "USE_IOS_NMS": V2_USE_IOS_NMS,
}


# ==========================
# 2. FLATBUG TILING ALGORITHMS
# ==========================

def equal_allocate_overlaps(total: int, segments: int, size: int) -> List[int]:
    """
    Generates cumulative positions for placing segments with controlled overlaps.
    
    This is a direct port of FlatBug's geometric.equal_allocate_overlaps function.
    
    The overlap is distributed uniformly, with the first few gaps adjusted slightly
    to ensure the segments collectively sum to `total`.
    
    Args:
        total: The total length to be covered by the segments
        segments: The number of segments to place
        size: The size of each segment (tile size)
        
    Returns:
        List of cumulative positions where each segment should be placed
        
    Example:
        >>> equal_allocate_overlaps(1000, 5, 250)
        [0, 187, 374, 562, 750]
    """
    if segments < 2:
        return [0] * segments
    
    overlap = segments * size - total
    partial_overlap, remainder = divmod(overlap, segments - 1)
    distance = size - partial_overlap
    
    return list(accumulate(
        [distance - (1 if i < remainder else 0) for i in range(segments - 1)], 
        initial=0
    ))


def calculate_tile_offsets(
    image_size: Tuple[int, int],
    tile_size: int,
    minimum_overlap: int
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Calculate tile offsets for sliding window with minimum overlap.
    
    This is a direct port of FlatBug's geometric.calculate_tile_offsets function.
    
    Args:
        image_size: (width, height) of the image
        tile_size: Size of each tile (1024)
        minimum_overlap: Minimum overlap between tiles (384 pixels)
        
    Returns:
        List of ((grid_x, grid_y), (pixel_y, pixel_x)) tuples
    """
    w, h = image_size
    
    # Calculate number of tiles needed in each dimension
    x_n_tiles = math.ceil((w - minimum_overlap) / (tile_size - minimum_overlap)) if w != tile_size else 1
    y_n_tiles = math.ceil((h - minimum_overlap) / (tile_size - minimum_overlap)) if h != tile_size else 1
    
    # Get evenly distributed offsets
    x_range = equal_allocate_overlaps(w, x_n_tiles, tile_size)
    y_range = equal_allocate_overlaps(h, y_n_tiles, tile_size)
    
    # Return as list of ((grid_m, grid_n), (y_offset, x_offset))
    return [((m, n), (j, i)) for n, j in enumerate(y_range) for m, i in enumerate(x_range)]


def calculate_pyramid_scales(
    image_w: int, 
    image_h: int, 
    tile_size: int,
    scale_increment: float = 2/3
) -> List[float]:
    """
    Calculate pyramid scales following FlatBug methodology.
    
    Starts from a scale that fits the whole image in one tile,
    then grows by 1/scale_increment (~1.5x) until reaching full resolution.
    
    Args:
        image_w: Image width
        image_h: Image height
        tile_size: Tile size (1024)
        scale_increment: Scale factor between levels (2/3 = 1.5x growth)
        
    Returns:
        Sorted list of scales from smallest to 1.0
    """
    max_dim = max(image_w, image_h)
    scales = []
    
    # Start with global scale (fit whole image in one tile)
    s = tile_size / max_dim
    
    if s >= 1.0:
        return [1.0]
    
    # Grow by ~1.5x steps until reaching 90% of full resolution
    while s <= 0.9:
        scales.append(s)
        s /= scale_increment  # Divide by 2/3 = multiply by 1.5
    
    # Always include full resolution
    scales.append(1.0)
    
    return sorted(scales)


def filter_by_edge_margin(
    boxes: np.ndarray,
    tile_size: int,
    edge_margin: int,
    tile_x: int,
    tile_y: int,
    layer_w: int,
    layer_h: int
) -> np.ndarray:
    """
    Filter out detections that are too close to tile edges (except real image edges).
    
    This replicates FlatBug's edge case filtering logic.
    
    Args:
        boxes: Array of [x0, y0, x1, y1] boxes in tile coordinates
        tile_size: Size of tile
        edge_margin: Margin threshold (16 pixels)
        tile_x, tile_y: Tile position in layer
        layer_w, layer_h: Layer dimensions
        
    Returns:
        Boolean mask of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=bool)
    
    # Determine which edges are real image edges
    is_real_left = (tile_x == 0)
    is_real_top = (tile_y == 0)
    is_real_right = (tile_x + tile_size >= layer_w)
    is_real_bottom = (tile_y + tile_size >= layer_h)
    
    keep = np.ones(len(boxes), dtype=bool)
    
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        
        # Check if box touches each edge
        touches_left = (x0 < edge_margin)
        touches_top = (y0 < edge_margin)
        touches_right = (x1 > tile_size - edge_margin)
        touches_bottom = (y1 > tile_size - edge_margin)
        
        # Reject if touches a tile edge that's not a real image edge
        if touches_left and not is_real_left:
            keep[i] = False
        elif touches_top and not is_real_top:
            keep[i] = False
        elif touches_right and not is_real_right:
            keep[i] = False
        elif touches_bottom and not is_real_bottom:
            keep[i] = False
    
    return keep


def filter_by_image_boundary(
    boxes: np.ndarray,
    image_w: int,
    image_h: int,
    margin: int
) -> np.ndarray:
    """
    Filter out detections that touch the actual image boundaries.
    
    This removes objects that are likely truncated/clipped at the image edge,
    which are often partial insects or artifacts.
    
    Args:
        boxes: Array of [x0, y0, x1, y1] boxes in global image coordinates
        image_w: Image width
        image_h: Image height
        margin: Distance from edge to consider as "touching" boundary
        
    Returns:
        Boolean mask of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=bool)
    
    if margin <= 0:
        return np.ones(len(boxes), dtype=bool)
    
    keep = np.ones(len(boxes), dtype=bool)
    
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        
        # Check if box touches any image boundary
        touches_left = (x0 < margin)
        touches_top = (y0 < margin)
        touches_right = (x1 > image_w - margin)
        touches_bottom = (y1 > image_h - margin)
        
        # Reject if touches any boundary
        if touches_left or touches_top or touches_right or touches_bottom:
            keep[i] = False
    
    return keep


def filter_by_object_size(
    boxes: np.ndarray,
    min_size: float,
    max_size: float
) -> np.ndarray:
    """
    Filter boxes by object size (sqrt of bbox area).
    
    This replicates FlatBug's MIN_MAX_OBJ_SIZE filtering.
    
    Args:
        boxes: Array of [x0, y0, x1, y1] boxes
        min_size: Minimum sqrt(area) threshold (32)
        max_size: Maximum sqrt(area) threshold (10^8)
        
    Returns:
        Boolean mask of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=bool)
    
    # Calculate sqrt(area) for each box
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    sizes = np.sqrt(areas)
    
    # Filter by size range
    keep = (sizes >= min_size) & (sizes <= max_size)
    
    return keep


def simplify_contour(contour: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
    """
    Simplify a contour using cv2.approxPolyDP.
    
    This is a direct port of FlatBug's geometric.simplify_contour function.
    
    Args:
        contour: Contour array of shape (N, 1, 2) or (N, 2)
        tolerance: Approximation tolerance (epsilon for cv2.approxPolyDP)
        
    Returns:
        Simplified contour of shape (M, 1, 2)
    """
    if len(contour) == 0:
        return contour
    
    # Ensure proper shape for cv2.approxPolyDP
    if contour.ndim == 2:
        contour = contour.reshape(-1, 1, 2)
    
    simplified = cv2.approxPolyDP(contour.astype(np.float32), epsilon=tolerance, closed=True)
    
    return simplified


def find_contours_flatbug(
    mask: np.ndarray,
    largest_only: bool = True,
    simplify: bool = False,
    tolerance: float = 1.0,
    use_chain_approx_none: bool = True
) -> List[np.ndarray]:
    """
    Find contours in a binary mask using FlatBug's methodology.
    
    Key differences from v1:
    - Uses CHAIN_APPROX_NONE to keep all contour points (like FlatBug)
    - Optionally keeps only the largest contour
    - Optionally applies simplification with given tolerance
    
    This is based on FlatBug's geometric.find_contours function.
    
    Args:
        mask: Binary mask (0/255 or 0/1)
        largest_only: If True, return only the largest contour by area
        simplify: If True, apply polygon simplification
        tolerance: Simplification tolerance (if simplify=True)
        use_chain_approx_none: If True, use CHAIN_APPROX_NONE (FlatBug style),
                               otherwise use CHAIN_APPROX_SIMPLE (SAM3 v1 style)
        
    Returns:
        List of contour arrays, each of shape (N, 1, 2)
    """
    # Ensure mask is uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Ensure mask is binary (0 or 255)
    if mask.max() == 1:
        mask = mask * 255
    
    # Find contours - V2 can use CHAIN_APPROX_NONE (keeps all points, like FlatBug)
    # or CHAIN_APPROX_SIMPLE (v1 behavior) for comparison
    approx_method = cv2.CHAIN_APPROX_NONE if use_chain_approx_none else cv2.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, approx_method)
    
    if len(contours) == 0:
        return []
    
    # Optionally keep only the largest contour
    if largest_only and len(contours) > 1:
        areas = [cv2.contourArea(c) for c in contours]
        largest_idx = np.argmax(areas)
        contours = [contours[largest_idx]]
    
    # Optionally simplify contours
    if simplify:
        contours = [simplify_contour(c, tolerance) for c in contours]
    
    return contours


def calculate_dynamic_tolerance(
    mask_height: int,
    mask_width: int,
    image_height: int,
    image_width: int
) -> float:
    """
    Calculate dynamic polygon simplification tolerance based on mask-to-image scale.
    
    This replicates FlatBug's contour_to_image_coordinates which uses:
        tolerance = (mask_to_image_scale / 2).mean()
    
    For FlatBug's 256x256 masks on 1024x1024 images:
        scale = (1023/255, 1023/255) ≈ (4.0, 4.0)
        tolerance = (4.0 / 2 + 4.0 / 2) / 2 = 2.0
    
    Args:
        mask_height: Height of the mask
        mask_width: Width of the mask
        image_height: Height of the target image
        image_width: Width of the target image
        
    Returns:
        Simplification tolerance value
    """
    # Calculate scale factors
    scale_h = (image_height - 1) / max(mask_height - 1, 1)
    scale_w = (image_width - 1) / max(mask_width - 1, 1)
    
    # Average scale divided by 2 (like FlatBug)
    tolerance = (scale_h + scale_w) / 2 / 2
    
    # Ensure minimum tolerance of 1.0
    return max(tolerance, 1.0)


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply morphological dilation to expand a binary mask.
    
    SAM3 produces tighter masks than GT annotations (~50% vs ~64% fill ratio).
    Dilation can help expand the masks to better match GT annotation style.
    
    Args:
        mask: Binary mask (0/1 or 0/255)
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
    
    This is similar to FlatBug's expand_by_one but allows configurable expansion.
    Each point is moved away from the centroid by the specified number of pixels.
    
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


def linear_interpolate_polygon(polygon: List[float], n_interp: int = 10) -> List[float]:
    """
    Add interpolated points along polygon edges for smoother scaling.
    
    This replicates FlatBug's linear_interpolate function from geometric.py.
    Before scaling a contour, FlatBug adds more points along each edge to ensure
    that the scaled polygon has smooth boundaries.
    
    Args:
        polygon: Flat list of coordinates [x0, y0, x1, y1, ...]
        n_interp: Number of interpolation points to add per edge (FlatBug uses ~10)
        
    Returns:
        Interpolated polygon as flat list with more points
    """
    if n_interp <= 0 or len(polygon) < 6:
        return polygon
    
    # Convert to Nx2 array
    pts = np.array(polygon).reshape(-1, 2)
    n_pts = len(pts)
    
    # Interpolate between each pair of consecutive points
    interpolated = []
    for i in range(n_pts):
        p1 = pts[i]
        p2 = pts[(i + 1) % n_pts]  # Wrap around for closed polygon
        
        # Add the original point
        interpolated.append(p1)
        
        # Add n_interp points between p1 and p2
        for j in range(1, n_interp + 1):
            t = j / (n_interp + 1)
            interp_pt = p1 + t * (p2 - p1)
            interpolated.append(interp_pt)
    
    result = np.array(interpolated).flatten().tolist()
    return result


def check_min_mask_area(mask: np.ndarray, min_area: int = 3) -> bool:
    """
    Check if a mask has sufficient pixel area.
    
    This replicates FlatBug's check in yolo_helpers.py:416-419:
        if masks.sum() < 3:
            continue  # Skip tiny noise masks
    
    Args:
        mask: Binary mask (0/1 or 0/255)
        min_area: Minimum number of pixels (FlatBug uses 3)
        
    Returns:
        True if mask has sufficient area, False otherwise
    """
    # Normalize to binary
    if mask.max() > 1:
        mask_binary = (mask > 127).astype(np.uint8)
    else:
        mask_binary = mask.astype(np.uint8)
    
    return mask_binary.sum() >= min_area


def compute_ios_matrix(boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoS (Intersection over Smaller) matrix for a set of boxes.
    
    This replicates FlatBug's ios_masks function from nms.py:170-198.
    IoS = intersection_area / min(area1, area2)
    
    Unlike IoU, IoS is asymmetric and more aggressively suppresses smaller
    boxes that are contained within larger boxes.
    
    Args:
        boxes: Nx4 array of boxes in [x0, y0, x1, y1] format
        
    Returns:
        NxN IoS matrix
    """
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 0))
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    ios_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            # Compute intersection
            ix1 = max(x1[i], x1[j])
            iy1 = max(y1[i], y1[j])
            ix2 = min(x2[i], x2[j])
            iy2 = min(y2[i], y2[j])
            
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
            else:
                intersection = 0.0
            
            # IoS = intersection / min(area1, area2)
            min_area = min(areas[i], areas[j])
            if min_area > 0:
                ios = intersection / min_area
            else:
                ios = 0.0
            
            ios_matrix[i, j] = ios
            ios_matrix[j, i] = ios
    
    return ios_matrix


def nms_ios(boxes: np.ndarray, scores: np.ndarray, ios_threshold: float = 0.2) -> np.ndarray:
    """
    Non-Maximum Suppression using IoS (Intersection over Smaller).
    
    This replicates FlatBug's IoS-based NMS strategy. Unlike standard IoU NMS,
    IoS more aggressively suppresses smaller boxes that overlap with larger ones.
    
    Args:
        boxes: Nx4 array of boxes in [x0, y0, x1, y1] format
        scores: N array of confidence scores
        ios_threshold: IoS threshold for suppression (FlatBug uses 0.2)
        
    Returns:
        Array of indices to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    
    # Sort by score (descending)
    order = np.argsort(scores)[::-1]
    
    keep = []
    suppressed = set()
    
    # Compute IoS matrix
    ios_matrix = compute_ios_matrix(boxes)
    
    for idx in order:
        if idx in suppressed:
            continue
        
        keep.append(idx)
        
        # Suppress boxes with high IoS overlap
        for other_idx in order:
            if other_idx not in suppressed and other_idx != idx:
                if ios_matrix[idx, other_idx] > ios_threshold:
                    suppressed.add(other_idx)
    
    return np.array(keep, dtype=np.int64)


def pad_bbox(bbox: List[float], padding: int, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    Pad a bounding box by the specified amount, clamping to image bounds.
    
    Like FlatBug's pad=5 in offset_scale_pad, this ensures boxes fully
    encapsulate the masks.
    
    Args:
        bbox: [x0, y0, x1, y1] format bounding box (corners)
        padding: Pixels to pad on each side
        img_w, img_h: Image dimensions for clamping
        
    Returns:
        Padded (x0, y0, x1, y1) tuple
    """
    x0, y0, x1, y1 = bbox
    
    if padding <= 0:
        return x0, y0, x1, y1
    
    # Expand box
    new_x0 = max(0, x0 - padding)
    new_y0 = max(0, y0 - padding)
    new_x1 = min(img_w, x1 + padding)
    new_y1 = min(img_h, y1 + padding)
    
    return new_x0, new_y0, new_x1, new_y1


def mask_to_polygon_v2(
    mask_uint8: np.ndarray, 
    x_off: int, 
    y_off: int, 
    scale: float,
    tile_size: int = 1024,
    use_dynamic_tolerance: bool = True,
    largest_only: bool = True,
    use_chain_approx_none: bool = True
) -> List[List[float]]:
    """
    Convert binary mask to polygon(s) in global coordinates.
    
    Version 2 improvements:
    1. Uses CHAIN_APPROX_NONE (all contour points) instead of CHAIN_APPROX_SIMPLE
    2. Dynamic tolerance based on mask-to-image scale (like FlatBug)
    3. Optionally keeps only largest contour to avoid fragmented detections
    
    Args:
        mask_uint8: Binary mask (0/1 or 0/255)
        x_off, y_off: Tile offset in layer coordinates
        scale: Current pyramid scale
        tile_size: Tile size (for tolerance calculation)
        use_dynamic_tolerance: If True, calculate tolerance dynamically
        largest_only: If True, keep only the largest contour
        use_chain_approx_none: If True, use CHAIN_APPROX_NONE (FlatBug style)
        
    Returns:
        List of polygon coordinate lists [x0, y0, x1, y1, ...]
    """
    # Calculate dynamic tolerance based on mask-to-image scale
    # For SAM3, the mask is typically same size as tile (1024x1024)
    # but we're converting to global coordinates, so scale matters
    if use_dynamic_tolerance:
        # Approximate FlatBug's behavior:
        # FlatBug has 256x256 masks -> 1024x1024 tiles = scale ~4
        # tolerance = scale / 2 = 2.0
        # For SAM3 with 1024x1024 masks on 1024x1024 tiles at scale 1.0:
        # We need to account for the global scale factor
        mask_h, mask_w = mask_uint8.shape[:2]
        # Target image size in global coords is tile_size / scale
        global_tile_size = tile_size / scale
        tolerance = calculate_dynamic_tolerance(mask_h, mask_w, int(global_tile_size), int(global_tile_size))
    else:
        tolerance = 1.0
    
    # Find contours using FlatBug methodology
    contours = find_contours_flatbug(
        mask_uint8, 
        largest_only=largest_only,
        simplify=False,  # We'll simplify after coordinate conversion
        use_chain_approx_none=use_chain_approx_none
    )
    
    polygons = []
    
    for cnt in contours:
        if len(cnt) < 3:
            continue
        
        # Reshape to (N, 2)
        points = cnt.reshape(-1, 2).astype(np.float64)
        
        # Convert to global coordinates
        points[:, 0] = (points[:, 0] + x_off) / scale
        points[:, 1] = (points[:, 1] + y_off) / scale
        points = np.maximum(points, 0)
        
        # Simplify polygon with dynamic tolerance (after coordinate conversion, like FlatBug)
        simplified = cv2.approxPolyDP(points.astype(np.float32).reshape(-1, 1, 2), 
                                       epsilon=tolerance, closed=True)
        
        if len(simplified) >= 3:
            polygons.append(simplified.reshape(-1).tolist())
    
    return polygons


def mask_to_bbox(mask_uint8: np.ndarray, x_off: int, y_off: int, scale: float) -> List[float]:
    """
    Convert binary mask to bounding box in global coordinates.
    
    Args:
        mask_uint8: Binary mask
        x_off, y_off: Tile offset in layer coordinates
        scale: Current pyramid scale
        
    Returns:
        [x, y, width, height] in global coordinates
    """
    x, y, w, h = cv2.boundingRect(mask_uint8)
    gx = (x + x_off) / scale
    gy = (y + y_off) / scale
    gw = w / scale
    gh = h / scale
    return [gx, gy, gw, gh]


# ==========================
# 3. SAM3 INFERENCE
# ==========================

def run_sam3_inference(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
    score_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run SAM3 inference on a single tile.
    
    Args:
        processor: SAM3 processor
        image: PIL Image tile
        prompt: Text prompt
        score_threshold: Minimum confidence threshold
        
    Returns:
        Tuple of (boxes, scores, masks) arrays filtered by threshold
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.inference_mode():
            state = processor.set_image(image)
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt, state)
    
    masks = state.get("masks", [])
    boxes = state.get("boxes", [])
    scores = state.get("scores", [])
    
    if len(boxes) > 0:
        b = boxes.float().detach().cpu().numpy()
        s = scores.float().detach().cpu().numpy()
        m = masks.float().detach().cpu().numpy()
        
        # Safe squeeze - only remove channel dim, not batch dim
        if m.ndim == 4:
            m = m.squeeze(1)
        
        # Ensure scores are flat
        s = s.flatten()
        
        # Filter by score threshold
        keep = s > score_threshold
        return b[keep], s[keep], m[keep]
    
    return np.array([]), np.array([]), np.array([])


def pad_image(image: Image.Image, padding: int) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Add padding to image borders (like FlatBug's padding_for_edge_cases).
    
    Args:
        image: Input PIL Image
        padding: Padding amount in pixels
        
    Returns:
        Tuple of (padded_image, (left, right, top, bottom) padding)
    """
    w, h = image.size
    new_w = w + 2 * padding
    new_h = h + 2 * padding
    
    # Create new image with black padding
    padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded.paste(image, (padding, padding))
    
    return padded, (padding, padding, padding, padding)


def remove_padding_from_boxes(
    boxes: np.ndarray,
    padding: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Remove padding offset from box coordinates.
    
    Args:
        boxes: Array of [x0, y0, x1, y1] boxes
        padding: (left, right, top, bottom) padding
        
    Returns:
        Boxes with padding offset removed
    """
    if len(boxes) == 0:
        return boxes
    
    boxes = boxes.copy()
    boxes[:, 0] -= padding[0]  # x0 - left
    boxes[:, 1] -= padding[2]  # y0 - top
    boxes[:, 2] -= padding[0]  # x1 - left
    boxes[:, 3] -= padding[2]  # y1 - top
    
    return boxes


# ==========================
# 4. MAIN PIPELINE
# ==========================

def process_image(
    processor: Sam3Processor,
    image_path: str,
    cfg: Dict[str, Any],
    device: str = "cuda:0"
) -> Tuple[List[Dict], Dict]:
    """
    Process a single image using FlatBug methodology with v2 improvements.
    
    Args:
        processor: SAM3 processor
        image_path: Path to input image
        cfg: Configuration dictionary
        device: CUDA device
        
    Returns:
        Tuple of (annotations_list, image_info_dict)
    """
    # Load configuration values
    TILE_SIZE = cfg["TILE_SIZE"]
    MINIMUM_TILE_OVERLAP = cfg["MINIMUM_TILE_OVERLAP"]
    EDGE_CASE_MARGIN = cfg["EDGE_CASE_MARGIN"]
    IMAGE_BOUNDARY_MARGIN = cfg.get("IMAGE_BOUNDARY_MARGIN", 10)  # Default 10px
    SCORE_THRESHOLD = cfg["SCORE_THRESHOLD"]
    IOU_THRESHOLD = cfg["IOU_THRESHOLD"]
    MIN_SIZE, MAX_SIZE = cfg["MIN_MAX_OBJ_SIZE"]
    SCALE_INCREMENT = cfg["SCALE_INCREMENT"]
    PADDING = cfg["PADDING"]
    PROMPT_PLURAL = cfg["PROMPT_PLURAL"]
    PROMPT_SINGULAR = cfg["PROMPT_SINGULAR"]
    
    # V2 enhancement options (from cfg, set by main() from V2_OPTIONS)
    USE_DYNAMIC_TOLERANCE = cfg.get("USE_DYNAMIC_TOLERANCE", V2_OPTIONS["USE_DYNAMIC_TOLERANCE"])
    LARGEST_CONTOUR_ONLY = cfg.get("LARGEST_CONTOUR_ONLY", V2_OPTIONS["LARGEST_CONTOUR_ONLY"])
    USE_CHAIN_APPROX_NONE = cfg.get("USE_CHAIN_APPROX_NONE", V2_OPTIONS["USE_CHAIN_APPROX_NONE"])
    MASK_DILATION = cfg.get("MASK_DILATION_PIXELS", V2_OPTIONS["MASK_DILATION_PIXELS"])
    POLYGON_EXPANSION = cfg.get("POLYGON_EXPANSION_PIXELS", V2_OPTIONS["POLYGON_EXPANSION_PIXELS"])
    BBOX_PADDING = cfg.get("BBOX_PADDING_PIXELS", V2_OPTIONS["BBOX_PADDING_PIXELS"])
    MIN_MASK_AREA = cfg.get("MIN_MASK_AREA_PIXELS", V2_OPTIONS["MIN_MASK_AREA_PIXELS"])
    LINEAR_INTERP_POINTS = cfg.get("LINEAR_INTERP_POINTS", V2_OPTIONS["LINEAR_INTERP_POINTS"])
    USE_IOS_NMS = cfg.get("USE_IOS_NMS", V2_OPTIONS["USE_IOS_NMS"])
    
    # Load image
    orig_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_image.size
    
    # Add padding to handle edge cases (like FlatBug)
    padded_image, pad_lrtb = pad_image(orig_image, PADDING)
    padded_w, padded_h = padded_image.size
    
    # Calculate pyramid scales
    scales = calculate_pyramid_scales(padded_w, padded_h, TILE_SIZE, SCALE_INCREMENT)
    
    print(f"   Scales: {[f'{s:.3f}' for s in scales]}")
    
    all_boxes = []
    all_scores = []
    all_masks_info = []
    
    # Process each scale level (from largest scale to smallest, like FlatBug)
    for scale in reversed(scales):
        is_max_scale = (scale == min(scales))  # Global view
        
        print(f"   > Scale {scale:.3f}...", end=" ", flush=True)
        
        # Choose prompt based on scale
        current_prompt = PROMPT_SINGULAR if is_max_scale else PROMPT_PLURAL
        
        # Resize image for this scale level
        if scale == 1.0:
            layer_img = padded_image
        else:
            new_w = round(padded_w * scale / 4) * 4  # Round to multiple of 4 (like FlatBug)
            new_h = round(padded_h * scale / 4) * 4
            layer_img = padded_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        layer_w, layer_h = layer_img.size
        
        # Calculate tile offsets using FlatBug algorithm
        offsets = calculate_tile_offsets(
            image_size=(layer_w, layer_h),
            tile_size=TILE_SIZE,
            minimum_overlap=int(MINIMUM_TILE_OVERLAP * scale)  # Scale overlap with image
        )
        
        tiles_count = 0
        for (grid_m, grid_n), (tile_y, tile_x) in offsets:
            tiles_count += 1
            
            # Extract tile
            x_end = min(tile_x + TILE_SIZE, layer_w)
            y_end = min(tile_y + TILE_SIZE, layer_h)
            tile = layer_img.crop((tile_x, tile_y, x_end, y_end))
            
            # Pad tile if needed (for edge tiles)
            if tile.size != (TILE_SIZE, TILE_SIZE):
                padded_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
                padded_tile.paste(tile, (0, 0))
                tile = padded_tile
            
            # Run inference
            t_boxes, t_scores, t_masks = run_sam3_inference(
                processor, tile, current_prompt, SCORE_THRESHOLD
            )
            
            if len(t_boxes) == 0:
                continue
            
            # Apply edge case margin filter (unless at max scale)
            edge_margin = 0 if is_max_scale else EDGE_CASE_MARGIN
            edge_keep = filter_by_edge_margin(
                t_boxes, TILE_SIZE, edge_margin,
                tile_x, tile_y, layer_w, layer_h
            )
            
            t_boxes = t_boxes[edge_keep]
            t_scores = t_scores[edge_keep]
            t_masks = t_masks[edge_keep]
            
            if len(t_boxes) == 0:
                continue
            
            # Apply object size filter (unless at max scale where we keep large objects)
            max_obj_size = 1e9 if is_max_scale else MAX_SIZE
            size_keep = filter_by_object_size(t_boxes, MIN_SIZE, max_obj_size)
            
            t_boxes = t_boxes[size_keep]
            t_scores = t_scores[size_keep]
            t_masks = t_masks[size_keep]
            
            # Project to global coordinates
            for i in range(len(t_boxes)):
                x0, y0, x1, y1 = t_boxes[i]
                
                # Convert to global coordinates (accounting for scale and padding)
                gx0 = (x0 + tile_x) / scale - PADDING
                gy0 = (y0 + tile_y) / scale - PADDING
                gx1 = (x1 + tile_x) / scale - PADDING
                gy1 = (y1 + tile_y) / scale - PADDING
                
                # Clamp to original image bounds
                gx0 = max(0, gx0)
                gy0 = max(0, gy0)
                gx1 = min(orig_w, gx1)
                gy1 = min(orig_h, gy1)
                
                mask_bin = (t_masks[i] > 0.5).astype(np.uint8)
                
                # V2: Skip masks that are too small (like FlatBug's masks.sum() < 3)
                if MIN_MASK_AREA > 0 and not check_min_mask_area(mask_bin, MIN_MASK_AREA):
                    continue
                
                # V2: Apply mask dilation if configured
                if MASK_DILATION > 0:
                    mask_bin = dilate_mask(mask_bin, MASK_DILATION)
                
                all_boxes.append([gx0, gy0, gx1, gy1])
                all_scores.append(float(t_scores[i]))
                
                all_masks_info.append({
                    "mask": mask_bin,
                    "x_off": tile_x,
                    "y_off": tile_y,
                    "scale": scale,
                    "padding": PADDING,
                    "tile_size": TILE_SIZE
                })
        
        if scale != 1.0:
            del layer_img
        gc.collect()
        print(f"Done ({tiles_count} tiles, {len(all_boxes)} candidates)")
    
    # Apply NMS with FlatBug's IoU threshold (0.2 = aggressive)
    print(f"   > Total candidates: {len(all_boxes)}...", end=" ")
    
    if len(all_boxes) > 0:
        boxes_arr = np.array(all_boxes, dtype=np.float32)
        scores_arr = np.array(all_scores, dtype=np.float32)
        
        # V2: Choose between IoS NMS (FlatBug-style) or standard IoU NMS
        if USE_IOS_NMS:
            # Use IoS (Intersection over Smaller) NMS like FlatBug
            keep_indices = nms_ios(boxes_arr, scores_arr, IOU_THRESHOLD)
            nms_type = "IoS NMS"
        else:
            # Use standard torchvision IoU NMS
            boxes_t = torch.tensor(boxes_arr, dtype=torch.float32).to(device)
            scores_t = torch.tensor(scores_arr, dtype=torch.float32).to(device)
            keep_indices = torchvision.ops.nms(boxes_t, scores_t, IOU_THRESHOLD)
            keep_indices = keep_indices.cpu().numpy()
            nms_type = "IoU NMS"
        
        final_boxes = np.array([all_boxes[i] for i in keep_indices])
        final_scores = np.array([all_scores[i] for i in keep_indices])
        final_masks = [all_masks_info[i] for i in keep_indices]
        
        print(f"-> {len(final_boxes)} (after {nms_type})", end=" ")
        
        # Apply image boundary filter to remove truncated objects at image edges
        if IMAGE_BOUNDARY_MARGIN > 0 and len(final_boxes) > 0:
            boundary_keep = filter_by_image_boundary(
                final_boxes, orig_w, orig_h, IMAGE_BOUNDARY_MARGIN
            )
            final_boxes = final_boxes[boundary_keep]
            final_scores = final_scores[boundary_keep]
            final_masks = [final_masks[i] for i, keep in enumerate(boundary_keep) if keep]
            print(f"-> {len(final_boxes)} (after boundary filter)")
        else:
            print()
    else:
        final_boxes = np.array([])
        final_scores = np.array([])
        final_masks = []
        print("-> 0")
    
    # Build annotations
    annotations = []
    for idx in range(len(final_boxes)):
        box = final_boxes[idx]
        score = final_scores[idx]
        m_info = final_masks[idx]
        
        mask_uint8 = m_info["mask"]
        
        # Generate polygon from mask using v2 methodology
        polys = mask_to_polygon_v2(
            mask_uint8, 
            m_info["x_off"], 
            m_info["y_off"], 
            m_info["scale"],
            tile_size=m_info["tile_size"],
            use_dynamic_tolerance=USE_DYNAMIC_TOLERANCE,
            largest_only=LARGEST_CONTOUR_ONLY,
            use_chain_approx_none=USE_CHAIN_APPROX_NONE
        )
        
        # Adjust polygon coordinates for padding
        adjusted_polys = []
        for poly in polys:
            adjusted = []
            for i in range(0, len(poly), 2):
                x = float(poly[i] - m_info["padding"])
                y = float(poly[i+1] - m_info["padding"])
                x = max(0.0, min(float(orig_w), x))
                y = max(0.0, min(float(orig_h), y))
                adjusted.extend([x, y])
            if len(adjusted) >= 6:  # At least 3 points
                # V2: Apply linear interpolation before scaling/expansion (like FlatBug)
                if LINEAR_INTERP_POINTS > 0:
                    adjusted = linear_interpolate_polygon(adjusted, LINEAR_INTERP_POINTS)
                
                # V2: Apply polygon expansion if configured
                if POLYGON_EXPANSION > 0:
                    adjusted = expand_polygon(adjusted, POLYGON_EXPANSION)
                    # Re-clamp after expansion
                    clamped = []
                    for j in range(0, len(adjusted), 2):
                        px = max(0.0, min(float(orig_w), adjusted[j]))
                        py = max(0.0, min(float(orig_h), adjusted[j+1]))
                        clamped.extend([px, py])
                    adjusted = clamped
                adjusted_polys.append(adjusted)
        
        if not adjusted_polys:
            continue
        
        # Calculate bbox from final boxes (ensure native Python floats)
        x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        
        # V2: Apply bbox padding if configured (like FlatBug's pad=5)
        if BBOX_PADDING > 0:
            x0, y0, x1, y1 = pad_bbox([x0, y0, x1, y1], BBOX_PADDING, orig_w, orig_h)
        
        bbox = [x0, y0, x1 - x0, y1 - y0]  # [x, y, w, h]
        area = (x1 - x0) * (y1 - y0)
        
        annotations.append({
            "bbox": bbox,
            "segmentation": adjusted_polys,
            "area": float(area),
            "score": float(score),
        })
    
    image_info = {
        "file_name": os.path.basename(image_path),
        "width": orig_w,
        "height": orig_h,
    }
    
    return annotations, image_info


def main():
    """Main entry point."""
    # Load configuration
    cfg = get_cfg()
    
    # Add v2 options to config (from V2_OPTIONS at top of file)
    cfg["USE_DYNAMIC_TOLERANCE"] = V2_OPTIONS["USE_DYNAMIC_TOLERANCE"]
    cfg["LARGEST_CONTOUR_ONLY"] = V2_OPTIONS["LARGEST_CONTOUR_ONLY"]
    cfg["MASK_DILATION_PIXELS"] = V2_OPTIONS["MASK_DILATION_PIXELS"]
    cfg["POLYGON_EXPANSION_PIXELS"] = V2_OPTIONS["POLYGON_EXPANSION_PIXELS"]
    cfg["BBOX_PADDING_PIXELS"] = V2_OPTIONS["BBOX_PADDING_PIXELS"]
    cfg["USE_CHAIN_APPROX_NONE"] = V2_OPTIONS["USE_CHAIN_APPROX_NONE"]
    cfg["MIN_MASK_AREA_PIXELS"] = V2_OPTIONS["MIN_MASK_AREA_PIXELS"]
    cfg["LINEAR_INTERP_POINTS"] = V2_OPTIONS["LINEAR_INTERP_POINTS"]
    cfg["USE_IOS_NMS"] = V2_OPTIONS["USE_IOS_NMS"]
    
    print("\n" + "="*60)
    print("SAM3 FlatBug Inference Script - VERSION 2")
    print("="*60)
    print("\nV2 Configuration:")
    print(f"  - USE_CHAIN_APPROX_NONE: {cfg['USE_CHAIN_APPROX_NONE']} (keep all contour points)")
    print(f"  - USE_DYNAMIC_TOLERANCE: {cfg['USE_DYNAMIC_TOLERANCE']} (scale-based simplification)")
    print(f"  - LARGEST_CONTOUR_ONLY: {cfg['LARGEST_CONTOUR_ONLY']} (avoid fragmented detections)")
    print(f"  - MASK_DILATION_PIXELS: {cfg['MASK_DILATION_PIXELS']} (expand masks before polygon)")
    print(f"  - POLYGON_EXPANSION_PIXELS: {cfg['POLYGON_EXPANSION_PIXELS']} (expand polygons outward)")
    print(f"  - BBOX_PADDING_PIXELS: {cfg['BBOX_PADDING_PIXELS']} (pad bounding boxes)")
    print(f"  - MIN_MASK_AREA_PIXELS: {cfg['MIN_MASK_AREA_PIXELS']} (filter tiny masks)")
    print(f"  - LINEAR_INTERP_POINTS: {cfg['LINEAR_INTERP_POINTS']} (interpolate polygon edges)")
    print(f"  - USE_IOS_NMS: {cfg['USE_IOS_NMS']} (IoS instead of IoU for NMS)")
    print()
    
    print_cfg(cfg)
    
    # Initialize SAM3 model
    print("\nLoading SAM3 Model...")
    model = build_sam3_image_model(bpe_path=BPE_PATH)
    model.to(DEVICE)
    model.eval()
    processor = Sam3Processor(model, device=DEVICE, confidence_threshold=cfg["SCORE_THRESHOLD"])
    print("Model Loaded.\n")
    
    # Load font for visualization
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Process each dataset
    for dataset_name in sorted(os.listdir(ROOT_DATASET)):
        if dataset_name not in ALLOWED_FOLDERS:
            continue
        
        dataset_path = os.path.join(ROOT_DATASET, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        output_json = os.path.join(dataset_path, "sam3_flatbug_strategy_v2.json")
        output_img_dir = os.path.join(dataset_path, "sam3_flatbug_strategy_v2")
        os.makedirs(output_img_dir, exist_ok=True)
        
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": [{"id": cfg["CATEGORY_ID"], "name": cfg["PROMPT_PLURAL"]}]
        }
        
        ann_id = 1
        img_id = 1
        
        # Get list of images
        image_files = [f for f in os.listdir(dataset_path) 
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        for filename in sorted(image_files):
            img_path = os.path.join(dataset_path, filename)
            
            try:
                print(f"\n[{img_id}] {filename}")
                
                annotations, image_info = process_image(
                    processor, img_path, cfg, DEVICE
                )
                
                # Add to COCO output
                coco_output["images"].append({
                    "id": img_id,
                    "file_name": image_info["file_name"],
                    "width": image_info["width"],
                    "height": image_info["height"]
                })
                
                # Visualization
                orig_image = Image.open(img_path).convert("RGBA")
                mask_layer = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
                mask_draw = ImageDraw.Draw(mask_layer)
                
                for ann in annotations:
                    # Draw polygons
                    for poly in ann["segmentation"]:
                        if len(poly) >= 6:
                            poly_tuples = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                            # Fill polygon
                            mask_draw.polygon(poly_tuples, fill=MASK_FILL_COLOR_RGBA)
                            # Draw bold border by stroking the polygon path
                            try:
                                # Ensure the polygon is closed by repeating first point
                                border_path = poly_tuples + [poly_tuples[0]]
                                mask_draw.line(border_path, fill=MASK_BORDER_COLOR_RGBA, width=MASK_BORDER_WIDTH)
                            except Exception:
                                # Fallback: ignore border drawing errors
                                pass
                    
                    # Add annotation to COCO
                    coco_output["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cfg["CATEGORY_ID"],
                        "bbox": ann["bbox"],
                        "segmentation": ann["segmentation"],
                        "area": ann["area"],
                        "score": ann["score"],
                        "iscrowd": 0
                    })
                    ann_id += 1
                
                # Composite mask layer
                comp_image = Image.alpha_composite(orig_image, mask_layer)
                final_vis = comp_image.convert("RGB")
                bbox_draw = ImageDraw.Draw(final_vis)
                
                # Draw bboxes and labels
                for ann in annotations:
                    x0, y0, w, h = ann["bbox"]
                    x1, y1 = x0 + w, y0 + h
                    
                    bbox_draw.rectangle([x0, y0, x1, y1], outline=BBOX_COLOR, width=3)
                    
                    label_txt = f"sam3 ({ann['score']:.2f})"
                    text_bbox = bbox_draw.textbbox((x0, y0), label_txt, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    label_y = y0 - text_h - 4
                    if label_y < 0:
                        label_y = y0 + 4
                    
                    bbox_draw.rectangle(
                        [x0, label_y, x0 + text_w + 6, label_y + text_h + 4],
                        fill=BBOX_COLOR
                    )
                    bbox_draw.text((x0 + 3, label_y + 1), label_txt, fill=LABEL_TEXT_COLOR, font=font)
                
                # Save visualization
                final_vis.save(os.path.join(output_img_dir, filename), quality=95)
                
                print(f"   Saved {len(annotations)} annotations")
                
            except Exception as e:
                print(f"   ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
            
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                img_id += 1
        
        # Save COCO JSON
        with open(output_json, "w") as f:
            json.dump(coco_output, f, indent=2)
        
        print(f"\nSaved: {output_json}")
        print(f"Total images: {img_id - 1}, Total annotations: {ann_id - 1}")


if __name__ == "__main__":
    main()
