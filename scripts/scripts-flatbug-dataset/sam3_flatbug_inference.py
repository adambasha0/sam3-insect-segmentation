"""
SAM3 FlatBug Inference Script

This script replicates the FlatBug repository's pyramid tiling inference methodology
for SAM3 to ensure a fair comparison between the models.

Key FlatBug methodology replicated:
1. Multi-scale pyramid with scale_increment = 2/3 (ratio 1.5x between levels)
2. Tile overlap = 384 pixels (MINIMUM_TILE_OVERLAP)
3. Edge case margin = 16 pixels (filter detections near tile borders)
4. Score threshold = 0.2 (confidence filtering)
5. IoU threshold = 0.2 (aggressive NMS)
6. Object size filtering: sqrt(area) in range [32, 10^8]
7. Padding = 32 pixels (added to image borders)
8. Tile offset calculation matching FlatBug's equal_allocate_overlaps algorithm

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
    "sticky-pi",
    "ubc-pitfall-traps",
    "ubc-scanned-sticky-cards",
}

# Visualization settings
MASK_FILL_COLOR_RGBA = (135, 206, 250, 120)
BBOX_COLOR = "#0051FF"
LABEL_TEXT_COLOR = "black"


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


def mask_to_polygon(mask_uint8: np.ndarray, x_off: int, y_off: int, scale: float) -> List[List[float]]:
    """
    Convert binary mask to polygon(s) in global coordinates.
    
    Args:
        mask_uint8: Binary mask (0/1 or 0/255)
        x_off, y_off: Tile offset in layer coordinates
        scale: Current pyramid scale
        
    Returns:
        List of polygon coordinate lists [x0, y0, x1, y1, ...]
    """
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for cnt in contours:
        if len(cnt) < 3:
            continue
        
        # Reshape and convert to global coordinates
        points = cnt.reshape(-1, 2).astype(np.float64)
        points[:, 0] = (points[:, 0] + x_off) / scale
        points[:, 1] = (points[:, 1] + y_off) / scale
        points = np.maximum(points, 0)
        
        # Simplify polygon (like FlatBug's simplify_contour)
        simplified = cv2.approxPolyDP(points.astype(np.float32), epsilon=1.0, closed=True)
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
    Process a single image using FlatBug methodology.
    
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
                
                all_boxes.append([gx0, gy0, gx1, gy1])
                all_scores.append(float(t_scores[i]))
                
                mask_bin = (t_masks[i] > 0.5).astype(np.uint8)
                all_masks_info.append({
                    "mask": mask_bin,
                    "x_off": tile_x,
                    "y_off": tile_y,
                    "scale": scale,
                    "padding": PADDING
                })
        
        if scale != 1.0:
            del layer_img
        gc.collect()
        print(f"Done ({tiles_count} tiles, {len(all_boxes)} candidates)")
    
    # Apply NMS with FlatBug's IoU threshold (0.2 = aggressive)
    print(f"   > Total candidates: {len(all_boxes)}...", end=" ")
    
    if len(all_boxes) > 0:
        boxes_t = torch.tensor(all_boxes, dtype=torch.float32).to(device)
        scores_t = torch.tensor(all_scores, dtype=torch.float32).to(device)
        
        # Use torchvision NMS with FlatBug's IoU threshold
        keep_indices = torchvision.ops.nms(boxes_t, scores_t, IOU_THRESHOLD)
        keep_indices = keep_indices.cpu().numpy()
        
        final_boxes = np.array([all_boxes[i] for i in keep_indices])
        final_scores = np.array([all_scores[i] for i in keep_indices])
        final_masks = [all_masks_info[i] for i in keep_indices]
        
        print(f"-> {len(final_boxes)} (after NMS)", end=" ")
        
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
        
        # Generate polygon from mask
        polys = mask_to_polygon(
            mask_uint8, 
            m_info["x_off"], 
            m_info["y_off"], 
            m_info["scale"]
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
                adjusted_polys.append(adjusted)
        
        if not adjusted_polys:
            continue
        
        # Calculate bbox from final boxes (ensure native Python floats)
        x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
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
        
        output_json = os.path.join(dataset_path, "sam3_flatbug_strategy-1.json")
        output_img_dir = os.path.join(dataset_path, "sam3_flatbug_strategy-1")
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
                            mask_draw.polygon(poly_tuples, fill=MASK_FILL_COLOR_RGBA)
                    
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
