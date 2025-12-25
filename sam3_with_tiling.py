import os
import json
import torch
import torchvision
import cv2
import numpy as np
import gc
from PIL import Image, ImageDraw
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ==========================
# 1. CONFIGURATION
# ==========================
DEVICE = "cuda:0"
BPE_PATH = "./assets/bpe_simple_vocab_16e6.txt.gz"
ROOT_DATASET = "./flatbug-dataset"

# Folders to process
ALLOWED_FOLDERS = {
    "cao2022",
}

PROMPT_TEXT = "insects"
CATEGORY_ID = 1

# --- HYPER-INFERENCE PARAMETERS ---
TILE_SIZE = 1024        # 1024x1024 is optimal for SAM-based models
TILE_OVERLAP = 0.25     # 25% overlap to ensure edge objects are captured
IOU_THRESHOLD = 0.5     # Intersection over Union for NMS (Deduplication)
CONF_THRESHOLD = 0.45   # Confidence score to keep a prediction

# ==========================
# 2. HELPER FUNCTIONS
# ==========================

def get_sliding_window_crops(image, tile_size, overlap_ratio):
    """
    Slices image into overlapping tiles.
    Returns: List of dicts {'image': PIL.Image, 'x': int, 'y': int}
    """
    w, h = image.size
    stride = int(tile_size * (1 - overlap_ratio))
    
    crops = []
    # If image is smaller than tile, just return original
    if w <= tile_size and h <= tile_size:
        return [{'image': image, 'x': 0, 'y': 0}]

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Calculate coordinates
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            
            # Adjust start point to ensure we don't have a tiny slice at the edge
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            
            crop = image.crop((x_start, y_start, x_end, y_end))
            crops.append({'image': crop, 'x': x_start, 'y': y_start})
            
    return crops

def mask_to_global_polygon(mask_np, x_off, y_off, scale_x, scale_y):
    """
    Extracts polygons from a local mask and transforms them to global coordinates.
    Memory efficient: does not create full-sized masks.
    """
    # Ensure binary uint8
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    
    # Find contours on the small tile mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        if len(cnt) < 3: continue # Ignore noise
            
        # Reshape to (N, 2)
        points = cnt.reshape(-1, 2).astype(np.float32)
        
        # Transform: Global = Local * Scale + Offset
        points[:, 0] = points[:, 0] * scale_x + x_off
        points[:, 1] = points[:, 1] * scale_y + y_off
        
        # Check bounds (clamp to 0)
        points = np.maximum(points, 0)
        
        # Flatten for COCO format
        polygons.append(points.flatten().tolist())
        
    return polygons

# ==========================
# 3. INITIALIZE MODEL
# ==========================
print("Loading SAM3 Model...")
model = build_sam3_image_model(bpe_path=BPE_PATH)
model.to(DEVICE)
model.eval()
processor = Sam3Processor(model, device=DEVICE, confidence_threshold=CONF_THRESHOLD)
print("Model Loaded.")

# ==========================
# 4. MAIN PIPELINE
# ==========================
for dataset_name in sorted(os.listdir(ROOT_DATASET)):
    if dataset_name not in ALLOWED_FOLDERS:
        continue

    dataset_path = os.path.join(ROOT_DATASET, dataset_name)
    print(f"\n==============================\nProcessing: {dataset_name}\n==============================")

    # Output paths
    output_json = os.path.join(dataset_path, f"sam3_results_tiled.json")
    output_img_dir = os.path.join(dataset_path, "sam3_vis_tiled")
    os.makedirs(output_img_dir, exist_ok=True)

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": CATEGORY_ID, "name": PROMPT_TEXT}]
    }

    ann_id = 1
    img_id = 1

    for filename in os.listdir(dataset_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(dataset_path, filename)
        
        try:
            orig_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping corrupt image {filename}: {e}")
            continue

        orig_w, orig_h = orig_image.size
        print(f"  [{img_id}] {filename} ({orig_w}x{orig_h})", end="... ")

        # --- A. PREPARE INFERENCE BATCH (Global + Tiles) ---
        inference_items = []

        # 1. Global Context View (Resized to TILE_SIZE)
        # This helps find large insects or clusters
        global_resized = orig_image.copy()
        global_resized.thumbnail((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
        inference_items.append({
            "image": global_resized,
            "x_off": 0, "y_off": 0,
            "scale_x": orig_w / global_resized.width,
            "scale_y": orig_h / global_resized.height,
            "type": "global"
        })

        # 2. Tiled Views (Full Resolution)
        # Only if image is larger than tile size
        if orig_w > TILE_SIZE or orig_h > TILE_SIZE:
            tiles = get_sliding_window_crops(orig_image, TILE_SIZE, TILE_OVERLAP)
            for t in tiles:
                inference_items.append({
                    "image": t['image'],
                    "x_off": t['x'], "y_off": t['y'],
                    "scale_x": 1.0, "scale_y": 1.0, # No scaling for tiles
                    "type": "tile"
                })

        print(f"Split into {len(inference_items)} views (1 Global + {len(inference_items)-1} Tiles)")

        # --- B. RUN INFERENCE ---
        raw_boxes = []
        raw_scores = []
        raw_masks_info = [] # Store metadata to reconstruct masks later

        for item in inference_items:
            # Run SAM3
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.inference_mode():
                    state = processor.set_image(item["image"])
                    processor.reset_all_prompts(state)
                    state = processor.set_text_prompt(PROMPT_TEXT, state)

            masks = state.get("masks", [])
            boxes = state.get("boxes", [])
            scores = state.get("scores", [])

            if len(boxes) == 0:
                continue

            # Transform results to Global Coordinates
            for i, box in enumerate(boxes):
                x0, y0, x1, y1 = box.detach().cpu().tolist()
                
                # Apply scaling (for global view) and offset (for tiles)
                gx0 = x0 * item["scale_x"] + item["x_off"]
                gy0 = y0 * item["scale_y"] + item["y_off"]
                gx1 = x1 * item["scale_x"] + item["x_off"]
                gy1 = y1 * item["scale_y"] + item["y_off"]

                raw_boxes.append([gx0, gy0, gx1, gy1])
                raw_scores.append(scores[i].item())
                
                # Keep mask as small numpy array to save RAM
                # We will process polygons only for survivors
                raw_masks_info.append({
                    "mask": masks[i].detach().cpu().numpy().squeeze(),
                    "x_off": item["x_off"],
                    "y_off": item["y_off"],
                    "scale_x": item["scale_x"],
                    "scale_y": item["scale_y"]
                })

            # Clear VRAM after every tile
            del state, masks, boxes, scores
            torch.cuda.empty_cache()

        # --- C. DEDUPLICATION (NMS) ---
        final_indices = []
        if len(raw_boxes) > 0:
            boxes_t = torch.tensor(raw_boxes, dtype=torch.float32).to(DEVICE)
            scores_t = torch.tensor(raw_scores, dtype=torch.float32).to(DEVICE)
            
            # Run NMS
            keep = torchvision.ops.nms(boxes_t, scores_t, IOU_THRESHOLD)
            final_indices = keep.cpu().numpy()

        print(f"    > Detections: {len(raw_boxes)} raw -> {len(final_indices)} unique")

        # --- D. SAVE RESULTS & VISUALIZE ---
        
        # Prepare visualization canvas
        vis_image = orig_image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        coco_output["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": orig_w,
            "height": orig_h
        })

        for idx in final_indices:
            x0, y0, x1, y1 = raw_boxes[idx]
            score = raw_scores[idx]
            m_info = raw_masks_info[idx]
            
            w_box = x1 - x0
            h_box = y1 - y0

            # 1. Get Polygons (Memory Efficient)
            # Threshold mask to binary
            local_mask_bin = (m_info["mask"] > 0.5).astype(np.uint8)
            
            polys = mask_to_global_polygon(
                local_mask_bin, 
                m_info["x_off"], m_info["y_off"], 
                m_info["scale_x"], m_info["scale_y"]
            )
            
            if not polys: continue

            # 2. Visualize
            # Box (Green)
            draw.rectangle([x0, y0, x1, y1], outline="#00FF00", width=3)
            # Mask Outline (Yellow)
            for poly in polys:
                draw.polygon(poly, outline="#FFFF00", fill=None, width=2)
            
            # 3. Add to COCO JSON
            # Calc approximate area
            local_area = np.sum(local_mask_bin)
            global_area = local_area * m_info["scale_x"] * m_info["scale_y"]

            coco_output["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "file_name": filename,
                "category_id": CATEGORY_ID,
                "bbox": [float(x0), float(y0), float(w_box), float(h_box)],
                "segmentation": polys,
                "area": float(global_area),
                "iscrowd": 0,
                "score": float(score)
            })
            ann_id += 1

        # Save Image
        vis_image.save(os.path.join(output_img_dir, filename))
        
        # Cleanup Memory
        del vis_image, draw, raw_boxes, raw_scores, raw_masks_info
        gc.collect()
        img_id += 1

    # Save Dataset JSON
    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=2)
    print(f"Saved JSON to {output_json}")

print("\nAll processing complete.")