import os
import json
import torch
import torchvision
import cv2
import numpy as np
import gc
from PIL import Image, ImageDraw, ImageFont
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ==========================
# 1. CONFIGURATION
# ==========================
DEVICE = "cuda:0"
BPE_PATH = "./assets/bpe_simple_vocab_16e6.txt.gz"
ROOT_DATASET = "./flatbug-dataset"

# Add your specific folder names here
ALLOWED_FOLDERS = {
    "Mothitor",
}

PROMPT_TEXT = "insects"
CATEGORY_ID = 1

# --- PYRAMID STRATEGY ---
# 1.0 = Native resolution (Best for small/tiny insects)
# 0.5 = Half resolution (Best for medium/large insects)
PYRAMID_SCALES = [1.0, 0.5] 

# --- INFERENCE PARAMETERS ---
TILE_SIZE = 1024        # SAM standard input size
TILE_OVERLAP = 0.25     # 25% overlap ensures no object is missed at edges
IOU_THRESHOLD = 0.5     # Intersection over Union for removing duplicates
CONF_THRESHOLD = 0.45   # Minimum confidence to keep a detection

# --- ARTIFACT REMOVAL ---
# Ignore detections touching the edge of a tile (unless it is the real image edge).
# This forces the system to wait for the adjacent tile where the object is centered.s
BORDER_THRESHOLD = 10 

# --- VISUALIZATION STYLING ---
MASK_DILATION_PIXELS = 8                # Expand mask by ~8 pixels
MASK_FILL_COLOR_RGBA = (135, 206, 250, 120) # Pale Blue with transparency
BBOX_COLOR = "#00FF00"                  # Green
LABEL_TEXT_COLOR = "black"

# ==========================
# 2. HELPER FUNCTIONS
# ==========================

def generate_sliding_window_crops(image, tile_size, overlap_ratio, current_scale):
    """
    Generator that yields image tiles one by one to save RAM.
    """
    w, h = image.size
    stride = int(tile_size * (1 - overlap_ratio))

    # If image is smaller than a single tile, return it as-is
    if w <= tile_size and h <= tile_size:
        yield {'image': image, 'x': 0, 'y': 0, 'scale': current_scale}
        return

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            
            # Adjust start to ensure we always have a full tile (unless image is too small)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            
            crop = image.crop((x_start, y_start, x_end, y_end))
            yield {'image': crop, 'x': x_start, 'y': y_start, 'scale': current_scale}

def mask_to_global_polygon(mask_uint8, x_off, y_off, scale_factor):
    """
    Converts a binary mask (uint8) into COCO polygons in the original global coordinates.
    """
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        if len(cnt) < 3: continue 
        
        # Reshape to (N, 2)
        points = cnt.reshape(-1, 2).astype(np.float32)
        
        # 1. Transform Local Tile -> Layer Global
        points[:, 0] += x_off
        points[:, 1] += y_off
        
        # 2. Transform Layer Global -> Original Image Global
        # (Divide by scale factor: e.g. if scale is 0.5, we multiply by 2)
        points[:, 0] /= scale_factor
        points[:, 1] /= scale_factor
        
        # Clamp to zero to avoid negative coordinates
        points = np.maximum(points, 0)
        
        polygons.append(points.flatten().tolist())
        
    return polygons

def run_inference(processor, image, prompt):
    """
    Runs SAM3 inference on a single tile.
    Includes fix for BFloat16 crash.
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
        return (
            # IMPORTANT: Cast to float32 before sending to NumPy/CPU
            boxes.float().detach().cpu().numpy(),
            scores.float().detach().cpu().numpy(),
            masks.float().detach().cpu().numpy().squeeze()
        )
    return [], [], []

# ==========================
# 3. INITIALIZE MODEL
# ==========================
print("Loading SAM3 Model...")
model = build_sam3_image_model(bpe_path=BPE_PATH)
model.to(DEVICE)
model.eval()
processor = Sam3Processor(model, device=DEVICE, confidence_threshold=CONF_THRESHOLD)
print("Model Loaded.")

# Try to load a nice font, otherwise default
try:
    font = ImageFont.truetype("arial.ttf", 16)
except IOError:
    font = ImageFont.load_default()

# ==========================
# 4. MAIN PIPELINE
# ==========================
for dataset_name in sorted(os.listdir(ROOT_DATASET)):
    if dataset_name not in ALLOWED_FOLDERS:
        continue

    dataset_path = os.path.join(ROOT_DATASET, dataset_name)
    print(f"\nProcessing Dataset: {dataset_name}")

    output_json = os.path.join(dataset_path, "sam3_results_pyramid.json")
    output_img_dir = os.path.join(dataset_path, "sam3_vis_pyramid")
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
            print(f"Skipping {filename}: {e}")
            continue

        orig_w, orig_h = orig_image.size
        print(f"[{img_id}] {filename} ({orig_w}x{orig_h}) | Scales: {PYRAMID_SCALES}")

        # Accumulators for results from ALL scales
        all_boxes = []
        all_scores = []
        all_masks_info = []

        # ==================================================
        # PART A: PYRAMID TILING LOOP
        # ==================================================
        for scale in PYRAMID_SCALES:
            print(f"   > Scale {scale}x...", end=" ", flush=True)
            
            # Resize image for this pyramid level (unless it's 1.0)
            if scale == 1.0:
                layer_img = orig_image
            else:
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                layer_img = orig_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            layer_w, layer_h = layer_img.size
            tile_gen = generate_sliding_window_crops(layer_img, TILE_SIZE, TILE_OVERLAP, scale)
            
            tiles_count = 0
            for t in tile_gen:
                tiles_count += 1
                
                # Run Model
                t_boxes, t_scores, t_masks = run_inference(processor, t['image'], PROMPT_TEXT)
                
                if len(t_boxes) == 0: continue

                # Identify Real Image Edges (to avoid deleting valid edge objects)
                is_real_left   = (t['x'] == 0)
                is_real_top    = (t['y'] == 0)
                is_real_right  = (t['x'] + TILE_SIZE >= layer_w)
                is_real_bottom = (t['y'] + TILE_SIZE >= layer_h)

                for i, box in enumerate(t_boxes):
                    lx0, ly0, lx1, ly1 = box
                    
                    # --- BORDER PATROL FILTER ---
                    # Reject detections that touch the "internal" seams of tiles
                    touches_left   = (lx0 < BORDER_THRESHOLD)
                    touches_top    = (ly0 < BORDER_THRESHOLD)
                    touches_right  = (lx1 > TILE_SIZE - BORDER_THRESHOLD)
                    touches_bottom = (ly1 > TILE_SIZE - BORDER_THRESHOLD)

                    reject = False
                    if touches_left and not is_real_left: reject = True
                    if touches_top and not is_real_top: reject = True
                    if touches_right and not is_real_right: reject = True
                    if touches_bottom and not is_real_bottom: reject = True
                    
                    if reject: continue 

                    # Projection: Convert Local Tile Box -> Original Global Box
                    layer_gx0, layer_gy0 = lx0 + t['x'], ly0 + t['y']
                    layer_gx1, layer_gy1 = lx1 + t['x'], ly1 + t['y']
                    
                    orig_gx0 = layer_gx0 / scale
                    orig_gy0 = layer_gy0 / scale
                    orig_gx1 = layer_gx1 / scale
                    orig_gy1 = layer_gy1 / scale

                    all_boxes.append([orig_gx0, orig_gy0, orig_gx1, orig_gy1])
                    all_scores.append(float(t_scores[i]))
                    
                    # Store mask as compact binary (uint8)
                    mask_bin = (t_masks[i] > 0.5).astype(np.uint8)
                    all_masks_info.append({
                        "mask": mask_bin,
                        "x_off": t['x'], 
                        "y_off": t['y'], 
                        "scale": scale
                    })
            
            # Free memory for this layer
            if scale != 1.0: del layer_img
            gc.collect()
            print(f"Done ({tiles_count} tiles)")

        # ==================================================
        # PART B: GLOBAL NMS (Deduplication)
        # ==================================================
        print(f"   > Resolving {len(all_boxes)} candidates...", end=" ")
        
        final_indices = []
        if len(all_boxes) > 0:
            boxes_t = torch.tensor(all_boxes, dtype=torch.float32).to(DEVICE)
            scores_t = torch.tensor(all_scores, dtype=torch.float32).to(DEVICE)
            
            # Run NMS across all scales combined
            keep = torchvision.ops.nms(boxes_t, scores_t, IOU_THRESHOLD)
            final_indices = keep.cpu().numpy()

        print(f"-> {len(final_indices)} unique insects.")

        # ==================================================
        # PART C: VISUALIZATION & SAVING
        # ==================================================
        
        coco_output["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": orig_w,
            "height": orig_h
        })

        # Prepare Visualization Layers
        base_image_rgba = orig_image.convert("RGBA")
        mask_layer = Image.new("RGBA", base_image_rgba.size, (0,0,0,0))
        mask_draw = ImageDraw.Draw(mask_layer)
        
        # We need a list to store annotations before adding them to COCO
        # so we can draw the bounding boxes ON TOP of the masks later
        final_anns = []

        for idx in final_indices:
            box = all_boxes[idx]
            score = all_scores[idx]
            m_info = all_masks_info[idx]

            # --- DILATION & POLYGON GENERATION ---
            # 1. Retrieve Binary Mask (0 or 1)
            mask_uint8 = m_info["mask"]
            
            # 2. Dilate Mask (Expand by ~8 pixels)
            # We dilate the *local* mask before converting to global polygons
            kernel = np.ones((3,3), np.uint8)
            dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=MASK_DILATION_PIXELS)

            # 3. Generate Global Polygons
            polys = mask_to_global_polygon(
                dilated_mask, 
                m_info["x_off"], 
                m_info["y_off"], 
                m_info["scale"]
            )
            
            if not polys: continue

            # 4. Draw FILLED Polygon on Transparent Layer
            for p in polys:
                # Convert flat list to tuples (x,y)
                poly_tuples = [(p[i], p[i+1]) for i in range(0, len(p), 2)]
                mask_draw.polygon(poly_tuples, fill=MASK_FILL_COLOR_RGBA)

            w_box = box[2] - box[0]
            h_box = box[3] - box[1]
            
            final_anns.append({
                "bbox": box,
                "polys": polys,
                "coco": {
                    "id": ann_id,
                    "image_id": img_id,
                    "file_name": filename,
                    "category_id": CATEGORY_ID,
                    "bbox": [box[0], box[1], w_box, h_box],
                    "segmentation": polys,
                    "area": float(w_box * h_box),
                    "score": score,
                    "iscrowd": 0
                }
            })
            ann_id += 1

        # Combine Base Image + Mask Layer
        comp_image = Image.alpha_composite(base_image_rgba, mask_layer)
        final_vis_image = comp_image.convert("RGB")
        bbox_draw = ImageDraw.Draw(final_vis_image)

        # Draw Boxes & Labels ON TOP of everything
        for item in final_anns:
            x0, y0, x1, y1 = item["bbox"]
            
            # Draw Box
            bbox_draw.rectangle([x0, y0, x1, y1], outline=BBOX_COLOR, width=3)
            
            # Draw Label Background & Text
            label_txt = "sam3"
            
            # Calculate text size
            text_bbox = bbox_draw.textbbox((x0, y0), label_txt, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            # Position label above box, but keep inside image
            label_y = y0 - text_h - 4
            if label_y < 0: label_y = y0 + 4
                
            # Draw Background Rectangle
            bbox_draw.rectangle(
                [x0, label_y, x0 + text_w + 6, label_y + text_h + 4], 
                fill=BBOX_COLOR
            )
            # Draw Text
            bbox_draw.text((x0 + 3, label_y + 1), label_txt, fill=LABEL_TEXT_COLOR, font=font)
            
            # Add to COCO list
            coco_output["annotations"].append(item["coco"])

        # Save result
        final_vis_image.save(os.path.join(output_img_dir, filename), quality=95)
        
        # Clean up per image
        del base_image_rgba, mask_layer, final_vis_image, bbox_draw, all_boxes, all_scores, all_masks_info
        gc.collect()
        torch.cuda.empty_cache()
        img_id += 1

    # Save final JSON
    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=2)
    print(f"Dataset complete. JSON saved to {output_json}")

print("\nAll processing complete.")