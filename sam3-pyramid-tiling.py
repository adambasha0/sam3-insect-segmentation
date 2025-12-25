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
ALLOWED_FOLDERS = { "Mothitor" }

PROMPT_TEXT = "insects"
CATEGORY_ID = 1

# --- PYRAMID CONFIGURATION ---
# The paper suggests analyzing the image at multiple scales.
# 1.0 = Native resolution (finds small bugs)
# 0.5 = Half resolution (finds medium beetles)
# 0.25 = Quarter resolution (finds large clusters/swarms)
PYRAMID_SCALES = [1.0, 0.5] 

TILE_SIZE = 1024        
TILE_OVERLAP = 0.25     
IOU_THRESHOLD = 0.5     
CONF_THRESHOLD = 0.45   

# BORDER PATROL: Ignore objects touching tile edges (unless real image edge)
# This prevents "chopped" detections.
BORDER_THRESHOLD = 10 

# ==========================
# 2. HELPER FUNCTIONS
# ==========================

def generate_sliding_window_crops(image, tile_size, overlap_ratio, current_scale):
    """
    Yields crops for a specific scale.
    Input image is ALREADY resized to the current_scale.
    """
    w, h = image.size
    stride = int(tile_size * (1 - overlap_ratio))

    if w <= tile_size and h <= tile_size:
        yield {'image': image, 'x': 0, 'y': 0, 'scale': current_scale}
        return

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            
            crop = image.crop((x_start, y_start, x_end, y_end))
            yield {'image': crop, 'x': x_start, 'y': y_start, 'scale': current_scale}

def mask_to_global_polygon(mask_np, x_off, y_off, scale_factor):
    """
    Converts binary mask to COCO polygon in ORIGINAL GLOBAL coordinates.
    scale_factor: The scale of the image layer (e.g., 0.5).
    We divide by scale_factor to get back to original 1.0 coords.
    """
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        if len(cnt) < 3: continue 
        points = cnt.reshape(-1, 2).astype(np.float32)
        
        # 1. Apply Tile Offset (Local -> Layer Global)
        points[:, 0] += x_off
        points[:, 1] += y_off
        
        # 2. Apply Scale Correction (Layer Global -> Original Global)
        # If we found it at 0.5x scale, we multiply by 2.0 (divide by 0.5) to get original size
        points[:, 0] /= scale_factor
        points[:, 1] /= scale_factor
        
        points = np.maximum(points, 0)
        polygons.append(points.flatten().tolist())
        
    return polygons

def run_inference(processor, image, prompt):
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
            # Fix: cast to float32 before numpy conversion
            boxes.float().detach().cpu().numpy(),
            scores.float().detach().cpu().numpy(),
            masks.float().detach().cpu().numpy().squeeze()
        )
    return [], [], []

# ==========================
# 3. INITIALIZATION
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
    output_json = os.path.join(dataset_path, f"sam3_results_pyramid.json")
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
        except:
            continue

        orig_w, orig_h = orig_image.size
        print(f"[{img_id}] {filename} ({orig_w}x{orig_h}) Processing Scales: {PYRAMID_SCALES}")

        # Accumulators for ALL scales
        all_boxes = []
        all_scores = []
        all_masks_info = []

        # ==================================================
        # PYRAMID LOOP
        # ==================================================
        for scale in PYRAMID_SCALES:
            print(f"   > Scale {scale}x...", end=" ", flush=True)
            
            # 1. Resize Image for this Layer
            if scale == 1.0:
                layer_img = orig_image
            else:
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                layer_img = orig_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            layer_w, layer_h = layer_img.size

            # 2. Tile this Layer
            tile_gen = generate_sliding_window_crops(layer_img, TILE_SIZE, TILE_OVERLAP, scale)
            
            tiles_processed = 0
            for t in tile_gen:
                tiles_processed += 1
                
                # Run Inference
                t_boxes, t_scores, t_masks = run_inference(processor, t['image'], PROMPT_TEXT)
                
                if len(t_boxes) == 0: continue

                # Identify Real Image Edges (for this layer)
                # If t['x'] is 0, it's the real left edge of the resized image
                is_real_left  = (t['x'] == 0)
                is_real_top   = (t['y'] == 0)
                is_real_right = (t['x'] + TILE_SIZE >= layer_w)
                is_real_bottom= (t['y'] + TILE_SIZE >= layer_h)

                for i, box in enumerate(t_boxes):
                    lx0, ly0, lx1, ly1 = box
                    
                    # --- BORDER PATROL (Crucial for Pyramid) ---
                    # We reject partial objects at EVERY scale
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

                    # --- COORDINATE PROJECTION ---
                    # 1. Tile -> Layer Global
                    layer_gx0 = lx0 + t['x']
                    layer_gy0 = ly0 + t['y']
                    layer_gx1 = lx1 + t['x']
                    layer_gy1 = ly1 + t['y']

                    # 2. Layer Global -> Original Global (Divide by scale)
                    orig_gx0 = layer_gx0 / scale
                    orig_gy0 = layer_gy0 / scale
                    orig_gx1 = layer_gx1 / scale
                    orig_gy1 = layer_gy1 / scale

                    all_boxes.append([orig_gx0, orig_gy0, orig_gx1, orig_gy1])
                    all_scores.append(float(t_scores[i]))
                    
                    # Store mask + metadata needed to reconstruct it later
                    # We store the BINARY mask to save RAM
                    mask_bin = (t_masks[i] > 0.5).astype(np.uint8)
                    all_masks_info.append({
                        "mask": mask_bin,
                        "x_off": t['x'], 
                        "y_off": t['y'],
                        "scale": scale # Important: we need to know which scale this came from
                    })
            
            # Clean up layer memory
            if scale != 1.0: del layer_img
            gc.collect()
            print(f"Done ({tiles_processed} tiles)")

        # ==================================================
        # GRAND NMS (Across All Scales)
        # ==================================================
        print(f"   > Resolving {len(all_boxes)} candidates...", end=" ")
        
        final_indices = []
        if len(all_boxes) > 0:
            boxes_t = torch.tensor(all_boxes, dtype=torch.float32).to(DEVICE)
            scores_t = torch.tensor(all_scores, dtype=torch.float32).to(DEVICE)
            
            # Standard NMS is usually sufficient here because:
            # 1. We removed border artifacts
            # 2. Boxes from 0.5x and 1.0x that are the "same" object will have high IoU
            # 3. The one with the higher confidence score wins
            keep = torchvision.ops.nms(boxes_t, scores_t, IOU_THRESHOLD)
            final_indices = keep.cpu().numpy()

        print(f"-> {len(final_indices)} unique insects.")

        # ==================================================
        # SAVE & VISUALIZE
        # ==================================================
        coco_output["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": orig_w,
            "height": orig_h
        })

        vis_image = orig_image.copy()
        draw = ImageDraw.Draw(vis_image)

        for idx in final_indices:
            box = all_boxes[idx]
            score = all_scores[idx]
            m_info = all_masks_info[idx]
            
            # Reconstruct Polygon
            # function takes (mask, x_off, y_off, scale_factor)
            polys = mask_to_global_polygon(
                m_info["mask"], 
                m_info["x_off"], 
                m_info["y_off"], 
                m_info["scale"]
            )
            
            if not polys: continue

            # Draw
            draw.rectangle(box, outline="#00FF00", width=3)
            # Add text for Scale Source (Optional debug)
            # draw.text((box[0], box[1]), f"{m_info['scale']}x", fill="white")
            
            for p in polys:
                draw.polygon(p, outline="#FFFF00", width=2)

            w_box = box[2] - box[0]
            h_box = box[3] - box[1]
            
            coco_output["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "file_name": filename,
                "category_id": CATEGORY_ID,
                "bbox": [box[0], box[1], w_box, h_box],
                "segmentation": polys,
                "area": float(w_box * h_box),
                "score": score,
                "iscrowd": 0
            })
            ann_id += 1

        vis_image.save(os.path.join(output_img_dir, filename))
        
        del vis_image, draw, all_boxes, all_scores, all_masks_info
        gc.collect()
        img_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=2)
    print(f"Done. Saved to {output_json}")