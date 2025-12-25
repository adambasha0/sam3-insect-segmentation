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

ALLOWED_FOLDERS = { 
    #"abram2023",
    #"AMI-traps",
    #"AMT",
    #"anTraX",
    #"biodiscover-arm",
    #"Diopsis",
    #"DIRT",
    #"Mothitor",
    #"PeMaToEuroPep",
    #"pinoy2023",
    #"sticky-pi",
    ## "ubc-pitfall-traps",
    ## to do
    #"ubc-scanned-sticky-cards",
    #"ALUS",
    #"BIOSCAN",
    "DiversityScanner",
    #"ArTaxOr",
    #"CollembolAI",    
} # Add your folders

PROMPT_TEXT = "insects"
CATEGORY_ID = 1

# --- PYRAMID STRATEGY ---
# 1.0  = Detail view (Small bugs)
# 0.5  = Context view (Medium bugs)
# 0.25 = Global view (Large bugs & "Standard SAM3" behavior) -> FIXES MISSING MIDDLE
PYRAMID_SCALES = [1.0, 0.5, 0.25] 

# --- INFERENCE PARAMETERS ---
TILE_SIZE = 1024
TILE_OVERLAP = 0.30     # Increased to 30% to help medium objects fit in tiles
IOU_THRESHOLD = 0.5     
CONF_THRESHOLD = 0.45   

# BORDER PATROL: Ignore objects touching tile edges
BORDER_THRESHOLD = 10 

# --- VISUALIZATION STYLING ---
MASK_DILATION_PIXELS = 8
MASK_FILL_COLOR_RGBA = (135, 206, 250, 120) # Pale Blue
BBOX_COLOR = "#0026ff" 
LABEL_TEXT_COLOR = "black"

# ==========================
# 2. HELPER FUNCTIONS
# ==========================

def generate_sliding_window_crops(image, tile_size, overlap_ratio, current_scale):
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

def mask_to_global_polygon(mask_uint8, x_off, y_off, scale_factor):
    """
    Converts a binary mask (uint8) into COCO polygons in original coordinates.
    """
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < 3: continue 
        points = cnt.reshape(-1, 2).astype(np.float32)
        
        # Transform: Tile -> Layer Global -> Original Global
        points[:, 0] = (points[:, 0] + x_off) / scale_factor
        points[:, 1] = (points[:, 1] + y_off) / scale_factor
        
        points = np.maximum(points, 0)
        polygons.append(points.flatten().tolist())
    return polygons

def mask_to_global_bbox(mask_uint8, x_off, y_off, scale_factor):
    """
    Calculates Bounding Box from the MASK (post-dilation) to ensure fit.
    """
    x, y, w, h = cv2.boundingRect(mask_uint8)
    
    # Transform: Tile -> Layer Global -> Original Global
    gx = (x + x_off) / scale_factor
    gy = (y + y_off) / scale_factor
    gw = w / scale_factor
    gh = h / scale_factor
    
    return [gx, gy, gw, gh]

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

try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()

# ==========================
# 4. MAIN PIPELINE
# ==========================
for dataset_name in sorted(os.listdir(ROOT_DATASET)):
    if dataset_name not in ALLOWED_FOLDERS:
        continue

    dataset_path = os.path.join(ROOT_DATASET, dataset_name)
    print(f"\nProcessing Dataset: {dataset_name}")

    output_json = os.path.join(dataset_path, "sam3_results_pyramid_v2.json")
    output_img_dir = os.path.join(dataset_path, "sam3_vis_pyramid_v2")
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
        print(f"[{img_id}] {filename} ({orig_w}x{orig_h}) | Scales: {PYRAMID_SCALES}")

        all_boxes = []
        all_scores = []
        all_masks_info = []

        # --- PYRAMID LOOP ---
        for scale in PYRAMID_SCALES:
            print(f"   > Scale {scale}x...", end=" ", flush=True)
            
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
                t_boxes, t_scores, t_masks = run_inference(processor, t['image'], PROMPT_TEXT)
                
                if len(t_boxes) == 0: continue

                is_real_left   = (t['x'] == 0)
                is_real_top    = (t['y'] == 0)
                is_real_right  = (t['x'] + TILE_SIZE >= layer_w)
                is_real_bottom = (t['y'] + TILE_SIZE >= layer_h)

                for i, box in enumerate(t_boxes):
                    lx0, ly0, lx1, ly1 = box
                    
                    # BORDER PATROL
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

                    # Transform Box to Global for NMS
                    gx0 = (lx0 + t['x']) / scale
                    gy0 = (ly0 + t['y']) / scale
                    gx1 = (lx1 + t['x']) / scale
                    gy1 = (ly1 + t['y']) / scale

                    all_boxes.append([gx0, gy0, gx1, gy1])
                    all_scores.append(float(t_scores[i]))
                    
                    # Store minimal mask info
                    mask_bin = (t_masks[i] > 0.5).astype(np.uint8)
                    all_masks_info.append({
                        "mask": mask_bin,
                        "x_off": t['x'], "y_off": t['y'], "scale": scale
                    })
            
            if scale != 1.0: del layer_img
            gc.collect()
            print(f"Done ({tiles_count} tiles)")

        # --- GLOBAL NMS ---
        print(f"   > Resolving {len(all_boxes)} candidates...", end=" ")
        final_indices = []
        if len(all_boxes) > 0:
            boxes_t = torch.tensor(all_boxes, dtype=torch.float32).to(DEVICE)
            scores_t = torch.tensor(all_scores, dtype=torch.float32).to(DEVICE)
            keep = torchvision.ops.nms(boxes_t, scores_t, IOU_THRESHOLD)
            final_indices = keep.cpu().numpy()
        print(f"-> {len(final_indices)} unique insects.")

        # --- VISUALIZATION ---
        coco_output["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": orig_w,
            "height": orig_h
        })

        base_image_rgba = orig_image.convert("RGBA")
        mask_layer = Image.new("RGBA", base_image_rgba.size, (0,0,0,0))
        mask_draw = ImageDraw.Draw(mask_layer)
        
        final_anns = []

        for idx in final_indices:
            score = all_scores[idx]
            m_info = all_masks_info[idx]

            # 1. Dilate Mask
            mask_uint8 = m_info["mask"]
            kernel = np.ones((3,3), np.uint8)
            dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=MASK_DILATION_PIXELS)

            # 2. Update BBox based on DILATED mask (FIX FOR BOX SIZE)
            # We ignore the original 'all_boxes[idx]' and recalculate it from the mask
            bbox_new = mask_to_global_bbox(
                dilated_mask, 
                m_info["x_off"], 
                m_info["y_off"], 
                m_info["scale"]
            )
            
            # 3. Generate Polygons
            polys = mask_to_global_polygon(
                dilated_mask, 
                m_info["x_off"], 
                m_info["y_off"], 
                m_info["scale"]
            )
            
            if not polys: continue

            # Draw Filled Mask
            for p in polys:
                poly_tuples = [(p[i], p[i+1]) for i in range(0, len(p), 2)]
                mask_draw.polygon(poly_tuples, fill=MASK_FILL_COLOR_RGBA)

            final_anns.append({
                "bbox": bbox_new, # Use the new expanded box
                "polys": polys,
                "score": score
            })

        # Composite & Draw Boxes
        comp_image = Image.alpha_composite(base_image_rgba, mask_layer)
        final_vis_image = comp_image.convert("RGB")
        bbox_draw = ImageDraw.Draw(final_vis_image)

        for item in final_anns:
            x0, y0, w, h = item["bbox"]
            x1, y1 = x0 + w, y0 + h
            
            # Draw Box (Expanded)
            bbox_draw.rectangle([x0, y0, x1, y1], outline=BBOX_COLOR, width=3)
            
            # Label
            label_txt = "sam3"
            text_bbox = bbox_draw.textbbox((x0, y0), label_txt, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            label_y = y0 - text_h - 4
            if label_y < 0: label_y = y0 + 4
                
            bbox_draw.rectangle([x0, label_y, x0 + text_w + 6, label_y + text_h + 4], fill=BBOX_COLOR)
            bbox_draw.text((x0 + 3, label_y + 1), label_txt, fill=LABEL_TEXT_COLOR, font=font)
            
            # Add to COCO
            coco_output["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "file_name": filename,
                "category_id": CATEGORY_ID,
                "bbox": [x0, y0, w, h],
                "segmentation": item["polys"],
                "area": float(w * h),
                "score": item["score"],
                "iscrowd": 0
            })
            ann_id += 1

        final_vis_image.save(os.path.join(output_img_dir, filename), quality=95)
        del base_image_rgba, mask_layer, final_vis_image, bbox_draw, all_boxes, all_scores, all_masks_info
        gc.collect()
        torch.cuda.empty_cache()
        img_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=2)
    print(f"Dataset complete. Saved to {output_json}")