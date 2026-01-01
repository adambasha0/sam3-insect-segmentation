import sys
import os
import json
import torch
import torchvision
import cv2
import numpy as np
import gc
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load the token from .env and login
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

# Add your folders here
## ALLOWED_FOLDERS = { "DiversityScanner" }

ALLOWED_FOLDERS = {
    "NHM-beetles-crops",
    #"cao2022",
    "gernat2018",
    "sittinger2023",
    "amarathunga2022",
    "biodiscover-arm",
    #"Mothitor",
    "DIRT",
    #"Diopsis",
    #"AMI-traps",
    #"AMT",
    "PeMaToEuroPep",
    "abram2023",
    "anTraX",
    "pinoy2023",
    "sticky-pi",
    "ubc-pitfall-traps",
    #"ALUS",
    "BIOSCAN",
    "DiversityScanner",
    "ArTaxOr",
    "CollembolAI",
    "ubc-scanned-sticky-cards",
}

# --- PROMPTS ---
PROMPT_PLURAL = "insects"  # For high-res tiles (expecting many)
PROMPT_SINGULAR = "insect" # For global view (expecting one big one)
CATEGORY_ID = 1

# --- FLATBUG PYRAMID CONFIG ---
FLATBUG_SCALE_INCREMENT = 2/3 

# --- INFERENCE PARAMETERS ---
TILE_SIZE = 1024        
TILE_OVERLAP = 0.25     
IOU_THRESHOLD = 0.5     

# Adaptive Thresholds
CONF_STRICT = 0.45     # For tiles (high detail)
CONF_PERMISSIVE = 0.25  # For global view (blurry, catch big objects)

# Border Patrol (Pixels)
BORDER_THRESHOLD = 10 

# --- VISUALIZATION ---
MASK_DILATION_PIXELS = 8
MASK_FILL_COLOR_RGBA = (135, 206, 250, 120) 
BBOX_COLOR = "#00FF00" 
LABEL_TEXT_COLOR = "black"

# ==========================
# 2. HELPER FUNCTIONS
# ==========================

def get_flatbug_scales(image_w, image_h, tile_size):
    """Calculates pyramid scales dynamically."""
    max_dim = max(image_w, image_h)
    scales = []
    
    # Global Scale (Fit whole image in one tile)
    s = tile_size / max_dim

    if s >= 1.0:
        return [1.0]
    
    # Grow by ~1.5x steps until 0.9
    while s <= 0.9:
        scales.append(s)
        s /= FLATBUG_SCALE_INCREMENT 
    
    scales.append(1.0)
    return sorted(scales)

def generate_sliding_window_crops(image, tile_size, overlap_ratio, current_scale):
    """Memory-safe generator."""
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
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < 3: continue 
        points = cnt.reshape(-1, 2).astype(np.float32)
        points[:, 0] = (points[:, 0] + x_off) / scale_factor
        points[:, 1] = (points[:, 1] + y_off) / scale_factor
        points = np.maximum(points, 0)
        polygons.append(points.flatten().tolist())
    return polygons

def mask_to_global_bbox(mask_uint8, x_off, y_off, scale_factor):
    x, y, w, h = cv2.boundingRect(mask_uint8)
    gx = (x + x_off) / scale_factor
    gy = (y + y_off) / scale_factor
    gw = w / scale_factor
    gh = h / scale_factor
    return [gx, gy, gw, gh]

def run_inference(processor, image, prompt, threshold):
    """Runs SAM3 and filters by dynamic threshold. FIXED for 1-object crash."""
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
        
        # --- FIX: Safe Squeeze ---
        # Only remove the channel dim (1), not the batch dim (0)
        if m.ndim == 4:
            m = m.squeeze(1)
        
        # Ensure scores are flat
        s = s.flatten()
        
        keep = s > threshold
        return b[keep], s[keep], m[keep]

    return [], [], []

def remove_contained_boxes(boxes, scores, masks_info, intersection_threshold=0.90):
    """Removes small boxes contained inside larger boxes."""
    if len(boxes) == 0:
        return boxes, scores, masks_info
    
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = np.ones(len(boxes), dtype=bool)
    
    for i in range(len(boxes)):
        if not keep[i]: continue
        for j in range(len(boxes)):
            if i == j or not keep[j]: continue
            
            ix0 = max(boxes[i][0], boxes[j][0])
            iy0 = max(boxes[i][1], boxes[j][1])
            ix1 = min(boxes[i][2], boxes[j][2])
            iy1 = min(boxes[i][3], boxes[j][3])
            
            iw = max(0, ix1 - ix0)
            ih = max(0, iy1 - iy0)
            intersection = iw * ih
            
            if (intersection > areas[i] * intersection_threshold) and (areas[j] > areas[i]):
                keep[i] = False
                break
                
    return boxes[keep], scores[keep], [masks_info[k] for k in range(len(masks_info)) if keep[k]]

# ==========================
# 3. INITIALIZATION
# ==========================
print("Loading SAM3 Model...")
model = build_sam3_image_model(bpe_path=BPE_PATH)
model.to(DEVICE)
model.eval()
processor = Sam3Processor(model, device=DEVICE, confidence_threshold=0.35) 
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

    output_json = os.path.join(dataset_path, "sam3_flatbug_strategy.json")
    output_img_dir = os.path.join(dataset_path, "sam3_flatbug_strategy")
    os.makedirs(output_img_dir, exist_ok=True)

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": CATEGORY_ID, "name": PROMPT_PLURAL}]
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
            print(f"Error loading {filename}: {e}")
            continue

        orig_w, orig_h = orig_image.size
        dynamic_scales = get_flatbug_scales(orig_w, orig_h, TILE_SIZE)
        
        print(f"[{img_id}] {filename} ({orig_w}x{orig_h})") 
        print(f"   > Scales: {[f'{s:.3f}' for s in dynamic_scales]}")

        all_boxes = []
        all_scores = []
        all_masks_info = []

        # --- PYRAMID LOOP ---
        for scale in dynamic_scales:
            print(f"   > Scale {scale:.3f}...", end=" ", flush=True)
            
            is_global = (scale == min(dynamic_scales))
            
            if is_global:
                current_conf = CONF_PERMISSIVE  # 0.20
                current_prompt = PROMPT_SINGULAR 
            else:
                current_conf = CONF_STRICT      # 0.45
                current_prompt = PROMPT_PLURAL   

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
                
                t_boxes, t_scores, t_masks = run_inference(
                    processor, t['image'], current_prompt, current_conf
                )
                
                if len(t_boxes) == 0: continue

                # Edge Logic
                is_real_left   = (t['x'] == 0)
                is_real_top    = (t['y'] == 0)
                is_real_right  = (t['x'] + TILE_SIZE >= layer_w)
                is_real_bottom = (t['y'] + TILE_SIZE >= layer_h)

                for i, box in enumerate(t_boxes):
                    lx0, ly0, lx1, ly1 = box
                    
                    # Border Patrol
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

                    # Projection
                    gx0 = (lx0 + t['x']) / scale
                    gy0 = (ly0 + t['y']) / scale
                    gx1 = (lx1 + t['x']) / scale
                    gy1 = (ly1 + t['y']) / scale

                    all_boxes.append([gx0, gy0, gx1, gy1])
                    all_scores.append(float(t_scores[i]))
                    
                    mask_bin = (t_masks[i] > 0.5).astype(np.uint8)
                    all_masks_info.append({
                        "mask": mask_bin,
                        "x_off": t['x'], "y_off": t['y'], "scale": scale
                    })
            
            if scale != 1.0: del layer_img
            gc.collect()
            print(f"Done ({tiles_count} tiles)")

        # --- NMS & CLEANUP ---
        print(f"   > Candidates: {len(all_boxes)}...", end=" ")
        if len(all_boxes) > 0:
            boxes_t = torch.tensor(all_boxes, dtype=torch.float32).to(DEVICE)
            scores_t = torch.tensor(all_scores, dtype=torch.float32).to(DEVICE)
            keep = torchvision.ops.nms(boxes_t, scores_t, IOU_THRESHOLD)
            
            nms_indices = keep.cpu().numpy()
            
            nms_boxes = np.array([all_boxes[i] for i in nms_indices])
            nms_scores = np.array([all_scores[i] for i in nms_indices])
            nms_masks = [all_masks_info[i] for i in nms_indices]
            
            clean_boxes, clean_scores, clean_masks = remove_contained_boxes(
                nms_boxes, nms_scores, nms_masks
            )
            print(f"-> {len(nms_indices)} (NMS) -> {len(clean_boxes)} (Cleaned)")
        else:
            clean_boxes, clean_scores, clean_masks = [], [], []
            print("-> 0")

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

        for idx in range(len(clean_boxes)):
            score = clean_scores[idx]
            m_info = clean_masks[idx]

            mask_uint8 = m_info["mask"]            
            
            # Generate BBox and Polygons directly from raw mask
            bbox_new = mask_to_global_bbox(mask_uint8, m_info["x_off"], m_info["y_off"], m_info["scale"])
            polys = mask_to_global_polygon(mask_uint8, m_info["x_off"], m_info["y_off"], m_info["scale"])
            
            if not polys: continue

            for p in polys:
                poly_tuples = [(p[i], p[i+1]) for i in range(0, len(p), 2)]
                mask_draw.polygon(poly_tuples, fill=MASK_FILL_COLOR_RGBA)

            final_anns.append({
                "bbox": bbox_new,
                "polys": polys,
                "score": score
            })

        comp_image = Image.alpha_composite(base_image_rgba, mask_layer)
        final_vis_image = comp_image.convert("RGB")
        bbox_draw = ImageDraw.Draw(final_vis_image)

        for item in final_anns:
            x0, y0, w, h = item["bbox"]
            x1, y1 = x0 + w, y0 + h
            
            bbox_draw.rectangle([x0, y0, x1, y1], outline=BBOX_COLOR, width=3)
            
            label_txt = f"sam3 (score: {item['score']:.2f})"
            text_bbox = bbox_draw.textbbox((x0, y0), label_txt, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            label_y = y0 - text_h - 4
            if label_y < 0: label_y = y0 + 4
                
            bbox_draw.rectangle([x0, label_y, x0 + text_w + 6, label_y + text_h + 4], fill=BBOX_COLOR)
            bbox_draw.text((x0 + 3, label_y + 1), label_txt, fill=LABEL_TEXT_COLOR, font=font)
            
            coco_output["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CATEGORY_ID,
                "bbox": [x0, y0, w, h],
                "segmentation": item["polys"],
                "area": float(w * h),
                "score": float(item["score"]),
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
    print(f"Done. Saved to {output_json}")