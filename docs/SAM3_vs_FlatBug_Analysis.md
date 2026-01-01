  # SAM3 vs FlatBug: Analysis of Metric Discrepancies

## Executive Summary

This document summarizes key findings from analyzing why SAM3 achieves lower segmentation metrics compared to FlatBug on the FlatBug dataset.

### Overall Performance (23 Datasets)

| Metric | SAM3 (Pyramid Tiling) | SAM3 (FlatBug Strategy) |
|--------|----------------------|-------------------------|
| **Bbox mAP** | 0.504 | 0.489 |
| **Bbox AP50** | 0.719 | 0.676 |
| **Segm mAP** | 0.328 | 0.305 |
| **Segm AP50** | 0.577 | 0.531 |

**Key Observations:**
1. **Segmentation mAP (~0.30-0.33)** is significantly lower than FlatBug's reported ~0.80 mAP
2. **Bbox mAP (~0.50)** is decent but still below FlatBug
3. **High variance across datasets** - some datasets perform well (BIOSCAN: 0.84 segm mAP), others very poorly (cao2022: 0.002 segm mAP)

### Dataset-Specific Performance Breakdown

**Top Performers (Segm mAP > 0.6):**
| Dataset | Bbox mAP | Segm mAP | Notes |
|---------|----------|----------|-------|
| NHM-beetles-crops | 0.902 | 0.839 | Single insects, clean background |
| BIOSCAN | 0.869 | 0.785 | Well-lit specimens |
| ArTaxOr | 0.759 | 0.695 | Clear insect images |
| amarathunga2022 | 0.844 | 0.639 | Good contrast |
| DiversityScanner | 0.763 | 0.604 | Controlled environment |

**Poor Performers (Segm mAP < 0.1):**
| Dataset | Bbox mAP | Segm mAP | Likely Issues |
|---------|----------|----------|---------------|
| cao2022 | 0.489 | 0.002 | ❓ Extreme failure case |
| AMI-traps | 0.091 | 0.018 | Complex trap backgrounds |
| PeMaToEuroPep | 0.107 | 0.025 | Unknown |
| sticky-pi | 0.172 | 0.039 | Sticky trap texture |
| AMT | 0.244 | 0.053 | Trap complexity |

---

## Part 1: Key Findings (Q&A Format)

### Q1: Why does SAM3 achieve ~0.30 segm mAP while FlatBug achieves ~0.80 mAP?

**A:** Multiple factors contribute:

1. **Mask size mismatch** (confirmed on ALUS dataset):
   - GT annotations: **63.4% fill ratio**
   - SAM3 predictions: **50.3% fill ratio**
   - SAM3 masks are systematically **smaller/tighter** than GT

2. **Dataset-specific challenges** (hypothesis - needs investigation):
   - Some datasets have **near-zero performance** (cao2022: 0.002 mAP)
   - These catastrophic failures drag down the average significantly
   - Could be prompt issues, scale issues, or annotation style mismatches

3. **Polygon processing differences** (confirmed):
   - FlatBug uses specific post-processing that SAM3 doesn't replicate

### Q2: What is the "fill ratio" and why does it matter?

**A:** Fill ratio = (mask pixel area) / (bounding box area). It measures how much of the bounding box is filled by the segmentation mask.

| Source | Fill Ratio | Interpretation |
|--------|------------|----------------|
| GT Annotations | 0.634 | Larger, more generous masks |
| SAM3 v1 | 0.503 | Tighter, more conservative masks |
| SAM3 Strategy | 0.562 | Slightly larger (uses FlatBug's code) |

**Implication:** Even when SAM3 correctly detects an insect, the mask IoU suffers because SAM3's tighter boundary doesn't overlap well with GT's looser boundary.

### Q3: Is SAM3's detection performance actually poor?

**A:** On ALUS dataset specifically, detection is strong:
- **True Positives:** 271
- **False Positives:** 29 
- **False Negatives:** 10
- **Precision:** 90.3%
- **Recall:** 96.4%

**However**, this was only analyzed on ALUS (one of the better-performing datasets). The catastrophic failures on datasets like cao2022 suggest detection issues on certain dataset types.

### Q4: What specific FlatBug techniques cause larger masks?

**A:** Several implementation details in FlatBug's pipeline:

1. **`expand_by_one=True`** in `scale_contour()`: Expands every contour by 1 pixel outward from centroid
   - **File:** `flat_bug/predictor.py` **Line 419**
   - **Function definition:** `flat_bug/geometric.py` **Lines 256-297**
   - Moves each contour point outward along its normal by 1 pixel

2. **`pad=5`** in `offset_scale_pad()`: Adds 5px padding to bounding boxes after inference
   - **File:** `flat_bug/predictor.py` **Line 1654**
   - **Function definition:** `flat_bug/predictor.py` **Lines 230-280**
   - Comment: "pad the boxes a bit to ensure they encapsulate the masks"

3. **`cv2.CHAIN_APPROX_NONE`**: Keeps ALL contour points before simplification
   - **File:** `flat_bug/geometric.py` **Line 112**
   - Unlike CHAIN_APPROX_SIMPLE which compresses horizontal/vertical segments

4. **Dynamic tolerance**: `tolerance = (mask_to_image_scale / 2).mean()` ≈ 2.0
   - **File:** `flat_bug/predictor.py` **Line 453**
   - Tolerance scales with the mask-to-image scaling factor
   - For 256→1024 upscaling (4x), tolerance ≈ 2.0

5. **256×256 mask upscaling**: FlatBug uses YOLOv8 with 256×256 masks, upscaled 4× to tile size
   - **Mask size:** Set by YOLOv8 architecture (MASK_SIZE = 256)
   - When upscaled from 256→1024, rounding during `scale_contour` inherently expands boundaries

### Q5: What is the optimal dilation to match GT?

**A:** Based on experiments on the ALUS dataset:

| Dilation (px) | Mask IoU |
|---------------|----------|
| 0 (baseline) | 0.740 |
| 5 | 0.836 |
| 7 | 0.860 |
| 9 | 0.872 |
| 11 | 0.865 |

**Optimal: 7-9 pixels** dilation brings SAM3 masks closest to GT annotation style.

### Q6: Does polygon expansion help?

**A:** Yes, but less than dilation:

| Expansion (px) | Mask IoU |
|----------------|----------|
| 0 (baseline) | 0.740 |
| 2 | 0.801 |
| 4 | 0.833 |
| 6 | 0.829 |

**Optimal: 4 pixels** expansion. However, mask dilation is more effective.

### Q7: Why does contour approximation method matter?

**A:** 
- **FlatBug:** Uses `cv2.CHAIN_APPROX_NONE` → keeps ALL contour points (~150 points/polygon)
- **SAM3 v1:** Uses `cv2.CHAIN_APPROX_SIMPLE` → compresses straight segments (~43 points/polygon)

More points = smoother polygon = potentially better IoU with GT (which was created the same way).

### Q8: What about False Positives and False Negatives?

**A:** Analysis of FP/FN patterns:

**False Positives (29 total):**
- Most are **small detections** (mean area: 1,847 px² vs TP mean: 8,291 px²)
- Located near image edges (boundary artifacts)
- Some are legitimate insects missed in GT

**False Negatives (10 total):**
- Mostly **small insects** that SAM3 missed
- Some are very dark/low-contrast insects
- A few are at extreme edges

---

## Part 2: Methodological Differences Summary

### Polygon Extraction Pipeline Comparison

| Aspect | FlatBug | SAM3 v1 |
|--------|---------|---------|
| Contour method | CHAIN_APPROX_NONE | CHAIN_APPROX_SIMPLE |
| Simplification | Dynamic (scale/2) | Fixed (epsilon=1.0) |
| Mask source | 256×256 YOLOv8 | 1024×1024 SAM |
| Contour expansion | expand_by_one=True | None |
| BBox padding | pad=5 after inference | None |
| Points per polygon | ~150 mean | ~43 mean |

### Architecture Differences

| Aspect | FlatBug | SAM3 |
|--------|---------|------|
| Model | YOLOv8-seg | SAM3 (Florence-2) |
| Mask resolution | 256×256 | Same as input tile |
| Prompt type | None (trained) | Text: "insects" |
| Inference approach | Direct | Prompted |

---

## Part 3: Brainstorming & Thesis Points

### Potential Thesis Discussion Points

1. **Annotation Bias Problem**
   - When using a model's own predictions as ground truth, any competing model inherits a systematic disadvantage
   - This is a form of "evaluation bias" that may not reflect true segmentation quality
   - **Thesis angle:** Fair evaluation requires model-agnostic ground truth

2. **Foundation Models vs Task-Specific Models**
   - SAM3 is a general-purpose model; FlatBug is trained specifically on insects
   - SAM3 may produce more "accurate" biological boundaries, but scores lower on FlatBug-style GT
   - **Thesis angle:** Metrics don't always capture true quality

3. **The Fill Ratio Hypothesis**
   - SAM3's tighter masks may actually be more accurate to true insect boundaries
   - FlatBug's looser masks may include background pixels
   - **Thesis angle:** Higher IoU ≠ better segmentation quality

4. **Post-Processing as a Confounder**
   - Significant performance can be "recovered" through post-processing (dilation)
   - This raises questions about what metrics truly measure
   - **Thesis angle:** Raw model output vs pipeline output distinction

5. **Dataset Heterogeneity Challenge**
   - Performance varies wildly: 0.84 mAP (NHM-beetles) vs 0.002 mAP (cao2022)
   - Foundation models may not generalize equally across all insect imaging conditions
   - **Thesis angle:** The importance of dataset stratification in evaluation

### Open Questions for Further Investigation

- [ ] Would manual re-annotation show SAM3 is actually more accurate?
- [ ] How do other foundation models (SAM, SAM2) compare?
- [ ] Is the optimal dilation dataset-dependent?
- [ ] Could we train a learned post-processing step?
- [ ] How does performance vary by insect size?
- [ ] **Why does cao2022 have 0.002 segm mAP but 0.489 bbox mAP?** (Critical mystery)
- [ ] What makes AMI-traps and sticky-pi so difficult?
- [ ] Are there systematic differences in annotation style across datasets?

### Recommendations for Fair Comparison

1. **Use independently annotated ground truth** when possible
2. **Report multiple metrics**: Detection (mAP@0.5) AND segmentation (mask IoU) separately
3. **Document post-processing steps** in both pipelines
4. **Consider evaluation at multiple IoU thresholds** (not just 0.5)
5. **Report fill ratios** to characterize annotation/prediction style
6. **Report per-dataset metrics**, not just averages (averages hide catastrophic failures)

---

## Part 4: Technical Implementation Notes

### V2 Script Configuration Options

The `sam3_flatbug_inference_v2.py` script includes configurable options to match FlatBug's behavior:

```python
# V2 Enhancement Options (top of file)
V2_USE_CHAIN_APPROX_NONE = True     # Use all contour points
V2_USE_DYNAMIC_TOLERANCE = True     # Scale-based simplification
V2_LARGEST_CONTOUR_ONLY = True      # Avoid fragmented detections
V2_ENABLE_MASK_DILATION = True      # Expand masks
V2_MASK_DILATION_KERNEL = 7         # Dilation size (7-9 optimal)
V2_ENABLE_POLYGON_EXPANSION = False # Polygon expansion
V2_POLYGON_EXPANSION_PX = 4         # Expansion amount
V2_ENABLE_BOX_PADDING = True        # BBox padding
V2_BOX_PADDING_PX = 5               # Like FlatBug's pad=5
```

### Files Created/Modified

| File | Purpose |
|------|---------|
| `sam3_flatbug_inference_v2.py` | Improved inference with FlatBug alignment |
| `compare_gt_sam3_polygons.py` | GT vs SAM3 polygon analysis |
| `find_best_dilation.py` | Dilation optimization experiments |

---

## Appendix: Raw Experimental Data

### Full Dataset Metrics (SAM3 FlatBug Inference Strategy)

| Dataset | Bbox mAP | Bbox AP50 | Segm mAP | Segm AP50 | Category |
|---------|----------|-----------|----------|-----------|----------|
| NHM-beetles-crops | 0.902 | 0.960 | 0.839 | 0.960 | ⭐ Top |
| BIOSCAN | 0.869 | 0.926 | 0.785 | 0.936 | ⭐ Top |
| ArTaxOr | 0.759 | 0.850 | 0.695 | 0.858 | ⭐ Top |
| amarathunga2022 | 0.844 | 0.940 | 0.639 | 0.940 | ⭐ Top |
| DiversityScanner | 0.763 | 0.851 | 0.604 | 0.851 | ⭐ Top |
| gernat2018 | 0.519 | 0.714 | 0.470 | 0.722 | Medium |
| ubc-pitfall-traps | 0.624 | 0.741 | 0.434 | 0.711 | Medium |
| CollembolAI | 0.457 | 0.528 | 0.408 | 0.519 | Medium |
| sittinger2023 | 0.637 | 0.804 | 0.404 | 0.777 | Medium |
| ALUS | 0.703 | 0.913 | 0.400 | 0.800 | Medium |
| ubc-scanned-sticky-cards | 0.611 | 0.787 | 0.375 | 0.737 | Medium |
| anTraX | 0.587 | 0.809 | 0.174 | 0.671 | Low |
| DIRT | 0.425 | 0.681 | 0.161 | 0.540 | Low |
| Diopsis | 0.287 | 0.498 | 0.132 | 0.416 | Low |
| pinoy2023 | 0.349 | 0.460 | 0.126 | 0.349 | Low |
| abram2023 | 0.198 | 0.300 | 0.090 | 0.269 | Low |
| Mothitor | 0.268 | 0.679 | 0.072 | 0.287 | ⚠️ Poor |
| biodiscover-arm | 0.350 | 0.614 | 0.071 | 0.322 | ⚠️ Poor |
| AMT | 0.244 | 0.696 | 0.053 | 0.187 | ⚠️ Poor |
| sticky-pi | 0.172 | 0.429 | 0.039 | 0.186 | ⚠️ Poor |
| PeMaToEuroPep | 0.107 | 0.204 | 0.025 | 0.102 | ❌ Very Poor |
| AMI-traps | 0.091 | 0.261 | 0.018 | 0.052 | ❌ Very Poor |
| cao2022 | 0.489 | 0.898 | 0.002 | 0.015 | ❌ Catastrophic |

**Average:** Bbox mAP=0.489, Segm mAP=0.305

### ALUS Dataset Detailed Analysis (30 images)

```
Ground Truth:
  - Total annotations: 281
  - Mean fill ratio: 0.634
  - Mean polygon points: ~150

SAM3 v1:
  - Total predictions: 300
  - Mean fill ratio: 0.503
  - Mean polygon points: ~43
  - Mean mask IoU: 0.740

SAM3 with 7px dilation:
  - Mean mask IoU: 0.860

SAM3 with 9px dilation:
  - Mean mask IoU: 0.872
```

### Detection Performance (ALUS only, IoU=0.5)

```
True Positives:  271
False Positives:  29
False Negatives:  10
Precision: 90.3%
Recall: 96.4%
```

### Critical Mystery: cao2022 Dataset

This dataset shows an unusual pattern that requires investigation:
- **Bbox mAP: 0.489** (decent detection)
- **Bbox AP50: 0.898** (very good detection at IoU=0.5!)
- **Segm mAP: 0.002** (near-zero segmentation)
- **Segm AP50: 0.015** (near-zero)

**Hypothesis:** SAM3 is detecting insects correctly but producing completely wrong segmentation masks. Possible causes:
1. Annotation format mismatch
2. Extreme scale differences
3. Unique image characteristics
4. Bug in polygon generation for this dataset

---

## Part 5: Additional FlatBug Processing Steps (Deep Dive)

This section documents additional pre/post-processing steps discovered in the FlatBug codebase that may explain metric discrepancies with SAM3.

### 5.1 Minimum Mask Area Filter

**File:** `flat_bug/yolo_helpers.py` **Lines 416-419**

FlatBug filters out predictions with very small masks:

```python
too_small = masks.sum(dim=[1, 2]) < 3
pred = pred[~too_small]
boxes = boxes[~too_small]
masks = masks[~too_small]
```

**Implication:** Any mask with fewer than 3 pixels is removed. SAM3 should implement similar filtering.

### 5.2 Edge Margin Filtering

**File:** `flat_bug/yolo_helpers.py` **Lines 393-397**

FlatBug can filter predictions too close to tile edges:

```python
if edge_margin is not None and edge_margin > 0:
    close_to_edge = (boxes[:, :2] < edge_margin).any(dim=1) | (boxes[:, 2:] > (tile_size - edge_margin)).any(dim=1)
    pred = pred[~close_to_edge]
    boxes = boxes[~close_to_edge]
```

**Implication:** Predictions near tile boundaries may be filtered. Check if SAM3 has similar edge handling.

### 5.3 Valid Size Range Filtering

**File:** `flat_bug/yolo_helpers.py` **Lines 386-391**

FlatBug can filter predictions outside a valid size range:

```python
if valid_size_range is not None and valid_size_range[0] > 0 and valid_size_range[1] > 0:
    valid_size = ((boxes[:, 2:] - boxes[:, :2]).log().sum(dim=1) / 2).exp()  # geometric mean of width/height
    valid = (valid_size >= valid_size_range[0]) & (valid_size <= valid_size_range[1])
    pred = pred[valid]
    boxes = boxes[valid]
```

**Implication:** Very small or very large detections can be filtered based on geometric mean of bbox dimensions.

### 5.4 Multi-Scale Pyramid Inference

**File:** `flat_bug/predictor.py` **Lines 1612-1632**

FlatBug performs inference at multiple scales using a pyramid approach:

```python
scales = []
if single_scale:
    scales = [1]
else:
    s = self.TILE_SIZE / max_dim
    if s >= 1:
        scales.append(s)
    else:
        while s <= 0.9:  # Cut off at 90%, to avoid having s~1 and s=1.
            scales.append(s)
            s /= scale_increment  # default: 2/3
        if s != 1:
            scales.append(1.0)
```

**Implication:** FlatBug runs detection at multiple scales (e.g., 1.0, 0.67, 0.44, ...) and merges results. This helps detect insects of varying sizes. SAM3's pyramid tiling strategy may differ.

### 5.5 NMS Types and IoU Threshold

**File:** `flat_bug/yolo_helpers.py` **Lines 399-414**

FlatBug supports multiple NMS strategies:

```python
if nms != 0:
    if nms == 1:
        nms_ind = nms_boxes(boxes, pred[:, 4], iou_threshold=iou_threshold)  # Standard box NMS
    elif nms == 2:
        nms_ind = fancy_nms(boxes, iou_boxes, pred[:, 4], iou_threshold=iou_threshold, return_indices=True)  # Custom NMS
    elif nms == 3:
        masks = process_mask(...)
        nms_ind = nms_masks(masks, pred[:, 4], iou_threshold=iou_threshold, ...)  # Mask-based NMS
```

**Implication:** FlatBug uses mask-based NMS (nms=3) for more accurate duplicate removal based on actual mask overlap, not just bounding boxes.

### 5.6 Polygon Linear Interpolation Before Scaling

**File:** `flat_bug/geometric.py` **Lines 236-257, 280-282**

Before scaling contours, FlatBug interpolates additional vertices:

```python
geometric.py line 236
def linear_interpolate(poly, scale):
    """Linearly interpolates a N x 2 polygon to have N x scale vertices."""
    new_poly = np.zeros((poly.shape[0] * scale, 2), dtype=np.float32)
    for i in range(poly.shape[0] - 1):
        new_poly[i*scale:(i+1)*scale] = np.linspace(poly[i], poly[i+1], scale, endpoint=False)
    ...

# In scale_contour:
n_interp = max(1, int(np.ceil(scale.max())) * 2)
contour = linear_interpolate(contour, n_interp)
```

**Implication:** Before upscaling 256→1024, FlatBug interpolates the polygon to 2× the scale factor's vertices, resulting in much smoother polygons. This is a key reason GT polygons have ~150 points vs SAM3's ~43.

### 5.7 Centroid-Based Drift Correction

**File:** `flat_bug/geometric.py` **Lines 280, 295-297**

After scaling, FlatBug corrects for centroid drift:

```python
centroid = contour.mean(axis=0)
# ... scaling and rounding operations ...
drift = centroid - contour.mean(axis=0)
return (contour + drift).round().astype(np.int32)[(n_interp // 2)::n_interp].copy()
```

**Implication:** Ensures the scaled polygon's centroid matches the original, preventing systematic shifts.

### 5.8 Post-NMS Mask Combination with IoS (Intersection over Smaller)

**File:** `flat_bug/nms.py` **Lines 170-200**

FlatBug can use IoS instead of IoU for mask comparisons: (this needs further investigation)

```python
def ios_masks_2sets(m1s, m2s, ...):
    """Computes IoS (Intersection over Smaller area) between all pairs between two sets of masks."""
    # IoS[i,j] = intersection[i, j] / min(m1s[i].sum(), m2s[j].sum())
```

**Implication:** IoS is more lenient for overlapping masks of different sizes, which may affect how duplicates are merged.

### 5.9 Confidence and Area Filtering in COCO Output

**File:** `flat_bug/coco_utils.py` **Lines 289-327**

When exporting to COCO format, FlatBug can apply post-hoc filtering:

```python
def filter_coco(coco, confidence=None, area=None, verbose=False):
    filtered_annotations = []
    for a in coco["annotations"]:
        if confidence is not None and "conf" in a:
            if a["conf"] < confidence:
                continue
        if area is not None and "bbox" in a:
            _, _, w, h = a["bbox"]
            if (w * h) < area:
                continue
        filtered_annotations += [a]
    return {..., "annotations": filtered_annotations}
```

**Implication:** FlatBug may apply confidence/area thresholds when saving predictions. Check if these filters are applied in evaluation. (this note needs further investigation)

### 5.10 256×256 Mask with 3× Upscaling for Contour Extraction

**File:** `flat_bug/yolo_helpers.py` **Line 123**

When merging tiles, FlatBug upscales masks 3× before finding contours:

```python
polygons = [find_contours(resize_mask(mask, [256 * 3, 256 * 3]), True) * (1024 / 256) / 3 + o.flip(0).unsqueeze(0) 
            for r, o in zip(results, box_offsetters) for mask in r.masks.data]
```

**Implication:** FlatBug upscales 256→768 (3×) before contour extraction to get smoother boundaries, then scales to final resolution. This is different from direct contour extraction at native resolution.

---

## Part 6: Summary of Key Differences to Address

| Step | FlatBug | SAM3 (Current) | Impact |
|------|---------|----------------|--------|
| Mask resolution | 256×256, upscaled 3× for contours | Native resolution | Smoother contours |
| Contour extraction | `cv2.CHAIN_APPROX_NONE` | `cv2.CHAIN_APPROX_SIMPLE` | More polygon points |
| Polygon interpolation | Linear interpolation before scaling | None | Smoother curves |
| Contour expansion | `expand_by_one=True` (1px outward) | None | Larger masks |
| BBox padding | `pad=5` after inference | None | Slightly larger boxes |
| Simplification tolerance | Dynamic: `scale/2` (~2.0 for 4× upscale) | Fixed: 1.0 | Different smoothing |
| Minimum mask filter | `area < 3 pixels` removed | Unknown | Removes noise |
| Mask dilation | None (but expansion in scaling) | Need 7-9px to match GT | Key metric difference |
| NMS type | Mask-based (IoU on actual masks) | Likely box-based | Affects overlap handling |
| Multi-scale pyramid | Yes (scales: 1.0, 0.67, 0.44, ...) | May differ | Detection at different scales |

---

### Recommendations for SAM3 Implementation

1. **Implement 3× upscaling before contour extraction** - Match FlatBug's smoother contour pipeline
2. **Use `cv2.CHAIN_APPROX_NONE`** - Keep all contour points (already documented)
3. **Add polygon linear interpolation** - Interpolate vertices before scaling
4. **Add `expand_by_one` equivalent** - Expand contours 1px outward from centroid
5. **Add centroid drift correction** - Ensure scaled polygon centroid matches original
6. **Apply 7-9px mask dilation** - Match GT annotation style (already tested)
7. **Filter minimum mask area** - Remove masks < 3 pixels
8. **Consider mask-based NMS** - More accurate duplicate removal

---

*Document created: December 31, 2025*  
*Last updated: December 31, 2025*  
*Project: SAM3 Insect Segmentation - Bachelor Thesis Research*
