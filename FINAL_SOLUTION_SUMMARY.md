# Final Summary: Improved Error Calculation for 2-Camera Reconstructions

## Question Raised
> "Is using the average error for all 2-camera frames the correct approach? It doesn't seem very informative on a frame-by-frame basis, especially if there are only two cameras. Is there a better, statistically sound approach to estimate the reconstruction error on a frame-by-frame basis?"

**Answer: You were absolutely right!** Averaging removes all frame-level information. The new approach is much better.

---

## The Solution: Per-Camera RMS Error

### Formula
```python
error = sqrt(sum_of_squared_pixel_errors / number_of_cameras)
```

### What It Means
- **RMS (Root Mean Square) reprojection error per camera**
- Interpretable: "The average pixel disagreement per camera view"
- Works for any number of cameras (2, 3, 4+)
- No statistical assumptions needed

---

## Real-World Example: Reye Track

### Data Overview
- **595 frames** with 2-camera reconstructions
- **112 frames** with 3-camera reconstructions

### Frame-by-Frame Comparison

| Frame | NCams | Error (New) | Error (Old Averaged) | Difference |
|-------|-------|-------------|---------------------|------------|
| 0     | 2     | 0.55 px     | 0.74 px             | Lost detail |
| 2     | 2     | 0.24 px     | 0.74 px             | Lost detail |
| 6     | 2     | 0.04 px     | 0.74 px             | Lost detail |
| 7     | 2     | 1.16 px     | 0.74 px             | Lost detail |
| 8     | 2     | 1.04 px     | 0.74 px             | Lost detail |
| 10    | 2     | 0.84 px     | 0.74 px             | Lost detail |
| 17    | 2     | 1.07 px     | 0.74 px             | Lost detail |
| 28    | 2     | 1.36 px     | 0.74 px             | Lost detail |

**Old approach**: Every single one of the 595 frames showed `0.74 px` — no variation!

**New approach**: Each frame shows its actual reconstruction quality:
- Best frame: `0.005 px` (excellent agreement)
- Worst frame: `2.48 px` (cameras disagree more)
- Most frames: `0.2 - 1.0 px` (good quality)

---

## Statistical Comparison

### 2-Camera Frames (Reye)
```
Old Averaged Approach:
  All 595 frames: 0.74 pixels (identical)
  Std:           0.00 pixels (no variation)
  Unique values: 1

New Per-Camera RMS Approach:
  Mean:          0.60 pixels
  Std:           0.45 pixels (natural variation!)
  Range:         0.01 to 2.48 pixels
  Unique values: 595 (every frame is informative!)
```

### 3-Camera Frames (Reye)
```
New approach:
  Mean:          2.34 pixels
  Std:           0.72 pixels
  Ratio:         0.26 (2cam vs 3cam)
```

---

## Why This Is Better

### 1. **Informative**
- Can identify excellent reconstructions (< 0.5 px)
- Can flag poor reconstructions (> 2.0 px)
- Can track quality over time

### 2. **Actionable**
- Filter frames by quality threshold
- Focus manual review on high-error frames
- Validate calibration quality

### 3. **Statistically Sound**
- RMS error is a standard, well-understood metric
- No problematic DOF assumptions
- Comparable across camera counts

### 4. **Interpretable**
- "0.5 pixels per camera" is meaningful
- Directly relates to image resolution
- Easy to set quality thresholds

---

## What About Statistical Uncertainty?

### Important Distinction
**Reconstruction error** ≠ **Statistical uncertainty**

- **Error (what we calculate)**: How well do the cameras agree?
- **Uncertainty (confidence)**: How reliable is this 3D point?

### For 2-Camera Cases
- **Error**: Can be calculated accurately (cameras disagree by X pixels)
- **Uncertainty**: Higher than 3+ cameras (less constrained system)

The per-camera RMS error correctly captures the **reconstruction quality** without confusing it with **statistical confidence**.

If you need confidence intervals, those should be calculated separately (e.g., via bootstrapping, which `argus-click` already does).

---

## Practical Impact

### Before (Averaged Approach)
```
User looking at data: "Why do 595 frames all show exactly 0.74 error?"
Answer: "Because we averaged them for stability"
Result: No useful information for quality control
```

### After (Per-Camera RMS)
```
User looking at data: "Frame 6 shows 0.04 error, frame 28 shows 1.36"
Answer: "Frame 6 has excellent reconstruction, frame 28 has more disagreement"
Result: Can make informed decisions about data quality
```

---
