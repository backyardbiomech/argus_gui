# Comparison: Old vs New Error Calculation Approaches

## The Problem
When only 2 cameras have data, how should we calculate reconstruction error?

---

## Original Approach (Pre-Fix)

```python
errors[j] = np.sqrt(epsilon / float(len(toSum) * 2 - 3))
```

**Issues:**
- 2 cameras: DOF = 1, dividing by 1 inflated errors by √3 ≈ 1.73x
- Example: 18.72 instead of reasonable value

**Result:** Artificially high, inconsistent with 3+ camera errors

---

## First Fix Attempt (Normalization + Averaging)

```python
if len(toSum) == 2:
    errors[j] = np.sqrt(epsilon / 3.0)  # Use 3-camera normalization
    # Then average all 2-camera errors together
else:
    errors[j] = np.sqrt(epsilon / dof)
```

**Improvements:**
✓ Fixed inflation issue (errors reduced by √3)
✓ Made 2-camera and 3-camera errors comparable

**Remaining Issues:**
✗ All 2-camera frames show same error value (averaged)
✗ No frame-by-frame information
✗ Not informative for quality assessment

**Example output:**
```
L1knee (2-cam):  Mean: 1.7267, Std: 0.0000, Unique: 1
L1ankle (2-cam): Mean: 10.8346, Std: 0.0000, Unique: 1
```

---

## Improved Approach (Current) ⭐

```python
errors[j] = np.sqrt(epsilon / float(n_cams))
```

**What it represents:**
- **RMS reprojection error per camera**
- Square root of (sum of squared pixel errors / number of cameras)
- Interpretable: "average pixel error per camera view"

**Advantages:**
✓ No DOF-related inflation
✓ Preserves frame-by-frame variation
✓ Informative for all camera counts
✓ Comparable across different numbers of cameras
✓ More interpretable than statistical RMSE
✓ Directly reflects reconstruction quality

**Example output:**
```
L1knee (2-cam):  Mean: 2.1148, Std: 1.4094, Unique: 137 frames
                 Range: 0.0050 to 7.2976
                 Sample: [2.54, 3.40, 3.36, 2.02, 2.13]

L1knee (3-cam):  Mean: 2.2047, Std: 1.2258, Unique: 37 frames
                 Ratio: 0.96 (very similar!)
```

---

## Why This Works Better

### Statistical Perspective
- **2-camera case**: Still has uncertainty due to minimal DOF, but that's about confidence, not the error value itself
- **Frame variation**: Real! Different frames have different reconstruction quality
- **Interpretability**: "This frame has 2.5 pixel average reprojection error" is meaningful

### Practical Benefits
1. **Quality assessment**: Can identify poor frames vs good frames
2. **Debugging**: Can spot systematic issues (e.g., one camera miscalibrated)
3. **Filtering**: Can filter by error threshold meaningfully
4. **Comparable**: 2-cam and 3-cam errors on similar scale

### What the Error Means
- **Low error (< 2 pixels)**: Good reconstruction, cameras agree well
- **Medium error (2-5 pixels)**: Acceptable, some camera disagreement
- **High error (> 5 pixels)**: Poor reconstruction, may need review
- Values apply regardless of number of cameras

---

## Mathematical Justification

### Old DOF approach:
```
RMSE_statistical = sqrt(sum_squared_errors / DOF)
where DOF = n_observations - n_parameters = (n_cams * 2) - 3
```
**Problem**: DOF=1 for 2 cameras makes denominator too small

### New RMS approach:
```
RMS_per_camera = sqrt(sum_squared_errors / n_cams)
```
**Advantage**: Normalizes by number of cameras, not DOF

### Relationship:
```
For 3 cameras:
  Old: sqrt(epsilon / 3)
  New: sqrt(epsilon / 3)  ← Same!

For 2 cameras:
  Old: sqrt(epsilon / 1) = sqrt(epsilon)
  New: sqrt(epsilon / 2)
  Reduction: sqrt(2) ≈ 1.41x
```

Actually, the new approach divides by 2 for 2 cameras, which is different from our first fix (dividing by 3). Let me recalculate the comparison...

---

## Numerical Comparison

### User's Example (18.72218751):

| Approach | Formula | Result | Notes |
|----------|---------|--------|-------|
| Original | sqrt(ε/1) | 18.72 | Inflated by √3 vs 3-cam |
| First Fix | sqrt(ε/3) | 10.81 | Same scale as 3-cam, but averaged |
| Current | sqrt(ε/2) | 13.24 | Per-camera RMS, frame variation |

### New Data (L1knee):
```
2-camera: Mean=2.11, Std=1.41, Range=[0.005, 7.30]
3-camera: Mean=2.20, Std=1.23
Ratio: 0.96 (nearly identical!)
```

---

## Recommendation

**Use the current improved approach** because:

1. It's statistically sound (RMS error per camera)
2. It preserves frame-level information
3. It's interpretable and actionable
4. It works consistently across all camera counts
5. Values directly reflect reprojection quality

The key insight: We don't need to "correct" for DOF. We just need to report the actual per-camera reprojection error, which is inherently meaningful and comparable.
