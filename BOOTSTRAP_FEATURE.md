# Bootstrap Feature Addition to DLCbatch.py

## Overview
Added bootstrapping functionality to `DLCbatch.py` to estimate 95% confidence intervals for 3D reconstructed points, matching the behavior of the `bootstrapXYZs` function in `tools.py`.

## Changes Made

### 1. Imports
- Added `bootstrapXYZs` to the imports from `argus_gui.tools`

### 2. Class Initialization (`DLCBatchProcessor.__init__`)
Added new parameters:
- `bootstrap` (bool, default=False): Enable/disable bootstrapping
- `bs_iterations` (int, default=250): Number of bootstrap iterations
- `display_progress` (bool, default=False): Show progress bars during bootstrapping
- `subframe_interp` (bool, default=True): Enable subframe interpolation

### 3. 3D Reconstruction Method (`_perform_3d_reconstruction`)
- Modified to return 5 values instead of 2:
  - `xyz_results`: 3D coordinates (original)
  - `error_results`: Reprojection errors (original)
  - `bootstrap_ci`: 95% confidence intervals (new, optional)
  - `bootstrap_weights`: Bootstrap weights (new, optional)
  - `bootstrap_tols`: Bootstrap tolerances (new, optional)

- When `self.bootstrap` is True:
  - Calls `bootstrapXYZs()` with the appropriate parameters
  - Handles exceptions gracefully, returning NaN arrays on failure

### 4. Save Results Method (`_save_results`)
Enhanced to save bootstrap results:
- Added optional parameters: `bootstrap_ci`, `bootstrap_weights`, `bootstrap_tols`
- For each track, adds columns (if bootstrap data available):
  - `{track}_ci_x`, `{track}_ci_y`, `{track}_ci_z`: 95% confidence intervals
  - `{track}_weight_x`, `{track}_weight_y`, `{track}_weight_z`: Bootstrap weights
- Creates separate CSV file for bootstrap tolerances: `{timestamp}_bootstrap_tolerances.csv`

### 5. Process All Trials Method (`process_all_trials`)
- Updated to unpack 5 return values from `_perform_3d_reconstruction`
- Passes bootstrap data to `_save_results`

### 6. Command-Line Arguments
Added new arguments:
- `--bootstrap` or `-b`: Enable bootstrapping
- `--bootstrap-iterations` or `-i`: Set number of iterations (default: 250)
- `--display-progress` or `-p`: Show progress bars
- `--no-subframe-interp`: Disable subframe interpolation

### 7. Main Function
- Updated to pass all bootstrap parameters to `DLCBatchProcessor`

### 8. Documentation
- Updated module docstring with bootstrap usage examples
- Updated argument parser epilog with bootstrap output description

## Usage Examples

### Basic 3D reconstruction (no bootstrap):
```bash
python DLCbatch.py /path/to/data --threshold 0.95
```

### With bootstrapping (default 250 iterations):
```bash
python DLCbatch.py /path/to/data --bootstrap
```

### With custom bootstrap iterations:
```bash
python DLCbatch.py /path/to/data --bootstrap --bootstrap-iterations 500
```

### With progress display:
```bash
python DLCbatch.py /path/to/data --bootstrap --display-progress
```

### Disable subframe interpolation:
```bash
python DLCbatch.py /path/to/data --bootstrap --no-subframe-interp
```

## Output Files

### Without Bootstrap
- `{timestamp}_3d_tracks.csv` with columns:
  - `frame`, `{track}_x`, `{track}_y`, `{track}_z`, `{track}_error`, `{track}_ncams`, `{track}_score`

### With Bootstrap
- `{timestamp}_3d_tracks.csv` (same columns as without bootstrap)
- `{timestamp}_xyz-cis.csv`:
  - `{track}_x_lower`, `{track}_y_lower`, `{track}_z_lower` (lower bounds)
  - `{track}_x_upper`, `{track}_y_upper`, `{track}_z_upper` (upper bounds)
- `{timestamp}_spline-weights.csv`:
  - `{track}_x`, `{track}_y`, `{track}_z` (bootstrap weights)
- `{timestamp}_spline-error-tolerances.csv`:
  - `{track}_x`, `{track}_y`, `{track}_z` (spline error tolerances)

## Bootstrap Methodology
The bootstrap implementation follows the same approach as in `tools.py`:

1. **Subframe Interpolation** (optional, 3+ cameras only):
   - Tests subframe offsets from -1 to +1 frames in 0.1 increments
   - Finds optimal offset for each camera to minimize RMSE
   - Adjusts point coordinates accordingly

2. **Bootstrap Sampling**:
   - For each frame and track, perturbs UV coordinates randomly
   - Perturbation follows normal distribution: N(0, rmse * sqrt(2) / sqrt(n_cams))
   - Reconstructs 3D points `bs_iterations` times (default: 250)
   - Calculates standard deviation across all iterations

3. **Confidence Intervals**:
   - Returns SD * 1.96 for 95% CI

4. **Weights and Tolerances**:
   - Weights: 1 / (SD / min(SD)) for each coordinate
   - Tolerances: Sum of (weight * SD²) for each coordinate
   - Used for downstream spline fitting with error weighting

## Technical Notes

- Bootstrap only runs when `--bootstrap` flag is used (not automatic)
- Handles cases where errors might not be computed (fills with NaN)
- Progress bars require `tqdm` library (from `tools.py` import)
- Bootstrap results are gracefully omitted if calculation fails
- Compatible with both pinhole and omnidirectional camera models
