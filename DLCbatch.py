#!/usr/bin/env python3
"""
DLC Batch Processing Script for Argus GUI
==========================================

This script processes DeepLabCut H5 files in batch mode, performing 3D reconstruction
and exporting results to CSV format.

Usage:
    python DLCtest.py /path/to/data/directory --threshold 0.95

Directory structure expected:

    ├── videos-raw/          # Contains H5 files named cam1_*, cam2_*, etc.
    ├── calibration/
    │   ├── *clicker-profile.txt     # Camera intrinsics
    │   └── *dlt-coefficients.csv    # DLT coefficients
    
Output:
    CSV files with columns: track_x, track_y, track_z, track_error, track_ncams, track_score
"""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
import re

# Add argus_gui to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from argus_gui.tools import uv_to_xyz, get_repo_errors
try:
    import argus.ocam
    ARGUS_OCAM_AVAILABLE = True
except ImportError:
    ARGUS_OCAM_AVAILABLE = False


class DLCBatchProcessor:
    def __init__(self, data_dir, likelihood_threshold=0.95):
        self.data_dir = data_dir
        self.likelihood_threshold = likelihood_threshold
        self.camera_profile = None
        self.dlt_coefficients = None
        self.videos_dir = os.path.join(data_dir, 'videos-raw')
        self.calibration_dir = os.path.join(data_dir, 'calibration')
        
        # Validate directory structure
        self._validate_directories()
        
        # Load calibration files
        self._load_calibration()
    
    def _validate_directories(self):
        """Check that required directories exist."""
        if not os.path.exists(self.videos_dir):
            raise FileNotFoundError(f"videos-raw directory not found: {self.videos_dir}")
        if not os.path.exists(self.calibration_dir):
            raise FileNotFoundError(f"calibration directory not found: {self.calibration_dir}")
    
    def _load_calibration(self):
        """Load camera profile and DLT coefficients."""
        # Find camera profile file
        profile_files = glob.glob(os.path.join(self.calibration_dir, '*clicker-profile.txt'))
        if not profile_files:
            raise FileNotFoundError("No clicker-profile.txt file found in calibration directory")
        profile_file = profile_files[0]
        # Load camera profile (similar to load_camera function in argus-click)
        camera_profile = np.loadtxt(profile_file)
        # Extract image dimensions before processing (needed for Y-coordinate flipping)
        if camera_profile.shape[1] == 12:
            # Columns: [cam_num, focal_len, width, height, cx, cy, skew, distortion_params...]
            self.image_heights = camera_profile[:, 3]  # Column 3 is image height
            self.image_widths = camera_profile[:, 2]   # Column 2 is image width for reference
        
        # Format camera profile based on type (from argus-click load_camera function)
        if camera_profile.shape[1] == 12:
            # Pinhole distortion - remove camera number, width, height, and skew columns
            self.camera_profile = np.delete(camera_profile, [0, 2, 3, 6], axis=1)
        elif camera_profile.shape[1] == 13:
            # CMei's omnidirectional distortion model
            new_list = []
            camera_number_checker = 1
            for profile in camera_profile:
                if profile[0] != camera_number_checker:
                    raise ValueError(f"Camera index mismatch in profile line {camera_number_checker}")
                # Remove camera index and create undistorter
                profile = np.delete(profile, [0])
                if ARGUS_OCAM_AVAILABLE:
                    new_list.append(argus.ocam.CMeiUndistorter(argus.ocam.ocam_model.from_array(profile)))
                else:
                    raise ImportError("argus.ocam module not available for omnidirectional cameras")
                camera_number_checker += 1
            self.camera_profile = new_list
        elif camera_profile.shape[1] == 19:
            # Scaramuzza's omnidirectional distortion model  
            new_list = []
            camera_number_checker = 1
            for profile in camera_profile:
                if profile[0] != camera_number_checker:
                    raise ValueError(f"Camera index mismatch in profile line {camera_number_checker}")
                # Remove camera index and create undistorter
                profile = np.delete(profile, [0])
                if ARGUS_OCAM_AVAILABLE:
                    new_list.append(argus.ocam.PointUndistorter(argus.ocam.ocam_model.from_array(profile)))
                else:
                    raise ImportError("argus.ocam module not available for omnidirectional cameras")
                camera_number_checker += 1
            self.camera_profile = new_list
        else:
            raise ValueError(f"Unsupported camera profile format: {camera_profile.shape[1]} columns")
        
        # Find DLT coefficients file
        dlt_files = glob.glob(os.path.join(self.calibration_dir, '*dlt-coefficients.csv'))
        if not dlt_files:
            raise FileNotFoundError("No dlt-coefficients.csv file found in calibration directory")
        dlt_file = dlt_files[0]
        # Load DLT coefficients (exact same as argus-click load_DLT function, lines 488-489)
        self.dlt_coefficients = np.loadtxt(dlt_file, delimiter=',')
        self.dlt_coefficients = self.dlt_coefficients.T  # Transpose like argus-click does
    
    def _find_trials(self):
        """Find all unique trials based on H5 file naming patterns."""
        h5_files = glob.glob(os.path.join(self.videos_dir, '*.h5'))
        if not h5_files:
            raise FileNotFoundError("No H5 files found in videos-raw directory")
        
        # Group files by trial (everything after camera number)
        trials = defaultdict(list)
        for h5_file in h5_files:
            basename = os.path.basename(h5_file)
            # Extract camera number and trial identifier
            match = re.match(r'cam(\d+)(.+)\.h5$', basename)
            if match:
                cam_num = int(match.group(1))
                trial_id = match.group(2)
                trials[trial_id].append((cam_num, h5_file))
            else:
                print(f"Warning: Could not parse camera number from {basename}")
        
        # Sort cameras within each trial
        for trial_id in trials:
            trials[trial_id].sort(key=lambda x: x[0])  # Sort by camera number
        
        print(f"Found {len(trials)} trials")
        return trials
    
    def _load_dlc_data(self, h5_files, trial_id):
        """Load DLC data from H5 files for a single trial."""
        print(f"Processing trial: {trial_id}")
        
        all_data = {}
        track_names = None
        max_frames = 0
        
        for cam_num, h5_file in h5_files:
            print(f"  Loading camera {cam_num}: {os.path.basename(h5_file)}")
            
            try:
                # Load H5 file
                df = pd.read_hdf(h5_file, key='df_with_missing')
                scorer_name = df.columns.get_level_values('scorer')[0]
                
                # Get track names from first file
                if track_names is None:
                    # Find bodypart level name
                    bodypart_level = None
                    for level_name in df.columns.names:
                        if level_name and isinstance(level_name, str) and 'bodypart' in level_name.lower():
                            bodypart_level = level_name
                            break
                    
                    if bodypart_level:
                        track_names = df.columns.get_level_values(bodypart_level).unique().tolist()
                    else:
                        raise ValueError("No bodypart level found in DLC data")
                
                # Determine if this is multi-animal DLC by checking MultiIndex structure
                is_multi_animal = hasattr(df.columns, 'nlevels') and df.columns.nlevels == 4
                
                if is_multi_animal:
                    # Multi-animal DLC - for now, just handle single animal case
                    print("    Multi-animal DLC detected - using first individual")
                    individuals = df.columns.get_level_values('individual').unique().tolist()
                    individual = individuals[0]
                    cam_data = df[scorer_name][individual]
                else:
                    # Single animal DLC
                    cam_data = df[scorer_name]
                
                # Apply likelihood threshold filtering
                for track in track_names:
                    if track in cam_data.columns.get_level_values(0):
                        likelihood = cam_data[track]['likelihood'].values
                        x_vals = cam_data[track]['x'].values
                        y_vals = cam_data[track]['y'].values
                        
                        # Flip Y coordinates from DeepLabCut (upper-left origin) to DLT (lower-left origin)
                        # Y_dlt = image_height - Y_dlc
                        if hasattr(self, 'image_heights') and cam_num-1 < len(self.image_heights):
                            image_height = self.image_heights[cam_num-1]
                            y_vals = image_height - y_vals
                        
                        # Set coordinates to NaN where likelihood is below threshold
                        low_likelihood_mask = likelihood <= self.likelihood_threshold
                        x_vals[low_likelihood_mask] = np.nan
                        y_vals[low_likelihood_mask] = np.nan
                        
                        # Store the filtered data
                        if cam_num not in all_data:
                            all_data[cam_num] = {}
                        all_data[cam_num][track] = {
                            'x': x_vals,
                            'y': y_vals,
                            'likelihood': likelihood
                        }
                
                max_frames = max(max_frames, len(df))
                
            except Exception as e:
                print(f"    Error loading {h5_file}: {e}")
                continue
        
        return all_data, track_names, max_frames
    
    def _format_data_for_reconstruction(self, all_data, track_names, max_frames):
        """Format loaded DLC data for 3D reconstruction."""
        n_cameras = len(all_data)
        n_tracks = len(track_names)
        
        # Create data array: frames x (2 * cameras * tracks)
        # IMPORTANT: Must match argus-click format which is:
        # [track1_cam1_x, track1_cam1_y, track1_cam2_x, track1_cam2_y, ..., track2_cam1_x, track2_cam1_y, ...]
        pts_data = np.full((max_frames, 2 * n_cameras * n_tracks), np.nan)
        
        for track_idx, track in enumerate(track_names):
            for cam_num in sorted(all_data.keys()):
                if cam_num in all_data and track in all_data[cam_num]:
                    cam_idx = cam_num - 1  # Convert to 0-based index
                    
                    # Calculate column indices for this camera and track
                    # argus-click format: each track gets (2 * n_cameras) columns
                    base_col = track_idx * (2 * n_cameras) + cam_idx * 2
                    
                    track_data = all_data[cam_num][track]
                    x_data = track_data['x']
                    y_data = track_data['y']
                    
                    # Ensure we don't exceed max_frames
                    data_len = min(len(x_data), max_frames)
                    
                    pts_data[:data_len, base_col] = x_data[:data_len]      # x coordinates
                    pts_data[:data_len, base_col + 1] = y_data[:data_len]  # y coordinates
        
        return pts_data
    
    def _calculate_scores_and_ncams(self, all_data, track_names, max_frames):
        """Calculate average likelihood scores and number of cameras per point."""
        n_tracks = len(track_names)
        
        scores = np.full((max_frames, n_tracks), np.nan)
        ncams = np.zeros((max_frames, n_tracks), dtype=int)
        
        for track_idx, track in enumerate(track_names):
            for frame in range(max_frames):
                valid_likelihoods = []
                n_valid_cams = 0
                
                for cam_num in sorted(all_data.keys()):
                    if (cam_num in all_data and track in all_data[cam_num] and 
                        frame < len(all_data[cam_num][track]['likelihood'])):
                        
                        likelihood = all_data[cam_num][track]['likelihood'][frame]
                        x_val = all_data[cam_num][track]['x'][frame]
                        y_val = all_data[cam_num][track]['y'][frame]
                        
                        # Check if this point passed the threshold (not NaN after filtering)
                        if not (np.isnan(x_val) or np.isnan(y_val)) and likelihood > self.likelihood_threshold:
                            valid_likelihoods.append(likelihood)
                            n_valid_cams += 1
                
                if valid_likelihoods:
                    scores[frame, track_idx] = np.mean(valid_likelihoods)
                    ncams[frame, track_idx] = n_valid_cams
        
        return scores, ncams
    
    def _perform_3d_reconstruction(self, pts_data, track_names):
        """Perform 3D reconstruction using uv_to_xyz."""
        n_tracks = len(track_names)
        max_frames = pts_data.shape[0]
        
        if self.camera_profile is None:
            raise ValueError("Camera profile is None")
        
        # Reconstruct each track separately
        xyz_results = []
        
        for track_idx in range(n_tracks):
            # Extract data for this track (all cameras) - exact argus-click pattern
            # argus-click: pts[:, j * 2 * len(camera_profile):(j + 1) * 2 * len(camera_profile)]
            track_pts = pts_data[:, track_idx * 2 * len(self.camera_profile):(track_idx + 1) * 2 * len(self.camera_profile)]
            
            try:
                # Perform 3D reconstruction
                xyz = uv_to_xyz(track_pts, self.camera_profile, self.dlt_coefficients)
                xyz_results.append(xyz)
                
            except Exception as e:
                print(f"    Error reconstructing {track_names[track_idx]}: {e}")
                # Fill with NaNs if reconstruction fails
                xyz_results.append(np.full((max_frames, 3), np.nan))
        
        # Now calculate reprojection errors for ALL tracks at once (like argus-click does)
        # Calculate reprojection errors for all tracks at once (like argus-click does)
        try:
            # Concatenate all XYZ data horizontally like argus-click does
            if xyz_results:
                xyz_all = xyz_results[0]  # Start with first track
                for k in range(1, len(xyz_results)):
                    xyz_all = np.hstack((xyz_all, xyz_results[k]))
                # Calculate reprojection errors using exact same call as argus-click (line 2882)
                all_errors = get_repo_errors(xyz_all, pts_data, self.camera_profile, self.dlt_coefficients)
                
                # get_repo_errors returns (n_tracks, n_frames), so transpose like argus-click does
                all_errors = all_errors.T  # Now (n_frames, n_tracks)
                
                # Split errors back by track
                error_results = []
                for track_idx in range(n_tracks):
                    track_errors = all_errors[:, track_idx]  # Each column corresponds to a track
                    error_results.append(track_errors)
            else:
                # No valid reconstructions
                error_results = [np.full(max_frames, np.nan) for _ in range(n_tracks)]
                
        except Exception as e:
            print(f"    Error calculating reprojection errors: {e}")
            # Fill with NaNs if error calculation fails
            error_results = [np.full(max_frames, np.nan) for _ in range(n_tracks)]
        
        return xyz_results, error_results
    
    def _extract_timestamp_from_trial_id(self, trial_id):
        """Extract timestamp from trial identifier for CSV filename."""
        # Look for datetime patterns in the trial ID
        # Common formats: _YYYY-MM-DD_HH-MM-SS, _YYYYMMDD_HHMMSS, etc.
        
        # Try different timestamp patterns
        patterns = [
            r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',  # YYYY-MM-DD_HH-MM-SS
            r'(\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2})',      # YYYYMMDD_HHMMSS
            r'(\d{8}_\d{6})',                           # YYYYMMDD_HHMMSS
            r'(\d{4}-\d{2}-\d{2})',                     # YYYY-MM-DD
            r'(\d{8})',                                 # YYYYMMDD
        ]
        
        for pattern in patterns:
            match = re.search(pattern, trial_id)
            if match:
                return match.group(1)
        
        # If no timestamp pattern found, use the trial_id itself (cleaned)
        # Remove leading/trailing underscores and replace problematic characters
        clean_id = trial_id.strip('_').replace('/', '-').replace('\\', '-').replace(' ', '_')
        return clean_id
    
    def _save_results(self, trial_id, track_names, xyz_results, error_results, scores, ncams):
        """Save results to CSV file."""
        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.data_dir, 'pose-3d-argus')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = self._extract_timestamp_from_trial_id(trial_id)
        output_file = os.path.join(output_dir, f"{timestamp}_3d_tracks.csv")
        
        max_frames = len(xyz_results[0])
        
        # Prepare data dictionary for DataFrame
        data_dict = {}
        
        # Add frame index
        data_dict['frame'] = range(max_frames)
        
        # Add data for each track
        for track_idx, track_name in enumerate(track_names):
            xyz = xyz_results[track_idx]
            errors = error_results[track_idx]
            track_scores = scores[:, track_idx]
            track_ncams = ncams[:, track_idx]
            
            # Add columns for this track
            data_dict[f'{track_name}_x'] = xyz[:, 0]
            data_dict[f'{track_name}_y'] = xyz[:, 1]
            data_dict[f'{track_name}_z'] = xyz[:, 2]
            data_dict[f'{track_name}_error'] = errors
            data_dict[f'{track_name}_ncams'] = track_ncams
            data_dict[f'{track_name}_score'] = track_scores
        
        # Create and save DataFrame
        df = pd.DataFrame(data_dict)
        df.to_csv(output_file, index=False, na_rep='NaN')
        
        print(f"  Saved results to: {output_file}")
        return output_file
    
    def process_all_trials(self):
        """Process all trials in the data directory."""
        trials = self._find_trials()
        
        processed_files = []
        
        for trial_id, h5_files in trials.items():
            try:
                # Load DLC data
                all_data, track_names, max_frames = self._load_dlc_data(h5_files, trial_id)
                
                if not all_data or not track_names:
                    print(f"  No valid data found for trial {trial_id}")
                    continue
                
                # Format data for reconstruction
                pts_data = self._format_data_for_reconstruction(all_data, track_names, max_frames)
                
                # Calculate scores and camera counts
                scores, ncams = self._calculate_scores_and_ncams(all_data, track_names, max_frames)
                
                # Perform 3D reconstruction
                xyz_results, error_results = self._perform_3d_reconstruction(pts_data, track_names)
                
                # Save results
                output_file = self._save_results(trial_id, track_names, xyz_results, error_results, scores, ncams)
                processed_files.append(output_file)
                
            except Exception as e:
                print(f"  Error processing trial {trial_id}: {e}")
                continue
        
        print(f"\nProcessing complete! Generated {len(processed_files)} files:")
        for file_path in processed_files:
            print(f"  {file_path}")
        
        return processed_files


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Process DeepLabCut H5 files for 3D reconstruction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory structure expected:
    data_directory/
    ├── videos-raw/          # Contains H5 files named cam1_*, cam2_*, etc.
    ├── calibration/
    │   ├── *clicker-profile.txt     # Camera intrinsics
    │   └── *dlt-coefficients.csv    # DLT coefficients

Output CSV columns for each track (example for 'L1hip'):
    L1hip_x, L1hip_y, L1hip_z, L1hip_error, L1hip_ncams, L1hip_score
        """
    )
    
    parser.add_argument('data_directory', 
                       help='Path to directory containing videos-raw and calibration folders')
    parser.add_argument('--threshold', '-t', type=float, default=0.95,
                       help='DLC likelihood threshold (default: 0.95)')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_directory):
        print(f"Error: Data directory does not exist: {args.data_directory}")
        sys.exit(1)
    
    try:
        # Create processor and run
        processor = DLCBatchProcessor(args.data_directory, args.threshold)
        processor.process_all_trials()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()