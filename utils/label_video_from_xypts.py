#!/usr/bin/env python3
"""
Label video with markers from xypts.csv file.

This script takes a video file and an Argus xypts.csv file containing 2D coordinates
for markers across multiple cameras, and overlays colored circles on the video for
each marker in the specified camera view.

Usage:
    python label_video_from_xypts.py <video_file> <xypts_csv> <camera_num> [options]

Example:
    python label_video_from_xypts.py video_cam1.mp4 pose_result_xypts.csv 1 --output labeled_video.mp4
    python label_video_from_xypts.py video_cam2.mp4 pose_result_xypts.csv 2 --valid-only --marker-size 5
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_distinct_colors(n):
    """
    Generate n visually distinct colors.
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List of (B, G, R) tuples for OpenCV
    """
    if n <= 10:
        # Use colorblind-safe palette from Argus colors
        base_colors = [
            (0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89), 
            (95, 158, 209), (200, 82, 0), (137, 137, 137), (162, 200, 236), 
            (255, 188, 121), (207, 207, 207)
        ]
        # Convert RGB to BGR for OpenCV
        colors = [(b, g, r) for (r, g, b) in base_colors[:n]]
    else:
        # Generate colors using HSV color space for more colors
        colors = []
        for i in range(n):
            hue = int(180 * i / n)  # Spread across hue spectrum
            hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))
    
    return colors


def parse_xypts_header(header, camera_num):
    """
    Parse the xypts.csv header to identify markers for a specific camera.
    
    Args:
        header: List of column names from CSV
        camera_num: Camera number to extract (1, 2, 3, etc.)
        
    Returns:
        Dictionary mapping marker names to column indices (x_col, y_col)
    """
    cam_suffix = f'_cam_{camera_num}_'
    markers = {}
    
    for idx, col_name in enumerate(header):
        if cam_suffix in col_name:
            if col_name.endswith('_x'):
                # Extract marker name
                marker_name = col_name.split(cam_suffix)[0]
                # Find corresponding y column
                y_col_name = f'{marker_name}{cam_suffix}y'
                if y_col_name in header:
                    y_idx = header.index(y_col_name)
                    markers[marker_name] = (idx, y_idx)
    
    return markers


def get_valid_frames(df, marker_dict):
    """
    Identify frames that have at least one valid (non-NaN) marker coordinate.
    
    Args:
        df: DataFrame containing the xypts data
        marker_dict: Dictionary mapping marker names to column indices
        
    Returns:
        List of frame indices with valid data
    """
    valid_frames = []
    
    for frame_idx in range(len(df)):
        has_valid = False
        for marker_name, (x_idx, y_idx) in marker_dict.items():
            x_val = df.iloc[frame_idx, x_idx]
            y_val = df.iloc[frame_idx, y_idx]
            if not (pd.isna(x_val) or pd.isna(y_val)):
                has_valid = True
                break
        if has_valid:
            valid_frames.append(frame_idx)
    
    return valid_frames


def label_video(video_path, csv_path, camera_num, output_path=None, 
                marker_size=3, valid_only=False, frame_limit=None, flip_y=False):
    """
    Create a labeled video with markers from xypts.csv.
    
    Args:
        video_path: Path to input video file
        csv_path: Path to xypts.csv file
        camera_num: Camera number (1, 2, 3, etc.)
        output_path: Path for output video (default: <input>_labeled.mp4)
        marker_size: Radius of marker circles in pixels
        valid_only: If True, only output frames with valid marker data
        frame_limit: Maximum number of frames to process (None for all)
        flip_y: If True, flip y-coordinates (for lower-left origin in CSV)
    """
    # Validate inputs
    video_path = Path(video_path)
    csv_path = Path(csv_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Set output path
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_cam{camera_num}_labeled.mp4"
    else:
        output_path = Path(output_path)
    
    print(f"Reading CSV file: {csv_path}")
    # Read CSV file - try comma first, then tab if that fails
    try:
        df = pd.read_csv(csv_path, sep=',')
        # Check if we got proper columns (more than 1 column means correct delimiter)
        if len(df.columns) == 1:
            # Probably tab-separated, try again
            df = pd.read_csv(csv_path, sep='\t')
    except Exception:
        # Fallback to tab-separated
        df = pd.read_csv(csv_path, sep='\t')
    
    header = list(df.columns)
    
    # Parse header to get marker information
    print(f"Parsing markers for camera {camera_num}...")
    marker_dict = parse_xypts_header(header, camera_num)
    
    if not marker_dict:
        raise ValueError(f"No markers found for camera {camera_num} in CSV file")
    
    print(f"Found {len(marker_dict)} markers: {', '.join(marker_dict.keys())}")
    
    # Generate colors for each marker
    colors = generate_distinct_colors(len(marker_dict))
    marker_colors = {name: colors[i] for i, name in enumerate(marker_dict.keys())}
    
    # Identify valid frames if needed
    if valid_only:
        print("Identifying frames with valid marker data...")
        valid_frames = get_valid_frames(df, marker_dict)
        print(f"Found {len(valid_frames)} valid frames out of {len(df)} total frames")
        if len(valid_frames) == 0:
            raise ValueError("No valid frames found in CSV file")
    else:
        valid_frames = list(range(len(df)))
    
    # Apply frame limit if specified
    if frame_limit is not None and len(valid_frames) > frame_limit:
        valid_frames = valid_frames[:frame_limit]
        print(f"Limited to first {frame_limit} valid frames")
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Could not create output video: {output_path}")
    
    print(f"Creating labeled video: {output_path}")
    print(f"Processing {len(valid_frames)} frames...")
    
    # Process frames
    current_frame = 0
    frames_written = 0
    
    with tqdm(total=len(valid_frames)) as pbar:
        for valid_frame_idx in valid_frames:
            # Seek to the correct frame if needed
            if current_frame != valid_frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, valid_frame_idx)
                current_frame = valid_frame_idx
            
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {valid_frame_idx}")
                current_frame += 1
                continue
            
            # Draw markers on frame
            for marker_name, (x_idx, y_idx) in marker_dict.items():
                if valid_frame_idx >= len(df):
                    break
                
                x_val = df.iloc[valid_frame_idx, x_idx]
                y_val = df.iloc[valid_frame_idx, y_idx]
                
                # Skip NaN values
                if pd.isna(x_val) or pd.isna(y_val):
                    continue
                
                # Convert coordinates
                x_coord = int(float(x_val))
                y_coord = int(float(y_val))
                
                # Flip y-coordinate if needed (CSV origin is upper-left, video origin is upper-left)
                # If flip_y is True, it means CSV has lower-left origin and needs to be flipped
                if flip_y:
                    y_coord = height - y_coord
                
                # Draw circle
                color = marker_colors[marker_name]
                center = (x_coord, y_coord)
                cv2.circle(frame, center, marker_size, color, -1)  # -1 = filled circle
            
            # Write frame
            out.write(frame)
            frames_written += 1
            current_frame += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\nSuccess! Wrote {frames_written} frames to {output_path}")
    
    # Print legend
    print("\nMarker Color Legend:")
    for marker_name, color in marker_colors.items():
        print(f"  {marker_name}: RGB{color}")


def main():
    parser = argparse.ArgumentParser(
        description='Label video with markers from Argus xypts.csv file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - label camera 1 video with all frames
  python label_video_from_xypts.py video_cam1.mp4 pose_result_xypts.csv 1
  
  # Only output frames with valid marker data
  python label_video_from_xypts.py video_cam1.mp4 pose_result_xypts.csv 1 --valid-only
  
  # Custom output path and larger markers
  python label_video_from_xypts.py video_cam2.mp4 pose_result_xypts.csv 2 \\
      --output labeled_cam2.mp4 --marker-size 5
  
  # Limit to first 1000 valid frames
  python label_video_from_xypts.py video_cam3.mp4 pose_result_xypts.csv 3 \\
      --valid-only --frame-limit 1000
  
  # Flip y-coordinates if CSV has lower-left origin
  python label_video_from_xypts.py video_cam1.mp4 pose_result_xypts.csv 1 \\
      --flip-y
        """
    )
    
    parser.add_argument('video', type=str,
                        help='Path to input video file')
    parser.add_argument('csv', type=str,
                        help='Path to xypts.csv file')
    parser.add_argument('camera', type=int,
                        help='Camera number (1, 2, 3, etc.)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output video path (default: <input>_cam<N>_labeled.mp4)')
    parser.add_argument('-s', '--marker-size', type=int, default=3,
                        help='Marker circle radius in pixels (default: 3)')
    parser.add_argument('-v', '--valid-only', action='store_true',
                        help='Only output frames with at least one valid marker')
    parser.add_argument('-l', '--frame-limit', type=int, default=None,
                        help='Maximum number of frames to process (default: all)')
    parser.add_argument('-f', '--flip-y', action='store_true',
                        help='Flip y-coordinates (use when CSV has lower-left origin)')
    
    args = parser.parse_args()
    
    try:
        label_video(
            video_path=args.video,
            csv_path=args.csv,
            camera_num=args.camera,
            output_path=args.output,
            marker_size=args.marker_size,
            valid_only=args.valid_only,
            frame_limit=args.frame_limit,
            flip_y=args.flip_y
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
