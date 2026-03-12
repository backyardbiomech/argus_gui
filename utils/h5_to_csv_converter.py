#!/usr/bin/env python3
"""
Quick H5 to CSV Converter for DeepLabCut Files
==============================================

Simple script to convert DeepLabCut H5 files to CSV format.
Preserves all data including coordinates, likelihood scores, and metadata.

Usage:
    python h5_to_csv_converter.py /path/to/file.h5
    python h5_to_csv_converter.py /path/to/directory/  # Convert all H5 files in directory
"""

import os
import sys
import pandas as pd
import argparse
import glob


def convert_h5_to_csv(h5_file_path):
    """
    Convert a single DeepLabCut H5 file to CSV format.
    
    Args:
        h5_file_path (str): Path to the H5 file
        
    Returns:
        str: Path to the generated CSV file
    """
    try:
        print(f"Converting: {os.path.basename(h5_file_path)}")
        
        # Load the H5 file
        df = pd.read_hdf(h5_file_path, key='df_with_missing')
        
        # Create output filename
        csv_file_path = h5_file_path.replace('.h5', '.csv')
        
        # Save as CSV
        df.to_csv(csv_file_path, index=True)
        
        print(f"  → Saved: {os.path.basename(csv_file_path)}")
        print(f"  → Shape: {df.shape}")
        
        # Print some basic info about the data
        if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
            # Multi-level columns (typical DLC format)
            try:
                scorer = df.columns.get_level_values('scorer')[0] if 'scorer' in df.columns.names else 'Unknown'
                print(f"  → Scorer: {scorer}")
                
                # Try to get bodyparts
                bodypart_levels = [name for name in df.columns.names if name and isinstance(name, str) and 'bodypart' in name.lower()]
                if bodypart_levels:
                    bodyparts = df.columns.get_level_values(bodypart_levels[0]).unique().tolist()
                    print(f"  → Bodyparts: {len(bodyparts)} ({', '.join(bodyparts[:5])}{'...' if len(bodyparts) > 5 else ''})")
                
                # Check for individuals (multi-animal DLC)
                if 'individual' in df.columns.names:
                    individuals = df.columns.get_level_values('individual').unique().tolist()
                    print(f"  → Individuals: {len(individuals)} ({', '.join(map(str, individuals))})")
                    
            except Exception as e:
                print(f"  → Could not extract detailed info: {e}")
        
        return csv_file_path
        
    except Exception as e:
        print(f"  → Error converting {os.path.basename(h5_file_path)}: {e}")
        return None


def convert_multiple_files(file_paths):
    """
    Convert multiple H5 files to CSV format.
    
    Args:
        file_paths (list): List of H5 file paths
        
    Returns:
        list: List of successfully converted CSV file paths
    """
    converted_files = []
    
    for h5_file in file_paths:
        if os.path.exists(h5_file) and h5_file.lower().endswith('.h5'):
            csv_file = convert_h5_to_csv(h5_file)
            if csv_file:
                converted_files.append(csv_file)
    
    return converted_files


def convert_specific_cameras(base_path, camera_numbers=[1, 2, 3]):
    """
    Convert specific camera files based on a base path pattern.
    
    Args:
        base_path (str): Base path with cam1 in it
        camera_numbers (list): List of camera numbers to convert
        
    Returns:
        list: List of successfully converted CSV file paths
    """
    converted_files = []
    
    for cam_num in camera_numbers:
        # Replace cam1 with the current camera number
        cam_file = base_path.replace('cam1_', f'cam{cam_num}_')
        
        if os.path.exists(cam_file):
            print(f"\n--- Processing Camera {cam_num} ---")
            csv_file = convert_h5_to_csv(cam_file)
            if csv_file:
                converted_files.append(csv_file)
        else:
            print(f"\n--- Camera {cam_num} file not found: {os.path.basename(cam_file)} ---")
    
    return converted_files


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Convert DeepLabCut H5 files to CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert single file
    python h5_to_csv_converter.py /path/to/file.h5
    
    # Convert all H5 files in directory
    python h5_to_csv_converter.py /path/to/directory/
    
    # Convert specific camera files (cam1, cam2, cam3) from cam1 path
    python h5_to_csv_converter.py --cameras /path/to/cam1_file.h5
        """
    )
    
    parser.add_argument('path', 
                       help='Path to H5 file or directory containing H5 files')
    parser.add_argument('--cameras', action='store_true',
                       help='Convert cam1, cam2, cam3 files based on the provided cam1 path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"Error: Path does not exist: {args.path}")
        sys.exit(1)
    
    try:
        converted_files = []
        
        if args.cameras:
            # Convert multiple camera files based on cam1 path
            if 'cam1_' not in args.path:
                print("Error: For --cameras option, provide a cam1_*.h5 file path")
                sys.exit(1)
            converted_files = convert_specific_cameras(args.path)
            
        elif os.path.isfile(args.path):
            # Convert single file
            if args.path.lower().endswith('.h5'):
                csv_file = convert_h5_to_csv(args.path)
                if csv_file:
                    converted_files.append(csv_file)
            else:
                print("Error: File must have .h5 extension")
                sys.exit(1)
                
        elif os.path.isdir(args.path):
            # Convert all H5 files in directory
            h5_files = glob.glob(os.path.join(args.path, '*.h5'))
            if not h5_files:
                print(f"No H5 files found in directory: {args.path}")
                sys.exit(1)
            h5_files.sort()  # Sort for consistent ordering
            converted_files = convert_multiple_files(h5_files)
        
        # Print summary
        print("\n=== Conversion Summary ===")
        print(f"Successfully converted {len(converted_files)} files:")
        for csv_file in converted_files:
            print(f"  {csv_file}")
            
        if converted_files:
            print("\nAll CSV files are saved in the same directory as the original H5 files.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()