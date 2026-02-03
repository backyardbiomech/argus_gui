#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for DLC labeling functionality
"""

import os
import sys
import pandas as pd
import numpy as np

# Add the argus_gui directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'argus_gui'))

from argus_gui.DLCclicker import DLCLabelFolder

def test_dlc_folder():
    """Test loading a DLC folder"""
    # Path to test DLC folder
    test_folder = r'c:\Users\jacksonbe3\Documents\repos\argus_gui\dlctest\DLCproject\labeled-data\cam1_08-Sep-25_21-34-16'
    
    print("=" * 60)
    print("Testing DLC Folder Loading")
    print("=" * 60)
    
    try:
        # Create DLC folder handler
        print(f"\nLoading folder: {test_folder}")
        dlc_folder = DLCLabelFolder(test_folder)
        
        print(f"✓ Folder loaded successfully")
        print(f"  - Video name: {dlc_folder.video_name}")
        print(f"  - CollectedData path: {dlc_folder.collected_data_path}")
        print(f"  - Machine labels files: {len(dlc_folder.machine_labels_paths)}")
        
        # Load the data
        print("\nLoading data from h5 files...")
        dlc_folder.load_data()
        
        print(f"✓ Data loaded successfully")
        print(f"  - Scorer: {dlc_folder.scorer}")
        print(f"  - Multi-animal: {dlc_folder.is_multi_animal}")
        print(f"  - Individuals: {dlc_folder.individuals}")
        print(f"  - Bodyparts: {dlc_folder.bodyparts}")
        print(f"  - Label frames: {dlc_folder.label_frames}")
        
        # Get tracks
        tracks = dlc_folder.get_tracks_for_argus()
        print(f"\n✓ Generated {len(tracks)} tracks for Argus:")
        for i, track in enumerate(tracks[:5]):  # Show first 5
            print(f"    {i+1}. {track}")
        if len(tracks) > 5:
            print(f"    ... and {len(tracks) - 5} more")
        
        # Find video
        print("\nLooking for video file...")
        video_path = dlc_folder.find_video(prompt_user=False)
        if video_path:
            print(f"✓ Video found: {video_path}")
        else:
            print("⚠ Video not found automatically (would prompt user)")
        
        # Test conversion to Argus format
        print("\nTesting conversion to Argus format...")
        # Use a test height
        test_height = 720
        argus_data = dlc_folder.convert_to_argus_format(test_height)
        print(f"✓ Converted to Argus format")
        print(f"  - Data shape: {argus_data.shape}")
        print(f"  - Non-zero entries: {np.count_nonzero(argus_data)}")
        
        # Check some specific frames
        print("\nSample data for first label frame:")
        if dlc_folder.label_frames:
            first_frame = dlc_folder.label_frames[0]
            print(f"  Frame {first_frame}:")
            for i in range(min(3, len(tracks))):
                x = argus_data[first_frame, i*2]
                y = argus_data[first_frame, i*2+1]
                print(f"    {tracks[i]}: ({x:.2f}, {y:.2f})")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dlc_folder()
    sys.exit(0 if success else 1)
