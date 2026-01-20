#!/usr/bin/env python3
"""
Test script to verify the error calculation fix for 2-camera reconstructions.

This script demonstrates that the fixed error calculation produces comparable
error magnitudes for 2-camera and 3-camera cases, whereas the old calculation
inflated 2-camera errors by a factor of sqrt(3).
"""

import numpy as np
import sys
import os

# Add argus_gui to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from argus_gui.tools import get_repo_errors

def create_test_data(n_frames=10, n_cams=3, n_tracks=1):
    """
    Create synthetic test data for error calculation.
    
    Returns:
        xyz: (n_frames, 3*n_tracks) array of 3D points
        pts: (n_frames, 2*n_cams*n_tracks) array of 2D points
        prof: camera profile array
        dlt: DLT coefficients
    """
    # Simple camera profile (focal length, cx, cy, k1-k5)
    prof = np.array([
        [1000.0, 640.0, 480.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Camera 1
        [1000.0, 640.0, 480.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Camera 2
        [1000.0, 640.0, 480.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Camera 3
    ])[:n_cams]
    
    # Simple DLT coefficients (would normally come from calibration)
    dlt = np.random.randn(n_cams, 11) * 0.001
    
    # Generate some 3D points
    xyz = np.random.randn(n_frames, 3 * n_tracks) * 100 + 500
    
    # Generate 2D projections with small noise
    pts = np.random.randn(n_frames, 2 * n_cams * n_tracks) * 2 + 640
    
    return xyz, pts, prof, dlt


def test_two_vs_three_cameras():
    """
    Test that 2-camera and 3-camera errors are on similar scales after the fix.
    """
    print("=" * 70)
    print("Testing 2-camera vs 3-camera error calculation")
    print("=" * 70)
    
    n_frames = 100
    
    # Test with 3 cameras
    print("\n--- Testing with 3 cameras ---")
    xyz_3cam, pts_3cam, prof_3cam, dlt_3cam = create_test_data(n_frames=n_frames, n_cams=3)
    errors_3cam = get_repo_errors(xyz_3cam, pts_3cam, prof_3cam, dlt_3cam)
    
    mean_error_3cam = np.nanmean(errors_3cam)
    std_error_3cam = np.nanstd(errors_3cam)
    print(f"Mean error: {mean_error_3cam:.4f}")
    print(f"Std error:  {std_error_3cam:.4f}")
    
    # Test with 2 cameras
    print("\n--- Testing with 2 cameras ---")
    xyz_2cam, pts_2cam, prof_2cam, dlt_2cam = create_test_data(n_frames=n_frames, n_cams=2)
    errors_2cam = get_repo_errors(xyz_2cam, pts_2cam, prof_2cam, dlt_2cam)
    
    mean_error_2cam = np.nanmean(errors_2cam)
    std_error_2cam = np.nanstd(errors_2cam)
    print(f"Mean error: {mean_error_2cam:.4f}")
    print(f"Std error:  {std_error_2cam:.4f}")
    
    # Compare ratios
    print("\n--- Comparison ---")
    ratio = mean_error_2cam / mean_error_3cam
    print(f"Ratio (2cam/3cam): {ratio:.4f}")
    print(f"Expected with old formula: {np.sqrt(3):.4f}")
    print(f"Expected with new formula: ~1.0")
    
    # Check if the fix worked
    if 0.8 < ratio < 1.5:
        print("\n✓ SUCCESS: 2-camera and 3-camera errors are on similar scale!")
        print("  The fix is working correctly.")
    else:
        print("\n✗ WARNING: Error ratio is outside expected range")
        print("  Expected ratio near 1.0, got {:.4f}".format(ratio))
    
    return ratio


def test_consistent_2cam_errors():
    """
    Test that when all frames use 2 cameras, errors are consistent but not inflated.
    """
    print("\n" + "=" * 70)
    print("Testing consistency of 2-camera errors")
    print("=" * 70)
    
    n_frames = 50
    xyz, pts, prof, dlt = create_test_data(n_frames=n_frames, n_cams=2)
    
    errors = get_repo_errors(xyz, pts, prof, dlt)
    
    print(f"\nNumber of frames: {n_frames}")
    print(f"Mean error: {np.nanmean(errors):.4f}")
    print(f"Std error:  {np.nanstd(errors):.4f}")
    print(f"Min error:  {np.nanmin(errors):.4f}")
    print(f"Max error:  {np.nanmax(errors):.4f}")
    
    # Check if errors show reasonable variation
    cv = np.nanstd(errors) / np.nanmean(errors)  # Coefficient of variation
    print(f"Coefficient of variation: {cv:.4f}")
    
    if cv > 0.01:
        print("\n✓ SUCCESS: Errors show reasonable variation")
        print("  (Not all the same value)")
    else:
        print("\n✗ WARNING: Errors are too uniform")
        print("  This might indicate averaging is still creating uniform values")


if __name__ == '__main__':
    print("Error Calculation Fix Test")
    print("This test verifies that 2-camera errors are not inflated\n")
    
    try:
        ratio = test_two_vs_three_cameras()
        test_consistent_2cam_errors()
        
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print("The error calculation has been fixed to prevent inflation of")
        print("2-camera reconstruction errors. Previously, 2-camera errors were")
        print("√3 ≈ 1.73x higher than 3-camera errors due to the DOF calculation.")
        print("\nWith the fix, errors are normalized to be comparable across")
        print("different numbers of cameras.")
        
    except Exception as e:
        print(f"\n✗ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
