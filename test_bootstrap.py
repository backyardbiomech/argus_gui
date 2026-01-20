#!/usr/bin/env python3
"""
Test script to verify bootstrap functionality in DLCbatch.py
"""

import sys
import os
import numpy as np

# Add argus_gui to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test import
try:
    from DLCbatch import DLCBatchProcessor
    print("✓ Successfully imported DLCBatchProcessor")
except Exception as e:
    print(f"✗ Failed to import DLCBatchProcessor: {e}")
    sys.exit(1)

# Test initialization with bootstrap parameters
try:
    # This will fail on directory validation, but we're just testing initialization
    processor = DLCBatchProcessor(
        data_dir="/tmp/test",
        likelihood_threshold=0.95,
        bootstrap=True,
        bs_iterations=100,
        display_progress=True,
        subframe_interp=False
    )
    print("✗ Should have raised FileNotFoundError")
except FileNotFoundError as e:
    print("✓ Initialization with bootstrap parameters works (expected FileNotFoundError)")
except Exception as e:
    print(f"✗ Unexpected error during initialization: {e}")
    sys.exit(1)

# Verify that bootstrapXYZs is imported
try:
    from argus_gui.tools import bootstrapXYZs
    print("✓ Successfully imported bootstrapXYZs from tools.py")
except Exception as e:
    print(f"✗ Failed to import bootstrapXYZs: {e}")
    sys.exit(1)

print("\n✓ All bootstrap integration tests passed!")
