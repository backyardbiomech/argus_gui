#!/usr/bin/env python

from __future__ import absolute_import

from .version import __version__

# Export public API
__all__ = ['__version__', 'resources', 'ArgusColors', 'FrameFinder', 'ClickerProject', 'MainWindow']

# load submodules conditionally to avoid import errors during module discovery
try:
    from .colors import ArgusColors
    from .frameFinderPyglet import FrameFinder
    from .logger import *
    from .output import *
    from .patterns import *
    from .sbaDriver import *
    from .sync import *
    from .tools import *
    from .triangulate import *
    from .undistort import *
    # Import graphers and Argus last as they depend on PySide6
    from .graphers import *
    from .Argus import MainWindow, ClickerProject
except ImportError as e:
    # If dependencies aren't available, skip the imports
    # This prevents module discovery warnings
    print(f"Warning: Could not import some argus_gui modules: {e}")
    # Re-raise if this is being imported by scripts that need these classes
    import traceback
    traceback.print_exc()

# Resources are always available
from . import resources
