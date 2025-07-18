#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

# Fix Qt platform plugin path for Windows - MUST be done before any Qt imports
import sys
import os
if sys.platform.startswith('win'):
    try:
        import PySide6
        pyside6_path = os.path.dirname(PySide6.__file__)
        
        # Try multiple possible plugin directory structures
        possible_plugin_paths = [
            os.path.join(pyside6_path, 'Qt6', 'plugins'),
            os.path.join(pyside6_path, 'Qt', 'plugins'),
            os.path.join(pyside6_path, 'plugins'),
            os.path.join(pyside6_path, '..', 'Library', 'plugins'),  # conda structure
            os.path.join(pyside6_path, '..', 'Lib', 'site-packages', 'PySide6', 'Qt6', 'plugins')  # pip structure
        ]
        
        qt_plugin_path = None
        for path in possible_plugin_paths:
            if os.path.exists(path):
                qt_plugin_path = path
                break
        
        if qt_plugin_path:
            os.environ['QT_PLUGIN_PATH'] = qt_plugin_path
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
            
        # Additional Qt environment variables for Windows
        os.environ['QT_QPA_PLATFORM'] = 'windows'
        
    except Exception as e:
        pass  # Silent fallback if Qt plugin setup fails

import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg for automatic Qt version detection, compatible with PySide6

# commented for pyqtgraph
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d
from PySide6.QtWidgets import QVBoxLayout, QWidget
from PySide6.QtGui import QFont
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.items.GLTextItem import GLTextItem
# from pyqtgraph import GraphicsLayoutWidget, LabelItem, PlotWidget
try:
    from moviepy.config import get_setting
except ImportError:
    # Fallback for newer moviepy versions
    def get_setting(setting_name):
        if setting_name == "FFMPEG_BINARY":
            return "ffmpeg"  # Default to system ffmpeg
        return None

# import wandOutputter
from .output import *
import pandas
import sys

from numpy import *
import numpy as np
import scipy.signal
import scipy.io.wavfile
import scipy.spatial
import scipy
import sys
import os.path
# import matplotlib.pyplot as plt
try:
    from moviepy.config import get_setting
except ImportError:
    # Fallback for newer moviepy versions
    def get_setting(setting_name):
        if setting_name == "FFMPEG_BINARY":
            return "ffmpeg"  # Default to system ffmpeg
        return None
# import matplotlib.patches as mpatches
import random
from .colors import *
import subprocess
from .tools import *

# import pickle # debug


# class that plots n wave files for the user to choose a time interval
class Shower():
    def __init__(self, tmpName, files, out):
        # temporary directory location
        self.tmpName = tmpName
        # actual wave file names
        self.files = files
        # expected temporary file names
        self.out = out

    def show(self):
        # Colors for plotting
        colors = ArgusColors().getMatplotlibColors()
        # Shuffle to make things interesting
        random.shuffle(colors)
        for k in range(len(self.files)):
            # If we don't find a wav with the same name as the file, rip one using moviepy's ffmpeg binary
            if not os.path.isfile(self.tmpName + '/' + self.out[k]):
                print('Ripping audio from file number ' + str(k + 1) + ' of ' + str(len(self.files)))
                sys.stdout.flush()
                cmd = [
                    get_setting("FFMPEG_BINARY"),
                    '-loglevel', 'panic',
                    '-hide_banner',
                    '-i', self.files[k],
                    '-ac', '1',
                    '-codec', 'pcm_s16le',
                    self.tmpName + '/' + self.out[k]
                ]
                subprocess.call(cmd)
            else:
                print('Found audio from file number ' + str(k + 1) + ' of ' + str(len(self.files)))
                sys.stdout.flush()
        # Put the full signals in a list
        signals_ = list()
        print('Reading waves and displaying...')
        sys.stdout.flush()
        for k in range(len(self.files)):
            rate, signal = scipy.io.wavfile.read(self.tmpName + '/' + self.out[k])
            signals_.append(signal)
        # Make a new list of signals but only using ever 100th sample
        signals = list()
        for k in range(len(signals_)):
            t = list()
            a = 0
            while a < len(signals_[k]):
                t.append(signals_[k][a])
                a += 100
            signals.append(np.asarray(t))
            
        # Platform-specific rendering fixes - MUST be set before QApplication creation
        # Import Qt classes needed for all platforms
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        
        # Windows-specific PyQtGraph configuration - BEFORE QApplication
        if sys.platform.startswith('win'):
            # Force software rendering to avoid GPU driver issues on Windows
            pg.setConfigOption('useOpenGL', False)
            pg.setConfigOption('antialias', False)  # Disable antialiasing on Windows
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            pg.setConfigOption('crashWarning', True)
            
            # Additional Windows-specific configurations
            pg.setConfigOption('leftButtonPan', False)  # Disable problematic pan behavior
            pg.setConfigOption('enableExperimental', False)  # Disable experimental features
            
            # Set Qt application attributes BEFORE creating QApplication
            try:
                # Set attributes before QApplication creation
                if hasattr(Qt, 'AA_UseDesktopOpenGL'):
                    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL, False)
                if hasattr(Qt, 'AA_UseSoftwareOpenGL'):
                    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, True)
                if hasattr(Qt, 'AA_ShareOpenGLContexts'):
                    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, False)
                if hasattr(Qt, 'AA_DisableShaderDiskCache'):
                    QApplication.setAttribute(Qt.AA_DisableShaderDiskCache, True)
            except Exception:
                pass  # Silent fallback
        else:
            # Conservative configuration for Mac/Linux - Disable OpenGL to avoid hanging
            pg.setConfigOption('useOpenGL', False)  # Disable OpenGL to prevent Mac hanging
            pg.setConfigOption('antialias', True)
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            pg.setConfigOption('enableExperimental', False)
            
            # Mac-specific Qt application attributes to prevent hanging
            try:
                if hasattr(Qt, 'AA_DisableWindowContextHelpButton'):
                    QApplication.setAttribute(Qt.AA_DisableWindowContextHelpButton, True)
                if hasattr(Qt, 'AA_DontCreateNativeWidgetSiblings'):
                    QApplication.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings, True)
            except Exception:
                pass  # Silent fallback
        
        # NOW create QApplication
        app = QApplication([])
        
        # Post-application Windows setup
        if sys.platform.startswith('win'):
            # Force immediate style refresh on Windows
            try:
                app.setStyle('Fusion')  # Use cross-platform Fusion style
            except Exception:
                pass  # Silent fallback
        
        # Platform-specific widget creation - SIMPLIFIED Windows approach
        if sys.platform.startswith('win'):
            try:
                # For Windows, try the most basic approach possible
                win = pg.PlotWidget()
                win.resize(1000, 600)
                win.setWindowTitle('Audio Streams - Windows Basic')
                win.setBackground('white')  # Use string instead of 'w'
                plot = win.getPlotItem()
                
                # Force basic Windows-compatible settings
                plot.setMenuEnabled(enableMenu=False)
                plot.setMouseEnabled(x=True, y=True)
                plot.enableAutoRange('xy', False)
                plot.setAutoVisible(y=False)
                
            except Exception:
                win = pg.GraphicsLayoutWidget()
                win.resize(1000, 600)
                win.setWindowTitle('Audio Streams - Windows Minimal')
                win.setBackground('white')
                plot = win.addPlot()
                plot.setMenuEnabled(enableMenu=False)
        else:
            try:
                # Mac/Linux: Optimized setup with hardware acceleration
                win = pg.PlotWidget(title="Audio Streams")
                win.resize(1000, 600)
                win.setWindowTitle('Audio Streams - PlotWidget')
                win.setBackground('w')
                plot = win.getPlotItem()
                
                # Enable better mouse interaction for Mac
                plot.setMouseEnabled(x=True, y=True)
                plot.setMenuEnabled(enableMenu=True)  # Enable context menu for Mac
                
                # Configure ViewBox for better responsiveness on Mac (software rendering)
                view_box = plot.getViewBox()
                if view_box:
                    # Use conservative settings to prevent hanging
                    view_box.setMouseEnabled(x=True, y=True)
                    view_box.setDefaultPadding(0.05)  # Slightly more padding for stability
                    # Use standard wheel zoom without custom scaling to avoid issues
                    view_box.aspectLocked = False  # Allow independent x/y zoom
                    
            except Exception:
                win = pg.GraphicsLayoutWidget(show=True, title="Audio Streams")
                win.resize(1000, 600)
                win.setWindowTitle('Audio Streams - GraphicsLayoutWidget')
                win.setBackground('w')
                plot = win.addPlot()
                
                # Configure for Mac stability (software rendering)
                plot.setMouseEnabled(x=True, y=True)
                view_box = plot.getViewBox()
                if view_box:
                    view_box.setMouseEnabled(x=True, y=True)
                    view_box.setDefaultPadding(0.05)  # Conservative padding for stability
                    # Use standard wheel zoom without custom scaling
                    view_box.aspectLocked = False  # Allow independent x/y zoom
        
        plot.showGrid(x=True, y=True)
        plot.setLabel('bottom', 'Minutes')
        plot.getAxis('left').setTicks([])
        plot.getAxis('left').setLabel('')
        
        legend = plot.addLegend(offset=(70, 30))
        
        # We'll set the range AFTER we calculate the actual data bounds
        # For now, just disable auto-range
        plot.disableAutoRange()
        
        # Calculate signal ranges and vertical offsets to avoid overflow
        signal_ranges = []
        for signal in signals:
            # Handle edge cases with NaN or infinite values
            finite_signal = signal[np.isfinite(signal)]
            if len(finite_signal) > 0:
                # Convert to float64 to prevent overflow in arithmetic operations
                signal_min = float(np.min(finite_signal))
                signal_max = float(np.max(finite_signal))
                signal_range = signal_max - signal_min
                signal_ranges.append(signal_range)
            else:
                signal_ranges.append(1.0)  # fallback for all NaN/inf signals
        
        # Use float64 to avoid overflow and calculate reasonable vertical separation
        if signal_ranges:
            max_range = signal_ranges[0]
            for sr in signal_ranges[1:]:
                if sr > max_range:
                    max_range = sr
        else:
            max_range = 1.0
        # Clamp the max_range to prevent extremely large offsets
        if max_range > 1e6:
            max_range = 1e6
        
        # Use a more reasonable vertical separation - normalize to a reasonable scale
        # Rather than using the full signal range, use a smaller multiplier for better visualization
        if max_range > 10000:
            # For large audio ranges, use a larger multiplier for better separation
            vertical_offset = float(max_range * 0.8)  # 80% separation for large ranges
        else:
            vertical_offset = float(max_range * 2.5)  # 250% separation for smaller ranges
        
        # CRITICAL: Calculate plot ranges BEFORE plotting anything
        # Calculate the overall coordinate bounds for all signals first
        all_t_values = []
        all_y_values = []
        for k in range(len(signals)):
            t = np.linspace(0, len(signals_[k]) / 48000., num=len(signals[k])) / 60.
            finite_signal = signals[k][np.isfinite(signals[k])]
            if len(finite_signal) > 0:
                signal_min = float(np.min(finite_signal))
                signal_max = float(np.max(finite_signal))
                signal_center = (signal_max + signal_min) / 2.0
            else:
                signal_center = 0.0
            y_offset = k * vertical_offset
            adjusted_signal = signals[k] - signal_center + y_offset
            
            all_t_values.extend([np.min(t), np.max(t)])
            all_y_values.extend([np.min(adjusted_signal), np.max(adjusted_signal)])
        
        # Set plot ranges BEFORE creating any plot items
        if all_t_values and all_y_values:
            x_min, x_max = min(all_t_values), max(all_t_values)
            y_min, y_max = min(all_y_values), max(all_y_values)
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            plot_x_min = x_min - x_padding
            plot_x_max = x_max + x_padding  
            plot_y_min = y_min - y_padding
            plot_y_max = y_max + y_padding
            
            plot.setXRange(plot_x_min, plot_x_max, padding=0)
            plot.setYRange(plot_y_min, plot_y_max, padding=0)
        
        for k in range(len(signals)):
            # Use bright, contrasting colors for better visibility
            bright_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = bright_colors[k % len(bright_colors)]
            t = np.linspace(0, len(signals_[k]) / 48000., num=len(signals[k])) / 60.
            # Use the signal's offset from its center, then apply vertical separation
            finite_signal = signals[k][np.isfinite(signals[k])]
            if len(finite_signal) > 0:
                # Convert to float64 to prevent overflow in arithmetic operations
                signal_min = float(np.min(finite_signal))
                signal_max = float(np.max(finite_signal))
                signal_center = (signal_max + signal_min) / 2.0
            else:
                signal_center = 0.0  # fallback for all NaN/inf signals
            y_offset = k * vertical_offset
            # Center the signal around zero, then add vertical offset for separation
            adjusted_signal = signals[k] - signal_center + y_offset
            
        # CRITICAL WINDOWS FIX: Use matplotlib as fallback if PyQtGraph fails
        matplotlib_plot_created = False
        if sys.platform.startswith('win'):
            # Try matplotlib as a backup for Windows - create ONE plot for ALL signals
            try:
                import matplotlib.pyplot as plt
                
                # Create matplotlib plot ONCE for all signals
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.set_title('Audio Streams - Matplotlib Windows Fallback')
                ax.set_xlabel('Minutes')
                ax.grid(True)
                
                # Set the same ranges as PyQtGraph
                ax.set_xlim(plot_x_min, plot_x_max)
                ax.set_ylim(plot_y_min, plot_y_max)
                
                matplotlib_plot_created = True
                
            except Exception:
                pass  # Silent fallback
        
        for k in range(len(signals)):
            # Use bright, contrasting colors for better visibility
            bright_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = bright_colors[k % len(bright_colors)]
            t = np.linspace(0, len(signals_[k]) / 48000., num=len(signals[k])) / 60.
            # Use the signal's offset from its center, then apply vertical separation
            finite_signal = signals[k][np.isfinite(signals[k])]
            if len(finite_signal) > 0:
                # Convert to float64 to prevent overflow in arithmetic operations
                signal_min = float(np.min(finite_signal))
                signal_max = float(np.max(finite_signal))
                signal_center = (signal_max + signal_min) / 2.0
            else:
                signal_center = 0.0  # fallback for all NaN/inf signals
            y_offset = k * vertical_offset
            # Center the signal around zero, then add vertical offset for separation
            adjusted_signal = signals[k] - signal_center + y_offset
            
            # Windows matplotlib plotting - add each signal to the SAME plot
            if matplotlib_plot_created and sys.platform.startswith('win'):
                try:
                    # Plot with matplotlib using the same data
                    bright_colors_mpl = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
                    color_mpl = bright_colors_mpl[k % len(bright_colors_mpl)]
                    
                    # Plot the audio waveform with matplotlib on the existing axes
                    ax.plot(t, adjusted_signal, color=color_mpl, linewidth=1, label=self.files[k].split('/')[-1])
                    
                except Exception:
                    matplotlib_plot_created = False  # Fall back to PyQtGraph for remaining signals
            
            # Original PyQtGraph plotting (for non-Windows or if matplotlib fails)
            if not matplotlib_plot_created or not sys.platform.startswith('win'):
                # Optimized plotting approach - thin lines for better Mac performance
                if sys.platform.startswith('win'):
                    # Windows: Use thicker, simpler pen
                    pen = pg.mkPen(color=color, width=10)
                else:
                    # Mac/Linux: Use thin lines for better performance and appearance
                    pen = pg.mkPen(color=color, width=1)
                
                # Plot the audio signal
                curve = plot.plot(t, adjusted_signal, pen=pen)
                
                # Force updates for Windows
                if sys.platform.startswith('win'):
                    plot.update()
                    curve.update() 
                    win.repaint()
                    app.processEvents()
                
                legend.addItem(curve, self.files[k].split('/')[-1])
        
        # Show matplotlib plot if it was created (after all signals are added)
        if matplotlib_plot_created and sys.platform.startswith('win'):
            try:
                # Add legend without test triangle
                ax.legend()
                plt.show()
                
                # Since matplotlib worked and is displayed, we can skip the PyQtGraph display
                return  # Exit early since we have a working plot
                
            except Exception:
                pass  # Continue with PyQtGraph display
        # Platform-specific window management and final display
        if sys.platform.startswith('win'):
            # Critical Windows fix: Force ViewBox to recalculate bounds and redraw
            view_box = plot.getViewBox()
            if view_box:
                # Force recalculation of data bounds
                view_box.updateAutoRange()
                view_box.enableAutoRange()
                view_box.disableAutoRange()  # Reset to our manual range
                
                # Set our range again to make sure it sticks
                if all_t_values and all_y_values:
                    view_box.setRange(xRange=[plot_x_min, plot_x_max], yRange=[plot_y_min, plot_y_max], padding=0)
                
                # Force ViewBox scene to update
                if hasattr(view_box, 'scene'):
                    view_box.scene().update()
            
            # Force immediate processing of Qt events on Windows
            for _ in range(5):
                app.processEvents()
            
            # Show window with Windows-specific settings
            win.show()
            win.raise_()
            win.activateWindow()
            
            # Force window to be topmost temporarily
            try:
                from PySide6.QtCore import Qt
                win.setWindowFlags(win.windowFlags() | Qt.WindowStaysOnTopHint)
                win.show()
                app.processEvents()
                # Remove topmost flag after a moment
                win.setWindowFlags(win.windowFlags() & ~Qt.WindowStaysOnTopHint)
                win.show()
            except Exception:
                pass  # Silent fallback
        else:
            # Optimized display for Mac/Linux with hardware acceleration
            win.show()
            win.raise_()
            win.activateWindow()
            
            # Force OpenGL context refresh for better performance on Mac
            if hasattr(win, 'getViewBox'):
                view_box = win.getViewBox() if hasattr(win, 'getViewBox') else plot.getViewBox()
                if view_box and hasattr(view_box, 'update'):
                    view_box.update()
        
        # Final event processing
        app.processEvents()
        
        # Platform-specific event loop handling
        if sys.platform.startswith('win'):
            # Windows-specific event loop with timeout to prevent hanging
            try:
                # Add a timer to prevent infinite hanging
                from PySide6.QtCore import QTimer
                
                # Create a timer that will allow the app to be responsive
                timer = QTimer()
                timer.timeout.connect(lambda: None)  # Empty callback
                timer.start(100)  # Process events every 100ms
                
                # Set a flag to allow graceful shutdown
                win.closeEvent = lambda event: app.quit()
                
                # Start the application event loop with timeout handling
                app.exec_()
            except Exception:
                try:
                    app.quit()
                except:
                    pass
        else:
            # Standard event loop for Mac/Linux
            app.exec_()
            
        # Cleanup
        signals_ = None
        # a = 0
        # patches = list()
        # width = 35
        # height = 3 * len(signals)
        # plt.figure(figsize=(width, height))
        # frame1 = plt.gca()
        # frame1.axes.get_yaxis().set_visible(False)
        # # Make a plot with colors chosen by circularly pulling from the colors vector
        # for k in range(len(signals)):
        #     color = colors[k % len(colors)]
        #     patches.append(mpatches.Patch(color=color, label=self.files[k].split('/')[-1]))
        #     t = np.linspace(0, len(signals_[k]) / 48000., num=len(signals[k])) / 60.
        #     plt.plot(t, signals[k] + float(a), color=color)
        #     a += np.nanmax(signals[k]) * 2
        # plt.legend(handles=patches)
        # plt.title('Audio Streams')
        # plt.xlabel('Minutes')
        # signals_ = None
        # plt.show()


# rigid_transform_3D
# Returns the estimated translation and rotation matrix for a rigid transform from one set of points to another.
# used here to transform points based on specified axis directions
#
# Uses a nifty SVD method
# https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
#
# Input: expects Nx3 matrices of points in A and B of matched N
# Returns: R,t
#   R = 3x3 rotation matrix
#   t = 3x1 column vector
def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)

    # center the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA).dot(BB)

    U, S, Vt = linalg.svd(H)

    R = Vt.T.dot(U.T)

    # special reflection case
    if linalg.det(R) < 0:
        print('Reflection detected - likely due to an underlying left-handed coordinate system')
        # do nothing (commented the below lines out on 2020-05-26)
        #Vt[2, :] *= -1
        #R = Vt.T.dot(U.T)
    t = -R.dot(centroid_A.T) + centroid_B.T

    return R, t


def rotation(a, b):
    # make unit vectors
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    # find cross product and cross product matrix
    v = np.cross(a, b)
    v_x = np.zeros((3, 3))
    v_x[1, 0] = v[2]
    v_x[0, 1] = -v[2]
    v_x[2, 0] = -v[1]
    v_x[0, 2] = v[1]
    v_x[2, 1] = v[0]
    v_x[1, 2] = -v[0]
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    R = np.eye(3) + v_x + np.linalg.matrix_power(v_x, 2) * ((1. - c) / s ** 2)
    return R

# calculate camera xyz position from DLT coefficients
def DLTtoCamXYZ(dlts):
    camXYZ = []
    for i in range(len(dlts)):
        m1=np.hstack([dlts[i,0:3],dlts[i,4:7],dlts[i,8:11]]).T
        m2=np.vstack([-dlts[i,3],-dlts[i,7],-1])
        camXYZ.append(np.dot(np.linalg.inv(m1),m2))
        
    camXYZa = np.array(camXYZ)
    return camXYZa

# takes unpaired and paired points along with other information about the scene, and manipulates the data for outputting and graphing in 3D
class wandGrapher(QWidget):
    def __init__(self, my_app, key, nppts, nuppts, scale, ref, indices, ncams, npframes, nupframes=None, name=None, temp=None,
                 display=True, uvs=None, nRef=0, order=None, report=True, cams=None, reference_type='Axis points', recording_frequency=100):
        super().__init__()
        self.my_app = my_app
        layout = QVBoxLayout(self)
        # Create a GL View widget for displaying 3D data with grid and axes (data will come later)
        self.view = gl.GLViewWidget()
        self.view.setWindowTitle('3D Graph')
        self.view.setCameraPosition(distance=20)
        # Create grid items for better visualization
        grid = gl.GLGridItem()
        grid.scale(2, 2, 1)
        self.view.addItem(grid)
        
        # Add x, y, z axes lines
        axis_length = 10
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_length, 0, 0]]), color=(1, 0, 0, 1), width=2)  # Red line for x-axis
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]]), color=(0, 1, 0, 1), width=2)  # Green line for y-axis
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]]), color=(0, 0, 1, 1), width=2)  # Blue line for z-axis
        self.view.addItem(x_axis)
        self.view.addItem(y_axis)
        self.view.addItem(z_axis)
                
        self.nppts = nppts
        self.nuppts = nuppts
        self.scale = scale
        self.ref = ref
        self.indices = indices
        self.ncams = ncams
        self.npframes = npframes
        self.nupframes = nupframes
        self.name = name
        self.temp = temp
        self.display = display
        self.uvs = uvs
        self.nRef = nRef
        self.order = order
        self.key = key
        self.report = report
        cams = cams
        self.reference_type = reference_type
        self.recording_frequency = recording_frequency

        self.cams = []

        for c in cams:
            self.cams.append([c[u] for u in [0, 1, 2, 5, 6, 7, 8, 9]])

    # finds the average distance between two point sets
    # used to find the scale defined by the wand
    def averDist(self, pts1, pts2):
        dist = list()
        for k in range(pts1.shape[0]):
            dist.append(np.linalg.norm(pts1[k] - pts2[k]))
        return np.nanmean(np.asarray(dist)), np.nanstd(np.asarray(dist))

    # transform the points with the rigid transform function
    def transform(self, xyzs, ref):
        
        # Subtract the origin from the points and reference
        t = ref[0]
        ret = xyzs - np.tile(t, (xyzs.shape[0], 1))
        ref = ref - ref[0]
            
        # process Axis points
        # if we only got one reference point
        if ref.shape[0] == 1 and self.reference_type == 'Axis points':
            print('Using 1-point (origin) reference axes')
            # we actually already did this above since it's the starting point
            # for all alignment operations
            
        # If we only have 2 reference points: origin, +Z (plumb line):
        elif ref.shape[0] == 2 and self.reference_type == 'Axis points':
            print('Using 2-point (origin,+Z) reference axes')    
            a = (ref[1] - ref[0]) * (1. / np.linalg.norm(ref[1] - ref[0]))

            # Get the current z-axis and the wanted z-axis
            pts1 = np.asarray([[0., 0., 1.]])
            pts2 = np.asarray([a])

            # Get the transform from one to the other
            R = rotation(pts2[0], pts1[0])

            # Perform the transform
            for k in range(ret.shape[0]):
                ret[k] = R.dot(ret[k].T).T
            
        # If we have origin,+x,+y axes:
        elif ref.shape[0] == 3 and self.reference_type == 'Axis points':
            print('Using 3-point (origin,+x,+y) reference axes')        
            A = ref

            # define an Nx4 matrix containing origin (same in both), a point on the x axis, a point on the y axis, and a point on z
            A = np.vstack((A, np.cross(A[1], A[2]) / np.linalg.norm(np.cross(A[1], A[2]))))

            # define the same points in our coordinate system
            B = np.zeros((4, 3))
            B[1] = np.array([np.linalg.norm(A[1]), 0., 0.])
            B[2] = np.array([0., np.linalg.norm(A[2]), 0.])
            B[3] = np.array([0., 0., 1.])

            # find rotation and translation, translation ~ 0 by definition
            R, t = rigid_transform_3D(A, B)

            # rotate
            ret = R.dot(ret.T).T
        
        # If we have origin,+x,+y,+z axes:
        elif ref.shape[0] == 4 and self.reference_type == 'Axis points':
            print('Using 4-point (origin,+x,+y,+z) reference axes')
            A = ref

            # define the same points in our coordinate system
            B = np.zeros((4, 3))
            B[1] = np.array([np.linalg.norm(A[1]), 0., 0.])
            B[2] = np.array([0., np.linalg.norm(A[2]), 0.])
            B[3] = np.array([0., 0., np.linalg.norm(A[3])])

            # find rotation and translation, translation ~ 0 by definition
            R, t = rigid_transform_3D(A, B)

            # rotate
            ret = R.dot(ret.T).T
            
        # If we have a gravity reference
        elif self.reference_type == 'Gravity':
            print('Using gravity alignment, +Z will point anti-parallel to gravity')
            rfreq=float(self.recording_frequency)
            t=np.arange(ref.shape[0]) # integer timebase
            
            print(ref) # debug
            
            # perform a least-squares fit of a 2nd order polynomial to each
            # of x,y,z components of the reference, evaluate the polynomial
            # and get the acceleration
            acc=np.zeros(3)
            idx=np.where(np.isfinite(ref[:,0])) # can only send real data to polyfit
            for k in range(3):
                p=np.polyfit(t[idx[0]],ref[idx[0],k],2)
                pv=np.polyval(p,t)
                acc[k]=np.mean(np.diff(np.diff(pv)))*rfreq*rfreq
            
            # need a rotation to point acceleration at -1 (i.e. -Z is down)
            an=acc/np.linalg.norm(acc) # unit acceleration vector
            vv=np.array([0,0,-1]) # target vector
            rv=np.cross(an,vv) # axis for angle-axis rotation
            rv=rv/np.linalg.norm(rv) # unit axis
            ang=np.arccos(np.dot(an,vv)) # rotation magnitude
            r = scipy.spatial.transform.Rotation.from_rotvec(rv*ang) # compose angle-axis rotation 
            ret = r.apply(ret) # apply it
            
            # reporting
            pg=np.linalg.norm(acc)/9.81*100
            print('Gravity measured with {:.2f}% accuracy!'.format(pg))
            
        # If we have a reference plane
        elif self.reference_type == 'Plane':
            print('Aligning to horizontal plane reference points')

            avg = np.mean(ref.T, axis=1)
            centered = ref - avg  # mean centered plane
            
            # do a principal components analysis via SVD
            uu, ss, vh = np.linalg.svd(centered, full_matrices=True)
            vh=vh.T # numpy svd vector is the transpose of the MATLAB version
            #print('svd results')
            #print(vh)
            
            #print('plane xyz points pre-rotation')
            #print(centered)
            
            # check to see if vh is a rotation matrix
            if np.linalg.det(vh) == -1:
                #print('found det of -1')
                vh=-vh
                
            # test application of rotation to plane points
            #rTest = np.matmul(centered,vh)
            #print('Rotation test on plane points')
            #print(rTest)
            
            # apply to the whole set of input values
            centered = xyzs - avg   # center on center of reference points
            rCentered = np.matmul(centered,vh)
            
            # check to see if Z points are on average + or -
            # if they're negative, multiply in a 180 degree rotation about the X axis
            rca=np.mean(rCentered.T,axis=1)
            if rca[2]<0:
                #print('reversing the direction')
                r180 = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
                vh = np.matmul(vh,r180)
                rCentered = np.matmul(centered,vh)

            ret = rCentered

        return ret

    # makes two sets of isomorphic paired points that share the same frame
    def pairedIsomorphism(self, pts):
        # get the length on the indices vector to split the paired points apart
        n1 = len(self.indices['paired'][0])
        n2 = len(self.indices['paired'][1])
        # get the maximum frame number which we found a triangulatable point at
        a = np.max(self.indices['paired'][1])
        b = np.max(self.indices['paired'][0])
        # split the paired points
        p1 = pts[:n1, :]
        p2 = pts[n1:, :]
        b = np.min([a, b])
        set1 = list()
        set2 = list()
        k = 0
        # if two paired points were found at the same frame put them each in the set
        while k <= b:
            if k in self.indices['paired'][0] and k in self.indices['paired'][1]:
                set1.append(p1[self.indices['paired'][0].index(k)])
                set2.append(p2[self.indices['paired'][1].index(k)])
            k += 1
        # return the split paired points as well as the isomorphic sets for drawing wands
        return p1, p2, np.asarray(set1), np.asarray(set2)

    # writes the DLT coefficients to a CSV using pandas
    def outputDLT(self, cos, error):
        print('\nDLT Errors: ')
        print(error)
        sys.stdout.flush()
        dat = np.asarray(cos)
        dat = dat.T[0]

        if self.order is not None:
            dat = dat[:, list(self.order)]

        dat = pandas.DataFrame(dat)
        dat.to_csv(self.name + '-dlt-coefficients.csv', header=False, index=False)

    # solves an overdetermined linear system for DLT coefficients constructed via
    # these instructions: http://kwon3d.com/theory/dlt/dlt.html
    def getCoefficients(self, xyz, uv, cam_index=0):
        # delete rows which have nan

        indices = list(np.where(~np.isnan(uv[:, 0]) == True)[0])

        # print(xyz.shape)
        # print(uv.shape)

        A = np.zeros((len(indices) * 2, 11))

        # construct matrix based on uv pairs and xyz coordinates
        for k in range(len(indices)):
            A[2 * k, :3] = xyz[indices[k]]
            A[2 * k, 3] = 1
            A[2 * k, 8:] = xyz[indices[k]] * -uv[indices[k], 0]
            A[2 * k + 1, 4:7] = xyz[indices[k]]
            A[2 * k + 1, 7] = 1
            A[2 * k + 1, 8:] = xyz[indices[k]] * -uv[indices[k], 1]

        B = np.zeros((len(indices) * 2, 1))

        for k in range(len(indices)):
            B[2 * k] = uv[indices[k], 0]
            B[2 * k + 1] = uv[indices[k], 1]

        # solve using numpy's least squared algorithm
        # added rcond option 2020-05-26 in response to FutureWarning from numpy
        L = np.linalg.lstsq(A, B, rcond=None)[0]

        # reproject to calculate rmse
        reconsted = np.zeros((len(indices), 2))
        for k in range(len(indices)):
            u = (np.dot(L[:3].T, xyz[indices[k]]) + L[3]) / (np.dot(L[-3:].T, xyz[indices[k]]) + 1.)
            v = (np.dot(L[4:7].T, xyz[indices[k]]) + L[7]) / (np.dot(L[-3:].T, xyz[indices[k]]) + 1.)
            reconsted[k] = [u[0], v[0]]

        errors = list()
        dof = float(self.ncams * 2 - 3)

        _ = np.power(reconsted - uv[indices], 2)
        _ = _[:, 0] + _[:, 1]
        errors = np.sqrt(_)
        error = np.sum(errors)

        """
        error = 0
        for k in range(len(indices)):
            s = np.sqrt((reconsted[k,0] - uv[indices[k],0])**2 + (reconsted[k,1] - uv[indices[k],1])**2)
            errors.append(s)
            error += s
        """

        # This part finds outliers and their frames
        merr = np.mean(errors)
        stderr = np.std(errors)
        outliers = list()
        ptsi = list()

        # pickle.dump(self.indices['paired'], open('paired.pkl', 'w'))

        if self.indices['paired'] is not None:
            pb_1 = len(self.indices['paired'][0])
        else:
            pb_1 = -1

        if self.indices['unpaired'] is not None:
            upindices = self.indices['unpaired'][0]
            for k in range(1, len(self.indices['unpaired'])):
                upindices = np.hstack((upindices, self.indices['unpaired'][k]))

        for k in range(len(errors)):
            if errors[k] >= 3 * stderr + merr:
                if indices[k] not in ptsi:
                    ptsi.append(indices[k])

                if self.nRef - 1 < indices[k] < pb_1 + self.nRef:
                    # if not self.indices['paired'][0][k] in frames:
                    outliers.append([self.indices['paired'][0][indices[k] - self.nRef] + 1,
                                     redistort_pts(np.array([uv[indices[k]]]), self.cams[cam_index])[0],
                                     'Paired (set 1)', errors[k]])
                    # frames.append(self.indices['paired'][0][k])
                elif pb_1 + self.nRef <= indices[k] < self.nppts + self.nRef:
                    # if not self.indices['paired'][1][k - len(self.indices['paired'][0])] in frames:
                    outliers.append(
                        [self.indices['paired'][1][indices[k] - len(self.indices['paired'][0]) - self.nRef] + 1,
                         redistort_pts(np.array([uv[indices[k]]]), self.cams[cam_index])[0], 'Paired (set 2)',
                         errors[k]])
                    # frames.append(self.indices['paired'][1][k - len(self.indices['paired'][0])])
                elif self.nppts + self.nRef <= indices[k] < self.nuppts + self.nppts + self.nRef:
                    try:
                        # if not upindices[(k - self.nppts)] in frames:
                        i = 0
                        _ = 0

                        while True:
                            if (indices[k] - self.nppts) - self.nRef < len(self.indices['unpaired'][i]) + _:
                                outliers.append([upindices[(indices[k] - self.nppts) - self.nRef] + 1,
                                                 redistort_pts(np.array([uv[indices[k]]]), self.cams[cam_index])[0],
                                                 'Unpaired ', errors[k], i])
                                break
                            else:
                                _ += len(self.indices['unpaired'][i])
                                i += 1
                        # frames.append(upindices[k - self.nppts])
                    except:
                        print('Looking for unpaired indices failed')
                        pass
                else:
                    # print pb1, self.nppts
                    pass

        rmse = error / float(len(errors))

        return L, rmse, outliers, ptsi

    def graph(self):
        # Load the points and camera profile from SBA
        xyzs = np.loadtxt(self.temp + '/' + self.key + '_np.txt')
        cam = np.loadtxt(self.temp + '/' + self.key + '_cn.txt')

        qT = cam[:, -7:]
        quats = qT[:, :4]
        trans = qT[:, 4:]

        """
        cameraPositions = list()
        cameraOrientations = list()

        for k in range(quats.shape[0]):
            Q = quaternions.Quaternion(quats[k][0], quats[k][1], quats[k][2], quats[k][3])
            R = Q.asRotationMatrix()

            cameraPositions.append(np.asarray(-R.T.dot(trans[k])))
            cameraOrientations.append(R.T.dot(np.asarray([0,0,np.nanmin(xyzs[:,2])/2.])))

        vP = np.zeros((quats.shape[0], 6))
        for k in range(quats.shape[0]):
            vP[k,:3] = cameraPositions[k]
            vP[k,3:] = cameraPositions[k] + cameraOrientations[k]

        """
        # If we've got paired points, define a scale
        if self.nppts != 0:
            paired = xyzs[self.nRef:self.nppts + self.nRef]
            p1, p2, pairedSet1, pairedSet2 = self.pairedIsomorphism(paired)
            dist, std = self.averDist(pairedSet1, pairedSet2)
            factor = self.scale / dist
        else:
            # else no scale, just arbitrary
            p1, p2 = None, None
            factor = 1.

        xyzs = xyzs*factor # apply scale factor to all xyz points
        
        if self.ref:
            print('Using reference points')
            xyzs = self.transform(xyzs, xyzs[:self.nRef, :])
            ref = xyzs[:self.nRef, :] # transformed reference points

        else:
            print('No reference points available - centering the calibration on the mean point location.')
            ref = None
            t = np.mean(xyzs, axis=0)
            for k in range(xyzs.shape[0]):
                xyzs[k] = xyzs[k] - t # changed by Ty from + to - to center an unaligned calibration 2020-05-26 version 2.1.2
        # now that we've applied the scale and alignment, re-extract the paired points for proper display
        if self.nppts != 0:
            paired = xyzs[self.nRef:self.nppts + self.nRef]
            p1, p2, pairedSet1, pairedSet2 = self.pairedIsomorphism(paired)

        # get DLT coefficients
        camn = 0
        errs = list()
        dlts = list()
        outliers = []
        ptsi = []
        for uv in self.uvs:
            cos, error, outlier, ind = self.getCoefficients(xyzs, uv, camn)
            camn += 1
            outliers = outliers + outlier
            ptsi = ptsi + ind
            # print len(ind)
            # print len(outlier)
            dlts.append(cos)
            errs.append(error)

        dlts = np.asarray(dlts)
        errs = np.asarray(errs)
        self.dlterrors = errs
        #print errors and wand score to the log
        self.outputDLT(dlts, errs)
        sys.stdout.flush()
        
        if self.nppts != 0:
            self.wandscore = 100. * (std / dist)
            print('\nWand score: ' + str(self.wandscore))
            sys.stdout.flush()
        else:
            print('\nWand score: not applicable')
        sys.stdout.flush()
        
        # start making the graph
        # app = QApplication([])
        # Create a main widget

        # main_layout = QVBoxLayout()
        # main_widget.setLayout(main_layout)
        
        # Create a GL View widget for displaying 3D data
        # view = gl.GLViewWidget()
        # view.show()
        # view.setWindowTitle('3D Graph')
        # view.setCameraPosition(distance=20)

        # # Create grid items for better visualization
        # grid = gl.GLGridItem()
        # grid.scale(2, 2, 1)
        # view.addItem(grid)

        # # Add x, y, z axes lines
        # axis_length = 10
        # x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_length, 0, 0]]), color=(1, 0, 0, 1), width=2)  # Red line for x-axis
        # y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]]), color=(0, 1, 0, 1), width=2)  # Green line for y-axis
        # z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]]), color=(0, 0, 1, 1), width=2)  # Blue line for z-axis
        # view.addItem(x_axis)
        # view.addItem(y_axis)
        # view.addItem(z_axis)

        # Add labels for x, y, z axes - not working
        # font = QFont()
        # font.setPointSize(20)
        # x_label = GLTextItem(text='X', color=(1, 0, 0, 1), pos=(axis_length, 0, 0), font=font)
        # y_label = GLTextItem(text='Y', color=(0, 1, 0, 1), pos=(0, axis_length, 0), font=font)
        # z_label = GLTextItem(text='Z', color=(0, 0, 1, 1), pos=(0, 0, axis_length), font=font)
        # view.addItem(x_label)
        # view.addItem(y_label)
        # view.addItem(z_label)
        
        # Trim off the reference points as we don't want to graph them with the other xyz
        xyzs = xyzs[self.nRef:, :]

        # Plot unpaired points
        if self.nuppts != 0:
            up = xyzs[self.nppts:, :]
            if self.display:
                x = up[:, 0]
                y = up[:, 1]
                z = up[:, 2]
                scatter = gl.GLScatterPlotItem(pos=np.array([x, y, z]).T, color=(0, 1, 1, 1), size=20)  # Cyan color, larger markers
                scatter.setGLOptions('translucent')
                self.view.addItem(scatter)

        # Plot paired points and draw lines between each paired set
        if self.nppts != 0 and self.display:
            for k in range(len(pairedSet1)):
                points = np.vstack((pairedSet1[k], pairedSet2[k]))
                x = points[:, 0]
                y = points[:, 1]
                z = points[:, 2]
                line = gl.GLLinePlotItem(pos=np.array([x, y, z]).T, color=(1, 0, 1, 1), width=5, antialias=True)  # Magenta color
                self.view.addItem(line)

        # Plot reference points
        if self.nRef != 0 and self.display:
            scatter = gl.GLScatterPlotItem(pos=ref, color=(1, 0, 0, 1), size=20)  # Red color, larger markers
            scatter.setGLOptions('translucent')
            self.view.addItem(scatter)

        # Get the camera locations as expressed in the DLT coefficients
        camXYZ = DLTtoCamXYZ(dlts)
        plotcamXYZ = np.array(camXYZ).reshape(-1, 3)  # Ensure camXYZ is a 2D array of shape (n_points, 3)
        scatter = gl.GLScatterPlotItem(pos=plotcamXYZ, color=(0, 1, 0, 1), size=10)  # Green color, larger markers
        scatter.setGLOptions('translucent')
        self.view.addItem(scatter)

        # Calculate automatic orientation length based on data range
        # Combine all visible 3D points to determine the overall scale
        all_points = []
        if len(plotcamXYZ) > 0:
            all_points.append(plotcamXYZ)
        if self.nRef != 0 and ref is not None:
            all_points.append(ref)
        if self.nppts != 0:
            all_points.append(pairedSet1)
            all_points.append(pairedSet2)
        if self.nuppts != 0:
            all_points.append(xyzs[self.nppts:, :])  # unpaired points
        
        if all_points:
            all_coords = np.vstack(all_points)
            # Calculate the range in each dimension
            x_range = np.max(all_coords[:, 0]) - np.min(all_coords[:, 0])
            y_range = np.max(all_coords[:, 1]) - np.min(all_coords[:, 1])
            z_range = np.max(all_coords[:, 2]) - np.min(all_coords[:, 2])
            # Use 5-10% of the maximum range as orientation length
            max_range = max(x_range, y_range, z_range)
            orientation_length = max_range * 0.1  # 10% of the data range
        else:
            orientation_length = 1.0  # fallback value

        # Add camera orientation lines (lollipop style)
        # Calculate orientation vectors for each camera using quaternions
        import scipy.spatial.transform
        for k in range(len(plotcamXYZ)):
            # Convert quaternion to rotation matrix
            rotation = scipy.spatial.transform.Rotation.from_quat([quats[k][1], quats[k][2], quats[k][3], quats[k][0]])  # Note: scipy expects [x,y,z,w] format
            R = rotation.as_matrix()
            
            # Camera optical axis is typically the negative z-axis in camera coordinates
            optical_axis = np.array([0, 0, -1])
            world_optical_axis = R @ optical_axis
            
            # Scale the orientation vector for visibility
            start_point = plotcamXYZ[k]
            end_point = start_point + world_optical_axis * orientation_length
            
            # Create line from camera position to show orientation
            line_points = np.array([start_point, end_point])
            orientation_line = gl.GLLinePlotItem(pos=line_points, color=(0, 0.8, 0, 1), width=3)  # Darker green line
            self.view.addItem(orientation_line)
        
        outputter = WandOutputter(self.name, self.ncams, self.npframes, pairedSet1, pairedSet2, self.indices['paired'], up, self.indices['unpaired'], self.nupframes)
        outputter.output()

        # if self.display:
        #     app.exec_()
            
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.set_aspect('equal') # doesn't look good for 3D
        # # main trick for getting axes to be equal (getting equal scaling) is to create "bounding box" points that set
        # # upper and lower axis limits to the same values on all three axes (https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to)
        # # trim off the reference points as we don't want to graph them with the other xyz
        # xyzs = xyzs[self.nRef:, :]

        # # vP = vP
        # # x = vP[:,:3][:,0]
        # # y = vP[:,:3][:,1]
        # # z = vP[:,:3][:,2]

        # # plot unpaired points
        # # ax.scatter(x,y,z)
        # if self.nuppts != 0:
        #     up = xyzs[self.nppts:, :]
        #     if self.display:
        #         x = up[:, 0]
        #         y = up[:, 1]
        #         z = up[:, 2]
        #         ax.scatter(x, y, z,c='c',label='Unpaired points')
        # else:
        #     up = None

        # # plot the paired points if there are any. draw a line between each paired set.
        # if self.nppts != 0 and self.display:
        #     ax.set_xlabel('X (Meters)')
        #     ax.set_ylabel('Y (Meters)')
        #     ax.set_zlabel('Z (Meters)')
        #     for k in range(len(pairedSet1)):
        #         _ = np.vstack((pairedSet1[k], pairedSet2[k]))
        #         x = _[:, 0]
        #         y = _[:, 1]
        #         z = _[:, 2]
        #         if k == 0:
        #             ax.plot(x, y, z,c='m',label='Paired points')
        #         else:
        #             ax.plot(x, y, z,c='m')
                
        # # plot the reference points if there are any
        # if self.nRef != 0 and self.display:
        #     ax.scatter(ref[:,0],ref[:,1],ref[:,2], c='r', label='Reference points')
            
        # # get the camera locations as expressed in the DLT coefficients
        # camXYZ = DLTtoCamXYZ(dlts)
        # ax.scatter(camXYZ[:,0],camXYZ[:,1],camXYZ[:,2], c='g', label='Camera positions')
            
        # # add the legend, auto-generated from label='' values for each plot entry
        # if self.display:
        #     ax.legend()

        # outputter = WandOutputter(self.name, self.ncams, self.npframes, p1, p2, self.indices['paired'], up,
        #                           self.indices['unpaired'], self.nupframes)
        # outputter.output()

        # if self.display:
        #     try:
        #         if sys.platform == 'linux2':
        #             # have to block on Linux, looking for fix...
        #             plt.show()
        #         else:
        #             if self.report:
        #                 plt.show(block=False)
        #             else:
        #                 plt.show()
        #     except Exception as e:
        #         print('Could not graph!\n' + e)

        return outliers, ptsi