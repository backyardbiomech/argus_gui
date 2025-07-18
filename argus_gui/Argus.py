# Fix Qt platform plugin path for Windows - MUST be done before any Qt imports
import sys
import os
if sys.platform.startswith('win'):
    print("Argus.py: Setting up Windows Qt environment variables...")
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
                print(f"  Found Qt plugins at: {path}")
                break
        
        if qt_plugin_path:
            os.environ['QT_PLUGIN_PATH'] = qt_plugin_path
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
            print(f"  Set QT_PLUGIN_PATH to: {qt_plugin_path}")
            
            # Check if the windows platform plugin specifically exists
            windows_plugin = os.path.join(qt_plugin_path, 'platforms', 'qwindows.dll')
            if os.path.exists(windows_plugin):
                print(f"  [OK] Windows platform plugin found: {windows_plugin}")
            else:
                print(f"  [WARN] Windows platform plugin NOT found at: {windows_plugin}")
                # List what's actually in the platforms directory
                platforms_dir = os.path.join(qt_plugin_path, 'platforms')
                if os.path.exists(platforms_dir):
                    plugins_found = os.listdir(platforms_dir)
                    print(f"  Available platform plugins: {plugins_found}")
        else:
            print("  [WARN] Could not find Qt plugins directory in any expected location")
            
        # Additional Qt environment variables for Windows
        os.environ['QT_QPA_PLATFORM'] = 'windows'
        print("  Set QT_QPA_PLATFORM=windows")
        
    except Exception as e:
        print(f"  Qt plugin path setup error: {e}")

from PySide6 import QtGui, QtWidgets, QtCore
# import pkg_resources
import importlib.resources
import os.path
import shutil
import subprocess
import cv2
import numpy as np

# import pandas as pd
import sys
import yaml
import string
import random
import tempfile
import psutil
import time
from PySide6.QtCore import QStandardPaths

# from argus_gui import Logger

# Ensure the package is available
try:
    # Use the newer files() API instead of deprecated path()
    resource_files = importlib.resources.files("argus_gui.resources")
    RESOURCE_PATH = os.path.abspath(str(resource_files))
except ModuleNotFoundError:
    raise ImportError("The package 'argus_gui' is required but not installed.")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app):
        super().__init__()

        # Set the window title
        self.setWindowTitle("Argus")
        self.app = app

        # Set the window icon
        self.setWindowIcon(QtGui.QIcon(os.path.join(RESOURCE_PATH, "icons/eye-8x.gif")))

        # Set the initial directory to the user's home directory
        self.current_directory = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.HomeLocation
        )

        # Create central widget
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        # Create left tabbed panel
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setMaximumWidth(700)
        # Create right panel
        self.right_panel = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QGridLayout()
        self.right_layout.setColumnMinimumWidth(0, 300)
        self.logwindow = QtWidgets.QTextEdit()
        self.logwindow.setReadOnly(True)
        font = self.logwindow.font()
        font.setFamily("Courier New")
        self.logwindow.setFont(font)
        self.cancel_button = QtWidgets.QPushButton("Cancel running process")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.on_cancel)
        self.about_button = QtWidgets.QPushButton("About this software")
        self.about_button.setToolTip("Show information about this software")
        self.about_button.clicked.connect(self.about)
        self.quit_button = QtWidgets.QPushButton("Quit Argus")
        self.quit_button.setToolTip("Quit Argus")
        self.quit_button.clicked.connect(self.quit_all)
        self.right_layout.addWidget(self.logwindow, 0, 0, 6, 3)
        self.right_layout.addWidget(self.cancel_button, 7, 0)
        self.right_layout.addWidget(self.about_button, 7, 1)
        self.right_layout.addWidget(self.quit_button, 7, 2)
        self.right_panel.setLayout(self.right_layout)

        # Set up some variables
        # general
        self.tmps = []
        self.pids = []

        # clicker
        self.drivers = []

        self.tmps.append(tempfile.mkdtemp())
        # dictionary of cached files, relating random key to movie location
        self.cached = dict()

        # Add tabs with different widgets
        self.add_clicker_tab()
        self.add_sync_tab()
        self.add_wand_tab()
        self.add_patterns_tab()
        self.add_calibrate_tab()
        self.add_dwarp_tab()

        # Create main layout
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addWidget(self.tab_widget)
        self.main_layout.addWidget(self.right_panel)
        self.central_widget.setLayout(self.main_layout)

    def add_clicker_tab(self):
        # Create the Clicker tab
        # Create file table with video paths and offsets
        self.file_table = QtWidgets.QTableWidget()
        self.file_table.setColumnCount(2)
        self.file_table.setHorizontalHeaderLabels(["Video File", "Frame Offset"])
        self.file_table.setToolTip(
            "List of movies to click through.\nPress '+' button to add movie.\nDouble-click offset to edit."
        )
        # Make the video file column stretch to fill available space
        header = self.file_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        
        # Create add button
        self.add_button = QtWidgets.QPushButton("+")
        self.add_button.setMaximumWidth(30)
        self.add_button.setToolTip("Add a movie to the list")
        self.add_button.clicked.connect(self.add)
        # Create clear button
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.setMaximumWidth(60)
        self.clear_button.setToolTip("Clear the list of movies")
        self.clear_button.clicked.connect(self.clear)
        # Create remove button
        self.remove_button = QtWidgets.QPushButton("-")
        self.remove_button.setMaximumWidth(30)
        self.remove_button.setToolTip("Remove the selected movie from the list")
        self.remove_button.clicked.connect(self.delete)
        # Create resolution dropdown
        self.resolution_label = QtWidgets.QLabel("Diplay Resolution: ")
        self.resolution_var = QtWidgets.QComboBox()
        self.resolution_var.addItems(["Half", "Full"])
        # Create load config button
        self.load_button = QtWidgets.QPushButton("Load Config")
        self.load_button.setToolTip("Load Clicker configuration from a file")
        self.load_button.clicked.connect(self.load)
        # Create go button
        self.go_button = QtWidgets.QPushButton("Go")
        self.go_button.setToolTip("Start digitizing through the movies")
        self.go_button.clicked.connect(self.clicker_go)
        
        # Connect cell changed signal to update offsets
        self.file_table.cellChanged.connect(self.update_offset)
        
        # Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.file_table, 0, 0, 3, 4)  # Give table more width
        layout.addWidget(self.add_button, 0, 4)
        layout.addWidget(self.remove_button, 1, 4)
        layout.addWidget(self.clear_button, 2, 4)
        layout.addWidget(self.resolution_label, 3, 0)
        layout.addWidget(self.resolution_var, 3, 1)
        layout.addWidget(self.load_button, 5, 0)
        layout.addWidget(self.go_button, 5, 4)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)

        self.tab_widget.addTab(
            tab,
            QtGui.QIcon(os.path.join(RESOURCE_PATH, "icons/location-8x.gif")),
            "Clicker",
        )

    def add_sync_tab(self):
        # Create the Sync tab
        # Create file list
        self.sync_file_list = QtWidgets.QListWidget()
        self.sync_file_list.setToolTip(
            "List of movies to syncronize\nPress '+' button to add movie"
        )
        # Create add button
        self.sync_add_button = QtWidgets.QPushButton(" + ")
        self.sync_add_button.setToolTip("Add a movie to the list")
        self.sync_add_button.clicked.connect(self.add)
        # Create clear button
        self.sync_clear_button = QtWidgets.QPushButton("Clear")
        self.sync_clear_button.setToolTip("Clear the list of movies")
        self.clear_button.clicked.connect(self.clear)
        # Create remove button
        self.sync_remove_button = QtWidgets.QPushButton(" - ")
        self.sync_remove_button.setToolTip("Remove the selected movie from the list")
        self.sync_remove_button.clicked.connect(self.delete)
        # Create go button
        self.sync_go_button = QtWidgets.QPushButton("Go")
        self.sync_go_button.setToolTip("Synchronize the videos")
        self.sync_go_button.clicked.connect(self.sync_go)
        # create log check box
        self.sync_log = QtWidgets.QCheckBox("Write Log")

        # Sync specific
        self.show_waves_button = QtWidgets.QPushButton("Show waves")
        self.show_waves_button.setToolTip(
            "Graph the audio tracks from the movies\nHelps better select a reasonable time range"
        )
        self.show_waves_button.clicked.connect(self.sync_show)
        self.crop = QtWidgets.QCheckBox("Specify time range")
        self.crop.stateChanged.connect(self.updateCropOptions)
        self.time_label = QtWidgets.QLabel("Time range containing sync sounds")
        self.start_crop = QtWidgets.QLineEdit()
        self.start_crop.setValidator(QtGui.QDoubleValidator())
        self.start_crop.setText("0.0")
        self.start_crop.setEnabled(False)
        self.start_label = QtWidgets.QLabel("Start time (decimal minutes)")
        self.end_crop = QtWidgets.QLineEdit()
        self.end_crop.setValidator(QtGui.QDoubleValidator())
        self.end_crop.setText("4.0")
        self.end_crop.setEnabled(False)
        self.end_label = QtWidgets.QLabel("End time  (decimal minutes)")

        self.sync_onam_label = QtWidgets.QLabel("Output filename")
        self.sync_onam = QtWidgets.QLineEdit()
        self.sync_onam_button = QtWidgets.QPushButton("Specify")
        self.sync_onam_button.clicked.connect(self.save_loc)

        # time box
        time_layout = QtWidgets.QGridLayout()
        # time_layout.addWidget(self.time_label, 0, 0)
        time_layout.addWidget(self.start_label, 1, 0)
        time_layout.addWidget(self.start_crop, 1, 1)
        time_layout.addWidget(self.end_label, 2, 0)
        time_layout.addWidget(self.end_crop, 2, 1)
        time_box = QtWidgets.QGroupBox("Time range containing sync sounds")
        time_box.setLayout(time_layout)

        # Layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.sync_file_list, 0, 0, 3, 2)
        layout.addWidget(self.sync_add_button, 0, 2)
        layout.addWidget(self.sync_clear_button, 2, 2)
        layout.addWidget(self.sync_remove_button, 1, 2)
        layout.addWidget(self.show_waves_button, 3, 0, 1, 2)
        layout.addWidget(self.crop, 4, 0)
        layout.addWidget(time_box, 6, 0, 2, 3)
        layout.addWidget(self.sync_log, 11, 0)
        layout.addWidget(self.sync_onam_label, 9, 0)
        layout.addWidget(self.sync_onam_button, 10, 0)
        layout.addWidget(self.sync_onam, 10, 1, 1, 2)
        layout.addWidget(self.sync_go_button, 11, 2)
        tab = QtWidgets.QWidget()
        tab.setLayout(layout)

        self.tab_widget.addTab(
            tab, QtGui.QIcon(os.path.join(RESOURCE_PATH, "icons/pulse-8x.gif")), "Sync"
        )

    def add_wand_tab(self):
        # Create the Wand tab

        self.intModeDict = {
            "Optimize none": "0",
            "Optimize focal length": "1",
            "Optimize focal length and principal point": "2",
        }

        self.disModeDict = {
            "Optimize none": "0",
            "Optimize r2": "1",
            "Optimize r2, r4": "2",
            "Optimize all distortion coefficients": "3",
        }

        self.refModeDict = {"Axis points": "0", "Gravity": "1", "Plane": "2"}

        # Create go button
        self.wand_go_button = QtWidgets.QPushButton("Go")
        self.wand_go_button.setToolTip("Calculate the DLT calibration")
        self.wand_go_button.clicked.connect(self.wand_go)

        ops_label = QtWidgets.QLabel("Options")

        self.ppts = QtWidgets.QLineEdit()
        self.ppts_button = QtWidgets.QPushButton("Select paired points file")
        self.ppts_button.setToolTip(
            "Open a CSV file with paired (wand) pixel coordinates"
        )
        self.ppts_button.clicked.connect(self.add)
        self.uppts = QtWidgets.QLineEdit()
        self.uppts_button = QtWidgets.QPushButton("Select unpaired points file")
        self.uppts_button.setToolTip(
            "Open a CSV file with unpaired (background) pixel coordinates"
        )
        self.uppts_button.clicked.connect(self.add)
        self.wand_cams = QtWidgets.QLineEdit()
        self.wand_cams_button = QtWidgets.QPushButton("Select camera profile")
        self.wand_cams_button.setToolTip(
            "Open a CSV or TXT file with camera intrinsic and extrinsics"
        )
        self.wand_cams_button.clicked.connect(self.add)
        self.wand_refs = QtWidgets.QLineEdit()
        self.wand_refs_button = QtWidgets.QPushButton("Select reference points")
        self.wand_refs_button.setToolTip("Open a CSV file with axes pixel coordinates")
        self.wand_refs_button.clicked.connect(self.add)

        # reference point options
        self.wand_reftype_label = QtWidgets.QLabel("Reference point type:")
        self.wand_reftype = QtWidgets.QComboBox()
        for key in self.refModeDict.keys():
            self.wand_reftype.addItem(key)
        self.wand_reftype.currentIndexChanged.connect(self.updateFreqBoxState)
        self.wand_reftype.setToolTip(
            "Set the reference type. \nAxis points are 1-4 points defining the origin and axes, \nGravity is an object accelerating due to gravity, \nPlane are 3+ points that define the X-Y plane"
        )
        # recording frequency
        self.wand_freq_label = QtWidgets.QLabel("Recording frequency (fps)")
        self.wand_freq = QtWidgets.QLineEdit()
        self.wand_freq.setValidator(QtGui.QDoubleValidator())
        self.updateFreqBoxState()

        self.wand_scale_label = QtWidgets.QLabel("Wand length")
        self.wand_scale = QtWidgets.QLineEdit()
        self.wand_scale.setValidator(QtGui.QDoubleValidator())
        self.wand_scale.setText("1.0")
        self.wand_scale.setToolTip(
            "Enter the distance between paired points (wand length) as m"
        )

        self.wand_instrics_label = QtWidgets.QLabel("Intriniscs: ")
        self.wand_intrinsics = QtWidgets.QComboBox()
        for key in self.intModeDict.keys():
            self.wand_intrinsics.addItem(key)

        self.wand_dist_label = QtWidgets.QLabel("Distortion: ")
        self.wand_dist = QtWidgets.QComboBox()
        for key in self.disModeDict.keys():
            self.wand_dist.addItem(key)
        # options boxes
        self.wand_outliers = QtWidgets.QCheckBox("Report on outliers")
        self.wand_outliers.setToolTip("Process outlier point with option to remove")
        self.wand_chooseRef = QtWidgets.QCheckBox("Choose reference cameras")
        self.wand_outputProf = QtWidgets.QCheckBox("Output camera profiles")
        self.wand_display = QtWidgets.QCheckBox("Display results")
        self.wand_display.setChecked(True)
        self.wand_display.stateChanged.connect(self.update_outliers_checkbox_state)
        self.wand_log = QtWidgets.QCheckBox("Write log")

        self.wand_onam_label = QtWidgets.QLabel("Output file prefix and location")
        self.wand_onam_button = QtWidgets.QPushButton("Specify")
        self.wand_onam_button.clicked.connect(self.save_loc)
        self.wand_onam = QtWidgets.QLineEdit()

        # options box
        options_layout = QtWidgets.QGridLayout()
        # options_layout.addWidget(ops_label, 7, 0)
        options_layout.addWidget(self.wand_instrics_label, 8, 0)
        options_layout.addWidget(self.wand_intrinsics, 8, 1)
        options_layout.addWidget(self.wand_dist_label, 9, 0)
        options_layout.addWidget(self.wand_dist, 9, 1)
        options_layout.addWidget(self.wand_outliers, 10, 0)
        options_layout.addWidget(self.wand_chooseRef, 10, 1)
        options_layout.addWidget(self.wand_outputProf, 10, 2)
        options_layout.addWidget(self.wand_display, 10, 3)
        options_box = QtWidgets.QGroupBox("Options")
        options_box.setLayout(options_layout)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.ppts_button, 1, 0)
        layout.addWidget(self.ppts, 1, 1, 1, 3)
        layout.addWidget(self.uppts_button, 2, 0)
        layout.addWidget(self.uppts, 2, 1, 1, 3)
        layout.addWidget(self.wand_refs_button, 3, 0)
        layout.addWidget(self.wand_refs, 3, 1, 1, 3)
        layout.addWidget(self.wand_cams_button, 4, 0)
        layout.addWidget(self.wand_cams, 4, 1, 1, 3)
        layout.addWidget(self.wand_reftype_label, 5, 2)
        layout.addWidget(self.wand_reftype, 5, 3)
        layout.addWidget(self.wand_freq_label, 6, 2)
        layout.addWidget(self.wand_freq, 6, 3)
        layout.addWidget(self.wand_scale_label, 5, 0)
        layout.addWidget(self.wand_scale, 5, 1)
        layout.addWidget(options_box, 7, 0, 4, 4)
        layout.addWidget(self.wand_onam_label, 11, 0)
        layout.addWidget(self.wand_onam_button, 12, 0)
        layout.addWidget(self.wand_onam, 12, 1, 1, 3)
        layout.addWidget(self.wand_log, 13, 0)
        layout.addWidget(self.wand_go_button, 13, 3)

        # layout.setColumnMinimumWidth(1, 400)

        # Set initial state for outliers checkbox based on display checkbox
        self.update_outliers_checkbox_state()

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)

        self.tab_widget.addTab(
            tab, QtGui.QIcon(os.path.join(RESOURCE_PATH, "icons/wand.gif")), "Wand"
        )

    def add_patterns_tab(self):
        # Create the Patterns tab
        # Create go button
        self.patt_go_button = QtWidgets.QPushButton("Go")
        self.patt_go_button.setToolTip("Detect patterns in video")
        self.patt_go_button.clicked.connect(self.pattern_go)

        self.patt_file_button = QtWidgets.QPushButton("Select video of a pattern")
        self.patt_file_button.clicked.connect(self.add)
        self.patt_file = QtWidgets.QLineEdit()

        # settings options
        self.patt_display = QtWidgets.QCheckBox(
            "Display pattern recognition in progress"
        )
        self.patt_display.setToolTip(
            "Check to view pattern detection live - slows down processing"
        )
        self.patt_type_label = QtWidgets.QLabel("Pattern type: ")
        self.patt_dots = QtWidgets.QRadioButton("Dots")
        self.patt_dots.setChecked(True)
        self.patt_chess = QtWidgets.QRadioButton("Chess board")
        set_layout = QtWidgets.QGridLayout()
        set_layout.addWidget(self.patt_display, 0, 0)
        set_layout.addWidget(self.patt_type_label, 1, 0)
        set_layout.addWidget(self.patt_dots, 1, 1)
        set_layout.addWidget(self.patt_chess, 1, 2)

        # settings box
        sett_box = QtWidgets.QGroupBox("Settings")
        sett_box.setLayout(set_layout)

        # Parameters
        param_box = QtWidgets.QGroupBox("Parameters")
        # pattern box
        patt_box = QtWidgets.QGroupBox("Pattern")
        self.patt_rows_label = QtWidgets.QLabel("Shapes per row:")
        self.patt_rows = QtWidgets.QSpinBox()
        self.patt_rows.setValue(12)
        self.patt_cols_label = QtWidgets.QLabel("Shapes per column:")
        self.patt_cols = QtWidgets.QSpinBox()
        self.patt_cols.setValue(9)
        self.patt_space_label = QtWidgets.QLabel("Spacing between shapes (m)")
        self.patt_space = QtWidgets.QLineEdit()
        self.patt_space.setValidator(QtGui.QDoubleValidator())
        self.patt_space.setText("0.02")
        patt_layout = QtWidgets.QGridLayout()
        patt_layout.addWidget(self.patt_rows_label, 0, 0)
        patt_layout.addWidget(self.patt_rows, 0, 1)
        patt_layout.addWidget(self.patt_cols_label, 1, 0)
        patt_layout.addWidget(self.patt_cols, 1, 1)
        patt_layout.addWidget(self.patt_space_label, 2, 0)
        patt_layout.addWidget(self.patt_space, 2, 1)
        patt_box.setLayout(patt_layout)
        # movie box
        mov_box = QtWidgets.QGroupBox("Movie")
        self.patt_start_label = QtWidgets.QLabel("Start time:")
        self.patt_start = QtWidgets.QLineEdit()
        self.patt_start.setToolTip("Time in the video to begin pattern recognition.")
        self.patt_start.setValidator(QtGui.QDoubleValidator())
        self.patt_end_label = QtWidgets.QLabel("End time:")
        self.patt_end = QtWidgets.QLineEdit()
        self.patt_end.setToolTip("Time in the video to stop pattern recognition.")
        self.patt_end.setValidator(QtGui.QDoubleValidator())
        mov_layout = QtWidgets.QGridLayout()
        mov_layout.addWidget(self.patt_start_label, 0, 0)
        mov_layout.addWidget(self.patt_start, 0, 1)
        mov_layout.addWidget(self.patt_end_label, 1, 0)
        mov_layout.addWidget(self.patt_end, 1, 1)
        mov_box.setLayout(mov_layout)
        param_layout = QtWidgets.QGridLayout()
        param_layout.addWidget(patt_box, 0, 0)
        param_layout.addWidget(mov_box, 0, 1)
        param_box.setLayout(param_layout)

        self.patt_log = QtWidgets.QCheckBox("Write log")
        # Create about button
        self.patt_about_button = QtWidgets.QPushButton("About")
        self.patt_about_button.setToolTip("Show information about this software")
        self.patt_about_button.clicked.connect(self.about)
        self.patt_onam_label = QtWidgets.QLabel("Output file prefix and location")
        self.patt_onam_button = QtWidgets.QPushButton("Specify")
        self.patt_onam_button.clicked.connect(self.save_loc)
        self.patt_onam = QtWidgets.QLineEdit()

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.patt_file_button, 1, 0)
        layout.addWidget(self.patt_file, 1, 1, 1, 3)
        layout.addWidget(sett_box, 2, 0, 1, 6)
        layout.addWidget(param_box, 3, 0, 2, 6)
        layout.addWidget(self.patt_onam_label, 5, 0)
        layout.addWidget(self.patt_onam_button, 6, 0)
        layout.addWidget(self.patt_onam, 6, 1, 1, 5)
        layout.addWidget(self.patt_log, 7, 0)
        layout.addWidget(self.patt_go_button, 7, 5)
        # layout.addWidget(self.patt_quit_button, 9, 0)
        # layout.addWidget(self.patt_about_button, 9, 6)
        tab = QtWidgets.QWidget()
        tab.setLayout(layout)

        self.tab_widget.addTab(
            tab,
            QtGui.QIcon(os.path.join(RESOURCE_PATH, "icons/grid-four-up-8x.gif")),
            "Patterns",
        )

    def add_calibrate_tab(self):
        # Create the Calibrate tab
        self.cal_log = QtWidgets.QCheckBox("Write log")
        # Create about button
        # self.cal_about_button = QtWidgets.QPushButton('About')
        # self.cal_about_button.setToolTip('Show information about this software')
        # self.cal_about_button.clicked.connect(self.about)
        self.cal_onam_label = QtWidgets.QLabel("Output file prefix and location")
        self.cal_onam_button = QtWidgets.QPushButton("Specify")
        self.cal_onam_button.clicked.connect(self.save_loc)
        self.cal_onam = QtWidgets.QLineEdit()
        # Create go button
        self.cal_go_button = QtWidgets.QPushButton("Go")
        self.cal_go_button.setToolTip("Calculate camera calibration")
        self.cal_go_button.clicked.connect(self.calibrate_go)
        # Create quit button
        # self.cal_quit_button = QtWidgets.QPushButton('Quit')
        # self.cal_quit_button.setToolTip('Quit Argus')
        # self.cal_quit_button.clicked.connect(self.quit_all)

        self.cal_file_button = QtWidgets.QPushButton("Select patterns file")
        self.cal_file_button.clicked.connect(self.add)
        self.cal_file_button.setToolTip('Find pickle ("pkl") file of detected patterns')
        self.cal_file = QtWidgets.QLineEdit()
        # Options Box
        self.cal_replicates_label = QtWidgets.QLabel("Number of replications: ")
        self.cal_replicates = QtWidgets.QSpinBox()
        self.cal_replicates.setRange(0, 2**31 - 1)
        self.cal_replicates.setValue(100)
        self.cal_replicates.setToolTip(
            "Number of times to sample the frames and solve the distortion equations"
        )

        self.cal_patterns_label = QtWidgets.QLabel(
            "Sample size (frames) per replicate: "
        )
        self.cal_patterns = QtWidgets.QSpinBox()
        self.cal_patterns.setValue(20)
        self.cal_patterns.setRange(0, 2**31 - 1)
        self.cal_patterns.setToolTip(
            "Number of frames to include in each replicate. \nHigher numbers may improve calibration but exponentially increase processing time"
        )
        self.cal_inv = QtWidgets.QCheckBox("Invert grid coordinates")
        self.cal_inv.setToolTip(
            "If you're getting poor results, try checking this option"
        )
        self.cal_dist_label = QtWidgets.QLabel("Distortion:")
        self.cal_dist_model = QtWidgets.QComboBox()
        self.cal_dist_model.addItems(["Pinhole model", "Omnidirectional model"])
        self.cal_dist_model.setCurrentText("Pinhole model")
        self.cal_dist_model.currentIndexChanged.connect(self.updateCalOptions)
        self.cal_dist_option = QtWidgets.QComboBox()
        self.cal_dist_option.addItems(
            [
                "Optimize k1, k2",
                "Optimize k1, k2, and k3",
                "Optimize all distortion coefficients",
            ]
        )
        self.cal_dist_option.setCurrentText("Optimize k1, k2")

        options_box = QtWidgets.QGroupBox("Options")
        opts_layout = QtWidgets.QGridLayout()
        opts_layout.addWidget(self.cal_replicates_label, 0, 0)
        opts_layout.addWidget(self.cal_replicates, 0, 1)
        opts_layout.addWidget(self.cal_patterns_label, 1, 0)
        opts_layout.addWidget(self.cal_patterns, 1, 1)
        opts_layout.addWidget(self.cal_dist_label, 3, 0)
        opts_layout.addWidget(self.cal_dist_model, 3, 1)
        opts_layout.addWidget(self.cal_dist_option, 3, 2)

        options_box.setLayout(opts_layout)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.cal_file_button, 1, 0)
        layout.addWidget(self.cal_file, 1, 1, 1, 3)
        layout.addWidget(self.cal_inv, 2, 0, 1, 2)
        layout.addWidget(options_box, 3, 0, 3, 4)
        layout.addWidget(self.cal_onam_label, 6, 0, 1, 2)
        layout.addWidget(self.cal_onam_button, 7, 0)
        layout.addWidget(self.cal_onam, 7, 1, 1, 4)
        layout.addWidget(self.cal_log, 8, 0)
        layout.addWidget(self.cal_go_button, 8, 5)
        # layout.addWidget(self.cal_quit_button, 8, 0)
        # layout.addWidget(self.cal_about_button, 9, 5)
        tab = QtWidgets.QWidget()
        tab.setLayout(layout)
        self.tab_widget.addTab(
            tab,
            QtGui.QIcon(os.path.join(RESOURCE_PATH, "icons/calculator-8x.gif")),
            "Calibrate",
        )

    def add_dwarp_tab(self):
        # Create the Dwarp tab

        # common elements
        self.dwarp_log = QtWidgets.QCheckBox("Write log")
        # Create about button
        # self.dwarp_about_button = QtWidgets.QPushButton('About')
        # self.dwarp_about_button.setToolTip('Show information about this software')
        # self.dwarp_about_button.clicked.connect(self.about)
        self.dwarp_onam_label = QtWidgets.QLabel("Output file prefix and location")
        self.dwarp_onam_button = QtWidgets.QPushButton("Specify")
        self.dwarp_onam_button.clicked.connect(self.save_loc)
        self.dwarp_onam = QtWidgets.QLineEdit()
        # Create go button
        self.dwarp_go_button = QtWidgets.QPushButton("Go")
        self.dwarp_go_button.setToolTip("Calculate camera calibration")
        self.dwarp_go_button.clicked.connect(self.dwarp_go)
        # Create quit button
        # self.dwarp_quit_button = QtWidgets.QPushButton('Quit')
        # self.dwarp_quit_button.setToolTip('Quit the program')
        # self.dwarp_quit_button.clicked.connect(self.quit_all)

        self.dwarp_file_button = QtWidgets.QPushButton("Select movie file")
        self.dwarp_file_button.clicked.connect(self.add)
        self.dwarp_file = QtWidgets.QLineEdit()

        # create and fill the camera model and shooting mode drop down menus from the calibrations files
        self.calibFiles = list()

        self.calibFolder = os.path.join(RESOURCE_PATH, "calibrations/")
        for file in os.listdir(self.calibFolder):
            if file.endswith(".csv"):
                self.calibFiles.append(file)

        # self.modes = list()
        self.models = {}

        for file in self.calibFiles:
            if file.split(".")[0] != "":
                mod = file.split(".")[0]
                self.models[mod] = {}
            ifile = open(self.calibFolder + file)
            line = ifile.readline()
            while line != [""]:
                line = ifile.readline().split(",")
                mode = line[0]
                vals = line[1:]
                if mode != "":
                    self.models[mod][mode] = vals

        dwarp_model_label = QtWidgets.QLabel("Camera model:")
        self.dwarp_models = QtWidgets.QComboBox()
        self.dwarp_models.addItems(list(self.models.keys()))
        self.dwarp_models.currentIndexChanged.connect(self.updateCam)
        dwarp_mode_label = QtWidgets.QLabel("Shooting Mode:")
        self.dwarp_modes = QtWidgets.QComboBox()
        self.dwarp_modes.currentIndexChanged.connect(self.calibParse)
        dwarp_fl_label = QtWidgets.QLabel("Focal length (mm)")
        self.dwarp_fl = QtWidgets.QLineEdit()
        self.dwarp_fl.setValidator(QtGui.QDoubleValidator())
        dwarp_cx_label = QtWidgets.QLabel("Horizontal center: ")
        self.dwarp_cx = QtWidgets.QLineEdit()
        self.dwarp_cx.setValidator(QtGui.QDoubleValidator())
        self.dwarp_cx.setToolTip("x-coordinate of optical midpoint")
        dwarp_cy_label = QtWidgets.QLabel("Vertical center")
        self.dwarp_cy = QtWidgets.QLineEdit()
        self.dwarp_cy.setValidator(QtGui.QDoubleValidator())
        self.dwarp_cy.setToolTip("y-coordinate of optical midpoint")
        dwarp_k1_label = QtWidgets.QLabel("Radial distortion:  k1")
        self.dwarp_k1 = QtWidgets.QLineEdit()
        self.dwarp_k1.setValidator(QtGui.QDoubleValidator())
        self.dwarp_k1.setToolTip("2n order radial\ndistortion coefficient")
        dwarp_k2_label = QtWidgets.QLabel("k2")
        self.dwarp_k2 = QtWidgets.QLineEdit()
        self.dwarp_k2.setValidator(QtGui.QDoubleValidator())
        self.dwarp_k2.setToolTip("4th order radial\ndistortion coefficient")
        dwarp_k3_label = QtWidgets.QLabel("k3")
        self.dwarp_k3 = QtWidgets.QLineEdit()
        self.dwarp_k3.setValidator(QtGui.QDoubleValidator())
        self.dwarp_k3.setToolTip("6th order radial\ndistortion coefficient")
        dwarp_t1_label = QtWidgets.QLabel("t1")
        self.dwarp_t1 = QtWidgets.QLineEdit()
        self.dwarp_t1.setValidator(QtGui.QDoubleValidator())
        self.dwarp_t1.setToolTip("1st decentering\ndistortion coefficient")
        dwarp_t2_label = QtWidgets.QLabel("t2")
        self.dwarp_t2 = QtWidgets.QLineEdit()
        self.dwarp_t2.setValidator(QtGui.QDoubleValidator())
        self.dwarp_t2.setToolTip("2nd decentering\ndistortion coefficient")
        dwarp_xi_label = QtWidgets.QLabel("xi")
        self.dwarp_xi = QtWidgets.QLineEdit()
        self.dwarp_xi.setValidator(QtGui.QDoubleValidator())
        self.dwarp_xi.setToolTip(
            "Camera shape parameter for CMei's \n omnidirectional model"
        )
        # lens parameters box
        param_box = QtWidgets.QGroupBox("Lens Parameters")
        param_layout = QtWidgets.QGridLayout()
        param_layout.addWidget(dwarp_model_label, 0, 0)
        param_layout.addWidget(self.dwarp_models, 0, 1, 1, 2)
        param_layout.addWidget(dwarp_mode_label, 0, 3)
        param_layout.addWidget(self.dwarp_modes, 0, 4, 1, 2)
        param_layout.addWidget(dwarp_fl_label, 1, 0)
        param_layout.addWidget(self.dwarp_fl, 1, 1)
        param_layout.addWidget(dwarp_cx_label, 1, 2)
        param_layout.addWidget(self.dwarp_cx, 1, 3)
        param_layout.addWidget(dwarp_cy_label, 1, 4)
        param_layout.addWidget(self.dwarp_cy, 1, 5)
        param_layout.addWidget(dwarp_k1_label, 2, 0)
        param_layout.addWidget(self.dwarp_k1, 2, 1)
        param_layout.addWidget(dwarp_k2_label, 2, 2)
        param_layout.addWidget(self.dwarp_k2, 2, 3)
        param_layout.addWidget(dwarp_k3_label, 2, 4)
        param_layout.addWidget(self.dwarp_k3, 2, 5)
        param_layout.addWidget(dwarp_t1_label, 3, 0)
        param_layout.addWidget(self.dwarp_t1, 3, 1)
        param_layout.addWidget(dwarp_t2_label, 3, 2)
        param_layout.addWidget(self.dwarp_t2, 3, 3)
        param_layout.addWidget(dwarp_xi_label, 3, 4)
        param_layout.addWidget(self.dwarp_xi, 3, 5)

        param_box.setLayout(param_layout)
        # output type options box
        out_box = QtWidgets.QGroupBox("Output type options")
        self.dwarp_opts_wd = QtWidgets.QRadioButton("Write and display video")
        self.dwarp_opts_wd.setChecked(True)
        self.dwarp_opts_do = QtWidgets.QRadioButton("Display only")
        self.dwarp_opts_wo = QtWidgets.QRadioButton("Write only")
        out_layout = QtWidgets.QVBoxLayout()
        out_layout.addWidget(self.dwarp_opts_wd)
        out_layout.addWidget(self.dwarp_opts_do)
        out_layout.addWidget(self.dwarp_opts_wo)
        out_box.setLayout(out_layout)

        # output movie options box
        mov_box = QtWidgets.QGroupBox("Output movie options")
        dwarp_qual_label = QtWidgets.QLabel("Compression level:")
        self.dwarp_qual = QtWidgets.QSpinBox()
        self.dwarp_qual.setValue(12)
        self.dwarp_qual.setRange(0, 63)
        self.dwarp_qual.setToolTip("Must be an integer between 0 and 63.")
        dwarp_int_label = QtWidgets.QLabel("Full frame interval")
        self.dwarp_int = QtWidgets.QSpinBox()
        self.dwarp_int.setValue(25)
        self.dwarp_int.setToolTip(
            "Number of frames in between full frames.\nHigher numbers mean larger file size but faster seek"
        )
        self.dwarp_crop = QtWidgets.QCheckBox("Crop video to undistorted region")
        self.dwarp_copy = QtWidgets.QCheckBox(
            "Copy video and audio codec before undistortion"
        )
        mov_layout = QtWidgets.QGridLayout()
        mov_layout.addWidget(dwarp_qual_label, 0, 0)
        mov_layout.addWidget(self.dwarp_qual, 0, 1)
        mov_layout.addWidget(dwarp_int_label, 1, 0)
        mov_layout.addWidget(self.dwarp_int, 1, 1)
        mov_layout.addWidget(self.dwarp_crop, 2, 0, 1, 2)
        mov_layout.addWidget(self.dwarp_copy, 3, 0, 1, 2)
        mov_box.setLayout(mov_layout)

        # Create informational text at the top
        dwarp_info_label = QtWidgets.QLabel(
            "Dwarp is only necessary for creating undistorted videos for presentations or analysis in other software. "
            "The other Argus modules will undistort behind the scenes when you load a properly formatted camera profile."
        )
        dwarp_info_label.setWordWrap(True)
        dwarp_info_label.setStyleSheet("QLabel { font-style: italic; margin: 10px; }")

        layout = QtWidgets.QGridLayout()
        layout.addWidget(dwarp_info_label, 0, 0, 1, 4)
        layout.addWidget(self.dwarp_file_button, 1, 0)
        layout.addWidget(self.dwarp_file, 1, 1, 1, 3)
        layout.addWidget(param_box, 2, 0, 3, 4)
        layout.addWidget(out_box, 6, 0, 1, 2)
        layout.addWidget(mov_box, 6, 2, 1, 2)
        layout.addWidget(self.dwarp_onam_label, 7, 0)
        layout.addWidget(self.dwarp_onam_button, 8, 0)
        layout.addWidget(self.dwarp_onam, 8, 1, 1, 3)
        layout.addWidget(self.dwarp_log, 9, 0)
        layout.addWidget(self.dwarp_go_button, 9, 3)
        # layout.addWidget(self.dwarp_quit_button, 11, 0)
        # layout.addWidget(self.dwarp_about_button, 11, 5)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)

        self.updateCam()
        self.tab_widget.addTab(
            tab, QtGui.QIcon(os.path.join(RESOURCE_PATH, "icons/eye-8x.gif")), "Dwarp"
        )

    ## General Functions
    def closeEvent(self, event):
        # Call the quit_all function when the window is closed
        self.quit_all()

    def about(self):
        w = QtWidgets.QDialog(self)
        license_text = QtWidgets.QPlainTextEdit()

        with open(os.path.join(RESOURCE_PATH, "LICENSE.txt"), "r") as f:
            text = f.read()
        license_text.setPlainText(text)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(license_text)
        w.setLayout(layout)
        w.exec_()

    def quit_all(self):
        self.close()
        self.kill_pids()
        for tmp in self.tmps:
            # Delete temporary directory in use if it still exists
            if os.path.isdir(tmp):
                shutil.rmtree(tmp)

    def kill_proc_tree(self, pid, including_parent=True):
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        if including_parent:
            parent.kill()
            parent.wait(5)

    def kill_pids(self):
        for pid in self.pids:
            try:
                self.kill_proc_tree(pid)
            except:
                pass

    # def go(self, cmd, wlog=False, mode='DEBUG'):
    #     cmd = [str(wlog), ''] + cmd
    #     rcmd = [sys.executable, os.path.join(RESOURCE_PATH, 'scripts/argus-log')]

    #     rcmd = rcmd + cmd

    #     startupinfo = None
    #     if sys.platform == "win32" or sys.platform == "win64":  # Make it so subprocess brings up no console window
    #         startupinfo = subprocess.STARTUPINFO()
    #         startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    #     print(type(rcmd), rcmd)
    #     print(type(subprocess.PIPE))
    #     print(type(startupinfo))

    #     proc = subprocess.Popen(rcmd, stdout=subprocess.PIPE, shell=False, startupinfo=startupinfo)
    #     self.pids.append(proc.pid)

    # def set_in_filename(self, var, filetypes=''):
    #     # Define the set_in_filename function
    #     options = QtWidgets.QFileDialog.Options()
    #     filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select File', '', filetypes, options=options)
    #     if filename:
    #         var.set(filename)

    # def set_out_filename(self, var, filetypes=''):
    #     options = QtWidgets.QFileDialog.Options()
    #     filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Select File', '', filetypes, options=options)
    #     if filename:
    #         var.set(filename)

    # Function for bringing up file dialogs; adds selected file to listbox
    def add(self):
        """
        Adds files to various tabs
        """
        # This function is used on several tabs, so get the tab name
        current_tab_name = self.tab_widget.tabText(self.tab_widget.currentIndex())

        # Depending on tab name and/or button pushed, call the window something different, filter for different files, and save to different locations

        # Clicker
        if current_tab_name == "Clicker":
            title = "Select Movie File"
            filter = "All files (*)"
            target = self.file_table

        if current_tab_name == "Sync":
            title = "Select Movie File"
            filter = "All files (*)"
            target = self.sync_file_list
            onam = self.sync_onam

        if current_tab_name == "Wand":
            button = self.sender()
            # paired
            if button == self.ppts_button:
                title = "Select paired points file"
                filter = "Wand points file (*xypts.csv);;All files (*)"
                target = self.ppts
                onam = self.wand_onam
            # unpaired
            if button == self.uppts_button:
                title = "Select unpaired points file"
                filter = "Wand points file (*xypts.csv);;All files (*)"
                target = self.uppts
            # camera profile
            if button == self.wand_cams_button:
                title = "Select camera profile"
                filter = "TXT (*.txt);;CSV (*.csv);;All files (*)"
                target = self.wand_cams
            # reference points
            if button == self.wand_refs_button:
                title = "Select reference points"
                filter = "reference points (*xypts.csv);;All files (*)"
                target = self.wand_refs
        if current_tab_name == "Patterns":
            title = "Select pattern video"
            filter = "All files (*)"
            target = self.patt_file
            onam = self.patt_onam
        if current_tab_name == "Calibrate":
            title = "Select detected patterns pickle"
            filter = "pickle (*.pkl)"
            target = self.cal_file
            onam = self.cal_onam
        if current_tab_name == "Dwarp":
            title = "Select movie file to dewarp"
            filter = "All files (*)"
            target = self.dwarp_file
            onam = self.dwarp_onam
        # Create a file dialog with the specified title and filter
        file_dialog = QtWidgets.QFileDialog(self, title, self.current_directory)
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)  # type: ignore
        file_dialog.setNameFilter(filter)

        # Show the file dialog and get the selected file path
        if file_dialog.exec():
            file_name = file_dialog.selectedFiles()[0]

        if file_name:
            # Update the current directory
            self.current_directory = QtCore.QFileInfo(file_name).absolutePath()
            # Clicker
            if current_tab_name == "Clicker":
                # Check if file already exists in table
                file_exists = False
                for row in range(self.file_table.rowCount()):
                    existing_item = self.file_table.item(row, 0)
                    if existing_item:
                        # Compare against the full path stored in UserRole
                        existing_full_path = existing_item.data(QtCore.Qt.UserRole)
                        if existing_full_path == file_name:
                            file_exists = True
                            break
                
                if not file_exists:
                    print(f"adding {file_name}")
                    # Get offset from user
                    if self.file_table.rowCount() != 0:
                        offset, ok = QtWidgets.QInputDialog.getInt(
                            self, "Enter Offset", "Frame offset: ", value=0
                        )
                    else:
                        offset = 0

                    try:
                        offset_int = int(offset)
                    except ValueError:
                        QtWidgets.QMessageBox.warning(None, "Error", "Frame offset must be an integer")  # type: ignore
                        return

                    # Add new row to table
                    row_position = self.file_table.rowCount()
                    self.file_table.insertRow(row_position)
                    
                    # Add filename item with formatted display path
                    display_path = self.format_file_path_for_display(file_name)
                    filename_item = QtWidgets.QTableWidgetItem(display_path)
                    filename_item.setFlags(filename_item.flags() & ~QtCore.Qt.ItemIsEditable)  # Make read-only
                    # Store the full path as tooltip and user data for later retrieval
                    filename_item.setToolTip(file_name)
                    filename_item.setData(QtCore.Qt.UserRole, file_name)
                    self.file_table.setItem(row_position, 0, filename_item)
                    
                    # Add offset item
                    offset_item = QtWidgets.QTableWidgetItem(str(offset_int))
                    self.file_table.setItem(row_position, 1, offset_item)
                else:
                    QtWidgets.QMessageBox.warning(
                        None,  # type: ignore
                        "Error",
                        "You cannot click through two of the same movies",
                    )

            # Sync
            if current_tab_name == "Sync":
                if not self.sync_file_list.findItems(file_name, QtCore.Qt.MatchExactly):  # type: ignore
                    print(f"adding {file_name}")
                    self.sync_file_list.addItem(file_name)
                    self.cached[file_name] = (
                        self.id_generator()
                        + "-"
                        + file_name.split("/")[-1].split(".")[0]
                        + ".wav"
                    )
                    onam.setText(target.item(0).text().split(".")[0] + "_offsets.csv")  # type: ignore
                else:
                    QtWidgets.QMessageBox.warning(
                        None,  # type: ignore
                        "Error",
                        "You cannot click through two of the same movies",
                    )

            # Wand
            if current_tab_name == "Wand":
                target.setText(file_name)  # type: ignore
                if button == self.ppts_button and onam.text() == "":
                    onam.setText(self.ppts.text().split(".")[0] + "_cal")

            if current_tab_name == "Calibrate":
                target.setText(file_name)  # type: ignore
                onam.setText(file_name.split(".")[0] + ".csv")

            if current_tab_name == "Dwarp":
                target.setText(file_name)  # type: ignore
                onam.setText(
                    file_name.split(".")[0] + "_dwarped." + file_name.split(".")[1]
                )

            # Pattern
            if current_tab_name == "Patterns":
                target.setText(file_name)  # type: ignore
                try:
                    mov = cv2.VideoCapture(file_name)  # type: ignore
                    dur = float(mov.get(cv2.CAP_PROP_FRAME_COUNT)) / float(mov.get(cv2.CAP_PROP_FPS))  # type: ignore
                    self.patt_start.setText("0.0")
                    self.patt_end.setText(str(dur))
                    self.patt_onam.setText(file_name.split(".")[0] + "_patterns.pkl")
                except:
                    QtWidgets.QMessageBox.warning(
                        None, "Error", "Cannot read selected video"  # type: ignore
                    )

    def format_file_path_for_display(self, file_path, max_length=70):
        """
        Format file path to show filename and truncate beginning if too long.
        For example: /very/long/path/to/video.mp4 becomes ...path/to/video.mp4
        """
        if len(file_path) <= max_length:
            return file_path
        
        # Split the path and always keep the filename
        parts = file_path.split('/')
        filename = parts[-1]
        
        # If just the filename is too long, return it as is
        if len(filename) > max_length:
            return filename
        
        # Build path from the end, keeping as much as possible
        result = filename
        for i in range(len(parts) - 2, -1, -1):
            part = parts[i]
            # Check if adding this part (plus separator and "...") would exceed limit
            test_length = len("..." + part + "/" + result)
            if test_length <= max_length:
                result = part + "/" + result
            else:
                result = "..." + result
                break
        
        return result

    def update_offset(self, row, column):
        """
        Called when a cell in the file table is changed.
        Validates offset values and updates the table.
        """
        if column == 1:  # Only validate offset column
            item = self.file_table.item(row, column)
            if item:
                try:
                    # Try to convert to integer
                    offset_value = int(item.text())
                    # Update the item text to ensure it's properly formatted
                    item.setText(str(offset_value))
                except ValueError:
                    # If conversion fails, reset to 0
                    QtWidgets.QMessageBox.warning(
                        None, "Error", "Frame offset must be an integer. Resetting to 0."
                    )
                    item.setText("0")

    def delete(self):
        """
        Removes videos from various tabs
        """
        # This function is used on several tabs, so get the tab name
        current_tab_name = self.tab_widget.tabText(self.tab_widget.currentIndex())

        # for Clicker
        if current_tab_name == "Clicker":
            # Get the selected row
            current_row = self.file_table.currentRow()

            if current_row >= 0:
                # If a row is selected, delete it
                self.file_table.removeRow(current_row)
            else:
                # If no row is selected, delete the last row in the table
                if self.file_table.rowCount() > 0:
                    self.file_table.removeRow(self.file_table.rowCount() - 1)
        # for Sync
        if current_tab_name == "Sync":
            # Get the selected item
            selected_items = self.sync_file_list.selectedItems()

            if selected_items:
                # If item(s) selected, delete it from list
                for item in selected_items:
                    self.sync_file_list.takeItem(self.sync_file_list.row(item))
                    # check if there is a cached wav, and delete it
                    if os.path.isfile(self.tmps[0] + "/" + self.cached[item]):
                        os.remove(self.tmps[0] + "/" + self.cached[item])
                    del self.cached[item]
            else:
                # If no item is selected, delete the last item in the list
                if self.sync_file_list.count() > 0:
                    item = self.sync_file_list[self.sync_file_list.count() - 1]  # type: ignore
                    if os.path.isfile(self.tmps[0] + "/" + self.cached[item]):
                        os.remove(self.tmps[0] + "/" + self.cached[item])
                    del self.cached[item]
                    self.sync_file_list.takeItem(self.sync_file_list.count() - 1)

    def clear(self):

        # This function is used on several tabs, so get the tab name
        current_tab_name = self.tab_widget.tabText(self.tab_widget.currentIndex())

        # for Clicker
        if current_tab_name == "Clicker":
            self.file_table.setRowCount(0)  # Clear all rows

        # for Sync
        if current_tab_name == "Sync":
            self.sync_file_list.selectAll()
            self.delete()

    # general function to select save location
    def save_loc(self):
        # This function is used on several tabs, so get the tab name
        current_tab_name = self.tab_widget.tabText(self.tab_widget.currentIndex())
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, caption="Select save location", dir=self.current_directory
        )  # , dir = init, filter = filt)
        if filename:
            # Update the current directory
            self.current_directory = QtCore.QFileInfo(filename).absolutePath()
            # for Sync
            if current_tab_name == "Sync":
                self.sync_onam.setText(filename)
            if current_tab_name == "Wand":
                self.wand_onam.setText(filename)
            if current_tab_name == "Patterns":
                self.patt_onam.setText(filename)
            if current_tab_name == "Calibrate":
                self.cal_onam.setText(filename)
            if current_tab_name == "Dwarp":
                self.dwarp_onam.setText(filename)

    ## Clicker Functions
    def load(self):
        """
        Loads config file on clicker window
        """
        options = QtWidgets.QFileDialog.Options()  # type: ignore
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select an Argus clicker config file",
            "",
            "Argus clicker config files (*.yaml)",
            options=options,
        )
        if filename:
            cmd = [sys.executable, os.path.join(RESOURCE_PATH, "scripts/argus-click")]
            args = [f"configload@{filename}"]
            cmd = cmd + args
            if hasattr(sys, "frozen"):
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    startupinfo=None,
                )
            else:
                proc = subprocess.Popen(cmd)

    def clicker_go(self):

        if self.resolution_var.currentText() == "Full":
            res_var = 1
        else:
            res_var = 2

        # Read movies and offsets from the table
        movies = []
        offsets = []
        for row in range(self.file_table.rowCount()):
            filename_item = self.file_table.item(row, 0)
            offset_item = self.file_table.item(row, 1)
            if filename_item and offset_item:
                # Get the full path from UserRole data instead of displayed text
                full_path = filename_item.data(QtCore.Qt.UserRole)
                if full_path:
                    movies.append(str(full_path))
                else:
                    # Fallback to displayed text if UserRole data is not available
                    movies.append(str(filename_item.text()))
                try:
                    offsets.append(int(offset_item.text()))
                except ValueError:
                    offsets.append(0)  # Default to 0 if invalid

        if len(movies) != 0:
            driver = PygletDriver(movies, offsets, res=str(res_var))
            driver.run()
            self.drivers.append(driver)
        else:
            QtWidgets.QMessageBox.warning(
                None, "Error", "No movies to click through!"  # type: ignore
            )
            return

    ## Sync Functions

    # Gets seconds from 'hours:minutes:seconds' string
    def getSec(self, s):
        return 60.0 * float(s)

    def updateCropOptions(self):
        self.start_crop.setEnabled(self.crop.isChecked())
        self.end_crop.setEnabled(self.crop.isChecked())

    def sync_go(self):
        cropArg = ""
        files = [
            str(self.sync_file_list.item(x).text())
            for x in range(self.sync_file_list.count())
        ]
        if len(files) <= 1:
            QtWidgets.QMessageBox.warning(
                None, "Error", "Need at least two videos to sync"  # type: ignore
            )
            return
        for k in range(len(files)):
            try:
                open(files[k])
            except:
                QtWidgets.QMessageBox.warning(
                    None,  # type: ignore
                    "Error",
                    "Could not find one or more of the specified videos",
                )
                return
        if self.crop.isChecked():
            try:
                float(self.start_crop.text())
                float(self.end_crop.text())
            except:
                QtWidgets.QMessageBox.warning(
                    None, "Error", "Start and end time must be floats"  # type: ignore
                )
                return
            for k in range(len(files)):
                cap = cv2.VideoCapture(files[k])  # type: ignore
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore
                dur = length * float(cap.get(cv2.CAP_PROP_FPS))  # type: ignore
                # dur = VideoFileClip(files[k]).duration
                if (
                    self.getSec(self.start_crop.text()) >= dur
                    or self.getSec(self.end_crop.text()) > dur
                ):
                    QtWidgets.QMessageBox.warning(
                        None,  # type: ignore
                        "Error",
                        "Time range does not exist for one or more of the specified videos",
                    )
                    return
                elif self.getSec(self.start_crop.text()) >= self.getSec(
                    self.end_crop.text()
                ):
                    QtWidgets.QMessageBox.warning(
                        None,  # type: ignore
                        "Error",
                        "Start time is further along than end time",
                    )
                    return
            cropArg = "1"

        for k in range(len(files)):
            try:
                self.cached[files[k]]
            except:
                self.cached[files[k]] = (
                    self.id_generator()
                    + "-"
                    + files[k].split("/")[-1].split(".")[0]
                    + ".wav"
                )
        out = list()
        for k in range(len(files)):
            out.append(self.cached[files[k]])

        logBool = self.sync_log.isChecked()

        # check for properly named output file (if it exists) & fix it if appropriate
        of = self.sync_onam.text()
        if of:
            ofs = of.split(".")
            if ofs[-1].lower() != "csv":
                of = of + ".csv"
                self.sync_onam.setText(of)

        file_str = ",".join(files)
        out_str = ",".join(out)

        cmd = [sys.executable, os.path.join(RESOURCE_PATH, "scripts/argus-sync")]
        # Create args list, order is important
        args = [
            file_str,
            "--tmp",
            self.tmps[0],
            "--start",
            self.start_crop.text(),
            "--end",
            self.end_crop.text(),
            "--ofile",
            self.sync_onam.text(),
            "--out",
            out_str,
        ]

        if self.crop.isChecked():
            args = args + ["--crop"]
        cmd = cmd + args
        self.go(cmd, self.sync_onam.text(), logBool)

    # Graph the wave files
    def sync_show(self):
        files = [
            str(self.sync_file_list.item(x).text())
            for x in range(self.sync_file_list.count())
        ]
        for k in range(len(files)):
            try:
                self.cached[files[k]]
            except:
                self.cached[files[k]] = (
                    self.id_generator()
                    + "-"
                    + files[k].split("/")[-1].split(".")[0]
                    + ".wav"
                )
        out = list()
        for k in range(len(files)):
            out.append(self.cached[files[k]])

        cmd = [sys.executable, os.path.join(RESOURCE_PATH, "scripts/argus-show")]
        if isinstance(files, str):
            args = [self.tmps[0], files] + out
        else:
            args = [self.tmps[0]] + list(files) + out

        cmd = cmd + args

        self.go(cmd)

    def id_generator(self, size=12, chars=string.ascii_uppercase + string.digits):
        return "".join(random.choice(chars) for _ in range(size))

    ## Wand specific
    def updateFreqBoxState(self):
        if self.wand_reftype.currentText() == "Gravity":
            self.wand_freq.setEnabled(True)
            self.wand_freq_label.setEnabled(True)
        else:
            self.wand_freq.setEnabled(False)
            self.wand_freq_label.setEnabled(False)

    def update_outliers_checkbox_state(self):
        """Update the outliers checkbox state based on display checkbox."""
        if self.wand_display.isChecked():
            # When display is enabled, automatically enable outliers reporting
            # and gray it out since GUI handles outliers interactively
            self.wand_outliers.setChecked(True)
            self.wand_outliers.setEnabled(False)
            self.wand_outliers.setToolTip("Outlier processing is handled through the GUI when display is enabled")
        else:
            # When display is disabled, allow user to control outliers checkbox
            self.wand_outliers.setEnabled(True)
            self.wand_outliers.setToolTip("Process outlier point with option to remove")

    def wand_go(self):
        cmd = [sys.executable, os.path.join(RESOURCE_PATH, "scripts/argus-wand")]
        tmp = tempfile.mkdtemp()
        write_bool = False

        args = [
            self.wand_cams.text(),
            "--intrinsics_opt",
            self.intModeDict[self.wand_intrinsics.currentText()],
            "--distortion_opt",
            self.disModeDict[self.wand_dist.currentText()],
            self.wand_onam.text(),
            "--paired_points",
            self.ppts.text(),
            "--unpaired_points",
            self.uppts.text(),
            "--scale",
            self.wand_scale.text(),
            "--reference_points",
            self.wand_refs.text(),
            "--reference_type",
            self.wand_reftype.currentText(),
            "--recording_frequency",
            self.wand_freq.text(),
            "--tmp",
            tmp,
        ]

        if self.wand_log.isChecked():
            write_bool = True

        if self.wand_display.isChecked():
            args = args + ["--graph"]

        if self.wand_outliers.isChecked():
            args = args + ["--outliers"]

        if self.wand_outputProf.isChecked():
            args = args + ["--output_camera_profiles"]

        if self.wand_chooseRef.isChecked():
            args = args + ["--choose_reference"]

        cmd = cmd + args
        print(cmd)
        self.go(cmd, self.wand_onam.text(), wlog=write_bool)

    def pattern_go(self):
        # check and fix the output filename
        if self.patt_onam.text() == "":  # no name at all
            self.patt_onam.setText(self.patt_file.text().split(".")[0] + "pkl")
        of = self.patt_onam.text()
        if of:  # check extension
            ofs = of.split(".")
            if ofs[-1].lower() != "pkl":
                of = of + ".pkl"
                self.patt_onam.setText(of)

        cmd = [sys.executable, os.path.join(RESOURCE_PATH, "scripts/argus-patterns")]
        writeBool = False

        if self.patt_log.isChecked():
            writeBool = True
        args = [
            self.patt_file.text(),
            self.patt_onam.text(),
            "--rows",
            self.patt_rows.text(),
            "--cols",
            self.patt_cols.text(),
            "--spacing",
            self.patt_space.text(),
            "--start",
            self.patt_start.text(),
            "--stop",
            self.patt_end.text(),
        ]
        if self.patt_dots.isChecked():
            args = args + ["--dots"]
        if self.patt_display.isChecked():
            args = args + ["--display"]
        cmd = cmd + args

        self.go(cmd, self.patt_onam.text(), writeBool)

    # Calibrate
    def updateCalOptions(self):
        if self.cal_dist_model.currentText() == "Pinhole model":
            self.cal_dist_option.setEnabled(True)
        else:
            self.cal_dist_option.setEnabled(False)

    def calibrate_go(self):
        if self.cal_file.text().split(".")[-1].lower() != "pkl":
            QtWidgets.QMessageBox.warning(
                None, "Error", "Input file must be a Pickle"  # type: ignore
            )
            return
        try:
            int(self.cal_replicates.text())
            int(self.cal_patterns.text())
        except:
            QtWidgets.QMessageBox.warning(
                None,  # type: ignore
                "Error",
                "Number of samples and replicates must both be integers",
            )
            return
        if self.cal_onam.text() == "":
            self.cal_onam.setText(self.cal_file.text()[:-3] + "csv")

        if self.cal_onam.text().split(".")[-1].lower() != "csv":
            self.cal_onam.setText(self.cal_onam.text() + "_calibrations.csv")

        cmd = [sys.executable, os.path.join(RESOURCE_PATH, "scripts/argus-calibrate")]
        writeBool = False
        args = [
            self.cal_file.text(),
            self.cal_onam.text(),
            "--replicates",
            self.cal_replicates.text(),
            "--patterns",
            self.cal_patterns.text(),
        ]
        if self.cal_dist_option.currentText() == "Optimize k1, k2, and k3":
            args = args + ["--k3"]
        elif (
            self.cal_dist_option.currentText() == "Optimize all distortion coefficients"
        ):
            args = args + ["--tangential"]

        if self.cal_dist_model.currentText() == "Omnidirectional model":
            args = args + ["--omnidirectional"]
        if self.cal_log.isChecked():
            writeBool = True

        if self.cal_inv.isChecked():
            args = args + ["--inverted"]

        cmd = cmd + args

        self.go(cmd, self.cal_onam.text(), writeBool)

    # Dwarp
    def disableEntries(self):
        self.dwarp_crop.setChecked(False)
        self.dwarp_crop.setCheckable(False)
        self.dwarp_fl.setEnabled(False)
        self.dwarp_cx.setEnabled(False)
        self.dwarp_cy.setEnabled(False)
        self.dwarp_k1.setEnabled(False)
        self.dwarp_k2.setEnabled(False)
        self.dwarp_k3.setEnabled(False)
        self.dwarp_t1.setEnabled(False)
        self.dwarp_t2.setEnabled(False)
        self.dwarp_xi.setEnabled(False)

    def enableEntries(self):
        self.dwarp_crop.setChecked(True)
        self.dwarp_crop.setCheckable(True)
        self.dwarp_fl.setEnabled(True)
        self.dwarp_cx.setEnabled(True)
        self.dwarp_cy.setEnabled(True)
        self.dwarp_k1.setEnabled(True)
        self.dwarp_k2.setEnabled(True)
        self.dwarp_k3.setEnabled(True)
        self.dwarp_t1.setEnabled(True)
        self.dwarp_t2.setEnabled(True)
        self.dwarp_xi.setEnabled(True)

    def updateCam(self):
        self.dwarp_modes.clear()
        model = self.dwarp_models.currentText()
        self.dwarp_modes.addItems(list(self.models[model].keys()))

    # Define function for filling the entry fields for the undistortion coefficients and other relevant numbers
    def calibParse(self):
        model = self.dwarp_models.currentText()
        mode = self.dwarp_modes.currentText()
        if len(mode) == 0:
            return
        vals = self.models[model][mode]
        if "(Fisheye)" in mode:
            # no entries for Scaramuzzas Fisheye
            self.disableEntries()
        elif "(CMei)" in mode:
            # enable everythign except k3
            self.enableEntries()
            self.dwarp_k1.setText(vals[6])
            self.dwarp_k2.setText(vals[7])
            self.dwarp_t1.setText(vals[8])
            self.dwarp_t2.setText(vals[9])
            self.dwarp_width = int(vals[1])
            self.dwarp_height = int(vals[2])
            self.dwarp_xi.setText(vals[10])
            self.dwarp_k3.setText("0.0")
            self.dwarp_k3.setEnabled(False)
            self.dwarp_fl.setText(vals[0])
            self.dwarp_cx.setText(vals[3])
            self.dwarp_cy.setText(vals[4])
        else:
            self.enableEntries()
            self.dwarp_k1.setText(vals[6])
            self.dwarp_k2.setText(vals[7])
            self.dwarp_t1.setText(vals[8])
            self.dwarp_t2.setText(vals[9])
            self.dwarp_width = int(vals[1])
            self.dwarp_height = int(vals[2])
            self.dwarp_xi.setText("1.0")
            self.dwarp_xi.setEnabled(False)
            self.dwarp_k3.setText(vals[10])
            self.dwarp_fl.setText(vals[0])
            self.dwarp_cx.setText(vals[3])
            self.dwarp_cy.setText(vals[4])

    def getCoefficients(self):
        try:
            if not "(CMei)" in self.dwarp_modes.currentText():
                co = [
                    self.dwarp_fl.text(),
                    self.dwarp_cx.text(),
                    self.dwarp_cy.text(),
                    self.dwarp_k1.text(),
                    self.dwarp_k2.text(),
                    self.dwarp_t1.text(),
                    self.dwarp_t2.text(),
                    self.dwarp_k3.text(),
                ]
                ret = ",".join(co)
                return ret
            else:
                co = [
                    self.dwarp_fl.text(),
                    str(self.width),
                    str(self.height),
                    self.dwarp_cx.text(),
                    self.dwarp_cy.text(),
                    self.dwarp_k1.text(),
                    self.dwarp_k2.text(),
                    self.dwarp_t1.text(),
                    self.dwarp_t2.text(),
                    self.dwarp_xi.text(),
                ]
                ret = ",".join(co)
                return ret
        except:
            QtWidgets.QMessageBox.warning(None, "Error", "Undistortion coefficients must all be floats")  # type: ignore
            return

    def omniParse(self):
        return ",".join(
            self.models[self.dwarp_models.currentText()][self.dwarp_modes.currentText()]
        )

    def dwarp_go(self):
        of = self.dwarp_onam.text()
        # check for properly named output file (if it exists) & fix it if appropriate
        if of != "":
            ofs = of.split(".")
            if ofs[-1].lower() != "mp4":
                of = of + ".mp4"
                self.dwarp_onam.setText(of)

        tmpName = ""
        # Extra bools and a string for passing temp dir, write option and display option to the undistorter object
        if self.getCoefficients():
            cmd = [sys.executable, os.path.join(RESOURCE_PATH, "scripts/argus-dwarp")]

            # build basic command line arguments
            args = [
                self.dwarp_file.text(),
                "--frameint",
                self.dwarp_int.text(),
                "--crf",
                self.dwarp_qual.text(),
            ]

            if self.dwarp_opts_wd.isChecked() or self.dwarp_opts_wo.isChecked():
                args = args + ["--ofile", self.dwarp_onam.text()]
                tmpName = tempfile.mkdtemp()
                args = args + ["--write", "--tmp", tmpName]

            if (
                "(Fisheye)" in self.dwarp_modes.currentText()
            ):  # assume it is not a fisheye calibration unless it says that it is
                omni = self.omniParse()
                args = args + ["--omni", omni]
            else:
                omni = ""
                args = args + ["--coefficients", self.getCoefficients()]

            if self.dwarp_opts_wd.isChecked() or self.dwarp_opts_do.isChecked():
                args = args + ["--disp"]

            if self.dwarp_crop.isChecked():
                args = args + ["--crop"]

            if self.dwarp_copy.isChecked():
                args = args + ["--copy"]

            if "(CMei)" in self.dwarp_modes.currentText():
                args = args + ["--cmei"]

            cmd = cmd + args
        else:
            return

        if self.dwarp_log.isChecked():
            logBool = True
        else:
            logBool = False

        self.go(cmd, self.dwarp_onam.text(), logBool)

    # main command caller used by all but clicker
    def go(self, cmd, opath="", wlog=False, mode="DEBUG"):
        print("Running the following command: ")
        print(" ".join(cmd))
        if opath is None:
            wlog = False
        self.wLog = wlog
        if self.wLog:
            self.logpath = os.path.join(
                os.path.dirname(opath), f'Log--{time.strftime("%Y-%m-%d-%H-%M")}.txt'
            )
            self.fo = open(self.logpath, "wb")
        else:
            self.fo = None

        self.cancel_button.setEnabled(True)
        self.worker = WorkerThread(cmd, self, opath, wlog)
        self.worker.output.connect(self.on_output)
        self.worker.complete.connect(self.on_complete)
        self.worker.start()

        # proc = subprocess.Popen(rcmd, stdout=subprocess.PIPE, shell=True, startupinfo=startupinfo)
        # self.pids.append(proc.pid)

    def on_output(self, text):
        bad_phrases = [
            "iters",
            "it/s",
            "InsecureRequestWarning",
            "axes3d.py",
            "Python",
            "_createMenuRef",
            "0x",
            "ApplePersistenceIgnoreState",
            "self._edgecolors",
            "objc",
            "warnings.warn",
            "UserWarning",
        ]

        if not any(bad_phrase in text for bad_phrase in bad_phrases):
            self.logwindow.append(text)
            if self.fo:
                text = "\n" + text
                self.fo.write(text.encode("utf-8"))

    def on_complete(self):
        self.on_output("Process complete!")
        self.kill_pids()
        self.cancel_button.setEnabled(False)
        if self.fo:
            self.fo.close()

    def on_cancel(self):
        self.worker.stop()
        self.on_output("Process cancelled!")
        if self.fo:
            self.fo.close()


class WorkerThread(QtCore.QThread):
    output = QtCore.Signal(str)
    complete = QtCore.Signal()

    def __init__(self, cmd, logWindow, opath, wLog):
        super().__init__()
        self.cmd = cmd
        self.logWindow = logWindow
        self.process = None

    def start(self):
        self.process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            text=True,
        )
        self.logWindow.pids.append(self.process.pid)

        while True:
            line = self.process.stdout.readline()  # type: ignore
            if self.process.poll() is not None:
                QtWidgets.QApplication.processEvents()
                self.complete.emit()
                break
            if line:
                self.output.emit(line.strip())

            QtWidgets.QApplication.processEvents()

    def stop(self):
        if self.process:
            self.process.terminate()


# Makes a subprocess with Pyglet windows for all camera views
class PygletDriver:
    def __init__(self, movies, offsets, res=None):
        self.movies = movies
        self.offsets = offsets
        self.res = res
        starts = []
        for k in range(len(offsets)):
            starts.append(np.max(offsets) - offsets[k])
        self.processes = []
        self.end = int(cv2.VideoCapture(movies[0]).get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore

    def run(self):
        movie_string = ""
        for index, movie in enumerate(self.movies):
            if index != len(self.movies) - 1:
                movie_string = movie_string + movie + "@"
            else:
                movie_string = movie_string + movie

        offset_string = ""
        for k in range(len(self.offsets)):
            if k != len(self.offsets) - 1:
                offset_string = offset_string + str(self.offsets[k]) + "@"
            else:
                offset_string = offset_string + str(self.offsets[k])

        cmd = [sys.executable, os.path.join(RESOURCE_PATH, "scripts/argus-click")]
        args = [movie_string, str(self.end), offset_string, self.res]
        cmd = cmd + args
        if hasattr(sys, "frozen"):
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                startupinfo=None,
            )
        else:
            proc = subprocess.Popen(cmd)

        self.processes.append(proc)

    def kill(self):
        for proc in self.processes:
            proc.kill()


class ClickerProject:
    @staticmethod
    def create(
        proj_path,
        video_paths,
        points,
        resolution,
        last_frame,
        offsets,
        dlt_coefficents=None,
        camera_profile=None,
        settings=None,
        window_positions=None,
    ):
        project = [
            {"videos": video_paths},
            {"points": points},
            {"resolution": resolution},
            {"last_frame": int(last_frame)},
            {"offsets": offsets},
            {"dlt_coefficents": dlt_coefficents},
            {"camera_profile": camera_profile},
            {"settings": settings},
        ]
        with open(f"{proj_path}-config.yaml", "w") as f:
            f.write(yaml.dump(project))

def main():
    """Main entry point for the argus_gui application."""
    app = QtWidgets.QApplication()
    window = MainWindow(app)
    window.show()
    sys.exit(app.exec())

def startArgus():
    """Legacy function for backwards compatibility."""
    main()
    
if __name__ == "__main__":
    main()
