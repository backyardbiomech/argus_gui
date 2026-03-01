#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLC Clicker - Integration of Argus clicker GUI with DeepLabCut labeling workflow

This module allows users to:
1. Load a DLC labeled-data folder
2. View the associated video in argus-click
3. Navigate between frames that need labeling
4. Edit/add/remove labels
5. Save back to DLC format (.h5 and .csv)

"""

from __future__ import absolute_import
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PySide6 import QtWidgets, QtCore


class DLCLabelFolder:
    """Manages a DLC labeled-data folder and its contents"""
    
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        self.video_name = self.folder_path.name
        self.collected_data_path = None
        self.machine_labels_paths = []
        self.video_path = None
        
        # Data structures
        self.collected_data = None  # DataFrame with manual labels
        self.machine_labels = None  # DataFrame with machine labels (highest iter)
        self.combined_data = None  # Combined data for Argus to use
        
        # Metadata
        self.scorer = None
        self.individuals = []
        self.bodyparts = []
        self.label_frames = []  # List of frame numbers to label
        self.frame_to_original_index = {}  # maps frame_num -> original index entry (for saving)
        self.is_multi_animal = False
        
        # Load the folder structure
        self._scan_folder()
        
    def _get_actual_frame_filenames(self):
        """Scan the folder for PNG files and return a dict mapping frame_num -> actual_filename.

        This is the authoritative source for image filenames – it reflects what is
        actually on disk rather than what may have been stored in a previous h5/csv
        (which could have used a different zero-padding width).
        """
        import re
        frame_map = {}
        for png_file in self.folder_path.glob("*.png"):
            m = re.search(r'(\d+)', png_file.stem)
            if m:
                frame_num = int(m.group(1))
                frame_map[frame_num] = png_file.name
        return frame_map

    def _scan_folder(self):
        """Scan the folder for h5 files and extracted frames"""
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")
        
        # Find CollectedData file
        collected_data_files = sorted(self.folder_path.glob("CollectedData*.h5"), key=lambda p: p.name)
        if not collected_data_files:
            raise FileNotFoundError(f"No CollectedData*.h5 file found in {self.folder_path}")
        if len(collected_data_files) == 1:
            self.collected_data_path = collected_data_files[0]
        else:
            # Multiple CollectedData files – ask the user which one to use
            app = QtWidgets.QApplication.instance()
            if app is None:
                app = QtWidgets.QApplication([])
            items = [p.name for p in collected_data_files]
            chosen, ok = QtWidgets.QInputDialog.getItem(
                None,
                "Multiple CollectedData files found",
                "Select the CollectedData file to use:",
                items,
                0,
                False,
            )
            if ok and chosen:
                self.collected_data_path = self.folder_path / chosen
            else:
                # Default to the first file if the user cancels
                self.collected_data_path = collected_data_files[0]
        
        # Find machine labels files
        machine_labels_files = list(self.folder_path.glob("machinelabels-iter*.h5"))
        if machine_labels_files:
            # Sort by iteration number
            def get_iter_num(path):
                stem = path.stem  # e.g., 'machinelabels-iter22'
                try:
                    return int(stem.split('iter')[-1])
                except:
                    return 0
            self.machine_labels_paths = sorted(machine_labels_files, key=get_iter_num)
    
    def load_data(self):
        """Load the h5 files and prepare data for Argus"""
        # Load CollectedData
        self.collected_data = pd.read_hdf(self.collected_data_path)
        
        # Get metadata from the structure
        self.scorer = self.collected_data.columns.get_level_values('scorer')[0]
        
        # Find bodyparts level name (could be 'bodypart' or 'bodyparts')
        bodypart_level = None
        for level_name in self.collected_data.columns.names:
            if 'bodypart' in level_name.lower():
                bodypart_level = level_name
                break
        
        if not bodypart_level:
            raise ValueError("No bodypart level found in DLC data")
        
        self.bodyparts = self.collected_data.columns.get_level_values(bodypart_level).unique().tolist()
        
        # Check if multi-animal
        if 'individuals' in self.collected_data.columns.names:
            self.is_multi_animal = True
            # Find the actual individual level name
            individual_level = None
            for level_name in self.collected_data.columns.names:
                if level_name and isinstance(level_name, str) and level_name.lower() in ['individual', 'individuals', 'animal']:
                    individual_level = level_name
                    break
            if individual_level:
                self.individuals = self.collected_data.columns.get_level_values(individual_level).unique().tolist()
        
        # Extract frame numbers from index
        # Index format may be a MultiIndex tuple ('labeled-data', 'video_name', 'img123.png')
        # OR a plain path string 'labeled-data\\video_name\\img123.png'
        import re
        self.label_frames = []
        self.frame_to_original_index = {}
        for idx in self.collected_data.index:
            # Extract frame number from image filename (e.g., 'img123.png' -> 123)
            if isinstance(idx, tuple):
                img_name = idx[2]
            else:
                # Flat path string – grab only the basename to avoid matching
                # digits in folder/video names (e.g. 'cam1', '23-Feb-21')
                img_name = str(idx).replace('\\', '/').split('/')[-1]
            try:
                match = re.search(r'(\d+)', img_name)
                if match:
                    frame_num = int(match.group(1))
                    self.label_frames.append(frame_num)
                    self.frame_to_original_index[frame_num] = idx
            except:
                pass
        
        # Sort frame numbers
        self.label_frames = sorted(set(self.label_frames))
        
        # Load machine labels if available
        if self.machine_labels_paths:
            # Load the highest iteration
            latest_ml_path = self.machine_labels_paths[-1]
            self.machine_labels = pd.read_hdf(latest_ml_path)
            
            # Get frames from machine labels
            ml_frames = []
            for idx in self.machine_labels.index:
                if isinstance(idx, tuple):
                    img_name = idx[-1]
                else:
                    img_name = str(idx).replace('\\', '/').split('/')[-1]
                try:
                    import re
                    match = re.search(r'(\d+)', img_name)
                    if match:
                        frame_num = int(match.group(1))
                        ml_frames.append(frame_num)
                except:
                    pass
            
            ml_frames = set(ml_frames)
            collected_frames = set(self.label_frames)
            
            # Add machine label frames that aren't already in collected data
            new_frames = ml_frames - collected_frames
            if new_frames:
                print(f"Adding {len(new_frames)} frames from machine labels")
                self.label_frames.extend(sorted(new_frames))
                self.label_frames = sorted(set(self.label_frames))
    
    def find_video(self, prompt_user=False):
        """
        Find the video file associated with this labeling folder.
        First checks ../videos/, then prompts user if not found and prompt_user=True
        """
        # Try ../videos/ directory
        videos_dir = self.folder_path.parent.parent / 'videos'
        if videos_dir.exists():
            # Look for video with matching name (without extension)
            video_exts = ['.mov', '.mp4', '.avi', '.MOV', '.MP4', '.AVI']
            for ext in video_exts:
                video_path = videos_dir / f"{self.video_name}{ext}"
                if video_path.exists():
                    self.video_path = str(video_path)
                    return self.video_path
        
        # If not found and prompt_user is True, ask user
        if prompt_user:
            app = QtWidgets.QApplication.instance()
            if app is None:
                app = QtWidgets.QApplication([])
            
            file_dialog = QtWidgets.QFileDialog()
            video_path, _ = file_dialog.getOpenFileName(
                None,
                f"Select video file for {self.video_name}",
                str(self.folder_path.parent.parent),
                "Video files (*.mov *.mp4 *.avi *.MOV *.MP4 *.AVI);;All files (*.*)"
            )
            
            if video_path:
                self.video_path = video_path
                return self.video_path
        
        return None
    
    def get_tracks_for_argus(self):
        """
        Generate track names for Argus.
        Returns list of track names in format:
        - Single animal: ['bodypart1', 'bodypart2', ...]
        - Multi-animal: ['ind1_bodypart1', 'ind1_bodypart2', 'ind2_bodypart1', ...]
        """
        tracks = []
        if self.is_multi_animal:
            for ind in self.individuals:
                for bp in self.bodyparts:
                    tracks.append(f"{ind}_{bp}")
        else:
            tracks = self.bodyparts.copy()
        return tracks
    
    def convert_to_argus_format(self, video_height):
        """
        Convert DLC data to Argus format (sparse matrix with frame x coordinates)
        Argus expects: columns are [track1_cam1_x, track1_cam1_y, track2_cam1_x, ...]
        
        Args:
            video_height: Height of video (needed to flip y-coordinates from DLC to Argus)
        
        Returns:
            numpy array of shape (max_frame, n_tracks * 2)
        """
        tracks = self.get_tracks_for_argus()
        
        # Determine max frame from label_frames
        max_frame = max(self.label_frames) if self.label_frames else 0
        
        # Initialize array (frames x (tracks * 2 for x,y))
        data = np.zeros((max_frame + 1, len(tracks) * 2))
        
        # Process collected data
        import re
        if self.collected_data is not None:
            for idx, row in self.collected_data.iterrows():
                # Extract frame number from image filename only (not from folder names)
                if isinstance(idx, tuple):
                    img_name = idx[2]
                else:
                    img_name = str(idx).replace('\\', '/').split('/')[-1]
                match = re.search(r'(\d+)', img_name)
                if not match:
                    continue
                frame_num = int(match.group(1))
                
                if frame_num > max_frame:
                    continue
                
                # Get data for each track
                for track_idx, track in enumerate(tracks):
                    col_offset = track_idx * 2
                    
                    if self.is_multi_animal:
                        # Split track name back to individual and bodypart
                        ind, bp = track.split('_', 1)
                        try:
                            x = row[(self.scorer, ind, bp, 'x')]
                            y = row[(self.scorer, ind, bp, 'y')]
                        except:
                            continue
                    else:
                        try:
                            x = row[(self.scorer, track, 'x')]
                            y = row[(self.scorer, track, 'y')]
                        except:
                            continue
                    
                    # Check if valid
                    if not pd.isna(x) and not pd.isna(y):
                        data[frame_num, col_offset] = x
                        # Flip y-coordinate (DLC origin is top-left, Argus is bottom-left)
                        data[frame_num, col_offset + 1] = video_height - y
        
        # Add machine labels for frames not in collected data
        if self.machine_labels is not None:
            collected_frames = set()
            for idx in self.collected_data.index:
                if isinstance(idx, tuple):
                    img_name = idx[-1]
                else:
                    img_name = str(idx).replace('\\', '/').split('/')[-1]
                import re
                match = re.search(r'(\d+)', img_name)
                if match:
                    collected_frames.add(int(match.group(1)))
            
            ml_scorer = self.machine_labels.columns.get_level_values('scorer')[0]
            
            for idx, row in self.machine_labels.iterrows():
                # Extract frame number from image filename only
                if isinstance(idx, tuple):
                    img_name = idx[2]
                else:
                    img_name = str(idx).replace('\\', '/').split('/')[-1]
                match = re.search(r'(\d+)', img_name)
                if not match:
                    continue
                frame_num = int(match.group(1))
                
                # Skip if already in collected data
                if frame_num in collected_frames:
                    continue
                
                if frame_num > max_frame:
                    continue
                
                # Get data for each track
                for track_idx, track in enumerate(tracks):
                    col_offset = track_idx * 2
                    
                    if self.is_multi_animal:
                        ind, bp = track.split('_', 1)
                        try:
                            x = row[(ml_scorer, ind, bp, 'x')]
                            y = row[(ml_scorer, ind, bp, 'y')]
                            likelihood = row[(ml_scorer, ind, bp, 'likelihood')]
                        except:
                            continue
                    else:
                        try:
                            x = row[(ml_scorer, track, 'x')]
                            y = row[(ml_scorer, track, 'y')]
                            likelihood = row[(ml_scorer, track, 'likelihood')]
                        except:
                            continue
                    
                    # Check likelihood threshold (could make this configurable)
                    if not pd.isna(x) and not pd.isna(y) and likelihood >= 0.9:
                        data[frame_num, col_offset] = x
                        data[frame_num, col_offset + 1] = video_height - y
        
        return data
    
    def save_to_dlc_format(self, argus_data, video_height):
        """
        Save Argus data back to DLC format (both .h5 and .csv)
        
        Args:
            argus_data: numpy array from Argus (frames x (tracks * 2))
            video_height: Height of video (to flip y-coordinates back)
        """
        tracks = self.get_tracks_for_argus()
        
        # Create multi-index columns
        if self.is_multi_animal:
            individual_level = None
            for level_name in self.collected_data.columns.names:
                if level_name and isinstance(level_name, str) and level_name.lower() in ['individual', 'individuals', 'animal']:
                    individual_level = level_name
                    break
            if not individual_level:
                individual_level = 'individuals'
            
            bodypart_level = None
            for level_name in self.collected_data.columns.names:
                if 'bodypart' in level_name.lower():
                    bodypart_level = level_name
                    break
            if not bodypart_level:
                bodypart_level = 'bodyparts'
            
            columns = []
            for track in tracks:
                ind, bp = track.split('_', 1)
                columns.append((self.scorer, ind, bp, 'x'))
                columns.append((self.scorer, ind, bp, 'y'))
            
            multi_index = pd.MultiIndex.from_tuples(
                columns,
                names=['scorer', individual_level, bodypart_level, 'coords']
            )
        else:
            bodypart_level = None
            for level_name in self.collected_data.columns.names:
                if 'bodypart' in level_name.lower():
                    bodypart_level = level_name
                    break
            if not bodypart_level:
                bodypart_level = 'bodyparts'
            
            columns = []
            for track in tracks:
                columns.append((self.scorer, track, 'x'))
                columns.append((self.scorer, track, 'y'))
            
            multi_index = pd.MultiIndex.from_tuples(
                columns,
                names=['scorer', bodypart_level, 'coords']
            )
        
        # Create rows with multi-index
        index_data = []
        data_rows = []
        
        # Determine the index format used in the original file (MultiIndex tuple vs flat path string)
        original_is_tuple = (self.collected_data is not None and
                             len(self.collected_data.index) > 0 and
                             isinstance(self.collected_data.index[0], tuple))
        # Detect path separator used in original flat-string index
        if not original_is_tuple and self.collected_data is not None and len(self.collected_data.index) > 0:
            first_idx_str = str(self.collected_data.index[0])
            original_sep = '\\' if '\\' in first_idx_str else '/'
        else:
            original_sep = '/'

        # Build an authoritative map of frame_num -> actual PNG filename from disk.
        # This ensures saved index entries always match what is actually on disk,
        # regardless of the zero-padding width used by a previous save.
        actual_filenames = self._get_actual_frame_filenames()

        # Infer fallback padding width from the existing index (used only when
        # no PNG is found on disk for a given frame number).
        _fallback_pad = 4
        if self.collected_data is not None and len(self.collected_data.index) > 0:
            import re as _re
            first_idx = self.collected_data.index[0]
            sample_img = first_idx[-1] if isinstance(first_idx, tuple) else str(first_idx).replace('\\', '/').split('/')[-1]
            _m = _re.search(r'(\d+)', sample_img)
            if _m:
                _fallback_pad = len(_m.group(1))

        for frame_num in self.label_frames:
            if frame_num >= len(argus_data):
                continue

            # Determine the correct image filename, preferring actual files on disk.
            if frame_num in actual_filenames:
                # Authoritative: use the filename that exists on disk.
                img_name = actual_filenames[frame_num]
            elif frame_num in self.frame_to_original_index:
                # Fall back to what was stored in the original h5.
                orig = self.frame_to_original_index[frame_num]
                img_name = orig[-1] if isinstance(orig, tuple) else str(orig).replace('\\', '/').split('/')[-1]
            else:
                # New frame with no PNG on disk yet – generate a name that
                # matches the padding style of the existing files.
                img_name = f"img{frame_num:0{_fallback_pad}d}.png"

            # Construct the index entry in the format used by the original file.
            if original_is_tuple:
                if frame_num in self.frame_to_original_index:
                    orig = self.frame_to_original_index[frame_num]
                    idx_parts = list(orig)
                    idx_parts[-1] = img_name
                    index_entry = tuple(idx_parts)
                else:
                    index_entry = ('labeled-data', self.video_name, img_name)
            else:
                index_entry = f"labeled-data{original_sep}{self.video_name}{original_sep}{img_name}"

            index_data.append(index_entry)
            
            # Create data row
            row_data = []
            for track_idx, track in enumerate(tracks):
                col_offset = track_idx * 2
                x = argus_data[frame_num, col_offset]
                y = argus_data[frame_num, col_offset + 1]
                
                # Flip y back to DLC format (top-left origin)
                if y != 0:
                    y = video_height - y
                
                # Convert 0,0 to NaN
                if x == 0 and y == 0:
                    x = np.nan
                    y = np.nan
                
                row_data.extend([x, y])
            
            data_rows.append(row_data)
        
        # Create DataFrame – preserve original index format
        if original_is_tuple:
            row_index = pd.MultiIndex.from_tuples(index_data)
        else:
            row_index = pd.Index(index_data)
        df = pd.DataFrame(data_rows, index=row_index, columns=multi_index)
        
        # Save to h5
        h5_path = self.collected_data_path
        df.to_hdf(h5_path, key='df_with_missing', mode='w')
        
        # Save to csv
        csv_path = h5_path.with_suffix('.csv')
        df.to_csv(csv_path)
        
        print(f"Saved DLC data to {h5_path} and {csv_path}")


def select_dlc_label_folder():
    """Open dialog to select a DLC labeled-data folder"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    
    folder_dialog = QtWidgets.QFileDialog()
    folder_path = folder_dialog.getExistingDirectory(
        None,
        "Select DLC labeled-data folder",
        "",
        QtWidgets.QFileDialog.ShowDirsOnly
    )
    
    if folder_path:
        return DLCLabelFolder(folder_path)
    return None


if __name__ == "__main__":
    # Test the module
    folder = select_dlc_label_folder()
    if folder:
        print(f"Loaded folder: {folder.folder_path}")
        folder.load_data()
        print(f"Scorer: {folder.scorer}")
        print(f"Multi-animal: {folder.is_multi_animal}")
        print(f"Individuals: {folder.individuals}")
        print(f"Bodyparts: {folder.bodyparts}")
        print(f"Label frames: {folder.label_frames}")
        print(f"Tracks for Argus: {folder.get_tracks_for_argus()}")
        
        video = folder.find_video(prompt_user=True)
        if video:
            print(f"Video: {video}")
