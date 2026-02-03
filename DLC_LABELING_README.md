# DeepLabCut Labeling Integration for Argus

This feature integrates Argus clicker GUI with DeepLabCut's labeling workflow, allowing you to use Argus to label frames selected by DeepLabCut.

## Features

- Load DLC `labeled-data` folders directly
- View the full video instead of extracted frame images
- Navigate between label frames using keyboard shortcuts
- Visual indicator showing when you're on a frame that needs labeling
- Automatically imports machine labels (if available)
- Saves back to DLC format (both `.h5` and `.csv`)
- Supports both single-animal and multi-animal DLC projects

## How to Use

### From the Main Argus GUI

1. Launch Argus (`python -m argus_gui` or use the Argus launcher)
2. Go to the **Clicker** tab
3. Click the **"DLC Labeling"** button
4. Select a DLC `labeled-data` folder (e.g., `.../labeled-data/cam1_08-Sep-25_21-34-16`)
5. If the video isn't found automatically in `../videos/`, you'll be prompted to locate it
6. The argus-click window will open with your video loaded

### From Command Line

You can also launch DLC labeling mode directly:

```bash
python argus_gui/resources/scripts/argus-click --dlc-label
```

Or pass the folder path directly to skip the file dialog:

```bash
python argus_gui/resources/scripts/argus-click --dlc-label "path/to/labeled-data/video_name"
```

## Controls

### Navigation
- **Shift + Right Arrow**: Jump to next label frame
- **Shift + Left Arrow**: Jump to previous label frame
- **F**: Advance one frame forward
- **B**: Go back one frame
- **G**: Go to specific frame (opens dialog)

### Labeling
- **Left Click**: Add or edit point for current track
- **Period (.)**: Switch to next track
- **Comma (,)**: Switch to previous track
- **O**: Open options menu

### Saving
- **S**: Save labels to DLC format (saves to original CollectedData file)
- **Ctrl+S**: Save As (not typically needed in DLC mode)

### View
- **Mouse Scroll**: Zoom in/out
- **Shift + Drag**: Pan the view
- **V**: Toggle viewfinder (zoomed region)
- **+/-**: Zoom in/out at mouse position

## File Structure

### Expected DLC Project Structure
```
DLCproject/
├── config.yaml
├── videos/
│   └── video_name.mov  (video file)
└── labeled-data/
    └── video_name/
        ├── CollectedData_*.h5  (manual labels)
        ├── CollectedData_*.csv
        ├── machinelabels-iter*.h5  (optional machine labels)
        ├── img001.png
        ├── img002.png
        └── ...
```

### What Gets Loaded

1. **CollectedData_*.h5**: Your manual labels (required)
2. **machinelabels-iter*.h5**: Machine-generated labels (optional)
   - The highest iteration number is used
   - Only frames not in CollectedData are added
   - Uses likelihood threshold (default 0.9)

### Frame Numbers

Frame numbers are extracted from the image filenames:
- `img103.png` → Frame 103
- `img0042.png` → Frame 42
- etc.

## Multi-Index Header Structure

The DLC h5 files use a multi-index header:

### Single Animal Projects
- Levels: `scorer`, `bodyparts`, `coords`
- Example: `('DLC', 'nose', 'x')`

### Multi-Animal Projects
- Levels: `scorer`, `individuals`, `bodyparts`, `coords`
- Example: `('DLC', 'ind1', 'nose', 'x')`

### Track Naming in Argus

- **Single animal**: Tracks named by bodypart (e.g., `nose`, `left_ear`)
- **Multi-animal**: Tracks named `individual_bodypart` (e.g., `ind1_nose`, `ind2_left_ear`)

## Coordinate Systems

⚠️ **Important**: DLC and Argus use different coordinate systems:

- **DLC**: Origin at top-left (0,0), y increases downward
- **Argus**: Origin at bottom-left (0,0), y increases upward

The integration automatically handles this conversion:
- When loading: `y_argus = video_height - y_dlc`
- When saving: `y_dlc = video_height - y_argus`

## Saving

When you press **S** in DLC labeling mode:

1. Data is converted back to DLC format
2. Coordinates are flipped back to DLC coordinate system
3. Files are saved:
   - `CollectedData_*.h5` (overwrites original)
   - `CollectedData_*.csv` (matching CSV)
4. Only frames in `label_frames` list are saved
5. Points at (0,0) are converted to NaN (DLC's missing data marker)

## Visual Indicators

- Window title shows **[LABEL FRAME]** when on a frame that needs labeling
- Use Shift+Arrow to jump between label frames quickly

## Limitations

- Currently supports **single camera only** for DLC labeling
- Machine labels are imported but not updated during the session
- Likelihood values are set to 1.0 for all manually added/edited points

## Troubleshooting

### "No CollectedData*.h5 file found"
Make sure you're selecting a folder inside `labeled-data/`, not the project root.

### "Video not found"
The video should be in `../videos/` relative to the labeled-data folder. If not, you'll be prompted to locate it manually.

### "Could not import DLCclicker module"
Make sure argus_gui is properly installed: `pip install -e .`

### Coordinate flipping issues
If your labels appear upside-down, check:
1. Video height is being correctly detected
2. The `flipy=True` parameter is set in `convert_to_argus_format()`

## Example Workflow

1. Create DLC project and extract frames as usual
2. Launch Argus and click "DLC Labeling"
3. Select your labeled-data folder
4. Argus opens with video at first label frame
5. Click to add/edit labels for visible bodyparts
6. Press Shift+Right to jump to next label frame
7. Repeat until all frames are labeled
8. Press S to save
9. Labels are saved back to DLC format
10. Continue with DLC training pipeline

## Technical Details

### Module: `argus_gui/DLCclicker.py`

Main class: `DLCLabelFolder`

Key methods:
- `load_data()`: Loads h5 files and extracts metadata
- `find_video()`: Locates the video file
- `get_tracks_for_argus()`: Generates track names
- `convert_to_argus_format()`: Converts DLC data → Argus sparse matrix
- `save_to_dlc_format()`: Converts Argus data → DLC h5/csv

### Modified Files

- `argus_gui/resources/scripts/argus-click`: Added DLC mode globals, navigation, and save functions
- `argus_gui/Argus.py`: Added "DLC Labeling" button and launcher method

### Global Variables in DLC Mode

```python
dlc_label_mode = True           # Flag indicating DLC mode active
dlc_label_frames = [103, 140, ...] # List of frames to label
dlc_folder_handler = DLCLabelFolder(...)  # Handler object
dlc_video_height = 720          # Video height for conversion
```

## Future Enhancements

Potential improvements:
- Support for multiple cameras
- Live update of machine labels
- Custom likelihood threshold per session
- Split individuals mode for multi-animal projects
- Batch processing of multiple labeled-data folders
- Integration with DLC's refinement GUI workflow
