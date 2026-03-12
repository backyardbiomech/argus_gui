# Working with DeepLabCut
Argus GUI can be integrated with [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), a popular tool for animal pose estimation, to enhance your 3D tracking capabilities. This section provides guidance on how to set up and use Argus with DeepLabCut.

## 3D in DeepLabCut
DeepLabCut natively supports 3D tracking with several significant limitations that Argus can help overcome:
1. **Number of Cameras**: DeepLabCut is limited to two cameras for 3D tracking, while Argus can handle multiple cameras.
2. **Camera Synchronization**: Argus provides robust synchronization tools using audio, while DeepLabCut requires that video files are synchronized before importing.
3. **Filming Volume Calibration**: Argus can calibrate the filming volume using a wand, which is not supported in DeepLabCut, which requires filming of checkerboard patterns to simultaneous determine camera instrinsics and extrinsics. The wand calibration (structure-from-motion) allows for 3D tracking in difficult to access filming volumes by separating the camera intrinsics and extrinsics calibration steps.

For a robust 2-camera setup in a controlled envinroment, we strongly recommend using DeepLabCut directly for the simplest workflow. However, for more complex setups or when using more than two cameras, Argus provides a more flexible and powerful solution, and can take advantage of DeepLabCut's pose estimation capabilities.

## Installation
We do not recommend trying to install Argus and DeepLabCut in the same Python environment. It might work, but both have a number of dependencies that may conflict. Instead, we recommend using a separate environment for Argus and DeepLabCut.

## Using Argus with DeepLabCut
Of all of the modules in Argus, the **Clicker** module is the manual analog of DeepLabCut pose estimation; and it can both import and export DeepLabCut data. This provides several possible uses and workflows:

### 1. Using Argus Clicker to manually clean up DeepLabCut data
Even for single-camera setups, DeepLabCut may not produce perfect tracking data. The ideal fix would be to iterate through extracting outliers, refining the model, retraining the model, and then re-tracking. Where that timeline isn't practical, you can use Argus Clicker to manually clean up the tracking data.
1. Track your video in DeepLabCut as normal. This should produce a .h5 file with the tracking data.
2. Open Argus Clicker and load the video in question.
3. Once the video window is open, type `O` for the options window and click the `Load DeepLabCut data` button to import the DeepLabCut tracking data.
4. You can now use Argus Clicker to manually edit the tracking data, deleting or correcting points as needed. 
5. When you are done, type `O` again to open the options window. You can either save it as a dense or spare Argus file, or select the `DeepLabCut .h5` option to save the data back to a DeepLabCut .h5 file. If you use this option, the likelihood values for any points you edited will be set to 1.0.
6. Note that to use this edited file in DeepLabCut functions (like `create_labeled_video`), you will need to rename the file to match the original DeepLabCut file name, so make a copy of the original file and put in a new folder before renaming the edited file.

### 2. Using Argus Clicker to export DeepLabCut data for training
If you have previously used Argus Clicker (or DLTdv) to manually track points in a video, and you want to use that data to train a DeepLabCut model, you can export the data in the correct format.
A function to do that is available in [DLCconverterDLT](https://github.com/backyardbiomech/DLCconverterDLT), and will be included in a future version of Argus.

### 3. Using Argus Clicker to import DeepLabCut data for 3D tracking
If you have a DeepLabCut model trained for 3D tracking, you can use Argus **Wand** for 3D calibration and **Clicker** to import the tracking data and use it for 3D triangulation.
1. Train your DeepLabCut model to track each video. Ignore DeepLabCut's 3D tracking capabilities, as Argus will handle that. This might be one model per camera, or if all cameras have similar enough views you might make a single model for all cameras.
2. Track your animals in each camera using DeepLabCut. This should produce a .h5 file with the tracking data for each camera.
3. You can train a separate model to track a wand with DeepLabCut. However, since only about 60 wand points are needed, it may be faster for single set-ups to manually track the wand using Argus **Clicker**. If you used DeepLabCut to track the wand, open all of the wand videos in Argus **Clicker** with the appropriate offsets. Once the video windows are open, type `O` for the options window and click the `Load DeepLabCut data` button to import the DeepLabCut tracking data. You will be asked for one tracking file per video, so make sure you open them in the same order as you loaded the videos. Clean up the wand tracks and save the data as an Argus file.
4. Re-open the videos and manually digitize some unpaired points and reference points if needed (see the [Wand module documentation](user-guide.md#wand) for details).
5. Open the **Wand** module and load the wand tracking data (paired points), unpaired points, and reference points, and camera profiles. Run wand, remove outliers if needed, and save the results.
6. Open the **Clicker** module and load the animal tracking videos with appropriate offsets.
7. Once the video windows are open, type `O` for the options window and click the `Load DeepLabCut data` button to import the DeepLabCut tracking data for each camera. You will be asked for one tracking file per video, so make sure you open them in the same order as you loaded the videos.
8. Load the DLT coefficients file generated by the **Wand** module and the camera profile. 
9. Saving as an Argus file will now produce an `xyzpts.csv` file with the 3D triangulated points for each camera.
10. There are additional functions to streamline the triangulation available at [DLCconverterDLT](https://github.com/backyardbiomech/DLCconverterDLT), and will be included in a future version of Argus.

### 4. Labeling and Refining DeepLabCut projects in Argus
Argus Clicker includes a specialized mode for labeling and refining DeepLabCut projects. This feature allows you to use Argus *as an alternative* to the DeepLabCut GUI for adding or correcting labels, with some advantages including keyboard navigation between labeled frames, the ability to step through unlabeled sequential frames, and the ability to refine machine-generated labels without overwriting existing manual labels.

#### Starting DLC Labeling Mode

**From the Argus GUI:**
1. Launch the main Argus GUI
2. Click the `DLC Labeling` button
3. Select the `labeled-data` folder for the video you want to label (e.g., `project-name/labeled-data/video_name/`)
4. Select the associated video file when prompted (Argus will first check the `project_name/videos/` directory for a video and load it automatically if it finds a match)

**From the command line:**
```bash
# Launch DLC labeling mode with folder selection dialog
argus-click --dlc-label

# Or specify the labeled-data folder directly
argus-click --dlc-label /path/to/project/labeled-data/video_name
```

#### How DLC Labeling Mode Works

When you open a DLC labeled-data folder, Argus automatically:

1. **Loads existing manual labels**: If a `CollectedData*.h5` file exists, Argus loads all frames that have been manually labeled
2. **Loads machine-generated labels**: Argus finds the most recent `machinelabels-iter*.h5` file (highest iteration number) and loads the predicted labels
3. **Combines the data**: 
   - Frames from `CollectedData` are prioritized - **manual labels are never overwritten by machine labels**
   - Frames that exist only in machine labels are added to the label list for refinement
   - This allows you to review model predictions on new frames without losing your manual work
4. **Prepares the interface**: All body parts (and individuals for multi-animal projects) are loaded as tracks

#### Navigation Controls

DLC labeling mode includes special keyboard shortcuts for efficient navigation:

- **Shift + Right Arrow**: Jump to the next labeled frame
- **Shift + Left Arrow**: Jump to the previous labeled frame
- **Regular Arrow Keys**: Navigate to next frame with a label for the currently selected point (as in normal Argus mode)
- `,` and `.` keys: step forward or backward through the bodyparts list.
- **f** and **b** Keys: Step forward or backward one frame at a time, regardless of labels. Adding `shift` to these keys steps 50 frames at a time.
- `+`, `-`, and `r` keys: zoom in and out at the current mouse position, or reset zoom to fit the video.
- `d` key: deletes the currently selected body part point on the current frame.
- These and other clicker keyboard shortcuts are the same as in normal [**Clicker** function](user-guide.md#clicker).


#### Editing Labels

Once in DLC labeling mode:

1. **View existing labels**: All body parts are displayed on the current frame with color-coded markers
2. **Add, correct, or delete labels**: Click on the video to place or move a marker for the currently selected body part
   - The current track is highlighted with a larger pink circle.
   - There is no dragging a point. Instead, simply click to place the point where it should be.
3. **Navigate between frames**: Use Shift+Arrow keys to jump between labeled frames, or `f` and `b` keys to move frame-by-frame
   - If you place a label on a frame that is not in the extracted frames list, you will get a warning and that point will not be saved. If you find a frame you want to label that is not in the extracted frames list, you need to extract it using DLC first (this ability may be added to Argus in the future).
4. **Note**: Auto-advance is disabled by default in DLC mode to give you more control when clicking points. Changing offsets is also disabled to prevent accidentally shifting the data in dlc mode.

#### Saving Your Work

When you're ready to save:

1. Press `S` to save. This will automatically save back to the `CollectedData_*DLC*.h5` and `csv` files in the proper `labeled-data` file.

**Important notes about saving:**
- If a `CollectedData*.h5` file already exists, Argus will save to that file
- If no `CollectedData` file exists (you're working with machine labels only), Argus will create a new file called `CollectedData_DLC.h5`
- Only frames that contain labels will be saved
- **Manual edits are preserved**: Points you manually added or corrected will not be overwritten by machine labels when you save or re-open the project

#### Workflow Tips

- **Refining model predictions**: After training a DeepLabCut model and generating predictions with `analyze_videos()`, extract outlier frames with `extract_outlier_frames()`. Open the labeled-data folder in Argus to quickly review and correct the machine labels on those frames.
- **Adding new labels**: Argus works equally well for labeling frames from scratch. Use the Shift+Arrow navigation to move between extraction frames efficiently.
- **Multi-animal projects**: Argus fully supports multi-animal DeepLabCut projects. Each individual-bodypart combination is treated as a separate track (e.g., "mouse1_nose", "mouse2_nose").
- **Preserving manual work**: Because Argus never overwrites existing `CollectedData` entries with machine labels, you can safely load both old manual labels and new predictions without losing your previous work.