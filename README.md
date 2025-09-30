Argus
=======

Argus is a python package with multiple tools for 3D camera calibration and reconstruction. Argus Panoptes had thousands of eyes, but you may only have two or more cameras.  Hopefully this will help.

![Argus GUI](https://img.shields.io/badge/version-3.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-GPL%20v3-green)

Updated: 2025-06-30, tested on Windows 11 and MacOS 15.5

Read the [full documentation](docs/index.md).

Find our original website at [https://argus.web.unc.edu](https://argus.web.unc.edu).

### How do I get set up?

Users new to Python should follow the full [installation instructions](docs/installation.md) to get started.

### Quick installation instructions

#### Option 1: Install with pip (recommended for most users)

**Step 1: Create a virtual environment (recommended)**

A virtual environment keeps your argus_gui installation separate from other Python packages on your system.

1. Open a terminal (macOS/Linux) or Command Prompt (Windows)
2. Create a new virtual environment:
   ```bash
   python -m venv argus_env
   ```
3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source argus_env/bin/activate
     ```
   - On Windows:
     ```bash
     argus_env\Scripts\activate
     ```
   
   You should see `(argus_env)` at the beginning of your command prompt when the environment is active.

**Step 2: Install argus_gui**

```bash
pip install git+https://github.com/backyardbiomech/argus_gui.git
```

**Step 3: Run the GUI**

```bash
argus-gui
```

**For future use:** Remember to activate your virtual environment each time before using argus_gui:
- macOS/Linux: `source argus_env/bin/activate`
- Windows: `argus_env\Scripts\activate`

Then run: `argus-gui`

**Note:** The installation will automatically handle the `sba` and `argus` dependencies from GitHub.

**Troubleshooting:** 

- If during use you encounter an error related to `cv2` or `opencv`, or if omnidirectional calibration fails with `INITIAL_FISHEYE` not defined, you need the contrib package:
  ```bash
  # With pip:
  pip uninstall opencv-python
  pip install opencv-contrib-python
  
  # With uv:
  uv pip uninstall opencv-python
  uv pip install opencv-contrib-python
  ```

- If you (especially on Windows) encounter an error related to `ffmpeg`, or `ffplay`, download the latest version of `ffmpeg` add it to your system's PATH using <a href="https://www.wikihow.com/Install-FFmpeg-on-Windows" target="_blank">these instructions</a>.


#### Option 2: Install with conda (if you prefer conda)
<details>
<summary><strong>👈 Click the triangle to expand!</strong></summary>

1. Right-click this link and select "Save Link As..." or "Download Linked File As..." : <a href="https://raw.githubusercontent.com/backyardbiomech/argus_gui/main/Argus.yaml">Argus.yaml</a> (save it as `Argus.yaml`, not `Argus.yaml.txt`).
2. Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) or anaconda on your computer. 
3. Open a terminal (macOS/Linux) or Anaconda Prompt (Windows).
4. Navigate to the directory where you downloaded `Argus.yaml` (probably your Downloads folder). You can use the `cd` command to change directories. For example:
   ```
   cd ~/Downloads
   ```
   or on Windows:
   ```   
   cd C:\Users\<YourUsername>\Downloads
   ```

5. Run the command:
   ```
   conda env create -f Argus.yaml
   ```
6. Activate the environment:
   ```
   conda activate argus
    ```
7. Open the gui with the command:
   ```
   argus-gui
   ```

8. To start the GUI in the the future, open a terminal or Anaconda Prompt, activate the environment with:
   ```
   conda activate argus
   ```
   and then run:
   ```
   argus-gui
   ```

</details>   
   

## Citation

If you use Argus in your research, please cite the following paper:

**Jackson, B.E., Evangelista, D.J., Ray, D.D., and Hedrick, T.L.** (2016). 3D for the people: multi-camera motion capture in the field with consumer-grade cameras and open source software. *Biology Open*, 5(9), 1334-1342. [https://doi.org/10.1242/bio.018713](https://doi.org/10.1242/bio.018713)

<details>
<summary><strong>📋 BibTeX Citation</strong></summary>

```bibtex
@article{Jackson3Dpeoplemulticamera2016,
  title = {{3D for the people: multi-camera motion capture in the field with consumer-grade cameras and open source software}},
  author = {Jackson, Brandon E. and Evangelista, Dennis J. and Ray, Dylan D. and Hedrick, Tyson L.},
  year = {2016},
  journal = {Biology Open},
  volume = {5},
  number = {9},
  pages = {1334--1342},
  doi = {10.1242/bio.018713},
  pmid = {27444791},
  url = {https://doi.org/10.1242/bio.018713}
}
```

</details>


### Who do I talk to?

Any questions or comments can be emailed to:
jacksonbe3@longwood.edu or ddray@email.unc.edu
