# Stereo_Vision_Camera_Box
A stereo vision system with custom camera box design and GUI interface for depth measurement and point cloud generation.
# Stereo Vision Camera Box

A stereo vision system with custom camera box design and GUI interface for depth measurement and point cloud generation.

## Features

- **GUI Interface**: User-friendly interface for real-time stereo vision capture
- **Real-time Distance Measurement**: Click on depth map to measure distance
- **Point Cloud Generation**: Automatic PLY format point cloud file generation
- **Data Management**: Organized saving of stereo images, depth maps, and metadata

## Hardware Configuration

### Stereo Camera Box Design
- **Box Structure**: Custom box with stereo cameras mounted on the internal top
- **Lighting System**: LED beads surrounding cameras for uniform illumination
- **Reflection Optimization**: Reflective stickers on internal walls to enhance lighting
- **Diffuse Bottom Surface**: White diffuse stickers on the bottom to reduce shadows
- **Calibration Board**: 9x6 chessboard pattern printed on A4 paper

### Camera Parameters
- **Resolution**: 640x480 (per camera)
- **Concatenated Stream**: 1280x480 (left and right cameras combined)
- **Baseline Distance**: ~120mm

## Installation

```bash
pip install opencv-python
pip install opencv-contrib-python
pip install numpy
pip install pillow
pip install tkinter
```

## Usage

### Camera Calibration
First, calibrate your stereo cameras using the provided (chessboard)[https://smiler488.github.io/app/targets/] pattern and update the camera parameters in the code.

### Run the GUI Application
```bash
python stereo_capture_box_gui.py
```

**Features:**
- Enter sample ID
- Click "Start Preview" to view real-time stereo images and depth map
- Press `s` or click "Save Current Sample" to save data
- Click on depth map to measure distance at that point
- Click "Next Sample" to switch to new sample

## Output Data

The system automatically saves:
- **Left/Right Images**: PNG format stereo image pairs
- **Depth Maps**: Pseudo-color PNG + raw disparity NPY files
- **Point Cloud Files**: PLY format with 3D coordinates and RGB colors
- **Metadata**: JSON format with sample info, timestamp, and camera parameters

Data is organized in folders by sample ID:
```
data/
└── [sample_id]/
    ├── left/          # Left camera images
    ├── right/         # Right camera images
    ├── depth/         # Depth maps and disparity data
    └── pointcloud/    # Point cloud files (.ply)
```

## Hardware Setup Tips

1. **Lighting**: Ensure even LED illumination around cameras
2. **Calibration**: Use the provided 9x6 chessboard pattern for accurate calibration
3. **Surface**: White diffuse bottom surface helps reduce shadows
4. **Positioning**: Mount cameras securely at the top of the box

**Note**: Proper camera calibration is essential for accurate depth measurements. Update the camera parameters in the code after calibration.
