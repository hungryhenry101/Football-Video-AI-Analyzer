# AGENTS.md

This file provides guidance to coding agent when working with code in this repository. 

Edit this file every time when any changes to the structure or core code have been made, so that when you view it next time, it won't be outdated.

## Project Overview

Football Video AI Analyzer - Research/learning project for tracking people and ball in football match videos, identifying goalkeepers, and calculating threat levels to automatically generate goalkeeper highlight reels.

## Quick Start

```bash
conda activate ml
pip install -r requirements.txt
```

**Run tracking:**
```bash
python main.py  # Edit model paths and VIDEO_PATH in main.py first
```

**Visualize trajectories:**
```bash
python draw_2d.py  # Reads from output/track_log.csv, visualizes raw vs compensated paths
```

## Architecture

```
├── main.py                    # Entry point - YOLO detection + ByteTrack tracking + CMC
├── draw_2d.py                 # Post-processing visualization of raw vs compensated trajectories
├── test_pitch_segmentation.py # Test script for soccer pitch segmentation model
├── bytetrack.yaml             # ByteTrack tracker configuration
├── core/
│   ├── ball_tracker.py        # Kalman Filter for ball tracking with occlusion handling
│   ├── cmc.py                 # Camera Motion Compensation using optical flow
│   ├── calibration.py         # Utilities for homography estimation from correspondences
│   ├── ui_renderer.py         # Enhanced UI rendering with stats, mini-map, threat level
│   └── pitch_detection/
│       ├── line_det.py        # Segmentation network and line extraction logic
│       ├── homography_estimator.py # Estimates homography using detected line intersections
│       └── soccerpitch.py     # Physical dimensions and geometry of a standard pitch
├── models/
│   ├── pitch_seg_npy/                 # mean.npy & std.npy for pitch segmentation
│   ├── soccer_pitch_segmentation.pth  # DeepLabV3 ResNet50 pitch segmentation (29 classes)
│   ├── field_pose_best.pt             # Field pose detection model (32 keypoints)
│   ├── football_best.pt               # Player/ball detection model
│   └── yolo11m.pt                     # YOLOv11 backbone
├── input_vids/                # Input videos
├── output/                    # Tracking logs (CSV), rendered videos, and test outputs
└── output/test_pitch_segmentation/  # Segmentation test outputs
```

## Field Pose Model (32 Keypoints)

Used in `field_pose_best.pt` to identify specific points on the pitch for calibration:
- **Left Area (0-12)**: Penalty area corners, goal area corners, goal line intersections.
- **Center (13-16, 30-31)**: Center circle edges, center line midpoint, center circle center.
- **Right Area (17-29)**: Penalty area corners, goal area corners, goal line intersections.

more details can be found in the comments of `core/pitch_detection/soccerpitch.py`

## Core Components

**`main.py`**:
- Integrates `LineDetector` for pitch analysis.
- Performs YOLO-based tracking for players and ball.
- Applies `CMC` for camera motion compensation.
- Outputs rendered video (`output/out.mp4`) and trajectory logs (`output/track_log.csv`).

**`core/pitch_detection/line_det.py`**:
- **`SegmentationNetwork`**: Renamed `analyse_img` to `forward`. Processes BGR images into 29-class semantic masks.
- **`LineDetector`**: Now initialized with `project_dir`, `width`, and `height`. Its `detect(img)` method handles internal segmentation and returns fitted polylines for detected pitch features.

**`core/pitch_detection/homography_estimator.py`**:
- Uses intersections of detected lines (e.g., center line vs touch lines) to calculate the Homography matrix `H`.
- Provides `warp(img)` for Bird's Eye View (BEV) and `draw_pitch_lines` for reprojection.

**`core/ball_tracker.py`**: `BallTracker` class
- Uses FilterPy Kalman Filter (4D state: x, y, vx, vy)
- Selects ball detections based on proximity to predicted position
- States: `VISIBLE` / `OCCLUDED`

**`core/cmc.py`**: `CMC` class
- Computes 2x3 affine transform from current frame to reference frame
- Uses Shi-Tomasi corners + Lucas-Kanade optical flow (masks out players/ball)

**`core/ui_renderer.py`**: `UIRenderer` class
- Features: Mini-map, frame info, GK identification, velocity estimation, ball trajectory, camera motion vector, threat level, and possession stats.

## Pitch Segmentation Model

**`test_pitch_segmentation.py`**: Test script for DeepLabV3 ResNet50 segmentation.

**29 Segmentation Classes:**
- **Boundary**: Side lines (T/B/L/R), Middle line.
- **Penalty Areas**: Big rectangles (L/R) - top, bottom, main.
- **Goal Areas**: Small rectangles (L/R) - top, bottom, main.
- **Circles**: Center, Left, Right.
- **Goals**: Crossbar, Left Post, Right Post (L/R sides).

## Key Configuration

- **Detection threshold**: `CONF_THRES = 0.15` in `main.py`
- **Tracker settings**: `bytetrack.yaml`
- **Ball tracker**: `MAX_MISS = 6`
- **CMC**: Requires ≥6 good feature points, ≥50% inlier ratio.
