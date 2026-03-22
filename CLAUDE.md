# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository. 

Edit this file every time after making any changes to the structure or core code, so that when you view it next time, it won't be outdated.

## Project Overview

AI Goalkeeper Highlight Generator - Research/learning project for tracking people and ball in football match videos, identifying goalkeepers, and calculating threat levels to automatically generate goalkeeper highlight reels.

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
python draw_2d.py  # Reads from output/track_log.csv
```

## Architecture

```
├── main.py              # Entry point - YOLO detection + ByteTrack tracking + CMC
├── draw_2d.py           # Post-processing visualization from CSV logs
├── test.py              # Unit tests for calibration and other components
├── bytetrack.yaml       # ByteTrack tracker configuration
├── core/
│   ├── ball_tracker.py  # Kalman Filter for ball tracking with occlusion handling
│   ├── cmc.py           # Camera Motion Compensation using optical flow
│   └── calibration.py   # Camera calibration using field pose detection
├── models/              # YOLO models (football_best.pt, yolo11m.pt, field_pose_best.pt)
├── input_vids/          # Input videos
├── output/              # Tracking logs (CSV), rendered videos, and test outputs
└── output/test_calibration/  # Test visualization outputs
```

## Field Pose Model

Here are the complete meanings of the 32 key points in models/field_pose_best.pt:

**Left Goal / Penalty Area (KP 0-12)**

- KP 00: Left Penalty Area Top-Left Corner
- KP 01: Left Penalty Area Upper Corner
- KP 02: Left Penalty Area Left Corner
- KP 03: Left Goal Area Top-Left Corner
- KP 04: Left Goal Line Top
- KP 05: Left Goal Line Bottom
- KP 06: Left Penalty Corner 1
- KP 07: Left Goal Area Corner
- KP 08: Left Penalty Corner 2
- KP 09: Left Penalty Area Top-Right Corner
- KP 10: Left Penalty Area Bottom-Right Corner
- KP 11: Left Penalty Area Bottom Corner
- KP 12: Left Penalty Area Bottom-Left Corner

**Center Line / Center Circle (KP 13-16, 30-31)**

- KP 13: Center Circle Top
- KP 14: Center Circle Upper
- KP 15: Center Circle Lower
- KP 16: Center Circle Bottom
- KP 30: Center Line Midpoint
- KP 31: Center Circle Center

**Right Goal / Penalty Area (KP 17-29)**

- KP 17: Right Penalty Area Top-Left Corner
- KP 18: Right Penalty Area Upper Corner
- KP 19: Right Penalty Area Right Corner
- KP 20: Right Goal Area Bottom-Right Corner
- KP 21: Right Penalty Corner 1
- KP 22: Right Penalty Corner 2
- KP 23: Right Goal Area Corner
- KP 24: Right Penalty Area Top-Right Corner
- KP 25: Right Penalty Area Top Edge
- KP 26: Right Penalty Area Upper Corner
- KP 27: Right Goal Area Top-Right Corner
- KP 28: Right Goal Line Top
- KP 29: Right Goal Line Bottom

## Core Components

**`main.py`**: Processes video frame-by-frame:
- Ball detection via YOLO + Kalman Filter (`core/ball_tracker.py`)
- Player detection via YOLO + ByteTrack
- Camera motion compensation (`core/cmc.py`)
- Outputs: `output/out.mp4`, `output/track_log.csv`

**`core/ball_tracker.py`**: `BallTracker` class
- Uses FilterPy Kalman Filter (4D state: x, y, vx, vy)
- Selects ball detections based on proximity to predicted position
- States: `VISIBLE` / `OCCLUDED` (after 3 consecutive misses)

**`core/cmc.py`**: `CMC` class
- Computes 2x3 affine transform from current frame to reference frame
- Uses Shi-Tomasi corners + Lucas-Kanade optical flow (masks out players/ball)
- Outputs `M_to_ref` for compensating player positions

## Key Configuration

- **Detection threshold**: `CONF_THRES = 0.15` in `main.py`
- **Tracker settings**: `bytetrack.yaml` (thresholds, buffer, match_thresh)
- **Ball tracker**: `MAX_MISS = 6`, validation gate threshold in `ball_tracker.py`
- **CMC**: Requires ≥6 good feature points, ≥50% inlier ratio for valid transform