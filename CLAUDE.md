# CLAUDE.md

This file provides guidance to Claude Code when working in this project. To make this file always up-to-date, modify it if major changes are made

## Project Overview

A research/learning prototype (not production) that ingests a football match video and, frame-by-frame, detects and tracks players, referees, and the ball, then renders a camera view ("cam") and a top-down Bird's Eye View ("bev"). The long-term goal is automatic goalkeeper highlight generation and match analytics — **not yet implemented** (see README TODOs). Everything is orchestrated by `main.py` as a single pass over the video.

## Running

- activate the `ml` conda environment

## Architecture

The pipeline has three stages wired together in `main.py`'s loop.

### 1. Pitch detection & calibration — `core/pnl/`
`PnLCalib.estimate(frame)` runs two HRNet heatmap models (`weights/SV_kp` for 57 pitch keypoints, `weights/SV_lines` for 23 line types), completes missing keypoints from line intersections, then runs `FramebyFrameCalib` (from `utils/utils_calib.py`) to compute the per-frame camera model. It returns a dict with **`K` (intrinsics), `R` (rotation), `t` (translation), and `P = K[R | -Rt]`** (3×4 projection matrix).

This module also owns the **BEV (top-down) representation**: `create_bev_template()` builds a static pitch image, and `world_to_bev_px(x, y)` maps world meters → BEV pixels (pitch is 105×68 m, origin at center, `bev_scale = 10 px/m`).

### 2. Player tracking — `core/player_tracker.py`
`PlayerTracker` wraps a YOLO model (`model.track(..., tracker="config/botsort.yaml")`) giving BoT-SORT (handles GMC, ReID, Kalman). `update(frame)` returns a list of dicts `{id, bbox, cls, conf}` (classes: 0=ball, 1=goalkeeper, 2=player, 3=referee).

### 3. Ball tracking — `core/ball_tracker.py`
`BallDetector` (YOLO, class 0 only) produces pixel detections; `project_to_ground()` uses the PnL camera model to lift them to ground-plane world coords. `BallTracker` (a `filterpy` Kalman filter, state `[x, y, vx, vy]` in BEV meters) selects the best candidate each frame via **Mahalanobis-distance gating with a Chi-Square threshold** (`chi2.ppf(0.95, df=2)`), then smooths and predicts to ride through occlusions.

### Coordinate systems & projection — `core/pnl/projection_utils.py`
Read it before touching any projection code.
- **World coords**: meters, origin at pitch center, ground plane `z=0`.
- `pixel_to_ground(u,v,K,R,t)` → ground (z=0) intersection using the **full pinhole model** (preferred over homography when camera tilt ≠ 0).
- `pixel_to_3d(u,v,bbox_width,K,R,t)` → full 3D position from apparent ball size; **currently unused** but is the planned path for aerial/high balls (see problems.md).
- `build_projection_matrix` / `project_3d_to_pixel` build and apply `P = K[R | -Rt]`.

## Files and Docs
Don't modify them before asking me or i told you so.

### under `demos/`


### gitignore
- `output`: folder in root and in test folder
- `weights`: model weights

### under `docs/`
- `problems.md`: solved and to be solved problems/bugs, some may have solutions and some may require deeper research
- `What Have I Learnt.md`: For academic report or sth like that

### READMEs
- `README.md`: Chinese ver.
- `README_EN.md`: English ver.