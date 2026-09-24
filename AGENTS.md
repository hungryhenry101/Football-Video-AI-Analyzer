# AGENTS.md

This file provides guidance to coding agents when working in this project. To make this file always up-to-date, modify it if major changes are made

## Project Overview

A research/learning prototype (not production) that ingests a football match video and, frame-by-frame, detects and tracks players, referees, and the ball, then renders a camera view ("cam") and a top-down Bird's Eye View ("bev"). The long-term goal is automatic goalkeeper highlight generation and match analytics — **not yet implemented** (see README TODOs). Everything is orchestrated by `main.py` as a single pass over the video.

## Running

- activate the `football` conda environment (`conda env create -f environment.yml`; conda-forge, Python 3.12)
- build the calibration extension:
  ```bash
  python scripts/build_minimal_opencv.py   # once, ~20 min
  python scripts/build_minimal_ceres.py    # once, ~5 min
  python scripts/build_native.py           # rerun after any C++ change
  ```
  OpenCV and Ceres are compiled locally, not taken from a package manager: every
  packaged build of either links a second OpenMP runtime, and PyTorch's wheel
  already ships one, which makes libomp abort the interpreter on import. Reasons
  and trade-offs in `core/BroadTrack/README.md`. `build_native.py` falls
  back to the system libraries (with a warning) if they are absent.

## Architecture

The pipeline has three stages wired together in `main.py`'s loop.

### 1. Pitch detection & calibration — `core/broadtrack_calib/`
BroadTrack's temporal camera tracker, wrapped for Python. Two TorchScript
detectors run in Python (`models.py`: `nbjw_keypoint_model.pt` for 57 pitch
keypoints, `tvcalib_model.pt` for 28 pitch-line classes); everything numerically
expensive is C++ in `core/BroadTrack/python/`, built as the `_broadtrack`
extension. `BroadTrackCalib.estimate(frame, boxes=None)` returns a dict with
**`K` (intrinsics), `R` (rotation), `t` (translation), `P = K[R | -Rt]`**, plus
`score` (line-IoU), `reinit`, `k1` and the pan/tilt/roll decomposition.

Unlike the per-frame solver it replaced, the tracker is stateful: it carries the
camera, the optical-flow tracks, an anchored tripod position and a lost-tracking
counter across frames. Call `reset()` on a video cut. See
`core/BroadTrack/README.md` for the build knobs, the measured per-stage timings,
and the platform gotchas (import order on macOS, 16:9 requirement).

`PitchGeometry` owns the **BEV (top-down) representation**: `create_bev_template()`
builds a static pitch image, `world_to_bev_px(x, y)` maps world meters → BEV
pixels (pitch is 105×68 m, origin at center, `bev_scale = 10 px/m`), and
`draw_pitch_lines()` projects the markings with a given `P`.

### 2. Player tracking — `core/player_tracker.py`
`PlayerTracker` wraps a YOLO model (`model.track(..., tracker="config/botsort.yaml")`) giving BoT-SORT (handles GMC, ReID, Kalman). `update(frame)` returns a list of dicts `{id, bbox, cls, conf}` (classes: 0=ball, 1=goalkeeper, 2=player, 3=referee). In `main.py` it runs *before* the calibration so its boxes can be handed to BroadTrack as its `human-bboxes` input.

### 3. Ball tracking — `core/ball_tracker.py`
`BallDetector` (YOLO, class 0 only) produces pixel detections; `project_to_ground()` uses the camera model to lift them to ground-plane world coords. `BallTracker` (a `filterpy` Kalman filter, state `[x, y, vx, vy]` in BEV meters) selects the best candidate each frame via **Mahalanobis-distance gating with a Chi-Square threshold** (`chi2.ppf(0.95, df=2)`), then smooths and predicts to ride through occlusions.

### Coordinate systems & projection — `core/projection_utils.py`
Read it before touching any projection code.
- **World coords**: meters, origin at pitch center, x towards the right goal line, y towards the bottom touch line, ground plane `z=0`, and **z is negative above the ground** (the camera sits at negative z). This is BroadTrack's frame and the one PnLCalib used, so the two are interchangeable.
- `pixel_to_ground(u,v,K,R,t)` → ground (z=0) intersection using the **full pinhole model** (preferred over homography when camera tilt ≠ 0).
- `pixel_to_3d(u,v,bbox_width,K,R,t)` → full 3D position from apparent ball size; **currently unused** but is the planned path for aerial/high balls (see problems.md).
- `build_projection_matrix` / `project_3d_to_pixel` build and apply `P = K[R | -Rt]`.

## Tests

`python -m pytest tests/ -q`. The geometry and detector-label-map tests always
run; the end-to-end calibration tests need the BroadTrack TorchScript models and
`core/BroadTrack/frames/`, and skip when those are absent. When present they are
regression checks against the scores the upstream C++ binary recorded for the
same sequence (`core/BroadTrack/out/anchored_pos.json`).

## Files and Docs
Don't modify them before asking me or i told you so.

### under `demos/`

### gitignore
- `output`: folder in root and in test folder
- `models`: all four model weights — the two YOLO detectors (`football_best.pt`,
  `football_best_big.pt`) and BroadTrack's two TorchScript detectors (see `core/BroadTrack/README.md`)
- `core/broadtrack_calib/_broadtrack*.so`: the compiled extension

### under `docs/`
- `problems.md`: solved and to be solved problems/bugs, some may have solutions and some may require deeper research
- `What Have I Learnt.md`: For academic report or sth like that

### READMEs
- `README.md`: Chinese ver.
- `README_EN.md`: English ver.