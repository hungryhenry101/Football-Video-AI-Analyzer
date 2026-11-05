# AGENTS.md

This file provides guidance to coding agents when working in this project. To make this file always up-to-date, modify it if major changes are made

## Developing Guidelines

### Before Making Changes
1. Read relevant documentation
2. Inspect the existing implementation and understand the current module boundaries.

### Implementation Principles
1. If anything is unsure, ask the user rather than guessing and proceeding.
2. Prioritise readability for humans. Avoid overly concise or abstract expressions that make code difficult to understand.
3. Write concise comments for non-obvious logic, design decisions, or variables. Not everywhere.
4. Never hardcode API keys, tokens, or other credentials.
5. Stay within the requested scope, reuse existing code where appropriate, and avoid unrelated refactoring.

### After Making Changes
1. Report what was changed and any known limitations.
2. Update relevant documentation (including this file AGENTS.md) to reflect the actual implementation and any changed interfaces or decisions.

## Project Overview

A research/learning prototype (not production) that ingests a football match video and, frame-by-frame, detects and tracks players, referees, and the ball, then renders a camera view ("cam") and a top-down Bird's Eye View ("bev"). The long-term goal is automatic goalkeeper highlight generation and match analytics — **not yet implemented** (see README TODOs). Everything is orchestrated by `main.py` as a single pass over the video.

## Running

### For the first time
- activate the `football` conda environment (`conda env create -f environment.yml`; conda-forge, Python 3.12)
- build the calibration extension fist time:
  ```bash
  python scripts/build_minimal_opencv.py   # once, ~20 min
  python scripts/build_minimal_ceres.py    # once, ~5 min
  python scripts/build_native.py           # rerun after any C++ change
  ```
  OpenCV and Ceres are compiled locally. For reason see `core/BroadTrack/README.md`

### After that
- run the pipeline: `python main.py --video input_vids/test2.mp4 --max-frames 200 --no-display`

### Logging
`core/log.py` is the single setup; every module does `logger = get_logger(__name__)`.
Nothing is printed unless an entry point calls `configure_logging` (`main.py` and both
demos do; tests stay silent).

Levels, chosen by how much output they produce:

| level | contents |
|---|---|
| `TRACE` (5, non-standard) | raw model output, individual detections and ball candidates |
| `DEBUG` | one summary line per frame, state transitions, decisions |
| `INFO` (default) | start-up, model loading, video metadata, run summary |
| `WARNING` | recoverable problems (calibration lost, unreadable frame) |

```bash
python main.py -v                          # DEBUG for every module
python main.py -vv                         # TRACE (per detection / candidate)
python main.py --log-module ball_tracker=TRACE --log-module calib=DEBUG
python main.py --log-file output/run.log   # file captures DEBUG and above
python main.py --log-file output/run.log --log-file-level TRACE  # include per-detection
python main.py --no-log-color
```

The same switches exist as environment variables, which is how the demos are
configured: `FOOTBALL_LOG_LEVEL`, `FOOTBALL_LOG_MODULES="ball_tracker=TRACE"`,
`FOOTBALL_LOG_FILE`, `FOOTBALL_LOG_FILE_LEVEL`, `NO_COLOR`.

`set_frame_context(frame_idx)` (called in each frame loop) tags every record with
`[f=0042]`, so logs from different modules can be lined up without threading the frame
index through every call. Module names may be short (`ball_tracker`) or full
(`core.ball_tracker`); loggers outside this project are hidden unless named explicitly.

When adding logs: use `%s` arguments rather than f-strings, keep per-detection output at
`TRACE`, per-frame output at `DEBUG`, and guard loops whose arguments are expensive with
`logger.isEnabledFor(TRACE)`.

## Architecture

The pipeline has three stages wired together in `main.py`'s loop.

### 1. Pitch detection & calibration — `core/broadtrack_calib/`
BroadTrack's temporal camera tracker, wrapped for Python from C++.
It runs two TorchScript detectors (keypoint and lines).
`BroadTrackCalib.estimate(frame, boxes=None)` returns a dict with
**`K` (intrinsics), `R` (rotation), `t` (translation), `P = K[R | -Rt]`**, plus
`score` (line-IoU), `reinit`, `k1` and the pan/tilt/roll decomposition.

The tracker is stateful, carrying data across frames. 
See `core/BroadTrack/README.md` for platform gotchas and more detailed info.

`PitchGeometry` owns the **BEV (top-down) representation**: `create_bev_template()`
builds a static pitch image, `world_to_bev_px(x, y)` maps world meters → BEV
pixels (pitch is 105×68 m, origin at centre, `bev_scale = 10 px/m`), and
`draw_pitch_lines()` projects the markings with a given `P`.

### 2. Player tracking — `core/player_tracker.py`
`PlayerTracker` wraps a YOLO-seg model (`model.track(..., tracker="config/botsort.yaml")`) giving BoT-SORT (handles GMC, ReID, Kalman). `update(frame)` returns a list of dicts `{id, bbox, cls, conf, mask}`; `cls` is always 0 (COCO person) — player/referee/goalkeeper roles are inferred later by `core/team_assigner.py`. In `main.py` it runs *before* the calibration so its boxes can be handed to BroadTrack as its `human-bboxes` input.

Turning a detection into a pitch position is `core/player_position.py`: `foot_point` is the box's float bottom-centre, and `PlayerPositionFilter` Kalman-filters each track's world position. The projection magnifies a pixel of foot jitter into 0.12 m at the near touchline and 0.61 m at the far goal line, so the filtering happens *after* projection, and both the BEV dots and `TeamAssigner`'s on-pitch verdict use the filtered point — the mask-based foot points that look more precise measured worse (0.170 vs 0.148 m/frame on test1).

### 3. Ball tracking — `core/ball_tracker.py`
`BallDetector` (YOLO, class 0 only) produces pixel detections; `get_candidates()` returns `BallCandidate`s that carry both the pixel box and the ground-plane BEV position (`pixel_to_ground`), plus a diagnostic-only `z_est` from `pixel_to_3d`.

`BallTracker` is a `filterpy` Kalman filter (`[x, y, vx, vy]` in BEV metres) driven by counters only: `miss_count` (frames since the last accepted measurement), `reject_count` (consecutive frames with plausible candidates but none through the gate) and `reacq_count` (consecutive candidates near the confirmation anchor). State is derived once per frame at the end, and the KF predicts exactly once per frame:

| state | meaning | output |
|---|---|---|
| `TENTATIVE` | no track yet; needs `reacquire_confirm` consecutive candidates | `None` until confirmed |
| `TRACKING` | measurement accepted this frame | position |
| `OCCLUDED` | `1 <= miss_count < N_occ`; predict only | predicted position (drawn hollow) |
| `LOST` | `miss_count >= N_occ`; prediction stops after `N_drop` | `None` |

Association has two tiers. Tier 1 is the Chi-Square test (`chi2.ppf(chi2_thres, df=2)`) capped at `gate_max_m` and updates the KF directly. Tier 2 is a **physical** bound — `ball_max_speed_mps × time since the last update + gate_margin_m` — and is what keeps a correctly detected fast ball alive after the filter has coasted; tier-2 candidates never update the filter directly and must pass the reacquisition confirmation. `BallTrackerConfig` holds every tunable, including the process noise (`process_noise_xy`, `process_noise_v`), which must be sized for a kicked ball rather than a pedestrian.

Candidate filtering is always-on for projection, pitch bounds and bbox shape. **Static pitch marks and player foot bands only gate new tracks** (`TENTATIVE`, `LOST`) and tier-2 recovery candidates; a tier-1 candidate that is consistent with the prediction is never discarded by them. This matters: a ball at a player's feet sits on the prediction, and filtering it there cost 14 frames of track in the test clip. Reacquisition arms after `reacquire_after` frames since the last update, needs `reacquire_confirm` candidates (allowed up to `reacquire_confirm_window` frames apart, so detector dropouts do not reset the run), and hard-resets the KF.

### Coordinate systems & projection — `core/projection_utils.py`
Read it before touching any projection code.
- **World coords**: meters, origin at pitch centre, x towards the right goal line, y towards the bottom touch line, ground plane `z=0`, and **z is negative above the ground** (the camera sits at negative z). This is BroadTrack's frame and the one PnLCalib used, so the two are interchangeable.
- `pixel_to_ground(u,v,K,R,t)` → ground (z=0) intersection using the **full pinhole model** (preferred over homography when camera tilt ≠ 0).
- `pixel_to_3d(u,v,bbox_width,K,R,t)` → full 3D position from apparent ball size; **currently unused** but is the planned path for aerial/high balls (see problems.md).
- `project_3d_to_pixel(points_3d,K,R,t)` applies `P = K[R | -Rt]`; only the tests use it, to synthesise detections.

## Docs
### `docs/`
- `problems.md`: solved and to be solved problems/bugs, some may have solutions and some may require deeper research
- `What Have I Learnt.md`: For academic report or sth like that

### READMEs
- `README.md`: Chinese ver.
- `README_EN.md`: English ver.