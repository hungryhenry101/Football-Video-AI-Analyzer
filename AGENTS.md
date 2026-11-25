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
- bootstrap a hard venue from one manually marked frame (interactive OpenCV window):
  ```bash
  python -m core.pitchcal.cli annotate \
      --video input_vids/test2.mp4 --frame 0 \
      --out config/anchors/test2.json --pitch 105x68
  python main.py --video input_vids/test2.mp4 --anchor config/anchors/test2.json \
      --start-frame 0 --max-frames 200 --no-display
  ```
  `--pitch` also accepts a non-regulation size (the arena's markings decide it; do
  not infer from the age group). `--anchor-free-camera` stops the camera centre
  from being pinned, for footage where the camera itself moves.
  The default backend is `pitchcal` (LK + sparse white-line evidence); pass
  `--calib-backend broadtrack` for the semantic line-segmentation solve every
  frame. Useful schedule knobs: `--player-interval 0.15` (seconds),
  `--line-every 4`, `--line-correct-every 20`, `--broadtrack-every N`
  (0 = BroadTrack only for the anchor/bootstrap).

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

### 0. Manual bootstrap (optional) — `core/pitchcal/` + `core/anchor_bootstrap.py`
BroadTrack's automatic bootstrap is the keypoint model, and it fails on low
camera angles and some youth footage. A human can instead mark one frame with
`python -m core.pitchcal.cli annotate` (point clicks and/or traced white lines),
which writes a **pitchcal anchor JSON**:

```json
{"pitch": [105, 68], "anchors": [{"frame": 7000, "width": 1920, "height": 1080,
  "H": [[...]], "points": [...], "lines": [...]}]}
```

Only the annotator/solver/tracker modules of PitchCal are vendored
(`geometry`, `solve`, `anchors`, `segments`, `linemask`, `annotate`,
`tracking`); its standalone BEV renderer and HTML report are not.

`core/anchor_bootstrap.py` converts an anchor to the runtime representation:
- it rescales `H` from the anchor's native frame to the processing resolution;
- the centred-principal decomposition is used when it reproduces `H` exactly
  (BroadTrack-derived anchors). A hand-fitted `H` often does **not** satisfy
  the square-pixel/centred constraints; forcing the centred solution shifts the
  whole projection by tens of pixels and was the cause of a visibly wrong first
  anchored frame. For those anchors the code scans the focal for the exact
  off-centre solution whose principal point is closest to the image centre
  (the constraints are 2 equations in `f, cx, cy`), so the ground projection
  matches the annotation exactly. This is why `Camera` carries an explicit
  `_principalPoint` (`setPrincipalPointOffset`); BroadTrack keeps it fixed
  while solving pan/tilt/focal;
- it picks the homography sign whose camera rays actually reach the ground
  (`_ray_valid_camera`), preferring the solution above the ground; this is what
  z_est and a BroadTrack keyframe need. Ground projection does not depend on
  it — that uses `H` directly;
- it validates the result (positive focal, orthonormal R, `t[2] < 0`).

With the default `--calib-backend pitchcal`, the anchor seeds the LK tracker
directly; BroadTrack's `set_camera` is used only for the `broadtrack` backend
or for `--broadtrack-every` keyframes. With `broadtrack`, `set_camera` pins the
camera centre, k1 and roll when `fix_position=True` (the default).

The anchor's pitch size is passed through to the calibration (C++
`Config.pitch_length/pitch_width` -> `SoccerPitch3D`), to the BEV template
(`pitch_wireframe(length, width)`), and to the downstream gates
(`PlayerPositionFilter`, `BallTrackerConfig`, `TeamAssigner`).

### 1. Calibration — two backends in `main.py`

**`--calib-backend pitchcal` (default)** — `core/pitch_tracker.py`
A `PitchHomographyTracker` wraps PitchCal's `GroundTracker` (LK + forward-
backward check + RANSAC + SIFT keyframes, ~10-20 ms/frame) and PitchCal's
white-line evidence (`segments.py`: LSD + "white paint on turf" filter,
`linemask.py`: oriented evidence). The schedule is PitchCal's:
- line evidence every `--line-every` frames (default 4): `detect_segments` ->
  `paint_like_segments` -> `filter_segments` -> `hit_from_segments`, giving
  `line_hit` and `line_coverage`;
- a gated homography correction every `--line-correct-every` frames (default
  20) via `refine_homography`; it is accepted only when the global paint
  support improves;
- the tracker carries the **homography**, and the rest of the pipeline projects
  through it (`homography_pixel_to_ground`). A camera `K, R, t` is derived per
  frame only for the diagnostic ball `z_est` and for keyframe seeding. This is
  deliberate: a hand-fitted `H` generally is not exactly representable by a
  pinhole camera, and converting it per frame would reintroduce the first-frame
  offset.
- BroadTrack runs only to bootstrap when there is no anchor, and every
  `--broadtrack-every N` frames as an absolute keyframe (`0` = only
  bootstrap/anchors). The keyframe is seeded from the tracked camera, then
  becomes the new reference for the LK tracker.
- `calib_score` is the tracker confidence (0.55 * inlier support +
  0.45 * line hit); `calib_valid` is false on a cut/lost track.

**`--calib-backend broadtrack`** — `core/broadtrack_calib/`
BroadTrack's temporal camera tracker, wrapped for Python from C++; it runs the
semantic line-segmentation model and Ceres every frame. `estimate(frame,
boxes=None)` returns `K, R, t, P` plus `score` (line-IoU), `reinit`, `k1` and
the pan/tilt/roll decomposition. The tracker is stateful. This is the
higher-accuracy, slower path, best for professional footage.

`PitchGeometry` owns the **BEV (top-down) representation**:
`create_bev_template()` builds a static pitch image, `world_to_bev_px(x, y)`
maps world metres -> BEV pixels (`bev_scale = 10 px/m`), and
`draw_pitch_lines()` projects the markings with a `P`. `main.py` draws the
wireframe through the current `H` with `draw_homography_lines()`, so both
backends share the projection path.

### 2. Player tracking — `core/player_tracker.py`
`PlayerTracker` wraps a YOLO-seg model (`model.track(..., tracker="config/botsort.yaml")`) giving BoT-SORT (handles GMC, ReID, Kalman). `update(frame)` returns a list of dicts `{id, bbox, cls, conf, mask}`; `cls` is always 0 (COCO person) — player/referee/goalkeeper roles are inferred later by `core/team_assigner.py`. In `main.py` it runs *before* the calibration so its boxes can be handed to the calibration as its "moving player" mask.

Detection is sparse: `--player-interval` seconds (default 0.15). On skipped
frames the previous detections are reused for the calibration's player masks,
the BEV dots and the team assignment; `PlayerPositionFilter` and
`TeamAssigner` are **not** updated again with the same measurement, and the
camera-view track overlay is only drawn on detection frames.

Turning a detection into a pitch position is `core/player_position.py`: `foot_point` is the box's float bottom-centre, and `PlayerPositionFilter` Kalman-filters each track's world position. The projection magnifies a pixel of foot jitter into 0.12 m at the near touchline and 0.61 m at the far goal line, so the filtering happens *after* projection, and both the BEV dots and `TeamAssigner`'s on-pitch verdict use the filtered point — the mask-based foot points that look more precise measured worse (0.170 vs 0.148 m/frame on test1).

### 3. Ball tracking — `core/ball_tracker.py`
`BallDetector` (YOLO, class 0 only) produces pixel detections; `get_candidates()` returns `BallCandidate`s that carry both the pixel box and the ground-plane BEV position (`homography_pixel_to_ground` on the pitchcal path, `pixel_to_ground` on the camera path), plus a diagnostic-only `z_est` from `pixel_to_3d` (camera path only).

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
- `homography_pixel_to_ground(u,v,H)` → ground intersection through a world→image ground homography. This is the default projection on the pitchcal backend: the LK tracker carries `H`, not a camera, and a hand-fitted `H` need not be exactly representable as a pinhole camera. `PlayerPositionFilter` and `BallDetector.get_candidates` accept either `H` or `K, R, t`.
- `pixel_to_3d(u,v,bbox_width,K,R,t)` → full 3D position from apparent ball size; **currently unused** but is the planned path for aerial/high balls (see problems.md).
- `project_3d_to_pixel(points_3d,K,R,t)` applies `P = K[R | -Rt]`; only the tests use it, to synthesise detections.

## Docs
### `docs/`
- `problems.md`: solved and to be solved problems/bugs, some may have solutions and some may require deeper research
- `What Have I Learnt.md`: For academic report or sth like that

### READMEs
- `README.md`: Chinese ver.
- `README_EN.md`: English ver.