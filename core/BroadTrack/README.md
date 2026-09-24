<div align=center>
<h1>BroadTrack: Broadcast Camera Tracking for Soccer</h1>


[![arXiv](https://img.shields.io/badge/arXiv-2404.11335-<COLOR>.svg)](https://arxiv.org/abs/2412.01721) 
</div>


>**[BroadTrack: Broadcast Camera Tracking for Soccer, WACV'25](https://arxiv.org/abs/2412.01721)**
>Floriane Magera, Thomas Hoyoux, Olivier Barnich, Marc Van Droogenbroeck


> **Note:** this directory is [BroadTrack](https://github.com/evs-broadcast/BroadTrack)
> vendored into [Football-Video-AI-Analyzer](../../README_EN.md)


---

## Using BroadTrack from Python

`core.broadtrack_calib` is this repo's pitch registration backend, and the only
part of BroadTrack that `main.py` uses. It runs BroadTrack's temporal camera
tracker — a per-frame Ceres bundle adjustment over segmented pitch lines,
regularised by Lucas-Kanade ground correspondences, with a line-IoU pan
refinement, a tripod position anchor and keypoint-driven re-initialisation.

```python
from core.broadtrack_calib import BroadTrackCalib

calib = BroadTrackCalib(
    weights_kp="models/nbjw_keypoint_model.pt",
    weights_line="models/tvcalib_model.pt",
    device="mps",            # or "cuda" / "cpu"
    width=960, height=540,
)

result = calib.estimate(frame, boxes=player_boxes)   # dict, or None
K, R, t, P = result["K"], result["R"], result["t"], result["P"]
```

Runnable examples: `main.py` (the full loop, with the boxes fed from the player
tracker) and `demos/broadtrack_demo.py` (minimal).

### Model weights

All four weights live in the repo-root `models/` (gitignored, ~690 MB, not
redistributed). `BroadTrackCalib` takes the first two as `weights_kp` /
`weights_line`:

| File | What it is |
| --- | --- |
| `models/tvcalib_model.pt` | line segmentation, 28 pitch-line classes — runs every frame |
| `models/nbjw_keypoint_model.pt` | 57 pitch keypoints — runs only on a re-initialisation |

The models can be downloaded from https://github.com/evs-broadcast/BroadTrack/tree/main/models

### What `estimate()` returns

- **`P = K[R | -R t]`**, in metres, origin at the pitch centre, `x` towards the
  right goal line, `y` towards the bottom touch line, `z = 0` on the ground and
  negative above it.
- Extra keys: `score` (line-IoU, 0-1), `reinit`, `frames_lost`, `k1`,
  `focal_length`, `pan_deg`, `tilt_deg`, `roll_deg`, `line_mask`.
- `estimate` returns `None` only for a numerically broken solve. The tracker
  always continues from the previous camera, so a bad frame yields the last good
  calibration rather than a failure. Watch `score`.
- BroadTrack solves radial distortion `k1`. The returned `K` is **not**
  undistorted; pass `k1` to `pixel_to_ground` if you want it applied.
- `boxes=` is optional but useful — it is BroadTrack's `human-bboxes` input, used
  to discard optical-flow points that landed on a moving player.
- The tracker is stateful: one instance tracks one continuous shot and is not
  thread-safe. Call `reset()` on a video cut.
- Rendering lives on `PitchGeometry` / `BroadTrackCalib`: `create_bev_template()`
  builds the static pitch image, `world_to_bev_px(x, y)` maps world metres to BEV
  pixels, and `draw_pitch_lines()` projects the markings with a given `P`.

### Gotchas

- **16:9 matters.** The segmentation model resizes to a fixed shortest side, so
  the line mask is only geometrically consistent with the frame at 16:9. A
  mismatch is reported on the first frame.
- **Feed full-resolution frames.** The tracker works in a fixed 1920x1080
  reference space and rescales everything at the boundary, so any frame size
  works — but do not pre-shrink to 1280x720 the way upstream's own CLI
  (`main.cpp`, `resizeFrameForProcessing`) does. That contradicts its own
  constants; measured on the reference sequence it costs ~0.18 line-IoU and
  leaves the pan estimate several degrees off.
- **Ceres asks for more threads than it has.** `CameraTracker::update` hardcodes
  `options.num_threads = 32`; with an 8-core machine Ceres logs a one-line
  warning per solve and clamps. Harmless, and the only line the bundled Ceres
  still prints.

### Performance

Measured on a 960x540 broadcast sequence (Apple M-series):

| Stage | Time |
| --- | --- |
| C++ tracker (point extraction, Ceres BA, line-IoU, optical flow) | ~25 ms |
| Line segmentation model (every frame) | ~0.5 s on MPS |
| Keypoint model (only when tracking is lost) | ~1.1 s on MPS, ~11.7 s on CPU |

The C++ half is not the bottleneck — the segmentation network is, exactly as
upstream, where it dominated the reported per-frame time. The keypoint model runs
only on a re-initialisation, so its cost is a one-off stall when the tracker has
already lost lock; it is ~10x faster on MPS, hence `keypoint_device="mps"` is
worth setting if you calibrate on CPU.

## Compiling

Needs a C++17 compiler, CMake >= 3.18, pybind11, Boost, RapidJSON, OpenCV 4 and
Ceres >= 2.1 (2.1 is the first release with `SubsetManifold`). OpenCV and Ceres
are built from source here rather than taken from a package manager — see below.

### The Python extension

This is the part `main.py` needs:

```bash
conda env create -f environment.yml && conda activate football
python scripts/build_minimal_opencv.py    # once, ~20 min
python scripts/build_minimal_ceres.py     # once, ~5 min
python scripts/build_native.py            # fast, rerun after any C++ change
```

`build_native.py` runs CMake over `core/BroadTrack/python/` and installs the
resulting `_broadtrack*.so` into `core/broadtrack_calib/`. The two
`build_minimal_*` scripts are one-time and cache their results under
`core/BroadTrack/third_party/` (gitignored); `build_native.py` picks them up
automatically, and if they are missing it falls back to the system OpenCV and
Ceres and says so — which on most platforms means the module will refuse to
import.

Only the numerical/geometric half is compiled. The two detectors and the CLI
entry points are deliberately excluded:

- the networks need libtorch, and the host process already has the Python `torch`
  package's own libtorch loaded. Linking a second, version-mismatched libtorch
  into the same process breaks the c10 operator registry. They run in Python
  instead — which `ultralytics` needs anyway;
- the CLI entry points need Boost.ProgramOptions and RapidJSON's writers, neither
  of which the library path uses.

| Stage | Where |
| --- | --- |
| Pitch line segmentation, pitch keypoints | Python (`core/broadtrack_calib/models.py`, TorchScript) |
| Point extraction, Ceres BA, line-IoU score | C++ (`_broadtrack`) |
| Optical-flow association, lost-tracking state machine | C++ (`_broadtrack`) |
| Projection, BEV rendering, pitch model | Python (`core/broadtrack_calib/calib.py`) |

### Why OpenCV and Ceres are built from source

Not for performance — for a single OpenMP runtime per process.

PyTorch's wheel ships its own OpenMP and that one cannot be removed, so anything
else in the process that also pulls an OpenMP runtime makes libomp abort the
interpreter:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
initialized.
```

Every packaged OpenCV and Ceres on macOS does exactly that:

* Homebrew's `ceres-solver` links SuiteSparse and OpenBLAS, which link
  `/opt/homebrew/opt/libomp/lib/libomp.dylib`;
* Homebrew's `opencv` and `opencv@4` both link `libopenblas` directly;
* conda-forge's `libopencv` lists `llvm-openmp` as a direct dependency, and
  conda-forge's `libopenblas` does too, so picking the pthreads BLAS variant
  does not help.

This all used to be survivable by accident. Up to PyTorch 2.9 the wheel stamped
its bundled OpenMP with the install name
`/opt/homebrew/opt/libomp/lib/libomp.dylib` — the same name as Homebrew's — so
dyld collapsed the two into a single image. PyTorch 2.14 stamps it
`/opt/llvm-openmp/lib/libomp.dylib`; the names no longer match, both copies load,
and the process dies.

So both dependencies are built here instead:

| Script | What it builds | Why that configuration |
| --- | --- | --- |
| `build_minimal_ceres.py` | Ceres 2.2.0, static, Eigen-only, MINIGLOG | No SuiteSparse, LAPACK, BLAS, glog or gflags — so no second OpenMP runtime and no duplicated gflags registry. BroadTrack solves one 8-parameter block against a few hundred residuals; Eigen's sparse Cholesky is far more than enough. |
| `build_minimal_opencv.py` | OpenCV 4.14.0, static, `core imgproc video` only | `WITH_OPENMP=OFF`, `WITH_LAPACK=OFF`, no imgcodecs/dnn/imgcodecs. Only the three modules the extension calls. |

Both are linked **statically** into the extension, which leaves it with no
OpenCV, Ceres, BLAS or OpenMP dynamic dependency at all. `libmetis` is the one
remaining non-system dependency: Ceres genuinely calls `METIS_NodeND` for
fill-reducing ordering, and Ceres' CMake exports that dependency unconditionally.
METIS uses no OpenMP, so it is harmless — `brew install metis` or your
distribution's equivalent is enough.

The two Ceres scripts also apply upstream patches, because Ceres 2.2.0 predates
Eigen 5 (the same two commits Homebrew's formula applies) and because
miniglog's `VLOG` has no level gate at all, which would print a dozen lines of
solver trace per frame.

**Caveat:** Ceres' build system warns that MINIGLOG "will likely cause problems
if glog is later linked". Nothing in this stack links glog, so the warning does
not apply here — but if you swap in a glog-based Ceres via `-DCeres_DIR=...`,
that is the risk you are taking.

### The standalone C++ binary

Upstream's own CLI, for reproducing upstream's numbers:

```bash
cmake -S core/BroadTrack -B core/BroadTrack/build -DLIBTORCH_DIR=<path to libtorch>
cmake --build core/BroadTrack/build -j
```

`CMakeLists.txt` builds two targets: `broadtrack` (from `main.cpp`, the tracker
CLI) and `reinit` (from `singleFrameCalibration.cpp`, single-frame calibration).
`LIBTORCH_DIR` defaults to the repo-local `core/BroadTrack/libtorch/`. On macOS
the build expects Homebrew's keg-only OpenCV 4 — `/opt/homebrew/opt/opencv@4` is
already on `CMAKE_PREFIX_PATH` — because OpenCV 5 is API-incompatible with this
codebase.

`run_broadtrack.sh` wraps the binary, setting `DYLD_LIBRARY_PATH` for the
repo-local libtorch and passing `--l` / `--k` from `models/`:

```bash
./core/BroadTrack/run_broadtrack.sh --f <image folder> --o <output json file>
```

Upstream's Docker image (**Install**, below) is the alternative route; it bakes
its own build and needs no local toolchain.

---

## Citation
Please cite our work if you use BroadTrack:
```bibtex
@inproceedings{Magera2025BroadTrack,
title = {BroadTrack: Broadcast Camera Tracking for Soccer},
author = {Magera, Floriane and Hoyoux, Thomas and Barnich, Olivier and Van Droogenbroeck, Marc},
booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
month = {February},
year = {2025},
address = {Tucson, Arizona, USA}
}
```
