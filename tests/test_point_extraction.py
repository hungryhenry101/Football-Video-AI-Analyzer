"""Regression tests for the pitch-line sample points the solver is built on.

``PointExtractor`` turns the segmentation model's label map into a few hundred
sample points, which are the *entire* input to the per-frame bundle adjustment.
Two things about it were wrong, and both are invisible unless you look for them:

* it seeded a fresh ``std::mt19937`` from ``std::random_device`` on **every**
  ``run()``, so two passes over the same video disagreed with each other and a
  single pass wobbled against itself (measured: 0.03 deg of pan, 0.03 deg of
  roll and 0.3% of focal between six solves of one identical frame);
* the loop that carries the previous frame's points over -- the only thing
  tying the residuals together in time -- was dead. It was guarded by
  ``_scaling == 1.``, but the points were stored already scaled into the
  540-row reference space, so with a 256-row mask (which is every mask: the
  segmentation model resizes its shortest side to 256) the guard never passed
  and every frame re-drew its sample positions from scratch.

``_points`` now lives in mask coordinates and the generator is a fixed-seed
member, so a frame reuses the last frame's points (re-snapped onto the current
mask) and identical input gives identical output.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.broadtrack_calib import BroadTrackCalib
from tests.conftest import KEYPOINT_WEIGHTS, LINE_WEIGHTS, requires_sequence

#: LineID codes the extractor accepts, from SoccerPitch3D. Any of the 150-173
#: range works; these are the halfway line and the two touch lines.
HALFWAY_LINE = 154
T_TOUCH_LINE = 151
B_TOUCH_LINE = 152


def pose_of(result):
    return (result["pan_deg"], result["tilt_deg"], result["roll_deg"],
            result["focal_length"])


def synthetic_line_mask(rows, cols):
    """A label map with three thick lines.

    Thick on purpose: ``PointExtractor::getSupportCenter`` rejects a candidate
    whose support disc is less than half full, and the disc has a 10 px radius
    in the 540-row reference space, so a hairline yields no points at all.
    """
    mask = np.zeros((rows, cols), dtype=np.uint8)
    thickness = max(16, rows // 20)
    cv2.line(mask, (cols // 2, 0), (cols // 2, rows - 1), HALFWAY_LINE, thickness)
    cv2.line(mask, (0, rows // 4), (cols - 1, rows // 4), T_TOUCH_LINE, thickness)
    cv2.line(mask, (0, 3 * rows // 4), (cols - 1, 3 * rows // 4), B_TOUCH_LINE, thickness)
    return mask


def run_over_frames(calib, frames):
    poses = []
    for _, frame in frames:
        result = calib.estimate(frame)
        assert result is not None, "the tracker refused a frame it used to accept"
        poses.append(pose_of(result))
    return np.array(poses)


# --------------------------------------------------------------------------- #
# the two properties S2 restored
# --------------------------------------------------------------------------- #

@requires_sequence
def test_two_passes_over_the_same_frames_calibrate_identically(calib, frames):
    """Same input, same output -- across a full reset.

    ``reset()`` throws the tracker away and rebuilds it, so the second pass is a
    genuinely independent run of the same code on the same frames. With the old
    per-frame ``std::random_device`` seeding the two passes picked different
    sample points and drifted apart immediately.

    The leading reset matters: ``calib`` is session-scoped and shared with the
    other calibration tests, so without it the "first" pass would start from
    whatever state they left the tracker in and the two halves would not be the
    same experiment.
    """
    try:
        calib.reset()
        first = run_over_frames(calib, frames[:8])
        calib.reset()
        second = run_over_frames(calib, frames[:8])
    finally:
        calib.reset()
    np.testing.assert_allclose(first, second, atol=1e-9)


@pytest.fixture(scope="module")
def still_calib():
    """A tracker with the optical-flow prior off.

    The optical-flow correspondences are unprojected through the *previous*
    frame's camera, so they are a feedback path of their own: with them on, a
    frozen mask still drifts by ~0.006 deg of pan per frame whatever the point
    extractor does, which would blur what this file is looking at. Turning them
    off leaves the extracted points as the only thing driving the solve.
    """
    calib = BroadTrackCalib(weights_kp=str(KEYPOINT_WEIGHTS),
                            weights_line=str(LINE_WEIGHTS),
                            device="cpu", width=1920, height=1080,
                            tracker_config={"optical_flow": False})
    yield calib
    calib.reset()


@requires_sequence
def test_reusing_the_previous_frames_points_steadies_a_still_frame(still_calib, frames, monkeypatch):
    """Carrying the points over is what stops a frozen mask from wandering.

    Not a fixed point -- once the reused points run out, ``run()`` tops the set
    up with fresh random ones until the suppression discs have covered the line,
    so a little variation is left. But it is far smaller than re-drawing every
    sample position from scratch, which is what this code effectively did before
    (see the module docstring). Measured over five repeats of one frame: pan
    0.004 vs 0.174 deg (45x), roll 0.006 vs 0.064 deg (11x), focal 0.7 vs 46 px
    (64x).

    ``BROADTRACK_NO_POINT_REUSE`` is the escape hatch that turns the carry-over
    off, so both halves run here, back to back, on the same tracker.
    """
    _, frame = frames[3]

    def spread_over_repeats():
        still_calib.estimate(frame)  # settle
        poses = np.array([pose_of(still_calib.estimate(frame)) for _ in range(5)])
        return poses.max(axis=0) - poses.min(axis=0)

    monkeypatch.delenv("BROADTRACK_NO_POINT_REUSE", raising=False)
    with_reuse = spread_over_repeats()

    still_calib.reset()
    monkeypatch.setenv("BROADTRACK_NO_POINT_REUSE", "1")
    without_reuse = spread_over_repeats()

    assert with_reuse[0] < without_reuse[0] / 3, (
        f"carrying the points over bought nothing: pan spread {with_reuse[0]:.6f} deg "
        f"with reuse vs {without_reuse[0]:.6f} deg without"
    )
    assert with_reuse[0] < 2e-2, f"pan wandered {with_reuse[0]:.6f} deg on a repeated frame"


# --------------------------------------------------------------------------- #
# the reuse path's precondition
# --------------------------------------------------------------------------- #

def test_the_tracker_survives_a_change_of_mask_size(extension):
    """Points carried over from a mask of a different size must be dropped.

    They live in mask coordinates, so indexing the new mask with them would run
    off the end -- and ``cv::Mat::at`` does not bounds-check outside debug
    builds. Blank masks would not exercise this: with no points there is
    nothing to carry over.
    """
    tracker = extension.Tracker({"hd_width": 1920.0, "hd_height": 1080.0})
    sizes = [(540, 960), (1080, 1920), (256, 455), (540, 960)]

    for rows, cols in sizes:
        frame = np.zeros((rows, cols, 3), dtype=np.uint8)
        mask = np.ascontiguousarray(synthetic_line_mask(rows, cols))
        result = tracker.estimate(frame, mask, np.zeros((0, 4)), None)
        assert np.isfinite(result["K"]).all()
        assert np.isfinite(result["t"]).all()
        assert np.isfinite(result["R"]).all()
