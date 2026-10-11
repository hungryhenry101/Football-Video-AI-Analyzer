"""Tests for the zero-phase calibration smoother (``broadtrack_calib.smoothing``).

The filter itself is pure numpy/scipy, so most of this runs without the models.
The tests that need the real pipeline are marked accordingly.

The properties that matter, roughly in order:

* the streamed output is *exactly* an offline Savitzky-Golay filter -- no
  hand-rolled edge behaviour that quietly differs from ``scipy``;
* every frame comes back exactly once, ``lag`` frames late, so a caller pairing
  frames with results cannot drift out of step;
* a genuine camera pan survives untouched. A smoother that eats the pan is
  worse than none -- that is what the causal median filter in C++ measurably
  does (line-IoU minimum 0.63 -> 0.55 over 130 frames of test1.mp4);
* a re-initialisation is never smoothed across: the jump is deliberate.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import savgol_filter

from core.broadtrack_calib import (
    BroadTrackCalib,
    CalibrationSmoother,
    pose_from_parameters,
)
from core.broadtrack_calib.smoothing import SMOOTHED_KEYS
from tests.conftest import (
    KEYPOINT_WEIGHTS,
    LINE_WEIGHTS,
    requires_models,
    requires_sequence,
)

WINDOW = 17
POLY = 3
LAG = WINDOW // 2


def synthetic_result(index, pan=0.0, tilt=75.0, roll=0.0, focal=1000.0,
                     reinit=False, score=0.6):
    """A calibration dict shaped like ``BroadTrackCalib.estimate``'s output.

    ``index`` is carried through unsmoothed; it is how the tests know which
    frame an emitted result belongs to.
    """
    K = np.array([[focal, 0.0, 480.0], [0.0, focal, 270.0], [0.0, 0.0, 1.0]])
    t = np.array([-1.3, 73.8, -17.9])
    R = np.eye(3)
    It = np.eye(4)[:3]
    It[:, 3] = -t
    return {
        "P": K @ (R @ It), "K": K, "R": R, "t": t, "k1": 0.01,
        "focal_length": focal, "pan_deg": pan, "tilt_deg": tilt, "roll_deg": roll,
        "score": score, "reinit": reinit, "frames_lost": 0, "line_mask": None,
        "index": index,
    }


def stream(smoother, results):
    """Push every result through and return the raw list of emissions."""
    return [smoother.push(result) for result in results]


def noisy_ramp(n, slope, noise=0.04, seed=0):
    """A real camera move (a ramp) plus the per-frame wander we want gone."""
    rng = np.random.default_rng(seed)
    return slope * np.arange(n) + rng.normal(0.0, noise, n)


def targets(emitted):
    """The frame index each non-None emission describes."""
    return [entry["index"] for entry in emitted if entry is not None]


# --------------------------------------------------------------------------- #
# the filter itself
# --------------------------------------------------------------------------- #

def test_streamed_output_is_exactly_an_offline_savgol_filter():
    n = 200
    pans = noisy_ramp(n, 0.05)
    emitted = stream(CalibrationSmoother(WINDOW, POLY),
                     [synthetic_result(i, pan=float(p)) for i, p in enumerate(pans)])

    # Nothing can be emitted until `lag` frames of lookahead exist.
    assert emitted[:LAG] == [None] * LAG

    got = np.array([entry["pan_deg"] for entry in emitted[LAG:]])
    reference = savgol_filter(pans, WINDOW, POLY, mode="interp")

    # Warm-up: the window is not full yet, so the raw solve is passed through
    # rather than a one-sided fit that would lag.
    np.testing.assert_allclose(got[:LAG], pans[:LAG], atol=1e-12)
    # Interior: bit-for-bit the offline filter.
    np.testing.assert_allclose(got[LAG:], reference[: n - LAG][LAG:], atol=1e-9)


def test_every_frame_comes_back_exactly_once_lag_frames_late():
    n = 60
    smoother = CalibrationSmoother(WINDOW, POLY)
    emitted = stream(smoother, [synthetic_result(i) for i in range(n)])

    assert sum(entry is None for entry in emitted) == LAG
    assert targets(emitted) == list(range(n - LAG))
    # ... and the tail comes back from flush(), still in order.
    assert [entry["index"] for entry in smoother.flush()] == list(range(n - LAG, n))


def test_a_steady_pan_passes_through_without_bias():
    """A degree-3 fit reproduces a cubic exactly, so a constant-velocity pan is
    untouched. This is the property the causal median filter in C++ lacks: it
    lagged the pan and cost the tracker measurable line-IoU."""
    pans = 0.08 * np.arange(80) - 4.0
    emitted = stream(CalibrationSmoother(WINDOW, POLY),
                     [synthetic_result(i, pan=float(p)) for i, p in enumerate(pans)])
    got = np.array([entry["pan_deg"] for entry in emitted[LAG:]])
    np.testing.assert_allclose(got[LAG:], pans[: len(got)][LAG:], atol=1e-9)


def test_jitter_is_removed_but_the_ramp_is_kept():
    pans = noisy_ramp(300, 0.05, seed=7)
    emitted = stream(CalibrationSmoother(WINDOW, POLY),
                     [synthetic_result(i, pan=float(p)) for i, p in enumerate(pans)])
    got = np.array([entry["pan_deg"] for entry in emitted[LAG:]])

    def roughness(x):
        return np.abs(np.diff(x, 2)).mean()

    assert roughness(got[LAG:]) < 0.25 * roughness(pans[: len(got)][LAG:])
    # The real motion is a 0.05 deg/frame ramp; it must still be there.
    assert np.polyfit(np.arange(len(got)), got, 1)[0] == pytest.approx(0.05, abs=2e-3)


def test_all_four_parameters_are_smoothed_and_the_pose_is_rebuilt():
    smoother = CalibrationSmoother(WINDOW, POLY)
    results = [
        synthetic_result(i,
                         pan=float(noisy_ramp(1, 0.1, seed=i)[0]),
                         tilt=75.0 + float(noisy_ramp(1, 0.01, seed=1000 + i)[0]),
                         roll=-0.5 + float(noisy_ramp(1, 0.002, seed=2000 + i)[0]),
                         focal=1000.0 + float(noisy_ramp(1, 3.0, seed=3000 + i)[0]))
        for i in range(60)
    ]
    emitted = [entry for entry in stream(smoother, results) if entry is not None]
    last = emitted[-1]
    raw = results[last["index"]]

    assert any(last[key] != raw[key] for key in SMOOTHED_KEYS), "nothing was smoothed"
    # ... and K/R/P describe the *smoothed* scalars, not the raw ones.
    K, R, t, P = pose_from_parameters(last)
    np.testing.assert_allclose(K, last["K"], atol=1e-12)
    np.testing.assert_allclose(R, last["R"], atol=1e-12)
    np.testing.assert_allclose(t, last["t"], atol=1e-12)
    np.testing.assert_allclose(P, last["P"], atol=1e-12)


# --------------------------------------------------------------------------- #
# the awkward frames
# --------------------------------------------------------------------------- #

def test_a_refused_solve_carries_through_the_delay_line():
    smoother = CalibrationSmoother(WINDOW, POLY)
    results = [synthetic_result(i) for i in range(40)]
    results[10] = None
    emitted = stream(smoother, results)

    assert emitted[10 + LAG] is None
    # Every other frame still comes back, once, in order.
    assert targets(emitted) == [i for i in range(40 - LAG) if i != 10]


def test_a_reinitialisation_is_not_smoothed_across():
    smoother = CalibrationSmoother(WINDOW, POLY)
    results = [synthetic_result(i, pan=float(noisy_ramp(1, 0.02, seed=i)[0]))
               for i in range(80)]
    results[40]["reinit"] = True
    results[40]["pan_deg"] += 3.0  # the deliberate pose jump

    emitted = stream(smoother, results)
    got = {entry["index"]: entry["pan_deg"] for entry in emitted if entry is not None}
    raw = {entry["index"]: entry["pan_deg"] for entry in results}

    # Frames whose window never reaches the jump are unaffected by it.
    assert any(got[i] != pytest.approx(raw[i], abs=1e-9)
               for i in range(2 * LAG, 40 - LAG))
    # Only a window that straddles the jump is refused, so LAG frames either
    # side of it come back unsmoothed -- the jump is not smeared over them.
    for index in range(40 - LAG, 40 + LAG):
        assert got[index] == pytest.approx(raw[index], abs=1e-12)
    # Once the window starts at or after the re-init, smoothing resumes.
    assert got[40 + LAG] != pytest.approx(raw[40 + LAG], abs=1e-9)


def test_flush_is_empty_when_nothing_is_buffered():
    assert CalibrationSmoother(WINDOW, POLY).flush() == []


def test_reset_forgets_everything():
    smoother = CalibrationSmoother(WINDOW, POLY)
    stream(smoother, [synthetic_result(i) for i in range(30)])
    smoother.reset()
    assert smoother.flush() == []
    assert smoother.push(synthetic_result(0)) is None  # back to warm-up


@pytest.mark.parametrize("window,poly", [(0, 3), (4, 3), (17, 0), (17, 17)])
def test_nonsense_parameters_are_rejected(window, poly):
    with pytest.raises(ValueError):
        CalibrationSmoother(window, poly)


# --------------------------------------------------------------------------- #
# against the real calibration
# --------------------------------------------------------------------------- #

@requires_sequence
def test_pose_reconstruction_reproduces_the_extension(calib, frames):
    """``pose_from_parameters`` must invert what the C++ wrote into K/R/P.

    Everything the smoother emits is rebuilt with this function, so a sign or
    transpose error here would silently move the whole pitch.
    """
    for _, frame in frames[:5]:
        result = calib.estimate(frame)
        assert result is not None
        K, R, t, P = pose_from_parameters(result)
        np.testing.assert_allclose(K, result["K"], atol=1e-9)
        np.testing.assert_allclose(R, result["R"], atol=1e-9)
        np.testing.assert_allclose(t, result["t"], atol=1e-12)
        np.testing.assert_allclose(P, result["P"], atol=1e-9)


@requires_models
def test_smoothing_is_off_by_default():
    calib = BroadTrackCalib(weights_kp=str(KEYPOINT_WEIGHTS),
                            weights_line=str(LINE_WEIGHTS),
                            device="cpu", width=960, height=540)
    assert calib.lag == 0
    assert calib.flush() == []


@requires_sequence
def test_estimate_holds_back_the_stream_when_smoothing_is_on(frames):
    """The end-to-end wiring: ``estimate`` delays, ``flush`` gives the tail back."""
    height, width = frames[0][1].shape[:2]
    calib = BroadTrackCalib(weights_kp=str(KEYPOINT_WEIGHTS),
                            weights_line=str(LINE_WEIGHTS),
                            device="cpu", width=width, height=height,
                            smooth_window=WINDOW)
    # Frame 0 initialises from keypoints, and the keypoint model costs ~12 s on
    # CPU. Not what this test is about, so stub it out.
    calib._keypoint_model.compute_keypoints = lambda frame: []

    emissions = [calib.estimate(frame) for _, frame in frames[:12]]

    assert calib.lag == LAG
    assert emissions[:LAG] == [None] * LAG, "output must not lead the delay line"
    assert any(entry is not None for entry in emissions), "nothing ever came out"
    assert len(calib.flush()) == LAG, "flush must hand back exactly the held frames"
