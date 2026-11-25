"""The pitchcal LK tracker wrapper: reset/rebase semantics."""

from __future__ import annotations

import numpy as np
import pytest

from core.anchor_bootstrap import homography_from_camera
from core.pitch_tracker import PitchHomographyTracker, PitchTrackConfig

# A real 960x540 camera (frame 0 of input_vids/test1.mp4, as BroadTrack solved it).
FOCAL, PAN, TILT, ROLL = 1179.2, 2.148, 74.801, -0.280
POSITION = np.array([-1.813, 73.705, -17.503])


def _camera():
    p, t, r = np.radians([PAN, TILT, ROLL])
    rot_tilt = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])
    rot_pan = np.array([[np.cos(p), -np.sin(p), 0.0], [np.sin(p), np.cos(p), 0.0], [0.0, 0.0, 1.0]])
    rot_roll = np.array([[np.cos(r), -np.sin(r), 0.0], [np.sin(r), np.cos(r), 0.0], [0.0, 0.0, 1.0]])
    K = np.array([[FOCAL, 0.0, 480.0], [0.0, FOCAL, 270.0], [0.0, 0.0, 1.0]])
    return K, (rot_pan @ rot_tilt @ rot_roll).T, POSITION


K, R, T = _camera()
H = homography_from_camera(K, R, T)


def _tracker(homography=H):
    return PitchHomographyTracker(
        homography, 960, 540, (105.0, 68.0),
        principal=(K[0, 2], K[1, 2]), focal=float(K[0, 0]),
        config=PitchTrackConfig(line_every=0, correct_every=0),
    )


def test_first_update_preserves_the_anchor_homography():
    tracker = _tracker()
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    result = tracker.update(frame)

    assert result.valid
    assert np.allclose(result.homography, H)
    assert np.isfinite(result.K).all()
    assert np.isfinite(result.R).all()
    assert np.isfinite(result.t).all()
    assert result.K[0, 2] == pytest.approx(480.0)
    assert result.K[1, 2] == pytest.approx(270.0)


def test_reset_rebases_on_the_new_homography():
    tracker = _tracker()
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    tracker.update(frame)

    shifted = H.copy()
    shifted[0, 2] += 10.0
    tracker.reset(shifted)
    result = tracker.update(frame)

    assert np.allclose(result.homography, shifted)
    assert result.valid


def test_boxes_are_optional_and_accept_lists_or_arrays():
    tracker = _tracker()
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    tracker.update(frame, boxes=None)
    tracker.reset(H)
    tracker.update(frame, boxes=[[10.0, 10.0, 50.0, 90.0]])
