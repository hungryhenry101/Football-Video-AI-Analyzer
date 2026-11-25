"""Tests for the player foot point and its temporal filter.

``core.player_position`` is pure numpy, so nothing here needs the models. The
camera used is a real one -- frame 0 of ``input_vids/test1.mp4``, recorded from
``BroadTrackCalib`` -- because the whole point of the filter is how the
projection magnifies a pixel of foot jitter, and a made-up camera would not
reproduce that.

The properties that matter, roughly in order:

* the foot point is the box's bottom-centre and nothing else. The mask-based
  alternatives were measured and are worse (see the module docstring);
* the filter removes jitter from a noisy stationary track;
* the filter does *not* lag a moving player. That is what rules out the simpler
  EMA: on a clean 7 m/s run EMA(0.5) sits 0.28 m behind, which is the size of
  the jitter it removes, except that it is a bias;
* a track that was not detected this frame is not reported, so the caller draws
  what the detector saw;
* the on-pitch split is the filter's own verdict, so the BEV and the team
  assigner cannot disagree about it.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.player_position import (
    PITCH_LENGTH,
    PITCH_MARGIN,
    PITCH_WIDTH,
    PlayerPositionFilter,
    foot_point,
    inside_pitch,
)
from core.projection_utils import project_3d_to_pixel

#: Frame 0 of input_vids/test1.mp4, as BroadTrackCalib solved it: 960x540.
FOCAL, PAN, TILT, ROLL = 1179.2, 2.148, 74.801, -0.280
POSITION = np.array([-1.813, 73.705, -17.503])


def camera():
    p, t, r = np.radians([PAN, TILT, ROLL])
    rot_tilt = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])
    rot_pan = np.array([[np.cos(p), -np.sin(p), 0.0], [np.sin(p), np.cos(p), 0.0], [0.0, 0.0, 1.0]])
    rot_roll = np.array([[np.cos(r), -np.sin(r), 0.0], [np.sin(r), np.cos(r), 0.0], [0.0, 0.0, 1.0]])
    K = np.array([[FOCAL, 0.0, 480.0], [0.0, FOCAL, 270.0], [0.0, 0.0, 1.0]])
    return K, (rot_pan @ rot_tilt @ rot_roll).T, POSITION


K, R, T = camera()


def detection(track_id, world_xy, width=30.0, height=70.0):
    """A detection whose foot point lands on ``world_xy`` (ground metres)."""
    u, v = project_3d_to_pixel(np.array([world_xy[0], world_xy[1], 0.0]), K, R, T)
    return {"id": track_id, "bbox": np.array([u - width / 2, v - height, u + width / 2, v],
                                             dtype=np.float32)}


# --------------------------------------------------------------- foot point


def test_foot_point_is_the_float_bottom_centre():
    obj = {"id": 1, "bbox": np.array([10.4, 20.7, 30.6, 80.9], dtype=np.float32)}
    u, v = foot_point(obj)
    assert u == pytest.approx((10.4 + 30.6) / 2)
    assert v == pytest.approx(80.9)


def test_foot_point_keeps_the_sub_pixel_part_the_old_code_dropped():
    """``int()`` on the box bottom threw away half a pixel -- 0.3 m at the far
    goal line, where a pixel of foot jitter is worth 0.6 m."""
    obj = {"id": 1, "bbox": np.array([0.0, 0.0, 10.0, 80.9], dtype=np.float32)}
    assert foot_point(obj)[1] == pytest.approx(80.9)
    assert int(foot_point(obj)[1]) == 80          # what the old code returned


def test_foot_point_does_not_use_the_mask():
    """A mask that disagrees with the box must not move the foot point: the
    mask bottom was measured to be the noisier of the two."""
    mask = np.zeros((80, 20), np.uint8)
    mask[30:79, :] = 1                      # mask bottom at local row 78
    obj = {"id": 1, "bbox": np.array([0.0, 0.0, 20.0, 80.0], dtype=np.float32), "mask": mask}
    assert foot_point(obj) == (10.0, 80.0)


# ------------------------------------------------------------------ filter


def test_filter_removes_noise_from_a_stationary_player():
    rng = np.random.default_rng(0)
    filt = PlayerPositionFilter()
    raw, filtered = [], []
    for i in range(120):
        noisy = rng.normal(0.0, 0.35, 2)
        raw.append(noisy)
        filtered.append(filt.update([detection(1, noisy)], K, R, T)[0][1])
    raw, filtered = np.array(raw[10:]), np.array(filtered[10:])
    assert np.linalg.norm(filtered.std(axis=0)) < np.linalg.norm(raw.std(axis=0)) * 0.85
    # and it stays centred on the truth rather than drifting off it
    assert np.linalg.norm(filtered.mean(axis=0)) < 0.15


def test_filter_does_not_lag_a_moving_player():
    """The property that rules out EMA and a causal median: a player running at
    a constant 7 m/s comes back where they are, not where they were."""
    filt = PlayerPositionFilter()
    lagged = []
    for i in range(60):
        truth = np.array([-20.0 + 0.28 * i, 0.0])
        got = filt.update([detection(2, truth)], K, R, T)[0][2]
        if i >= 5:
            lagged.append(np.linalg.norm(np.array(got) - truth))
    assert max(lagged) < 0.10, f"the filter lagged a clean run by {max(lagged):.3f} m"


def test_filter_reports_only_what_was_detected_this_frame():
    filt = PlayerPositionFilter()
    on_pitch, off_pitch = filt.update([detection(7, (1.0, 1.0))], K, R, T)
    assert list(on_pitch) == [7]
    assert off_pitch == set()
    assert filt.update([], K, R, T) == ({}, set())


def test_a_missing_track_is_coasted_then_forgotten():
    filt = PlayerPositionFilter(max_coast=3)
    filt.update([detection(7, (1.0, 1.0))], K, R, T)
    for _ in range(3):
        filt.update([], K, R, T)
    # still known: coming back a few frames later should be filtered, not fresh
    assert np.linalg.norm(np.array(filt.update([detection(7, (1.2, 1.0))], K, R, T)[0][7])
                          - np.array([1.2, 1.0])) < 0.30
    for _ in range(4):                      # past max_coast: starts from scratch
        filt.update([], K, R, T)
    assert filt.update([detection(7, (1.0, 1.0))], K, R, T)[0][7] == pytest.approx((1.0, 1.0), abs=1e-5)


def test_a_new_track_starts_at_its_measurement():
    filt = PlayerPositionFilter()
    assert filt.update([detection(5, (10.0, -4.0))], K, R, T)[0][5] == pytest.approx((10.0, -4.0))


def test_reset_forgets_every_track():
    filt = PlayerPositionFilter()
    filt.update([detection(1, (1.0, 1.0))], K, R, T)
    filt.reset()
    assert filt.update([detection(2, (2.0, 2.0))], K, R, T)[0][2] == pytest.approx((2.0, 2.0))


def test_a_detection_off_the_ground_plane_is_not_on_pitch():
    """A box whose ray points above the horizon has no ground position; the
    track must not be reported with a bogus one.

    The test1 camera never sees above its own horizon -- even the top of the
    frame lands 360 m out -- so this uses a camera pitched up instead.
    """
    filt = PlayerPositionFilter()
    looking_up = np.diag([1.0, -1.0, -1.0])
    obj = {"id": 1, "bbox": np.array([400.0, 240.0, 420.0, 270.0], dtype=np.float32)}
    assert filt.update([obj], K, looking_up, T) == ({}, {1})


# -------------------------------------------------------------- on-pitch gate


def test_update_accepts_the_equivalent_homography():
    """The pitchcal path projects through H; the result must match K/R/t."""
    from core.anchor_bootstrap import homography_from_camera

    H = homography_from_camera(K, R, T)
    detection_ = detection(3, (2.0, 3.0))
    camera_result = PlayerPositionFilter().update([detection_], K, R, T)[0]
    homography_result = PlayerPositionFilter().update([detection_], H=H)[0]

    assert homography_result.keys() == camera_result.keys()
    for track_id in camera_result:
        assert homography_result[track_id] == pytest.approx(
            camera_result[track_id], abs=1e-6)


def test_inside_pitch_grants_the_margin_on_every_line():
    assert inside_pitch((0.0, 0.0))
    assert inside_pitch((PITCH_LENGTH / 2 + PITCH_MARGIN, 0.0))
    assert not inside_pitch((PITCH_LENGTH / 2 + PITCH_MARGIN + 0.01, 0.0))
    assert inside_pitch((0.0, -(PITCH_WIDTH / 2 + PITCH_MARGIN)))
    assert not inside_pitch((0.0, -(PITCH_WIDTH / 2 + PITCH_MARGIN + 0.01)))
    assert not inside_pitch(None)


def test_off_pitch_detections_are_split_out():
    filt = PlayerPositionFilter()
    on_pitch, off_pitch = filt.update(
        [detection(1, (0.0, 0.0)), detection(2, (0.0, 40.0))], K, R, T
    )
    assert on_pitch.keys() == {1}
    assert off_pitch == {2}


def test_an_off_pitch_track_keeps_its_filter_state():
    """Stepping over the touchline must not restart the Kalman filter, or the
    player comes back with a fresh measurement's jitter."""
    filt = PlayerPositionFilter()
    for i in range(30):                     # build up +y velocity inside the pitch
        filt.update([detection(1, (0.0, float(i)))], K, R, T)
    for i in range(30, 40):                 # walk over the touchline
        _, off_pitch = filt.update([detection(1, (0.0, float(i)))], K, R, T)
    assert off_pitch == {1}                 # the last foot was 5 m out
    got = filt.update([detection(1, (0.0, 25.0))], K, R, T)[0][1]
    # a fresh track would return the measurement exactly; the retained +y
    # velocity pulls the estimate past it
    assert got[1] > 25.0
