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
* the on-pitch verdict uses the same positions the BEV does.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.player_position import PlayerPositionFilter, foot_point
from core.projection_utils import project_3d_to_pixel
from core.team_assigner import TeamAssigner

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
        filtered.append(filt.update([detection(1, noisy)], K, R, T)[1])
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
        got = filt.update([detection(2, truth)], K, R, T)[2]
        if i >= 5:
            lagged.append(np.linalg.norm(np.array(got) - truth))
    assert max(lagged) < 0.10, f"the filter lagged a clean run by {max(lagged):.3f} m"


def test_filter_reports_only_what_was_detected_this_frame():
    filt = PlayerPositionFilter()
    assert list(filt.update([detection(7, (1.0, 1.0))], K, R, T)) == [7]
    assert filt.update([], K, R, T) == {}


def test_a_missing_track_is_coasted_then_forgotten():
    filt = PlayerPositionFilter(max_coast=3)
    filt.update([detection(7, (1.0, 1.0))], K, R, T)
    for _ in range(3):
        filt.update([], K, R, T)
    # still known: coming back a few frames later should be filtered, not fresh
    assert np.linalg.norm(np.array(filt.update([detection(7, (1.2, 1.0))], K, R, T)[7])
                          - np.array([1.2, 1.0])) < 0.30
    for _ in range(4):                      # past max_coast: starts from scratch
        filt.update([], K, R, T)
    assert filt.update([detection(7, (1.0, 1.0))], K, R, T)[7] == pytest.approx((1.0, 1.0), abs=1e-5)


def test_a_new_track_starts_at_its_measurement():
    filt = PlayerPositionFilter()
    assert filt.update([detection(5, (10.0, -4.0))], K, R, T)[5] == pytest.approx((10.0, -4.0))


def test_reset_forgets_every_track():
    filt = PlayerPositionFilter()
    filt.update([detection(1, (1.0, 1.0))], K, R, T)
    filt.reset()
    assert filt.update([detection(2, (2.0, 2.0))], K, R, T)[2] == pytest.approx((2.0, 2.0))


def test_a_detection_off_the_ground_plane_is_ignored():
    """A box whose ray points above the horizon has no ground position; the
    track must not be reported with a bogus one.

    The test1 camera never sees above its own horizon -- even the top of the
    frame lands 360 m out -- so this uses a camera pitched up instead.
    """
    filt = PlayerPositionFilter()
    looking_up = np.diag([1.0, -1.0, -1.0])
    obj = {"id": 1, "bbox": np.array([400.0, 240.0, 420.0, 270.0], dtype=np.float32)}
    assert filt.update([obj], K, looking_up, T) == {}


# --------------------------------------------------------- team assignment


def test_ground_position_prefers_the_filtered_position():
    obj = detection(3, (0.0, 0.0))
    raw = TeamAssigner.ground_position(obj, K, R, T)
    given = TeamAssigner.ground_position(obj, K, R, T, {3: (12.0, -7.0)})
    assert given == (12.0, -7.0)
    assert raw is not None and raw != given


def test_ground_position_falls_back_to_the_projection():
    obj = detection(4, (5.0, 5.0))
    assert TeamAssigner.ground_position(obj, K, R, T) == pytest.approx((5.0, 5.0), abs=0.01)


def test_positions_are_found_for_the_ids_the_detector_actually_produces():
    """``PlayerTracker`` hands out numpy integers, not Python ones. If the
    lookup missed them this would silently fall back to the raw projection --
    no error, just the jitter back."""
    obj = detection(np.int64(3), (0.0, 0.0))
    assert TeamAssigner.ground_position(obj, K, R, T, {3: (12.0, -7.0)}) == (12.0, -7.0)


def test_filter_on_pitch_uses_the_positions_it_is_given():
    """The verdict must follow whatever the caller drew, or a player can be on
    the BEV and off the pitch in the same frame."""
    assigner = TeamAssigner()
    on_pitch = detection(1, (0.0, 0.0))
    off_pitch = detection(2, (0.0, 0.0))
    positions = {1: (0.0, 0.0), 2: (0.0, 60.0)}     # 60 m out: off the pitch
    kept, dropped = assigner.filter_on_pitch([on_pitch, off_pitch], K, R, T, positions)
    assert [o["id"] for o in kept] == [1]
    assert dropped == {2}
