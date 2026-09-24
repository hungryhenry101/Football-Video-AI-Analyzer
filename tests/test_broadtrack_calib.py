"""End-to-end calibration tests.

These need BroadTrack's TorchScript detectors and a frame sequence; they skip
when either is missing. ``core/BroadTrack/frames/`` is the 1920x1080 sequence the
upstream C++ binary was run on, and ``core/BroadTrack/out/anchored_pos.json``
holds the per-frame scores it recorded, so most of these are genuine regression
checks against the reference implementation rather than self-consistency checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_models, requires_sequence


@requires_models
def test_estimate_returns_a_usable_camera(calib, frames):
    name, frame = frames[0]
    result = calib.estimate(frame)
    assert result is not None, f"no calibration for {name}"

    assert result["K"].shape == (3, 3)
    assert result["R"].shape == (3, 3)
    assert result["t"].shape == (3,)
    assert result["P"].shape == (3, 4)
    assert np.isfinite(result["K"]).all()
    assert np.isfinite(result["R"]).all()
    assert np.isfinite(result["t"]).all()

    # R must be a rotation.
    assert np.allclose(result["R"] @ result["R"].T, np.eye(3), atol=1e-6)
    assert np.linalg.det(result["R"]) == pytest.approx(1.0, abs=1e-6)

    # P = K [R | -R t], the convention shared with core.projection_utils.
    It = np.eye(4)[:3]
    It[:, 3] = -result["t"]
    assert np.allclose(result["P"], result["K"] @ (result["R"] @ It))


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (960, 540), (640, 360)])
def test_intrinsics_scale_with_the_frame_resolution(extension, size):
    """The returned K must describe the caller's frame, at any frame size.

    Regression test: the HD -> frame rescale of the focal length and principal
    point used the reciprocal of the right factor. At 1920x1080 that factor is
    1.0, so every other test passed while a 960x540 caller -- which is what
    main.py feeds in -- got a principal point of (1920, 1080) and a focal length
    squared, projecting the whole pitch off screen.

    Uses a blank mask and no models: the scaling is applied to whatever camera
    the solve ends up with, so the solve's quality is irrelevant here.
    """
    width, height = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    tracker = extension.Tracker({"hd_width": 1920.0, "hd_height": 1080.0})

    result = tracker.estimate(frame, mask, np.zeros((0, 4)), None)

    assert result["K"][2, 2] == 1.0, "the homogeneous entry must not be rescaled"
    assert result["K"][0, 2] == pytest.approx(width / 2)
    assert result["K"][1, 2] == pytest.approx(height / 2)
    assert result["K"][0, 0] == pytest.approx(result["focal_length"])
    assert result["K"][1, 1] == pytest.approx(result["focal_length"])
    assert result["focal_length"] > 0


@requires_models
def test_intrinsics_are_in_frame_pixels(calib, frames):
    """Focal length and principal point must come back at the caller's
    resolution, not the tracker's internal 1920x1080."""
    name, frame = frames[0]
    result = calib.estimate(frame)
    height, width = frame.shape[:2]

    assert result["K"][0, 0] == pytest.approx(result["focal_length"])
    assert result["K"][2, 2] == 1.0, "the homogeneous entry must not be rescaled"
    assert result["K"][0, 2] == pytest.approx(width / 2)
    assert result["K"][1, 2] == pytest.approx(height / 2)
    # A broadcast camera at this framing sits around 2500-2900 px at 1080p.
    assert 0.8 * width < result["focal_length"] < 2.0 * width


@requires_models
def test_camera_is_above_the_ground_and_near_the_pitch(calib, frames):
    result = calib.estimate(frames[0][1])
    x, y, z = result["t"]
    assert z < 0, "z is negative upwards; the camera must be above the ground plane"
    assert -40.0 < z < -5.0, f"implausible camera height: {z:.1f} m"
    assert -PITCH_MARGIN < x < PITCH_MARGIN
    assert -PITCH_MARGIN < y < PITCH_MARGIN


PITCH_MARGIN = 150.0


@requires_sequence
def test_scores_track_the_reference_binary(calib, frames, reference):
    """The whole point of the port: same input, same quality.

    The reference recorded a line-IoU score per frame. Ours must match it
    closely — a loose bound would pass even if the mask conversion or the
    coordinate scaling drifted, so this is deliberately tight.
    """
    checked = 0
    for name, frame in frames:
        entry = reference.get(f"frames/{name}")
        if entry is None:
            continue
        result = calib.estimate(frame)
        assert result is not None, f"no calibration for {name}"
        assert result["score"] == pytest.approx(entry["score"], abs=0.03), (
            f"{name}: score {result['score']:.3f} vs reference {entry['score']:.3f}"
        )
        checked += 1
    assert checked >= 10, "not enough reference frames to compare against"


@requires_sequence
def test_pan_tilt_match_the_reference_binary(calib, frames, reference):
    for name, frame in frames[:10]:
        entry = reference.get(f"frames/{name}")
        if entry is None:
            continue
        result = calib.estimate(frame)
        camera = entry["cp"]
        assert result["pan_deg"] == pytest.approx(camera["panDegrees"], abs=0.5), name
        assert result["tilt_deg"] == pytest.approx(camera["tiltDegrees"], abs=0.5), name


@requires_sequence
def test_calibration_is_temporally_smooth(calib, frames):
    """BroadTrack is a tracker, so consecutive frames must not jump.

    This is the property PnLCalib lacked: its per-frame solves jittered even when
    the camera was still.
    """
    poses = []
    for _, frame in frames:
        result = calib.estimate(frame)
        assert result is not None
        poses.append((result["pan_deg"], result["tilt_deg"], result["focal_length"]))

    pans = np.array([p[0] for p in poses])
    tilts = np.array([p[1] for p in poses])
    focals = np.array([p[2] for p in poses])

    assert np.abs(np.diff(pans)).max() < 1.0, "pan jumped between consecutive frames"
    assert np.abs(np.diff(tilts)).max() < 1.0, "tilt jumped between consecutive frames"
    assert np.abs(np.diff(focals) / focals[:-1]).max() < 0.05, "focal length jumped"


@requires_sequence
def test_player_boxes_do_not_break_the_solve(calib, frames):
    """Player boxes only feed the optical-flow rejection, so the camera must
    come out essentially unchanged with and without them."""
    name, frame = frames[5]
    height, width = frame.shape[:2]

    without = calib.estimate(frame)
    calib.reset()
    calib.estimate(frame)

    boxes = np.array([[0.3 * width, 0.6 * height, 0.4 * width, 0.9 * height]], dtype=np.float64)
    with_boxes = calib.estimate(frame, boxes=boxes)

    assert with_boxes is not None
    assert with_boxes["pan_deg"] == pytest.approx(without["pan_deg"], abs=0.5)
    assert with_boxes["tilt_deg"] == pytest.approx(without["tilt_deg"], abs=0.5)


@requires_models
def test_estimate_rejects_malformed_input(calib):
    with pytest.raises(ValueError):
        calib.estimate(None)
    with pytest.raises(ValueError):
        calib.estimate(np.zeros((100, 100), dtype=np.uint8))


@requires_models
def test_reset_reinitialises_from_keypoints(calib, frames):
    _, frame = frames[0]
    before = calib.estimate(frame)
    calib.reset()
    after = calib.estimate(frame)
    assert after is not None
    # After a reset the tracker restarts from the prior camera, so the first
    # solve must attempt the keypoint-driven re-initialisation.
    assert after["reinit"] is True
    assert after["pan_deg"] == pytest.approx(before["pan_deg"], abs=1.0)


@requires_models
def test_bev_round_trip_places_players_on_the_pitch(calib, frames):
    """A player standing on the centre mark must land at the BEV centre."""
    from core.projection_utils import pixel_to_ground, project_3d_to_pixel

    result = calib.estimate(frames[0][1])
    pixel = project_3d_to_pixel(np.array([0.0, 0.0, 0.0]), result["K"], result["R"], result["t"])
    back = pixel_to_ground(pixel[0], pixel[1], result["K"], result["R"], result["t"])
    assert back is not None
    assert back[0] == pytest.approx(0.0, abs=1.0)
    assert back[1] == pytest.approx(0.0, abs=1.0)

    px, py = calib.world_to_bev_px(back[0], back[1])
    template = calib.create_bev_template()
    assert 0 <= px < template.shape[1]
    assert 0 <= py < template.shape[0]


@requires_models
def test_line_mask_is_labelled_with_pitch_line_ids(calib, frames):
    """The mask that reaches the solver must be a label map, not a binary image:
    PointExtractor groups points per line id and skips UNDEFINED_LINE."""
    result = calib.estimate(frames[0][1])
    mask = result["line_mask"]
    assert mask.dtype == np.uint8
    assert mask.ndim == 2

    values = set(np.unique(mask).tolist())
    assert 0 in values, "background must be zero"
    labelled = values - {0}
    assert labelled, "the segmentation produced no pitch lines"
    assert all(150 <= value <= 200 for value in labelled), labelled
    assert 150 not in labelled, "UNDEFINED_LINE must be mapped to background"
