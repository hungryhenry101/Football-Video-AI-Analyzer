"""Manual anchor -> BroadTrack camera conversion, and the C++ seeding entry."""

from __future__ import annotations

import numpy as np
import pytest

from core.anchor_bootstrap import (
    _camera_from_anchor,
    _decompose_homography,
    _reproduces,
    homography_from_camera,
    load_manual_bootstrap,
)
from core.pitchcal.anchors import Anchor, save_anchor_file
from core.pitchcal.geometry import PinholeCamera, look_at_rotation
from core.projection_utils import (
    homography_pixel_to_ground,
    pixel_to_ground,
    project_3d_to_pixel,
)


def _synthetic_anchor(width=1920, height=1080, focal=1500.0,
                      centre=(0.0, -80.0, 20.0), target=(0.0, 0.0, 0.0),
                      frame=0) -> Anchor:
    """An anchor holding the ground homography of a known z-up camera.

    Used only for format/round-trip tests.  This project's camera frame is
    z-down, so a z-up camera's H is mirrored here and cannot be ray-cast; the
    conversion tests below start from a real z-down camera instead.
    """
    camera = PinholeCamera(
        focal, width / 2.0, height / 2.0,
        look_at_rotation(np.asarray(centre, float), np.asarray(target, float)),
        np.asarray(centre, float),
    )
    return Anchor(frame=frame, width=width, height=height,
                  f=0.0, R=np.zeros((3, 3)), C=np.zeros(3),
                  H=camera.homography())


# Frame 0 of input_vids/test1.mp4, as BroadTrack solved it: a real z-down
# camera at 960x540 (the same fixture core.player_position uses).
_ZD_FOCAL, _ZD_PAN, _ZD_TILT, _ZD_ROLL = 1179.2, 2.148, 74.801, -0.280
_ZD_POSITION = np.array([-1.813, 73.705, -17.503])


def _zdown_camera(pan=_ZD_PAN, tilt=_ZD_TILT, roll=_ZD_ROLL,
                  position=None, focal=_ZD_FOCAL):
    p, t, r = np.radians([pan, tilt, roll])
    rot_tilt = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])
    rot_pan = np.array([[np.cos(p), -np.sin(p), 0.0], [np.sin(p), np.cos(p), 0.0], [0.0, 0.0, 1.0]])
    rot_roll = np.array([[np.cos(r), -np.sin(r), 0.0], [np.sin(r), np.cos(r), 0.0], [0.0, 0.0, 1.0]])
    K = np.array([[focal, 0.0, 480.0], [0.0, focal, 270.0], [0.0, 0.0, 1.0]])
    return K, (rot_pan @ rot_tilt @ rot_roll).T, (
        _ZD_POSITION if position is None else np.asarray(position, float))


def _zdown_anchor(width=1920, height=1080, frame=0, **camera_kwargs) -> Anchor:
    """A physical z-down camera's ground homography, stored as an anchor."""
    scale = width / 960.0
    K, R, t = _zdown_camera(**camera_kwargs)
    K = K.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return Anchor(frame=frame, width=width, height=height,
                  f=0.0, R=np.zeros((3, 3)), C=np.zeros(3),
                  H=homography_from_camera(K, R, t))


def test_conversion_keeps_the_ground_homography_and_puts_the_camera_above_ground():
    width, height = 1920, 1080
    anchor = _zdown_anchor(width, height)
    K_expected, _, t_expected = _zdown_camera()

    K, R, t, k1 = _camera_from_anchor(anchor, width // 2, height // 2)

    assert K[0, 0] == pytest.approx(K_expected[0, 0], rel=1e-5)
    assert K[0, 2] == pytest.approx(width / 4.0)
    assert K[1, 2] == pytest.approx(height / 4.0)
    assert t == pytest.approx(t_expected, abs=1e-3)
    assert t[2] < 0.0, "z points down, so the camera must be at a negative z"
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)

    # The converted camera must project the ground exactly like the anchor H.
    world = np.array([[-52.5, -34.0], [52.5, -34.0], [0.0, 0.0],
                      [52.5, 34.0], [-52.5, 34.0]])
    points_3d = np.column_stack([world, np.zeros(len(world))])
    projected = project_3d_to_pixel(points_3d, K, R, t)

    H_half = anchor.homography(width // 2, height // 2)
    homogeneous = np.column_stack([world, np.ones(len(world))])
    expected = (H_half @ homogeneous.T).T
    expected = expected[:, :2] / expected[:, 2:3]

    assert np.max(np.abs(projected - expected)) < 1e-6


def test_inconsistent_homography_gets_an_exact_offcentre_camera():
    """A hand-fitted H need not satisfy the centred-camera constraints.

    The decomposition must still reproduce it exactly, using a principal-point
    offset, instead of shifting the whole projection (which is what made the
    first anchored frame disagree with the annotator).
    """
    anchor = _synthetic_anchor()
    H = np.asarray(anchor.H, float)
    # A small projective perturbation preserves "is a homography" but makes
    # the squared-pixel/centred-principal constraints fail.
    H[0, 1] *= 1.05

    camera = _decompose_homography(H, 960, 540)
    assert camera is not None
    assert _reproduces(camera.homography(), H), "the camera must reproduce H"
    assert (abs(camera.cx - 480.0) > 1.0 or abs(camera.cy - 270.0) > 1.0), \
        "an inconsistent H needs an off-centre principal point"


def test_extension_preserves_an_offcentre_principal_point(extension):
    K = np.array([[835.6, 0.0, 250.6], [0.0, 835.6, 234.3], [0.0, 0.0, 1.0]])
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    t = np.array([-1.3, 38.9, -6.5])

    tracker = extension.Tracker({"pitch_length": 105.0, "pitch_width": 68.0})
    tracker.set_camera(K, R, t, 0.0, 960, 540, True)
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    mask = np.zeros((540, 960), dtype=np.uint8)
    result = tracker.estimate(frame, mask, np.zeros((0, 4)), None)

    assert result["K"][0, 2] == pytest.approx(250.6, abs=1e-6)
    assert result["K"][1, 2] == pytest.approx(234.3, abs=1e-6)
    assert np.allclose(result["t"], t)


@pytest.mark.parametrize("pan,tilt,roll,position", [
    (2.148, 74.801, -0.280, (-1.813, 73.705, -17.503)),
    (-5.0, 78.0, 1.0, (10.0, 70.0, -15.0)),
    (8.0, 70.0, -2.0, (-20.0, 65.0, -20.0)),
    (0.0, 80.0, 0.0, (0.0, 80.0, -12.0)),
])
def test_conversion_is_stable_across_camera_positions(pan, tilt, roll, position):
    anchor = _zdown_anchor(width=1920, height=1080,
                           pan=pan, tilt=tilt, roll=roll, position=position)
    K, R, t, _ = _camera_from_anchor(anchor, 960, 540)
    assert t[2] < 0.0
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(K).all() and np.isfinite(R).all() and np.isfinite(t).all()
    # The ray must reach the ground: z_est and a BroadTrack keyframe rely on it.
    pixel = project_3d_to_pixel(np.zeros(3), K, R, t)
    assert pixel_to_ground(float(pixel[0]), float(pixel[1]), K, R, t) is not None


def test_homography_projection_matches_the_camera_projection():
    """The LK path projects through H; the camera form must agree on the ground."""
    anchor = _zdown_anchor()
    K, R, t, _ = _camera_from_anchor(anchor, 960, 540)
    H = homography_from_camera(K, R, t)

    world = np.array([[-20.0, -10.0], [0.0, 0.0], [25.0, 15.0]])
    pixels = project_3d_to_pixel(
        np.column_stack([world, np.zeros(len(world))]), K, R, t)
    for (u, v), expected in zip(pixels, world):
        got = homography_pixel_to_ground(u, v, H)
        assert got == pytest.approx(expected, abs=1e-4)
        assert pixel_to_ground(u, v, K, R, t) == pytest.approx(got, abs=1e-4)


def test_anchor_file_roundtrip_and_frame_selection(tmp_path):
    path = tmp_path / "anchors.json"
    anchors = [_zdown_anchor(frame=100), _zdown_anchor(frame=300)]
    save_anchor_file(path, "match.mp4", (105.0, 68.0), anchors)

    bootstrap = load_manual_bootstrap(path, 960, 540)
    assert bootstrap.pitch == pytest.approx((105.0, 68.0))
    assert [a.frame for a in bootstrap.anchors] == [100, 300]

    assert bootstrap.at_or_before(99) is None
    assert bootstrap.at_or_before(100).frame == 100
    assert bootstrap.at_or_before(299).frame == 100
    assert bootstrap.at_or_before(300).frame == 300
    assert bootstrap.next_after(100).frame == 300
    assert bootstrap.next_after(300) is None
    assert bootstrap.exact(300).frame == 300
    assert bootstrap.exact(200) is None


def test_mismatched_aspect_ratio_is_rejected(tmp_path):
    path = tmp_path / "anchors.json"
    save_anchor_file(path, "match.mp4", (105.0, 68.0),
                     [_zdown_anchor(width=1280, height=960)])
    with pytest.raises(ValueError, match="aspect"):
        load_manual_bootstrap(path, 960, 540)


def test_a_camera_below_the_ground_is_rejected():
    # A homography whose decomposition would put the camera "above" the z-up
    # ground in the wrong sense: reflect x so the solver's C_z flips sign.
    anchor = _synthetic_anchor()
    anchor.H = anchor.H @ np.diag([-1.0, 1.0, 1.0])
    # The reflected homography still decomposes; the important assertion is
    # that the conversion either raises or returns a camera above the ground.
    try:
        _, _, t, _ = _camera_from_anchor(anchor, 960, 540)
    except ValueError:
        return
    assert t[2] < 0.0


def test_extension_accepts_a_seeded_camera(extension):
    """The C++ tracker must start from K/R/t and keep it on a blank frame."""
    K = np.array([[1200.0, 0.0, 480.0], [0.0, 1200.0, 270.0], [0.0, 0.0, 1.0]])
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    t = np.array([0.0, 60.0, -15.0])

    tracker = extension.Tracker({"pitch_length": 105.0, "pitch_width": 68.0})
    tracker.set_camera(K, R, t, 0.0, 960, 540, True)

    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    mask = np.zeros((540, 960), dtype=np.uint8)
    result = tracker.estimate(frame, mask, np.zeros((0, 4)), None)

    assert result["ok"]
    assert np.allclose(result["K"], K)
    # The per-frame solve may refine pan/tilt (and the line-IoU pan refinement
    # runs even with an empty mask), but the pinned position must survive.
    assert np.allclose(result["t"], t)
    assert np.linalg.det(result["R"]) == pytest.approx(1.0, abs=1e-6)


def test_pitch_wireframe_honours_custom_dimensions(extension):
    wireframe = extension.pitch_wireframe(1.0, 90.0, 64.0)
    points = np.concatenate(wireframe, axis=0)
    assert points[:, 0].min() == pytest.approx(-45.0)
    assert points[:, 0].max() == pytest.approx(45.0)
    assert points[:, 1].min() == pytest.approx(-32.0)
    assert points[:, 1].max() == pytest.approx(32.0)
