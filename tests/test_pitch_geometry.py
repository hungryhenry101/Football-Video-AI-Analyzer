"""Pitch model, BEV rendering and the projection helpers.

These need the compiled extension but no model weights, so they run everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.broadtrack_calib import CENTER_CIRCLE_RADIUS, PITCH_LENGTH, PITCH_WIDTH, PitchGeometry, pitch_wireframe
from core.broadtrack_calib.calib import _project_polyline, _in_frame


@pytest.fixture(scope="module")
def geometry() -> PitchGeometry:
    return PitchGeometry()


def test_wireframe_covers_the_pitch(geometry):
    """Every marking must lie on the ground, inside the touch and goal lines."""
    assert geometry.wireframe, "the pitch model returned no polylines"
    for polyline in geometry.wireframe:
        assert polyline.ndim == 2 and polyline.shape[1] == 3, "polylines are (N, 3) world points"
        assert np.isfinite(polyline).all()
        assert np.allclose(polyline[:, 2], 0.0), "pitch markings are on z = 0"
        assert polyline[:, 0].min() >= -PITCH_LENGTH / 2 - 1e-6
        assert polyline[:, 0].max() <= PITCH_LENGTH / 2 + 1e-6
        assert polyline[:, 1].min() >= -PITCH_WIDTH / 2 - 1e-6
        assert polyline[:, 1].max() <= PITCH_WIDTH / 2 + 1e-6


def test_wireframe_includes_the_goal_lines(geometry):
    """Sanity-check the coordinate frame against the regulation pitch."""
    xs = np.concatenate([p[:, 0] for p in geometry.wireframe])
    ys = np.concatenate([p[:, 1] for p in geometry.wireframe])
    assert xs.min() == pytest.approx(-PITCH_LENGTH / 2)
    assert xs.max() == pytest.approx(PITCH_LENGTH / 2)
    assert ys.min() == pytest.approx(-PITCH_WIDTH / 2)
    assert ys.max() == pytest.approx(PITCH_WIDTH / 2)


def test_bev_mapping_puts_the_origin_at_the_centre(geometry):
    px, py = geometry.world_to_bev_px(0.0, 0.0)
    assert (px, py) == (int(PITCH_LENGTH / 2 * geometry.bev_scale), int(PITCH_WIDTH / 2 * geometry.bev_scale))


def test_bev_mapping_orientation(geometry):
    """+x runs right across the BEV, -y (the "top" touch line) runs towards row 0."""
    left, _ = geometry.world_to_bev_px(-PITCH_LENGTH / 2, 0.0)
    right, _ = geometry.world_to_bev_px(PITCH_LENGTH / 2, 0.0)
    _, top = geometry.world_to_bev_px(0.0, -PITCH_WIDTH / 2)
    _, bottom = geometry.world_to_bev_px(0.0, PITCH_WIDTH / 2)
    assert left == 0
    assert right == pytest.approx(int(PITCH_LENGTH * geometry.bev_scale), abs=1)
    assert top == 0
    assert bottom == pytest.approx(int(PITCH_WIDTH * geometry.bev_scale), abs=1)


def test_bev_template_shape_and_centre_circle(geometry):
    template = geometry.create_bev_template()
    assert template.shape == (int(PITCH_WIDTH * geometry.bev_scale), int(PITCH_LENGTH * geometry.bev_scale), 3)
    assert template.dtype == np.uint8

    # Halfway line: the column through the centre should be drawn on.
    cx, cy = geometry.world_to_bev_px(0.0, 0.0)
    assert (template[cy, cx] == 255).all(), "no centre mark drawn"
    radius = int(CENTER_CIRCLE_RADIUS * geometry.bev_scale)
    assert (template[cy, cx + radius] == 255).all(), "no centre circle drawn"


def test_pitch_wireframe_is_the_extension_wireframe(extension):
    wireframe = pitch_wireframe()
    assert len(wireframe) == len(extension.pitch_wireframe(1.0))
    assert np.allclose(wireframe[0], np.asarray(extension.pitch_wireframe(1.0)[0]))


def _nadir_camera(height_m: float, focal: float = 2000.0, width: int = 1920, height: int = 1080):
    """Projection matrix for a camera ``height_m`` above the pitch centre,
    looking straight down.

    World z is negative upwards, so the ground is z = 0 and a camera above it
    sits at negative z. Its optical axis points along world +z, which makes the
    rotation the identity: x_cam = X - t. Builds P = K [R | -R t], the same
    convention PnLCalib and core.projection_utils use.
    """
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3)
    t = np.array([0.0, 0.0, -height_m])
    It = np.eye(4)[:3]
    It[:, 3] = -t
    return K @ (R @ It)


def test_projection_puts_the_centre_mark_at_the_image_centre():
    P = _nadir_camera(20.0)
    centre = _project_polyline(np.array([[0.0, 0.0, 0.0]]), P)
    assert centre is not None
    assert centre[0] == (960, 540)


def test_projection_axes_follow_the_world_frame():
    """+x is to the right and +y (towards the bottom touch line) is downwards."""
    P = _nadir_camera(20.0)
    centre = _project_polyline(np.array([[0.0, 0.0, 0.0]]), P)[0]

    right = _project_polyline(np.array([[10.0, 0.0, 0.0]]), P)[0]
    below = _project_polyline(np.array([[0.0, 10.0, 0.0]]), P)[0]

    assert right[0] > centre[0] and right[1] == centre[1]
    assert below[1] > centre[1] and below[0] == centre[0]
    # 10 m at 20 m height with f = 2000 px is 1000 px from centre.
    assert right[0] - centre[0] == pytest.approx(1000, abs=1)


@pytest.mark.parametrize("point", [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
def test_projection_drops_points_at_or_behind_the_camera(point):
    """The camera plane is w = 0; anything at or beyond it must be dropped
    rather than mirrored through the principal point."""
    P = np.eye(3, 4, dtype=np.float64)
    assert _project_polyline(np.array([point]), P) is None


def test_projection_drops_empty_polylines():
    assert _project_polyline(np.empty((0, 3)), np.eye(3, 4)) is None


def test_draw_pitch_lines_paints_the_image(geometry):
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Default colour is BGR green, so test channel 1.
    geometry.draw_pitch_lines(img, _nadir_camera(30.0))
    painted = img[..., 1] > 0
    assert painted.any(), "nothing was drawn"

    # The overlay must be centred on the pitch, not smeared into a corner.
    ys, xs = np.nonzero(painted)
    assert xs.min() < 960 < xs.max()
    assert ys.min() < 540 < ys.max()


def test_in_frame_bounds():
    assert _in_frame((0, 0), 10, 10)
    assert _in_frame((9, 9), 10, 10)
    assert not _in_frame((-1, 5), 10, 10)
    assert not _in_frame((5, 10), 10, 10)
