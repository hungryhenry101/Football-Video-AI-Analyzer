import numpy as np

from core.pitchcal.geometry import (PinholeCamera, PitchTemplate, camera_from_homography,
                               look_at_rotation)


def test_pitch_template_landmarks():
    p = PitchTemplate(105, 68)
    assert len(p.landmarks) > 30
    assert p.landmarks["center_mark"] == (0.0, 0.0)
    assert abs(p.landmarks["left_penalty_top_inner"][0] + 36.0) < 1e-9
    assert abs(p.landmarks["left_penalty_top_inner"][1] + 20.16) < 1e-9


def test_homography_decomposition_roundtrip():
    C = np.array([0.0, -95.0, 24.0])
    R = look_at_rotation(C, np.array([0.0, 0.0, 0.0]))
    cam = PinholeCamera(1600.0, 960.0, 540.0, R, C)
    H = cam.homography()
    recovered = camera_from_homography(H, 1920, 1080, focal=1600.0)
    assert recovered is not None
    err = np.abs(recovered.homography() - H).max()
    assert err < 1e-6


def test_penalty_arcs_are_outside_the_penalty_area():
    """The visible arc cap must face the pitch centre, not the goal."""
    p = PitchTemplate(105, 68)
    arcs = {name: pts for name, pts in p.polylines() if name.endswith("_penalty_arc")}
    left = arcs["left_penalty_arc"]
    right = arcs["right_penalty_arc"]
    # Penalty-area front lines are at x = -36 and +36.
    assert left[:, 0].min() >= -36.01
    assert left[:, 0].max() < -30.0
    assert right[:, 0].max() <= 36.01
    assert right[:, 0].min() > 30.0
    # Endpoints are the named landmarks on the front line.
    lt = p.landmarks["left_penalty_arc_top"]
    assert np.min(np.linalg.norm(left - np.array(lt), axis=1)) < 0.2


def test_radial_projection_matches_opencv_convention():
    import cv2
    from core.pitchcal.geometry import rotation_to_rvec
    C = np.array([0.0, -45.0, 12.0])
    R = look_at_rotation(C, np.array([2.0, 1.0, 0.0]), roll=3.0)
    cam = PinholeCamera(1200.0, 960.0, 540.0, R, C)
    rng = np.random.default_rng(3)
    world = np.column_stack([rng.uniform(-50, 50, 40),
                             rng.uniform(-32, 32, 40),
                             np.zeros(40)])
    k1 = -0.18
    uv, _, valid = cam.project_distorted(world, k1)
    expected, _ = cv2.projectPoints(world, rotation_to_rvec(R),
                                    (-R @ C).reshape(3, 1), cam.K,
                                    np.array([k1, 0.0, 0.0, 0.0, 0.0]))
    expected = expected.reshape(-1, 2)
    assert np.allclose(uv[valid], expected[valid], atol=1e-6)
    # k1 = 0 is exactly the pinhole model.
    uv0, _, valid0 = cam.project_distorted(world, 0.0)
    uv_p, _, valid_p = cam.project(world)
    assert np.array_equal(valid0, valid_p)
    assert np.allclose(uv0[valid0], uv_p[valid_p], atol=1e-9)


def test_projection_does_not_mirror_points_behind_the_camera():
    """A line whose near end is behind the camera must not flip over the horizon.

    Anchors fitted from far-side clicks can extrapolate the near touchline
    behind the camera; dividing such points by a negative w used to mirror
    them above the horizon and draw the halfway line into the stands.
    """
    from core.pitchcal.geometry import project_polyline_segments
    H = np.array([[50.0, 0.0, 640.0],
                  [0.0, 40.0, 2251.0],
                  [0.0, -0.05, 1.0]])
    # One endpoint sits behind the camera (w = 1 - 0.05 * 34 < 0).
    assert not project_polyline_segments(H, np.array([[0.0, -34.0], [0.0, 34.0]]), 1280, 720)
    # A line entirely in front is drawn and clipped to the frame.
    segs = project_polyline_segments(H, np.array([[-52.5, -34.0], [52.5, -34.0]]), 1280, 720)
    assert len(segs) == 1
    for x, y in segs[0]:
        assert 0 <= x <= 1279 and 0 <= y <= 719
    assert abs(segs[0][0][1] - 330.0) < 1.0
