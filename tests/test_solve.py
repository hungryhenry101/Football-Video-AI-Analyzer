import numpy as np
import cv2
import pytest

from core.pitchcal.geometry import PinholeCamera, PitchTemplate, look_at_rotation
from core.pitchcal.solve import solve_homography_from_clicks


def test_ransac_click_homography_ignores_outlier():
    template = PitchTemplate(105, 68)
    names = ["center_mark", "center_circle_top", "center_circle_bottom",
             "halfway_top_touch", "left_penalty_top_inner", "right_penalty_top_inner",
             "left_corner_top", "right_corner_bottom"]
    world = np.array([template.landmarks[n] for n in names], float)
    C = np.array([0.0, -50.0, 20.0])
    R = look_at_rotation(C, np.array([0.0, 0.0, 0.0]))
    cam = PinholeCamera(1500.0, 960.0, 540.0, R, C)
    image = cam.ground_homography_points(world)
    image[3] += (55.0, -40.0)  # one bad click
    H, info = solve_homography_from_clicks(world, image)
    projected = cv2.perspectiveTransform(world[None].astype(np.float32), H)[0]
    err = np.linalg.norm(projected - image, axis=1)
    assert info["used_points"] >= 6
    assert np.median(err[np.asarray(info["inlier_mask"], bool)]) < 3.0
    # The bad click must be *named*, not just out-voted: this is what lets the
    # annotator paint it red and the human re-click it.
    assert 3 in info["outlier_indices"]
    assert info["outlier_detection"] == "ransac"
    outlier_err = dict(zip(info["outlier_indices"], info["outlier_errors_px"]))
    assert outlier_err[3] > 40.0


def test_four_click_solve_says_outliers_cannot_be_checked():
    names = ["center_mark", "center_circle_top", "center_circle_left",
             "left_penalty_top_inner"]
    world, image = _synthetic_clicks(names, noise_px=1.0)
    _, info = solve_homography_from_clicks(world, image)
    assert info["outlier_indices"] == []
    assert "fewer than 6" in info["outlier_detection"]


def _synthetic_clicks(names, noise_px=0.0, seed=0):
    template = PitchTemplate(105, 68)
    world = np.array([template.landmarks[n] for n in names], float)
    C = np.array([0.0, -50.0, 20.0])
    R = look_at_rotation(C, np.array([0.0, 0.0, 0.0]))
    cam = PinholeCamera(1500.0, 960.0, 540.0, R, C)
    image = cam.ground_homography_points(world)
    if noise_px:
        image = image + np.random.default_rng(seed).normal(0, noise_px, image.shape)
    return world, image


def test_three_collinear_clicks_are_rejected():
    """The old RECOMMENDED first four (three on the halfway line) must fail.

    ``cv2.findHomography`` happily returns an arbitrary member of a
    one-parameter family for this layout; the solver must refuse it instead.
    """
    names = ["center_mark", "center_circle_top", "center_circle_bottom",
             "center_circle_left"]
    world, image = _synthetic_clicks(names)
    with pytest.raises(ValueError, match="lie on one line"):
        solve_homography_from_clicks(world, image)


def test_four_collinear_plus_one_off_line_is_rejected():
    """Four halfway-line points plus one penalty corner still leave a DOF."""
    names = ["center_mark", "center_circle_top", "center_circle_bottom",
             "halfway_top_touch", "left_penalty_top_inner"]
    world, image = _synthetic_clicks(names)
    with pytest.raises(ValueError, match="lie on one line"):
        solve_homography_from_clicks(world, image)


def test_two_off_line_clicks_are_enough():
    names = ["center_mark", "center_circle_top", "center_circle_left",
             "left_penalty_top_inner", "right_penalty_top_inner"]
    world, image = _synthetic_clicks(names, noise_px=2.0)
    H, info = solve_homography_from_clicks(world, image)
    assert np.all(np.isfinite(H))
    assert abs(H[2, 2]) > 1e-9
    assert info["median_reproj_px"] < 5.0
