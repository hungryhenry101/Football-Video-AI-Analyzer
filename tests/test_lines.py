"""White-line annotation solving, independent of the GUI."""
from __future__ import annotations

import numpy as np

from core.pitchcal.anchors import Anchor, load_anchor_file, save_anchor_file
from core.pitchcal.geometry import PinholeCamera, PitchTemplate, look_at_rotation
from core.pitchcal.solve import (line_metrics_from_homography,
                            refine_homography_to_lines,
                            solve_homography_from_lines)


W, H = 1600, 900


def _project(Hm, xy):
    p = (Hm @ np.column_stack([xy, np.ones(len(xy))]).T).T
    ok = np.isfinite(p).all(axis=1) & (p[:, 2] > 1e-9)
    uv = np.full((len(xy), 2), np.nan)
    uv[ok] = p[ok, :2] / p[ok, 2:3]
    return uv


def _resample(pts, step):
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(round(arc[-1] / step)) + 1)
    q = np.linspace(0.0, arc[-1], n)
    return np.stack([np.interp(q, arc, pts[:, 0]),
                     np.interp(q, arc, pts[:, 1])], axis=1)


def _synthetic_scene(noise_px: float = 1.0, seed: int = 7):
    template = PitchTemplate(100.0, 64.0)
    C = np.array([0.0, 42.0, 7.0])
    R = look_at_rotation(C, np.array([0.0, 0.0, 0.0]))
    cam = PinholeCamera(820.0, W / 2.0, H / 2.0, R, C)
    H_true = cam.homography()
    rng = np.random.default_rng(seed)
    lines = []
    for name, world in template.polylines():
        uv = _project(H_true, _resample(world, 0.15))
        inside = (np.isfinite(uv).all(axis=1)
                  & (uv[:, 0] >= 0) & (uv[:, 0] < W)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < H))
        uv = uv[inside]
        if len(uv) < 20:
            continue
        # Keep a contiguous visible run, then downsample to ~4 px spacing.
        seg = np.linalg.norm(np.diff(uv, axis=0), axis=1)
        if seg.sum() < 60:
            continue
        uv = _resample(uv, 4.0)
        uv = uv + rng.normal(0.0, noise_px, uv.shape)
        lines.append({"name": name, "pixels": uv.tolist()})
    return template, H_true, cam, lines


def _perturbed(H_true, scale=1.01, shift=15.0):
    C = np.array([[scale, 0.0006, shift],
                  [-0.0006, scale, -shift],
                  [3e-5, -5e-5, 1.0]])
    H0 = C @ H_true
    return H0 / H0[2, 2]


def _template_reprojection_error(H_true, H_est, template):
    world = np.vstack([_resample(p, 0.4) for _, p in template.polylines()])
    a = _project(H_true, world)
    b = _project(H_est, world)
    inside = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
              & (a[:, 0] >= 0) & (a[:, 0] < W) & (a[:, 1] >= 0) & (a[:, 1] < H))
    return float(np.median(np.linalg.norm(a[inside] - b[inside], axis=1)))


def test_refine_homography_to_lines_recovers_perturbed_anchor():
    template, H_true, _, lines = _synthetic_scene()
    assert len(lines) >= 3
    H0 = _perturbed(H_true)
    before = line_metrics_from_homography(H0, lines, template, W, H)
    H_est, info = refine_homography_to_lines(H0, lines, template, W, H)
    assert info["accepted"], info
    assert info["line_median_px"] < 3.0, info
    assert info["line_median_px"] <= before["line_median_px"]
    assert _template_reprojection_error(H_true, H_est, template) < 4.0


def test_outlier_line_is_named():
    """A trace on the wrong line is reported, not silently averaged in."""
    template, H_true, _, lines = _synthetic_scene()
    corrupted = []
    for entry in lines:
        entry = dict(entry)
        if entry["name"] == "halfway_line":
            entry["pixels"] = (np.asarray(entry["pixels"], float)
                               + np.array([70.0, 0.0])).tolist()
        corrupted.append(entry)
    metrics = line_metrics_from_homography(H_true, corrupted, template, W, H)
    names = [o["name"] for o in metrics["outlier_lines"]]
    assert names == ["halfway_line"], metrics["outlier_lines"]
    assert metrics["outlier_lines"][0]["median_px"] > 40.0
    # A clean scene must not produce false positives.
    clean = line_metrics_from_homography(H_true, lines, template, W, H)
    assert clean["outlier_lines"] == [], clean["outlier_lines"]


def test_solve_homography_from_lines_with_previous_anchor():
    template, H_true, _, lines = _synthetic_scene()
    H0 = _perturbed(H_true, scale=1.02, shift=30.0)
    H_est, info = solve_homography_from_lines(lines, template, W, H,
                                              init_H=H0)
    assert info["method"] == "previous-anchor+lines", info
    assert info["accepted"], info
    assert info["line_median_px"] < 3.0, info
    assert _template_reprojection_error(H_true, H_est, template) < 6.0


def test_sparse_center_circle_clicks_are_arc_observations():
    template, H_true, _, lines = _synthetic_scene()
    circle = next(l for l in lines if l["name"] == "center_circle")
    pts = np.asarray(circle["pixels"], float)
    idx = np.linspace(0, len(pts) - 1, 7).astype(int)
    sparse = [{"name": "center_circle", "pixels": pts[idx].tolist()}]
    context = [l for l in lines if l["name"] != "center_circle"]
    H0 = _perturbed(H_true)
    before = line_metrics_from_homography(H0, sparse + context, template, W, H)
    H_est, info = refine_homography_to_lines(H0, sparse + context, template, W, H)
    # Sparse clicks are kept as points on the arc, not resampled into chords.
    assert info["per_line"]["center_circle"]["points"] == 7, info
    assert info["accepted"], info
    assert info["line_median_px"] < 4.0
    assert info["line_median_px"] <= before["line_median_px"]


def test_partial_line_traces_are_supported():
    """A line clipped by the frame edge must still constrain the homography."""
    template, H_true, _, lines = _synthetic_scene()
    partial = []
    for entry in lines:
        pts = np.asarray(entry["pixels"], float)
        lo, hi = int(0.30 * len(pts)), int(0.70 * len(pts))
        if hi - lo >= 2:
            partial.append({"name": entry["name"], "pixels": pts[lo:hi].tolist()})
    assert len(partial) >= 2
    H0 = _perturbed(H_true)
    H_est, info = solve_homography_from_lines(partial, template, W, H,
                                              init_H=H0)
    assert info["accepted"], info
    assert info["line_median_px"] < 4.0, info
    assert _template_reprojection_error(H_true, H_est, template) < 6.0


def test_bad_point_seed_is_rescued_by_the_traced_lines():
    template, H_true, _, lines = _synthetic_scene()
    # A projective seed that puts the projected template lines far outside the
    # frame; the old 80 px-clipped residual had no gradient to recover from.
    C = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [-0.0015, 0.0, 1.0]])
    H_bad = C @ H_true
    H_bad = H_bad / H_bad[2, 2]
    H_est, info = solve_homography_from_lines(lines, template, W, H,
                                              init_H=H_bad)
    assert info["accepted"], info
    assert info["line_median_px"] < 3.0, info
    if info.get("method") == "lines-only-multistart":
        assert info.get("point_seed_rejected"), info


def test_well_conditioned_point_seed_is_not_replaced_by_lines_only():
    """A good click seed carries the far field; a wrong trace must not discard it.

    This is the failure mode that made a manual U15 anchor worse than the
    automatic one: the traced lines disagreed with the clicks, and the solver
    silently returned a lines-only extrapolation that explained the traces but
    put the rest of the pitch in the wrong place.
    """
    template, H_true, _, lines = _synthetic_scene()
    names = ["center_mark", "center_circle_top", "center_circle_left",
             "left_penalty_top_inner", "right_penalty_top_inner"]
    world = np.array([template.landmarks[n] for n in names], float)
    image = _project(H_true, world)
    points = [{"world": w.tolist(), "pixel": i.tolist()}
              for w, i in zip(world, image)]
    bad = [dict(entry) for entry in lines]
    for entry in bad:
        if entry["name"] == "halfway_line":
            entry["pixels"] = (np.asarray(entry["pixels"], float)
                               + np.array([120.0, 0.0])).tolist()
            break
    else:
        raise AssertionError("synthetic scene has no halfway_line")
    H_est, info = solve_homography_from_lines(bad, template, W, H, points=points)
    assert info["method"] != "lines-only-multistart", info
    assert not info.get("point_seed_rejected"), info
    assert "clicked points" in info.get("warning", ""), info
    assert _template_reprojection_error(H_true, H_est, template) < 5.0, info
    # The clicks were used and the mislabelled trace is reported by name.
    assert info["used_points"] == len(points), info
    assert "halfway_line" in [o["name"] for o in info["outlier_lines"]], info


def test_line_anchor_roundtrip(tmp_path):
    Hm = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])
    lines = [{"name": "halfway_line", "pixels": [[1.0, 2.0], [3.0, 4.0]]}]
    anchor = Anchor(frame=12, width=W, height=H, f=0.0,
                    R=np.zeros((3, 3)), C=np.zeros(3), source="manual-lines",
                    score=0.9, lines=lines, line_hit=0.8, line_cov=1.0, H=Hm)
    path = tmp_path / "lines.json"
    save_anchor_file(path, "demo.mp4", (100.0, 64.0), [anchor], source="manual-lines")
    loaded = load_anchor_file(path).anchors[0]
    assert loaded.lines == lines
    assert abs(loaded.line_hit - 0.8) < 1e-9
    assert abs(loaded.line_cov - 1.0) < 1e-9
    assert np.allclose(loaded.H, Hm)


def test_unknown_line_names_are_rejected():
    template, _, _, _ = _synthetic_scene()
    lines = [{"name": "not_a_pitch_line", "pixels": [[0, 0], [10, 10]]}]
    try:
        solve_homography_from_lines(lines, template, W, H)
    except ValueError as exc:
        assert "no usable line annotations" in str(exc)
    else:
        raise AssertionError("unknown line name should not solve")
