"""Camera from manual ground points and white-line annotations."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .geometry import (PinholeCamera, PitchTemplate, camera_from_parameters,
                       camera_parameters, look_at_rotation, rvec_to_rotation,
                       solve_camera_from_points)

# A traced line is reported as an *outlier* (never silently deleted) when its
# median distance to the projected template is far above the global fit, or
# when it barely touches the paint at all.  A line with fewer observations is
# too short to judge.
OUTLIER_LINE_MIN_POINTS = 8
OUTLIER_LINE_MIN_PX = 8.0
OUTLIER_LINE_REL = 3.0
# A clicked landmark is reported when its reprojection is this far from the
# robust median of the other clicks.  Used only when no RANSAC mask exists
# (a joint point+line solve); RANSAC's own inlier mask is authoritative.
OUTLIER_POINT_MIN_PX = 12.0
OUTLIER_POINT_REL = 3.0


def _project_points(cam: PinholeCamera, world_xy: np.ndarray) -> np.ndarray:
    return cam.ground_homography_points(world_xy)


def refine_from_points(cam: PinholeCamera, world_xy: np.ndarray,
                       image_xy: np.ndarray,
                       camera_height: Optional[float] = None,
                       max_nfev: int = 120) -> PinholeCamera:
    world = np.asarray(world_xy, float).reshape(-1, 2)
    image = np.asarray(image_xy, float).reshape(-1, 2)
    p0 = camera_parameters(cam)
    R0 = cam.R.copy()
    C0 = cam.C.copy()

    def unpack(p):
        f = cam.f * math.exp(float(p[0]))
        from .geometry import rvec_to_rotation
        R = rvec_to_rotation(p[1:4]) @ R0
        C = C0 + np.array([p[4], p[5], C0[2] * (math.exp(p[6]) - 1.0)])
        return PinholeCamera(f, cam.cx, cam.cy, R, C)

    def resid(p):
        c = unpack(p)
        uv = _project_points(c, world)
        r = (uv - image).ravel()
        prior = np.array([
            12.0 * p[0], 40.0 * p[1], 40.0 * p[2], 40.0 * p[3],
            0.5 * p[4], 0.5 * p[5], 4.0 * p[6],
        ])
        if camera_height is not None:
            prior = np.concatenate([prior, [6.0 * (c.height - camera_height) / max(camera_height, 1.0)]])
        return np.concatenate([r, prior])

    x0 = np.zeros(7)
    lb = np.array([-1.0, -0.7, -0.7, -0.7, -60, -60, -1.0])
    ub = np.array([1.0, 0.7, 0.7, 0.7, 60, 60, 1.0])
    result = least_squares(resid, x0, bounds=(lb, ub), loss="soft_l1",
                           f_scale=3.0, max_nfev=max_nfev,
                           x_scale=[0.2, 0.08, 0.08, 0.08, 6, 6, 0.2])
    return unpack(result.x)


def solve_from_clicks(world_xy: Sequence[Sequence[float]],
                      image_xy: Sequence[Sequence[float]],
                      width: int, height: int,
                      f_prior: Optional[float] = None,
                      camera_height: Optional[float] = None) -> tuple[PinholeCamera, dict]:
    world = np.asarray(world_xy, float).reshape(-1, 2)
    image = np.asarray(image_xy, float).reshape(-1, 2)
    if len(world) < 4:
        raise ValueError("need at least 4 point correspondences")
    layout = point_layout_problem(world, image)
    if layout is not None:
        raise ValueError(layout)
    import cv2
    # Human clicks are mostly good, but one or two often miss a line centre
    # (the circle left/right points in the existing U15 profile are a good
    # example).  RANSAC on the point homography finds the coherent subset
    # before the camera is decomposed; without it a single outlier can drag
    # the focal length and height into a different basin.
    inlier_mask = np.ones(len(world), bool)
    if len(world) >= 6:
        H_r, mask = cv2.findHomography(world, image, cv2.RANSAC, 12.0,
                                       maxIters=5000, confidence=0.999)
        if mask is not None:
            m = mask.ravel().astype(bool)
            if int(m.sum()) >= 5:
                inlier_mask = m
    w_in = world[inlier_mask]
    i_in = image[inlier_mask]
    cam = solve_camera_from_points(w_in, i_in, width, height, f_prior=f_prior,
                                   camera_height=camera_height)
    if cam is None:
        raise RuntimeError("point homography could not be decomposed")
    cam = refine_from_points(cam, w_in, i_in, camera_height=camera_height)
    uv = _project_points(cam, world)
    err = np.linalg.norm(uv - image, axis=1)
    outliers = ~inlier_mask
    return cam, {
        "median_reproj_px": float(np.median(err[inlier_mask])),
        "max_reproj_px": float(np.max(err[inlier_mask])),
        "errors_px": err.tolist(),
        "inlier_mask": inlier_mask.astype(int).tolist(),
        "used_points": int(inlier_mask.sum()),
        "outlier_detection": ("ransac" if len(world) >= 6
                              else "none (fewer than 6 clicks)"),
        "outlier_indices": [int(i) for i in np.flatnonzero(outliers)],
        "outlier_errors_px": [float(err[i]) for i in np.flatnonzero(outliers)],
    }


def refine_with_paint(cam: PinholeCamera, frame: np.ndarray, template,
                      world_xy: Optional[np.ndarray] = None,
                      image_xy: Optional[np.ndarray] = None,
                      point_weight: float = 2.0,
                      max_nfev: int = 100) -> tuple[PinholeCamera, dict]:
    """Small line-based correction of a point-solved camera.

    LSD + the paint-support filter gives a sparse set of trustworthy line
    samples.  Projecting the template and pulling it toward those samples is
    enough to correct the focal/height bias of a 4-8 click solve without the
    local minima that a dense binary line mask can introduce.
    """
    import cv2
    from scipy.spatial import cKDTree
    from .geometry import rvec_to_rotation
    from .linemask import paint_like_segments

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return cam, {"accepted": False, "reason": "no LSD segments"}
    segs = paint_like_segments(frame, np.asarray(lines, np.float32).reshape(-1, 4))
    if not len(segs):
        return cam, {"accepted": False, "reason": "no paint segments"}
    inv = np.linalg.inv(cam.homography())
    kept = []
    samples = []
    for x1, y1, x2, y2 in segs:
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / 2))
        t = np.linspace(0, 1, n)
        pts = np.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], axis=1)
        mid = pts[len(pts) // 2]
        p = inv @ np.array([mid[0], mid[1], 1.0])
        if abs(p[2]) < 1e-9:
            continue
        wx, wy = p[0] / p[2], p[1] / p[2]
        if abs(wx) <= template.hx + 4 and abs(wy) <= template.hy + 4:
            kept.append((x1, y1, x2, y2))
            samples.append(pts)
    if not kept:
        return cam, {"accepted": False, "reason": "no segments in pitch"}
    tree = cKDTree(np.vstack(samples).astype(np.float64))
    model = template.sample_points(step=0.7)
    world_pts = np.vstack([np.column_stack([p, np.zeros(len(p))]) for _, p in model])
    W, H = cam.cx * 2, cam.cy * 2

    def line_metrics(c: PinholeCamera):
        uv, depth, valid = c.project(world_pts)
        inside = (valid & np.isfinite(uv).all(axis=1)
                  & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H))
        if inside.sum() < 20:
            return 0.0, 99.0, 0
        d, _ = tree.query(uv[inside], k=1)
        d = np.minimum(d, 60.0)
        return float(np.mean(d < 5.0)), float(np.median(d)), int(inside.sum())

    hit0, med0, nvis0 = line_metrics(cam)
    R0, C0 = cam.R.copy(), cam.C.copy()

    def unpack(p):
        f = cam.f * math.exp(float(p[0]))
        R = rvec_to_rotation(p[1:4]) @ R0
        C = C0 + np.array([p[4], p[5], C0[2] * (math.exp(p[6]) - 1.0)])
        return PinholeCamera(f, cam.cx, cam.cy, R, C)

    pw = np.asarray(world_xy, float).reshape(-1, 2) if world_xy is not None else None
    pi = np.asarray(image_xy, float).reshape(-1, 2) if image_xy is not None else None

    def resid(p):
        c = unpack(p)
        uv, depth, valid = c.project(world_pts)
        inside = (valid & np.isfinite(uv).all(axis=1)
                  & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H))
        d = np.full(len(uv), 30.0)
        if inside.any():
            dd, _ = tree.query(uv[inside], k=1)
            d[inside] = np.minimum(dd, 30.0)
        pieces = [d]
        if pw is not None and pi is not None and len(pw):
            pieces.append(((c.ground_homography_points(pw) - pi) * point_weight).ravel())
        pieces.append(np.array([18.0 * p[0], 80.0 * p[1], 80.0 * p[2],
                                80.0 * p[3], 1.5 * p[4], 1.5 * p[5], 8.0 * p[6]]))
        return np.concatenate(pieces)

    lb = np.array([-0.35, -0.10, -0.10, -0.10, -8, -8, -0.35])
    ub = np.array([0.35, 0.10, 0.10, 0.10, 8, 8, 0.35])
    res = least_squares(resid, np.zeros(7), bounds=(lb, ub), loss="soft_l1",
                        f_scale=3.0, max_nfev=max_nfev,
                        x_scale=[0.15, 0.04, 0.04, 0.04, 2, 2, 0.15])
    cand = unpack(res.x)
    hit1, med1, nvis1 = line_metrics(cand)
    point_before = point_after = None
    if pw is not None and pi is not None and len(pw):
        point_before = float(np.median(np.linalg.norm(cam.ground_homography_points(pw) - pi, axis=1)))
        point_after = float(np.median(np.linalg.norm(cand.ground_homography_points(pw) - pi, axis=1)))
    accepted = (hit1 >= hit0 + 0.015 or med1 <= med0 * 0.85)
    if point_after is not None and point_before is not None:
        accepted = accepted and point_after <= max(12.0, point_before * 1.8)
    info = {
        "accepted": bool(accepted),
        "hit_before": hit0, "hit_after": hit1,
        "median_before_px": med0, "median_after_px": med1,
        "visible_before": nvis0, "visible_after": nvis1,
        "point_before_px": point_before, "point_after_px": point_after,
        "segments": len(kept),
    }
    return (cand if accepted else cam), info


def _max_collinear_count(points: np.ndarray, rel_tol: float = 1e-3) -> int:
    """Largest number of points lying on one line (within a relative tolerance).

    Used to catch degenerate click layouts before a homography is fitted.  The
    tolerance is relative to the point-set extent, so it works both in metres
    and in native image pixels.
    """
    p = np.asarray(points, float).reshape(-1, 2)
    n = len(p)
    if n < 3:
        return n
    span = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])), 1e-9)
    tol = float(rel_tol) * span
    best = 2
    for i in range(n):
        for j in range(i + 1, n):
            d = p[j] - p[i]
            nd = float(np.hypot(d[0], d[1]))
            if nd < 1e-9:
                continue
            normal = np.array([-d[1], d[0]], float) / nd
            dist = np.abs((p - p[i]) @ normal)
            best = max(best, int(np.count_nonzero(dist <= tol)))
    return best


def point_layout_problem(world_xy, image_xy) -> Optional[str]:
    """Return a human-readable reason when clicks cannot determine a homography.

    A projectivity is fixed by 4 points in general position.  If all but one of
    the clicks lie on a single line, a one-parameter family of homographies
    explains them: the DLT is rank-deficient and ``cv2.findHomography`` returns
    an arbitrary (often near-singular) member of that family.  This is not a
    rare edge case -- clicking the centre mark, the circle top/bottom and the
    halfway-line touch points is the natural way to start, and it produces
    exactly such a set.
    """
    world = np.asarray(world_xy, float).reshape(-1, 2)
    image = np.asarray(image_xy, float).reshape(-1, 2)
    if len(world) != len(image):
        return "the world and image point counts differ"
    n = len(world)
    if n < 4:
        return f"need at least 4 clicks (have {n})"
    if _max_collinear_count(world) >= n - 1:
        return (f"{n - 1} of {n} landmarks lie on one line, so the homography "
                "is not determined.  Click at least two landmarks off that "
                "line (for example the centre circle plus a penalty-area "
                "corner on each side).")
    if _max_collinear_count(image) >= n - 1:
        return ("the clicked pixels are nearly collinear; check that every "
                "click is on the landmark named in the panel.")
    return None


def solve_homography_from_clicks(world_xy: Sequence[Sequence[float]],
                                 image_xy: Sequence[Sequence[float]],
                                 ransac_px: float = 12.0) -> tuple[np.ndarray, dict]:
    """Robust world->image ground homography from manual clicks.

    This is the dependency-free manual path.  It intentionally returns the
    homography, not a decomposed camera: the homography is exactly what the
    BEV and the frame-to-frame tracker need, and it has no mirrored-pose
    ambiguity.
    """
    import cv2
    world = np.asarray(world_xy, float).reshape(-1, 2)
    image = np.asarray(image_xy, float).reshape(-1, 2)
    if len(world) < 4:
        raise ValueError("need at least 4 point correspondences")
    layout = point_layout_problem(world, image)
    if layout is not None:
        raise ValueError(layout)
    H, mask = cv2.findHomography(world, image, cv2.RANSAC, ransac_px,
                                 maxIters=5000, confidence=0.999)
    if H is None or not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-9:
        raise RuntimeError("homography solve failed (degenerate solution)")
    inliers = np.ones(len(world), bool) if mask is None else mask.ravel().astype(bool)
    if int(inliers.sum()) >= 4:
        H_refit, _ = cv2.findHomography(world[inliers], image[inliers], 0)
        if H_refit is not None and np.all(np.isfinite(H_refit)):
            H = H_refit
    projected = cv2.perspectiveTransform(world[None].astype(np.float32), H)[0]
    err = np.linalg.norm(projected - image, axis=1)
    if err[inliers].size == 0 or (float(np.median(err[inliers])) > 8.0
                                  or float(np.max(err[inliers])) > 40.0):
        raise RuntimeError(
            "the clicks do not agree on one homography (inlier residual "
            f"{float(np.median(err[inliers])) if err[inliers].size else 0:.1f} px); "
            "check the landmark labels and the pitch size")
    outliers = ~inliers
    return H, {
        "inlier_mask": inliers.astype(int).tolist(),
        "used_points": int(inliers.sum()),
        "median_reproj_px": float(np.median(err[inliers])),
        "max_reproj_px": float(np.max(err[inliers])),
        "errors_px": err.tolist(),
        # RANSAC needs redundancy to identify a bad click.  With 4-5 clicks a
        # homography can pass exactly through every one of them, so an
        # "outlier free" report is not evidence that every click is right.
        "outlier_detection": ("ransac" if len(world) >= 6
                              else "none (fewer than 6 clicks)"),
        "outlier_indices": [int(i) for i in np.flatnonzero(outliers)],
        "outlier_errors_px": [float(err[i]) for i in np.flatnonzero(outliers)],
    }


# ---------------------------------------------------------------------------
# Manual white-line annotations
# ---------------------------------------------------------------------------
#
# A line annotation is a named template line plus a polyline the human traced
# on the white paint.  Unlike a click, a line carries information along its
# whole length; missing intersections no longer prevent a solve.  The price is
# that a line only constrains the homography up to sliding along itself, so
# lines are used either to refine an existing anchor or, when enough of them
# are visible, to seed a camera search.

def _resample_2d(points, step: float) -> np.ndarray:
    pts = np.asarray(points, float).reshape(-1, 2)
    if len(pts) < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < 1e-9:
        return pts[:1].copy()
    n = max(2, int(round(total / max(float(step), 1e-3))) + 1)
    q = np.linspace(0.0, total, n)
    return np.stack([np.interp(q, arc, pts[:, 0]),
                     np.interp(q, arc, pts[:, 1])], axis=1)


def _normalise_line_annotations(lines, template: PitchTemplate) -> Dict[str, List[np.ndarray]]:
    """Group drawn polylines by template line name.

    ``lines`` is a sequence of ``{"name": str, "pixels": [[x, y], ...]}``.
    Several segments may share a name -- an occluded touchline traced in two
    pieces, or a centre circle traced as two arcs -- and are merged here.
    """
    valid = {name for name, _ in template.polylines()}
    polys = dict(template.polylines())
    grouped: Dict[str, List[np.ndarray]] = {}
    for entry in lines or []:
        name = str(entry.get("name", ""))
        if name not in valid:
            continue
        pts = np.asarray(entry.get("pixels", []), float)
        if pts.ndim != 2 or len(pts) < 2:
            continue
        pts = pts[:, :2]
        if len(polys[name]) == 2:
            # Straight template line: resample so the per-line weight reflects
            # the drawn length rather than the number of clicks.
            pts = _resample_2d(pts, 3.0)
        else:
            # Curved template line: keep the human's vertices as observations.
            # Connecting sparse clicks with chords and resampling them would
            # place samples *inside* the arc (chord sagitta) and bias the fit.
            pts = pts.copy()
        if len(pts) >= 2:
            grouped.setdefault(name, []).append(pts)
    return grouped


def _line_distances_from_H(H: np.ndarray, drawn: Dict[str, List[np.ndarray]],
                           template: PitchTemplate, width: int, height: int,
                           step: float = 0.08, max_px: float = 80.0) -> Dict[str, np.ndarray]:
    """Distance from every drawn point to its projected template line.

    Straight template lines are compared to their *supporting* image line: a
    human tracing only the visible middle of a touchline should not be
    penalised for not reaching its painted ends.  Curves (the centre circle,
    penalty arcs, corner arcs) use nearest points on a densely sampled
    projected polyline.
    """
    H = np.asarray(H, float)
    polys = dict(template.polylines())
    out: Dict[str, np.ndarray] = {}
    for name, segments in drawn.items():
        world = np.asarray(polys[name], float)
        ds: List[np.ndarray] = []
        if len(world) == 2:
            ends = np.column_stack([world, np.ones(2)])
            q = H @ ends.T
            if np.all(q[2] > 1e-9):
                uv = q[:2] / q[2:3]
                line = np.cross(np.array([uv[0, 0], uv[1, 0], 1.0]),
                                np.array([uv[0, 1], uv[1, 1], 1.0]))
                norm = float(math.hypot(line[0], line[1]))
                if norm > 1e-9:
                    line = line / norm
                    for pts in segments:
                        hom = np.column_stack([pts, np.ones(len(pts))])
                        ds.append(np.minimum(np.abs(hom @ line), max_px))
        if not ds:
            samples = _resample_2d(world, step)
            q = H @ np.column_stack([samples, np.ones(len(samples))]).T
            valid = np.isfinite(q).all(axis=0) & (q[2] > 1e-9)
            if int(valid.sum()) >= 2:
                uv = (q[:2, valid] / q[2, valid]).T
                inside = ((uv[:, 0] >= -max_px) & (uv[:, 0] <= width + max_px)
                          & (uv[:, 1] >= -max_px) & (uv[:, 1] <= height + max_px))
                uv = uv[inside]
                if len(uv) >= 2:
                    tree = cKDTree(uv)
                    for pts in segments:
                        d, _ = tree.query(pts)
                        ds.append(np.minimum(d, max_px))
        n = sum(len(s) for s in segments)
        out[name] = np.concatenate(ds) if ds else np.full(n, max_px, float)
    return out


def _line_metrics_from_distances(distances: Dict[str, np.ndarray],
                                 max_px: float = 80.0) -> dict:
    per_line = {}
    all_d: List[np.ndarray] = []
    for name, d in distances.items():
        if len(d) == 0:
            per_line[name] = {"median_px": float(max_px), "p90_px": float(max_px),
                              "points": 0, "hit": 0.0}
            continue
        per_line[name] = {
            "median_px": float(np.median(d)),
            "p90_px": float(np.quantile(d, 0.9)),
            "points": int(len(d)),
            "hit": float(np.mean(d < 5.0)),
        }
        all_d.append(d)
    if not all_d:
        return {"line_median_px": float(max_px), "line_hit": 0.0,
                "line_coverage": 0.0, "per_line": per_line, "line_points": 0,
                "worst_line": None, "worst_line_px": float(max_px),
                "outlier_lines": [], "outlier_threshold_px": float(max_px)}
    joined = np.concatenate(all_d)
    medians = [v["median_px"] for v in per_line.values() if v["points"]]
    coverage = float(np.mean([m < 5.0 for m in medians])) if medians else 0.0
    worst = max(per_line, key=lambda k: per_line[k]["median_px"])
    return {
        "line_median_px": float(np.median(joined)),
        "line_hit": float(np.mean(joined < 5.0)),
        "line_coverage": coverage,
        "per_line": per_line,
        "line_points": int(len(joined)),
        "worst_line": worst,
        "worst_line_px": float(per_line[worst]["median_px"]),
        "outlier_lines": _outlier_lines_from_metrics(per_line, float(np.median(joined))),
        "outlier_threshold_px": _outlier_line_threshold(float(np.median(joined))),
    }


def _outlier_line_threshold(global_median_px: float) -> float:
    return max(OUTLIER_LINE_MIN_PX, OUTLIER_LINE_REL * max(float(global_median_px), 0.0))


def _outlier_lines_from_metrics(per_line: Dict[str, dict],
                                global_median_px: float) -> List[dict]:
    """Name the traced lines that the solved homography does not explain.

    This is a *report*, not an automatic rejection: a mislabelled line (top vs
    bottom touchline, or a line from the other half) is exactly the kind of
    human error the annotator should point at so it can be undone and retraced.
    """
    threshold = _outlier_line_threshold(global_median_px)
    out: List[dict] = []
    for name, m in per_line.items():
        n = int(m.get("points", 0))
        if n < OUTLIER_LINE_MIN_POINTS:
            continue
        median = float(m.get("median_px", OUTLIER_LINE_MIN_PX * 10))
        hit = float(m.get("hit", 0.0))
        if median > threshold or (hit < 0.2 and median > 5.0):
            out.append({
                "name": name,
                "median_px": median,
                "p90_px": float(m.get("p90_px", median)),
                "hit": hit,
                "points": n,
            })
    out.sort(key=lambda item: item["median_px"], reverse=True)
    return out


def line_metrics_from_homography(H: np.ndarray, lines, template: PitchTemplate,
                                 width: int, height: int) -> dict:
    drawn = _normalise_line_annotations(lines, template)
    return _line_metrics_from_distances(_line_distances_from_H(H, drawn, template, width, height))


def _line_quality_score(m: dict) -> float:
    return (float(m.get("line_median_px", 80.0))
            + 0.25 * min(float(m.get("worst_line_px", 80.0)), 80.0)
            + 20.0 * (1.0 - float(m.get("line_hit", 0.0))))


_PITCH_SIZE_LINES = {
    "touchline_top", "touchline_bottom",
    "goal_line_left", "goal_line_right",
    "left_penalty_front", "right_penalty_front",
    "corner_tl", "corner_tr", "corner_br", "corner_bl",
}


def _pitch_size_hint(H: np.ndarray, drawn: Dict[str, List[np.ndarray]],
                     template: PitchTemplate, image_w: int, image_h: int,
                     worst_name: Optional[str], worst_px: float) -> Optional[str]:
    """If one line is far off, check whether a different outer pitch size fits.

    Touchlines move with the pitch *width*; goal/penalty fronts move with the
    *length*.  The regulation markings inside (centre circle, penalty areas)
    keep their absolute size, so this is a cheap way to catch a wrong
    ``--pitch`` before blaming the trace.
    """
    if worst_name not in _PITCH_SIZE_LINES or worst_px <= 12.0:
        return None
    lengths = sorted({round(template.length + d) for d in (-15, -10, -5, 0, 5, 10)})
    widths = sorted({round(template.width + d) for d in (-8, -6, -4, -2, 0, 2, 4, 6)})
    best = None
    for length in lengths:
        for pitch_width in widths:
            if length <= 40 or pitch_width <= 30:
                continue
            candidate = PitchTemplate(float(length), float(pitch_width))
            try:
                m = _line_metrics_from_distances(
                    _line_distances_from_H(H, drawn, candidate, image_w, image_h))
            except Exception:
                continue
            line_px = float(m["per_line"].get(worst_name, {}).get("median_px", 80.0))
            score = _line_quality_score(m)
            if best is None or score < best[0]:
                best = (score, float(length), float(pitch_width), line_px)
    if best is None:
        return None
    _, length, pitch_width, line_px = best
    if (abs(length - template.length) < 0.5
            and abs(pitch_width - template.width) < 0.5):
        return None
    if line_px <= max(6.0, worst_px * 0.5) and line_px < worst_px - 5.0:
        return (f"try --pitch {length:.0f}x{pitch_width:.0f}: {worst_name} "
                f"drops from {worst_px:.0f} px to {line_px:.0f} px")
    return None


def _line_bases_for_correction(H: np.ndarray, drawn: Dict[str, List[np.ndarray]],
                               template: PitchTemplate,
                               step: float = 0.08) -> Dict[str, dict]:
    """Pre-project template samples so a projective correction is cheap."""
    H = np.asarray(H, float)
    polys = dict(template.polylines())
    bases: Dict[str, dict] = {}
    for name in drawn:
        world = np.asarray(polys[name], float)
        samples = _resample_2d(world, step)
        entry = {
            "base_samples": H @ np.column_stack([samples, np.ones(len(samples))]).T,
            "straight": False,
        }
        if len(world) == 2:
            ends = np.column_stack([world, np.ones(2)])
            entry["base_ends"] = H @ ends.T
            entry["straight"] = True
        bases[name] = entry
    return bases


def _line_distances_from_correction(C: np.ndarray, bases: Dict[str, dict],
                                    drawn: Dict[str, List[np.ndarray]],
                                    width: int, height: int,
                                    max_px: float = 80.0) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, entry in bases.items():
        segments = drawn[name]
        ds: List[np.ndarray] = []
        if entry["straight"]:
            q = C @ entry["base_ends"]
            if np.all(q[2] > 1e-9):
                uv = q[:2] / q[2:3]
                line = np.cross(np.array([uv[0, 0], uv[1, 0], 1.0]),
                                np.array([uv[0, 1], uv[1, 1], 1.0]))
                norm = float(math.hypot(line[0], line[1]))
                if norm > 1e-9:
                    line = line / norm
                    for pts in segments:
                        hom = np.column_stack([pts, np.ones(len(pts))])
                        ds.append(np.minimum(np.abs(hom @ line), max_px))
        if not ds:
            q = C @ entry["base_samples"]
            valid = np.isfinite(q).all(axis=0) & (q[2] > 1e-9)
            if int(valid.sum()) >= 2:
                uv = (q[:2, valid] / q[2, valid]).T
                inside = ((uv[:, 0] >= -max_px) & (uv[:, 0] <= width + max_px)
                          & (uv[:, 1] >= -max_px) & (uv[:, 1] <= height + max_px))
                uv = uv[inside]
                if len(uv) >= 2:
                    tree = cKDTree(uv)
                    for pts in segments:
                        d, _ = tree.query(pts)
                        ds.append(np.minimum(d, max_px))
        n = sum(len(s) for s in segments)
        out[name] = np.concatenate(ds) if ds else np.full(n, max_px, float)
    return out


def _line_weight_map(distances: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Balance lines of very different traced length.

    A full touchline supplies hundreds of samples while a short penalty-area
    line supplies twenty.  Without balancing, the long line wins every argument
    and the short far-side line -- often the only left/right evidence -- is
    ignored as an outlier.  The weights make each line contribute roughly the
    same total squared residual.
    """
    sizes = [len(d) for d in distances.values() if len(d)]
    ref = float(np.median(sizes)) if sizes else 1.0
    return {name: math.sqrt(ref / max(1, len(d))) for name, d in distances.items()}


def _point_annotations(points) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not points:
        return None, None
    if isinstance(points[0], dict):
        world = np.asarray([p["world"] for p in points], float).reshape(-1, 2)
        image = np.asarray([p["pixel"] for p in points], float).reshape(-1, 2)
    else:
        world = np.asarray(points, float).reshape(-1, 2)
        image = None
    if image is None or len(world) == 0 or len(world) != len(image):
        return None, None
    return world, image


def _project_ground(H: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    hom = np.column_stack([world_xy, np.ones(len(world_xy))])
    q = (np.asarray(H, float) @ hom.T).T
    return q[:, :2] / q[:, 2:3]


def refine_homography_to_lines(H0: np.ndarray, lines, template: PitchTemplate,
                               width: int, height: int, points=None,
                               point_weight: float = 0.5,
                               max_nfev: int = 120) -> Tuple[np.ndarray, dict]:
    """Pull an existing anchor onto manually traced white lines.

    The fit has two stages:

    1. an **escape** stage with a large distance cap, a weak robust loss and
       wide bounds.  If the starting H is a bad point seed, the projected
       template lines can sit hundreds of pixels outside the frame; clipping
       them at 80 px removes the gradient and the optimiser can never find
       them.  The escape stage keeps a real gradient.
    2. the old **fine** robust correction around the escaped H, which snaps the
       lines to the paint without letting a single mislabelled line dominate.

    The previous anchor/point seed is still valuable as an initialiser, but the
    points are down-weighted (0.5) because they are usually rough guidance for
    the line fit, not measurements.
    """
    drawn = _normalise_line_annotations(lines, template)
    if not drawn:
        raise ValueError("no usable line annotations")
    H0 = np.asarray(H0, float).copy()
    if abs(H0[2, 2]) > 1e-12:
        H0 = H0 / H0[2, 2]
    base_dist = _line_distances_from_H(H0, drawn, template, width, height)
    metrics0 = _line_metrics_from_distances(base_dist)
    world_pts, image_pts = _point_annotations(points)

    def _optimise(H_base: np.ndarray, coarse: bool):
        bases = _line_bases_for_correction(H_base, drawn, template)
        weights = _line_weight_map(
            _line_distances_from_H(H_base, drawn, template, width, height))

        def unpack(p: np.ndarray) -> np.ndarray:
            return np.array([[1.0 + p[0], p[1], p[2]],
                             [p[3], 1.0 + p[4], p[5]],
                             [p[6], p[7], 1.0]], float)

        def resid(p: np.ndarray) -> np.ndarray:
            C = unpack(p)
            cap = 5000.0 if coarse else 2000.0
            d = _line_distances_from_correction(
                C, bases, drawn, width, height, max_px=cap)
            weighted = [d[name] * weights.get(name, 1.0) for name in d]
            if coarse:
                reg = np.array([1.0 * p[0], 1.0 * p[1], 0.01 * p[2],
                                1.0 * p[3], 1.0 * p[4], 0.01 * p[5],
                                2.0 * p[6], 2.0 * p[7]])
            else:
                reg = np.array([2.0 * p[0], 2.0 * p[1], 0.04 * p[2],
                                2.0 * p[3], 2.0 * p[4], 0.04 * p[5],
                                25.0 * p[6], 25.0 * p[7]])
            pieces = [np.concatenate(weighted) if weighted else np.zeros(0), reg]
            if world_pts is not None and len(world_pts):
                # Keep the rough clicks in the escape stage too.  Without them
                # an unobservable far field (central view) is free to slide to
                # a different member of the projective family, which is how a
                # degenerate seed used to be "rescued" into a wrong anchor.
                Hm = C @ H_base
                if abs(Hm[2, 2]) > 1e-12:
                    Hm = Hm / Hm[2, 2]
                uv = _project_ground(Hm, world_pts)
                w = point_weight * (0.5 if coarse else 1.0)
                pieces.append(((uv - image_pts) * w).ravel())
            return np.concatenate(pieces)

        if coarse:
            lb = np.array([-0.60, -0.60, -2000.0, -0.60, -0.60, -2000.0, -0.05, -0.05])
            ub = np.array([0.60, 0.60, 2000.0, 0.60, 0.60, 2000.0, 0.05, 0.05])
            res = least_squares(resid, np.zeros(8), bounds=(lb, ub), loss="soft_l1",
                                f_scale=20.0, max_nfev=max(60, max_nfev // 2),
                                x_scale=[0.05, 0.05, 100.0, 0.05, 0.05, 100.0, 0.002, 0.002])
        else:
            lb = np.array([-0.20, -0.20, -300.0, -0.20, -0.20, -300.0, -0.005, -0.005])
            ub = np.array([0.20, 0.20, 300.0, 0.20, 0.20, 300.0, 0.005, 0.005])
            res = least_squares(resid, np.zeros(8), bounds=(lb, ub), loss="soft_l1",
                                f_scale=3.0, max_nfev=max_nfev,
                                x_scale=[0.03, 0.03, 40.0, 0.03, 0.03, 40.0, 0.0008, 0.0008])
        C = unpack(res.x)
        Hn = C @ H_base
        if not np.all(np.isfinite(Hn)) or abs(Hn[2, 2]) < 1e-12:
            return None, res
        return Hn / Hn[2, 2], res

    H_start = H0
    coarse_used = False
    coarse_metrics = None
    # Only escape a genuinely poor starting fit.  A good anchor with one
    # mislabelled line should be reported, not dragged around by that line.
    if metrics0["line_median_px"] > 5.0:
        H_coarse, _ = _optimise(H0, coarse=True)
        if H_coarse is not None:
            coarse_metrics = _line_metrics_from_distances(
                _line_distances_from_H(H_coarse, drawn, template, width, height))
            if (coarse_metrics["line_median_px"] <= metrics0["line_median_px"] * 0.95
                    or coarse_metrics["line_median_px"] <= 5.0):
                H_start = H_coarse
                coarse_used = True

    Hn, res = _optimise(H_start, coarse=False)
    if Hn is None:
        return H0, {"accepted": False, "reason": "degenerate correction",
                    "method": "line-refine", "used_lines": len(drawn),
                    "line_median_px": metrics0["line_median_px"],
                    "worst_line_px": metrics0["worst_line_px"],
                    "per_line": metrics0["per_line"],
                    "outlier_lines": metrics0.get("outlier_lines", []),
                    "outlier_threshold_px": metrics0.get("outlier_threshold_px"),
                    "coarse_used": coarse_used}
    metrics1 = _line_metrics_from_distances(
        _line_distances_from_H(Hn, drawn, template, width, height))
    # The fine stage should not make the fit worse than its starting point.
    start_metrics = _line_metrics_from_distances(
        _line_distances_from_H(H_start, drawn, template, width, height))
    if ((metrics1["line_median_px"], metrics1["worst_line_px"])
            > (start_metrics["line_median_px"], start_metrics["worst_line_px"])):
        Hn, metrics1 = H_start, start_metrics
    worst_after = metrics1["worst_line_px"]
    accepted = (metrics1["line_median_px"] <= max(10.0, metrics0["line_median_px"] * 1.05)
                and worst_after <= 12.0
                and (metrics1["line_median_px"] <= metrics0["line_median_px"] * 0.95
                     or metrics1["line_median_px"] <= 5.0))
    point_before = point_after = None
    point_warning = False
    if world_pts is not None and len(world_pts):
        point_before = float(np.median(np.linalg.norm(
            _project_ground(H0, world_pts) - image_pts, axis=1)))
        point_after = float(np.median(np.linalg.norm(
            _project_ground(Hn, world_pts) - image_pts, axis=1)))
        # The points are a seed, not ground truth.  A large gap is reported but
        # does not reject a good line fit.
        if point_after > max(40.0, point_before * 4.0):
            point_warning = True
    final = metrics1 if accepted else metrics0
    H_final = Hn if accepted else H0
    point_errors = None
    point_outlier_indices: List[int] = []
    point_outlier_threshold = None
    if world_pts is not None and len(world_pts):
        errs = np.linalg.norm(_project_ground(H_final, world_pts) - image_pts, axis=1)
        point_errors = [float(e) for e in errs]
        point_median = float(np.median(errs))
        point_outlier_threshold = max(OUTLIER_POINT_MIN_PX,
                                      OUTLIER_POINT_REL * point_median)
        point_outlier_indices = [int(i) for i in
                                 np.flatnonzero(errs > point_outlier_threshold)]
    info = {
        "method": "line-refine",
        "accepted": bool(accepted),
        "line_median_px": final["line_median_px"],
        "line_hit": final["line_hit"],
        "line_coverage": final["line_coverage"],
        "per_line": final["per_line"],
        "worst_line": final["worst_line"],
        "worst_line_px": final["worst_line_px"],
        "line_points": final["line_points"],
        "outlier_lines": final.get("outlier_lines", []),
        "outlier_threshold_px": final.get("outlier_threshold_px"),
        "line_median_before_px": metrics0["line_median_px"],
        "line_median_candidate_px": metrics1["line_median_px"],
        "worst_line_candidate": metrics1["worst_line"],
        "worst_line_candidate_px": metrics1["worst_line_px"],
        "per_line_candidate": metrics1["per_line"],
        "outlier_lines_candidate": metrics1.get("outlier_lines", []),
        "line_hit_before": metrics0["line_hit"],
        "line_hit_candidate": metrics1["line_hit"],
        "line_coverage_before": metrics0["line_coverage"],
        "line_coverage_candidate": metrics1["line_coverage"],
        "coarse_used": coarse_used,
        "coarse_median_px": (coarse_metrics["line_median_px"]
                             if coarse_metrics else None),
        "coarse_worst_px": (coarse_metrics["worst_line_px"]
                            if coarse_metrics else None),
        "used_lines": len(drawn),
        "used_points": int(0 if world_pts is None else len(world_pts)),
        "point_before_px": point_before,
        "point_after_px": point_after,
        "point_errors_px": point_errors,
        "point_outlier_indices": point_outlier_indices,
        "point_outlier_threshold_px": point_outlier_threshold,
        "point_warning": point_warning,
        "correction": res.x.tolist(),
        "correction_base": "coarse" if coarse_used else "initial",
    }
    if not accepted:
        pitch_hint = _pitch_size_hint(
            Hn, drawn, template, width, height,
            metrics1["worst_line"], metrics1["worst_line_px"])
        if pitch_hint:
            info["pitch_hint"] = pitch_hint
        if metrics1["worst_line_px"] > 12.0:
            info["warning"] = (
                f"worst line {metrics1['worst_line']} is "
                f"{metrics1['worst_line_px']:.0f} px off; check the top/bottom "
                f"line label and the --pitch size")
        else:
            info["warning"] = (
                "line fit did not improve enough; check the traced line or add "
                "another visible line")
        if pitch_hint:
            info["warning"] = info["warning"] + "; " + pitch_hint
    elif coarse_used and metrics0["line_median_px"] > 5.0:
        info["warning"] = ("the initial seed was far from the traced lines; a "
                           "wide correction was used.  Check the far field in "
                           "the BEV.")
    return (Hn if accepted else H0), info


def _camera_candidates_for_lines(template: PitchTemplate, width: int, height: int,
                                 focal_prior: Optional[float] = None):
    """Plausible touchline/tactical broadcast cameras for a from-scratch solve."""
    hx, hy = template.hx, template.hy
    foci = ([focal_prior * 0.8, focal_prior, focal_prior * 1.25]
            if focal_prior else [0.75 * width, 1.0 * width, 1.35 * width])
    target = np.array([0.0, 0.0, 0.0])
    for side in (1.0, -1.0):
        for off in (4.0, 15.0, 30.0):
            cy = side * (hy + off)
            for cx in (-0.2 * hx, 0.0, 0.2 * hx):
                for cz in (5.0, 12.0, 22.0):
                    C = np.array([cx, cy, cz], float)
                    R = look_at_rotation(C, target)
                    for f in foci:
                        yield PinholeCamera(f, width / 2.0, height / 2.0, R, C)
    # Cameras behind a goal line are less common but appear in tactical footage.
    for side in (1.0, -1.0):
        for off in (5.0, 20.0):
            cx = side * (hx + off)
            for cy in (-0.25 * hy, 0.0, 0.25 * hy):
                for cz in (6.0, 14.0):
                    C = np.array([cx, cy, cz], float)
                    R = look_at_rotation(C, target)
                    for f in foci:
                        yield PinholeCamera(f, width / 2.0, height / 2.0, R, C)


def _fit_camera_to_lines(cam0: PinholeCamera, drawn: Dict[str, List[np.ndarray]],
                         template: PitchTemplate, width: int, height: int,
                         focal_prior: Optional[float] = None,
                         camera_height_prior: Optional[float] = None,
                         max_nfev: int = 70) -> Tuple[PinholeCamera, object]:
    f0, R0, C0 = float(cam0.f), cam0.R.copy(), cam0.C.copy()
    weights = _line_weight_map(
        _line_distances_from_H(cam0.homography(), drawn, template, width, height))

    def unpack(p: np.ndarray) -> PinholeCamera:
        f = f0 * math.exp(float(p[0]))
        R = rvec_to_rotation(p[1:4]) @ R0
        C = C0 + np.array([p[4], p[5], p[6]], float)
        return PinholeCamera(f, cam0.cx, cam0.cy, R, C)

    def resid(p: np.ndarray) -> np.ndarray:
        cam = unpack(p)
        dists = _line_distances_from_H(cam.homography(), drawn, template, width, height)
        d = np.concatenate([dists[n] * weights.get(n, 1.0) for n in dists]) \
            if dists else np.zeros(0, float)
        reg = np.array([15.0 * p[0], 20.0 * p[1], 20.0 * p[2],
                        20.0 * p[3], 0.15 * p[4], 0.15 * p[5], 0.15 * p[6]])
        pieces = [d, reg]
        if focal_prior is not None:
            pieces.append(np.array([5.0 * (math.log(max(cam.f, 1e-3))
                                            - math.log(max(focal_prior, 1e-3)))]))
        if camera_height_prior is not None:
            pieces.append(np.array([8.0 * (math.log(max(cam.height, 0.1))
                                            - math.log(max(camera_height_prior, 0.1)))]))
        return np.concatenate(pieces)

    lb = np.array([-0.6, -0.55, -0.55, -0.55, -35.0, -35.0,
                   max(-35.0, 1.0 - C0[2])])
    ub = np.array([0.6, 0.55, 0.55, 0.55, 35.0, 35.0, 40.0])
    res = least_squares(resid, np.zeros(7), bounds=(lb, ub), loss="soft_l1",
                        f_scale=3.0, max_nfev=max_nfev,
                        x_scale=[0.2, 0.1, 0.1, 0.1, 5.0, 5.0, 5.0])
    return unpack(res.x), res


def _lines_only_multistart(drawn: Dict[str, List[np.ndarray]],
                           template: PitchTemplate, width: int, height: int,
                           focal_prior: Optional[float] = None,
                           camera_height_prior: Optional[float] = None,
                           max_camera_candidates: int = 6,
                           used_points: int = 0) -> Tuple[np.ndarray, dict]:
    """Fit a plausible camera to the lines when no anchor/point seed works."""
    scored = []
    for cam in _camera_candidates_for_lines(template, width, height, focal_prior):
        m = _line_metrics_from_distances(
            _line_distances_from_H(cam.homography(), drawn, template, width, height))
        scored.append((_line_quality_score(m), cam, m))
    if not scored:
        raise ValueError("line solve produced no camera candidates")
    scored.sort(key=lambda z: z[0])
    best_cam = None
    best_metrics = None
    best_info = None
    for _, cam0, _ in scored[:max(1, int(max_camera_candidates))]:
        cam, res = _fit_camera_to_lines(cam0, drawn, template, width, height,
                                        focal_prior=focal_prior,
                                        camera_height_prior=camera_height_prior)
        m = _line_metrics_from_distances(
            _line_distances_from_H(cam.homography(), drawn, template, width, height))
        if best_metrics is None or _line_quality_score(m) < _line_quality_score(best_metrics):
            best_cam, best_metrics = cam, m
            best_info = {"cost": float(res.cost), "camera_candidates": len(scored)}
    if best_cam is None or best_metrics is None:
        raise RuntimeError("line-only solve did not converge")
    H = best_cam.homography()
    H = H / H[2, 2]
    accepted = (best_metrics["line_median_px"] <= 8.0
                and best_metrics["worst_line_px"] <= 12.0)
    info = {
        "method": "lines-only-multistart",
        "accepted": bool(accepted),
        "line_median_px": best_metrics["line_median_px"],
        "line_hit": best_metrics["line_hit"],
        "line_coverage": best_metrics["line_coverage"],
        "per_line": best_metrics["per_line"],
        "worst_line": best_metrics["worst_line"],
        "worst_line_px": best_metrics["worst_line_px"],
        "outlier_lines": best_metrics.get("outlier_lines", []),
        "outlier_threshold_px": best_metrics.get("outlier_threshold_px"),
        "line_points": best_metrics["line_points"],
        "used_lines": len(drawn),
        "used_points": int(used_points),
        "camera": {"f": float(best_cam.f), "C": best_cam.C.tolist(),
                   "height": float(best_cam.height)},
        **(best_info or {}),
        "warning": ("lines-only solve: the far field is extrapolated."
                    + (f"  Only {len(drawn)} line(s) traced; add the centre "
                       "circle or another visible line." if len(drawn) < 3 else "")
                    + "  Mark a previous anchor (--append) or click 4 rough "
                      "points before tracing lines for a metric anchor."),
    }
    if best_metrics["line_median_px"] > 25.0:
        raise RuntimeError(
            "line-only solve is too far off (median {:.1f} px on the traced "
            "lines).  Check the line labels, or add 4 rough point clicks / a "
            "previous anchor.".format(best_metrics["line_median_px"]))
    return H, info


def solve_homography_from_lines(lines, template: PitchTemplate, width: int, height: int,
                                points=None, init_H: Optional[np.ndarray] = None,
                                allow_from_scratch: bool = True,
                                focal_prior: Optional[float] = None,
                                camera_height_prior: Optional[float] = None,
                                max_camera_candidates: int = 6) -> Tuple[np.ndarray, dict]:
    """Solve a world->image homography from manually traced white lines.

    Preferred order:
    1. refine the previous anchor / a point-solved anchor with the lines;
    2. if that is rejected, try a lines-only camera fit; a bad point seed can
       otherwise trap the refinement in a basin the lines cannot escape;
    3. if there is no seed at all, use the lines-only fit directly (marked as
       extrapolated, because a central view does not observe the far field).
    """
    drawn = _normalise_line_annotations(lines, template)
    if not drawn:
        raise ValueError("no usable line annotations")
    world_pts, image_pts = _point_annotations(points)
    layout_problem = None
    if world_pts is not None:
        layout_problem = point_layout_problem(world_pts, image_pts)
        if layout_problem is not None:
            # Do not let a degenerate click set drag the line fit.  The clicks
            # are guidance, not measurements, but a rank-deficient set has no
            # information in at least one direction and can only mislead it.
            world_pts, image_pts = None, None
    used_points = 0 if world_pts is None else len(world_pts)

    candidates: List[Tuple[str, np.ndarray]] = []
    if init_H is not None:
        candidates.append(("previous-anchor", np.asarray(init_H, float)))
    if world_pts is not None and len(world_pts) >= 4:
        try:
            Hp, _ = solve_homography_from_clicks(world_pts, image_pts)
            candidates.append(("points", Hp))
        except Exception:
            pass

    best = None
    for method, H0 in candidates:
        try:
            Hn, info = refine_homography_to_lines(
                H0, lines, template, width, height,
                points=(points if world_pts is not None else None))
        except Exception:
            continue
        score = (float(info.get("line_median_px", 80.0))
                 + 0.2 * float(info.get("point_after_px") or 0.0))
        key = (0 if info.get("accepted") else 1, score)
        if best is None or key < best[0]:
            best = (key, method, Hn, info)

    if best is not None:
        _, method, Hn, info = best
        info["method"] = f"{method}+lines"
        info["used_lines"] = len(drawn)
        if layout_problem:
            info["point_layout_problem"] = layout_problem
        if info.get("accepted") or not allow_from_scratch:
            return Hn, info
        if method == "points" and world_pts is not None and len(world_pts) >= 4:
            # A well-conditioned point seed carries the far-field scale that
            # lines alone cannot observe.  A lines-only fit would replace a
            # known metric anchor with an arbitrary extrapolation, so keep the
            # seeded result and report the disagreement instead.
            info["warning"] = (
                "the traced lines do not fully agree with the clicked points; "
                "the point seed was kept for the far field.  Check the line "
                "labels (top vs bottom) and the --pitch size.")
            return Hn, info
        # The seed candidate was rejected.  Try the lines alone: the point
        # seed may be in a basin that the local correction cannot leave.
        try:
            Hs, infos = _lines_only_multistart(
                drawn, template, width, height, focal_prior=focal_prior,
                camera_height_prior=camera_height_prior,
                max_camera_candidates=max_camera_candidates,
                used_points=used_points)
        except Exception:
            return Hn, info
        def _q(inf: dict) -> Tuple[int, float]:
            return (0 if inf.get("accepted") else 1,
                    _line_quality_score(inf))
        if _q(infos) < _q(info):
            infos["point_seed_rejected"] = True
            infos["warning"] = (
                "the point/previous seed disagreed with the traced lines, so "
                "the lines were solved without it; the far field is "
                "extrapolated."
                + (f"  Only {len(drawn)} line(s) traced; add the centre circle "
                   "or another visible line." if len(drawn) < 3 else "")
                + "  If the lines are correct, check the seed points; if the "
                  "far field matters, add a known anchor.")
            return Hs, infos
        return Hn, info
    if not allow_from_scratch:
        raise ValueError("line annotations need a previous anchor or at least "
                         "4 point clicks (switch to point mode with 'm')")
    H, info = _lines_only_multistart(
        drawn, template, width, height, focal_prior=focal_prior,
        camera_height_prior=camera_height_prior,
        max_camera_candidates=max_camera_candidates,
        used_points=used_points)
    if layout_problem:
        info["point_layout_problem"] = layout_problem
        info["warning"] = info.get("warning", "") + "  " + layout_problem
    return H, info
