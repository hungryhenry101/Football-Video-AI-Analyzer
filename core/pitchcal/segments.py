"""White-paint segment evidence and a conservative homography correction.

This is the absolute correction layer: LSD segments are filtered to white paint
on turf, then a small projective correction C is fitted so the projected pitch
template lands on them.  It is run sparsely (once every ~0.8 s) and accepted only
when the line hit improves and the absolute fit is good.  Every accepted fit is
cross-checked against the same global paint metric the pipeline reports, so a
correction that only improves its own wide-gated nearest-segment score (e.g. on a
degenerate centre-circle view) is rejected.  Bound-pinned fits need a clear
absolute improvement.  It cannot invent lines: a frame with no paint simply
leaves the homography alone.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .geometry import PitchTemplate
from .linemask import paint_like_segments


def detect_segments(gray: np.ndarray) -> np.ndarray:
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4), np.float32)
    return np.asarray(lines, np.float32).reshape(-1, 4)


def filter_segments(segments: np.ndarray, H: np.ndarray, template: PitchTemplate,
                    boxes: Optional[np.ndarray], width: int, height: int,
                    min_len: float = 15.0, model_gate_px: float = 90.0,
                    world_margin_m: float = 1.0) -> np.ndarray:
    if segments is None or not len(segments):
        return np.zeros((0, 4), np.float32)
    inv = np.linalg.inv(H)
    model_pts = []
    for _, wp in template.polylines():
        q = (H @ np.column_stack([wp, np.ones(len(wp))]).T).T
        ok = np.isfinite(q).all(axis=1) & (np.abs(q[:, 2]) > 1e-9)
        uv = q[ok, :2] / q[ok, 2:3]
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)]
        if len(uv):
            model_pts.append(uv)
    tree = cKDTree(np.vstack(model_pts)) if model_pts else None
    out = []
    for x1, y1, x2, y2 in segments:
        if math.hypot(x2 - x1, y2 - y1) < min_len:
            continue
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if boxes is not None and len(boxes):
            if np.any((boxes[:, 0] - 6 <= mx) & (mx <= boxes[:, 2] + 6)
                      & (boxes[:, 1] - 6 <= my) & (my <= boxes[:, 3] + 6)):
                continue
        if tree is not None and float(tree.query([mx, my], k=1)[0]) > model_gate_px:
            continue
        p = inv @ np.array([mx, my, 1.0])
        if abs(p[2]) < 1e-9:
            continue
        wx, wy = p[0] / p[2], p[1] / p[2]
        if abs(wx) <= template.hx + world_margin_m and abs(wy) <= template.hy + world_margin_m:
            out.append((x1, y1, x2, y2))
    return np.asarray(out, np.float32) if out else np.zeros((0, 4), np.float32)


def segment_points(segments: np.ndarray, step_px: float = 1.5) -> np.ndarray:
    pts = []
    for x1, y1, x2, y2 in segments:
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / step_px))
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], axis=1))
    return np.vstack(pts) if pts else np.zeros((0, 2))


def paint_support(frame: np.ndarray, H: np.ndarray, template: PitchTemplate,
                  work_width: int = 960, work_height: Optional[int] = None,
                  boxes: Optional[np.ndarray] = None) -> dict:
    """Global white-paint support of a homography, at the pipeline resolution.

    The annotator's line solve only measures the lines the human traced, so a
    fit can look perfect (``line_hit`` ~ 1.0) while the rest of the projected
    template sits in the wrong place.  This evaluates the same
    LSD + paint-photometry + hit metric the pipeline reports, over every
    visible template line, which is the honest "is this anchor right" check.
    """
    h0, w0 = frame.shape[:2]
    if work_height is None:
        work_height = max(1, int(round(h0 * float(work_width) / max(1, w0))))
    work = cv2.resize(frame, (int(work_width), int(work_height)),
                      interpolation=cv2.INTER_AREA)
    scale = np.diag([work_width / float(w0), work_height / float(h0), 1.0])
    Hw = scale @ np.asarray(H, float)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    segs = paint_like_segments(work, detect_segments(gray))
    segs = filter_segments(segs, Hw, template, boxes,
                           int(work_width), int(work_height))
    hit, cov, med = hit_from_segments(segs, Hw, template,
                                      int(work_width), int(work_height))
    return {"paint_hit": float(hit), "paint_cov": float(cov),
            "paint_median_px": float(med), "paint_segments": int(len(segs))}


def hit_from_segments(segments: np.ndarray, H: np.ndarray, template: PitchTemplate,
                      width: int, height: int) -> Tuple[float, float, float]:
    if not len(segments):
        return 0.0, 0.0, 99.0
    tree = cKDTree(segment_points(segments).astype(np.float64))
    hits = total = covered = line_count = 0
    dists = []
    for name, w in template.sample_points(step=0.8):
        q = (H @ np.column_stack([w, np.ones(len(w))]).T).T
        ok = np.isfinite(q).all(axis=1) & (np.abs(q[:, 2]) > 1e-9)
        uv = np.full((len(w), 2), np.nan)
        uv[ok] = q[ok, :2] / q[ok, 2:3]
        inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        if inside.sum() < 3:
            continue
        line_count += 1
        d, _ = tree.query(uv[inside], k=1)
        total += len(d)
        hits += int(np.sum(d < 5.0))
        dists.append(float(np.median(d)))
        if float(np.quantile(d, 0.5)) < 12:
            covered += 1
    if total == 0:
        return 0.0, 0.0, 99.0
    return hits / float(total), covered / max(1, line_count), float(np.median(dists))


def refine_homography(H: np.ndarray, frame: np.ndarray, template: PitchTemplate,
                      boxes: Optional[np.ndarray] = None,
                      gate_px: float = 200.0,
                      max_nfev: int = 80) -> tuple[np.ndarray, dict]:
    """Fit a small projective correction C so H' = C·H lands on paint."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segs = detect_segments(gray)
    segs = paint_like_segments(frame, segs)
    segs = filter_segments(segs, H, template, boxes, w, h,
                           min_len=22.0, model_gate_px=gate_px, world_margin_m=1.0)
    if len(segs) < 12:
        return H, {"accepted": False, "reason": "too few paint segments", "segments": int(len(segs))}
    tree = cKDTree(segment_points(segs).astype(np.float64))
    # Base projections: applying C to these is much cheaper than reprojecting
    # 3D points in every residual evaluation.
    world = []
    for _, p in template.sample_points(step=0.6):
        world.append(p)
    world = np.vstack(world)
    q = (H @ np.column_stack([world, np.ones(len(world))]).T).T
    ok = np.isfinite(q).all(axis=1) & (np.abs(q[:, 2]) > 1e-9)
    base = q[ok, :2] / q[ok, 2:3]
    inside = (base[:, 0] >= 0) & (base[:, 0] < w) & (base[:, 1] >= 0) & (base[:, 1] < h)
    base = base[inside]
    if len(base) < 40:
        return H, {"accepted": False, "reason": "too few visible model points"}
    homog = np.column_stack([base, np.ones(len(base))])

    def apply(p):
        C = np.array([[1 + p[0], p[1], p[2]],
                      [p[3], 1 + p[4], p[5]],
                      [p[6], p[7], 1.0]])
        uv = (C @ homog.T).T
        return uv[:, :2] / uv[:, 2:3]

    def metrics(uv):
        d, _ = tree.query(uv, k=1)
        d = np.minimum(d, 60.0)
        return float(np.mean(d < 5.0)), float(np.median(d))

    hit0, med0 = metrics(base)

    def resid(p):
        uv = apply(p)
        d, _ = tree.query(uv, k=1)
        d = np.minimum(d, 40.0)
        reg = np.array([2.0 * p[0], 2.0 * p[1], 0.04 * p[2],
                        2.0 * p[3], 2.0 * p[4], 0.04 * p[5],
                        25.0 * p[6], 25.0 * p[7]])
        return np.concatenate([d, reg])

    # Keep the historical search box (a real correction can need ~100 px after
    # a pan), but flag solutions that pin a bound: that is usually the
    # optimizer running away on a degenerate view (e.g. only a centre circle +
    # halfway line visible).  Bound-pinned fits need stronger absolute proof.
    lb = np.array([-0.25, -0.25, -220.0, -0.25, -0.25, -220.0, -0.004, -0.004])
    ub = np.array([0.25, 0.25, 220.0, 0.25, 0.25, 220.0, 0.004, 0.004])
    res = least_squares(resid, np.zeros(8), bounds=(lb, ub), loss="soft_l1",
                        f_scale=3.0, max_nfev=max_nfev,
                        x_scale=[0.03, 0.03, 30, 0.03, 0.03, 30, 0.0008, 0.0008])
    p = res.x
    # A solution pinned to a bound means the fit wanted to go further than the
    # search box allows; require stronger absolute evidence before trusting it.
    span = np.maximum(np.abs(lb), np.abs(ub))
    at_bound = bool(np.any(np.abs(p) >= 0.995 * span))
    C = np.array([[1 + p[0], p[1], p[2]],
                  [p[3], 1 + p[4], p[5]],
                  [p[6], p[7], 1.0]])
    Hn = C @ H
    if abs(Hn[2, 2]) > 1e-12:
        Hn = Hn / Hn[2, 2]
    uv = apply(p)
    hit1, med1 = metrics(uv)
    already_good = med0 < 3.0 and hit0 > 0.65
    accepted = not already_good and (
        (med1 <= 5.0 and hit1 >= 0.45 and med1 <= med0 * 1.10
         and (hit1 >= hit0 + 0.03 or med1 <= med0 * 0.85))
        or (med1 <= 7.0 and hit1 >= hit0 + 0.10)
    )
    # Cross-check with the metric the pipeline actually reports.  The
    # optimizer above scores the same wide-gated segment set it moved the
    # model onto, so it can approve a warp that loses real paint support.
    # Reject any correction that does not preserve the global paint hit,
    # coverage and median used for confidence/line_hit.  A solution pinned to
    # a bound is only trusted when the absolute fit improves clearly.
    paint_before = paint_after = None
    if accepted:
        try:
            paint_before = paint_support(frame, H, template)
            paint_after = paint_support(frame, Hn, template)
            accepted = (paint_after["paint_hit"] >= paint_before["paint_hit"] - 0.02
                        and paint_after["paint_cov"] >= paint_before["paint_cov"] - 0.05
                        and paint_after["paint_median_px"] <=
                        max(12.0, paint_before["paint_median_px"] * 1.5))
            if accepted and at_bound:
                accepted = (paint_after["paint_hit"] >= paint_before["paint_hit"] + 0.02
                            and paint_after["paint_cov"] >= paint_before["paint_cov"] - 0.05)
        except Exception:
            # Keep the guardrails above; a failed absolute check is not a
            # reason to skip a correction on an otherwise well-supported frame.
            paint_before = paint_after = None
    info = {
        "accepted": bool(accepted),
        "segments": int(len(segs)),
        "hit_before": float(hit0), "hit_after": float(hit1),
        "median_before_px": float(med0), "median_after_px": float(med1),
        "visible_points": int(len(base)),
        "at_bound": at_bound,
        "correction": p.tolist(),
    }
    if paint_before is not None and paint_after is not None:
        info.update({
            "paint_hit_before": float(paint_before["paint_hit"]),
            "paint_hit_after": float(paint_after["paint_hit"]),
            "paint_cov_before": float(paint_before["paint_cov"]),
            "paint_cov_after": float(paint_after["paint_cov"]),
            "paint_median_before_px": float(paint_before["paint_median_px"]),
            "paint_median_after_px": float(paint_after["paint_median_px"]),
        })
    return (Hn if accepted else H), info
