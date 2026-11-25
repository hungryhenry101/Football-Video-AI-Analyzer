"""Pitch geometry, camera model and planar homography utilities.

World frame (metres): origin at the pitch centre, x along the length towards
the right goal line, y across the width, z up.  Image frame: pixels, u right,
v down.  Camera frame: x right, y down, z forward.

The ground-plane homography is the object every other module uses:

    H = K [r1 r2 -R C]      (z = 0 world plane -> image)

It is all that is needed to project pixels/feet to a BEV.  The pinhole camera
(f, R, C) is kept because it makes temporal tracking and line registration much
better conditioned than optimising eight unconstrained homography entries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CENTER_CIRCLE_RADIUS = 9.15
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.32
GOAL_WIDTH = 7.32
PENALTY_MARK_DISTANCE = 11.0
CORNER_ARC_RADIUS = 1.0


def _arc_points(cx: float, cy: float, radius: float, a0: float, a1: float, n: int = 48) -> np.ndarray:
    angles = np.linspace(np.deg2rad(a0), np.deg2rad(a1), n)
    return np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)


class PitchTemplate:
    """FIFA/IFAB 11-a-side pitch template with configurable outer dimensions.

    Penalty/goal areas, the centre circle, penalty marks and goals keep their
    regulation absolute sizes when the outer length/width change -- exactly how
    youth 90x64 pitches are marked.
    """

    def __init__(self, length: float = 105.0, width: float = 68.0):
        if length <= 0 or width <= 0:
            raise ValueError("pitch dimensions must be positive")
        self.length = float(length)
        self.width = float(width)
        self.hx = self.length / 2.0
        self.hy = self.width / 2.0
        self._landmarks = self._build_landmarks()

    # ---------------------------------------------------------------- template
    def _build_landmarks(self) -> Dict[str, Tuple[float, float]]:
        hx, hy = self.hx, self.hy
        pa_x, pa_y = hx - PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH / 2.0
        ga_x, ga_y = hx - GOAL_AREA_LENGTH, GOAL_AREA_WIDTH / 2.0
        spot_x = hx - PENALTY_MARK_DISTANCE
        arc_dx = PENALTY_AREA_LENGTH - PENALTY_MARK_DISTANCE
        arc_y = math.sqrt(max(CENTER_CIRCLE_RADIUS**2 - arc_dx**2, 0.0))
        d: Dict[str, Tuple[float, float]] = {
            "center_mark": (0.0, 0.0),
            "center_circle_top": (0.0, -CENTER_CIRCLE_RADIUS),
            "center_circle_bottom": (0.0, CENTER_CIRCLE_RADIUS),
            "center_circle_left": (-CENTER_CIRCLE_RADIUS, 0.0),
            "center_circle_right": (CENTER_CIRCLE_RADIUS, 0.0),
            "halfway_top_touch": (0.0, -hy),
            "halfway_bottom_touch": (0.0, hy),
            "left_goal_center": (-hx, 0.0),
            "right_goal_center": (hx, 0.0),
            "left_post_top": (-hx, -GOAL_WIDTH / 2.0),
            "left_post_bottom": (-hx, GOAL_WIDTH / 2.0),
            "right_post_top": (hx, -GOAL_WIDTH / 2.0),
            "right_post_bottom": (hx, GOAL_WIDTH / 2.0),
        }
        for side, sx in (("left", -1.0), ("right", 1.0)):
            goal_line = sx * hx
            inner_x = sx * pa_x
            goal_inner_x = sx * ga_x
            d.update({
                f"{side}_corner_top": (goal_line, -hy),
                f"{side}_corner_bottom": (goal_line, hy),
                f"{side}_penalty_top_inner": (inner_x, -pa_y),
                f"{side}_penalty_bottom_inner": (inner_x, pa_y),
                f"{side}_penalty_top_goal_line": (goal_line, -pa_y),
                f"{side}_penalty_bottom_goal_line": (goal_line, pa_y),
                f"{side}_goal_area_top_inner": (goal_inner_x, -ga_y),
                f"{side}_goal_area_bottom_inner": (goal_inner_x, ga_y),
                f"{side}_goal_area_top_goal_line": (goal_line, -ga_y),
                f"{side}_goal_area_bottom_goal_line": (goal_line, ga_y),
                f"{side}_penalty_mark": (sx * spot_x, 0.0),
                f"{side}_penalty_arc_top": (inner_x, -arc_y),
                f"{side}_penalty_arc_bottom": (inner_x, arc_y),
            })
        return d

    @property
    def landmarks(self) -> Dict[str, Tuple[float, float]]:
        return self._landmarks

    def landmark_array(self, names: Sequence[str]) -> np.ndarray:
        return np.asarray([self._landmarks[n] for n in names], dtype=np.float64)

    # ------------------------------------------------------------- line model
    def polylines(self) -> List[Tuple[str, np.ndarray]]:
        """Named ground-plane polylines, sampled densely (world metres)."""
        hx, hy = self.hx, self.hy
        pa_x, pa_y = hx - PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH / 2.0
        ga_x, ga_y = hx - GOAL_AREA_LENGTH, GOAL_AREA_WIDTH / 2.0
        spot_x = hx - PENALTY_MARK_DISTANCE
        arc_dx = PENALTY_AREA_LENGTH - PENALTY_MARK_DISTANCE
        arc_y = math.sqrt(max(CENTER_CIRCLE_RADIUS**2 - arc_dx**2, 0.0))
        lines: List[Tuple[str, np.ndarray]] = []
        lines.append(("touchline_top", np.array([[-hx, -hy], [hx, -hy]], float)))
        lines.append(("touchline_bottom", np.array([[-hx, hy], [hx, hy]], float)))
        lines.append(("goal_line_left", np.array([[-hx, -hy], [-hx, hy]], float)))
        lines.append(("goal_line_right", np.array([[hx, -hy], [hx, hy]], float)))
        lines.append(("halfway_line", np.array([[0.0, -hy], [0.0, hy]], float)))
        lines.append(("center_circle", _arc_points(0, 0, CENTER_CIRCLE_RADIUS, 0, 360, 96)))
        for side, sx in (("left", -1.0), ("right", 1.0)):
            goal = sx * hx
            inner = sx * pa_x
            goal_inner = sx * ga_x
            lines += [
                (f"{side}_penalty_front", np.array([[inner, -pa_y], [inner, pa_y]], float)),
                (f"{side}_penalty_top", np.array([[goal, -pa_y], [inner, -pa_y]], float)),
                (f"{side}_penalty_bottom", np.array([[goal, pa_y], [inner, pa_y]], float)),
                (f"{side}_goal_area_front", np.array([[goal_inner, -ga_y], [goal_inner, ga_y]], float)),
                (f"{side}_goal_area_top", np.array([[goal, -ga_y], [goal_inner, -ga_y]], float)),
                (f"{side}_goal_area_bottom", np.array([[goal, ga_y], [goal_inner, ga_y]], float)),
                (f"{side}_penalty_arc", _arc_points(
                    sx * spot_x, 0.0, CENTER_CIRCLE_RADIUS,
                    (-math.degrees(math.acos(arc_dx / CENTER_CIRCLE_RADIUS)) if sx < 0
                     else 180.0 - math.degrees(math.acos(arc_dx / CENTER_CIRCLE_RADIUS))),
                    (math.degrees(math.acos(arc_dx / CENTER_CIRCLE_RADIUS)) if sx < 0
                     else 180.0 + math.degrees(math.acos(arc_dx / CENTER_CIRCLE_RADIUS))),
                    32)),
            ]
            # Only the part outside the penalty area is painted: for the left
            # goal this is the cap facing +x (angles -phi..+phi), for the
            # right goal the cap facing -x (180-phi..180+phi).  The old code
            # used the opposite cap, which drew the arc inside the penalty box.
        # Corner arcs (quarter circles inside the pitch).
        r = CORNER_ARC_RADIUS
        lines += [
            ("corner_tl", _arc_points(-hx, -hy, r, 0, 90, 16)),
            ("corner_tr", _arc_points(hx, -hy, r, 90, 180, 16)),
            ("corner_br", _arc_points(hx, hy, r, 180, 270, 16)),
            ("corner_bl", _arc_points(-hx, hy, r, 270, 360, 16)),
        ]
        return lines

    def sample_points(self, step: float = 0.5, include_circles: bool = True) -> List[Tuple[str, np.ndarray]]:
        """Resample every polyline to roughly ``step`` metres."""
        out: List[Tuple[str, np.ndarray]] = []
        for name, pts in self.polylines():
            pts = np.asarray(pts, float)
            if not include_circles and ("circle" in name or "arc" in name or "corner" in name):
                continue
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            if len(seg) == 0:
                continue
            s = np.concatenate([[0.0], np.cumsum(seg)])
            total = float(s[-1])
            if total < 1e-6:
                continue
            n = max(2, int(round(total / step)) + 1)
            q = np.linspace(0, total, n)
            x = np.interp(q, s, pts[:, 0])
            y = np.interp(q, s, pts[:, 1])
            out.append((name, np.stack([x, y], axis=1)))
        return out

    # ------------------------------------------------------- BEV conversions
    def world_to_bev(self, xy: np.ndarray, scale: float = 10.0, margin: float = 2.0) -> np.ndarray:
        """World metres -> BEV pixels, origin top-left, with a small margin."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        px = (xy[:, 0] + self.hx + margin) * scale
        py = (xy[:, 1] + self.hy + margin) * scale
        return np.stack([px, py], axis=1)

    def bev_size(self, scale: float = 10.0, margin: float = 2.0) -> Tuple[int, int]:
        return (int(round((self.length + 2 * margin) * scale)),
                int(round((self.width + 2 * margin) * scale)))


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def look_at_rotation(center: np.ndarray, target: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """World->camera rotation for a camera at ``center`` looking at ``target``."""
    center = np.asarray(center, float).reshape(3)
    target = np.asarray(target, float).reshape(3)
    forward = target - center
    n = np.linalg.norm(forward)
    if n < 1e-9:
        raise ValueError("camera and target coincide")
    z = forward / n
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    if roll:
        a = math.radians(roll)
        xr = math.cos(a) * x + math.sin(a) * y
        yr = -math.sin(a) * x + math.cos(a) * y
        x, y = xr, yr
    return np.stack([x, y, z], axis=0)


def rotation_to_rvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, float))
    return rvec.reshape(3)


def rvec_to_rotation(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    return R


def rotation_about_axes(rx: float, ry: float, rz: float) -> np.ndarray:
    """Incremental rotation in camera frame, radians."""
    return rvec_to_rotation(np.array([rx, ry, rz], float))


@dataclass
class PinholeCamera:
    f: float
    cx: float
    cy: float
    R: np.ndarray  # world -> camera
    C: np.ndarray  # camera centre, world metres (z up)

    def copy(self) -> "PinholeCamera":
        return PinholeCamera(float(self.f), float(self.cx), float(self.cy),
                             self.R.copy(), self.C.copy())

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]])

    @property
    def height(self) -> float:
        return float(self.C[2])

    def homography(self) -> np.ndarray:
        H = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], -self.R @ self.C])
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        return H

    def project(self, world: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project Nx3 world points -> (Nx2 uv, N depth, N valid)."""
        pts = np.asarray(world, float).reshape(-1, 3)
        cam = (self.R @ (pts - self.C).T).T
        depth = cam[:, 2]
        valid = depth > 1e-6
        uv = np.full((len(pts), 2), np.nan)
        uv[valid] = (self.K @ cam[valid].T).T[:, :2] / depth[valid, None]
        return uv, depth, valid

    def project_ground(self, xy: np.ndarray) -> np.ndarray:
        uv, _, valid = self.project(np.column_stack([np.asarray(xy, float).reshape(-1, 2),
                                                     np.zeros(len(np.asarray(xy).reshape(-1, 2)))]))
        return uv

    def project_distorted(self, world: np.ndarray, k1: float = 0.0):
        """Project Nx3 world points through a first-order radial distortion.

        Uses the same convention as the reference project and OpenCV:
        ``x_d = x_u * (1 + k1 * r^2)`` where ``(x_u, y_u)`` are pinhole
        normalised coordinates.  ``k1 == 0`` is exactly :meth:`project`.
        """
        pts = np.asarray(world, float).reshape(-1, 3)
        cam = (self.R @ (pts - self.C).T).T
        depth = cam[:, 2]
        valid = depth > 1e-6
        uv = np.full((len(pts), 2), np.nan)
        if not np.any(valid):
            return uv, depth, valid
        xn = cam[valid, 0] / depth[valid]
        yn = cam[valid, 1] / depth[valid]
        r2 = xn * xn + yn * yn
        scale = 1.0 + float(k1) * r2
        uv[valid] = np.stack([self.f * xn * scale + self.cx,
                              self.f * yn * scale + self.cy], axis=1)
        return uv, depth, valid

    def ground_homography_points(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        h = np.column_stack([xy, np.ones(len(xy))])
        p = (self.homography() @ h.T).T
        return p[:, :2] / p[:, 2:3]

    def in_front(self, world: np.ndarray) -> np.ndarray:
        _, depth, _ = self.project(world)
        return depth > 1e-6


def focal_from_homography(H: np.ndarray, width: int, height: int) -> Optional[float]:
    """Zhang square-pixel focal estimate for a ground homography."""
    H = np.asarray(H, float)
    cx, cy = width / 2.0, height / 2.0
    p0 = np.array([H[0, 0] - cx * H[2, 0], H[1, 0] - cy * H[2, 0]])
    p1 = np.array([H[0, 1] - cx * H[2, 1], H[1, 1] - cy * H[2, 1]])
    q0, q1 = float(H[2, 0]), float(H[2, 1])
    a = float(np.dot(p0, p1))
    b = float(q0 * q1)
    c = float(np.dot(p0, p0) - np.dot(p1, p1))
    d = float(q0 * q0 - q1 * q1)
    den = a * a + c * c
    if den < 1e-18:
        return None
    w = -(a * b + c * d) / den
    if w <= 0 or not math.isfinite(w):
        return None
    f = math.sqrt(1.0 / w)
    if not math.isfinite(f):
        return None
    return float(np.clip(f, 0.15 * max(width, height), 8.0 * max(width, height)))


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def camera_from_homography(H: np.ndarray, width: int, height: int,
                           focal: Optional[float] = None) -> Optional[PinholeCamera]:
    """Decompose a ground homography into a pinhole camera (principal point centred)."""
    H = np.asarray(H, float)
    if not np.all(np.isfinite(H)):
        return None
    if focal is None:
        focal = focal_from_homography(H, width, height)
        if focal is None:
            return None
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    A = np.linalg.inv(K) @ H
    scale = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]) + 1e-12)
    r1 = scale * A[:, 0]
    r2 = scale * A[:, 1]
    t = scale * A[:, 2]
    r3 = np.cross(r1, r2)
    R = _orthonormalize(np.column_stack([r1, r2, r3]))
    if np.linalg.det(R) < 0:
        H = -H
        A = np.linalg.inv(K) @ H
        scale = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]) + 1e-12)
        r1 = scale * A[:, 0]
        r2 = scale * A[:, 1]
        t = scale * A[:, 2]
        R = _orthonormalize(np.column_stack([r1, r2, np.cross(r1, r2)]))
    C = -R.T @ t
    if not np.all(np.isfinite(C)) or not np.all(np.isfinite(R)):
        return None
    # Keep the camera above the ground.  A homography sign flip is physically
    # the same view; reject the mirrored solution.
    if C[2] <= 0:
        R = R.copy()
        R[:, :2] *= -1.0
        R[:, 2] = np.cross(R[:, 0], R[:, 1])
        C = -R.T @ (-t)
    return PinholeCamera(float(focal), float(cx), float(cy), R, C)


def solve_camera_from_points(world_xy: np.ndarray, image_xy: np.ndarray,
                             width: int, height: int,
                             f_prior: Optional[float] = None,
                             camera_height: Optional[float] = None,
                             f_search: bool = True) -> Optional[PinholeCamera]:
    """Initial camera from >=4 manual ground point correspondences.

    The homography gives an exact ground mapping; the decomposition into
    (f, R, C) is the physically meaningful starting point for line refinement.
    When the point layout is near-degenerate the focal from H is unreliable, so
    a small 1-D focal search picks the decomposition with the lowest point
    reprojection error.
    """
    world = np.asarray(world_xy, float).reshape(-1, 2)
    image = np.asarray(image_xy, float).reshape(-1, 2)
    if len(world) < 4:
        raise ValueError("at least 4 point correspondences are required")
    H, _ = cv2.findHomography(world, image, 0)
    if H is None:
        return None
    f0 = focal_from_homography(H, width, height)
    if f_prior is not None and f0 is None:
        f0 = f_prior
    if f0 is None:
        f0 = 1.1 * width
    candidates: List[float] = []
    if f_search:
        for s in np.linspace(0.35, 3.0, 54):
            candidates.append(f0 * s)
        if f_prior:
            candidates += [f_prior * s for s in (0.7, 0.85, 1.0, 1.18, 1.4)]
    else:
        candidates = [f0]
    best = None
    best_err = float("inf")
    for f in candidates:
        f = float(np.clip(f, 0.12 * max(width, height), 8.0 * max(width, height)))
        cam = camera_from_homography(H, width, height, float(f))
        if cam is None:
            continue
        if camera_height is not None and abs(cam.height - camera_height) > 0.8 * camera_height + 8:
            continue
        uv = cam.ground_homography_points(world)
        err = float(np.median(np.linalg.norm(uv - image, axis=1)))
        # Prefer plausible focal lengths unless a prior says otherwise.
        f_ref = f_prior if f_prior else 1.2 * width
        err += 0.02 * abs(math.log(f / f_ref))
        if camera_height is not None:
            err += 0.05 * abs(cam.height - camera_height)
        if err < best_err and cam.height > 0:
            best_err = err
            best = cam
    return best


def camera_parameters(cam: PinholeCamera) -> np.ndarray:
    """[log f, rotvec(3), Cx, Cy, log Cz] -- a stable 7-vector for optimisation."""
    rvec = rotation_to_rvec(cam.R)
    return np.array([math.log(max(cam.f, 1e-3)), *rvec,
                     cam.C[0], cam.C[1], math.log(max(cam.C[2], 1e-3))], float)


def camera_from_parameters(p: np.ndarray, width: int, height: int,
                           R_prior: Optional[np.ndarray] = None,
                           C_prior: Optional[np.ndarray] = None) -> PinholeCamera:
    p = np.asarray(p, float)
    f = math.exp(float(np.clip(p[0], math.log(50.0), math.log(20.0 * max(width, height)))))
    R = rvec_to_rotation(p[1:4])
    if R_prior is not None:
        # p[1:4] is a local rotation applied in camera frame.
        R = R @ R_prior
    C = np.array([p[4], p[5], math.exp(float(np.clip(p[6], math.log(0.5), math.log(200.0))))])
    if C_prior is not None:
        C = C_prior + np.array([p[4], p[5], p[6]])
    return PinholeCamera(f, width / 2.0, height / 2.0, R, C)


def camera_from_local_parameters(f: float, local_rvec: np.ndarray, C: np.ndarray,
                                 R_prior: np.ndarray, width: int, height: int) -> PinholeCamera:
    R = rvec_to_rotation(local_rvec) @ R_prior
    return PinholeCamera(float(f), width / 2.0, height / 2.0, R, np.asarray(C, float).copy())


def project_wireframe(cam: PinholeCamera, template: PitchTemplate,
                      step: float = 0.5) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Project template polylines; returns (name, uv, valid) per line."""
    out = []
    for name, pts in template.polylines():
        world = np.column_stack([pts, np.zeros(len(pts))])
        uv, _, valid = cam.project(world)
        out.append((name, uv, valid))
    return out


# ---------------------------------------------------------------------------
# Image-space drawing of ground-plane lines
# ---------------------------------------------------------------------------

def clip_segment(a, b, width: int, height: int):
    """Liang-Barsky: clip the pixel segment ``a``->``b`` to the image rectangle.

    Returns ``(a, b)`` clipped coordinates, or ``None`` when the segment lies
    completely outside.  Works on float coordinates, so it also tames points
    that project to huge values near the horizon.
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    if not all(math.isfinite(v) for v in (ax, ay, bx, by)):
        return None
    dx, dy = bx - ax, by - ay
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return None
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax), (dx, (width - 1) - ax),
                 (-dy, ay), (dy, (height - 1) - ay)):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    return (ax + t0 * dx, ay + t0 * dy), (ax + t1 * dx, ay + t1 * dy)


def project_polyline_segments(H: np.ndarray, world_xy: np.ndarray,
                              width: int, height: int) -> List[Tuple[Tuple[float, float],
                                                                    Tuple[float, float]]]:
    """Project a ground-plane polyline to pixel segments ready to draw.

    Points with homogeneous ``w <= 0`` are *behind* the camera plane: dividing
    by w would mirror them across the horizon, which is how a badly
    extrapolated anchor used to draw the halfway line up into the stands.
    Such points are dropped, and so is any segment that touches one.  Every
    remaining segment is clipped to the image rectangle, so extreme
    near-horizon projections cannot produce overflowing drawing coordinates.

    Returns a list of ``((x0, y0), (x1, y1))`` float segments.
    """
    pts = np.asarray(world_xy, float).reshape(-1, 2)
    if len(pts) < 2:
        return []
    hom = np.column_stack([pts, np.ones(len(pts))])
    p = (np.asarray(H, float) @ hom.T).T
    w = p[:, 2]
    valid = np.isfinite(p).all(axis=1) & (w > 1e-9)
    uv = np.full((len(pts), 2), np.nan)
    uv[valid] = p[valid, :2] / w[valid, None]
    out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for i in range(len(pts) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        seg = clip_segment(uv[i], uv[i + 1], width, height)
        if seg is not None:
            out.append(seg)
    return out
