"""Domain-independent white-pitch-line evidence extraction.

The broadcast-domain segmentation networks used by most public pipelines fall
apart on tactical / youth footage: different turf, faint far touchlines and
stands whose seats are green.  This module uses no learned model.  A bank of
oriented bright-bar matched filters responds to thin, high-contrast paint on
turf; the response is trained by the geometry (pitch template) later, not by a
dataset.  It is deliberately conservative: a missing far line simply supplies
no evidence, instead of supplying a wrong class label.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LineEvidence:
    image: np.ndarray          # BGR working image
    response: np.ndarray       # float32, matched-filter line strength
    field: np.ndarray          # uint8 field mask (255 = grass)
    gray: np.ndarray
    scale_x: float             # response pixel -> original pixel
    scale_y: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.response.shape


def _line_kernel(length: int, thickness: int, gap: int, angle_deg: float) -> np.ndarray:
    """Zero-mean oriented bright-bar matched filter."""
    size = int(length + 2 * (thickness + gap)) | 1
    k = np.zeros((size, size), np.float32)
    c = size // 2
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    n_t = max(2, length // 2 + 1)
    for t in np.linspace(-length / 2, length / 2, n_t):
        for u in np.linspace(-(thickness - 1) / 2, (thickness - 1) / 2, max(1, thickness)):
            x = int(round(c + t * ca - u * sa))
            y = int(round(c + t * sa + u * ca))
            if 0 <= x < size and 0 <= y < size:
                k[y, x] += 1.0
    for t in np.linspace(-length / 2, length / 2, n_t):
        for sgn in (-1, 1):
            u = sgn * (thickness / 2 + gap)
            x = int(round(c + t * ca - u * sa))
            y = int(round(c + t * sa + u * ca))
            if 0 <= x < size and 0 <= y < size and k[y, x] >= 0:
                k[y, x] = -1.0
    pos = float(k[k > 0].sum())
    neg = float(-k[k < 0].sum())
    if pos > 0:
        k[k > 0] /= pos
    if neg > 0:
        k[k < 0] /= neg
    return k


def _field_mask(image: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    exg = 2.0 * g.astype(np.float32) - r.astype(np.float32) - b.astype(np.float32)
    mask = ((exg > 8) & (hsv[:, :, 2] > 25)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == idx).astype(np.uint8) * 255
    # Fill holes so painted lines (low saturation) stay inside the field mask.
    flood = mask.copy()
    h, w = flood.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def compute_line_evidence(image: np.ndarray, work_size: int = 960,
                          orientations: int = 12,
                          scales: Optional[Tuple[Tuple[int, int], ...]] = None,
                          field_mask: Optional[np.ndarray] = None) -> LineEvidence:
    """Compute white-line evidence at a working resolution.

    ``work_size`` is the long side.  The line filters are evaluated on a
    slightly downsampled copy for speed and then reused at working resolution.
    """
    if image is None or image.ndim != 3:
        raise ValueError("image must be HxWx3 BGR")
    h, w = image.shape[:2]
    if work_size and max(h, w) > work_size:
        s = work_size / float(max(h, w))
        work = cv2.resize(image, (int(round(w * s)), int(round(h * s))),
                          interpolation=cv2.INTER_AREA)
    else:
        work = image
    H, W = work.shape[:2]
    b, g, r = cv2.split(work)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float32)
    minc = np.minimum(np.minimum(b, g), r).astype(np.float32)
    bg_g = cv2.GaussianBlur(gray, (0, 0), 11)
    bg_m = cv2.GaussianBlur(minc, (0, 0), 11)
    dg = gray - bg_g
    dm = minc - bg_m

    if scales is None:
        scales = ((11, 1), (19, 1), (19, 3), (29, 3))
    resp = np.zeros_like(gray)
    for length, thick in scales:
        # Small filters are cheap; evaluate all of them on the work image.
        for ai in range(orientations):
            ang = ai * 180.0 / orientations
            k = _line_kernel(length, thick, 2, ang)
            rg = cv2.filter2D(dg, -1, k, borderType=cv2.BORDER_REPLICATE)
            rm = cv2.filter2D(dm, -1, k, borderType=cv2.BORDER_REPLICATE)
            resp = np.maximum(resp, 0.62 * rg + 0.58 * rm)
    resp = cv2.GaussianBlur(resp, (0, 0), 0.8)
    if field_mask is None:
        field = _field_mask(work)
    else:
        field = cv2.resize(field_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return LineEvidence(work, resp, field, gray.astype(np.uint8),
                        W / float(w), H / float(h))


def _roi_mask(shape: Tuple[int, int], polygon: Optional[np.ndarray],
              margin: int = 25) -> np.ndarray:
    H, W = shape
    if polygon is None:
        return np.ones((H, W), np.uint8) * 255
    poly = np.asarray(polygon, np.float32).reshape(-1, 2)
    if len(poly) < 3:
        return np.ones((H, W), np.uint8) * 255
    roi = np.zeros((H, W), np.uint8)
    cv2.fillPoly(roi, [np.round(poly).astype(np.int32)], 255)
    if margin:
        roi = cv2.dilate(roi, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    return roi


def line_mask(evidence: LineEvidence, polygon: Optional[np.ndarray] = None,
              quantile: float = 0.90, min_threshold: float = 4.0,
              players: Optional[np.ndarray] = None) -> np.ndarray:
    """Binary line mask restricted to a pitch polygon (image coords).

    ``polygon`` is the projected ground footprint of the pitch (or a margin
    around the visible pitch); it keeps green stands and advertising edges out.
    ``players`` is an optional list of ``(x1, y1, x2, y2)`` boxes to blank.
    """
    resp = evidence.response
    field = evidence.field
    roi = _roi_mask(resp.shape, polygon, margin=25)
    valid = (field > 0) & (roi > 0)
    values = resp[valid]
    if values.size == 0:
        return np.zeros_like(resp, np.uint8)
    thr = max(float(np.quantile(values, quantile)), float(min_threshold))
    mask = ((resp > thr) & valid).astype(np.uint8)
    if players is not None:
        for box in players:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(mask, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), 0, -1)
    # Remove isolated pixels but keep 1 px-wide far lines.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask * 255


def line_distance(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(255 - (mask > 0).astype(np.uint8) * 255,
                                 cv2.DIST_L2, 5)


def pitch_polygon_image(cam, template, margin_m: float = 3.0) -> Optional[np.ndarray]:
    """Project the pitch outer rectangle to an image polygon (clipped to w>0).

    A camera near the touchline can have a corner behind the camera plane.  We
    clip the *world* polygon against the ground-plane camera depth before
    projecting; the returned polygon may still extend far outside the image.
    """
    from .geometry import PinholeCamera
    hx, hy = template.hx + margin_m, template.hy + margin_m
    corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], float)
    R, C = cam.R, cam.C
    # Ground point depth: z_cam = r3 . (X - C) = r3_xy . (xy - C_xy) - r3_z*Cz
    a = R[2, 0] * (corners[:, 0] - C[0]) + R[2, 1] * (corners[:, 1] - C[1]) - R[2, 2] * C[2]
    poly = corners[a > 0.15]
    if len(poly) < 3:
        return None
    uv = cam.ground_homography_points(poly)
    if not np.all(np.isfinite(uv)):
        return None
    return uv.astype(np.float32)


def pitch_polygon_h(H: np.ndarray, template, margin_m: float = 3.0) -> Optional[np.ndarray]:
    """Project the pitch outer rectangle through a ground homography.

    This is the homography-native version used by the per-frame tracker; the
    camera-based ``pitch_polygon_image`` is for anchor solving.
    """
    hx, hy = template.hx + margin_m, template.hy + margin_m
    corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], float)
    p = (np.asarray(H, float) @ np.column_stack([corners, np.ones(4)]).T).T
    valid = np.isfinite(p).all(axis=1) & (np.abs(p[:, 2]) > 1e-9)
    if valid.sum() < 3:
        return None
    uv = p[valid, :2] / p[valid, 2:3]
    return uv.astype(np.float32)


def polygon_to_work(polygon: np.ndarray, evidence: LineEvidence) -> np.ndarray:
    p = np.asarray(polygon, np.float64).copy()
    p[:, 0] *= evidence.scale_x
    p[:, 1] *= evidence.scale_y
    return p


def paint_like_segments(frame: np.ndarray, segments: np.ndarray,
                   min_contrast: float = 6.0) -> np.ndarray:
    """Keep LSD segments that actually look like white paint on turf.

    LSD is happy to latch onto mowing-stripe boundaries, player limbs and
    advertising edges.  A white line is a *thin bright bar*: brighter than the
    turf on both sides, reasonably bright, and not strongly coloured.  This
    cheap photometric check removes most of the false segments before they are
    back-projected into the BEV, which is what makes the observed-line layer
    readable rather than a pile of diagonals.
    """
    if segments is None or not len(segments):
        return np.zeros((0, 4), np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(frame)
    minc = np.minimum(np.minimum(b, g), r).astype(np.float32)
    sat = hsv[:, :, 1]
    h, w = gray.shape
    keep = []
    for x1, y1, x2, y2 in np.asarray(segments).reshape(-1, 4):
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 22:
            continue
        nx, ny = -(y2 - y1) / length, (x2 - x1) / length
        contrasts = []
        brights = []
        sats = []
        for t in np.linspace(0.08, 0.92, max(4, int(length / 12))):
            cx = x1 + t * (x2 - x1)
            cy = y1 + t * (y2 - y1)
            ix, iy = int(round(cx)), int(round(cy))
            if not (1 <= ix < w - 1 and 1 <= iy < h - 1):
                continue
            best = -1.0
            for d in (2, 4, 7, 11):
                xp, yp = int(round(cx + d * nx)), int(round(cy + d * ny))
                xm, ym = int(round(cx - d * nx)), int(round(cy - d * ny))
                if not (0 <= xp < w and 0 <= yp < h and 0 <= xm < w and 0 <= ym < h):
                    continue
                c = float(minc[iy, ix] - 0.5 * (minc[yp, xp] + minc[ym, xm]))
                best = max(best, c)
            if best > -1:
                contrasts.append(best)
                brights.append(float(minc[iy, ix]))
                sats.append(float(sat[iy, ix]))
        if len(contrasts) < 3:
            continue
        if (np.median(contrasts) >= min_contrast
                and np.median(brights) >= 65
                and (np.median(sats) <= 150 or np.median(brights) >= 135)):
            keep.append((x1, y1, x2, y2))
    return np.asarray(keep, np.float32) if keep else np.zeros((0, 4), np.float32)
