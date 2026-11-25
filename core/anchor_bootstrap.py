"""Manual calibration anchors: pitchcal JSON -> BroadTrack's camera.

The annotator in :mod:`core.pitchcal` writes anchors in its own world frame:
metres, origin at the pitch centre, x along the length, y across the width,
**z up**.  BroadTrack (and the rest of this project) uses the same x/y but
**z down** -- the camera sits at a negative z.  This module is the single
place that converts between them, so the z flip never leaks into ``main.py``.

An anchor is a ground-plane homography ``H`` for one frame.  BroadTrack wants
``K, R, t``, so the homography is decomposed into a pinhole camera with the
principal point at the image centre and square pixels -- exactly the camera
model BroadTrack solves.  The decomposition is only used to *start* the
temporal tracker; from then on the per-frame line solve refines the camera.

Example
-------
    bootstrap = load_manual_bootstrap("config/anchors/venue.json", 960, 540)
    anchor = bootstrap.at_or_before(frame_idx)
    calib.set_camera(anchor.K, anchor.R, anchor.t, anchor.k1,
                     frame_width=anchor.width, frame_height=anchor.height)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from core.log import get_logger
from core.pitchcal.anchors import Anchor, AnchorFile, load_anchor_file
from core.pitchcal.geometry import (
    PinholeCamera,
    _orthonormalize,
    camera_from_homography,
)
from core.projection_utils import pixel_to_ground, project_3d_to_pixel

logger = get_logger(__name__)

# A ground homography determines a camera only up to the sign of P.  pitchcal's
# ``camera_from_homography`` picks the solution whose camera is above the *z-up*
# ground plane; BroadTrack's frame has z pointing down, so it needs the other
# solution.  The two differ by R -> R @ diag(-1, -1, 1) and C -> diag(1, 1, -1) C.
# This was checked against a real BroadTrack solve (exact match) and against
# synthetic cameras (reprojection exact, camera above the ground).  The ground
# homography itself is identical for both solutions.
_ROTATION_SIGN = np.diag([-1.0, -1.0, 1.0])
_POSITION_FLIP_Z = np.diag([1.0, 1.0, -1.0])


@dataclass(frozen=True)
class AnchorCamera:
    """One anchor, already converted to BroadTrack's frame and resolution."""

    frame: int
    width: int
    height: int
    K: np.ndarray
    R: np.ndarray          # world -> camera, z down
    t: np.ndarray          # camera centre, z down (camera sits at t[2] < 0)
    k1: float = 0.0
    source: str = "manual"
    score: float = 0.0


@dataclass
class ManualBootstrap:
    """All anchors in a pitchcal anchor file, for one working resolution."""

    path: str
    pitch: tuple[float, float]
    width: int
    height: int
    anchors: list[AnchorCamera] = field(default_factory=list)

    def at_or_before(self, frame: int) -> Optional[AnchorCamera]:
        """The latest anchor that is not in the future.

        Applying an anchor to an earlier frame would be wrong: it describes
        the camera at one instant, and a moving PTZ camera changes by the
        frame.  ``None`` means the tracker must bootstrap itself.
        """
        candidates = [a for a in self.anchors if a.frame <= frame]
        return max(candidates, key=lambda a: a.frame) if candidates else None

    def next_after(self, frame: int) -> Optional[AnchorCamera]:
        """The first anchor strictly after ``frame``, if any."""
        candidates = [a for a in self.anchors if a.frame > frame]
        return min(candidates, key=lambda a: a.frame) if candidates else None

    def exact(self, frame: int) -> Optional[AnchorCamera]:
        for anchor in self.anchors:
            if anchor.frame == frame:
                return anchor
        return None


def _reproduces(homography: np.ndarray, reference: np.ndarray, tol: float = 1e-4) -> bool:
    """Whether two ground homographies are the same mapping (up to scale)."""
    a = np.asarray(homography, float) / homography[2, 2]
    b = np.asarray(reference, float) / reference[2, 2]
    return bool(np.allclose(a, b, atol=tol, rtol=tol))


def _constraints(homography: np.ndarray, focal: float, cx: float, cy: float) -> np.ndarray:
    """The two square-pixel/centred-camera constraints for H = K [r1 r2 t].

    ``K^-1 H`` must have orthogonal, equal-length first two columns.  Both
    residuals are dimensionless so one tolerance works at every resolution.
    """
    K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    A = np.linalg.inv(K) @ homography
    a, b = A[:, 0], A[:, 1]
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return np.array([float(a @ b) / (na * nb),
                     (na * na - nb * nb) / (na * na + nb * nb)])


def _camera_from_k(homography: np.ndarray, focal: float, cx: float, cy: float) -> Optional[PinholeCamera]:
    """Build (f, cx, cy, R, C) from H once K is known; ``None`` if degenerate."""
    K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    A = np.linalg.inv(K) @ homography
    n0, n1 = float(np.linalg.norm(A[:, 0])), float(np.linalg.norm(A[:, 1]))
    if n0 <= 1e-12 or n1 <= 1e-12 or not np.isfinite(n0 + n1):
        return None
    scale = 2.0 / (n0 + n1)
    r1, r2, t = scale * A[:, 0], scale * A[:, 1], scale * A[:, 2]
    R = _orthonormalize(np.column_stack([r1, r2, np.cross(r1, r2)]))
    C = -R.T @ t
    if not np.isfinite(R).all() or not np.isfinite(C).all():
        return None
    # Two poses give the same H.  Pick the one above the z-up ground, which
    # is the convention camera_from_homography() returns and the one the
    # z-down conversion below expects.
    if C[2] <= 0.0:
        R = R @ _ROTATION_SIGN
        C = _POSITION_FLIP_Z @ C
    return PinholeCamera(float(focal), float(cx), float(cy), R, C)


def _decompose_homography(homography: np.ndarray, width: int, height: int) -> Optional[PinholeCamera]:
    """Decompose a ground homography into a pinhole camera.

    The centred-principal-point decomposition is exact for a homography that
    came from a physical camera (BroadTrack's own anchors are).  A *manually*
    fitted H -- line/paint fitting with only a few clicks -- need not satisfy
    the square-pixel/centred constraints.  Forcing the centred solution there
    shifts the whole projection by tens of pixels, which is what makes the
    first anchored frame disagree with the annotator.

    The constraints are two equations in three unknowns (f, cx, cy), so an
    exact solution generically exists as a 1-D family.  We scan the focal and
    keep the member whose principal point is closest to the image centre,
    preferring a focal near the centred estimate.  The resulting camera
    reproduces H to machine precision, so the ground projection (BEV, players,
    ball) matches the anchor exactly.  The principal-point offset is a real
    parameter of BroadTrack's camera after our extension.
    """
    H = np.asarray(homography, float)
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return None
    H = H / H[2, 2]

    centred = camera_from_homography(H, width, height)
    if centred is not None and _reproduces(centred.homography(), H):
        logger.trace("anchor decomposition: centred principal point (exact)")
        return centred

    focal_hint = centred.f if centred is not None else 1.2 * width
    cx0, cy0 = width / 2.0, height / 2.0
    best = None  # (score, focal, cx, cy)
    for focal in np.geomspace(0.2 * max(width, height), 4.0 * max(width, height), 56):
        for start in ((cx0, cy0), (0.25 * width, 0.5 * height), (0.75 * width, 0.5 * height)):
            result = least_squares(
                lambda p: _constraints(H, focal, p[0], p[1]),
                start,
                bounds=([-width, -height], [2.0 * width, 2.0 * height]),
                xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=300,
            )
            if not np.isfinite(result.fun).all() or np.linalg.norm(result.fun) > 1e-9:
                continue
            cx, cy = float(result.x[0]), float(result.x[1])
            camera = _camera_from_k(H, float(focal), cx, cy)
            if camera is None or not _reproduces(camera.homography(), H):
                continue
            score = ((cx - cx0) ** 2 + (cy - cy0) ** 2
                     + (0.02 * width) ** 2 * np.log(float(focal) / focal_hint) ** 2)
            if best is None or score < best[0]:
                best = (score, float(focal), cx, cy)

    if best is None:
        logger.warning(
            "anchor decomposition: no exact off-centre solution; using the "
            "centred camera (the anchor frame may be offset from the annotation)"
        )
        return centred

    _, focal, cx, cy = best
    camera = _camera_from_k(H, focal, cx, cy)
    logger.debug(
        "anchor decomposition: off-centre principal point (%.1f, %.1f) in "
        "%dx%d, f=%.1f (centred estimate %.1f); H reproduced exactly",
        cx, cy, width, height, focal, focal_hint,
    )
    return camera


def _ray_valid_camera(homography: np.ndarray, K: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Pick the H sign whose camera actually looks at the ground.

    A ground homography has two camera solutions (P and -P).  Both project the
    ground correctly, but one has the pitch behind the camera: the pinhole
    ray used by ``pixel_to_ground``/BroadTrack then misses the plane.  Prefer
    the solution above the ground in this project's z-down frame.
    """
    A = np.linalg.inv(K) @ homography
    n0, n1 = float(np.linalg.norm(A[:, 0])), float(np.linalg.norm(A[:, 1]))
    if n0 <= 1e-12 or n1 <= 1e-12:
        return None
    scale = 2.0 / (n0 + n1)
    r1, r2, th = scale * A[:, 0], scale * A[:, 1], scale * A[:, 2]

    candidates = []
    for sign in (1.0, -1.0):
        R = _orthonormalize(np.column_stack([
            sign * r1, sign * r2, np.cross(sign * r1, sign * r2),
        ]))
        C = -R.T @ (sign * th)
        if not np.isfinite(R).all() or not np.isfinite(C).all():
            continue
        pixel = project_3d_to_pixel(np.zeros(3), K, R, C)
        ground = (pixel_to_ground(float(pixel[0]), float(pixel[1]), K, R, C)
                  if np.isfinite(pixel).all() else None)
        candidates.append((R, C, ground is not None))

    ray_valid = [c for c in candidates if c[2]]
    above_ground = [c for c in ray_valid if c[1][2] < 0.0]
    chosen = (above_ground or ray_valid)
    if not chosen:
        return None
    return chosen[0][0], chosen[0][1]


def _focal_at_principal(homography: np.ndarray, cx: float, cy: float) -> Optional[float]:
    """Best square-pixel focal for H when the principal point is fixed.

    Same Zhang least-squares as ``geometry.focal_from_homography`` but with an
    arbitrary principal point.  Used every frame on the LK-tracked homography,
    so it is a closed form rather than an iterative solve.
    """
    p0 = np.array([homography[0, 0] - cx * homography[2, 0],
                   homography[1, 0] - cy * homography[2, 0]])
    p1 = np.array([homography[0, 1] - cx * homography[2, 1],
                   homography[1, 1] - cy * homography[2, 1]])
    q0, q1 = float(homography[2, 0]), float(homography[2, 1])
    a = float(p0 @ p1)
    b = float(q0 * q1)
    c = float(p0 @ p0 - p1 @ p1)
    d = float(q0 * q0 - q1 * q1)
    denominator = a * a + c * c
    if denominator < 1e-18:
        return None
    w = -(a * b + c * d) / denominator  # w = 1/f^2
    if w <= 0.0 or not np.isfinite(w):
        return None
    return float(1.0 / np.sqrt(w))


def homography_from_camera(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World->image ground homography for a camera in this project's frame."""
    H = np.asarray(K, float) @ np.column_stack([
        np.asarray(R, float)[:, 0],
        np.asarray(R, float)[:, 1],
        -np.asarray(R, float) @ np.asarray(t, float),
    ])
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def camera_from_tracked_homography(
    homography: np.ndarray,
    width: int,
    height: int,
    principal: tuple[float, float],
    focal_hint: float,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Convert one tracked ground homography to ``(K, R, t)`` in this frame.

    The principal point is fixed by the anchor that started the track; only
    the focal follows zoom.  This is the per-frame counterpart of
    ``_decompose_homography`` and must stay cheap: it is called for every
    frame by the LK tracker.  Returns ``None`` when the homography is
    degenerate, in which case the caller should keep the last camera.
    """
    H = np.asarray(homography, float)
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return None
    H = H / H[2, 2]
    cx, cy = float(principal[0]), float(principal[1])
    focal = _focal_at_principal(H, cx, cy)
    limit = max(width, height)
    if focal is None or not (0.1 * limit <= focal <= 8.0 * limit):
        focal = float(focal_hint)
    K = np.array([[float(focal), 0.0, cx],
                  [0.0, float(focal), cy],
                  [0.0, 0.0, 1.0]])
    chosen = _ray_valid_camera(H, K)
    if chosen is None:
        return None
    R, t = chosen
    if not np.isfinite(R).all() or not np.isfinite(t).all() or t[2] >= 0.0:
        return None
    return K, R, t


def _camera_from_anchor(anchor: Anchor, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Decompose one pitchcal anchor at ``width`` x ``height`` into K, R, t.

    Returns the camera in this project's frame (z down).  Raises ``ValueError``
    for an anchor that cannot produce a physical camera -- a silent zero
    camera would be far worse than a clear failure at start-up.
    """
    if anchor.H is not None:
        homography = anchor.homography(width, height)
        k1 = 0.0
    elif anchor.f > 0.0 and np.isfinite(anchor.C).all():
        # A camera-form anchor still defines a ground homography.  Going
        # through H (rather than using its R directly) keeps one conversion
        # path: pitchcal's camera-form R may be stored with det = -1 because it
        # was itself imported from a z-down camera.
        homography = anchor.homography(width, height)
        k1 = float(anchor.k1)
    else:
        raise ValueError(
            f"anchor at frame {anchor.frame}: neither a homography nor a "
            "camera (f, R, C) is stored"
        )

    camera = _decompose_homography(homography, width, height)
    if camera is None:
        raise ValueError(
            f"anchor at frame {anchor.frame}: the homography cannot be "
            "decomposed into a pinhole camera (degenerate focal estimate); "
            "re-mark the anchor with more spread-out landmarks"
        )

    _validate_camera(camera, anchor.frame)

    K = camera.K
    chosen = _ray_valid_camera(homography, K)
    if chosen is None:
        raise ValueError(
            f"anchor at frame {anchor.frame}: no camera solution looks at the "
            "ground; the anchor is mirrored or degenerate"
        )
    R, t = chosen
    _validate_rotated(R, t, anchor.frame)
    return K, R, t, k1


def _validate_camera(camera: PinholeCamera, frame: int) -> None:
    if not np.isfinite(camera.f) or camera.f <= 0.0:
        raise ValueError(f"anchor at frame {frame}: non-positive focal length")
    if not np.isfinite(camera.R).all() or not np.isfinite(camera.C).all():
        raise ValueError(f"anchor at frame {frame}: non-finite camera pose")


def _validate_rotated(R: np.ndarray, t: np.ndarray, frame: int) -> None:
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError(f"anchor at frame {frame}: non-finite converted camera")
    if abs(float(np.linalg.det(R)) - 1.0) > 1e-3:
        raise ValueError(f"anchor at frame {frame}: rotation is not orthonormal")
    # z is negative above the ground.
    if float(t[2]) >= 0.0:
        raise ValueError(
            f"anchor at frame {frame}: camera centre z={t[2]:.2f} is not above "
            "the ground; the anchor probably has the pitch flipped"
        )


def load_manual_bootstrap(
    path: str | Path,
    width: int,
    height: int,
    pitch: Optional[tuple[float, float]] = None,
) -> ManualBootstrap:
    """Read a pitchcal anchor file and convert every anchor to ``width`` x ``height``.

    Args:
        path: anchor JSON written by ``python -m core.pitchcal.cli annotate``.
        width, height: the analyzer's processing resolution.  The anchor's
            native frame is rescaled to this; the aspect ratio must match.
        pitch: override the pitch size stored in the file.

    Raises:
        ValueError: for an unreadable/degenerate anchor.  Failing at start-up
            is better than silently calibrating against a wrong camera.
    """
    path = Path(path)
    anchor_file: AnchorFile = load_anchor_file(path)
    if not anchor_file.anchors:
        raise ValueError(f"{path}: no anchors in the file")

    pitch_size = tuple(pitch) if pitch is not None else tuple(anchor_file.pitch)
    converted = []
    for anchor in anchor_file.anchors:
        aspect = anchor.width / float(anchor.height)
        if abs(aspect - width / float(height)) > 1e-3:
            raise ValueError(
                f"{path}: anchor at frame {anchor.frame} is {anchor.width}x{anchor.height} "
                f"(aspect {aspect:.4f}) but the processing frame is {width}x{height} "
                f"(aspect {width / height:.4f})"
            )
        K, R, t, k1 = _camera_from_anchor(anchor, width, height)
        converted.append(AnchorCamera(
            frame=int(anchor.frame),
            width=int(width),
            height=int(height),
            K=K,
            R=R,
            t=t,
            k1=k1,
            source=str(anchor.source),
            score=float(anchor.score),
        ))
        logger.debug(
            "anchor frame %d: f=%.1f C=(%.1f, %.1f, %.1f) source=%s",
            anchor.frame, K[0, 0], t[0], t[1], t[2], anchor.source,
        )
    converted.sort(key=lambda a: a.frame)
    return ManualBootstrap(
        path=str(path),
        pitch=(float(pitch_size[0]), float(pitch_size[1])),
        width=int(width),
        height=int(height),
        anchors=converted,
    )
