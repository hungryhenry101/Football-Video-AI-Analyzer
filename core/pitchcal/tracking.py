"""Frame-to-frame ground-plane tracking with Lucas-Kanade features.

A camera looking at a planar pitch induces a homography between consecutive
frames.  Tracking static features inside the projected pitch polygon and
composing their RANSAC homography is both much faster and much more robust than
running a pitch detector on every frame.  The absolute anchor/broadcast solve
only has to be right once; this carries it through pans, tilts and zooms.

Features on players are moving outliers.  They are rejected by RANSAC and,
when player boxes are supplied (the YOLO detector runs on a sparse schedule),
removed from the mask outright.  The LK layer stays on the ground plane, so
features on stands/advertising are excluded by the projected pitch polygon;
that also keeps the tracker usable when the camera pans away from the centre.

On top of LK, a SIFT keyframe (~0.5 s old) measures accumulated drift directly
from static off-pitch structures (stands, roofs, advertising boards).  Those
are the only SIFT-rich support in a wide tactical shot: turf has almost no
blobs and the strong in-pitch ones sit on players.  Off-pitch structures are
not on the ground plane, so their image homography only equals the ground one
while the camera pans/tilts/zooms about a fixed optical centre.  Per-interval
and cumulative agreement checks bound how far the correction may move the
ground homography, and a screen-fixed score bug is rejected as inconsistent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .geometry import PinholeCamera, camera_from_homography, look_at_rotation
from .linemask import LineEvidence, pitch_polygon_h


@dataclass
class TrackerConfig:
    max_corners: int = 900
    # qualityLevel is relative to the strongest corner in the image.  Wide,
    # low-contrast tactical views have weak grass/paint corners, and 0.005 left
    # only ~35 of them: RANSAC then fits noise and the projection drifts even
    # with a static camera.  0.001 keeps ~5x more features and holds a static
    # view to a couple of pixels over several seconds.
    quality_level: float = 0.001
    min_distance: int = 8
    block_size: int = 7
    lk_window: int = 21
    lk_levels: int = 3
    fb_error_px: float = 1.5
    ransac_px: float = 2.0
    min_inliers: int = 30
    min_ratio: float = 0.30
    max_reproj_px: float = 1.5
    redetect_below: int = 120
    reinit_after_lost: int = 4
    max_frame_delta: float = 18.0  # cut detector threshold (mean abs gray diff)
    min_hist_corr: float = 0.55
    polygon_margin_px: int = 30
    green_features: bool = True
    green_min_exg: float = 6.0
    sift_features: int = 2000
    sift_ratio: float = 0.75
    sift_ransac_px: float = 3.0
    sift_min_matches: int = 25
    sift_keyframe_every: int = 30
    # A stand-based keyframe solution may only differ from the LK trajectory by
    # accumulated drift.  A much larger disagreement means a different depth
    # (camera translation) or a screen-fixed graphic (score bug) was matched;
    # reject the correction instead of jumping the ground homography.
    sift_max_disagree_px: float = 18.0
    # Stands/boards are not on the ground plane.  Let SIFT correct drift, but
    # never let it carry the ground homography far away from the pure-LK
    # trajectory: a depth-dependent bias otherwise accumulates without bound.
    sift_max_trajectory_px: float = 60.0


@dataclass
class TrackState:
    homography: np.ndarray
    valid: bool
    inliers: int = 0
    ratio: float = 0.0
    reproj_px: float = 0.0
    confidence: float = 0.0
    lost_frames: int = 0
    cut: bool = False
    features: int = 0


class GroundTracker:
    def __init__(self, camera: Optional[PinholeCamera] = None, width: int = 960,
                 height: int = 540, template=None,
                 config: Optional[TrackerConfig] = None,
                 homography: Optional[np.ndarray] = None):
        from .geometry import PitchTemplate
        if camera is None and homography is None:
            raise ValueError("GroundTracker needs a camera or a homography")
        self.camera = camera
        self.template = template or PitchTemplate()
        self.width = int(width)
        self.height = int(height)
        self.cfg = config or TrackerConfig()
        self.H = (np.asarray(homography, float) if homography is not None
                  else camera.homography())
        # Pure-LK trajectory, maintained alongside the SIFT-corrected H.  It is
        # the reference that bounds how far a stand-based correction may move
        # the ground homography.
        self._H_lk = self.H.copy()
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._prev_hist = None
        self._last_cut = False
        self.lost_frames = 0
        self.last_state = TrackState(self.H.copy(), True, confidence=1.0)
        self._sift_kf = None
        self._sift_kf_frame = 0
        self.frame_count = -1
        self._sift = cv2.SIFT_create(nfeatures=self.cfg.sift_features)
        self._bf = cv2.BFMatcher(cv2.NORM_L2)

    # ------------------------------------------------------------------ public
    def rebase(self, homography: np.ndarray) -> None:
        """Replace the absolute homography and the LK shadow together.

        Used by the fixed-centre stabiliser, which rewrites H every frame.
        """
        self.H = np.asarray(homography, float)
        self._H_lk = self.H.copy()

    def reset(self, camera: Optional[PinholeCamera] = None,
              homography: Optional[np.ndarray] = None) -> None:
        if camera is not None:
            self.camera = camera
            self.H = camera.homography()
        elif homography is not None:
            self.H = np.asarray(homography, float)
        self._H_lk = self.H.copy()
        self._prev_gray = None
        self._prev_pts = None
        self._prev_hist = None
        self._sift_kf = None
        self._sift_kf_frame = 0
        self.frame_count = -1
        self.lost_frames = 0
        self.last_state = TrackState(self.H.copy(), True, confidence=1.0)

    def update(self, frame: np.ndarray,
               boxes: Optional[Sequence[Sequence[float]]] = None,
               line_hit: Optional[float] = None) -> TrackState:
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cut = self._is_cut(gray)
        if cut:
            self._prev_gray = gray
            self._prev_pts = None
            self._prev_hist = self._histogram(gray)
            self._sift_kf = None
            self.lost_frames += 1
            state = TrackState(self.H.copy(), False, confidence=0.0,
                               lost_frames=self.lost_frames, cut=True)
            self.last_state = state
            return state
        mask = self._feature_mask(frame, boxes)
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_pts = self._detect(gray, mask)
            self._prev_hist = self._histogram(gray)
            self._init_sift_keyframe(frame, gray, boxes)
            # No frame-to-frame measurement happened yet (clip start or the
            # first frame after reset); do not report perfect confidence.
            state = TrackState(self.H.copy(), True, features=_count(self._prev_pts),
                               confidence=0.0)
            self.last_state = state
            return state

        pts_prev = self._prev_pts
        if pts_prev is None or len(pts_prev) < self.cfg.redetect_below:
            pts_prev = self._detect(self._prev_gray, mask)
        if pts_prev is None or len(pts_prev) < 12:
            self.lost_frames += 1
            self._prev_gray = gray
            self._prev_pts = self._detect(gray, mask)
            self._prev_hist = self._histogram(gray)
            state = TrackState(self.H.copy(), False, confidence=0.0,
                               lost_frames=self.lost_frames)
            self.last_state = state
            return state

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, pts_prev, None,
            winSize=(self.cfg.lk_window, self.cfg.lk_window),
            maxLevel=self.cfg.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, p1, None,
            winSize=(self.cfg.lk_window, self.cfg.lk_window),
            maxLevel=self.cfg.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        good = st.ravel().astype(bool) & st2.ravel().astype(bool)
        fb = np.linalg.norm(pts_prev.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)
        good &= fb < self.cfg.fb_error_px
        a = pts_prev.reshape(-1, 2)[good]
        b = p1.reshape(-1, 2)[good]

        H_rel = None
        inliers = 0
        ratio = 0.0
        reproj = 0.0
        if len(a) >= 20:
            H_rel, mask_i = cv2.findHomography(a, b, cv2.RANSAC,
                                               self.cfg.ransac_px, maxIters=2000,
                                               confidence=0.995)
            if H_rel is not None and mask_i is not None:
                inliers = int(mask_i.sum())
                ratio = inliers / float(len(a))
                if inliers >= 8:
                    projected = cv2.perspectiveTransform(a[None].astype(np.float32), H_rel)[0]
                    reproj = float(np.median(np.linalg.norm(projected - b, axis=1)))

        valid = (H_rel is not None
                 and inliers >= self.cfg.min_inliers
                 and ratio >= self.cfg.min_ratio
                 and reproj <= self.cfg.max_reproj_px
                 and _plausible_homography(H_rel, self.width, self.height))
        if valid:
            new_H = H_rel @ self.H
            if abs(new_H[2, 2]) > 1e-12:
                new_H = new_H / new_H[2, 2]
            new_lk = H_rel @ self._H_lk
            if abs(new_lk[2, 2]) > 1e-12:
                new_lk = new_lk / new_lk[2, 2]
            # H is world->image; the image-corner plausibility test above
            # applies to the image->image relative homography, not to H.
            # Composing a validated relative motion onto a validated anchor
            # cannot change the domain, so trust it here.
            self.H = new_H
            self._H_lk = new_lk
            self.lost_frames = 0
        if not valid:
            self.lost_frames += 1

        # Re-detect features on the new frame when the pool gets thin.
        fresh = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.cfg.max_corners,
            qualityLevel=self.cfg.quality_level,
            minDistance=self.cfg.min_distance, mask=mask,
            blockSize=self.cfg.block_size)
        self._prev_gray = gray
        self._prev_pts = fresh if fresh is not None else p1
        self._prev_hist = self._histogram(gray)

        inlier_component = min(1.0, ratio / 0.75) * min(1.0, inliers / 250.0)
        confidence = inlier_component
        if line_hit is not None:
            confidence = 0.55 * inlier_component + 0.45 * float(np.clip(line_hit, 0, 1))
        if not valid:
            confidence *= 0.15
        state = TrackState(self.H.copy(), valid, inliers=inliers, ratio=ratio,
                           reproj_px=reproj, confidence=float(np.clip(confidence, 0, 1)),
                           lost_frames=self.lost_frames,
                           features=_count(self._prev_pts))
        state = self._sift_keyframe_correct(frame, gray, boxes, line_hit, state)
        self.last_state = state
        return state

    # ------------------------------------------------------------------ SIFT
    def _sift_mask(self, frame: np.ndarray, boxes) -> np.ndarray:
        from .linemask import pitch_polygon_h
        poly = pitch_polygon_h(self.H, self.template, margin_m=1.0)
        mask = np.ones(frame.shape[:2], np.uint8) * 255
        if poly is not None:
            cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 0)
        # Static off-pitch structures (stands, roofs, advertising boards) are
        # the only reliable SIFT features in a wide tactical shot: turf has
        # almost none at working resolution, and the strong in-pitch ones sit
        # on players, which are moving outliers.  Those structures are not on
        # the ground plane, so they share the ground homography only while the
        # camera pans/tilts/zooms about a fixed optical centre (no dolly).
        # sift_max_disagree_px guards the translation case, and RANSAC wins
        # over screen-fixed graphics such as a score bug.
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        if boxes is not None and len(boxes):
            for box in boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
                cv2.rectangle(mask, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), 0, -1)
        return mask

    def _init_sift_keyframe(self, frame, gray, boxes) -> None:
        if self._sift is None:
            return
        mask = self._sift_mask(frame, boxes)
        kp, des = self._sift.detectAndCompute(gray, mask)
        self._sift_kf = (kp, des, self.H.copy())
        self._sift_kf_frame = self.frame_count

    def _sift_keyframe_correct(self, frame, gray, boxes, line_hit, state):
        """Correct accumulated LK drift against the last SIFT keyframe.

        The keyframe is matched on static off-pitch structures (stands, roofs,
        boards), the only SIFT-rich support in a wide shot.  A keyframe
        baseline of ~0.5 s is much less ambiguous than consecutive-frame
        matching on repetitive mowing stripes, and the keyframe H is already
        corrected, so errors do not integrate.  Off-pitch points are not on the
        ground plane, so the result is accepted only when it is consistent with
        the LK trajectory (see ``sift_max_disagree_px``/
        ``sift_max_trajectory_px``).
        """
        if self._sift is None:
            return state
        if self._sift_kf is None:
            self._init_sift_keyframe(frame, gray, boxes)
            return state
        every = max(2, int(self.cfg.sift_keyframe_every))
        mask = self._sift_mask(frame, boxes)
        kp, des = self._sift.detectAndCompute(gray, mask)
        pkp, pdes, kf_H = self._sift_kf
        if des is None or pdes is None or len(kp) < 20 or len(pkp) < 20:
            return state
        good = [m for m, n in self._bf.knnMatch(pdes, des, k=2)
                if m.distance < self.cfg.sift_ratio * n.distance]
        if len(good) < self.cfg.sift_min_matches:
            return state
        a = np.float32([pkp[m.queryIdx].pt for m in good])
        b = np.float32([kp[m.trainIdx].pt for m in good])
        H_rel, mask_i = cv2.findHomography(a, b, cv2.RANSAC,
                                           self.cfg.sift_ransac_px,
                                           maxIters=3000, confidence=0.995)
        if H_rel is None or mask_i is None:
            return state
        inliers = int(mask_i.sum())
        ratio = inliers / float(len(good))
        proj = cv2.perspectiveTransform(a[None], H_rel)[0]
        reproj = float(np.median(np.linalg.norm(proj - b, axis=1)))
        valid = (inliers >= self.cfg.min_inliers and ratio >= self.cfg.min_ratio
                 and reproj <= self.cfg.max_reproj_px
                 and _plausible_homography(H_rel, self.width, self.height))
        if not valid:
            return state
        # A stand-based keyframe solution is a drift correction, not a new
        # independent trajectory: it should agree with LK to within accumulated
        # drift.  A large disagreement means either real camera translation
        # (off-pitch features then sit on the wrong plane) or a screen-fixed
        # graphic (a score bug is static in screen space, not in the world).
        try:
            H_lk_rel = self.H @ np.linalg.inv(kf_H)
            if abs(H_lk_rel[2, 2]) > 1e-12:
                H_lk_rel = H_lk_rel / H_lk_rel[2, 2]
            probe = np.array([[[self.width / 2.0, self.height / 2.0],
                               [0, 0], [self.width - 1, 0],
                               [self.width - 1, self.height - 1],
                               [0, self.height - 1]]], np.float32)
            p_lk = cv2.perspectiveTransform(probe, H_lk_rel)[0]
            p_sift = cv2.perspectiveTransform(probe, H_rel)[0]
            disagree = float(np.median(np.linalg.norm(p_lk - p_sift, axis=1)))
        except np.linalg.LinAlgError:
            return state
        if not np.isfinite(disagree):
            return state
        # If LK itself failed this frame there is no trajectory to agree with;
        # let SIFT re-anchor instead of rejecting the only working measurement.
        if state.valid and disagree > self.cfg.sift_max_disagree_px:
            return state
        new_H = H_rel @ kf_H
        if abs(new_H[2, 2]) > 1e-12:
            new_H = new_H / new_H[2, 2]
        # Bound the total correction against the pure-LK trajectory.  A small
        # stand/depth bias can pass the per-interval check yet accumulate to a
        # large ground-plane error over hundreds of frames.
        grid = np.array([(x, y) for x in np.linspace(-52.0, 52.0, 7)
                         for y in np.linspace(-33.0, 33.0, 5)], float)
        hom = np.column_stack([grid, np.ones(len(grid))])
        qa, qb = hom @ new_H.T, hom @ self._H_lk.T
        ok = (np.isfinite(qa).all(axis=1) & np.isfinite(qb).all(axis=1)
              & (np.abs(qa[:, 2]) > 1e-9) & (np.abs(qb[:, 2]) > 1e-9))
        if state.valid and ok.any():
            drift = float(np.median(np.linalg.norm(
                qa[ok, :2] / qa[ok, 2:3] - qb[ok, :2] / qb[ok, 2:3], axis=1)))
            if not np.isfinite(drift) or drift > self.cfg.sift_max_trajectory_px:
                return state
        self.H = new_H
        self.lost_frames = 0
        # Refresh the keyframe when its baseline is ~1 s old.  Every frame is
        # still matched to the current keyframe, so H never integrates drift
        # for more than one keyframe interval.
        if self.frame_count - self._sift_kf_frame >= every:
            self._sift_kf = (kp, des, self.H.copy())
            self._sift_kf_frame = self.frame_count
        # Re-seed LK on this corrected frame.
        m2 = self._feature_mask(frame, boxes)
        fresh = cv2.goodFeaturesToTrack(gray, maxCorners=self.cfg.max_corners,
                                        qualityLevel=self.cfg.quality_level,
                                        minDistance=self.cfg.min_distance, mask=m2,
                                        blockSize=self.cfg.block_size)
        self._prev_gray = gray
        self._prev_pts = fresh
        inlier_comp = min(1.0, ratio / 0.75) * min(1.0, inliers / 120.0)
        conf = inlier_comp
        if line_hit is not None:
            conf = 0.55 * inlier_comp + 0.45 * float(np.clip(line_hit, 0, 1))
        return TrackState(self.H.copy(), True, inliers=inliers, ratio=ratio,
                          reproj_px=reproj, confidence=float(np.clip(conf, 0, 1)),
                          lost_frames=self.lost_frames, features=len(kp))

    # ------------------------------------------------------------------ helpers
    def _detect(self, gray: np.ndarray, mask: np.ndarray):
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=self.cfg.max_corners,
            qualityLevel=self.cfg.quality_level,
            minDistance=self.cfg.min_distance, mask=mask,
            blockSize=self.cfg.block_size)

    def _feature_mask(self, frame: np.ndarray, boxes: Optional[Sequence[Sequence[float]]]) -> np.ndarray:
        poly = pitch_polygon_h(self.H, self.template, margin_m=3.0)
        mask = np.zeros((self.height, self.width), np.uint8)
        if poly is None:
            return mask
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
        mask = cv2.dilate(mask, np.ones((2 * self.cfg.polygon_margin_px + 1,) * 2, np.uint8))
        if self.cfg.green_features and frame is not None:
            b, g, r = cv2.split(frame)
            exg = 2.0 * g.astype(np.float32) - r.astype(np.float32) - b.astype(np.float32)
            green = (exg > self.cfg.green_min_exg).astype(np.uint8) * 255
            green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            green = cv2.dilate(green, np.ones((5, 5), np.uint8))
            mask = cv2.bitwise_and(mask, green)
        if boxes is not None and len(boxes):
            for box in boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
                cv2.rectangle(mask, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), 0, -1)
        # Keep features away from the frame edge and from the horizon/stands.
        return mask


    def _histogram(self, gray: np.ndarray):
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([small], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _is_cut(self, gray: np.ndarray) -> bool:
        if self._prev_gray is None or self._prev_hist is None:
            return False
        diff = float(np.mean(np.abs(gray.astype(np.float32) - self._prev_gray.astype(np.float32))))
        hist = self._histogram(gray)
        corr = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL))
        return diff > self.cfg.max_frame_delta and corr < self.cfg.min_hist_corr


def _count(pts) -> int:
    return 0 if pts is None else int(len(pts))


def _plausible_homography(H: np.ndarray, width: int, height: int) -> bool:
    if H is None or not np.all(np.isfinite(H)):
        return False
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    moved = cv2.perspectiveTransform(corners[None], H.astype(np.float64))[0]
    if not np.all(np.isfinite(moved)):
        return False
    area0 = float(width * height)
    area1 = abs(float(cv2.contourArea(moved.astype(np.float32))))
    ratio = area1 / area0
    if not (0.25 < ratio < 4.0):
        return False
    # Any point must not have flipped to the far side of the horizon.
    if np.any(moved[:, 0] < -4 * width) or np.any(moved[:, 0] > 5 * width):
        return False
    if np.any(moved[:, 1] < -4 * height) or np.any(moved[:, 1] > 5 * height):
        return False
    return True


def line_hit_rate(frame: np.ndarray, homography: np.ndarray,
                  template, camera: PinholeCamera,
                  max_segments: int = 300,
                  threshold_px: float = 5.0) -> tuple[float, float]:
    """Fraction of projected template points near an LSD line segment.

    Returns ``(hit_rate, coverage)`` where coverage is the fraction of template
    lines with at least some observed support.  This is a confidence aid, not
    an optimiser: missing far lines reduce coverage but do not make the camera
    wrong, and a correct camera over faint paint can still score low.
    """
    if frame is None:
        return 0.0, 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return 0.0, 0.0
    segs = np.asarray(lines).reshape(-1, 4)
    h, w = gray.shape
    points = []
    keep = []
    for idx, (x1, y1, x2, y2) in enumerate(segs[:max_segments]):
        if math.hypot(x2 - x1, y2 - y1) < 15:
            continue
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / 2))
        for t in np.linspace(0, 1, n):
            points.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        keep.append(idx)
    if not points:
        return 0.0, 0.0
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(points, np.float64))
    lines_world = template.sample_points(step=0.8)
    hit = 0
    total = 0
    covered = 0
    for name, pts in lines_world:
        world = np.column_stack([pts, np.zeros(len(pts))])
        uv, _, valid = camera.project(world)
        inside = valid & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if inside.sum() < 3:
            continue
        d, _ = tree.query(uv[inside], k=1)
        total += len(d)
        hit += int(np.sum(d < threshold_px))
        if float(np.quantile(d, 0.5)) < 12:
            covered += 1
    if total == 0:
        return 0.0, 0.0
    coverage = covered / max(1, len(lines_world))
    return hit / float(total), float(coverage)


class FixedCenterStabilizer:
    """Project a tracked homography onto a fixed-centre PTZ camera.

    Broadcast and tactical cameras pan/tilt/zoom on a tripod: their optical
    centre does not move.  Free-homography tracking has eight degrees of
    freedom and can drift into translation/height modes that are physically
    impossible.  Re-fitting only ``f`` and ``R`` with the anchor centre pinned
    removes those modes.  If the solve cannot explain the tracked motion (a
    real dolly move, or a bad frame), the raw homography is kept.
    """

    def __init__(self, camera: PinholeCamera, width: int = 960, height: int = 540):
        self.camera = camera.copy()
        self.width = int(width)
        self.height = int(height)
        self.prev = camera.copy()
        self.C = camera.C.copy()
        self.f0 = camera.f
        # A world grid spanning the pitch plus a small run-off.  Only the
        # visible subset is used each frame.
        self.world_xy = np.array([(x, y)
                                  for x in np.linspace(-55.0, 55.0, 23)
                                  for y in np.linspace(-37.0, 37.0, 15)], float)

    def stabilize(self, H: np.ndarray) -> tuple[np.ndarray, PinholeCamera, float, bool]:
        hom = np.column_stack([self.world_xy, np.ones(len(self.world_xy))])
        q = (H @ hom.T).T
        ok = np.isfinite(q).all(axis=1) & (np.abs(q[:, 2]) > 1e-9)
        uv = np.full((len(self.world_xy), 2), np.nan)
        uv[ok] = q[ok, :2] / q[ok, 2:3]
        inside = (ok & (uv[:, 0] >= -50) & (uv[:, 0] < self.width + 50)
                  & (uv[:, 1] >= -50) & (uv[:, 1] < self.height + 50))
        if int(inside.sum()) < 40:
            return H, self.prev, 99.0, False
        world = np.column_stack([self.world_xy[inside], np.zeros(int(inside.sum()))])
        target = uv[inside]
        R0 = self.prev.R.copy()
        C = self.C

        def make(p):
            f = self.f0 * math.exp(float(p[0]))
            R = rvec_to_rotation(p[1:4]) @ R0
            return PinholeCamera(f, self.width / 2.0, self.height / 2.0, R, C)

        def resid(p):
            cam = make(p)
            proj, _, valid = cam.project(world)
            err = np.where(valid[:, None], proj - target, 20.0)
            return np.concatenate([err.ravel(),
                                   [3.0 * p[0], 18.0 * p[1], 18.0 * p[2], 18.0 * p[3]]])

        try:
            res = least_squares(resid, np.zeros(4),
                                bounds=([-0.6, -0.3, -0.3, -0.3],
                                        [0.6, 0.3, 0.3, 0.3]),
                                loss="soft_l1", f_scale=3.0, max_nfev=50,
                                x_scale=[0.1, 0.04, 0.04, 0.04])
        except Exception:
            return H, self.prev, 99.0, False
        cam = make(res.x)
        proj, _, valid = cam.project(world)
        if not np.any(valid):
            return H, self.prev, 99.0, False
        err = float(np.median(np.linalg.norm(proj[valid] - target[valid], axis=1)))
        plausible = (err <= 22.0 and 0.30 * self.f0 <= cam.f <= 2.5 * self.f0
                     and cam.height > 0.5)
        if not plausible:
            return H, self.prev, err, False
        self.prev = cam
        Hn = cam.homography()
        if abs(Hn[2, 2]) > 1e-12:
            Hn = Hn / Hn[2, 2]
        return Hn, cam, err, True
