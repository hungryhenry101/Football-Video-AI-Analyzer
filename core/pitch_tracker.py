"""Per-frame calibration from pitchcal's ground tracker and line evidence.

BroadTrack solves the camera with a semantic line-segmentation network every
frame.  That network is the dominant cost and is unreliable on youth/low-angle
footage.  PitchCal's design is the opposite: Lucas-Kanade carries the
frame-to-frame motion (CPU, ~10-20 ms) and a domain-independent white-paint
detector (LSD + turf-photometry) is run on a sparse schedule for confidence and
a conservative homography correction.

This module wraps that pair into one object.  BroadTrack stays available for
anchor/keyframe absolute calibration; between those frames the returned ground
homography is the authoritative calibration, so the rest of the pipeline
projects through it directly (``homography_pixel_to_ground``) instead of an
approximate camera.

    tracker = PitchHomographyTracker(H, 960, 540, (105, 68), principal, focal)
    calib = tracker.update(frame, boxes=player_boxes)
    ground = homography_pixel_to_ground(u, v, calib.homography)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from core.anchor_bootstrap import camera_from_tracked_homography
from core.log import get_logger
from core.pitchcal.geometry import PitchTemplate
from core.pitchcal.linemask import paint_like_segments
from core.pitchcal.segments import (
    detect_segments,
    filter_segments,
    hit_from_segments,
    refine_homography,
)
from core.pitchcal.tracking import GroundTracker, TrackerConfig

logger = get_logger(__name__)


@dataclass
class PitchTrackConfig:
    """Sparse-schedule knobs, matching pitchcal's pipeline defaults."""

    line_every: int = 4          # frames between LSD + white-paint measurements
    correct_every: int = 20      # frames between gated H corrections (0 = off)
    sift_keyframe_every: int = 15  # frames between SIFT drift checks


@dataclass
class PitchTrackFrame:
    homography: np.ndarray
    valid: bool
    confidence: float
    line_hit: float
    line_coverage: float
    lost_frames: int
    cut: bool
    inliers: int
    ratio: float
    # Camera approximation, for diagnostics/z_est and BroadTrack keyframes.
    # Ground projection must use ``homography`` instead.
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray


class PitchHomographyTracker:
    """Stateful LK tracker + periodic white-line evidence for one video."""

    def __init__(
        self,
        homography: np.ndarray,
        width: int,
        height: int,
        pitch: tuple[float, float],
        principal: tuple[float, float],
        focal: float,
        config: Optional[PitchTrackConfig] = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.template = PitchTemplate(float(pitch[0]), float(pitch[1]))
        self.config = config or PitchTrackConfig()
        self._tracker: Optional[GroundTracker] = None
        self._principal = (float(principal[0]), float(principal[1]))
        self._focal = float(focal)
        self._last_hit = 0.0
        self._last_coverage = 0.0
        self._last_camera = None
        self.reset(homography, principal, focal)

    def reset(
        self,
        homography: np.ndarray,
        principal: Optional[tuple[float, float]] = None,
        focal: Optional[float] = None,
    ) -> None:
        """Rebase the tracker on a new absolute homography (new anchor/keyframe)."""
        if principal is not None:
            self._principal = (float(principal[0]), float(principal[1]))
        if focal is not None:
            self._focal = float(focal)
        self._last_hit = 0.0
        self._last_coverage = 0.0
        self._tracker = GroundTracker(
            homography=np.asarray(homography, float),
            width=self.width,
            height=self.height,
            template=self.template,
            config=TrackerConfig(
                sift_keyframe_every=max(2, int(self.config.sift_keyframe_every)),
            ),
        )
        self._last_camera = camera_from_tracked_homography(
            homography, self.width, self.height, self._principal, self._focal,
        )
        if self._last_camera is not None:
            self._focal = float(self._last_camera[0][0, 0])
        logger.debug(
            "PitchHomographyTracker reset at %dx%d (principal %.1f, %.1f, f=%.1f)",
            self.width, self.height, self._principal[0], self._principal[1], self._focal,
        )

    def update(self, frame: np.ndarray, boxes=None) -> PitchTrackFrame:
        tracker = self._tracker
        if tracker is None:
            raise RuntimeError("PitchHomographyTracker.update before reset")
        if boxes is not None:
            boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)

        state = tracker.update(
            frame, boxes=boxes,
            line_hit=self._last_hit if self._last_hit > 0.0 else None,
        )
        index = tracker.frame_count

        # Sparse white-paint evidence: the same LSD + photometric filter
        # pitchcal uses.  It feeds the confidence and the gated correction; it
        # is not a per-frame solve input.
        if self.config.line_every > 0 and index % self.config.line_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            segments = detect_segments(gray)
            segments = paint_like_segments(frame, segments)
            segments = filter_segments(
                segments, state.homography, self.template, boxes,
                self.width, self.height,
            )
            self._last_hit, self._last_coverage, _ = hit_from_segments(
                segments, state.homography, self.template, self.width, self.height,
            )
            logger.debug(
                "line evidence: hit=%.2f coverage=%.2f segments=%d",
                self._last_hit, self._last_coverage, len(segments),
            )

        # Conservative absolute correction against the observed paint.  It is
        # accepted only when the global paint support improves (see
        # segments.refine_homography), so a bad trace cannot jump the track.
        if (self.config.correct_every > 0 and index > 0
                and index % self.config.correct_every == 0 and state.valid):
            corrected, info = refine_homography(
                state.homography, frame, self.template, boxes)
            if info.get("accepted"):
                state.homography = corrected
                tracker.rebase(corrected)
                logger.debug(
                    "line correction accepted at frame %d (hit %.2f->%.2f, "
                    "median %.1f->%.1f px)",
                    index, info.get("hit_before", 0.0), info.get("hit_after", 0.0),
                    info.get("median_before_px", 0.0), info.get("median_after_px", 0.0),
                )
            else:
                logger.debug(
                    "line correction rejected at frame %d (%s)",
                    index, info.get("reason", "no clear improvement"),
                )

        H = np.asarray(state.homography, float)
        camera = camera_from_tracked_homography(
            H, self.width, self.height, self._principal, self._focal,
        )
        if camera is not None:
            self._last_camera = camera
            self._focal = float(camera[0][0, 0])
            K, R, t = camera
        elif self._last_camera is not None:
            K, R, t = self._last_camera
        else:
            K = np.full((3, 3), np.nan)
            R = np.full((3, 3), np.nan)
            t = np.full(3, np.nan)

        return PitchTrackFrame(
            homography=H,
            valid=bool(state.valid),
            confidence=float(state.confidence),
            line_hit=float(self._last_hit),
            line_coverage=float(self._last_coverage),
            lost_frames=int(state.lost_frames),
            cut=bool(state.cut),
            inliers=int(state.inliers),
            ratio=float(state.ratio),
            K=K, R=R, t=t,
        )

    @property
    def homography(self) -> Optional[np.ndarray]:
        return None if self._tracker is None else self._tracker.H
