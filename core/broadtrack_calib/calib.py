"""
``BroadTrackCalib`` keeps original public surface — ``estimate``,
``draw_pitch_lines``, ``world_to_bev_px`` and ``create_bev_template``

But the calibration itself is BroadTrack's temporal tracker:
a per-frame Ceres bundle adjustment over the segmented pitch lines
regularised by Lucas-Kanade ground correspondences,
with a line-IoU pan refinement
and a keypoint-driven re-initialisation when tracking is lost.

What runs where
---------------
Python
    the two TorchScript detectors (line segmentation, pitch keypoints)
    and all of the drawing / BEV code.
C++ (``core/BroadTrack/python/``, built by ``scripts/build_native.py``)
    everything per-frame that is numerically expensive: point extraction, the
    Ceres solve, the line-IoU score, the optical-flow association, and the
    state machine that decides when to re-initialise.

``PitchGeometry`` is split out of ``BroadTrackCalib``
because the pitch model and the BEV rendering need neither the detectors nor the extension's heavy parts;

Example
-------
    calib = BroadTrackCalib(weights_kp="models/nbjw_keypoint_model.pt",
                            weights_line="models/tvcalib_model.pt",
                            device="cuda")
    result = calib.estimate(frame, boxes=player_boxes)   # dict or None
    K, R, t = result["K"], result["R"], result["t"]
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import cv2
import numpy as np

from core.log import TRACE, get_logger

# Import order matters on macOS.
# Ceres pulls in the system OpenMP runtime through OpenBLAS/SuiteSparse, and PyTorch uses its own copy;
# if Ceres initialises first, libomp aborts.
# So, DO NOT reorder these two imports.
from .models import KeypointDetectionModel, LineSegmentationModel, keypoints_to_list

from . import _broadtrack

# in meters
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
CENTER_CIRCLE_RADIUS = 9.15

logger = get_logger(__name__)


def pitch_wireframe(delta: float = 1.0,
                    length: float = PITCH_LENGTH,
                    width: float = PITCH_WIDTH) -> list:
    """List of (N, 3) world-metre polylines.
    Comes from BroadTrack's own ``SoccerPitch3D::getWireframe``.
    ``length``/``width`` only change the outer dimensions; the markings keep
    their regulation sizes, as on a youth pitch.
    """
    return [np.asarray(polyline, dtype=np.float64)
            for polyline in _broadtrack.pitch_wireframe(delta, length, width)]


class PitchGeometry:
    """ Pitch constants & world <-> BEV mapping. """

    def __init__(
        self,
        length: float = PITCH_LENGTH,
        width: float = PITCH_WIDTH,
        bev_scale: float = 10.0,
        wireframe: Optional[Sequence[np.ndarray]] = None,
    ):
        self.length = float(length)
        self.width = float(width)
        self.bev_scale = float(bev_scale)
        self._wireframe = (list(wireframe) if wireframe is not None
                           else pitch_wireframe(length=self.length, width=self.width))

    @property
    def wireframe(self) -> list:
        return self._wireframe

    def world_to_bev_px(self, x: float, y: float) -> tuple:
        """World metres -> BEV pixels"""
        px = int((x + self.length / 2) * self.bev_scale)
        py = int((self.width / 2 + y) * self.bev_scale)
        return px, py

    def create_bev_template(self) -> np.ndarray:
        """Static top-down pitch image."""
        width = int(self.length * self.bev_scale)
        height = int(self.width * self.bev_scale)
        canvas = np.ones((height, width, 3), dtype=np.uint8) * np.array([76, 156, 76], dtype=np.uint8)

        for polyline in self._wireframe:
            points = [self.world_to_bev_px(float(p[0]), float(p[1])) for p in polyline]
            for start, end in zip(points[:-1], points[1:]):
                cv2.line(canvas, start, end, (255, 255, 255), 1)

        center = self.world_to_bev_px(0.0, 0.0)
        cv2.circle(canvas, center, int(CENTER_CIRCLE_RADIUS * self.bev_scale), (255, 255, 255), 1)
        cv2.circle(canvas, center, 3, (255, 255, 255), -1)

        return canvas

    def draw_pitch_lines(self, img: np.ndarray, P: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Project the pitch markings onto ``img`` with projection matrix ``P``."""
        height, width = img.shape[:2]
        for polyline in self._wireframe:
            projected = _project_polyline(polyline, P)
            if projected is None:
                continue
            for start, end in zip(projected[:-1], projected[1:]):
                if not (_in_frame(start, width, height) and _in_frame(end, width, height)):
                    continue
                cv2.line(img, start, end, color, thickness)
        return img


class BroadTrackCalib:
    """Temporal pitch calibration for a single broadcast camera.

    One instance tracks one video stream, frame by frame, in order. Calls are
    not thread-safe: the tracker's camera, optical-flow points and lost-tracking
    counter are shared mutable state.
    """

    def __init__(
        self,
        weights_kp: str,
        weights_line: str,
        device: str = "cpu",
        width: int = 960,
        height: int = 540,
        *,
        pitch_length: float = PITCH_LENGTH,
        pitch_width: float = PITCH_WIDTH,
        bev_scale: float = 10.0,
        line_device: Optional[str] = None,
        keypoint_device: Optional[str] = None,
        tracker_config: Optional[dict] = None,
        keypoint_model_kwargs: Optional[dict] = None,
        line_model_kwargs: Optional[dict] = None,
    ):
        """
        Args:
            weights_kp: TorchScript pitch keypoint model (``nbjw_keypoint_model.pt``).
            weights_line: TorchScript pitch-line segmentation model (``tvcalib_model.pt``).
            device: ``"cuda"``, ``"mps"`` or ``"cpu"``; falls back to CPU when the
                requested backend is unavailable.
            width, height: expected frame size. Used to report a mismatch; the
                tracker itself adapts to whatever frame it is handed.
            pitch_length, pitch_width: pitch size in metres, for BEV rendering.
            bev_scale: BEV pixels per metre.
            line_device, keypoint_device: per-model device overrides. They
                default to ``device``. Worth setting apart: the segmentation
                model is small and runs about as fast on CPU as on MPS, while
                the 58-channel keypoint model is roughly ten times faster on
                MPS (~1.1 s vs ~11.7 s per frame at 1080p in our measurements).
                The keypoint model only runs when the tracker loses lock, so on
                CPU it costs a stall only when something has already gone wrong.
            tracker_config: overrides forwarded to the C++ tracker. See
                ``core/BroadTrack/python/BroadTrackTracker.h`` for the keys —
                ``position_mode`` (``"free"``/``"soft"``), ``tripod_x/y/z``,
                ``tripod_radius``, ``optical_flow``, ``radial_distortion``,
                ``reinit_score`` and friends.
            keypoint_model_kwargs, line_model_kwargs: forwarded to the two
                model wrappers in :mod:`core.broadtrack.models`.
        """
        self.weights_kp = str(weights_kp)
        self.weights_line = str(weights_line)
        self.width = int(width)
        self.height = int(height)

        self.geometry = PitchGeometry(pitch_length, pitch_width, bev_scale)

        self._line_model = LineSegmentationModel(
            self.weights_line, line_device or device, **(line_model_kwargs or {})
        )
        self._keypoint_model = KeypointDetectionModel(
            self.weights_kp, keypoint_device or device, **(keypoint_model_kwargs or {})
        )
        self.device = self._line_model.device

        config = dict(tracker_config or {})
        config.setdefault("hd_width", 1920.0)
        config.setdefault("hd_height", 1080.0)
        config.setdefault("pitch_length", float(pitch_length))
        config.setdefault("pitch_width", float(pitch_width))
        self._tracker = _broadtrack.Tracker(config)
        self._tracker_config = config

        self._last_result: Optional[dict] = None
        self._last_frame_shape: Optional[tuple] = None

        logger.info(
            "BroadTrackCalib initialized (line model on %s, keypoint model on %s; "
            "expected %dx%d, HD reference %.0fx%.0f, position_mode=%s)",
            self._line_model.device,
            self._keypoint_model.device,
            self.width,
            self.height,
            config["hd_width"],
            config["hd_height"],
            config.get("position_mode", "free"),
        )

    # ------------------------------------------------------------------ core

    def estimate(self, frame: np.ndarray, boxes: Optional[np.ndarray] = None) -> Optional[dict]:
        """Run one frame of the BroadTrack pipeline.

        Args:
            frame: HxWx3 uint8 BGR image. The pipeline works best at 16:9; the
                aspect ratio is checked on the first frame and a mismatch is
                reported once.
            boxes: optional Nx4 array of player bounding boxes ``(x1, y1, x2, y2)``
                in frame pixels. They are used to discard optical-flow points
                that landed on a moving player, which is what BroadTrack's
                ``human-bboxes`` input does. Passing the player tracker's
                detections for the *same* frame measurably helps.

        Returns:
            A dict with ``P``, ``K``, ``R``, ``t`` (unchanged semantics relative
            to PnLCalib) plus ``score``, ``reinit``, ``frames_lost``, ``k1``,
            ``focal_length``, ``pan_deg``, ``tilt_deg``, ``roll_deg`` and
            ``line_mask``; or ``None`` if the solve did not produce a usable
            camera.
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 BGR image")
        self._check_frame_shape(frame)

        line_mask = self._line_model.compute_line_mask(frame)
        box_array = self._normalise_boxes(boxes)
        logger.trace("estimate: %d player boxes handed to the tracker", len(box_array))

        # The extension only calls this when it decides a re-initialisation is
        # worth attempting, which is what keeps the keypoint model off the
        # hot path.
        def keypoints_fn():
            keypoints = keypoints_to_list(self._keypoint_model.compute_keypoints(frame))
            if logger.isEnabledFor(TRACE):
                logger.trace(
                    "estimate: keypoint model returned %d point ids (%s)",
                    len(keypoints),
                    ", ".join(str(point_id) for point_id, _ in keypoints),
                )
            return keypoints

        native = self._tracker.estimate(frame, line_mask, box_array, keypoints_fn)

        if not native["ok"]:
            logger.debug("estimate: tracker solve failed (ok=false)")
            return None
        rejection = self._rejection_reason(native)
        if rejection is not None:
            logger.debug("estimate: rejecting solve (%s)", rejection)
            return None

        K = np.ascontiguousarray(native["K"], dtype=np.float64)
        R = np.ascontiguousarray(native["R"], dtype=np.float64)
        t = np.ascontiguousarray(native["t"], dtype=np.float64)

        It = np.eye(4, dtype=np.float64)[:3]
        It[:, 3] = -t
        P = K @ (R @ It)

        result = {
            "P": P,
            "K": K,
            "R": R,
            "t": t,
            "score": native["score"],
            "reinit": native["reinit"],
            "frames_lost": native["frames_lost"],
            "k1": native["k1"],
            "focal_length": native["focal_length"],
            "pan_deg": native["pan_deg"],
            "tilt_deg": native["tilt_deg"],
            "roll_deg": native["roll_deg"],
            "line_mask": line_mask,
        }
        self._last_result = result
        logger.debug(
            "estimate: score=%.3f reinit=%s frames_lost=%s focal=%.1f "
            "pan=%.2f tilt=%.2f roll=%.2f k1=%.4f",
            result["score"],
            result["reinit"],
            result["frames_lost"],
            result["focal_length"],
            result["pan_deg"],
            result["tilt_deg"],
            result["roll_deg"],
            result["k1"],
        )
        return result

    def reset(self) -> None:
        """Forget all temporal state. Call this when the video cuts."""
        self._tracker.reset()
        self._last_result = None
        logger.debug("reset: calibration temporal state cleared")

    def set_camera(
        self,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        k1: float = 0.0,
        *,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        fix_position: bool = True,
    ) -> None:
        """Bootstrap the tracker from a known camera.

        Args:
            K: 3x3 intrinsics in the caller's frame resolution. The principal
                point may be off-centre: a hand-fitted anchor homography can
                require it, and the tracker keeps it fixed while solving.
            R: 3x3 world->camera rotation (world frame z points *down*, as
                described in the class docstring).
            t: camera centre in world metres.
            k1: first radial-distortion coefficient.
            frame_width, frame_height: resolution ``K`` was measured at;
                defaults to the constructor's ``width``/``height``.
            fix_position: pin the camera centre, k1 and roll. Use this for a
                fixed PTZ camera marked by a human; the per-frame solve then
                only follows pan/tilt and zoom. Pass ``False`` for a camera
                that physically moves.

        A manual anchor from :mod:`core.anchor_bootstrap` is the intended
        producer. After this call the tracker runs from the anchor instead of
        BroadTrack's default broadcast prior.
        """
        width = int(frame_width or self.width)
        height = int(frame_height or self.height)
        self._tracker.set_camera(
            np.asarray(K, dtype=np.float64),
            np.asarray(R, dtype=np.float64),
            np.asarray(t, dtype=np.float64),
            float(k1), width, height, bool(fix_position),
        )
        self._last_result = None
        logger.info(
            "set_camera: bootstrapped at %dx%d (f=%.1f, fix_position=%s)",
            width, height, float(np.asarray(K, dtype=np.float64)[0, 0]), fix_position,
        )

    @property
    def score(self) -> Optional[float]:
        """Line-IoU score of the most recent solve, or ``None``."""
        return None if self._last_result is None else self._last_result["score"]

    # ------------------------------------------------------------- rendering

    def draw_pitch_lines(self, img: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Project the pitch markings onto ``img`` with the current camera.

        Draws nothing until the first successful :meth:`estimate`.
        """
        if self._last_result is None:
            return img
        return self.geometry.draw_pitch_lines(img, self._last_result["P"], color, thickness)

    def world_to_bev_px(self, x: float, y: float) -> tuple:
        """World metres -> BEV pixels. Same mapping PnLCalib used."""
        return self.geometry.world_to_bev_px(x, y)

    def create_bev_template(self) -> np.ndarray:
        """Static top-down pitch image, drawn from BroadTrack's own pitch model."""
        return self.geometry.create_bev_template()

    # -------------------------------------------------------------- internals

    def _check_frame_shape(self, frame: np.ndarray) -> None:
        shape = frame.shape[:2]
        if shape == self._last_frame_shape:
            return
        self._last_frame_shape = shape
        rows, cols = shape
        if (cols, rows) != (self.width, self.height):
            logger.warning(
                "frame is %dx%d but the tracker was built for %dx%d; adapting "
                "(the tracker rescales internally)",
                cols, rows, self.width, self.height,
            )
        if abs(cols / rows - 16.0 / 9.0) > 0.02:
            logger.warning(
                "frame aspect %.3f is not 16:9. The pitch-line segmentation model resizes "
                "to a fixed shortest side, so the line mask is only geometrically "
                "consistent with the frame at 16:9.",
                cols / rows,
            )

    @staticmethod
    def _normalise_boxes(boxes: Optional[np.ndarray]) -> np.ndarray:
        if boxes is None:
            return np.zeros((0, 4), dtype=np.float64)
        array = np.asarray(boxes, dtype=np.float64)
        if array.size == 0:
            return np.zeros((0, 4), dtype=np.float64)
        array = array.reshape(-1, 4)
        if not np.isfinite(array).all():
            array = array[np.isfinite(array).all(axis=1)]
        return np.ascontiguousarray(array)

    @staticmethod
    def _rejection_reason(native: dict) -> Optional[str]:
        """Why a solve cannot be used, or ``None`` when it is fine.

        The tracker always returns *something* — it starts from a prior camera
        and only ever improves on it — so unlike PnLCalib there is no "failed to
        estimate" state. What can still happen is a numerically broken solve, and
        that is what this filters.
        """
        if not np.isfinite(native["K"]).all() or not np.isfinite(native["R"]).all():
            return "non-finite intrinsics or rotation"
        if not np.isfinite(native["t"]).all():
            return "non-finite camera position"
        if native["K"][0, 0] <= 0.0:
            return "non-positive focal length"
        # The camera must sit above the ground plane (z is negative upwards).
        if native["t"][2] >= 0.0:
            return "camera below the ground plane"
        return None

    @classmethod
    def _is_usable(cls, native: dict) -> bool:
        """Whether :meth:`estimate` can return this solve to the caller."""
        return cls._rejection_reason(native) is None


def _project_polyline(polyline: np.ndarray, P: np.ndarray) -> Optional[list]:
    """Project an (N, 3) world polyline to pixels; ``None`` if any point is not
    safely in front of the camera.

    Dividing by a negative w mirrors the point through the principal point
    instead of rejecting it, which would draw a line across the frame in the
    wrong place, so the whole polyline is dropped rather than partly drawn.
    """
    if len(polyline) == 0:
        return None
    homogeneous = np.hstack([polyline, np.ones((len(polyline), 1))])
    projected = (P @ homogeneous.T).T
    w = projected[:, 2]
    if np.any(w <= 1e-9):
        return None
    pixels = projected[:, :2] / w[:, None]
    if not np.isfinite(pixels).all():
        return None
    return [(int(round(px)), int(round(py))) for px, py in pixels]


def _in_frame(point: Sequence[int], width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height
