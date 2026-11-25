"""
Foot point: the box's float bottom-centre, not the mask's lowest pixel.
KF: a per-track constant-velocity Kalman filter over the world position after projection.
On-pitch gate: ``update`` splits the filtered positions to on_pitch and off_pitch
"""

from __future__ import annotations

from typing import NamedTuple
import numpy as np

from core.log import TRACE, get_logger
from core.projection_utils import homography_pixel_to_ground, pixel_to_ground

logger = get_logger(__name__)

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PITCH_MARGIN = 1.0

# Constant-velocity model in world metres, one step per frame:
# state = [x, y, vx, vy]
_TRANSITION = np.array(
    [[1.0, 0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0, 1.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
)
_OBSERVATION = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0]])


class _Track(NamedTuple):
    """One player's filter state, and how long since they were last detected."""
    state: np.ndarray
    covariance: np.ndarray
    missing: int


def foot_point(obj) -> tuple:
    """Ground-contact pixel of a detection: the bottom-centre of its box.

    Args:
        obj: ``PlayerTracker.update``'s return value.
    Returns:
        ``(u, v)`` in frame pixels.
    """
    x1, _, x2, y2 = (float(v) for v in obj["bbox"])
    return (x1 + x2) / 2.0, y2


def inside_pitch(point, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH) -> bool:
    """Whether a world-metre ground point is on the pitch, within ``PITCH_MARGIN``.
    Args:
        point: ``(x, y)`` in metres, or None.
        pitch_length, pitch_width: outer pitch size; a manual anchor may
            describe a youth pitch.
    """
    if point is None:
        return False
    return (
        abs(float(point[0])) <= pitch_length / 2 + PITCH_MARGIN
        and abs(float(point[1])) <= pitch_width / 2 + PITCH_MARGIN
    )


class PlayerPositionFilter:
    """Per-track Kalman filter over players' BEV positions.
    Args:
        process_noise: variance added to every state each frame.
        measurement_noise: variance of the ground position a detection has, in m².
        initial_covariance: initial state covariance, loose on purpose
        max_coast: frames a track may go undetected before it is forgotten
    """

    def __init__(
        self,
        process_noise: float = 0.05,
        measurement_noise: float = 0.25,
        initial_covariance: float = 1.0,
        max_coast: int = 25,
        pitch_length: float = PITCH_LENGTH,
        pitch_width: float = PITCH_WIDTH,
    ):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_covariance = float(initial_covariance)
        self.max_coast = int(max_coast)
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)

        self._q = np.eye(4) * self.process_noise
        self._r = np.eye(2) * self.measurement_noise
        self._tracks: dict[int, _Track] = {}
        logger.debug(
            "PlayerPositionFilter: process_noise=%.3f measurement_noise=%.3f "
            "initial_covariance=%.3f max_coast=%d",
            self.process_noise, self.measurement_noise,
            self.initial_covariance, self.max_coast,
        )

    def update(self, tracked_objects, K=None, R=None, t=None, H=None):
        """Filter this frame's detections
        Args:
            tracked_objects: as returned by ``PlayerTracker.update``.
            K, R, t: the calibration for current frame (camera form).
            H: world->image ground homography. Use this instead of K/R/t when
                the calibration is tracked as a homography (pitchcal).
        Returns:
            ``(on_pitch, off_pitch)``:
            ``on_pitch`` is ``{track_id: (x, y)}`` in world metres; ``off_pitch`` is the set of ids off pitch
        """
        if H is None and (K is None or R is None or t is None):
            raise ValueError("update needs either H or K, R, t")
        measured = {}
        for obj in tracked_objects:
            u, v = foot_point(obj)
            point = (homography_pixel_to_ground(u, v, H) if H is not None
                     else pixel_to_ground(u, v, K, R, t))
            if point is None or not np.isfinite(point).all():
                logger.trace("update: id=%s foot pixel (%.1f, %.1f) is unprojectable",
                             int(obj["id"]), u, v)
                continue
            measured[int(obj["id"])] = np.asarray(point, dtype=np.float64)
        n_projectable = len(measured)

        positions = {}
        for tid in list(self._tracks):
            track = self._tracks[tid]
            state, covariance = self._predict(track.state, track.covariance)
            point = measured.pop(tid, None)
            if point is None:
                if track.missing + 1 > self.max_coast:
                    del self._tracks[tid]
                    logger.trace("update: id=%s dropped after %d coasted frames",
                                 tid, track.missing + 1)
                else:
                    self._tracks[tid] = _Track(state, covariance, track.missing + 1)
                    logger.trace("update: id=%s coasting (%d/%d)",
                                 tid, track.missing + 1, self.max_coast)
                continue
            state, covariance = self._correct(state, covariance, point)
            self._tracks[tid] = _Track(state, covariance, 0)
            positions[tid] = (float(state[0]), float(state[1]))
            logger.trace("update: id=%s measured=(%.2f, %.2f) filtered=(%.2f, %.2f)",
                         tid, point[0], point[1], state[0], state[1])

        for tid, point in measured.items():
            state = np.array([point[0], point[1], 0.0, 0.0])
            self._tracks[tid] = _Track(state, np.eye(4) * self.initial_covariance, 0)
            positions[tid] = (float(point[0]), float(point[1]))
            logger.trace("update: id=%s new track at (%.2f, %.2f)", tid, point[0], point[1])

        on_pitch, off_pitch = {}, set()
        for obj in tracked_objects:
            tid = int(obj["id"])
            point = positions.get(tid)
            if inside_pitch(point, self.pitch_length, self.pitch_width):
                on_pitch[tid] = point
            else:
                off_pitch.add(tid)
        logger.debug(
            "update: %d detected, %d projectable, %d tracks; on pitch %s, off pitch %s",
            len(tracked_objects), n_projectable, len(positions),
            sorted(on_pitch), sorted(off_pitch),
        )
        return on_pitch, off_pitch

    def reset(self) -> None:
        """Forget every track. Call on a video cut."""
        self._tracks.clear()
        logger.debug("reset: all player position tracks cleared")

    def _predict(self, state, covariance):
        state = _TRANSITION @ state
        covariance = _TRANSITION @ covariance @ _TRANSITION.T + self._q
        return state, covariance

    def _correct(self, state, covariance, point):
        innovation = point - _OBSERVATION @ state
        innovation_covariance = _OBSERVATION @ covariance @ _OBSERVATION.T + self._r
        gain = covariance @ _OBSERVATION.T @ np.linalg.inv(innovation_covariance)
        state = state + gain @ innovation
        covariance = (np.eye(4) - gain @ _OBSERVATION) @ covariance
        return state, covariance
