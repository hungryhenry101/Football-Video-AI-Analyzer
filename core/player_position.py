"""
Foot point: the box's float bottom-centre, not the mask's lowest pixel
KF: a per-track constant-velocity Kalman filter over the world position after projection.
"""

from __future__ import annotations

from typing import NamedTuple
import numpy as np

from core.projection_utils import pixel_to_ground

#: Constant-velocity model in world metres, one step per frame:
#: state = [x, y, vx, vy].
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
    ):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_covariance = float(initial_covariance)
        self.max_coast = int(max_coast)

        self._q = np.eye(4) * self.process_noise
        self._r = np.eye(2) * self.measurement_noise
        self._tracks: dict[int, _Track] = {}

    def update(self, tracked_objects, K, R, t) -> dict:
        """Filter this frame's detections into BEV positions.
        Args:
            tracked_objects: as returned by ``PlayerTracker.update``.
            K, R, t: the calibration for *this* frame.
        Returns:
            ``{track_id: (x, y)}`` in world metres, for the tracks detected on this frame only.
        """
        measured = {}
        for obj in tracked_objects:
            u, v = foot_point(obj)
            point = pixel_to_ground(u, v, K, R, t)
            if point is None or not np.isfinite(point).all():
                continue
            measured[int(obj["id"])] = np.asarray(point, dtype=np.float64)

        positions = {}
        for tid in list(self._tracks):
            track = self._tracks[tid]
            state, covariance = self._predict(track.state, track.covariance)
            point = measured.pop(tid, None)
            if point is None:
                if track.missing + 1 > self.max_coast:
                    del self._tracks[tid]
                else:
                    self._tracks[tid] = _Track(state, covariance, track.missing + 1)
                continue
            state, covariance = self._correct(state, covariance, point)
            self._tracks[tid] = _Track(state, covariance, 0)
            positions[tid] = (float(state[0]), float(state[1]))

        for tid, point in measured.items():
            state = np.array([point[0], point[1], 0.0, 0.0])
            self._tracks[tid] = _Track(state, np.eye(4) * self.initial_covariance, 0)
            positions[tid] = (float(point[0]), float(point[1]))

        return positions

    def reset(self) -> None:
        """Forget every track. Call on a video cut."""
        self._tracks.clear()

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
