"""Where a tracked player stands: the foot point and its temporal filter.

A player's position on the pitch is read off the frame every frame, and the
pitch position is what the BEV draws and what the on-pitch filter judges. Both
steps used to be done inline, in three places, from the raw detection.

Foot point
----------
The bottom-centre of the bounding box, in float. The obvious alternative -- the
lowest pixel of the segmentation mask -- is *worse*, measured against real
detections on ``input_vids/test1.mp4`` with the camera frozen so it cannot
contribute: the mask bottom moved 0.170 m per frame against the box bottom's
0.148 m, and put 8.6% of its steps past the 0.32 m/frame a human can run
against 6.2%. YOLO's box is a regression-head output and is smooth; the mask
bottom is a thresholded contour, where one stray pixel moves the foot by a
whole row -- 0.6 m at the far goal line. So the box stays, minus the ``int()``
truncation, which only cost half a pixel of quantisation.

Filter
------
A per-track constant-velocity Kalman filter over the world position. Filtering
in world metres rather than in pixels is deliberate: the projection magnifies a
pixel of foot jitter into 0.12 m at the near touchline and 0.61 m at the far
goal line, and this is the only stage that sees that magnification. Measured on
the same 200 frames it cuts the frame-to-frame jitter (mean |second difference|
of the BEV position) from 0.148 m to 0.084 m, and the share of steps that are
physically impossible from 6.2% to 3.0%.

The filter is causal on purpose. A centred window is a little better again
(centred median-3: 0.105 m) but needs lookahead, and every causal alternative
that smooths harder pays in lag: on a clean 7 m/s run EMA(0.5) sits 0.276 m
behind the player and EMA(0.3) 0.638 m, which is the size of the jitter being
removed -- except that a lag is a bias and does not average out. The
constant-velocity model tracks the same run to 0.008 m.

``process_noise`` and ``measurement_noise`` come from a sweep over both the real
detections and the clean-motion reference: 0.05 / 0.25 is the knee, and 0.25 m²
(0.5 m) is also about the measured p90 frame-to-frame step, so it is not an
arbitrary number.

An innovation gate was tried and dropped: at chi-square 0.99 it rejected 1
sample in 4574 and changed the jitter by 0.0006 m. The linear update already
handles this noise, and a gate is one more thing that can reject a genuine
sprint.
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
    """One player's filter state, plus how long since they were last detected."""

    state: np.ndarray
    covariance: np.ndarray
    missing: int


def foot_point(obj) -> tuple:
    """Ground-contact pixel of a detection: the bottom-centre of its box.

    Float, not int: the box bottom is already a continuous coordinate, and
    truncating it to a pixel index throws away half a pixel for nothing. At the
    far goal line half a pixel is 0.3 m.

    Args:
        obj: one entry of ``PlayerTracker.update``'s return value.

    Returns:
        ``(u, v)`` in frame pixels.
    """
    x1, _, x2, y2 = (float(v) for v in obj["bbox"])
    return (x1 + x2) / 2.0, y2


class PlayerPositionFilter:
    """Per-track Kalman filter over players' BEV positions.

    One instance handles one continuous shot; :meth:`reset` on a video cut.
    :meth:`update` must be called exactly once per frame, in order.

    Args:
        process_noise: variance added to every state each frame.
        measurement_noise: variance of the ground position a detection implies,
            in m². 0.25 (0.5 m) matches the measured p90 frame-to-frame step.
        initial_covariance: state covariance a track starts with. Loose on
            purpose, so a new track follows its first detections rather than
            the zero velocity it is initialised with.
        max_coast: frames a track may go undetected before it is forgotten.
            BoT-SORT keeps track ids through roughly this much occlusion.
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
            ``{track_id: (x, y)}`` in world metres, for the tracks detected on
            this frame only. A track that was not detected keeps its (coasting)
            state for the next frame but is not reported -- the caller draws
            what the detector saw, and a coasted prediction would be a position
            with nothing on screen to justify it.
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
