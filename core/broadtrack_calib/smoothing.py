"""Zero-phase temporal smoothing of the calibration sequence.

Why
---
BroadTrack re-solves pan/tilt/roll/focal from scratch every frame with no
temporal prior on them (only the pan gets a local IoU refinement), and
``PointExtractor`` re-draws its sample points at random on every frame. The
result is a per-frame optimum that wanders: measured on ``input_vids/test1.mp4``
the pose of a *fixed* image point moves by ~4 cm per frame at the near touchline
and ~16 cm at the far goal line, even though the camera is on a tripod.

A causal filter cannot fix this without a real cost. The camera genuinely pans,
and a one-sided average always lags it: the built-in
``BROADTRACK_PAN_SMOOTH_WINDOW=5`` median filter was measured to make things
*worse* (line-IoU minimum 0.63 -> 0.55 over 130 frames) for exactly that reason.
So the smoothing here is zero-phase -- a symmetric Savitzky-Golay window
centred on the frame being emitted.

The price is lookahead: the result for frame ``t`` needs frames up to ``t + lag``.
This module therefore *delays* the whole calibration stream by ``lag`` frames
(``lag = window // 2``, 8 frames = 0.32 s at 25 fps). That is free in an offline
pipeline -- the caller renders frame ``t - lag`` when ``estimate`` hands back the
calibration for it -- and unacceptable in a live one.

What is smoothed
----------------
Only the four scalars that are actually noisy: ``pan_deg``, ``tilt_deg``,
``roll_deg`` and ``focal_length``. The camera position and ``k1`` are anchored in
C++ and are constant, and the principal point is a property of the frame size.
``K``, ``R`` and ``P`` are rebuilt from the smoothed scalars with
:func:`pose_from_parameters`, which reproduces the C++ convention exactly
(verified against the extension in ``tests/test_calib_smoothing.py``).

Frames where the tracker re-initialised are not smoothed across: a re-init is a
deliberate pose discontinuity, and blending it with the poses before it would
smear the jump over the whole window instead of removing it.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np
from scipy.signal import savgol_coeffs

#: The camera parameters the smoother filters. Everything else in the result
#: dict describes the frame (``score``, ``line_mask`` ...) or is constant
#: (``t``, ``k1``) and is passed through untouched.
SMOOTHED_KEYS = ("pan_deg", "tilt_deg", "roll_deg", "focal_length")


def pose_from_parameters(result: dict) -> tuple:
    """Rebuild ``(K, R, t, P)`` from the scalars in a calibration dict.

    Mirrors ``Camera::getOrientationFromPanTiltRoll`` /
    ``Camera::setPanTiltRoll``: the C++ builds ``M = Rp * Rt * Rr`` and returns
    ``M.transpose()`` as the world->camera rotation, which is what ``R`` is
    here. Focal length and principal point form the intrinsic matrix, and
    ``P = K [R | -R t]`` as everywhere else in this package.

    ``k1`` is *not* folded in: the returned ``K`` describes the distorted image,
    exactly like BroadTrack's own output. Pass ``k1`` to
    :func:`core.projection_utils.pixel_to_ground` separately.
    """
    pan, tilt, roll = np.radians(
        [result["pan_deg"], result["tilt_deg"], result["roll_deg"]]
    )
    cos_p, sin_p = np.cos(pan), np.sin(pan)
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    cos_r, sin_r = np.cos(roll), np.sin(roll)

    rot_tilt = np.array([[1.0, 0.0, 0.0],
                         [0.0, cos_t, -sin_t],
                         [0.0, sin_t, cos_t]])
    rot_pan = np.array([[cos_p, -sin_p, 0.0],
                        [sin_p, cos_p, 0.0],
                        [0.0, 0.0, 1.0]])
    rot_roll = np.array([[cos_r, -sin_r, 0.0],
                         [sin_r, cos_r, 0.0],
                         [0.0, 0.0, 1.0]])
    R = (rot_pan @ rot_tilt @ rot_roll).T

    focal = float(result["focal_length"])
    cx = float(result["K"][0, 2])
    cy = float(result["K"][1, 2])
    K = np.array([[focal, 0.0, cx],
                  [0.0, focal, cy],
                  [0.0, 0.0, 1.0]])

    t = np.asarray(result["t"], dtype=np.float64)
    It = np.eye(4, dtype=np.float64)[:3]
    It[:, 3] = -t
    P = K @ (R @ It)
    return K, R, t, P


class CalibrationSmoother:
    """Delay line + symmetric Savitzky-Golay filter over a calibration stream.

    One instance handles one continuous shot; :meth:`reset` on a video cut.
    ``push`` must be called exactly once per frame, in order.

    Args:
        window: odd filter length, in frames. 1 disables smoothing entirely.
        polyorder: degree of the local polynomial the window fits. 3 keeps a
            pan that is accelerating (a constant-velocity fit would bias it).
    """

    def __init__(self, window: int = 17, polyorder: int = 3):
        if window < 1:
            raise ValueError("window must be >= 1")
        if window % 2 == 0:
            raise ValueError("window must be odd, so the filter has a centre frame")
        if not 0 < polyorder < window:
            raise ValueError("polyorder must be >= 1 and < window")
        self.window = int(window)
        self.polyorder = int(polyorder)
        #: Frames of lookahead the caller must delay its rendering by.
        self.lag = self.window // 2

        self._coefficients = savgol_coeffs(self.window, self.polyorder)
        self._last_params: Optional[np.ndarray] = None
        self._history: deque = deque(maxlen=self.window)
        self._queue: deque = deque()
        self._seen = 0
        #: Earliest frame index whose samples may enter a window, or ``None``
        #: while there is no usable sample yet. Moved forward on every
        #: re-initialisation so a deliberate pose jump is never blended.
        self._valid_from: Optional[int] = None

    def push(self, result: Optional[dict]) -> Optional[dict]:
        """Feed the raw calibration for the next frame.

        Returns the calibration for the frame ``lag`` back, or ``None`` until
        ``lag`` frames have been seen. A ``None`` result (a solve the tracker
        refused) is carried through the delay line, so the caller still learns
        about that frame -- just late.
        """
        index = self._seen
        self._seen += 1

        # The history is what gets filtered, so a refused solve carries the
        # previous frame's pose forward instead of punching a hole in the
        # window. The queue keeps the None so the caller still sees it.
        if result is not None:
            self._last_params = np.array(
                [result[key] for key in SMOOTHED_KEYS], dtype=np.float64
            )
            if self._valid_from is None or result.get("reinit"):
                self._valid_from = index
        self._history.append(self._last_params)
        self._queue.append(result)

        if len(self._queue) <= self.lag:
            return None
        entry = self._queue.popleft()
        if entry is None:
            return None

        window_start = index - self.window + 1
        if (
            self._valid_from is None
            or len(self._history) < self.window
            or window_start < self._valid_from
        ):
            # Warm-up, or too close to a re-initialisation for a symmetric
            # window: hand back the unsmoothed solve rather than a one-sided fit.
            return entry

        params = self._coefficients @ np.stack(self._history)
        return self._with_parameters(entry, params)

    def flush(self) -> list:
        """Drain the delay line at the end of a video.

        The tail has no future frames left to average over, so these come back
        unsmoothed -- the mirror image of the warm-up at the start. One entry
        per leftover frame, ``None`` included, so a caller pairing frames with
        results stays aligned.
        """
        remaining = list(self._queue)
        self._queue.clear()
        return remaining

    def reset(self) -> None:
        """Forget everything. Call on a video cut."""
        self._history.clear()
        self._queue.clear()
        self._seen = 0
        self._valid_from = None
        self._last_params = None

    @staticmethod
    def _with_parameters(entry: dict, params: np.ndarray) -> dict:
        """A copy of ``entry`` with the smoothed scalars and rebuilt K/R/P."""
        out: dict[str, Any] = dict(entry)
        for key, value in zip(SMOOTHED_KEYS, params):
            out[key] = float(value)
        out["K"], out["R"], out["t"], out["P"] = pose_from_parameters(out)
        return out
