"""
Temporal smoothing of per-frame camera calibration to remove jitter.

The per-frame PnP calibration is independently re-estimated every frame, which
introduces frame-to-frame jitter in the camera pose (R, t) and intrinsics (K).
For the mostly-static broadcast main camera this jitter is pure noise and gets
amplified by ``pixel_to_ground`` for points far from the camera.

This module smooths the camera parameters across time with a One-Euro filter
(Casiez et al. 2012) — a low-latency adaptive low-pass filter — and resets the
filter on discontinuities (hard cuts / large camera motion) so a new shot is
adopted immediately instead of being smeared into the previous one.

Smoothing is done in a stable parameter space:
  - pan / tilt / roll   (Euler angles, rad) via ``rotation_matrix_to_pan_tilt_roll``
  - position (x, y, z)  (camera center, meters)
  - focal length fx, fy (pixels)
The smoothed parameters are rebuilt back into a ``cam_params`` dict with
``pan_tilt_roll_to_orientation(...).T`` while keeping the original principal
point and distortion coefficients.
"""

import numpy as np

from .utils.utils_calib import (
    pan_tilt_roll_to_orientation,
    rotation_matrix_to_pan_tilt_roll,
)

# Channels whose value is an angle in radians (unwrapped relative to the
# previous filtered value so a pan crossing +/-pi is not seen as a jump).
_ANGLE_CHANNELS = ("pan", "tilt", "roll")

# Per-channel scale used to normalize the One-Euro ``beta`` term so a single
# beta is meaningful across radians / meters / pixels.
_SCALE = {
    "pan": 1.0,     # rad
    "tilt": 1.0,    # rad
    "roll": 0.2,    # rad (kept small by the camera operator)
    "fx": 1000.0,   # px
    "fy": 1000.0,   # px
    "tx": 100.0,    # m
    "ty": 100.0,    # m
    "tz": 30.0,     # m (camera height)
}
_ALL_CHANNELS = tuple(_SCALE.keys())


class OneEuro:
    """One-Euro filter (Casiez et al. 2012) for a single scalar signal."""

    def __init__(self, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    @staticmethod
    def _alpha(cutoff, dt):
        # alpha = r / (r + 1), r = 2*pi*cutoff*dt (equivalent to the tau form)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, dt):
        if dt <= 0:
            return self.x_prev
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def reset(self, x0):
        self.x_prev = float(x0)
        self.dx_prev = 0.0


class CalibrationSmoother:
    """Smooth the per-frame ``cam_params`` dict across time.

    Args:
        dt: sampling period in seconds (1 / fps).
        min_cutoff: minimum cutoff frequency (Hz) for the One-Euro filters.
        beta: velocity coefficient — how fast the cutoff rises with speed.
            Higher beta = less lag during real camera motion but keeps more
            noise; lower beta = stronger jitter removal but lags pans/zooms.
        d_cutoff: cutoff (Hz) for the derivative low-pass.
        reset_pan_tilt_deg: per-frame pan/tilt change (deg) above which the
            smoother treats the frame as a cut and re-seeds.
        reset_position_m: per-frame camera-center displacement (m) above which
            the smoother re-seeds.
    """

    def __init__(self, dt, min_cutoff=1.0, beta=5.0, d_cutoff=1.0,
                 reset_pan_tilt_deg=5.0, reset_position_m=15.0):
        self.dt = float(dt)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset_pan_tilt_deg = reset_pan_tilt_deg
        self.reset_position_m = reset_position_m
        self._filters = None
        self._prev_raw = None
        self._prev_out = None
        self.n_resets = 0  # number of cut/large-motion resets performed

    def _init_filters(self, vec):
        # Seed each filter with the *normalized* value (matching what filter() is
        # fed), while _prev_out keeps raw units for angle unwrapping.
        self._filters = {
            name: OneEuro(vec[name] / _SCALE[name], self.min_cutoff, self.beta, self.d_cutoff)
            for name in _ALL_CHANNELS
        }
        self._prev_out = dict(vec)

    @staticmethod
    def _to_vector(cam_params):
        R = np.array(cam_params["rotation_matrix"], dtype=np.float64)
        pos = np.array(cam_params["position_meters"], dtype=np.float64).reshape(3)
        pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(R)
        return {
            "pan": float(pan), "tilt": float(tilt), "roll": float(roll),
            "fx": float(cam_params["x_focal_length"]),
            "fy": float(cam_params["y_focal_length"]),
            "tx": float(pos[0]), "ty": float(pos[1]), "tz": float(pos[2]),
        }

    @staticmethod
    def _to_cam_params(vec, raw):
        out = dict(raw)
        out["rotation_matrix"] = pan_tilt_roll_to_orientation(
            vec["pan"], vec["tilt"], vec["roll"]
        ).T.tolist()
        out["position_meters"] = [vec["tx"], vec["ty"], vec["tz"]]
        out["x_focal_length"] = vec["fx"]
        out["y_focal_length"] = vec["fy"]
        out["pan_degrees"] = np.rad2deg(vec["pan"])
        out["tilt_degrees"] = np.rad2deg(vec["tilt"])
        out["roll_degrees"] = np.rad2deg(vec["roll"])
        return out

    @staticmethod
    def _angle_diff(a, b):
        return (a - b + np.pi) % (2.0 * np.pi) - np.pi

    def _is_discontinuity(self, vec, prev):
        pan_delta = abs(self._angle_diff(vec["pan"], prev["pan"]))
        tilt_delta = abs(self._angle_diff(vec["tilt"], prev["tilt"]))
        max_angle_deg = max(np.rad2deg(pan_delta), np.rad2deg(tilt_delta))
        if max_angle_deg > self.reset_pan_tilt_deg:
            return True
        dp = np.array([vec["tx"] - prev["tx"],
                       vec["ty"] - prev["ty"],
                       vec["tz"] - prev["tz"]])
        if np.linalg.norm(dp) > self.reset_position_m:
            return True
        return False

    def smooth(self, cam_params):
        """Return a smoothed copy of ``cam_params`` (raw on first frame / cut)."""
        vec = self._to_vector(cam_params)

        if self._filters is None:
            self._init_filters(vec)
            self._prev_raw = vec
            return cam_params

        if self._is_discontinuity(vec, self._prev_raw):
            # Hard cut / large motion: adopt the new estimate and re-seed.
            self.n_resets += 1
            self._init_filters(vec)
            self._prev_raw = vec
            return cam_params

        out = {}
        for name in _ALL_CHANNELS:
            x = vec[name]
            if name in _ANGLE_CHANNELS:
                # Unwrap relative to the previous filtered value for continuity.
                prev_out = self._prev_out[name]
                while x - prev_out > np.pi:
                    x -= 2.0 * np.pi
                while x - prev_out < -np.pi:
                    x += 2.0 * np.pi
            x_norm = x / _SCALE[name]
            x_hat = self._filters[name].filter(x_norm, self.dt)
            out[name] = x_hat * _SCALE[name]

        self._prev_raw = vec
        self._prev_out = out
        return self._to_cam_params(out, cam_params)
