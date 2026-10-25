from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.stats import chi2
from core.projection_utils import pixel_to_ground, pixel_to_3d

PITCH_HALF_LENGTH = 105.0 / 2
PITCH_HALF_WIDTH = 68.0 / 2

TENTATIVE = "TENTATIVE"
TRACKING = "TRACKING"
OCCLUDED = "OCCLUDED"
LOST = "LOST"

FALLBACK_STATIC_MARKS = ((0.0, 0.0), (41.5, 0.0), (-41.5, 0.0))


@dataclass(frozen=True)
class BallCandidate:
    # pixel
    u: float
    v: float
    # BEV world metres; None if unprojectable
    x: float | None
    y: float | None
    width_px: float
    height_px: float
    conf: float
    z_est: float | None = None # pixel_to_3d z; diagnostics only


@dataclass(frozen=True)
class BallTrackerConfig:
    """Every tunable in one place; Q/R stay at their base values this round."""

    chi2_thres: float = 0.95
    n_occ_s: float = 0.5           # -> N_occ = max(1, round(fps * n_occ_s))
    n_drop_s: float = 4.0          # -> N_drop = max(N_occ, round(fps * n_drop_s))
    gate_max_m: float = 8.0        # Euclidean cap on the tier-1 gate, on top of chi2
    gate_margin_m: float = 1.0     # slack on the physical (tier-2) radius
    reacquire_after: int = 2       # frames since the last update before reacquiring
    reacquire_confirm: int = 2     # candidates needed to confirm a reacquisition
    reacquire_confirm_window: int = 3  # frames allowed between those candidates
    reacquire_radius_m: float = 3.0
    static_radius_m: float = 0.75
    pitch_margin_m: float = 1.0
    foot_band_frac: float = 0.25
    ball_max_speed_mps: float = 45.0
    process_noise_xy: float = 0.05
    process_noise_v: float = 6.0


class BallDetector:
    def __init__(self, model_path, device):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame):
        detections = self.model(
            frame,
            conf=0.15,
            iou=0.45,
            imgsz=1280,
            classes=[0],
            device=self.device,
            verbose=False
        )
        return detections

    def get_candidates(self, detections, K, R, t):
        """Turn YOLO output into candidates with BEV ground positions.
        the tracker uses z=0 projection.
        """
        if not detections:
            return []
        result = detections[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [1.0] * len(boxes)

        candidates = []
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = (float(v) for v in box)
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            u, v = (x1 + x2) / 2, (y1 + y2) / 2

            ground = pixel_to_ground(u, v, K, R, t)
            if ground is None or not np.isfinite(ground).all():
                gx = gy = None
            else:
                gx, gy = float(ground[0]), float(ground[1])
                if not (np.isfinite(gx) and np.isfinite(gy)):
                    gx = gy = None

            z_est = None
            try:
                point3d = pixel_to_3d(u, v, width, K, R, t)
            except (np.linalg.LinAlgError, ValueError):
                point3d = None
            if point3d is not None and np.isfinite(point3d).all():
                z_est = float(point3d[2])

            candidates.append(BallCandidate(
                u=u, v=v, x=gx, y=gy,
                width_px=width, height_px=height,
                conf=float(conf), z_est=z_est,
            ))
        return candidates


FRAME_FIELDS = [
    "frame_idx", "calib_score", "calib_reinit", "calib_frames_lost", "calib_valid",
    "state", "miss_count", "reject_count", "reacq_count",
    "n_raw", "n_plausible", "n_candidates", "accepted",
    "best_d2", "best_dist_m", "gate_chi2", "gate_m",
    "stat_gate_m", "phys_gate_m",
    "pred_x", "pred_y", "meas_x", "meas_y", "innov_x", "innov_y",
]

CANDIDATE_FIELDS = [
    "frame_idx", "cand_idx", "u", "v", "x", "y", "width_px", "height_px",
    "conf", "z_est", "d2", "dist_m", "plausible", "tier", "accepted", "reject_reason",
]

class BallTracker:
    """TENTATIVE -> TRACKING <-> OCCLUDED -> LOST state machine.

    Counters are the only state; the state itself is derived at the end of
    each frame.  The Kalman filter predicts exactly once per frame.
    """

    def __init__(
        self,
        fps,
        config: BallTrackerConfig | None = None,
        static_marks=None,
        chi2_thres=None,
    ):
        if chi2_thres is not None and config is None:
            config = BallTrackerConfig(chi2_thres=chi2_thres)
        self.config = config or BallTrackerConfig()
        self.fps = float(fps) if fps else 25.0
        self.dt = 1.0 / self.fps
        self.n_occ = max(1, round(self.fps * self.config.n_occ_s))
        self.n_drop = max(self.n_occ, round(self.fps * self.config.n_drop_s))
        self.gate_threshold = chi2.ppf(self.config.chi2_thres, df=2)

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = self.dt
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        self.kf.Q = np.diag([
            self.config.process_noise_xy, self.config.process_noise_xy,
            self.config.process_noise_v, self.config.process_noise_v,
        ]).astype(np.float64)
        self.kf.R = np.eye(2) * 0.25

        self.state = TENTATIVE
        self.miss_count = 0        # frames since the last kf.update()
        self.reject_count = 0      # consecutive plausible-but-not-gated frames
        self.reacq_count = 0       # candidates seen since the confirm anchor
        self._dropped = False      # track discarded after N_drop
        self._frame_counter = 0    # process_frame() calls, for time windows

        self._last_accepted = None     # (x, y) of the last accepted measurement
        self._confirm_anchor = None    # (x, y) of the first candidate in a run
        self._confirm_entry = None     # the anchor candidate itself
        self._confirm_frame = None     # frame counter when the anchor was set

        marks = list(static_marks) if static_marks else list(FALLBACK_STATIC_MARKS)
        self.static_marks = [(float(x), float(y)) for x, y in marks]

    # ---------------------------------------------------------------- public

    def process_frame(
        self,
        candidates,
        player_boxes=None,
        frame_idx=None,
        calib_score=None,
        calib_reinit=False,
        calib_frames_lost=None,
        calib_valid=True,
    ):
        """Advance one frame. Returns the BEV (x, y) to draw, or None."""
        candidates = list(candidates or [])
        player_boxes = self._normalise_boxes(player_boxes)
        self._frame_counter += 1
        self._stat_gate_m = None
        self._phys_gate_m = None

        # Static marks / foot bands only gate new tracks here. While OCCLUDED
        # the prediction is still meaningful, so a motion-consistent candidate
        # must not be thrown away by those priors; they are applied to tier-2
        # recovery candidates inside _gate_and_update instead.
        acquisition = self.state in (TENTATIVE, LOST)
        entries, n_plausible = self._build_entries(candidates, acquisition, player_boxes)

        pred = None
        best = None
        accepted = False

        # Predict exactly once per frame; TENTATIVE has no track yet.
        if self.state != TENTATIVE:
            pred = self._predict()

        if not calib_valid:
            # Calibration is stale: advance the track without trusting pixels.
            for entry in entries:
                entry["reason"] = "calib_invalid"
            self.reject_count = 0
            self._reset_confirm()
            if self.state != TENTATIVE:
                self.miss_count += 1
                self.state = OCCLUDED if self.miss_count < self.n_occ else LOST
                if self.state == LOST and self.miss_count >= self.n_drop:
                    self._dropped = True

        elif self.state == TENTATIVE:
            best = self._try_confirm([e for e in entries if e["plausible"]])
            if best is not None:
                accepted = True
                self._last_accepted = (best["x"], best["y"])
                self.state = TRACKING

        else:
            if self.state in (TRACKING, OCCLUDED):
                best, accepted, recoverable = self._gate_and_update(entries, pred)
                if not accepted:
                    if any(e["plausible"] for e in entries):
                        self.reject_count += 1
                    else:
                        self.reject_count = 0
                    self.miss_count += 1
                    self.state = OCCLUDED if self.miss_count < self.n_occ else LOST

                    # Physics-plausible candidates start the confirmation run;
                    # once LOST, the prediction is stale enough that any
                    # filtered candidate is a reacquisition candidate.
                    if self.miss_count >= self.config.reacquire_after:
                        if self.state == LOST:
                            recoverable = [e for e in entries if e["plausible"]]
                        best, accepted = self._try_reacquire(recoverable)
                        if accepted:
                            self._last_accepted = (best["x"], best["y"])

            elif self.state == LOST:
                best, accepted = self._try_reacquire(
                    [e for e in entries if e["plausible"]])
                if accepted:
                    self._last_accepted = (best["x"], best["y"])
                else:
                    if any(e["plausible"] for e in entries):
                        self.reject_count += 1
                    else:
                        self.reject_count = 0
                    self.miss_count += 1
                    if self.miss_count >= self.n_drop:
                        self._dropped = True

        if accepted:
            self.miss_count = 0
            self.reject_count = 0
            self._dropped = False

        position = self._output_position()
        return position

    def get_position(self):
        """Current BEV position, or None with no confirmed / dropped track."""
        if self.state == TENTATIVE or self._dropped:
            return None
        return float(self.kf.x[0].item()), float(self.kf.x[1].item())

    def get_velocity(self):
        """Kalman filter velocity (vx, vy) in m/s, or None without a track."""
        if self.get_position() is None:
            return None
        return float(self.kf.x[2].item()), float(self.kf.x[3].item())

    # ------------------------------------------------------------- decisions

    def _predict(self):
        if self.state == LOST and (self._dropped or self.miss_count >= self.n_drop):
            return None
        self.kf.predict()
        self.kf.P = (self.kf.P + self.kf.P.T) / 2
        if not (np.isfinite(self.kf.x).all() and np.isfinite(self.kf.P).all()):
            self._invalidate()
            return None
        return float(self.kf.x[0].item()), float(self.kf.x[1].item())

    def _gate_and_update(self, entries, pred):
        """Tier 1 updates the KF; tier 2 is only eligible for reacquisition.

        A covariance gate alone rejects a correctly detected fast ball as soon
        as the filter has coasted for a few frames, so a physical bound (max
        ball speed over the time since the last update) keeps those candidates
        alive. They never update the filter directly: a single false positive
        must not teleport the track.
        """
        if pred is None:
            return None, False, []
        elapsed = max(self.miss_count + 1, 1) * self.dt
        self._phys_gate_m = self.config.ball_max_speed_mps * elapsed + self.config.gate_margin_m
        self._stat_gate_m = self._statistical_gate_radius()

        passing = []
        recoverable = []
        for entry in entries:
            if not entry["plausible"]:
                continue
            d2, dist = self._innovation_stats(entry["x"], entry["y"])
            entry["d2"], entry["dist_m"] = d2, dist
            if not np.isfinite(d2):
                entry["reason"] = "nan_d2"
                if dist <= self._phys_gate_m:
                    entry["tier"] = 2
                    recoverable.append(entry)
            elif d2 < self.gate_threshold and dist <= self.config.gate_max_m:
                entry["tier"] = 1
                passing.append(entry)
            elif dist <= self._phys_gate_m and entry["acq_reason"] is not None:
                # Physically reachable, but a static mark or a player's feet:
                # too weak to rebuild a track on.
                entry["reason"] = entry["acq_reason"]
            elif dist <= self._phys_gate_m:
                entry["tier"] = 2
                entry["reason"] = "gate_recoverable"
                recoverable.append(entry)
            else:
                entry["reason"] = "gate"

        if not passing:
            return None, False, recoverable

        best = min(passing, key=lambda e: e["d2"])
        self.kf.update(np.array([best["x"], best["y"]], dtype=np.float64))
        if not (np.isfinite(self.kf.x).all() and np.isfinite(self.kf.P).all()):
            self._invalidate()
            return None, False, []
        best["accepted"] = 1
        best["reason"] = "accepted"
        self.state = TRACKING
        self._reset_confirm()
        return best, True, []

    def _statistical_gate_radius(self):
        S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
        return float(np.sqrt(self.gate_threshold * max(float(np.trace(S)) / 2.0, 0.0)))

    def _try_reacquire(self, entries):
        """Confirm a candidate over frames, then hard-reset the filter."""
        plausible = [e for e in entries if e["plausible"]]
        if self._last_accepted is not None:
            elapsed = max(self.miss_count, 1) * self.dt
            ax, ay = self._last_accepted
            plausible = [
                e for e in plausible
                if np.hypot(e["x"] - ax, e["y"] - ay) / elapsed
                <= self.config.ball_max_speed_mps
            ]
        if not plausible and self._confirm_anchor is None:
            return None, False
        # An active run survives empty frames; _confirm() expires it once the
        # confirmation window is exceeded.
        confirmed = self._confirm(plausible)
        if confirmed is None:
            return None, False
        self._set_track(confirmed)
        return confirmed, True

    def _try_confirm(self, plausible):
        if not plausible and self._confirm_anchor is None:
            return None
        confirmed = self._confirm(plausible)
        if confirmed is None:
            return None
        self._set_track(confirmed)
        return confirmed

    def _confirm(self, entries):
        """Confirm a candidate over a small frame window, around an anchor.

        Missing a frame does not discard the run: detectors drop the ball for
        one or two frames constantly. The run is only dropped once it has been
        quiet for longer than ``reacquire_confirm_window``, or when the
        candidate jumps outside the radius for the elapsed time.
        """
        now = self._frame_counter
        if self._confirm_anchor is None:
            if not entries:
                return None
            self._start_confirm(max(entries, key=lambda e: e["conf"]), now)
            return self._confirm_entry if self.config.reacquire_confirm <= 1 else None

        gap = now - self._confirm_frame
        if gap > self.config.reacquire_confirm_window:
            self._reset_confirm()
            if entries:
                self._start_confirm(max(entries, key=lambda e: e["conf"]), now)
            return None
        if not entries:
            return None

        radius = max(
            self.config.reacquire_radius_m,
            self.config.ball_max_speed_mps * gap * self.dt * 1.5,
        )
        ax, ay = self._confirm_anchor
        nearby = [e for e in entries if np.hypot(e["x"] - ax, e["y"] - ay) <= radius]
        if not nearby:
            self._reset_confirm()
            self._start_confirm(max(entries, key=lambda e: e["conf"]), now)
            return None
        chosen = min(nearby, key=lambda e: np.hypot(e["x"] - ax, e["y"] - ay))
        self.reacq_count += 1
        if self.reacq_count < self.config.reacquire_confirm:
            return None
        return chosen

    def _start_confirm(self, entry, frame):
        self._confirm_anchor = (entry["x"], entry["y"])
        self._confirm_entry = entry
        self._confirm_frame = frame
        self.reacq_count = 1

    def _set_track(self, entry):
        """Hard reset the KF to a confirmed measurement (init or reacquire)."""
        entry["accepted"] = 1
        entry["reason"] = "accepted"
        entry["tier"] = max(entry.get("tier", 0), 2)
        vx = vy = 0.0
        if self._confirm_anchor is not None and self.reacq_count > 1:
            ax, ay = self._confirm_anchor
            elapsed = (self._frame_counter - self._confirm_frame) * self.dt
            if elapsed > 0:
                vx = (entry["x"] - ax) / elapsed
                vy = (entry["y"] - ay) / elapsed
        limit = self.config.ball_max_speed_mps
        vx = float(np.clip(vx, -limit, limit))
        vy = float(np.clip(vy, -limit, limit))
        self.kf.x = np.array([[entry["x"]], [entry["y"]], [vx], [vy]], dtype=np.float64)
        self.kf.P = np.eye(4, dtype=np.float64) * 0.1
        self._dropped = False
        self.state = TRACKING
        self.miss_count = 0
        self.reject_count = 0
        self._reset_confirm()

    def _invalidate(self):
        self.state = LOST
        self._dropped = True
        self.miss_count = max(self.miss_count, self.n_drop)
        self._reset_confirm()

    def _reset_confirm(self):
        self.reacq_count = 0
        self._confirm_anchor = None
        self._confirm_entry = None
        self._confirm_frame = None

    def _output_position(self):
        if self.state == TRACKING:
            return self.get_position()
        if self.state == OCCLUDED and not self._dropped:
            return float(self.kf.x[0].item()), float(self.kf.x[1].item())
        return None

    def _innovation_stats(self, x, y):
        innovation = np.array([[x], [y]], dtype=np.float64) - self.kf.H @ self.kf.x
        S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
        S = (S + S.T) / 2
        dist = float(np.hypot(innovation[0, 0], innovation[1, 0]))
        try:
            solved = np.linalg.solve(S, innovation)
        except np.linalg.LinAlgError:
            return np.inf, dist
        d2 = float(np.asarray(innovation.T @ solved).item())
        return d2, dist

    # -------------------------------------------------------------- filtering

    def _filter(self, cand, acquisition, player_boxes):
        """Return (passes_always_on, plausible, reason, acquisition_reason)."""
        if cand.x is None or cand.y is None:
            return False, False, "unprojectable", None
        if not (np.isfinite(cand.x) and np.isfinite(cand.y)):
            return False, False, "unprojectable", None
        margin = self.config.pitch_margin_m
        if (abs(cand.x) > PITCH_HALF_LENGTH + margin
                or abs(cand.y) > PITCH_HALF_WIDTH + margin):
            return False, False, "outside_pitch", None
        if (not np.isfinite(cand.width_px) or not np.isfinite(cand.height_px)
                or cand.width_px <= 0 or cand.height_px <= 0):
            return False, False, "bbox_shape", None
        if not (2.0 <= cand.width_px <= 80.0):
            return False, False, "bbox_shape", None
        if not (0.5 <= cand.height_px / cand.width_px <= 2.0):
            return False, False, "bbox_shape", None

        acq_reason = self._acquisition_reason(cand, player_boxes)
        if acquisition and acq_reason is not None:
            return True, False, acq_reason, acq_reason
        return True, True, "none", acq_reason

    def _acquisition_reason(self, cand, player_boxes):
        for mx, my in self.static_marks:
            if np.hypot(cand.x - mx, cand.y - my) < self.config.static_radius_m:
                return "static_mark"
        if self._in_foot_band(cand.u, cand.v, player_boxes):
            return "foot_band"
        return None

    def _in_foot_band(self, u, v, player_boxes):
        frac = self.config.foot_band_frac
        for x1, y1, x2, y2 in player_boxes:
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            top = y2 - frac * height
            if (x1 - 0.15 * width) <= u <= (x2 + 0.15 * width) and top <= v <= y2:
                return True
        return False

    @staticmethod
    def _normalise_boxes(boxes):
        if boxes is None:
            return []
        array = np.asarray(boxes, dtype=np.float64)
        if array.size == 0:
            return []
        array = array.reshape(-1, 4)
        return array[np.isfinite(array).all(axis=1)]

    def _build_entries(self, candidates, acquisition, player_boxes):
        entries = []
        n_plausible = 0
        for cand in candidates:
            always_ok, plausible, reason, acq_reason = self._filter(
                cand, acquisition, player_boxes)
            if always_ok:
                n_plausible += 1
            entries.append({
                "u": cand.u, "v": cand.v, "x": cand.x, "y": cand.y,
                "width_px": cand.width_px, "height_px": cand.height_px,
                "conf": cand.conf, "z_est": cand.z_est,
                "plausible": int(plausible), "accepted": 0, "tier": 0,
                "reason": reason, "acq_reason": acq_reason,
                "d2": None, "dist_m": None,
            })
        return entries, n_plausible