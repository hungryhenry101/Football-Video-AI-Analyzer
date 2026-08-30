import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.stats import chi2
from core.pnl.projection_utils import pixel_to_ground


class BallDetector:
    def __init__(self, model_path, device):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame):
        detections = self.model(
            frame,
            conf=0.15,
            iou=0.45,
            imgsz=960,
            classes=[0],
            device=self.device,
            verbose=False
        )
        return detections

    def get_ball_centers(self, detections):
        if not detections:
            return []
        result = detections[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            centers.append((float(cx), float(cy), float(w)))
        return centers

    def project_to_ground(self, detections, K, R, t):
        """Project ball detections to ground plane (world meters) using PnL camera model."""
        centers = self.get_ball_centers(detections)
        ground_pts = []
        for cx, cy, _ in centers:
            pt = pixel_to_ground(cx, cy, K, R, t)
            if pt is not None:
                ground_pts.append((float(pt[0]), float(pt[1])))
        return ground_pts

class BallTracker:
    def __init__(self, fps, chi2_thres=0.95):
        if fps == 0: return
        self.fps = fps
        self.dt = 1.0 / self.fps
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # [x, y, vx, vy] in hg BEV (meters, m/s)
        # State Transition Matrix
        dt = self.dt
        self.kf.F = np.array([
            [1, 0, dt, 0], # x_pred = x_old + dt * vx_old
            [0, 1, 0, dt], # y_pred = y_old + dt * vy_old
            [0, 0, 1, 0],  # vx_pred = vx_old
            [0, 0, 0, 1]   # vy_pred = vy_old
        ], np.float32)

        # Measurement
        self.kf.H = np.array([
            [1, 0, 0, 0], # Zx
            [0, 1, 0, 0]  # Zy
        ], np.float32)

        # noise parameters in meter units
        self.kf.Q = np.eye(4) * 0.32  # process noise
        self.kf.R = np.eye(2) * 0.3  # measurement noise

        # gating threshold
        self.gate_threshold = chi2.ppf(chi2_thres, df=2)

        # state machine: UNINIT | INIT | NO_DET | GATE_FAIL | TRACKING
        self.state = "UNINIT"

    def initialize(self, x, y):
        self.kf.x = np.array([[x],[y],[0],[0]], dtype=np.float32)  # Current State
        self.kf.P = np.eye(4, dtype=np.float32) * 0.1  # Error Covariance
        self.state = "INIT"

    def process_frame(self, candidates):
        # main
        best = self.select_best_candidate(candidates)
        if best is not None:
            self.update(best[0], best[1])
        else:
            print("")
        return self.get_position()

    def select_best_candidate(self, candidates):
        """
        candidates: list of (x, y) detections in BEV coordinates after warp
        """
        if self.state == "UNINIT":
            # no gate on first frame
            if len(candidates) > 0:
                return candidates[0]
            return None

        if candidates is None:
            self.state = "NO_DET"
            return None

        self.predict()

        best_d2 = np.inf
        best_candidate = None

        for (x, y) in candidates:
            z = np.array([[x], [y]], dtype=np.float32)
            d2 = self.mahalanobis_distance_squared(z)
            if d2 < self.gate_threshold and d2 < best_d2:
                best_d2 = d2
                best_candidate = (x, y)

        if best_candidate is None:
            self.state = "GATE_FAIL"
        else:
            self.state = "TRACKING"

        return best_candidate


    def mahalanobis_distance_squared(self, z):
        """
        Compute d^2 = (z - Hx)^T S^{-1} (z - Hx)
        where S = H P H^T + R
        """
        H = self.kf.H
        x_pred = self.kf.x
        P_pred = self.kf.P
        R = self.kf.R

        y = z - H @ x_pred  # innovation
        S = H @ P_pred @ H.T + R
        # Compute quadratic form
        try:
            inv_S = np.linalg.inv(S)
            d2 = y.T @ inv_S @ y
            return float(d2)
        except np.linalg.LinAlgError:
            return np.inf

    def predict(self):
        if self.state == "UNINIT":
            return None
        self.kf.predict()
        return self.get_position()

    def update(self, x, y):
        if self.state == "UNINIT":
            self.initialize(x, y)
            return

        measurement = np.array([x, y], dtype=np.float32)
        self.kf.update(measurement)

    def get_position(self):
        if self.state == "UNINIT":
            return None
        return float(self.kf.x[0]), float(self.kf.x[1])

    def get_velocity(self):
        """Return Kalman filter velocity estimate (vx, vy) in m/s."""
        if self.state == "UNINIT":
            return None
        return float(self.kf.x[2]), float(self.kf.x[3])