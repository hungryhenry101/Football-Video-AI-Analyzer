import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.stats import chi2
from .pitch_detection.soccerpitch import SoccerPitch


class BallDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        detections = self.model(
            frame,
            conf=0.15,
            iou=0.45,
            imgsz=960,
            classes = [0]
        )
        return detections


    def project_to_pitch(self, detections, H):
        if H is None or len(detections) == 0:
            return []
        # project raw ball to BEV pitch ball using Homography
        bev_balls = []
        result = detections[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            p_homo = np.array([cx, cy, 1.0])
            p_bev = H @ p_homo
            bev_balls.append(p_bev)
        return bev_balls


class BallTracker:
    def __init__(self):
        soccer_pitch = SoccerPitch()
        self.penalty_mark_l = soccer_pitch.left_penalty_mark
        self.penalty_mark_r = soccer_pitch.right_penalty_mark

        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # [x, y, vx, vy], dt = 1 frame
        # State Transition Matrix
        self.kf.F = np.array([
            [1, 0, 1, 0], # x_pred = 1*x_old + 0*y_old + 1*vx_old + 0*vy_old = x_old + vx_old
            [0, 1, 0, 1], # y_pred
            [0, 0, 1, 0], # vx_pred
            [0, 0, 0, 1]  # vy_pred
        ], np.float32)

        # Measurement
        self.kf.H = np.array([
            [1, 0, 0, 0], # Zx
            [0, 1, 0, 0]  # Zy
        ], np.float32)

        self.kf.Q = np.eye(4) * 1.0  # process noise
        self.kf.R = np.eye(2) * 9.0  # measurement noise

        # gating threshold (90% confidence for chi^2 with df=2)
        self.gate_threshold = chi2.ppf(0.9, df=2)

        self.initialized = False

    def initialize(self, x, y):
        self.kf.x = np.array([[x],[y],[0],[0]], dtype=np.float32)
        self.kf.P = np.eye(4, dtype=np.float32) * 10.
        self.initialized = True

    def process_frame(self, candidates):
        # main
        best = self.select_best_candidate(candidates)
        if best is not None:
            self.update(best[0], best[1])
        return self.get_position()

    def select_best_candidate(self, candidates):
        """
        candidates: list of (x, y) detections in BEV coordinates after warp
        """
        if not self.initialized:
            # no gate on first frame
            if len(candidates) > 0:
                return candidates[0]
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
            print("no best candidate")

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
        if not self.initialized:
            return None
        self.kf.predict()
        return self.get_position()

    def update(self, x, y):
        if not self.initialized:
            self.initialize(x, y)
            return

        measurement = np.array([x, y], dtype=np.float32)
        self.kf.update(measurement)

    def get_position(self):
        if not self.initialized:
            return None
        return float(self.kf.x[0]), float(self.kf.x[1])