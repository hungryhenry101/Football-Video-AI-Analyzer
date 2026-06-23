import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.stats import chi2


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
        )
        return detections


    def project_to_pitch(self, detections, H):
        if H is None or len(detections) == 0:
            return []
        # project raw ball to hg BEV (meters) using Homography
        result = detections[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return []

        # 向量化：一次批量矩阵乘法完成所有框的投影，避免 Python 循环
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        ones = np.ones(len(boxes), dtype=np.float32)
        pts = np.column_stack([cx, cy, ones])  # N×3

        hg_pts = (H @ pts.T).T  # N×3，单次矩阵乘法替代逐个循环
        # 归一化齐次坐标 → (x_m, y_m) in meters
        hg_pts = hg_pts / hg_pts[:, 2:3]
        return [(float(p[0]), float(p[1])) for p in hg_pts]


class BallTracker:
    def __init__(self, chi2_thres=0.9):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # dt = 1 frame
        # [x, y, vx, vy] in hg BEV (meters)
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

        # noise parameters in meter units
        self.kf.Q = np.eye(4) * 0.08  # process noise
        self.kf.R = np.eye(2) * 0.05  # measurement noise

        # gating threshold (90% confidence for chi^2 with df=2)
        self.gate_threshold = chi2.ppf(chi2_thres, df=2)

        self.initialized = False

    def initialize(self, x, y):
        self.kf.x = np.array([[x],[y],[0],[0]], dtype=np.float32)  # Current State
        self.kf.P = np.eye(4, dtype=np.float32) * 0.1  # Error Covariance
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