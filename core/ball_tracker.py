import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
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


    def warp(self, balls, scale=10.0):
        """
        :param
            balls: output of project_to_pitch()，[x_phys, y_phys, w]
            scale: px/m，must be same as scale in HomographyEstimator.warp()

        :return:
            [(x_bev, y_bev), ...] px coords for visualization of BEV
        """
        if len(balls) == 0:
            return []

        soccer_pitch = SoccerPitch()
        half_w = soccer_pitch.PITCH_LENGTH / 2
        half_h = soccer_pitch.PITCH_WIDTH / 2

        tx = half_w * scale
        ty = half_h * scale
        T = np.array([[scale, 0, tx],
                      [0, scale, ty],
                      [0, 0, 1]], dtype=np.float32)

        bev_points = []
        for ball in balls:
            p = T @ ball[:3]
            p = p / p[2]  # normalize homogeneous coords
            bev_points.append((int(p[0]), int(p[1])))
        return bev_points

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

        self.kf.Q = np.eye(4) * 0.01  # process noise
        self.kf.R = np.eye(2) * 5.0  # measurement noise

        self.initialized = False

    def initialize(self, x, y):
        self.kf.x = np.array([
            [x],
            [y],
            [0],
            [0]
        ], dtype=np.float32)
        # Reset covariance matrix so previous filtering history doesn't leak
        self.kf.P = np.eye(4, dtype=np.float32) * 10.
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        self.kf.predict()  # updates self.kf.x in-place, returns None
        x = float(self.kf.x[0])
        y = float(self.kf.x[1])
        return x, y

    def update(self, x, y):
        if not self.initialized:
            self.initialize(x, y)
            return

        measurement = np.array([x, y], dtype=np.float32)
        self.kf.update(measurement)