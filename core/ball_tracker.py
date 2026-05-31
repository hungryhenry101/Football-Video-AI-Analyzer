import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import cv2
from .pitch_detection.soccerpitch import SoccerPitch


def is_valid_measurement(kf, z, thresh=40):  # 95%, 2D
    y = z - kf.H @ kf.x
    S = kf.H @ kf.P @ kf.H.T + kf.R
    d = y.T @ np.linalg.inv(S) @ y
    # Use chi-square gating for 2 DOF. 95% ~= 5.99, 99% ~= 9.21
    return d < thresh

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
        # 通过 Homography 将 raw ball 投射为 pitch ball
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
        """将物理坐标（米）转换为 BEV 像素坐标。

        参数:
            balls: project_to_pitch() 的输出，每个元素是 [x_phys, y_phys, w]
            scale: 像素/米，必须与 HomographyEstimator.warp() 的 scale 一致

        返回:
            [(x_bev, y_bev), ...] BEV 图像上的像素坐标列表
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
            p = p / p[2]  # 归一化齐次坐标
            bev_points.append((int(p[0]), int(p[1])))
        return bev_points



##TEST
if __name__ == "__main__":
    ball_detector = BallDetector("./models/football_best.pt")
    vid = cv2.VideoCapture("./input_vids/test2.mp4")

    from .pitch_detection.line_det import LineDetector
    from .pitch_detection.homography_estimator import HomographyEstimator
    width, height = 735, 404
    line_detection = LineDetector("./", width, height)
    while True:
        ret, frame = vid.read()
        frame = cv2.resize(frame,(width, height))

        detection = line_detection.detect(frame)
        canva = frame.copy()
        homo_est = HomographyEstimator(width, height)
        homo_est.estimate(detection)
        bev_img = homo_est.warp(canva)

        raw_ball = ball_detector.detect(frame)
        if len(raw_ball) > 0:
            result = raw_ball[0]
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            for box in boxes_xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 0), 2)

        p_balls = ball_detector.project_to_pitch(raw_ball, homo_est.H)
        p_balls_bev = ball_detector.warp(p_balls)
        for p_b in p_balls_bev:
            cv2.circle(bev_img, p_b, 5, (0, 0, 255), -1)

        cv2.imshow("frame", frame)
        cv2.imshow("bev_img", bev_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    vid.release()
    cv2.destroyAllWindows()