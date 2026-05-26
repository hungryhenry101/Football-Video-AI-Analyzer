import numpy as np
from filterpy.kalman import KalmanFilter


def is_valid_measurement(kf, z, thresh=40):  # 95%, 2D
    y = z - kf.H @ kf.x
    S = kf.H @ kf.P @ kf.H.T + kf.R
    d = y.T @ np.linalg.inv(S) @ y
    # Use chi-square gating for 2 DOF. 95% ~= 5.99, 99% ~= 9.21
    return d < thresh


class BallTracker:
    def __init__(self):
        self.miss_count = 0
        self.MAX_MISS = 6

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0

        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        self.kf.P *= 500.0
        self.kf.R = np.eye(2) * 10.0
        self.kf.Q = np.eye(4) * 0.1

        self.ball_xy = None
        self.miss = 0

        self.ball_state = "VISIBLE"

    def select_ball(self, dets, prev_xy, max_dist, is_initialized):
        if len(dets) == 0:
            return None

        # If not initialized, just choose the highest-confidence detection
        if not is_initialized:
            return max(dets, key=lambda d: d["conf"])

        # Fallback to simple nearest-by-euclidean if prev_xy is provided
        if prev_xy is None:
            return max(dets, key=lambda d: d["conf"])

        px, py = prev_xy
        best = None
        best_dist = max_dist

        for d in dets:
            cx, cy = d["cx"], d["cy"]
            dist = np.hypot(cx - px, cy - py)
            if dist < best_dist:
                best_dist = dist
                best = d
        return best

    def ball_detection(self, detections):
        # 1. Properly format YOLO detections from the Boxes object
        formatted_dets = []
        for box in detections:
            # Extract center x, center y, and confidence
            xywh = box.xywh.cpu().numpy()[0]
            conf = box.conf.cpu().numpy()[0]
            formatted_dets.append({"cx": xywh[0], "cy": xywh[1], "conf": conf})

        self.kf.predict()

        # Check if we have a valid previous position
        is_initialized = self.ball_xy is not None

        # 2. Select the ball. Prefer Mahalanobis gating over raw Euclidean thresholds
        sel = None
        if len(formatted_dets) > 0:
            if not is_initialized:
                sel = max(formatted_dets, key=lambda d: d["conf"])  # highest confidence
            else:
                # compute Mahalanobis distance for each candidate
                def mahalanobis_for_det(det):
                    z = np.array([[det["cx"]], [det["cy"]]])
                    y = z - self.kf.H @ self.kf.x
                    S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
                    return float(y.T @ np.linalg.inv(S) @ y)

                dists = [(mahalanobis_for_det(d), d) for d in formatted_dets]
                dists.sort(key=lambda x: x[0])
                best_d2, best_det = dists[0]
                # gating threshold: 95%~6, 99%~9.2. Use 9.21 for visible; use larger when occluded
                gate_thresh = 9.21 if self.ball_state == "VISIBLE" else 16.0
                if best_d2 < gate_thresh:
                    sel = best_det

        if sel is not None:
            z = np.array([[sel["cx"]], [sel["cy"]]])

            # 3. If first detection, jump the Kalman Filter to the ball's location
            if not is_initialized:
                self.kf.x[0, 0] = sel["cx"]
                self.kf.x[1, 0] = sel["cy"]
                self.ball_xy = (int(sel["cx"]), int(sel["cy"]))
                self.ball_state = "VISIBLE"
                self.miss = 0
                return  # Skip update for the first frame to stabilize

            # Validation gate (only if already tracking)
            # scale measurement noise by detection confidence (higher conf -> lower R)
            prev_R = self.kf.R.copy()
            try:
                conf = float(sel.get("conf", 1.0))
            except Exception:
                conf = 1.0
            conf = max(conf, 0.01)
            self.kf.R = np.eye(2) * (10.0 / conf)

            if is_valid_measurement(self.kf, z, thresh=9.21):
                self.kf.update(z)
                self.ball_state = "VISIBLE"
                self.miss = 0
            else:
                self.miss += 1

            # restore measurement noise
            self.kf.R = prev_R
        else:
            self.miss += 1

        if self.miss >= 3:
            self.ball_state = "OCCLUDED"
            print("Ball occluded.")

        # Update the visual coordinates
        self.ball_xy = (int(self.kf.x[0, 0]), int(self.kf.x[1, 0]))