from pathlib import Path
import sys
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


import cv2
import numpy as np
import torch
from core.ball_tracker import OCCLUDED, BallDetector, BallTracker
from core.broadtrack_calib import BroadTrackCalib

FALLBACK_STATIC_MARKS = [(0.0, 0.0), (41.5, 0.0), (-41.5, 0.0)]


def has_display():
    if sys.platform == "darwin" or sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY"))


def static_marks_from(calib_engine):
    marks = []
    for polyline in calib_engine.geometry.wireframe:
        points = np.asarray(polyline).reshape(-1, 3)
        if len(points) == 1:
            marks.append((float(points[0, 0]), float(points[0, 1])))
    return marks or FALLBACK_STATIC_MARKS


def main():
    vid = cv2.VideoCapture("input_vids/test2.mp4")
    fps = vid.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)   # CUSTOM WIDTH if needed
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)  # CUSTOM HEIGHT if needed

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    ball_detector = BallDetector("models/football_best.pt", device)

    calib_engine = BroadTrackCalib(
        weights_kp="models/nbjw_keypoint_model.pt",
        weights_line="models/tvcalib_model.pt",
        device=device,
        width=width,
        height=height
    )
    tracker = BallTracker(fps=fps, static_marks=static_marks_from(calib_engine))
    bev_template = calib_engine.create_bev_template()

    display = has_display()
    if display:
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow("out bev", cv2.WINDOW_NORMAL)
        cv2.moveWindow("frame", 50, 50)
        cv2.moveWindow("out bev", 50 + width + 20, 50)

    frame_idx = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))

        calib = calib_engine.estimate(frame)
        if calib is None:
            continue
        K, R, t = calib["K"], calib["R"], calib["t"]

        bev_img = bev_template.copy()

        raw_balls = ball_detector.detect(frame)
        candidates = ball_detector.get_candidates(raw_balls, K, R, t)
        # Always call process_frame, even with no detections, so the KF can coast.
        pred = tracker.process_frame(candidates, player_boxes=[], frame_idx=frame_idx)

        # raw candidates (red on BEV)
        for c in candidates:
            if c.x is not None and c.y is not None:
                px, py = calib_engine.world_to_bev_px(c.x, c.y)
                cv2.circle(bev_img, (px, py), 3, (0, 0, 255), -1)

        # tracker output (yellow on BEV): filled while TRACKING, hollow while OCCLUDED
        if pred is not None:
            px, py = calib_engine.world_to_bev_px(pred[0], pred[1])
            if tracker.state == OCCLUDED:
                cv2.circle(bev_img, (px, py), 5, (0, 255, 255), 2)
            else:
                cv2.circle(bev_img, (px, py), 5, (0, 255, 255), -1)
        cv2.putText(bev_img, f"{tracker.state} m{tracker.miss_count} r{tracker.reject_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # raw detections (green on camera)
        if raw_balls and raw_balls[0].boxes is not None:
            for box in raw_balls[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if display:
            cv2.imshow("frame", frame)
            cv2.imshow("out bev", bev_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1

    vid.release()
    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
