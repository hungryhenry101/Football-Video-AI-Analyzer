from pathlib import Path
import sys
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


import cv2
import torch
from core.ball_tracker import BallDetector, BallTracker
from core.pnl.pnl_calib import PnLCalib


def main():
    vid = cv2.VideoCapture("input_vids/test2.mp4")
    fps = vid.get(cv2.CAP_PROP_FPS)
    width, height = 735, 404

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    ball_detector = BallDetector("weights/football_best.pt", device)
    tracker = BallTracker(fps=fps)

    pnl_calib = PnLCalib(
        weights_kp="weights/SV_kp",
        weights_line="weights/SV_lines",
        device=device,
        width=width,
        height=height
    )
    bev_template = pnl_calib.create_bev_template()

    cv2.moveWindow("frame", 50, 50)
    cv2.moveWindow("out bev", 50 + width + 20, 50)

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))

        calib = pnl_calib.estimate(frame)
        if calib is None:
            continue
        K, R, t = calib["K"], calib["R"], calib["t"]

        bev_img = bev_template.copy()

        raw_balls = ball_detector.detect(frame)
        if len(raw_balls) > 0:
            # KF prediction (blue on BEV)
            candidates = ball_detector.project_to_ground(raw_balls, K, R, t)
            pred = tracker.process_frame(candidates)
            if pred is not None:
                px, py = pnl_calib.world_to_bev_px(pred[0], pred[1])
                cv2.circle(bev_img, (px, py), 5, (255, 0, 0), -1)

            # raw detections (green on camera)
            result = raw_balls[0]
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            for box in boxes_xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # all candidates (red on BEV)
        p_balls = ball_detector.project_to_ground(raw_balls, K, R, t)
        for bx, by in p_balls:
            px, py = pnl_calib.world_to_bev_px(bx, by)
            cv2.circle(bev_img, (px, py), 5, (0, 0, 255), -1)

        cv2.imshow("frame", frame)
        cv2.imshow("out bev", bev_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
