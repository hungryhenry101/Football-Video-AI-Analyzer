import os
os.chdir("../")

import cv2
import torch
from core.ball_tracker import BallDetector, BallTracker
from core.pitch_detection.line_det import LineDetector
from core.pitch_detection.homography_estimator import HomographyEstimator


def main():
    vid = cv2.VideoCapture("input_vids/test2.mp4")
    width, height = 735, 404

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    ball_detector = BallDetector("models/football_best.pt", device)
    tracker = BallTracker()

    line_detection = LineDetector(".", width, height, device)
    homo_est = HomographyEstimator(width, height)

    # Create windows once and lock positions so they don't move
    cv2.moveWindow("frame", 50, 50)
    cv2.moveWindow("out bev", 50 + width + 20, 50)

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))

        detection = line_detection.detect(frame)
        canva = frame.copy()
        homo_est.estimate(detection)
        bev_img = homo_est.warp(canva)

        raw_balls = ball_detector.detect(frame)
        if len(raw_balls) > 0:
            # KF
            candidates = ball_detector.project_to_pitch(raw_balls, homo_est.H)
            pred = tracker.process_frame(candidates)
            if pred is not None:
                pred_out = homo_est.warp_points([(pred[0], pred[1], 1.0)])
                if pred_out:
                    cv2.circle(bev_img, (int(pred_out[0][0]), int(pred_out[0][1])), 5, (255, 0, 0), -1)  # blue
            else:
                print("no candidates")

            # raw
            result = raw_balls[0]
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            for box in boxes_xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # green

        # all candidates (hg_bev → out_bev for display)
        p_balls = ball_detector.project_to_pitch(raw_balls, homo_est.H)
        p_balls_bev = homo_est.warp_points([(x, y, 1.0) for x, y in p_balls])
        for p_b in p_balls_bev:
            cv2.circle(bev_img, p_b, 5, (0, 0, 255), -1) #red

        cv2.imshow("frame", frame)
        cv2.imshow("out bev", bev_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
