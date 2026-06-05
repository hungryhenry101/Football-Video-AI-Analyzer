import os
os.chdir("../")

import cv2
from core.ball_tracker import BallDetector
from core.ball_tracker import BallTracker
from core.pitch_detection.line_det import LineDetector
from core.pitch_detection.homography_estimator import HomographyEstimator


def main():
    ball_detector = BallDetector("models/football_best.pt")
    tracker = BallTracker()
    vid = cv2.VideoCapture("input_vids/test2.mp4")

    width, height = 735, 404
    line_detection = LineDetector(".", width, height)

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))

        detection = line_detection.detect(frame)
        canva = frame.copy()
        homo_est = HomographyEstimator(width, height)
        homo_est.estimate(detection)
        bev_img = homo_est.warp(canva)

        raw_ball = ball_detector.detect(frame)
        if len(raw_ball) > 0:
            result = raw_ball[0]
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            pred = tracker.predict()
            for box in boxes_xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if pred is not None:
                    cv2.circle(frame, (int(pred[0]), int(pred[1])), 5, (255,0,0), -1)
                tracker.update((x1 + x2) / 2, (y1 + y2) / 2)

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


if __name__ == "__main__":
    main()
