import os
os.chdir("../")

import cv2
from core.player_tracker import PlayerTracker
from core.pitch_detection.homography_estimator import HomographyEstimator
from core.pitch_detection.line_det import LineDetector

def main():
    tracker = PlayerTracker("./models/football_best.pt")
    cap = cv2.VideoCapture("input_vids/test2.mp4")
    width, height = 735, 404
    homo_est = HomographyEstimator(width, height)
    line_detector = LineDetector(".", width, height)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        canvas = frame.copy()

        result = tracker.update(frame)

        l_detection = line_detector.detect(frame)
        homo_est.estimate(l_detection)
        bev_canva = homo_est.warp(canvas)
        if homo_est.H is not None:
            bev_players = tracker.project_to_pitch(result, homo_est.H)
            players = homo_est.warp_points(bev_players)
            for player in players:
                cv2.circle(bev_canva, (int(player[0]), int(player[1])), 5, (255, 0, 0), -1)
            cv2.imshow("bev", bev_canva)

        tracker.draw_tracks(canvas, result)
        cv2.imshow("frame", canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()