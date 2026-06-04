import os
os.chdir("../")

import cv2
from core.player_tracker import PlayerTracker

def main():
    tracker = PlayerTracker("./models/football_best.pt")
    cap = cv2.VideoCapture("./input_vids/test1.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        canvas = frame.copy()

        result = tracker.update(frame)
        tracker.draw_tracks(canvas, result)
        cv2.imshow("frame", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()