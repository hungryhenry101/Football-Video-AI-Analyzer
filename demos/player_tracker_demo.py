from pathlib import Path
import sys
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


import cv2
import torch
from core.player_tracker import PlayerTracker
from core.pnl.pnl_calib import PnLCalib
from core.pnl.projection_utils import pixel_to_ground


def main():
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    tracker = PlayerTracker("./weights/football_best.pt", device)
    cap = cv2.VideoCapture("input_vids/test2.mp4")
    width, height = 735, 404

    pnl_calib = PnLCalib(
        weights_kp="weights/SV_kp",
        weights_line="weights/SV_lines",
        device=device,
        width=width,
        height=height
    )
    bev_template = pnl_calib.create_bev_template()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        canvas = frame.copy()

        result = tracker.update(canvas)

        calib = pnl_calib.estimate(frame)
        if calib is None:
            continue
        K, R, t = calib["K"], calib["R"], calib["t"]

        bev_canva = bev_template.copy()
        player_centers = tracker.get_player_centers(result)
        for tid, (cx, cy) in player_centers.items():
            pt = pixel_to_ground(cx, cy, K, R, t)
            if pt is not None:
                px, py = pnl_calib.world_to_bev_px(pt[0], pt[1])
                cv2.circle(bev_canva, (px, py), 5, (255, 0, 0), -1)
        cv2.imshow("bev", bev_canva)

        tracker.draw_tracks(canvas, result)
        cv2.imshow("frame", canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
