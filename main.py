from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
import torch

from core.broadtrack_calib import BroadTrackCalib
from core.projection_utils import pixel_to_ground
from core.player_tracker import PlayerTracker
from core.ball_tracker import OCCLUDED, BallDetector, BallTracker
from core.team_assigner import TeamAssigner

BALL_MODEL_PATH = "models/yolo-sn-ball.pt"
PLAYER_MODEL_PATH = "models/yolo11n-seg.pt"

# BroadTrack's TorchScript detectors. See core/BroadTrack/README.md;
# all four weights live in models/ (that directory is gitignored).
BROADTRACK_KP_WEIGHTS = "models/nbjw_keypoint_model.pt"
BROADTRACK_LINE_WEIGHTS = "models/tvcalib_model.pt"

STATIC_MARKS = [(0.0, 0.0), (41.5, 0.0), (-41.5, 0.0)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="input_vids/test2.mp4")
    parser.add_argument("--max-frames", type=int, default=0, help="0 for the whole video")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--no-display", action="store_true", help="run headless")
    return parser.parse_args()

def main():
    args = parse_args()
    display = not args.no_display and bool(os.environ.get("DISPLAY"))

    # VIDEO PROCESSING
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(args.max_frames, total_frames) if args.max_frames else total_frames

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # CORE init
    calib_engine = BroadTrackCalib(
        weights_kp=BROADTRACK_KP_WEIGHTS,
        weights_line=BROADTRACK_LINE_WEIGHTS,
        device=device,
        width=width,
        height=height,
    )
    ball_detector = BallDetector(BALL_MODEL_PATH, device)
    ball_tracker = BallTracker(fps=fps, static_marks=STATIC_MARKS)
    player_tracker = PlayerTracker(PLAYER_MODEL_PATH, device)
    team_assigner = TeamAssigner()

    bev_template = calib_engine.create_bev_template()
    bev_canvas_h, bev_canvas_w = bev_template.shape[:2]

    # Video Output
    os.makedirs(args.out_dir, exist_ok=True)
    out_cam = cv2.VideoWriter(os.path.join(args.out_dir, "cam.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    out_bev = cv2.VideoWriter(os.path.join(args.out_dir, "bev.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (bev_canvas_w, bev_canvas_h))

    if display:
        cv2.namedWindow("cam", cv2.WINDOW_NORMAL)
        cv2.namedWindow("bev", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cam", width, height)
        cv2.resizeWindow("bev", bev_canvas_w, bev_canvas_h)
        cv2.moveWindow("cam", 50, 50)
        cv2.moveWindow("bev", 50 + width + 20, 50)

    last_calib = None
    try:
        for frame_idx in tqdm(range(limit)):
            ret, frame = cap.read()
            if not ret:
                tqdm.write("Failed to read frame")
                break
            frame = cv2.resize(frame, (width, height))
            cam_canva = frame.copy()
            bev_canva = bev_template.copy()

            # Players first:
            # BroadTrack uses the boxes to reject optical-flow points that landed on a moving player
            player_dets = player_tracker.update(frame)
            player_boxes = [d["bbox"] for d in player_dets]

            new_calib = calib_engine.estimate(frame, boxes=player_boxes)
            calib_valid = new_calib is not None
            if new_calib is not None:
                last_calib = new_calib
                calib = new_calib
            else:
                tqdm.write("Failed to estimate calib")
                calib = last_calib
            if calib is None:
                continue
            K = calib["K"]
            R = calib["R"]
            t = calib["t"]

            team_assigner.update(frame, player_dets, K, R, t)
            ball_dets = ball_detector.detect(frame)

            # BEV: players via pixel_to_ground (bottom-centre = ground contact point)
            player_centers = player_tracker.get_player_centers(player_dets)
            for tid, (cx, cy) in player_centers.items():
                pt = pixel_to_ground(cx, cy, K, R, t)
                if pt is not None:
                    px, py = calib_engine.world_to_bev_px(pt[0], pt[1])
                    cv2.circle(bev_canva, (px, py), 4, team_assigner.draw_colour(tid), -1)

            # BEV: ball
            ball_candidates = ball_detector.get_candidates(ball_dets, K, R, t)
            prediction = ball_tracker.process_frame(
                ball_candidates,
                player_boxes=player_boxes,
                frame_idx=frame_idx,
                calib_score=calib["score"],
                calib_reinit=calib["reinit"],
                calib_frames_lost=calib["frames_lost"],
                calib_valid=calib_valid,
            )
            cv2.putText(
                bev_canva,
                f"{ball_tracker.state} m{ball_tracker.miss_count} r{ball_tracker.reject_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
            )
            if prediction is not None:
                px, py = calib_engine.world_to_bev_px(prediction[0], prediction[1])
                if ball_tracker.state == OCCLUDED:
                    cv2.circle(bev_canva, (px, py), 5, (0, 255, 255), 2)  # hollow
                else:
                    cv2.circle(bev_canva, (px, py), 5, (0, 255, 255), -1)  # yellow

            # Camera Plane
            calib_engine.draw_pitch_lines(cam_canva, color=(0, 255, 0), thickness=2)
            player_tracker.draw_tracks(cam_canva, player_dets)
            if ball_dets and ball_dets[0].boxes is not None:
                ball_boxes = ball_dets[0].boxes.xyxy.cpu().numpy()
                for box in ball_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(cam_canva, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red

            # Calibration quality (line-IoU against the segmented pitch lines)
            cv2.putText(cam_canva, f"calib score {calib['score']:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            out_cam.write(cam_canva)
            out_bev.write(bev_canva)

            if display:
                cv2.imshow("cam", cam_canva)
                cv2.imshow("bev", bev_canva)
            delay = max(1, int(1000 / fps))
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
    finally:
        out_cam.release()
        out_bev.release()
        cap.release()
        if display:
            cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
