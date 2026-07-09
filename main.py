import os
import sys

import cv2
import torch
from tqdm import tqdm
from core.pitch_detection.line_det import LineDetector
from core.pitch_detection.homography_estimator import HomographyEstimator
from core.player_tracker import PlayerTracker
from core.ball_tracker import BallTracker, BallDetector

FOOTBALL_MODEL_PATH = "weights/football_best.pt"  # YOUR MODEL PATH
VIDEO_PATH = "input_vids/test2.mp4" # YOUR VIDEO PATH

# VIDEO PROCESSING
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)  # CUSTOM WIDTH
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)  # CUSTOM HEIGHT
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# GPU detect
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# CORE init
line_detector = LineDetector(os.path.dirname(os.path.abspath(__file__)), width, height, device)
homo_est = HomographyEstimator(width, height)
ball_detector = BallDetector(FOOTBALL_MODEL_PATH, device)
ball_tracker = BallTracker(fps=fps)
player_tracker = PlayerTracker(FOOTBALL_MODEL_PATH, device)

def has_display():
    if sys.platform == 'darwin' or sys.platform == 'win32':
        return True  # macOS and windows always has a display
    return 'DISPLAY' in os.environ and os.environ['DISPLAY']

if has_display():
    cv2.namedWindow("cam", cv2.WINDOW_NORMAL)
    cv2.namedWindow("bev", cv2.WINDOW_NORMAL)
    cv2.moveWindow("cam", 50, 50)
    cv2.moveWindow("bev", 50 + width + 20, 50)


# MAIN LOOP
for frame_idx in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        tqdm.write("Failed to read frame")
        break
    frame = cv2.resize(frame, (width, height))
    cam_canva = frame.copy()
    bev_canva = frame.copy()

    line_dets = line_detector.detect(frame)
    homo_est.estimate(line_dets)
    bev_canva = homo_est.warp(bev_canva)

    player_dets = player_tracker.update(frame)
    ball_dets = ball_detector.detect(frame)

    # BEV visualization
    if homo_est.H is not None:
        bev_players = player_tracker.project_to_pitch(player_dets, homo_est.H)
        player_pts = homo_est.warp_points(bev_players)
        if len(player_pts) > 0:
            for player_pt in player_pts:
                cv2.circle(bev_canva, (int(player_pt[0]), int(player_pt[1])), 5, (255, 0, 0), -1)  # blue
        else:
            print("no player detected")

        ball_candidates = ball_detector.project_to_pitch(ball_dets, homo_est.H)
        prediction = ball_tracker.process_frame(ball_candidates)
        if prediction is not None:
            # convert prediction from hg_bev (m) to out_bev (px) for display
            pred_out = homo_est.warp_points([(prediction[0], prediction[1], 1.0)])
            if pred_out:
                cv2.circle(bev_canva, (int(pred_out[0][0]), int(pred_out[0][1])), 5, (0, 0, 255), -1)  # red
        else:
            print("no ball candidates")

    # Camera Plane
    player_tracker.draw_tracks(cam_canva, player_dets)

    ball_boxes = ball_dets[0].boxes.xyxy.cpu().numpy()
    for box in ball_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(cam_canva, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red

    delay = max(1, int(1000 / fps))
    if has_display():
        cv2.imshow("cam", cam_canva)
        cv2.imshow("bev", bev_canva)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()
