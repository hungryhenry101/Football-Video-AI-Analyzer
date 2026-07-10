import os
import sys

import cv2
import torch
from tqdm import tqdm
from core.pnl.pnl_calib import PnLCalib
from core.pnl.projection_utils import pixel_to_ground
from core.player_tracker import PlayerTracker
from core.ball_tracker import BallTracker, BallDetector

FOOTBALL_WEIGHT_FILE = "weights/football_best.pt"  # YOUR FOOTBALL WEIGHT FILE
VIDEO_PATH = "input_vids/test2.mp4" # YOUR VIDEO PATH

PNL_KP_WEIGHTS = "weights/SV_kp"
PNL_LINE_WEIGHTS = "weights/SV_lines"

# VIDEO PROCESSING
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)  # CUSTOM WIDTH if needed
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)  # CUSTOM HEIGHT if needed
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# CORE init
pnl_calib = PnLCalib(
    weights_kp=PNL_KP_WEIGHTS,
    weights_line=PNL_LINE_WEIGHTS,
    device=device,
    width=width,
    height=height
)
ball_detector = BallDetector(FOOTBALL_WEIGHT_FILE, device)
ball_tracker = BallTracker(fps=fps)
player_tracker = PlayerTracker(FOOTBALL_WEIGHT_FILE, device)

bev_template = pnl_calib.create_bev_template()

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
    bev_canva = bev_template.copy()

    calib = pnl_calib.estimate(frame)
    if calib is None:
        tqdm.write("Failed to estimate calib")
        continue
    K = calib["K"]
    R = calib["R"]
    t = calib["t"]

    # project detected lines to cam view
    pnl_calib.draw_pitch_lines(cam_canva, color=(0,255,0), thickness=2)

    player_dets = player_tracker.update(frame)
    ball_dets = ball_detector.detect(frame)

    # BEV: players via pixel_to_ground (bottom-center = ground contact point)
    player_centers = player_tracker.get_player_centers(player_dets)
    for tid, (cx, cy) in player_centers.items():
        pt = pixel_to_ground(cx, cy, K, R, t)
        if pt is not None:
            px, py = pnl_calib.world_to_bev_px(pt[0], pt[1])
            cv2.circle(bev_canva, (px, py), 4, (255, 0, 0), -1)

    # BEV: ball via pixel_to_ground
    ball_candidates = ball_detector.project_to_ground(ball_dets, K, R, t)
    prediction = ball_tracker.process_frame(ball_candidates)
    if prediction is not None:
        px, py = pnl_calib.world_to_bev_px(prediction[0], prediction[1])
        cv2.circle(bev_canva, (px, py), 5, (0, 0, 255), -1)

    # Camera Plane
    player_tracker.draw_tracks(cam_canva, player_dets)

    if ball_dets[0].boxes is not None:
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
