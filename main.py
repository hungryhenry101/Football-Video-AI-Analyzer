import os
import sys
import csv
import random
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import core.ball_tracker as ball_tracker
import core.cmc as cmc
from core.ui_renderer import UIRenderer
from core.pitch_detection.line_det import LineDetector
from core.pitch_detection.homography_estimator import HomographyEstimator

FOOTBALL_MODEL_PATH = "models/football_best.pt"  # YOUR MODEL PATH HERE
OVERLAY_MODEL_PATH = "models/overlay.pt"
VIDEO_PATH = "./input_vids/test1.mp4" # YOUR VIDEO PATH HERE
OUTPUT_VIDEO = "./output/out.mp4"
CONF_THRES = 0.15  # 降低阈值以提高检测率

# PLAYER & BALL MODEL
player_model = YOLO(FOOTBALL_MODEL_PATH)
ball_model = YOLO(FOOTBALL_MODEL_PATH)
names = player_model.names
name2id = {v: k for k, v in player_model.names.items()}
ball_cls = name2id["ball"]

# CSV WRITER
csv_file = open("./output/track_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame","id","raw_x","raw_y","comp_x","comp_y","dx","dy","dist"])

# VIDEO PROCESSING
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# PITCH LINE DETECTOR
line_detector = LineDetector(os.path.dirname(os.path.abspath(__file__)), int(width/2), int(height/2))
homo_est = HomographyEstimator(width, height)

# Calculate new width for stats panel
stats_panel_width = 400
new_width = width + stats_panel_width
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (new_width, height))

def has_display():
    if sys.platform == 'darwin' or sys.platform == 'win32':
        return True  # macOS and windows always has a display
    return 'DISPLAY' in os.environ and os.environ['DISPLAY']

if has_display():
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.namedWindow("pitch", cv2.WINDOW_NORMAL)

# CMC
compensator = cmc.CMC(width, height, has_display())
# Ball Tracker
football_tracker = ball_tracker.BallTracker()
# UI Renderer
ui_renderer = UIRenderer(width, height, has_display())

track_colors = {}

# raw and compensated trajectories (players)
raw_traj = defaultdict(lambda: deque(maxlen=50))
comp_traj = defaultdict(lambda: deque(maxlen=50))

# Frame timing for UI
frame_times = deque(maxlen=30)

def get_color(track_id):
    if track_id not in track_colors:
        track_colors[track_id] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
    return track_colors[track_id]

def draw_traj(frame, traj, color):
    pts = list(traj)
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        x1, y1 = pts[i-1]
        x2, y2 = pts[i]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


for frame_idx in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret: # abbr. of return
        tqdm.write("Failed to read frame")
        break

    # Pitch Detection
    tqdm.write(f"[{frame_idx}] detecting pitch")
    pitch_dets = line_detector.detect(frame)
    homo_est.estimate(pitch_dets)
    visualized_pitch_det = homo_est.draw_pitch_lines(frame.copy())

    if has_display():
        cv2.imshow("pitch", visualized_pitch_det)

    raw_frame = frame.copy() # For CMC

    ball_detect = ball_model(
        frame,
        conf=0.05,
        classes=[ball_cls],
        verbose=False
    )

    # Ball Tracker
    football_tracker.ball_detection(ball_detect[0].boxes)
    ball_xy = football_tracker.ball_xy
    ball_state = football_tracker.ball_state

    compensator.gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

    # YOLO Player Tracker
    results = player_model.track(
        frame,
        persist=True,
        conf=CONF_THRES,
        tracker="bytetrack.yaml",
        verbose=False
    )

    if not results or results[0].boxes is None or results[0].boxes.id is None:
        tqdm.write(f"[{frame_idx}] no detections")
        # Still render UI even without detections
        rendered = ui_renderer.render(
            frame, ball_xy, ball_state,
            boxes=None, ids=None,
            M_to_ref=compensator.M_to_ref,
            frame_idx=frame_idx
        )
        writer.write(rendered)
        compensator.prev_gray = compensator.gray.copy()
        continue

    boxes = results[0].boxes
    ids = boxes.id.cpu().numpy()
    xys = boxes.xyxy.cpu().numpy()

    # Camera Motion Compensation
    compensator.compensate(football_tracker.ball_xy, xys, frame)

    # Update trajectories
    for tid, (x1, y1, x2, y2) in zip(ids, xys):
        tid = int(tid)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        color = get_color(tid)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        raw_traj[tid].append((cx, cy))

        if compensator.M_to_ref is not None:
            # M_to_ref 是从当前帧到参考帧的 2x3 仿射变换矩阵
            pt = np.array([cx, cy], dtype=np.float32)
            pt_c = compensator.M_to_ref[:, :2] @ pt + compensator.M_to_ref[:, 2]
            cx_c, cy_c = int(pt_c[0]), int(pt_c[1])
            comp_traj[tid].append((cx_c, cy_c))
            dx = cx - cx_c; dy = cy - cy_c; dist = np.hypot(dx, dy)
            csv_writer.writerow([frame_idx, tid, cx, cy, cx_c, cy_c, dx, dy, f"{dist:.2f}"])
        else:
            comp_traj[tid].append((cx, cy))

    # Render enhanced UI
    rendered = ui_renderer.render(
        frame, ball_xy, ball_state,
        boxes=xys, ids=ids,
        M_to_ref=compensator.M_to_ref,
        frame_idx=frame_idx
    )

    compensator.prev_gray = compensator.gray.copy()
    delay = max(1, int(1000 / fps))
    if has_display():
        cv2.imshow("preview", rendered)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    writer.write(rendered)

cv2.destroyAllWindows()
cap.release()
writer.release()

print(f"Tracking video saved to {OUTPUT_VIDEO}")
