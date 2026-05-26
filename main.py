import os
import sys

import cv2
from tqdm import tqdm
import core.ball_tracker as ball_tracker
from core.ui_renderer import UIRenderer
from core.pitch_detection.line_det import LineDetector
from core.pitch_detection.homography_estimator import HomographyEstimator
from core.player_tracker import PlayerTracker

FOOTBALL_MODEL_PATH = "models/football_best.pt"  # YOUR MODEL PATH HERE
VIDEO_PATH = "./input_vids/test1.mp4" # YOUR VIDEO PATH HERE
OUTPUT_VIDEO = "./output/out.mp4"

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
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

def has_display():
    if sys.platform == 'darwin' or sys.platform == 'win32':
        return True  # macOS and windows always has a display
    return 'DISPLAY' in os.environ and os.environ['DISPLAY']

if has_display():
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.namedWindow("pitch", cv2.WINDOW_NORMAL)

# Ball Tracker
football_tracker = ball_tracker.BallTracker()
# Player Tracker
player_tracker = PlayerTracker(FOOTBALL_MODEL_PATH)

# UI Renderer
ui_renderer = UIRenderer(width, height, has_display())


for frame_idx in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret: # abbr. of return
        tqdm.write("Failed to read frame")
        break


    # Pitch Detector
    pitch_dets = line_detector.detect(frame)
    homo_est.estimate(pitch_dets)
    visualized_pitch_det = homo_est.draw_pitch_lines(frame.copy())

    if has_display():
        cv2.imshow("pitch", visualized_pitch_det)


    # Ball Tracker
    ball_detect = ball_model(
        frame,
        conf=0.05,
        classes=[ball_cls],
        verbose=False
    )
    football_tracker.ball_detection(ball_detect[0].boxes)
    ball_xy = football_tracker.ball_xy
    ball_state = football_tracker.ball_state


    # Player Tracker
    player_results = player_tracker.update(frame)

    if not player_results or player_results[0].boxes is None or player_results[0].boxes.id is None:
        tqdm.write(f"[{frame_idx}] no player detected")
        # Still render UI even without detections
        rendered = ui_renderer.render(frame, ball_xy, ball_state)
        writer.write(rendered)
        continue

    boxes = player_results[0].boxes
    ids = boxes.id.cpu().numpy()
    xys = boxes.xyxy.cpu().numpy()

    # Render enhanced UI
    rendered = ui_renderer.render(
        frame, ball_xy, ball_state,
        boxes=xys, ids=ids,
        M_to_ref=compensator.M_to_ref,
        frame_idx=frame_idx
    )

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
