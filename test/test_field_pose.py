"""
Test file for field_pose_best.pt model
- Visualizes all 32 keypoints on test images/videos
- Outputs detection data to CSV
- Saves visualization results to output/test_field_pose/
"""

import cv2
import numpy as np
from ultralytics import YOLO
import csv
import os
from datetime import datetime

# Configuration
MODEL_PATH = "models/field_pose_best.pt"
OUTPUT_DIR = "output/test_field_pose"
DEFAULT_TEST_IMAGE = "input_vids/test.png"
DEFAULT_TEST_VIDEO = "input_vids/test1.mp4"

# Keypoint group names for visualization
KEYPOINT_GROUPS = {
    "left_penalty": list(range(0, 13)),      # KP 0-12: Left goal/penalty area
    "center": [13, 14, 15, 16, 30, 31],       # Center line/circle
    "right_penalty": list(range(17, 30)),    # KP 17-29: Right goal/penalty area
}

# Keypoint names (from CLAUDE.md)
KEYPOINT_NAMES = {
    0: "Left Penalty Top-Left",
    1: "Left Penalty Upper",
    2: "Left Penalty Left",
    3: "Left Goal Area Top-Left",
    4: "Left Goal Line Top",
    5: "Left Goal Line Bottom",
    6: "Left Penalty Corner 1",
    7: "Left Goal Area Corner",
    8: "Left Penalty Corner 2",
    9: "Left Penalty Top-Right",
    10: "Left Penalty Bottom-Right",
    11: "Left Penalty Bottom",
    12: "Left Penalty Bottom-Left",
    13: "Center Circle Top",
    14: "Center Circle Upper",
    15: "Center Circle Lower",
    16: "Center Circle Bottom",
    17: "Right Penalty Top-Left",
    18: "Right Penalty Upper",
    19: "Right Penalty Right",
    20: "Right Goal Area Bottom-Right",
    21: "Right Penalty Corner 1",
    22: "Right Penalty Corner 2",
    23: "Right Goal Area Corner",
    24: "Right Penalty Top-Right",
    25: "Right Penalty Top Edge",
    26: "Right Penalty Upper",
    27: "Right Goal Area Top-Right",
    28: "Right Goal Line Top",
    29: "Right Goal Line Bottom",
    30: "Center Line Midpoint",
    31: "Center Circle Center",
}

# Colors for different groups (BGR)
GROUP_COLORS = {
    "left_penalty": (0, 0, 255),      # Red
    "center": (0, 255, 0),             # Green
    "right_penalty": (255, 0, 0),      # Blue
}


def draw_keypoints(frame, keypoints_abs, confidences, show_conf_threshold=0.3):
    """
    Draw keypoints on frame with different colors per group.
    keypoints_abs: list of (x_abs, y_abs, conf) in absolute pixel coordinates.
    Returns the frame with visualizations.
    """
    h, w = frame.shape[:2]

    # Draw keypoints
    for kp_idx, (x, y, conf) in enumerate(keypoints_abs):
        if conf < show_conf_threshold:
            continue

        x, y = int(x), int(y)

        # Determine group and color
        group_color = None
        for group_name, kp_indices in KEYPOINT_GROUPS.items():
            if kp_idx in kp_indices:
                group_color = GROUP_COLORS[group_name]
                break

        if group_color is None:
            group_color = (0, 255, 255)  # Yellow for ungrouped

        # Draw circle
        cv2.circle(frame, (x, y), 8, group_color, -1)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)

        # Draw label
        label = f"KP{kp_idx}:{conf:.2f}"
        cv2.putText(frame, label, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, group_color, 2)

    # Draw connections (skeleton) for each group
    # Left penalty area box connections
    left_penalty_connections = [
        (0, 1), (1, 9), (9, 10), (10, 11), (11, 12), (12, 0),  # Outer box
        (3, 7), (7, 8), (8, 6), (6, 3),  # Goal area
        (4, 5),  # Goal line
    ]

    # Right penalty area box connections
    right_penalty_connections = [
        (17, 24), (24, 25), (25, 17),  # Outer box top
        (24, 26), (26, 18), (18, 17),  # Outer box sides
        (20, 23), (23, 27), (27, 20),  # Goal area
        (28, 29),  # Goal line
    ]

    # Center connections
    center_connections = [
        (13, 14), (14, 31), (31, 15), (15, 16),  # Circle
        (30, 31),  # Center line to circle center
    ]

    def draw_connections(connections, color):
        for kp1, kp2 in connections:
            if kp1 < len(keypoints_abs) and kp2 < len(keypoints_abs):
                x1, y1, c1 = keypoints_abs[kp1]
                x2, y2, c2 = keypoints_abs[kp2]
                if c1 > show_conf_threshold and c2 > show_conf_threshold:
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    cv2.line(frame, pt1, pt2, color, 2)

    draw_connections(left_penalty_connections, GROUP_COLORS["left_penalty"])
    draw_connections(right_penalty_connections, GROUP_COLORS["right_penalty"])
    draw_connections(center_connections, GROUP_COLORS["center"])

    return frame


def draw_legend(frame):
    """Draw a legend showing color meanings."""
    legend_y = 30
    cv2.putText(frame, "Legend:", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    legend_items = [
        ("Left Penalty (Red)", GROUP_COLORS["left_penalty"]),
        ("Center (Green)", GROUP_COLORS["center"]),
        ("Right Penalty (Blue)", GROUP_COLORS["right_penalty"]),
    ]

    for i, (text, color) in enumerate(legend_items):
        y = legend_y + (i + 1) * 25
        cv2.circle(frame, (15, y - 5), 8, color, -1)
        cv2.putText(frame, text, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def test_on_image(image_path, model, output_dir):
    """Test field pose model on a single image."""
    print(f"\n{'='*50}")
    print(f"Testing on image: {image_path}")
    print(f"{'='*50}")

    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return None

    h, w = frame.shape[:2]
    print(f"Image size: {w}x{h}")

    # Run inference
    results = model(frame, verbose=False)
    result = results[0]

    # Extract keypoints
    keypoints_data = []
    if result.keypoints is not None:
        kps = result.keypoints.xy[0].cpu().numpy()  # N x 2 (absolute pixel coordinates)
        confs = result.keypoints.conf[0].cpu().numpy()  # N

        for i, (kp, conf) in enumerate(zip(kps, confs)):
            x, y = kp
            keypoints_data.append({
                'kp_idx': i,
                'kp_name': KEYPOINT_NAMES.get(i, f"KP_{i}"),
                'x_abs': x,  # Already absolute
                'y_abs': y,  # Already absolute
                'confidence': conf,
            })

    # Print detection summary
    print(f"\nDetected {len(keypoints_data)} keypoints:")
    detected_count = sum(1 for kp in keypoints_data if kp['confidence'] > 0.3)
    print(f"  High confidence (>0.3): {detected_count}")

    # Group summary
    for group_name, kp_indices in KEYPOINT_GROUPS.items():
        group_detections = [kp for kp in keypoints_data
                          if kp['kp_idx'] in kp_indices and kp['confidence'] > 0.3]
        print(f"  {group_name}: {len(group_detections)}/{len(kp_indices)} detected")

    # Create visualization
    vis_frame = frame.copy()
    if result.keypoints is not None:
        kps_abs = result.keypoints.xy[0].cpu().numpy()  # Absolute coordinates
        confs = result.keypoints.conf[0].cpu().numpy()
        # Convert to list of tuples: (x_abs, y_abs, conf)
        keypoints_abs = [(float(x), float(y), float(c)) for (x, y), c in zip(kps_abs, confs)]
        vis_frame = draw_keypoints(vis_frame, keypoints_abs, confs)

    vis_frame = draw_legend(vis_frame)

    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"result_{timestamp}.png")
    cv2.imwrite(output_path, vis_frame)
    print(f"\nVisualization saved to: {output_path}")

    # Save CSV data
    csv_path = os.path.join(output_dir, f"data_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['kp_idx', 'kp_name', 'x_abs', 'y_abs', 'confidence'])
        writer.writeheader()
        writer.writerows(keypoints_data)
    print(f"Keypoint data saved to: {csv_path}")

    return keypoints_data


def test_on_video(video_path, model, output_dir, num_frames=10):
    """Test field pose model on a video (sample frames)."""
    print(f"\n{'='*50}")
    print(f"Testing on video: {video_path}")
    print(f"{'='*50}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Sample frames
    frame_interval = max(1, total_frames // num_frames)
    frame_indices = list(range(0, total_frames, frame_interval))[:num_frames]

    all_keypoints = []

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        print(f"\nProcessing frame {frame_idx}/{total_frames}...")

        # Run inference
        results = model(frame, verbose=False)
        result = results[0]

        # Extract keypoints
        if result.keypoints is not None:
            kps = result.keypoints.xy[0].cpu().numpy()  # Absolute pixel coordinates
            confs = result.keypoints.conf[0].cpu().numpy()

            for i, (kp, conf) in enumerate(zip(kps, confs)):
                x, y = kp
                all_keypoints.append({
                    'frame': frame_idx,
                    'kp_idx': i,
                    'kp_name': KEYPOINT_NAMES.get(i, f"KP_{i}"),
                    'x_abs': x,  # Already absolute
                    'y_abs': y,  # Already absolute
                    'confidence': conf,
                })

            # Draw visualization
            vis_frame = frame.copy()
            kps_abs = result.keypoints.xy[0].cpu().numpy()
            confs = result.keypoints.conf[0].cpu().numpy()
            keypoints_abs = [(float(x), float(y), float(conf)) for (x, y), conf in zip(kps_abs, confs)]
            vis_frame = draw_keypoints(vis_frame, keypoints_abs, confs)
            vis_frame = draw_legend(vis_frame)

            # Add frame info
            cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Save frame visualization
            frame_output = os.path.join(output_dir, f"frame_{frame_idx:06d}_{timestamp}.png")
            cv2.imwrite(frame_output, vis_frame)

    cap.release()

    # Save combined CSV
    csv_path = os.path.join(output_dir, f"video_data_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'kp_idx', 'kp_name', 'x_abs', 'y_abs', 'confidence'])
        writer.writeheader()
        writer.writerows(all_keypoints)

    print(f"\nSaved {len(all_keypoints)} keypoint detections to: {csv_path}")

    # Summary statistics
    print("\nSummary Statistics:")
    print(f"  Total detections: {len(all_keypoints)}")

    per_frame_stats = {}
    for kp in all_keypoints:
        frame = kp['frame']
        if frame not in per_frame_stats:
            per_frame_stats[frame] = {'total': 0, 'high_conf': 0}
        per_frame_stats[frame]['total'] += 1
        if kp['confidence'] > 0.3:
            per_frame_stats[frame]['high_conf'] += 1

    for frame, stats in sorted(per_frame_stats.items()):
        print(f"  Frame {frame}: {stats['high_conf']}/{stats['total']} high confidence")

    return all_keypoints


def main():
    """Main test function."""
    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("\nModel info:")
    print(f"  Task: {model.task}")
    print(f"  Names: {model.names}")

    # Test on image
    if os.path.exists(DEFAULT_TEST_IMAGE):
        test_on_image(DEFAULT_TEST_IMAGE, model, OUTPUT_DIR)
    else:
        print(f"Test image not found: {DEFAULT_TEST_IMAGE}")

    # Test on video
    if os.path.exists(DEFAULT_TEST_VIDEO):
        test_on_video(DEFAULT_TEST_VIDEO, model, OUTPUT_DIR, num_frames=5)
    else:
        print(f"Test video not found: {DEFAULT_TEST_VIDEO}")

    print(f"\n{'='*50}")
    print("Testing complete!")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
