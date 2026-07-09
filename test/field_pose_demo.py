import os
os.chdir("../")

import cv2
import torch
import numpy as np
from ultralytics import YOLO


def main():
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = YOLO("./models/field_pose_best.pt").to(device)
    print(f"Model task: {model.task}, keypoints: {model.model.kpt_shape}")

    cap = cv2.VideoCapture("input_vids/test2.mp4")
    width, height = 735, 404

    # Per-keypoint colors: hue cycles around the color wheel, 32 steps
    kpt_colors = []
    for i in range(32):
        hue = int(180 * i / 32)
        c = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        kpt_colors.append(tuple(int(v) for v in c))

    cv2.namedWindow("field_pose", cv2.WINDOW_NORMAL)
    cv2.moveWindow("field_pose", 50, 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        canva = frame.copy()

        results = model(frame, verbose=False)

        for r in results:
            if r.keypoints is None:
                continue

            kpts = r.keypoints.data  # shape: (N, 32, 3)
            if kpts.shape[0] == 0:
                continue

            # Take the best detection (highest average confidence over keypoints)
            confs = kpts[:, :, 2].mean(dim=1)  # (N,)
            best_idx = confs.argmax().item()
            keypoints = kpts[best_idx].cpu().numpy()  # (32, 3): x, y, conf

            for i, (x, y, conf) in enumerate(keypoints):
                if conf < 0.1:
                    continue
                px, py = int(x), int(y)

                # Draw index
                cv2.putText(canva, str(i), (px + 4, py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

                # Draw point with confidence as radius
                r = int(2 + conf * 6)
                cv2.circle(canva, (px, py), r, kpt_colors[i], -1)

                # Thin border for visibility
                cv2.circle(canva, (px, py), r, (0, 0, 0), 1)

            # Draw confidence summary
            valid = (keypoints[:, 2] > 0.1).sum()
            avg_conf = keypoints[keypoints[:, 2] > 0.1, 2].mean() if valid > 0 else 0
            cv2.putText(canva, f"keypoints: {valid}/32  avg_conf: {avg_conf:.2f}",
                        (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("field_pose", canva)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            # Pause on space
            while True:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord(" ") or key2 == ord("q"):
                    break
            if key2 == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
