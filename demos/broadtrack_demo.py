"""Camera calibration demo: BroadTrack pitch-line overlay on a video.

Requires the BroadTrack TorchScript detectors in models/ and the compiled
extension (python scripts/build_native.py).
"""

from pathlib import Path
import sys
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import cv2
from core.broadtrack_calib import BroadTrackCalib
import torch
from tqdm import tqdm

if __name__ == '__main__':
    KP_WEIGHTS = "models/nbjw_keypoint_model.pt"
    LINE_WEIGHTS = "models/tvcalib_model.pt"
    VIDEO_PATH = "input_vids/test2.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    calib_engine = BroadTrackCalib(
        weights_kp=KP_WEIGHTS,
        weights_line=LINE_WEIGHTS,
        device=device,
        width=width,
        height=height,
    )

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        calib = calib_engine.estimate(frame)
        calib_engine.draw_pitch_lines(frame, color=(0, 255, 0), thickness=2)
        if calib is not None:
            cv2.putText(frame, f"score {calib['score']:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
