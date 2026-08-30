from pathlib import Path
import sys
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import cv2
from core.pnl.pnl_calib import PnLCalib
import torch
from tqdm import tqdm

if __name__ == '__main__':
    PNL_KP_WEIGHTS = "weights/SV_kp"
    PNL_LINE_WEIGHTS = "weights/SV_lines"
    VIDEO_PATH = "input_vids/test2.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    
    output = cv2.VideoWriter("output/pnl_demo.mp4", cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (width, height))

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    pnl_calib = PnLCalib(
        weights_kp=PNL_KP_WEIGHTS,
        weights_line=PNL_LINE_WEIGHTS,
        device=device,
        width=width,
        height=height
    )

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        calib = pnl_calib.estimate(frame)
        pnl_calib.draw_pitch_lines(frame, color=(0, 255, 0), thickness=2)
        output.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    output.release()