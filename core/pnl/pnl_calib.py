"""
PnL Calibrator

Provides:
- Full 3×4 projection matrix P for 3D
- Camera intrinsics K, rotation R, position t

Usage:
    pnl = PnLCalib("weights/SV_kp", "weights/SV_lines", device="cuda")
    calib = pnl.estimate(frame)  # dict with P, K, R, t, kp_dict, lines_dict
"""

import cv2
import yaml
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as f
from PIL import Image

from .model.cls_hrnet import get_cls_net
from .model.cls_hrnet_l import get_cls_net as get_cls_net_l
from .utils.utils_calib import FramebyFrameCalib, line_world_coords_3D
from .utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints,
    coords_to_dict,
)


class PnLCalib:
    """
    1. Line Detection: PnLCalib's two HRNet models (keypoint + line extremity detection)
    2. FramebyFrameCalib camera calibration
    """

    def __init__(self, weights_kp, weights_line, device="cuda",
                 width=960, height=540,
                 kp_threshold=0.34, line_threshold=0.79):
        """
        Args:
            weights_kp: path to keypoint HRNet weights (e.g. "weights/SV_kp")
            weights_line: path to line HRNet weights (e.g. "weights/SV_lines")
            device: "cuda", "mps", or "cpu"
            width, height: input frame resolution
            kp_threshold: confidence threshold for keypoint heatmap peaks
            line_threshold: confidence threshold for line heatmap peaks
        """
        self.device = device
        self.width = width
        self.height = height
        self.kp_threshold = kp_threshold
        self.line_threshold = line_threshold

        # Load HRNet configs
        cfg_path = "config/hrnetv2_w48.yaml"
        cfg_l_path = "config/hrnetv2_w48_l.yaml"
        cfg = yaml.safe_load(open(cfg_path, 'r'))
        cfg_l = yaml.safe_load(open(cfg_l_path, 'r'))

        # Keypoint model (58 channels: 57 keypoints + background)
        loaded_state = torch.load(weights_kp, map_location=device)
        self.model_kp = get_cls_net(cfg)
        self.model_kp.load_state_dict(loaded_state)
        self.model_kp.to(device)
        self.model_kp.eval()

        # Line model (24 channels: 23 line types + background)
        loaded_state_l = torch.load(weights_line, map_location=device)
        self.model_line = get_cls_net_l(cfg_l)
        self.model_line.load_state_dict(loaded_state_l)
        self.model_line.to(device)
        self.model_line.eval()

        # Resize transform (models expect 960×540)
        self.resize = T.Resize((540, 960))

        # Calibration engine (instantiated per-frame)
        self.calib = None
        self._last_result = None
        self._last_cam_params = None  # previous frame's cam_params, used to warm-start

        # Pitch geometry constants for BEV rendering
        self._pitch_length = 105.0
        self._pitch_width = 68.0
        self._bev_scale = 10.0  # px/m

        print(f"PnLCalib initialized on {device} "
              f"({width}×{height})")

    def estimate(self, frame):
        """Run keypoint/line detection and camera calibration on a frame.

        Returns:
            dict with keys:
                P: 3×4 projection matrix (world 3D → image 2D)
                K: 3×3 intrinsic matrix
                R: 3×3 rotation matrix (world → camera)
                t: (3,) camera center in world coords (meters)
                kp_dict: detected keypoints {id: {x, y, p}}
                lines_dict: detected line endpoints {id: {x_1, y_1, x_2, y_2, ...}}
                rep_err: reprojection error (pixels), or None
                cam_params: full camera parameter dict from PnLCalib
            or None if calibration failed
        """
        self._last_result = None

        # Preprocess: BGR → RGB → PIL → tensor
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img_tensor = f.to_tensor(img).float().unsqueeze(0)
        _, _, h_orig, w_orig = img_tensor.size()
        if img_tensor.size(-1) != 960:
            img_tensor = self.resize(img_tensor)
        img_tensor = img_tensor.to(self.device)
        b, c, h, w = img_tensor.size()

        # Forward pass both models
        with torch.no_grad():
            heatmaps_kp = self.model_kp(img_tensor)
            heatmaps_line = self.model_line(img_tensor)

        # Extract keypoints and line endpoints from heatmaps
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps_kp[:, :-1])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_line[:, :-1])
        kp_dict = coords_to_dict(kp_coords, threshold=self.kp_threshold)
        lines_dict = coords_to_dict(line_coords, threshold=self.line_threshold)

        # Complete missing keypoints from line intersections, normalize coords
        kp_dict, lines_dict = complete_keypoints(
            kp_dict[0], lines_dict[0], w=w, h=h, normalize=True
        )

        # Run calibration (warm-started from the previous frame's camera params)
        self.calib = FramebyFrameCalib(
            iwidth=self.width, iheight=self.height, denormalize=True,
            warm_start=self._last_cam_params,
        )
        self.calib.update(kp_dict, lines_dict)

        result = self.calib.heuristic_voting(
            refine=True,
            refine_lines=True
        )

        if result is None:
            return None

        cam_params = result["cam_params"]
        rep_err = result["rep_err"]

        # Cache for next frame's warm-start
        self._last_cam_params = cam_params

        # Extract camera components from params dict
        K = np.array([
            [cam_params["x_focal_length"], 0, cam_params["principal_point"][0]],
            [0, cam_params["y_focal_length"], cam_params["principal_point"][1]],
            [0, 0, 1],
        ], dtype=np.float64)

        R = np.array(cam_params["rotation_matrix"], dtype=np.float64)
        t = np.array(cam_params["position_meters"], dtype=np.float64)

        # Build projection matrix P = K [R | -R t]
        It = np.eye(4, dtype=np.float64)[:3]
        It[:, 3] = -t
        P = K @ (R @ It)

        self._last_result = {
            "P": P,
            "K": K,
            "R": R,
            "t": t,
            "kp_dict": kp_dict,
            "lines_dict": lines_dict,
            "rep_err": rep_err,
            "cam_params": cam_params,
        }
        return self._last_result

    def draw_pitch_lines(self, img, color=(0, 255, 0), thickness=2):
        """Project standard pitch lines back onto the original frame using P matrix."""

        if self._last_result is None:
            return img

        P = self._last_result["P"]

        # Standard pitch line segments in 3D (from PnLCalib)

        for line in line_world_coords_3D:
            w1, w2 = np.array(line[0]), np.array(line[1])
            # Project both endpoints
            p1_h = P @ np.array([w1[0], w1[1], w1[2], 1.0])
            p2_h = P @ np.array([w2[0], w2[1], w2[2], 1.0])
            if abs(p1_h[2]) < 1e-6 or abs(p2_h[2]) < 1e-6:
                continue
            p1 = (int(p1_h[0] / p1_h[2]), int(p1_h[1] / p1_h[2]))
            p2 = (int(p2_h[0] / p2_h[2]), int(p2_h[1] / p2_h[2]))
            # Filter out-of-bounds points
            h, w = img.shape[:2]
            if (0 <= p1[0] < w and 0 <= p1[1] < h and
                    0 <= p2[0] < w and 0 <= p2[1] < h):
                cv2.line(img, p1, p2, color, thickness)

        return img

    def world_to_bev_px(self, x, y):
        """Convert world coords (meters, origin at pitch center) to BEV pixels."""
        px = int((x + self._pitch_length / 2) * self._bev_scale)
        py = int((self._pitch_width / 2 + y) * self._bev_scale)
        return px, py

    def create_bev_template(self):
        """Create a static top-down pitch image with lines and markings."""
        from .utils.utils_calib import line_world_coords_3D

        w = int(self._pitch_length * self._bev_scale)
        h = int(self._pitch_width * self._bev_scale)
        canvas = np.ones((h, w, 3), dtype=np.uint8) * np.array([76, 156, 76], dtype=np.uint8)

        for (x1, y1, _), (x2, y2, _) in line_world_coords_3D:
            px1, py1 = self.world_to_bev_px(x1, y1)
            px2, py2 = self.world_to_bev_px(x2, y2)
            cv2.line(canvas, (px1, py1), (px2, py2), (255, 255, 255), 1)

        cx, cy = self.world_to_bev_px(0, 0)
        cv2.circle(canvas, (cx, cy), int(9.15 * self._bev_scale), (255, 255, 255), 1)
        cv2.circle(canvas, (cx, cy), 3, (255, 255, 255), -1)

        return canvas
