"""PyTorch ports of BroadTrack's two pitch detectors.

BroadTrack's C++ loads two TorchScript models through libtorch
(``LineSegmentationModel.cpp`` and ``KeypointDetectionModel.cpp``).
This module reproduces both with Python ``torch`` package instead, so the
compiled extension never has to link a second copy of libtorch.

Three details of the upstream preprocessing are deliberate and easy to "fix" by
accident, each of which silently degrades accuracy rather than failing loudly:

* **The tensors stay in BGR order.** Both models copy the raw ``cv::Mat`` bytes
  into a ``[B, H, W, C]`` tensor without any channel swap.
* **The two models normalise differently.**
  ``LineSegmentationModel`` ends with ImageNet mean/std scaling;
  ``KeypointDetectionModel`` stops at ``/255``.
  This is not a typo upstream — feeding the keypoint model ImageNet-normalised input
  drops its detections from ~27 channels to zero, which in turn silently removes
  the keypoint-driven re-initialisation.
* **No resizing aside from the model's own.** The segmentation model resizes so
  its shortest side is 256 px, the keypoint model so its shortest side is 540 px.

The keypoint model emits heatmaps at half the resolution of its input, and the
upstream code compensates with ``scale = 2`` (see
``getKeypointsFromHeatmapBatchMaxpool``). We apply the same factor and then map
back to the caller's frame resolution.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Label maps, copied from SoccerPitch3D.h / the upstream model wrappers.
# ---------------------------------------------------------------------------

UNDEFINED_LINE = 150

LINE_CLASS_TO_ID = {
    1: 157,   # Big rect. left bottom       -> L_PENALTY_AREA_B_SIDE
    2: 158,   # Big rect. left main         -> L_PENALTY_AREA_R_SIDE
    3: 156,   # Big rect. left top          -> L_PENALTY_AREA_T_SIDE
    4: 160,   # Big rect. right bottom      -> R_PENALTY_AREA_B_SIDE
    5: 161,   # Big rect. right main        -> R_PENALTY_AREA_L_SIDE
    6: 159,   # Big rect. right top         -> R_PENALTY_AREA_T_SIDE
    7: 169,   # Circle central              -> CENTER_CIRCLE
    8: 168,   # Circle left                 -> L_PENALTY_ARC
    9: 170,   # Circle right                -> R_PENALTY_ARC
    10: UNDEFINED_LINE,  # Goal left crossbar
    11: UNDEFINED_LINE,  # Goal left post left
    12: UNDEFINED_LINE,  # Goal left post right
    13: UNDEFINED_LINE,  # Goal right crossbar
    14: UNDEFINED_LINE,  # Goal right post left
    15: UNDEFINED_LINE,  # Goal right post right
    16: UNDEFINED_LINE,  # Goal unknown
    17: UNDEFINED_LINE,  # Line unknown
    18: 154,  # Middle line                 -> HALFWAY_LINE
    19: 152,  # Side line bottom            -> B_TOUCH_LINE
    20: 153,  # Side line left              -> L_GOAL_LINE
    21: 155,  # Side line right             -> R_GOAL_LINE
    22: 151,  # Side line top               -> T_TOUCH_LINE
    23: 163,  # Small rect. left bottom     -> L_GOAL_AREA_B_SIDE
    24: 164,  # Small rect. left main       -> L_GOAL_AREA_R_SIDE
    25: 162,  # Small rect. left top        -> L_GOAL_AREA_T_SIDE
    26: 166,  # Small rect. right bottom    -> R_GOAL_AREA_B_SIDE
    27: 167,  # Small rect. right main      -> R_GOAL_AREA_L_SIDE
    28: 165,  # Small rect. right top       -> R_GOAL_AREA_T_SIDE
}
"""Segmentation class index -> ``SoccerPitch3D::LineID``. Class 0 is background."""

KEYPOINT_CLASS_TO_POINT_ID = {
    50: 101,  # CENTER_MARK
    44: 102,  # L_PENALTY_MARK
    56: 103,  # R_PENALTY_MARK
    0: 104,   # TL_PITCH_CORNER
    27: 106,  # BL_PITCH_CORNER
    2: 105,   # TR_PITCH_CORNER
    29: 107,  # BR_PITCH_CORNER
    3: 108,   # L_PENALTY_AREA_TL_CORNER
    4: 109,   # L_PENALTY_AREA_TR_CORNER
    23: 110,  # L_PENALTY_AREA_BL_CORNER
    24: 111,  # L_PENALTY_AREA_BR_CORNER
    5: 112,   # R_PENALTY_AREA_TL_CORNER
    6: 113,   # R_PENALTY_AREA_TR_CORNER
    25: 114,  # R_PENALTY_AREA_BL_CORNER
    26: 115,  # R_PENALTY_AREA_BR_CORNER
    7: 116,   # L_GOAL_AREA_TL_CORNER
    8: 117,   # L_GOAL_AREA_TR_CORNER
    19: 118,  # L_GOAL_AREA_BL_CORNER
    20: 119,  # L_GOAL_AREA_BR_CORNER
    9: 120,   # R_GOAL_AREA_TL_CORNER
    10: 121,  # R_GOAL_AREA_TR_CORNER
    21: 122,  # R_GOAL_AREA_BL_CORNER
    22: 123,  # R_GOAL_AREA_BR_CORNER
    1: 124,   # T_TOUCH_AND_HALFWAY_LINES_INTERSECTION
    28: 125,  # B_TOUCH_AND_HALFWAY_LINES_INTERSECTION
    31: 126,  # T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION
    34: 127,  # B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION
    30: 128,  # TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION
    32: 129,  # TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION
    33: 130,  # BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION
    35: 131,  # BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION
}
"""Keypoint heatmap channel -> ``SoccerPitch3D::PointID``."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_device(device: str | torch.device) -> torch.device:
    """Deal with devices"""
    if isinstance(device, torch.device):
        requested = device
    else:
        name = str(device).lower()
        if name.startswith("cuda") and not torch.cuda.is_available():
            requested = torch.device("cpu")
        elif name == "mps" and not torch.backends.mps.is_available():
            requested = torch.device("cpu")
        else:
            requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        requested = torch.device("cpu")
    return requested


def _resize_shortest_side_to(image: np.ndarray, shortest: int) -> np.ndarray:
    """Match BroadTrack's ``cv::resize`` geometry (shortest side pinned)."""
    rows, cols = image.shape[:2]
    if rows > cols:
        new_width = shortest
        new_height = int(shortest * rows / cols)
    else:
        new_height = shortest
        new_width = int(shortest * cols / rows)
    if (new_height, new_width) == (rows, cols):
        return image
    return cv2.resize(image, (new_width, new_height))


def _to_float_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR uint8 HWC -> ``[1, 3, H, W]`` float tensor in [0, 1] (no channel swap)."""
    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    return tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


def _to_normalised_tensor(
    image: np.ndarray, device: torch.device, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """As above, plus ImageNet mean/std. Only the segmentation model wants this."""
    return _to_float_tensor(image, device).sub_(mean).div_(std)


def _normalisation_tensors(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return mean, std


class LineSegmentationModel:
    """BroadTrack's pitch-line segmentation model (``tvcalib_model.pt``).

    ``compute_line_mask`` returns a ``uint8`` label map whose non-zero values are
    ``SoccerPitch3D::LineID`` codes (150-173), exactly as the C++ version does.
    """

    def __init__(self, model_path: str, device: str | torch.device = "cpu", shortest_side: int = 256):
        self.device = resolve_device(device)
        self.shortest_side = shortest_side
        self._model = torch.jit.load(str(model_path), map_location="cpu")
        self._model.eval()
        self._model.to(self.device)
        self._mean, self._std = _normalisation_tensors(self.device)

    @torch.no_grad()
    def compute_line_mask(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 BGR image")
        resized = _resize_shortest_side_to(frame, self.shortest_side)
        inputs = _to_normalised_tensor(resized, self.device, self._mean, self._std)
        output = self._model(inputs)
        logits = output["out"] if isinstance(output, dict) else output
        classes = logits.detach().squeeze(0).argmax(0).to("cpu").numpy().astype(np.int32)

        # Class 0 is background: those pixels keep the zero they start with.
        mask = np.zeros_like(classes, dtype=np.uint8)
        for class_index, line_id in LINE_CLASS_TO_ID.items():
            mask[classes == class_index] = line_id
        mask[mask == UNDEFINED_LINE] = 0
        return mask


class KeypointDetectionModel:
    """BroadTrack's pitch keypoint model (``nbjw_keypoint_model.pt``).

    ``compute_keypoints`` returns ``[(point_id, [(x, y), ...]), ...]`` in the
    resolution of the frame that was passed in.
    """

    def __init__(
        self,
        model_path: str,
        device: str | torch.device = "cpu",
        shortest_side: int = 540,
        heatmap_scale: int = 2,
        max_keypoints: int = 2,
        min_keypoint_pixel_distance: int = 15,
        score_threshold: float = 0.1,
    ):
        self.device = resolve_device(device)
        self.shortest_side = shortest_side
        self.heatmap_scale = heatmap_scale
        self.max_keypoints = max_keypoints
        self.min_keypoint_pixel_distance = min_keypoint_pixel_distance
        self.score_threshold = score_threshold
        self._model = torch.jit.load(str(model_path), map_location="cpu")
        self._model.eval()
        self._model.to(self.device)

    @torch.no_grad()
    def compute_keypoints(self, frame: np.ndarray) -> List[Tuple[int, List[Tuple[float, float]]]]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 BGR image")

        resized = _resize_shortest_side_to(frame, self.shortest_side)
        # /255 only — this model is NOT ImageNet-normalised (see the module docstring).
        # Adding mean/std scaling zeroes out every detection.
        inputs = _to_float_tensor(resized, self.device)
        heatmaps = self._model(inputs)
        if isinstance(heatmaps, dict):
            heatmaps = heatmaps["out"]

        coords, scores = self._local_maxima(heatmaps)
        # Heatmap space -> resized-input space -> caller's frame space.
        to_frame = frame.shape[0] / float(resized.shape[0]) * self.heatmap_scale
        coords = coords * to_frame
        scores = scores.to("cpu").numpy()

        out: List[Tuple[int, List[Tuple[float, float]]]] = []
        for channel, point_id in KEYPOINT_CLASS_TO_POINT_ID.items():
            points = [
                (float(coords[channel, k, 0]), float(coords[channel, k, 1]))
                for k in range(coords.shape[1])
                if scores[channel, k] > self.score_threshold
            ]
            if points:
                out.append((point_id, points))
        return out

    def _local_maxima(self, heatmaps: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        """Port of ``getKeypointsFromHeatmapBatchMaxpool`` with the upstream defaults."""
        if heatmaps.dim() != 4:
            raise ValueError("heatmap must be NxCxHxW")

        batch, channels, height, width = heatmaps.shape
        pad = self.min_keypoint_pixel_distance
        kernel = pad * 2 + 1

        padded = F.pad(heatmaps, (pad, pad, pad, pad), mode="constant", value=1.0)
        pooled = F.max_pool2d(padded, kernel, stride=1, padding=0)
        local_maxima = pooled == heatmaps
        filtered = heatmaps * local_maxima

        values, indices = torch.topk(
            filtered.reshape(batch, channels, -1), self.max_keypoints, dim=2, largest=True
        )
        x = (indices % width).float()
        y = (indices // width).float()
        coords = torch.stack((x, y), dim=-1)  # [B, C, K, 2]
        return coords[0].to("cpu").numpy(), values[0]


def keypoints_to_list(
    keypoints: Sequence[Tuple[int, Sequence[Tuple[float, float]]]]
) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """Normalise the callback payload the extension expects."""
    return [(int(point_id), [(float(x), float(y)) for x, y in points]) for point_id, points in keypoints]
