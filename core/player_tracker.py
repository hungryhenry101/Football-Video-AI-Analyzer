"""Player Detector (YOLO-seg) & Tracker (BoT-SORT)

Uses COCO pretrained YOLO segmentation model.
Will be used for colour clustering in downstream.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from core.projection_utils import pixel_to_ground

# person id in COCO
PERSON_CLASS = 0

CLASS_NAMES = {
    PERSON_CLASS: "person",
}

CLASS_COLORS = {
    PERSON_CLASS: (255, 255, 255),
}


class PlayerTracker:
    """YOLO-seg + BoT-SORT。

    Args:
        model_path: path of segmentation model
        conf_thres: confidence threshold
        imgsz: image size - 1280 instead of 640 as default
    """

    def __init__(
        self,
        model_path,
        device,
        tracker_config="config/botsort.yaml",
        conf_thres=0.25,
        imgsz=1280,
    ):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.conf_thres = conf_thres
        self.imgsz = imgsz
        self.device = device

    def update(self, frame):
        """
        Return [{id, bbox, cls, conf, mask}]
        mask is consistent with bbox, in shape of (w,h); 1 represents person
        """
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf_thres,
            imgsz=self.imgsz,
            classes=[PERSON_CLASS],
            tracker=self.tracker_config,
            device=self.device,
            verbose=False,
        )

        objs = []
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return objs

        boxes, ids, confs = [
            t.cpu().numpy() for t in (r.boxes.xyxy, r.boxes.id, r.boxes.conf)
        ]
        ids = ids.astype(int)
        polys = r.masks.xy if r.masks is not None else [None] * len(boxes)

        for box, tid, conf, poly in zip(boxes, ids, confs, polys):
            bbox, mask = self._clip_box_and_mask(poly, box, frame.shape[:2])
            objs.append({
                "id": tid,
                "bbox": bbox,
                "cls": PERSON_CLASS,
                "conf": conf,
                "mask": mask,
            })

        return objs

    @staticmethod
    def _clip_box_and_mask(poly, box, frame_hw):
        """ Deal with bbox to make sure its legal;
            Rasterise the semantic mask. """
        h, w = frame_hw
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)  # YOLO 给的是 float32
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        if x2 <= x1 or y2 <= y1:
            return bbox, None

        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        if poly is not None and len(poly) >= 3:
            shifted = np.round(np.asarray(poly, dtype=np.float64) - (x1, y1)) # origin at top-left of bbox
            cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
        return bbox, mask

    def project_to_pitch(self, tracked_objects, K, R, t):
        if tracked_objects is None or K is None or R is None or t is None:
            return []
        players_xy = self.get_player_centers(tracked_objects)
        out_bev_players = []
        for x, y in players_xy.values():
            # full P instead of H
            pt = pixel_to_ground(x, y, K, R, t)
            if pt is not None:
                out_bev_players.append(np.array([pt[0], pt[1], 1.0]))
        return out_bev_players

    def get_player_centers(self, tracked_objects):
        """get bottom-centre points (for homography projection)"""
        centers = {}
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            # We use bottom-center because that's where the player touches the pitch
            centers[obj["id"]] = (int((x1 + x2) / 2), int(y2))
        return centers

    def draw_tracks(self, frame, tracked_objects):
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj["bbox"])

            cls = obj["cls"]
            conf = obj["conf"]
            tid = obj["id"]

            color = CLASS_COLORS.get(cls, (255, 0, 0))
            name = CLASS_NAMES.get(cls, f"class_{cls}")

            mask = obj.get("mask")
            if mask is not None:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                offset = np.array([[x1, y1]], dtype=np.int32)
                cv2.drawContours(frame, [c + offset for c in contours], -1, color, 1)

            # bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # label
            label = f"{name} #{tid} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
