from ultralytics import YOLO
import cv2
import numpy as np
from core.pnl.projection_utils import pixel_to_ground

CLASS_NAMES = {
    0: "ball",
    1: "goalkeeper", ## TODO: identifying GK 1) color clustering; 2) position
    2: "player",
    3: "referee",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 255, 255),
    3: (0, 255, 255),
}

class PlayerTracker:
    def __init__(self, model_path, device, tracker_config="config/botsort.yaml", conf_thres=0.2):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.conf_thres = conf_thres
        self.device = device

    def update(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf_thres,
            tracker=self.tracker_config,
            device=self.device,
            verbose=False
        )

        objs = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # 一次性从 GPU 拉取所有检测结果，减少同步开销
            boxes_xyxy, ids, confs, clss = [
                t.cpu().numpy() for t in [
                    results[0].boxes.xyxy, results[0].boxes.id,
                    results[0].boxes.conf, results[0].boxes.cls
                ]
            ]
            ids = ids.astype(int)
            clss = clss.astype(int)

            for box, tid, conf, cls in zip(boxes_xyxy, ids, confs, clss):
                objs.append({
                    "id": tid,
                    "bbox": box,
                    "cls": cls,
                    "conf": conf,
                })

        return objs

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
        """get bottom-center points (for homography projection)"""
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

            # bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # label
            label = f"{name} #{tid} {conf:.2f}"
            cv2.putText(frame,label,(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        return frame
