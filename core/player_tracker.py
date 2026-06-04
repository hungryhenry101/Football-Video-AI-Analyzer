from ultralytics import YOLO
import cv2

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
    def __init__(self, model_path, tracker_config="botsort.yaml", conf_thres=0.2):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.conf_thres = conf_thres

    def update(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf_thres,
            tracker=self.tracker_config,
        )

        objs = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, conf, cls in zip(boxes_xyxy, ids, confs, clss):
                objs.append({
                    "id": tid,
                    "bbox": box,
                    "cls": cls,
                    "conf": conf,
                })

        return objs

    def get_player_centers(self, tracked_objects):
        """get bottom-center points (for homography projection)"""
        centers = {}
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            # We use bottom-center because that's where the player touches the pitch
            centers[obj["id"]] = (int((x1 + x2) / 2), int(y2))
        return centers

    def project_to_pitch(self, tracked_objects, H):
        players = self.get_player_centers(tracked_objects)



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
