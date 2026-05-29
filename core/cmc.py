import numpy as np
import cv2

class CameraMotionCompensator:
    def __init__(self):
        self.prev_intersections = None

    def calc_cmc(self, intersections):
        if self.prev_intersections is None:
            self.prev_intersections = intersections.copy()
            return np.eye(2, 3)

        src_pts = []
        dst_pts = []
        for name, point in intersections.items():
            if point is None:
                continue
            if name in self.prev_intersections and self.prev_intersections[name] is not None:
                dst_pts.append(point)
                src_pts.append(self.prev_intersections[name])

        # for next frame
        self.prev_intersections = intersections.copy()

        if len(dst_pts) < 3:
            return np.eye(2, 3)

        #calc
        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)
        matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        return matrix if matrix is not None else np.eye(2, 3)

    def visualize(self, matrix, canva):
        # Draw a compact overlay in the top-left to visualize camera translation
        tx, ty = float(matrix[0, 2]), float(matrix[1, 2])  # Extract translation

        # a small overlay
        overlay_w, overlay_h = 200, 200
        overlay = np.zeros((overlay_h, overlay_w, 3), dtype=np.uint8)
        overlay[:] = (40, 40, 40)  # dark bg

        # Dynamic scale so small translations are visible
        scale = 50.0
        center = (overlay_w // 2, overlay_h // 2)
        end_point = (int(center[0] + tx * scale), int(center[1] + ty * scale))

        cv2.arrowedLine(overlay, center, end_point, (0, 0, 255), 3, tipLength=0.3)
        text = f"tx={tx:.2f}, ty={ty:.2f}"
        cv2.putText(overlay, text, (8, overlay_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        h, w = canva.shape[:2]
        x0, y0 = 8, 8
        x1, y1 = x0 + overlay_w, y0 + overlay_h
        canva[y0:y1, x0:x1] = overlay

## TEST
if __name__ == "__main__":
    from .pitch_detection.line_det import LineDetector, Visualizer
    from .pitch_detection.homography_estimator import HomographyEstimator
    cmc = CameraMotionCompensator()
    cap = cv2.VideoCapture("./input_vids/test2.mp4")
    width, height = 735, 404
    line_detection = LineDetector("./", width, height)
    visualizer = Visualizer()
    homo_est = HomographyEstimator(width, height)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (width, height))

        detection = line_detection.detect(frame_resized)
        canva = frame_resized.copy()
        visualizer.draw_lines(canva, detection)
        homo_est.estimate(detection)
        projected_img = homo_est.draw_pitch_lines(canva.copy())
        bev_img = homo_est.warp(canva)
        cv2.imshow("lines dets", canva)
        cv2.imshow("Projected Pitch Lines", projected_img)
        cv2.imshow("Bird's Eye View", bev_img)

        # HERE's the real test
        intersections = homo_est.intersections
        M = cmc.calc_cmc(intersections)
        print(M)
        cmc.visualize(M, frame_resized)
        cv2.imshow("frame", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()