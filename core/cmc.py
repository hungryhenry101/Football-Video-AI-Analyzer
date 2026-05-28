import numpy as np
import cv2

class CameraMotionCompensator:
    def __init__(self):
        self.prev_intersections = None

    def calc_cmc(self, intersections):
        if self.prev_intersections is None:
            self.prev_intersections = intersections
            return np.eye(2,3)

        src_pts = []
        dst_pts = []
        for name, point in intersections:
            if name in self.prev_intersections:
                dst_pts.append(point)
                src_pts.append(self.prev_intersections[name])

        # for next frame
        self.prev_intersections = intersections

        if len(dst_pts) < 3:
            return np.eye(2,3)

        #calc
        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)
        matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        return matrix if matrix is not None else np.eye(2, 3)

    def visualize(self, matrix, canva):
        tx, ty = matrix[0, 2], matrix[1, 2]  # Extract translation
        center = (100, 100)  # Center of a small visualization box
        end_point = (int(center[0] + tx * 5), int(center[1] + ty * 5))
        cv2.arrowedLine(canva, center, end_point, (0, 255, 0), 2)

## TEST
if __name__ == "__main__":
    from pitch_detection.line_det import LineDetector
    cmc = CameraMotionCompensator()
    cap = cv2.VideoCapture("./input_vids/test1.mp4")