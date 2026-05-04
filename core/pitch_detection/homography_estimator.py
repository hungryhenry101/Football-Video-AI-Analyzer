import numpy as np
from ultralytics.utils.torch_utils import is_parallel
from soccerpitch import SoccerPitch
import cv2

class HomographyEstimator:
    def __init__(self, canva, lines):
        self.H = None
        self.soccer_pitch = SoccerPitch()
        self.canva = canva
        self.lines = lines

    def estimate(self):
        """
        Estimate homography using intersections of detected lines.
        """
        print("estimating homography...")

        # TODO: 先检测
        #  1. 中线与上下边线交点：
        #  T_TOUCH_AND_HALFWAY_LINES_INTERSECTION, B_TOUCH_AND_HALFWAY_LINES_INTERSECTION
        #  2. 场地四个角：TL_PITCH_CORNER, BL_PITCH_CORNER, TR_PITCH_CORNER, BR_PITCH_CORNER

        middel_line = side_line_bottom = side_line_left = side_line_right = side_line_top = None
        for class_name, params in self.lines.items():
            if class_name == 'Middle line':
                middel_line = params
            if class_name == 'Side line bottom':
                side_line_bottom = params
            if class_name == 'Side line left':
                side_line_left = params
            if class_name == 'Side line right':
                side_line_right = params
            if class_name == 'Side line top':
                side_line_top = params

        intersections = {}
        intersections['MIDDLE_and_SIDE_BOTTOM'] = self.intersect_lines(middel_line, side_line_bottom)
        intersections['MIDDLE_and_SIDE_TOP'] = self.intersect_lines(middel_line, side_line_top)
        intersections['SIDE_TOP_and_SIDE_RIGHT'] = self.intersect_lines(side_line_top, side_line_right)
        intersections['SIDE_TOP_and_SIDE_LEFT'] = self.intersect_lines(side_line_left, side_line_top)
        intersections['SIDE_BOTTOM_and_SIDE_RIGHT'] = self.intersect_lines(side_line_bottom, side_line_right)
        intersections['SIDE_BOTTOM_and_SIDE_LEFT'] = self.intersect_lines(side_line_bottom, side_line_left)

        for intersection in intersections.values():
            if intersection is not None:
                cv2.circle(self.canva, intersection, 5, (0, 0, 255), -1)

    def intersect_lines(self, line1, line2):
        if line1 is None or line2 is None or self.is_parallel(line1, line2):
            print("lines are None or paralleled")
            return None
        vx1, vy1, x1, y1 = line1.flatten()
        vx2, vy2, x2, y2 = line2.flatten()

        # 转换为一般式: Ax + By = C
        A1 = vy1
        B1 = -vx1
        C1 = A1 * x1 + B1 * y1

        A2 = vy2
        B2 = -vx2
        C2 = A2 * x2 + B2 * y2

        # 构造方程组
        A = np.array([[A1, B1],
                      [A2, B2]])
        C = np.array([C1, C2])

        # 求解
        x, y = np.linalg.solve(A, C)
        return int(x), int(y)

    def is_parallel(self, line1, line2, max_angle=20):
        """用「叉积 cross product」 判断平行"""
        # line_params are (vx, vy, x, y) from cv2.fitLine, where (vx, vy) is the normalized direction vector.
        # Two 2D vectors (v1x, v1y) and (v2x, v2y) are parallel if v1x*v2y - v1y*v2x = 0.

        tolerance = np.sin(max_angle * np.pi / 180)
        v1 = line1[:2]
        v2 = line2[:2]
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        return abs(cross_product) < tolerance

    def warp(self, img):
        if self.H is None:
            return img
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, self.H, (w, h))
