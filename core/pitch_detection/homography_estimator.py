import numpy as np
from .soccerpitch import SoccerPitch
import cv2

class HomographyEstimator:
    def __init__(self, width, height):
        self.H = None
        self.soccer_pitch = SoccerPitch()
        self.width = width
        self.height = height
        self.intersections = {}

    def estimate(self, lines):
        """
        Estimate homography using intersections of detected lines.
        """
        print("estimating homography...")

        # 检测
        #  1. 中线与上下边线交点：
        #  T_TOUCH_AND_HALFWAY_LINES_INTERSECTION, B_TOUCH_AND_HALFWAY_LINES_INTERSECTION
        #  2. 场地四个角：TL_PITCH_CORNER, BL_PITCH_CORNER, TR_PITCH_CORNER, BR_PITCH_CORNER

        middel_line = side_line_bottom = side_line_left = side_line_right = side_line_top = None
        for class_name, params in lines.items():
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

        # TODO: more geometric constraint (such as parallel and perpendicular with central line)

        self.intersections['B_TOUCH_AND_HALFWAY_LINES_INTERSECTION'] = self.intersect_lines(middel_line, side_line_bottom)
        self.intersections['T_TOUCH_AND_HALFWAY_LINES_INTERSECTION'] = self.intersect_lines(middel_line, side_line_top)
        self.intersections['TR_PITCH_CORNER'] = self.intersect_lines(side_line_top, side_line_right)
        self.intersections['TL_PITCH_CORNER'] = self.intersect_lines(side_line_left, side_line_top)
        self.intersections['BR_PITCH_CORNER'] = self.intersect_lines(side_line_bottom, side_line_right)
        self.intersections['BL_PITCH_CORNER'] = self.intersect_lines(side_line_bottom, side_line_left)

        # Finding the HOMOGRAPHY
        src = []
        dst = []
        
        for name, point in self.intersections.items():
            if point is not None:
                src.append(point)
                # 将物理坐标（以中心为原点）转换为 BEV 画布像素坐标
                phys_pt = self.soccer_pitch.point_dict[name]
                dst.append(phys_pt)
                
        if len(src) >= 4 and len(dst) >= 4:
            src = np.array(src, dtype=np.float32)
            dst = np.array(dst, dtype=np.float32)
            self.H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        else:
            print("not enough intersections")

    def draw_pitch_lines(self, img, color=(0, 255, 0), thickness=2):
        """将标准球场线条重投影到原始图像上"""
        if self.H is None:
            return img

        H_inv = np.linalg.inv(self.H)  # inverse

        # 获取球场所有线条的离散点
        field_polylines = self.soccer_pitch.sample_field_points()
        for name, polyline in field_polylines.items():
            pts_bev = []
            for pt_phys in polyline:
                pts_bev.append([pt_phys[0], pt_phys[1]])

            if not pts_bev:
                continue

            pts_bev = np.array(pts_bev, dtype=np.float32).reshape(-1, 1, 2)
            # 透视变换：从 BEV 映射回原图
            pts_img = cv2.perspectiveTransform(pts_bev, H_inv)

            for i in range(len(pts_img) - 1):
                p1 = tuple(pts_img[i][0].astype(int))
                p2 = tuple(pts_img[i + 1][0].astype(int))
                # 过滤掉变换后可能出现的异常坐标点
                if abs(p1[0]) > 1e4 or abs(p1[1]) > 1e4 or abs(p2[0]) > 1e4 or abs(p2[1]) > 1e4:
                    continue
                cv2.line(img, p1, p2, color, thickness)
        return img

    def intersect_lines(self, line1, line2):
        if line1 is None or line2 is None or self.is_parallel(line1, line2):
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
        A = np.array([[A1, B1],[A2, B2]])
        C = np.array([C1, C2])

        # 求解
        try:
            x, y = np.linalg.solve(A, C)
            return int(x), int(y)
        except np.linalg.LinAlgError:
            return None

    def is_parallel(self, line1, line2, max_angle=20):
        """用「叉积 cross product」 判断平行"""
        tolerance = np.sin(max_angle * np.pi / 180)
        v1 = line1[:2]
        v2 = line2[:2]
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        return abs(cross_product) < tolerance

    def warp(self, img):
        """生成俯视图BEV Bird's Eye View"""
        if self.H is None:
            return img

        scale = 10.0  # px / m
        half_w = self.soccer_pitch.PITCH_LENGTH / 2
        half_h = self.soccer_pitch.PITCH_WIDTH / 2

        # 将 (-half_w, -half_h) 平移到 (0,0)
        tx = half_w * scale
        ty = half_h * scale

        T = np.array([[scale, 0, tx],
                      [0, scale, ty],
                      [0, 0, 1]], dtype=np.float32)

        H_bev = T @ self.H

        out_w = int(2 * half_w * scale)
        out_h = int(2 * half_h * scale)

        return cv2.warpPerspective(img, H_bev, (out_w, out_h))