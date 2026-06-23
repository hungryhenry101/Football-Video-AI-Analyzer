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
        big_rect_left_bottom = big_rect_left_main = big_rect_left_top = big_rect_right_bottom = big_rect_right_top = big_rect_right_main = None
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
            if class_name == 'Big rect. left bottom':
                big_rect_left_bottom = params
            if class_name == 'Big rect. left main':
                big_rect_left_main = params
            if class_name == 'Big rect. left top':
                big_rect_left_top = params
            if class_name == 'Big rect. right bottom':
                big_rect_right_bottom = params
            if class_name == 'Big rect. right main':
                big_rect_right_main = params
            if class_name == 'Big rect. right top':
                big_rect_right_top = params

        # TODO: more geometric constraint (such as parallel and perpendicular with central line)

        self.intersections['B_TOUCH_AND_HALFWAY_LINES_INTERSECTION'] = self.intersect_lines(middel_line, side_line_bottom)
        self.intersections['T_TOUCH_AND_HALFWAY_LINES_INTERSECTION'] = self.intersect_lines(middel_line, side_line_top)
        self.intersections['TR_PITCH_CORNER'] = self.intersect_lines(side_line_top, side_line_right)
        self.intersections['TL_PITCH_CORNER'] = self.intersect_lines(side_line_left, side_line_top)
        self.intersections['BR_PITCH_CORNER'] = self.intersect_lines(side_line_bottom, side_line_right)
        self.intersections['BL_PITCH_CORNER'] = self.intersect_lines(side_line_bottom, side_line_left)

        self.intersections["L_PENALTY_AREA_TL_CORNER"] = self.intersect_lines(side_line_left, big_rect_left_top)
        self.intersections["L_PENALTY_AREA_TR_CORNER"] = self.intersect_lines(big_rect_left_top, big_rect_left_main)
        self.intersections["L_PENALTY_AREA_BL_CORNER"] = self.intersect_lines(side_line_left, big_rect_left_bottom)
        self.intersections["L_PENALTY_AREA_BR_CORNER"] = self.intersect_lines(big_rect_left_bottom, big_rect_left_main)

        self.intersections["R_PENALTY_AREA_TL_CORNER"] = self.intersect_lines(big_rect_right_top, big_rect_right_main)     # top ∩ main = TL (inner corner)
        self.intersections["R_PENALTY_AREA_TR_CORNER"] = self.intersect_lines(side_line_right, big_rect_right_top)       # touchline ∩ top = TR (outer corner on goal line)
        self.intersections["R_PENALTY_AREA_BL_CORNER"] = self.intersect_lines(big_rect_right_bottom, big_rect_right_main)  # bottom ∩ main = BL (inner corner)
        self.intersections["R_PENALTY_AREA_BR_CORNER"] = self.intersect_lines(side_line_right, big_rect_right_bottom)    # touchline ∩ bottom = BR (outer corner on goal line)
        # Finding the HOMOGRAPHY
        src = []
        dst = []
        
        for name, point in self.intersections.items():
            if point is not None:
                src.append(point)
                # 将物理坐标（以中心为原点）转换为 hg BEV 画布像素坐标
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
            hg_bev_pts = []
            for hg_bev_pt in polyline:
                hg_bev_pts.append([hg_bev_pt[0], hg_bev_pt[1]])

            if not hg_bev_pts:
                continue

            hg_bev_pts = np.array(hg_bev_pts, dtype=np.float32).reshape(-1, 1, 2)
            # 从 BEV 映射回原图
            pts_vid = cv2.perspectiveTransform(hg_bev_pts, H_inv)

            for i in range(len(pts_vid) - 1):
                p1 = tuple(pts_vid[i][0].astype(int))
                p2 = tuple(pts_vid[i + 1][0].astype(int))
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

    def _get_bev_transform(self, scale=10.0):
        """返回从 Homogenous BEV 坐标（米，原点在中心）到 out BEV 像素坐标的变换矩阵 T。

        T = [[scale, 0, tx],
             [0, scale, ty],
             [0, 0,    1]]

        其中 tx, ty 将 (-half_w, -half_h) 平移到 (0, 0)。
        """
        half_w = self.soccer_pitch.PITCH_LENGTH / 2
        half_h = self.soccer_pitch.PITCH_WIDTH / 2
        tx = half_w * scale
        ty = half_h * scale
        return np.array([[scale, 0, tx],
                         [0, scale, ty],
                         [0, 0, 1]], dtype=np.float32)

    def warp(self, img):
        """生成BEV Bird's Eye View"""
        if self.H is None:
            return img

        scale = 10.0  # px / m
        T = self._get_bev_transform(scale)
        H_bev = T @ self.H

        out_w = int(self.soccer_pitch.PITCH_LENGTH * scale)
        out_h = int(self.soccer_pitch.PITCH_WIDTH * scale)

        return cv2.warpPerspective(img, H_bev, (out_w, out_h))

    def warp_points(self, hg_points, scale=10.0):
        """将齐次坐标点（project_to_pitch 的输出，单位为m）转换为 out BEV 像素坐标。

        :param points: 齐次坐标列表，元素是 (x, y, w)
        :param scale: px/m，与 warp() 中的 scale 一致
        :return: [(x_bev, y_bev), ...] out BEV 像素坐标列表
        """
        if not hg_points:
            return []

        T = self._get_bev_transform(scale)
        # 矩阵乘法替代 Python 循环
        pts = np.array([p[:3] for p in hg_points], dtype=np.float32).T  # 3×N
        p = T @ pts  # 3×N
        p = p / p[2]  # 归一化齐次坐标
        return [(int(p[0, i]), int(p[1, i])) for i in range(p.shape[1])]