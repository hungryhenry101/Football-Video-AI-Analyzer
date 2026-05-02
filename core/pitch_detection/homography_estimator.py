import numpy as np
from soccerpitch import SoccerPitch
import cv2

class HomographyEstimator:
    def __init__(self):
        self.H = None
        self.soccer_pitch = SoccerPitch()

    def estimate(self, line_params_dict, image_shape):
        """
        Estimate homography using detected dominant lines.
        Currently uses a simple heuristic: intersection of lines.
        """
        points_det = []
        points_dim = []

        # TODO: 先检测
        #  1. 中线与上下边线交点：
        #  T_TOUCH_AND_HALFWAY_LINES_INTERSECTION, B_TOUCH_AND_HALFWAY_LINES_INTERSECTION
        #  2. 场地四个角：TL_PITCH_CORNER, BL_PITCH_CORNER, TR_PITCH_CORNER, BR_PITCH_CORNER


        def is_parallel(line1, line2, tolerance=0.1):
            """用「叉积 cross product」 判断平行"""

            # 理论上，叉积为0时，两条线平行
            # ps. 点积为0时，两条线垂直
            cross_product = np.cross(line1, line2)
            return abs(cross_product) < tolerance


    def warp(self, img):
        if self.H is None:
            return img
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, self.H, (w, h))
