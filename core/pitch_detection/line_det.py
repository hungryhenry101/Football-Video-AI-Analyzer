from collections import deque
import random
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.models.segmentation import deeplabv3_resnet50
from .soccerpitch import SoccerPitch

MEAN_PATH = 'models/pitch_seg_npy/mean.npy'
STD_PATH = 'models/pitch_seg_npy/std.npy'
MODEL_PATH = 'models/soccer_pitch_segmentation.pth'

class SegmentationNetwork:
    def __init__(self, project_dir, width, height, device):
        self.width = width
        self.height = height
        self.device = device

        self.mean = np.load(os.path.join(project_dir, MEAN_PATH))
        self.std = np.load(os.path.join(project_dir, STD_PATH))
        model = nn.DataParallel(deeplabv3_resnet50(weights=None,weights_backbone=None, num_classes=29)) # class 0 is bg

        self.init_weight(model, nn.init.kaiming_normal_,
                         nn.BatchNorm2d, 1e-3, 0.1,
                         mode='fan_in')

        state_dict = torch.load(os.path.join(project_dir, MODEL_PATH), map_location=self.device)
        model.load_state_dict(state_dict["model"])
        model.eval()
        self.model = model.to(self.device)

        print(f"SegmentationNetwork initialized on {self.device} with {width}x{height} resolution")


    def init_weight(self, feature, conv_init, norm_layer, bn_eps, bn_momentum,
                    **kwargs):
        for name, m in feature.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                conv_init(m.weight, **kwargs)
            elif isinstance(m, norm_layer):
                m.eps = bn_eps
                m.momentum = bn_momentum
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img): # img: BGR image
        img = cv2.resize(img, (self.width, self.height))
        img = np.asarray(img, np.float32) / 255. # Normalize
        img = (img - self.mean) / self.std # Standardize
        img = img.transpose((2, 0, 1)) # transpose to Channel, Height, Width
        img = torch.from_numpy(img).to(self.device, dtype=torch.float32).unsqueeze(0) # Add batch to shape

        with torch.no_grad():
            result = self.model(img)
        output = result['out'][0].cpu().numpy() # Classes, Height, Width
        output = np.asarray(np.argmax(output, axis=0), dtype=np.uint8) # argmax along the class dimension to get the most likely class for each pixel

        return output


class LineDetector:
    def __init__(self, project_dir, width, height, device, disk_radius=6, max_dist=40):
        """
        :param disk_radius: radius of the circles used for synthesizing the mask
        :param max_dist: maximal distance between two points to be joined in a polyline
        """
        self.width = width
        self.height = height
        self.disk_radius = disk_radius
        self.max_dist = max_dist
        self.seg_network = SegmentationNetwork(project_dir, self.width, self.height, device)

    def detect(self, img):
        """
        Finds the extremities or fitted arcs of each detected class in the semantic mask.
        :param semantic_mask: 2D mask of predicted classes
        :return: dictionary {class_name: [{'x': x1, 'y': y1}, ...]}
        """

        semantic_mask = self.seg_network.forward(img)
        skeletons = self._skeletonize(semantic_mask)
        results = self._fit(skeletons)
        return results

    def _fit(self, skeletons):
        results = dict()
        for class_name, skel_img in skeletons.items():
            points = np.transpose(np.nonzero(skel_img))
            if len(points) == 0:
                continue

            polyline_list = self._join_points(points)
            if not polyline_list:
                continue

            longest_polyline = max(polyline_list, key=len)

            ### to return the raw point
            # results[class_name] = longest_polyline

            # Swap [row, col] to [col, row] for image coordinates (x, y) with three dimensions
            pts_array = np.array(longest_polyline)[:, [1, 0]].reshape(-1, 1, 2).astype(np.float32)
            if ('Circle' not in class_name and
                    'Goal' not in class_name and
                    'Small rect.' not in class_name and
                    len(longest_polyline)>4):
                line_params = cv2.fitLine(pts_array, cv2.DIST_HUBER, 1.0, 0.01, 0.01).ravel()
                results[class_name] = line_params

            ### let's fit the arc AFTER homography
            # else:
            #     fitted = fit_ellipse_arc(pts_array)
            #     if fitted:
            #         results[class_name] = fitted

        return results

    def _skeletonize(self, semantic_mask):
        skeletons = dict()
        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(semantic_mask, kernel, iterations=1)

        for k, class_name in enumerate(SoccerPitch.lines_classes):
            # 生成布尔掩码并转换为 uint8 (0 或 255)
            mask = (eroded_mask == k + 1).astype(np.uint8) * 255
            if mask.sum() > 0:
                skeleton = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
                if skeleton is not None and np.any(skeleton):
                    skeletons[class_name] = skeleton
        return skeletons

    def _join_points(self, point_list):
        polylines = []
        if not len(point_list):
            return polylines

        remaining_points = [p.astype(np.float32) for p in point_list]

        while len(remaining_points) > 0:
            polyline = deque([remaining_points.pop(0)])
            head = polyline[0]
            tail = polyline[-1]

            changed = True
            while changed:
                changed = False
                best_dist = self.max_dist
                best_idx = -1
                best_side = None # 'head' or 'tail'

                for i, point in enumerate(remaining_points):
                    d_head = np.linalg.norm(point - head)
                    d_tail = np.linalg.norm(point - tail)

                    if d_head < best_dist:
                        best_dist = d_head
                        best_idx = i
                        best_side = 'head'
                    if d_tail < best_dist:
                        best_dist = d_tail
                        best_idx = i
                        best_side = 'tail'

                if best_idx != -1:
                    point = remaining_points.pop(best_idx)
                    if best_side == 'head':
                        polyline.appendleft(point)
                        head = polyline[0]
                    else:
                        polyline.append(point)
                        tail = polyline[-1]
                    changed = True

            polylines.append(list(polyline))
        return polylines

    def _synthesize_mask(self, semantic_mask): # WILL BE REPLACED BY SKELETONIZE
        """
        1. pick a random pixel
        2. draw a circle around the pixel with r of disk_radius
        3. calc the center of mass of all the white pixels in the circle
        4. shift towards the center and loop
        ps. literally like K-means clustering?
        """
        mask = semantic_mask.copy().astype(np.uint8)
        points = np.transpose(np.nonzero(mask))
        disks = []
        while len(points):
            start = random.choice(points)
            dist = 10.
            success = True
            while dist > 1.:
                enough_support, center = self._get_support_center(mask, start, self.disk_radius)
                if not enough_support:
                    bad_point = np.round(center).astype(np.int32)
                    cv2.circle(mask, (bad_point[1], bad_point[0]), self.disk_radius, 0, -1)
                    success = False
                    break
                dist = np.sqrt(np.sum(np.square(center - start)))
                start = center
            if success:
                disks.append(np.round(start).astype(np.int32))
                cv2.circle(mask, (disks[-1][1], disks[-1][0]), self.disk_radius, 0, -1)
            points = np.transpose(np.nonzero(mask))
        return disks

    def _get_support_center(self, mask, start, disk_radius, min_support=0.1):
        x = int(start[0])
        y = int(start[1])
        result = [x, y]
        xstart = max(0, x - disk_radius)
        xend = min(mask.shape[0] - 1, x + disk_radius)
        ystart = max(0, y - disk_radius)
        yend = min(mask.shape[1] - 1, y + disk_radius)

        # Optimization: use slicing and masking instead of nested loops
        y_grid, x_grid = np.ogrid[xstart:xend + 1, ystart:yend + 1]
        dist_sq = (x_grid - y) ** 2 + (y_grid - x) ** 2
        circle_mask = dist_sq < disk_radius ** 2

        mask_roi = mask[xstart:xend + 1, ystart:yend + 1]
        valid_pixels = np.logical_and(circle_mask, mask_roi > 0)

        support_pixels = np.sum(valid_pixels)
        if support_pixels < min_support * np.square(disk_radius) * np.pi:
            return False, np.array(start)

        coords = np.argwhere(valid_pixels)
        result[0] = np.mean(coords[:, 0] + xstart)
        result[1] = np.mean(coords[:, 1] + ystart)

        return True, np.array(result)


class Visualizer:
    def __init__(self):
        soccer_pitch = SoccerPitch()
        self.palette = soccer_pitch.palette

    def draw_lines(self, canvas, detections):
        h, w = canvas.shape[:2]
        for class_name, params in detections.items():
            vx, vy, x, y = params
            if abs(vx) < 1e-5:
                continue
            lefty = int(y - x * vy / vx)
            righty = int(y + (w - x) * vy / vx)
            color = self.palette[class_name][::-1] # RGB to BGR
            cv2.line(canvas, (w - 1, righty), (0, lefty), color, 2)
