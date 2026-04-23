import copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.models.segmentation import deeplabv3_resnet50
from soccerpitch import SoccerPitch
import random
import os
from ellipse_fit import fit_ellipse_arc

MEAN_PATH = 'models/pitch_seg_npy/mean.npy'
STD_PATH = 'models/pitch_seg_npy/std.npy'
MODEL_PATH = 'models/soccer_pitch_segmentation.pth'

class SegmentationNetwork:
    def __init__(self, project_dir, width=640, height=360):
        self.width = width
        self.height = height

        self.mean = np.load(os.path.join(project_dir, MEAN_PATH))
        self.std = np.load(os.path.join(project_dir, STD_PATH))
        model = nn.DataParallel(deeplabv3_resnet50(weights=None,weights_backbone=None, num_classes=29)) # class 0 is bg

        self.init_weight(model, nn.init.kaiming_normal_,
                         nn.BatchNorm2d, 1e-3, 0.1,
                         mode='fan_in')

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

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

    def analyse_img(self, img): # img: BGR image
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
    def __init__(self, disk_radius=6, max_dist=40):
        """
        :param disk_radius: radius of the circles used for synthesizing the mask
        :param max_dist: maximal distance between two points to be joined in a polyline
        """
        self.disk_radius = disk_radius
        self.max_dist = max_dist

    def detect(self, semantic_mask):
        """
        Finds the extremities or fitted arcs of each detected class in the semantic mask.
        :param semantic_mask: 2D mask of predicted classes
        :return: dictionary {class_name: [{'x': x1, 'y': y1}, ...]}
        """

        skeletons = self._generate_class_synthesis(semantic_mask)
        results = self._fit(skeletons)
        return results

    def _fit(self, buckets):
        results = dict()
        for class_name, disks_list in buckets.items():
            polyline_list = self._join_points(disks_list)
            if not polyline_list:
                continue

            longest_polyline = max(polyline_list, key=len)
            # Flatten longest_polyline to a list of (x,y) int points for fitting
            xy_points = [(int(p[1]), int(p[0])) for p in longest_polyline]

            if 'Circle' in class_name and len(longest_polyline) >= 5:
                # fit an arc
                fitted = fit_ellipse_arc(xy_points)

                if fitted:
                    results[class_name] = fitted

            else:
                # longest_polyline[0] is [row, col] -> [y, x]
                results[class_name] = [
                    xy_points[0], xy_points[-1]
                ]
        return results

    def _generate_class_synthesis(self, semantic_mask):
        buckets = dict()
        kernel = np.ones((5, 5), np.uint8)
        # Erode to remove small noise
        eroded_mask = cv2.erode(semantic_mask, kernel, iterations=1)

        for k, class_name in enumerate(SoccerPitch.lines_classes):
            # Class indices are k+1 because 0 is background
            mask = eroded_mask == k + 1
            if mask.sum() > 0:
                disk_list = self._synthesize_mask(mask)
                if len(disk_list):
                    buckets[class_name] = disk_list
        return buckets

    def _get_support_center(self, mask, start, disk_radius, min_support=0.1):
        x = int(start[0])
        y = int(start[1])
        support_pixels = 1
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

    def _synthesize_mask(self, semantic_mask):
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


# TEST
if __name__ == '__main__':
    test_file = "../../input_vids/test_right.png"
    image = cv2.imread(test_file)

    seg_network = SegmentationNetwork("../../", 735, 404)
    semantic_mask = seg_network.analyse_img(image)

    line_detection = LineDetector()
    detection = line_detection.detect(semantic_mask)
    print(detection)

    width, height = seg_network.width, seg_network.height
    canva = np.zeros((height, width, 3), dtype=np.uint8)
    soccer_pitch = SoccerPitch()
    for class_name, fitted in detection.items():
        color = soccer_pitch.palette[class_name]
        # Convert list of dicts to numpy array for cv2.polylines
        if "Circle" in class_name:
            center = (int(fitted['center'][0]), int(fitted['center'][1]))
            axes = (int(fitted['axes'][0]*2), int(fitted['axes'][1]*2))
            angle = fitted['angle_deg']
            start_angle = np.rad2deg(fitted['start_angle_rad'])
            end_angle = np.rad2deg(fitted['end_angle_rad'])
            cv2.ellipse(canva, center, axes, angle, start_angle, end_angle, color, 2)
        else:
            cv2.line(canva, fitted[0], fitted[1], color, 2)

    cv2.namedWindow('image')
    cv2.imshow("image", canva)
    cv2.waitKey(0)