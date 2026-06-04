import os
os.chdir("../")

import cv2
from core.pitch_detection.line_det import LineDetector, Visualizer
from core.pitch_detection.homography_estimator import HomographyEstimator

def main():
    test_file = "./input_vids/test1.png"
    image = cv2.imread(test_file)

    width, height = 735, 404
    line_detection = LineDetector("./", width, height)
    detection = line_detection.detect(image)

    canva = cv2.resize(image, (width, height)).copy()
    visualizer = Visualizer()
    # 可视化 1: 在原图上画识别到的线
    visualizer.draw_lines(canva, detection)

    homo_est = HomographyEstimator(width, height)
    homo_est.estimate(detection)

    # 可视化 2: 在原图上画根据 Homography 得出的投影球场线
    projected_img = homo_est.draw_pitch_lines(canva.copy())

    # 可视化 3: 俯视图
    bev_img = homo_est.warp(canva)

    cv2.imshow("Original with Lines", canva)
    cv2.imshow("Projected Pitch Lines", projected_img)
    cv2.imshow("Bird's Eye View", bev_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()