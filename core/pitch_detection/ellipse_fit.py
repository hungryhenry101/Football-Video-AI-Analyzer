# 等学了椭圆再来这里理解代码吧。。。
import numpy as np
import cv2

def fit_ellipse_and_sample(points, width, height):
    """
    Fits an ellipse to a set of points and returns sampled points along the fitted curve.
    Points are in [row, col] format.

    :param points: List of points in [row, col] format (y, x)
    :param width: Image width for normalization
    :param height: Image height for normalization
    :return: List of dicts {'x': x, 'y': y} in normalized coordinates, or None if fitting fails.
    """
    if len(points) < 5:
        print('Not enough points to fit ellipse')
        return None

    # points are in [row, col] (y, x), cv2.fitEllipse expects [col, row] (x, y)
    pts = np.array([[p[1], p[0]] for p in points], dtype=np.float32)
    ellipse = cv2.fitEllipse(pts)
    (xc, yc), (d1, d2), angle = ellipse
    a, b = d1 / 2, d2 / 2

    theta_rad = np.deg2rad(angle)
    cos_theta, sin_theta = np.cos(theta_rad), np.sin(theta_rad)

    # Find angular range of original points in local ellipse coordinates
    pts_centered = pts - np.array([xc, yc])
    # Rotation matrix to align with ellipse axes
    R = np.array([[cos_theta, sin_theta],
                  [-sin_theta, cos_theta]])
    pts_local = pts_centered @ R.T
    pt_angles = np.arctan2(pts_local[:, 1] / b, pts_local[:, 0] / a)

    # Find the arc range, handling potential wrap-around at -pi/pi
    pt_angles = np.sort(pt_angles)
    gaps = np.diff(pt_angles)
    max_gap_idx = np.argmax(gaps)
    if gaps[max_gap_idx] > np.pi:
        start_angle = pt_angles[max_gap_idx + 1]
        end_angle = pt_angles[max_gap_idx] + 2 * np.pi
    else:
        start_angle = pt_angles[0]
        end_angle = pt_angles[-1]

    num_samples = max(20, len(points))
    angles = np.linspace(start_angle, end_angle, num_samples)

    sampled = []
    for t in angles:
        x_l, y_l = a * np.cos(t), b * np.sin(t)
        x = xc + x_l * cos_theta - y_l * sin_theta
        y = yc + x_l * sin_theta + y_l * cos_theta
        sampled.append({'x': float(x / width), 'y': float(y / height)})
    return sampled
