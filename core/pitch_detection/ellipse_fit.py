import numpy as np
import cv2

def fit_ellipse_arc(points):
    """
    fits ellipse and calculate the range of angle, then return an arc

    :param points: List of points in (x, y) format
    :return: dict
        {
            'center': (x, y),
            'axes': (a, b),
            'angle': angle_deg,
            'start_angle_rad': start, # elliptical local coord sys
            'end_angle_rad': end,
        }
    """
    if len(points) < 5:
        print('Not enough points to fit ellipse')
        return None

    points = np.array(points)

    ellipse = cv2.fitEllipse(points)
    (xc, yc), (d1, d2), angle_deg = ellipse
    a, b = d1 / 2.0, d2 / 2.0   # semi-axes

    # calculate angles in local coords for each point
    theta_rad = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    pts_centered = points - np.array([xc, yc])
    # convert the point from global coords to local coords
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    pts_local = pts_centered @ R.T
    # equation of ellipse: (x_local/a)^2 + (y_local/b)^2 = 1
    # local angle = arctan2(y_local/b, x_local/a)
    pt_angles = np.arctan2(pts_local[:, 1] / b, pts_local[:, 0] / a)

    # sort and find the biggest gap
    pt_angles = np.sort(pt_angles)
    gaps = np.diff(pt_angles)
    max_gap_idx = np.argmax(gaps)

    if gaps[max_gap_idx] > np.pi:
        # max gap is greater than pi => the arc is the minor one 劣弧
        start_angle = pt_angles[max_gap_idx + 1]
        end_angle = pt_angles[max_gap_idx] + 2 * np.pi
    else:
        # otherwise the arc is the major one 优弧
        start_angle = pt_angles[0]
        end_angle = pt_angles[-1]

    # return an arc
    return {
        'center': (float(xc), float(yc)),
        'axes': (float(a), float(b)), # semi-axes
        'angle_deg': float(angle_deg),
        'start_angle_rad': float(start_angle),
        'end_angle_rad': float(end_angle),
    }