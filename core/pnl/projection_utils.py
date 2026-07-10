"""
3D ray casting and projection utilities for camera-based ball tracking.

Provides the bridge between 2D image pixel coordinates and 3D world coordinates
using a full pinhole camera model (intrinsics K, rotation R, position t).

For aerial balls: the full camera matrix P gives us the 3D ray through any pixel.
Combined with apparent ball size (bbox width), we estimate the perpendicular
depth Z along the optical axis, then compute the true 3D world position.
"""

import numpy as np

# Regulation soccer ball diameter in meters
BALL_DIAMETER_M = 0.22


def pixel_to_ray(u, v, K, R, t, k1=0.0, k2=0.0):
    """Back-project an image pixel to a 3D unit ray in world coordinates.

    Args:
        u, v: pixel coordinates (float)
        K: 3×3 camera intrinsic matrix
        R: 3×3 rotation matrix (world → camera)
        t: (3,) camera center in world coordinates
        k1, k2: radial distortion coefficients (default 0)

    Returns:
        origin: (3,) camera center in world coordinates
        direction: (3,) unit vector of the ray in world coordinates
    """
    p = np.array([u, v, 1.0], dtype=np.float64)
    p_norm = np.linalg.inv(K) @ p

    # Inverse radial distortion
    x_d, y_d = p_norm[0], p_norm[1]
    r2 = x_d**2 + y_d**2
    factor = 1.0 + k1 * r2 + k2 * (r2**2)
    x_u = x_d / factor
    y_u = y_d / factor

    # Direction in camera coords → world coords, then normalize
    d_cam = np.array([x_u, y_u, 1.0], dtype=np.float64)
    direction = R.T @ d_cam
    direction /= np.linalg.norm(direction)

    origin = np.asarray(t, dtype=np.float64)
    return origin, direction


def pixel_to_3d(u, v, bbox_width_px, K, R, t, k1=0.0, k2=0.0):
    """Image pixel → 3D world position using apparent ball size.

    The pinhole model relationship is:
        Z = (focal_px * real_diameter_m) / bbox_width_px

    where Z is the **perpendicular depth** along the camera's optical axis
    (NOT the Euclidean distance along the ray). The 3D point is then:

        P_cam = Z * [x_u, y_u, 1]          ← point in camera coordinates
        P_world = R^T @ P_cam + t           ← transform to world

    Args:
        u, v: ball center pixel coordinates
        bbox_width_px: width of ball bounding box in pixels
        K: 3×3 intrinsics (at the frame resolution used for detection)
        R: 3×3 rotation (world → camera)
        t: (3,) camera center in world coords
        k1, k2: distortion coefficients

    Returns:
        (3,) estimated 3D position [x, y, z] in meters (world coords),
        or None if bbox_width_px ≤ 0
    """
    if bbox_width_px <= 0:
        return None

    focal_px = float(K[0, 0])
    if focal_px <= 0:
        return None

    # 1. Normalize pixel → undistorted camera coordinates
    p = np.array([u, v, 1.0], dtype=np.float64)
    p_norm = np.linalg.inv(K) @ p
    x_d, y_d = p_norm[0], p_norm[1]

    # Radial distortion correction
    r2 = x_d**2 + y_d**2
    factor = 1.0 + k1 * r2 + k2 * (r2**2)
    x_u = x_d / factor
    y_u = y_d / factor

    # 2. Perpendicular depth from apparent size
    Z = (focal_px * BALL_DIAMETER_M) / bbox_width_px

    # 3. Point in camera coordinates: [x_u * Z, y_u * Z, Z]
    pt_cam = np.array([x_u * Z, y_u * Z, Z], dtype=np.float64)

    # 4. Transform to world coordinates
    R_t = np.asarray(R, dtype=np.float64).T
    t_vec = np.asarray(t, dtype=np.float64)
    pt_world = R_t @ pt_cam + t_vec

    return pt_world


def pixel_to_ground(u, v, K, R, t, k1=0.0, k2=0.0):
    """Image pixel → ground plane intersection (z=0).

    Uses the full camera model to find where the ray from the camera through
    pixel (u,v) intersects the pitch plane. More accurate than a homography
    when the camera has non-zero tilt.

    Args:
        u, v: pixel coordinates
        K: 3×3 intrinsics
        R: 3×3 rotation (world → camera)
        t: (3,) camera center in world

    Returns:
        (2,) ground position [x, y] in meters, or None if ray is parallel
        to ground or intersection is behind camera
    """
    origin, direction = pixel_to_ray(u, v, K, R, t, k1, k2)
    if abs(direction[2]) < 1e-10:
        return None
    t_scale = -origin[2] / direction[2]
    if t_scale < 0:
        return None
    pt = origin + t_scale * direction
    return np.array([pt[0], pt[1]], dtype=np.float64)


def project_3d_to_pixel(points_3d, K, R, t):
    """Project 3D world point(s) to image pixel coordinates.

    Args:
        points_3d: (3,) or (N, 3) world coordinates [x, y, z]
        K: 3×3 camera intrinsic matrix
        R: 3×3 rotation matrix (world → camera)
        t: (3,) camera center in world coordinates

    Returns:
        (2,) or (N, 2) pixel coordinates
    """
    points_3d = np.atleast_2d(np.asarray(points_3d, dtype=np.float64))

    # Build 3×4 projection matrix: P = K [R | -R t]
    It = np.eye(4, dtype=np.float64)[:3]
    It[:, 3] = -np.asarray(t, dtype=np.float64)
    P = K @ (np.asarray(R, dtype=np.float64) @ It)

    ones = np.ones((points_3d.shape[0], 1), dtype=np.float64)
    world_h = np.hstack([points_3d, ones])
    img_h = (P @ world_h.T).T
    img_h /= img_h[:, 2:3]
    pixels = img_h[:, :2]

    if pixels.shape[0] == 1:
        return pixels[0]
    return pixels


def build_projection_matrix(K, R, t):
    """Build the 3×4 camera projection matrix P = K [R | -R t].

    Maps homogeneous world coordinates [X, Y, Z, 1]^T to image pixels.
    """
    It = np.eye(4, dtype=np.float64)[:3]
    It[:, 3] = -np.asarray(t, dtype=np.float64)
    return K @ (np.asarray(R, dtype=np.float64) @ It)
