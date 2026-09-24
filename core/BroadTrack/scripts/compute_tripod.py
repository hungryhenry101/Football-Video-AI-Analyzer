import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from sklearn.covariance import MinCovDet
import argparse

# from core.camera import Camera

def pan_tilt_roll_to_orientation(pan, tilt, roll):
    """
    Conversion from euler angles to orientation matrix.
    :param pan:
    :param tilt:
    :param roll:
    :return: orientation matrix
    """
    Rpan = np.array([
        [np.cos(pan), -np.sin(pan), 0],
        [np.sin(pan), np.cos(pan), 0],
        [0, 0, 1]])
    Rroll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]])
    Rtilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt), np.cos(tilt)]])
    rotMat = np.dot(Rpan, np.dot(Rtilt, Rroll))
    return rotMat

class Camera:

    def from_json_parameters(self, calib_json_object):
        """
        Loads camera parameters from dictionary.
        :param calib_json_object: the dictionary containing camera parameters.
        """
        self.image_width = calib_json_object["sensorResolutionWidthPixels"]
        self.image_height = calib_json_object["sensorResolutionHeightPixels"]
        self.principal_point = (self.image_width/2., self.image_height/2. )
        hfov = calib_json_object["horizontalFieldOfViewDegrees"]*np.pi/180.
        focal = 2 * self.principal_point[0] / (2 * np.tan(hfov / 2.))
        self.xfocal_length = focal
        self.yfocal_length = focal


        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')

        pan = calib_json_object['panDegrees'] * np.pi / 180.
        tilt = calib_json_object['tiltDegrees'] * np.pi / 180.
        roll = calib_json_object['rollDegrees'] * np.pi / 180.

        self.rotation = np.array([
            [-np.sin(pan) * np.sin(roll) * np.cos(tilt) + np.cos(pan) * np.cos(roll),
             np.sin(pan) * np.cos(roll) + np.sin(roll) * np.cos(pan) * np.cos(tilt), np.sin(roll) * np.sin(tilt)],
            [-np.sin(pan) * np.cos(roll) * np.cos(tilt) - np.sin(roll) * np.cos(pan),
             -np.sin(pan) * np.sin(roll) + np.cos(pan) * np.cos(roll) * np.cos(tilt), np.sin(tilt) * np.cos(roll)],
            [np.sin(pan) * np.sin(tilt), -np.sin(tilt) * np.cos(pan), np.cos(tilt)]
        ], dtype='float')

        self.rotation = np.transpose(pan_tilt_roll_to_orientation(pan, tilt, roll))

        self.position = np.array([
            calib_json_object['positionXMeters'],
            calib_json_object['positionYMeters'],
            calib_json_object['positionZMeters']
                                  ], dtype='float')



    def to_json_parameters(self):
        pass

    def get_look_at_vector(self):
        return self.rotation[2]



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def compute_intersection_points(rays, tilt_full):
    points = []
    assert len(rays) == len(tilt_full)
    for i in range(len(rays)):
        for j in range(i+1, len(rays)):
            if np.abs(tilt_full[i] - tilt_full[j]) > 1.:
                break
            origin1, direction1 = rays[i]
            origin2, direction2 = rays[j]

            cos_theta = np.dot(direction1, direction2)/ np.linalg.norm(direction1) / np.linalg.norm(direction2)
            # May want to avoid too similar optical axes
            if cos_theta > 0.9:
                continue

            A = np.array([direction1, -direction2]).T
            b = origin2 - origin1
            t, s = np.linalg.lstsq(A, b, rcond=None)[0]
            point1 = origin1 + t * direction1
            point2 = origin2 + s * direction2

            intersection_point = (point1 + point2) / 2
            points.append(intersection_point)
    return np.array(points)


def get_eye_and_optical_axis_and_tilt_pan(file, iou_score=0.6):

    eyes = []
    axis = []
    tilts = []
    pans = []
    with open(file, "r") as f:
        calibs_results_per_frame = json.load(f)

    for k, v in calibs_results_per_frame.items():

        frame_score = calibs_results_per_frame[k]["score"]
        if frame_score > iou_score:
            cam = Camera()
            cam.from_json_parameters(v["cp"])
            eyes.append(cam.position)
            axis.append(cam.get_look_at_vector())
            tilts.append(v["cp"]["tiltDegrees"])
            pans.append(v["cp"]["panDegrees"])

    return eyes, axis, tilts, pans

def residuals(params, points, directions):
    c = params[:3]
    r = params[3]
    res = []

    for i in range(len(points)):
        p = points[i]
        d = directions[i]

        lambda_i = np.dot(c - p, d) / np.dot(d, d)
        t = p + lambda_i * d

        rad_constraint = np.linalg.norm(t - c) - r
        small_radius_constraint = r**2
        main_camera_X0_constraint = 10 * c[0]**2
        res.append(rad_constraint)
        res.append(main_camera_X0_constraint)
        res.append(small_radius_constraint)

    return res



def analyse_sequence( file, output_file):

    output_dict = {}

    match_eyes, match_axis, match_tilts, match_pans = get_eye_and_optical_axis_and_tilt_pan(file)
    intersect_optical_axes_match = compute_intersection_points(list(zip(match_eyes, match_axis)), match_tilts)

    assert len(intersect_optical_axes_match)

    robust_estimate = MinCovDet().fit(intersect_optical_axes_match)

    mean_robust = robust_estimate.location_
    points = np.array(match_eyes)
    directions = np.array(match_axis)

    initial_guess = np.array([0., mean_robust[1], mean_robust[2], 0.12])
    bounds = ([-np.inf, -np.inf, -np.inf, 0.], [np.inf, np.inf, np.inf, 0.3])
    result = least_squares(residuals, initial_guess, bounds=bounds, args=(points, directions), loss='soft_l1')

    center = result.x[:3]
    radius = result.x[3]

    print("Initial optical axis intersection:", mean_robust)
    print(f"center: {center} radius: {radius}")
    output_dict = {}
    output_dict["sphere"] = {}
    output_dict["sphere"]["center"] = center
    output_dict["sphere"]["radius"] = radius

    
    print(f"From tilt : {np.min(match_tilts)} -> {np.max(match_tilts)}")
    print(f"From pan : {np.min(match_pans)} -> {np.max(match_pans)}")


    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.axis('equal')

    Xs = [x[0] for x in match_eyes]
    Ys = [x[1] for x in match_eyes]
    Zs = [x[2] for x in match_eyes]
    dXs = [x[0] for x in match_axis]
    dYs = [x[1] for x in match_axis]
    dZs = [x[2] for x in match_axis]

    ax1.quiver(
        Xs,
        Ys,

        dXs,
        dYs,
        color='blue',
        angles='xy'

    )
    circ0 = plt.Circle((center[0], center[1]), radius, color='r')
    ax1.add_patch(circ0)
    ax1.scatter(mean_robust[0], mean_robust[1], marker=(5, 1))

    ax2.quiver(
        Ys,
        Zs,
        dYs,
        dZs,
        color='blue',
        angles='xy'
    )
    circ = plt.Circle((center[1], center[2]), radius, color='r')
    ax2.add_patch(circ)
    ax2.scatter(mean_robust[1], mean_robust[2], marker=(5, 1))

    ax1.invert_yaxis()
    ax2.invert_xaxis()
    ax2.invert_yaxis()
   
    plt.savefig(f'{output_file.replace(".json", ".png")}')

    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=4, cls=NumpyEncoder)

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Compute the tripod parameters from broadcast camera parameters. Generates json file with tripod parameters and visualizations plots(next to the output path provided).')

    parser.add_argument('-i', '--input', type=str,
                        help='Path to file containing camera parameters')
    parser.add_argument('-o', '--output', default="./tripod.json",
                        required=False, type=str,
                        help="Path to the output file.")
    args = parser.parse_args()

    analyse_sequence(args.input, args.output)
