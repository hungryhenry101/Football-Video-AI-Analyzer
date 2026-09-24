// Python-facing orchestration layer around BroadTrack's CameraTracker.
//
// This file is part of the pybind11 integration; it is NOT part of the
// upstream BroadTrack sources (Camera.cpp, CameraTracker.cpp, ...). It owns
// everything that main.cpp used to do around the tracker: the per-frame state
// machine, the Lucas-Kanade optical flow grid, the "tracking lost" counter and
// the coordinate-space conversion between the caller's frame resolution and
// the fixed HD reference space the C++ pipeline was tuned for.
//
// The neural networks of BroadTrack (line segmentation + pitch keypoints) are
// deliberately NOT called from here. They run in Python (see
// core/broadtrack/models.py) so that this extension does not have to link a
// second copy of libtorch next to the one the Python `torch` package already
// has in the process. Keypoints are pulled back in through a callback that is
// only invoked when the tracker actually needs a re-initialisation.

#pragma once

#include <array>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

class CameraTracker;
class Camera;
class Point3D;

namespace broadtrack
{

// A pitch point detection: (SoccerPitch3D::PointID, [(x, y), ...]).
// Coordinates are in the caller's frame resolution.
using KeypointList = std::vector<std::pair<int, std::vector<std::array<double, 2>>>>;

// Called from inside estimate() when the tracker needs pitch keypoints to
// re-initialise. Returns keypoints in the caller's frame resolution.
using KeypointFn = std::function<KeypointList()>;

struct Config
{
    // Reference resolution the C++ pipeline works in. BroadTrack's constants
    // (PointExtractor's 540-row normalisation, MASK_TO_HD_FACTOR,
    // scoreCameraWithMask's 960x540 resize) are only mutually consistent when
    // the camera is declared at HD. Every coordinate crossing the boundary is
    // mapped into this space, so the caller may hand in any frame size.
    double hd_width = 1920.0;
    double hd_height = 1080.0;

    // Lucas-Kanade ground correspondences (BroadTrack's "optical flow" prior).
    bool optical_flow = true;
    bool radial_distortion = true;

    // "free": camera position is solved per frame (BroadTrack's default when
    // no tripod calibration file is supplied). "soft": position is pulled
    // towards the tripod centre/radius below.
    std::string position_mode = "free";

    double tripod_x = 0.0;
    double tripod_y = 90.0;
    double tripod_z = -18.0;
    double tripod_radius = 0.0;

    // State machine thresholds, matching BroadTrack's main.cpp defaults.
    double reinit_score = 0.3;       // below this a re-initialisation is attempted
    double good_score = 0.5;         // above this the "tracking lost" counter resets
    double prior_reset_score = 0.2;  // combined with lost_frames_before_reset
    int lost_frames_before_reset = 5;
    int reinit_threshold = 12;       // RANSAC-style threshold for reinit()
    int reinit_cauchy_factor = 100;  // Cauchy parameter = factor * frames_lost

    // Optical flow seed grid, in caller-frame pixels (BroadTrack's main.cpp
    // uses a 50 px pitch with a 50 px margin on the processing frame, so the
    // number of seeded points scales with the frame size).
    int grid_step = 50;
    int grid_margin = 50;
    int min_tracked_points = 10;
};

struct Result
{
    bool ok = false;      // false only when no calibration exists at all yet
    double score = 0.0;   // line-IoU score of the returned camera
    bool reinit = false;  // a re-initialisation was attempted on this frame

    // Camera model in the caller's frame resolution:
    //   K (3x3 intrinsics), R (3x3 world->camera), t (3) camera centre (world, m)
    // World frame: metres, origin at the pitch centre, x towards the right goal
    // line, y towards the bottom touch line, z = 0 on the ground and negative
    // above it. This is the same frame PnLCalib used, so
    // P = K [R | -R t] and projection_utils work unchanged.
    std::array<double, 9> K{};
    std::array<double, 9> R{};
    std::array<double, 3> t{};

    double focal_length = 0.0;
    double pan_deg = 0.0, tilt_deg = 0.0, roll_deg = 0.0;
    double k1 = 0.0;
    int frames_lost = 0;
};

class Tracker
{
public:
    explicit Tracker(const Config &config = Config());
    ~Tracker();

    Tracker(const Tracker &) = delete;
    Tracker &operator=(const Tracker &) = delete;

    // One frame of the pipeline.
    //   frame    BGR uint8, HxWx3, caller's resolution (aliased, not copied)
    //   lineMask uint8 label map from the line segmentation model, any
    //            resolution with the same aspect ratio as `frame`
    //   boxes    player bounding boxes (x1, y1, x2, y2) in frame pixels, used
    //            to reject optical-flow points that sit on moving players
    //   keypointsFn invoked at most once, only when a re-initialisation is
    //            attempted
    Result estimate(const cv::Mat &frame,
                    const cv::Mat &lineMask,
                    const std::vector<cv::Rect2d> &boxes,
                    const KeypointFn &keypointsFn);

    // Drop all temporal state (camera, optical-flow points, lost counter).
    void reset();

    Config config;

private:
    void initPriorCamera();
    void rebuildOpticalFlowGrid(int rows, int cols);

    // Frame space -> HD reference space (a magnification when the caller hands
    // in frames smaller than HD, which is the normal case: 960x540 frames give
    // 2.0). Everything coming *in* -- mask points, optical-flow pixels,
    // keypoints -- is multiplied by this.
    double hdScale(const cv::Mat &frame) const { return config.hd_width / static_cast<double>(frame.cols); }

    // HD reference space -> frame space, the reciprocal. Camera intrinsics go
    // *out* through this. Using hdScale() here instead silently scales the
    // focal length and principal point by the square of the frame/HD ratio --
    // invisible on 1920x1080 input, where both factors are 1.
    double frameScale(const cv::Mat &frame) const { return static_cast<double>(frame.cols) / config.hd_width; }

    CameraTracker *_tracker = nullptr;
    Camera *_priorCamera = nullptr;

    bool _initialised = false;
    int _framesLost = 0;

    cv::Mat _previousFrame;
    std::vector<cv::Point2f> _trackedPoints;  // frame space
};

}  // namespace broadtrack
