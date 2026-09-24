#include "BroadTrackTracker.h"

#include "Camera.h"
#include "CameraTracker.h"
#include "SoccerPitch3D.h"
#include "core.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace broadtrack
{

namespace
{

constexpr double kDegreesPerRadian = 180.0 / CV_PI;

std::array<double, 9> toArray(const Matrix3x3 &m)
{
    std::array<double, 9> out{};
    for (int i = 0; i < 9; ++i)
    {
        out[i] = m[i];
    }
    return out;
}

std::array<double, 3> toArray(const Vector3x1 &v)
{
    std::array<double, 3> out{};
    for (int i = 0; i < 3; ++i)
    {
        out[i] = v[i];
    }
    return out;
}

}  // namespace

Tracker::Tracker(const Config &cfg) : config(cfg)
{
    if (config.hd_width <= 0.0 || config.hd_height <= 0.0)
    {
        throw std::invalid_argument("BroadTrack: hd_width/hd_height must be positive");
    }
    if (config.position_mode != "free" && config.position_mode != "soft")
    {
        throw std::invalid_argument("BroadTrack: position_mode must be \"free\" or \"soft\"");
    }

    _tracker = new CameraTracker();
    _priorCamera = new Camera(cv::Size(static_cast<int>(config.hd_width), static_cast<int>(config.hd_height)));
}

Tracker::~Tracker()
{
    delete _tracker;
    delete _priorCamera;
}

void Tracker::reset()
{
    delete _tracker;
    _tracker = new CameraTracker();
    _initialised = false;
    _framesLost = 0;
    _previousFrame.release();
    _trackedPoints.clear();
    if (config.position_mode == "soft")
    {
        _tracker->setTripodInfo(Point3D(config.tripod_x, config.tripod_y, config.tripod_z),
                                config.tripod_radius);
    }
    else
    {
        _tracker->setTripodInfo(Point3D(config.tripod_x, config.tripod_y, config.tripod_z), 0.0);
    }
}

void Tracker::initPriorCamera()
{
    const cv::Size resolution(static_cast<int>(config.hd_width), static_cast<int>(config.hd_height));

    _priorCamera->setPanTiltRoll(Vector3x1(0.0, 80.0 * CV_PI / 180.0, 0.0));
    _priorCamera->setPosition(Vector3x1(config.tripod_x, config.tripod_y, config.tripod_z));
    _priorCamera->setPixelResolution(resolution);
    _priorCamera->setFocalLength(std::sqrt(config.hd_width * config.hd_width +
                                           config.hd_height * config.hd_height));

    _tracker->setTripodInfo(Point3D(config.tripod_x, config.tripod_y, config.tripod_z),
                            config.position_mode == "soft" ? config.tripod_radius : 0.0);
    _tracker->setCamera(*_priorCamera);
}

void Tracker::rebuildOpticalFlowGrid(int rows, int cols)
{
    _trackedPoints.clear();
    for (int row = config.grid_margin; row < rows - config.grid_margin; row += config.grid_step)
    {
        for (int col = config.grid_margin; col < cols - config.grid_margin; col += config.grid_step)
        {
            _trackedPoints.emplace_back(cv::Point2f(static_cast<float>(col), static_cast<float>(row)));
        }
    }
}

Result Tracker::estimate(const cv::Mat &frame,
                         const cv::Mat &lineMask,
                         const std::vector<cv::Rect2d> &boxes,
                         const KeypointFn &keypointsFn)
{
    if (frame.empty() || frame.type() != CV_8UC3)
    {
        throw std::invalid_argument("BroadTrack: frame must be a non-empty 8UC3 BGR image");
    }
    if (lineMask.empty() || lineMask.type() != CV_8UC1)
    {
        throw std::invalid_argument("BroadTrack: lineMask must be a non-empty 8UC1 label map");
    }

    const int rows = frame.rows;
    const int cols = frame.cols;
    const double scale = hdScale(frame);

    bool didReinit = false;

    // ---------------------------------------------------------------- prior
    if (!_initialised)
    {
        initPriorCamera();
        if (keypointsFn)
        {
            const KeypointList detected = keypointsFn();
            if (detected.size() >= 2)
            {
                std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> pointDict;
                pointDict.reserve(detected.size());
                for (const auto &entry : detected)
                {
                    std::vector<Point2D> points;
                    points.reserve(entry.second.size());
                    for (const auto &xy : entry.second)
                    {
                        points.emplace_back(xy[0] * scale, xy[1] * scale);
                    }
                    if (!points.empty())
                    {
                        pointDict.emplace_back(static_cast<SoccerPitch3D::PointID>(entry.first), points);
                    }
                }
                if (pointDict.size() >= 2)
                {
                    _tracker->reinit(pointDict, config.reinit_threshold);
                    // The first frame (and the first after reset()) is
                    // initialised from keypoints, so report it the same way as a
                    // recovery later in the sequence.
                    didReinit = true;
                }
            }
        }
        _initialised = true;
    }

    // --------------------------------------------------------- optical flow
    SoccerPitch3D pitch;
    const Point2D topLeft = pitch.getPoint2D(SoccerPitch3D::TL_PITCH_CORNER);
    const Point2D bottomRight = pitch.getPoint2D(SoccerPitch3D::BR_PITCH_CORNER);

    std::vector<std::pair<Point3D, Point2D>> pitchProjections;

    if (config.optical_flow && _framesLost == 0)
    {
        if (_trackedPoints.empty())
        {
            rebuildOpticalFlowGrid(rows, cols);
        }
        else if (!_previousFrame.empty() && _previousFrame.size() == frame.size())
        {
            cv::Mat previousGray;
            cv::Mat currentGray;
            cv::cvtColor(_previousFrame, previousGray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(frame, currentGray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point2f> next;
            std::vector<uchar> status;
            std::vector<float> error;
            const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.03);
            cv::calcOpticalFlowPyrLK(previousGray, currentGray, _trackedPoints, next, status, error,
                                     cv::Size(55, 55), 2, criteria);

            const Camera previousCamera = _tracker->getCamera();
            std::vector<cv::Point2f> goodNew;
            goodNew.reserve(_trackedPoints.size());

            for (size_t i = 0; i < _trackedPoints.size() && i < next.size() && i < status.size(); ++i)
            {
                for (const auto &box : boxes)
                {
                    if (box.contains(cv::Point2d(next[i].x, next[i].y)))
                    {
                        status[i] = 0;
                        break;
                    }
                }
                if (status[i] != 1)
                {
                    continue;
                }

                // Unproject the *previous* pixel through the *previous* camera,
                // then associate it with the tracked pixel in the current frame.
                const Point2D previousHd(_trackedPoints[i].x * scale, _trackedPoints[i].y * scale);
                const Point3D point3D = pitch.getSurface().intersection(previousCamera.getRay(previousHd));
                if (point3D.x() > topLeft.x() && point3D.x() < bottomRight.x() &&
                    point3D.y() > topLeft.y() && point3D.y() < bottomRight.y())
                {
                    pitchProjections.emplace_back(point3D, Point2D(next[i].x * scale, next[i].y * scale));
                    goodNew.push_back(next[i]);
                }
            }

            _trackedPoints.swap(goodNew);
            if (static_cast<int>(_trackedPoints.size()) < config.min_tracked_points)
            {
                rebuildOpticalFlowGrid(rows, cols);
            }
        }
    }
    else if (!config.optical_flow)
    {
        // nothing to do
    }
    else
    {
        // Tracking was lost on a previous frame: drop the stale tracks so the
        // grid is re-seeded from scratch once the tracker recovers. main.cpp
        // keeps them, which feeds one frame of stale correspondences into the
        // solver after a recovery.
        _trackedPoints.clear();
    }

    // ---------------------------------------------------------------- update
    const bool softPosition = config.position_mode == "soft";

    double score = 0.0;
    Camera camera;
    std::tie(score, camera) = _tracker->update(lineMask, pitchProjections, softPosition, false,
                                               config.radial_distortion);

    if (score < config.reinit_score)
    {
        _framesLost += 1;
        didReinit = true;

        const Camera previousCamera = camera;
        const KeypointList detected = keypointsFn ? keypointsFn() : KeypointList();

        if (detected.size() >= 2)
        {
            std::vector<std::pair<SoccerPitch3D::PointID, std::vector<Point2D>>> pointDict;
            pointDict.reserve(detected.size());
            for (const auto &entry : detected)
            {
                std::vector<Point2D> points;
                points.reserve(entry.second.size());
                for (const auto &xy : entry.second)
                {
                    points.emplace_back(xy[0] * scale, xy[1] * scale);
                }
                if (!points.empty())
                {
                    pointDict.emplace_back(static_cast<SoccerPitch3D::PointID>(entry.first), points);
                }
            }

            if (pointDict.size() >= 2)
            {
                _tracker->reinit(pointDict, config.reinit_threshold);
                const double newScore = _tracker->evaluate(lineMask);
                if (newScore < score)
                {
                    _tracker->setCamera(previousCamera);
                }
                else
                {
                    score = newScore;
                }
            }
        }

        if (_framesLost > config.lost_frames_before_reset && score < config.prior_reset_score)
        {
            _tracker->setCamera(*_priorCamera);
        }

        const std::vector<std::pair<Point3D, Point2D>> empty;
        const int cauchy = config.reinit_cauchy_factor * _framesLost;
        std::tie(score, camera) = _tracker->update(lineMask, empty, true, false,
                                                   config.radial_distortion, cauchy);
        _trackedPoints.clear();
    }

    if (score > config.good_score)
    {
        _framesLost = 0;
    }

    frame.copyTo(_previousFrame);

    // ---------------------------------------------------------------- result
    const Camera solved = _tracker->getCamera();

    Result result;
    result.ok = true;
    result.score = score;
    result.reinit = didReinit;
    result.frames_lost = _framesLost;

    // Camera intrinsics are declared in HD; rescale focal length and principal
    // point to the caller's frame. The homogeneous bottom-right entry is 1 and
    // must stay 1, so the matrix is rebuilt rather than multiplied.
    const double toFrame = frameScale(frame);
    const double hdFocal = solved.getFocalLength();
    const Point2D hdPrincipal = solved.getPrincipalPoint();
    const double frameFocal = hdFocal * toFrame;
    result.K = {frameFocal, 0.0, hdPrincipal.x() * toFrame,
                0.0, frameFocal, hdPrincipal.y() * toFrame,
                0.0, 0.0, 1.0};

    result.R = toArray(solved.getRotation());
    result.t = toArray(solved.getPosition());

    const Vector3x1 panTiltRoll = solved.getPanTiltRoll();
    result.pan_deg = panTiltRoll[0] * kDegreesPerRadian;
    result.tilt_deg = panTiltRoll[1] * kDegreesPerRadian;
    result.roll_deg = panTiltRoll[2] * kDegreesPerRadian;
    result.focal_length = frameFocal;

    const std::vector<double> distortion = solved.getRadialDistortion();
    result.k1 = distortion.empty() ? 0.0 : distortion[0];

    return result;
}

}  // namespace broadtrack
