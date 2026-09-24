// pybind11 bindings for the BroadTrack camera tracker.
//
// Everything crossing this boundary is plain numpy / Python data: no cv::Mat
// is ever handed to Python, and no Python object is stored in C++. That keeps
// the extension independent of whichever OpenCV build the host process already
// loaded through `import cv2` (the two OpenCV copies only ever share raw
// buffers, which is safe).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BroadTrackTracker.h"
#include "SoccerPitch3D.h"
#include "core.h"

#include <opencv2/core.hpp>

#include <stdexcept>
#include <string>

namespace py = pybind11;
using broadtrack::Config;
using broadtrack::KeypointFn;
using broadtrack::KeypointList;
using broadtrack::Result;
using broadtrack::Tracker;

namespace
{

template <typename T>
T dictValue(const py::dict &dict, const char *key, const T &fallback)
{
    if (!dict.contains(key))
    {
        return fallback;
    }
    return py::cast<T>(dict[key]);
}

Config configFromDict(const py::dict &dict)
{
    Config config;
    config.hd_width = dictValue<double>(dict, "hd_width", config.hd_width);
    config.hd_height = dictValue<double>(dict, "hd_height", config.hd_height);
    config.optical_flow = dictValue<bool>(dict, "optical_flow", config.optical_flow);
    config.radial_distortion = dictValue<bool>(dict, "radial_distortion", config.radial_distortion);
    config.position_mode = dictValue<std::string>(dict, "position_mode", config.position_mode);
    config.tripod_x = dictValue<double>(dict, "tripod_x", config.tripod_x);
    config.tripod_y = dictValue<double>(dict, "tripod_y", config.tripod_y);
    config.tripod_z = dictValue<double>(dict, "tripod_z", config.tripod_z);
    config.tripod_radius = dictValue<double>(dict, "tripod_radius", config.tripod_radius);
    config.reinit_score = dictValue<double>(dict, "reinit_score", config.reinit_score);
    config.good_score = dictValue<double>(dict, "good_score", config.good_score);
    config.prior_reset_score = dictValue<double>(dict, "prior_reset_score", config.prior_reset_score);
    config.lost_frames_before_reset =
        dictValue<int>(dict, "lost_frames_before_reset", config.lost_frames_before_reset);
    config.reinit_threshold = dictValue<int>(dict, "reinit_threshold", config.reinit_threshold);
    config.reinit_cauchy_factor =
        dictValue<int>(dict, "reinit_cauchy_factor", config.reinit_cauchy_factor);
    config.grid_step = dictValue<int>(dict, "grid_step", config.grid_step);
    config.grid_margin = dictValue<int>(dict, "grid_margin", config.grid_margin);
    config.min_tracked_points = dictValue<int>(dict, "min_tracked_points", config.min_tracked_points);
    return config;
}

// A cv::Mat that borrows a numpy buffer. `owner` must outlive the view.
struct BorrowedMat
{
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> owner;
    cv::Mat view;
};

BorrowedMat frameView(const py::array &array)
{
    BorrowedMat borrowed{py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(array), cv::Mat()};
    if (borrowed.owner.ndim() != 3 || borrowed.owner.shape(2) != 3)
    {
        throw std::invalid_argument("frame must be an HxWx3 uint8 BGR array");
    }
    borrowed.view = cv::Mat(static_cast<int>(borrowed.owner.shape(0)),
                            static_cast<int>(borrowed.owner.shape(1)), CV_8UC3, borrowed.owner.mutable_data());
    return borrowed;
}

BorrowedMat maskView(const py::array &array)
{
    BorrowedMat borrowed{py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(array), cv::Mat()};
    if (borrowed.owner.ndim() != 2)
    {
        throw std::invalid_argument("line_mask must be a 2-D uint8 label map");
    }
    borrowed.view = cv::Mat(static_cast<int>(borrowed.owner.shape(0)),
                            static_cast<int>(borrowed.owner.shape(1)), CV_8UC1, borrowed.owner.mutable_data());
    return borrowed;
}

std::vector<cv::Rect2d> boxesFromArray(const py::array &array)
{
    std::vector<cv::Rect2d> boxes;
    if (array.size() == 0)
    {
        return boxes;
    }
    auto contiguous = py::array_t<double, py::array::c_style | py::array::forcecast>(array);
    if (contiguous.ndim() != 2 || contiguous.shape(1) != 4)
    {
        throw std::invalid_argument("boxes must be an Nx4 float array of (x1, y1, x2, y2)");
    }
    auto view = contiguous.unchecked<2>();
    boxes.reserve(static_cast<size_t>(contiguous.shape(0)));
    for (py::ssize_t i = 0; i < contiguous.shape(0); ++i)
    {
        boxes.emplace_back(view(i, 0), view(i, 1), view(i, 2) - view(i, 0), view(i, 3) - view(i, 1));
    }
    return boxes;
}

py::array_t<double> matrix3(const std::array<double, 9> &values)
{
    py::array_t<double> out(py::array::ShapeContainer{3, 3});
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
}

py::array_t<double> vector3(const std::array<double, 3> &values)
{
    py::array_t<double> out(py::array::ShapeContainer{3});
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
}

Result estimatePy(Tracker &self,
                  const py::array &frame,
                  const py::array &lineMask,
                  const py::array &boxes,
                  const py::object &keypointsFn)
{
    const BorrowedMat frameMat = frameView(frame);
    const BorrowedMat maskMat = maskView(lineMask);
    const std::vector<cv::Rect2d> boxList = boxesFromArray(boxes);

    KeypointFn callback;
    if (!keypointsFn.is_none())
    {
        callback = [keypointsFn]() -> KeypointList {
            py::object produced = keypointsFn();
            if (produced.is_none())
            {
                return {};
            }
            return py::cast<KeypointList>(produced);
        };
    }

    return self.estimate(frameMat.view, maskMat.view, boxList, callback);
}

py::dict resultToDict(const Result &result)
{
    py::dict out;
    out["ok"] = result.ok;
    out["score"] = result.score;
    out["reinit"] = result.reinit;
    out["frames_lost"] = result.frames_lost;
    out["K"] = matrix3(result.K);
    out["R"] = matrix3(result.R);
    out["t"] = vector3(result.t);
    out["focal_length"] = result.focal_length;
    out["pan_deg"] = result.pan_deg;
    out["tilt_deg"] = result.tilt_deg;
    out["roll_deg"] = result.roll_deg;
    out["k1"] = result.k1;
    return out;
}

py::list pitchWireframe(double delta)
{
    SoccerPitch3D pitch;
    py::list out;
    for (const Polyline3D &polyline : pitch.getWireframe(delta))
    {
        py::array_t<double> points({static_cast<py::ssize_t>(polyline.size()), static_cast<py::ssize_t>(3)});
        auto view = points.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(polyline.size()); ++i)
        {
            view(i, 0) = polyline[static_cast<size_t>(i)].x();
            view(i, 1) = polyline[static_cast<size_t>(i)].y();
            view(i, 2) = polyline[static_cast<size_t>(i)].z();
        }
        out.append(std::move(points));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_broadtrack, m)
{
    m.doc() = "BroadTrack soccer camera tracking (Ceres bundle adjustment + LK optical flow).";

    py::class_<Tracker>(m, "Tracker",
                        "Temporal camera tracker. One instance tracks one video stream.")
        .def(py::init([](const py::dict &config) { return new Tracker(configFromDict(config)); }),
             py::arg("config") = py::dict())
        .def(
            "estimate",
            [](Tracker &self, const py::array &frame, const py::array &lineMask,
               const py::array &boxes, const py::object &keypointsFn) {
                return resultToDict(estimatePy(self, frame, lineMask, boxes, keypointsFn));
            },
            py::arg("frame"), py::arg("line_mask"), py::arg("boxes"), py::arg("keypoints_fn"),
            "Track the camera on one frame.\n\n"
            "frame: HxWx3 uint8 BGR, in the caller's resolution.\n"
            "line_mask: HxW uint8 pitch-line label map from the segmentation model.\n"
            "boxes: Nx4 float (x1, y1, x2, y2) player boxes, or an empty array.\n"
            "keypoints_fn: zero-argument callable returning [(point_id, [(x, y), ...]), ...]\n"
            "    in the caller's resolution; invoked only when a re-initialisation is needed.\n\n"
            "The GIL is held for the whole call, including the keypoints_fn callback.")
        .def("reset", &Tracker::reset, "Drop all temporal state (camera, optical flow, lost counter).");

    m.def("pitch_wireframe", &pitchWireframe, py::arg("delta") = 1.0,
          "Pitch markings as a list of (N, 3) float arrays in world metres.");
}
