// Force-included ahead of every translation unit (see CMakeLists.txt). It
// bridges the vendored BroadTrack sources to the local build environment
// without editing them, in two ways.
//
// 1. OpenCV 5 relocated declarations.
//    The sources were written against OpenCV 4. OpenCV 5 kept the API but moved
//    the 2-D shape fitting declarations (cv::fitLine, cv::fitEllipse* and the
//    cv::DistanceTypes enum they take) out of <opencv2/imgproc.hpp> into a new
//    <opencv2/geometry/2d.hpp>, and <opencv2/opencv.hpp> does not pull it in.
//    Two symbols in core.cpp (fitLine with DIST_L2, fitEllipseDirect) therefore
//    fail to compile against OpenCV 5 over a missing include and nothing else.
//
// 2. Standard headers the sources only got transitively.
//    PointExtractor.cpp, CameraTracker.cpp and core.cpp use std::cout and
//    std::ostringstream without including <iostream> / <sstream>. That compiles
//    as long as something in the include graph happens to pull them in, which a
//    full packaged OpenCV does and a minimal BUILD_LIST=core,imgproc,video
//    build (scripts/build_minimal_opencv.py) does not. Include them explicitly
//    rather than growing the diff against upstream.

#pragma once

#include <iostream>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>

#if defined(CV_VERSION_MAJOR) && CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif
