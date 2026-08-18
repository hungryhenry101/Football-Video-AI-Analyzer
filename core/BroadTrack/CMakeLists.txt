cmake_minimum_required(VERSION 3.6)
project(evs-main-camera-tracking)

# libtorch installation directory (macOS build supports CPU + MPS).
# Override with -DLIBTORCH_DIR=<path>; default is the repo-local copy.
if(NOT DEFINED LIBTORCH_DIR)
    set(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
endif()

if(APPLE)
    # Homebrew on Apple Silicon provides Boost/OpenCV/Ceres/RapidJSON.
    # OpenCV 4 is keg-only; OpenCV 5 is API-incompatible with this codebase.
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/opt/homebrew/opt/opencv@4")
    include_directories("/opt/homebrew/include")
    link_directories("/opt/homebrew/lib")
endif()

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(Torch REQUIRED PATHS "${LIBTORCH_DIR}")

add_executable(broadtrack main.cpp
    KeypointDetectionModel.cpp
    KeypointDetectionModel.h
    LineSegmentationModel.cpp
    LineSegmentationModel.h
    Camera.cpp
    Camera.h
    core.h
    core.cpp
    LineIoUScore.cpp
    LineIoUScore.h
    SoccerPitch3D.cpp
    SoccerPitch3D.h
    CameraTracker.cpp
    CameraTracker.h
    Residuals.h
    Chronometer.h
    JsonCalibDict.h
    JsonCalibDict.cpp
    PointExtractor.h
    PointExtractor.cpp
)



add_executable(reinit singleFrameCalibration.cpp
    Camera.cpp
    Camera.h
    core.h
    core.cpp
    KeypointDetectionModel.cpp
    KeypointDetectionModel.h
    LineSegmentationModel.cpp
    LineSegmentationModel.h
    LineIoUScore.cpp
    LineIoUScore.h
    SoccerPitch3D.cpp
    SoccerPitch3D.h
    CameraTracker.cpp
    CameraTracker.h
    Residuals.h
    Chronometer.h
    JsonCalibDict.h
    JsonCalibDict.cpp
    PointExtractor.h
    PointExtractor.cpp
)


target_link_libraries(broadtrack
PUBLIC
boost_filesystem
boost_program_options
ceres
${OpenCV_LIBS}
${TORCH_LIBRARIES}

)

target_link_libraries(reinit 
PUBLIC 
boost_filesystem
boost_system
PUBLIC ceres
${OpenCV_LIBS}
${TORCH_LIBRARIES}
)

install(TARGETS broadtrack DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
