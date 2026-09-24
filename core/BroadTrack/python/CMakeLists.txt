# Builds BroadTrack's C++ core as a CPython extension module.
#
# Only the numerical/geometric sources are compiled here. The two neural
# networks (LineSegmentationModel.cpp, KeypointDetectionModel.cpp) and the
# command-line entry points (main.cpp, singleFrameCalibration.cpp,
# JsonCalibDict.cpp) are deliberately excluded:
#
#   * the networks need libtorch, and the host process already has the Python
#     `torch` package's own libtorch loaded. Linking a second, version-mismatched
#     libtorch into the same process breaks the c10 operator registry. They run
#     in Python instead (core/broadtrack/models.py);
#   * the CLI entry points need Boost.ProgramOptions and RapidJSON's writers,
#     neither of which the library path uses.
#
# The result links OpenCV (core/imgproc/video only), Ceres and RapidJSON, which
# are available from every mainstream package manager.

cmake_minimum_required(VERSION 3.23)  # CMAKE_IGNORE_PREFIX_PATH
project(broadtrack_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Homebrew keeps OpenCV 4 keg-only; prefer it over the (ABI-incompatible)
# OpenCV 5 that a bare `brew install opencv` puts on the default path.
if(APPLE AND EXISTS "/opt/homebrew/opt/opencv@4")
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/opencv@4")
endif()

# Pin the interpreter (and therefore the ABI tag of the produced .so). The
# build driver passes the interpreter it wants the module installed into.
if(DEFINED BROADTRACK_PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE "${BROADTRACK_PYTHON_EXECUTABLE}")
    set(PYTHON_EXECUTABLE "${BROADTRACK_PYTHON_EXECUTABLE}")
endif()
set(PYBIND11_FINDPYTHON ON)
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

find_package(pybind11 CONFIG REQUIRED)

# A Conda/Anaconda environment ships its own copies of the C++ stack next to the
# interpreter. Left in the search path, CMake happily mixes them with the system
# ones: CeresConfig.cmake names its gflags dependency unqualified, so CMake
# resolves the *bare* name `gflags` to whichever build it finds first, and
# loading that one next to the gflags glog already depends on puts two gflags
# registries in the process. The module then aborts on import with
#
#     ERROR: flag 'flagfile' was defined more than once
#
# Keep the interpreter's own prefixes out of the C++ dependency search. Python
# and pybind11 are located above, by path, so they are unaffected. If Ceres and
# OpenCV really do come from the interpreter's prefix, turn this off with
# -DBROADTRACK_ISOLATE_PYTHON_PREFIX=OFF.
option(BROADTRACK_ISOLATE_PYTHON_PREFIX
       "Exclude the Python interpreter's prefix from C++ dependency lookup" ON)
if(BROADTRACK_ISOLATE_PYTHON_PREFIX)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix); print(sys.base_prefix)"
        OUTPUT_VARIABLE BROADTRACK_PYTHON_PREFIXES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    string(REPLACE "\n" ";" BROADTRACK_PYTHON_PREFIXES "${BROADTRACK_PYTHON_PREFIXES}")
    foreach(_prefix IN LISTS BROADTRACK_PYTHON_PREFIXES)
        if(EXISTS "${_prefix}")
            list(APPEND CMAKE_IGNORE_PREFIX_PATH "${_prefix}")
        endif()
    endforeach()
endif()

# CeresConfig.cmake lists its gflags dependency as the *bare* name `gflags`, so
# CMake resolves it by searching for any package called gflags. When a second
# C++ stack is installed next to the interpreter (Conda/Anaconda ships one), that
# search can land on an unrelated gflags build, and loading it alongside the one
# glog already links puts two gflags registries in the process:
#
#     ERROR: flag 'flagfile' was defined more than once
#
# Nothing here calls a gflags symbol — glog's own load command supplies the
# right library — so satisfying the name with an empty target keeps the stray
# copy off the link line. Declared before find_package(Ceres) so it wins over
# whatever that pulls in.
if(NOT TARGET gflags)
    add_library(gflags INTERFACE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc video)
find_package(Ceres REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(Boost REQUIRED)

set(BROADTRACK_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

set(BROADTRACK_SOURCES
    "${BROADTRACK_CORE_DIR}/core.cpp"
    "${BROADTRACK_CORE_DIR}/Camera.cpp"
    "${BROADTRACK_CORE_DIR}/CameraTracker.cpp"
    "${BROADTRACK_CORE_DIR}/SoccerPitch3D.cpp"
    "${BROADTRACK_CORE_DIR}/PointExtractor.cpp"
    "${BROADTRACK_CORE_DIR}/LineIoUScore.cpp"
)

pybind11_add_module(_broadtrack MODULE
    bindings.cpp
    BroadTrackTracker.cpp
    ${BROADTRACK_SOURCES}
)

target_include_directories(_broadtrack PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${BROADTRACK_CORE_DIR}"
)

# Link the three OpenCV modules the library actually uses, not ${OpenCV_LIBS}.
# The full list drags in opencv_dnn, which in turn needs whatever protobuf it
# was built against — a dependency the tracker has no use for, and one that
# breaks the import outright when the local protobuf has moved on.
set(BROADTRACK_OPENCV_TARGETS "")

foreach(component core imgproc video)
    if(TARGET opencv_${component})
        list(APPEND BROADTRACK_OPENCV_TARGETS opencv_${component})
    else()
        # Module targets are missing from some OpenCV packages; fall back to
        # the library paths find_package resolved for the requested components.
        list(APPEND BROADTRACK_OPENCV_TARGETS ${OpenCV_${component}_LIBRARY})
    endif()
endforeach()

target_link_libraries(_broadtrack PRIVATE
    ${BROADTRACK_OPENCV_TARGETS}
    Ceres::ceres
)

# RapidJSON ships either an imported target or plain include directories
# depending on how it was packaged.
if(TARGET RapidJSON::RapidJSON)
    target_link_libraries(_broadtrack PRIVATE RapidJSON::RapidJSON)
elseif(DEFINED RapidJSON_INCLUDE_DIRS)
    target_include_directories(_broadtrack PRIVATE ${RapidJSON_INCLUDE_DIRS})
elseif(DEFINED RAPIDJSON_INCLUDE_DIRS)
    target_include_directories(_broadtrack PRIVATE ${RAPIDJSON_INCLUDE_DIRS})
endif()

if(TARGET Boost::boost)
    target_link_libraries(_broadtrack PRIVATE Boost::boost)
else()
    target_include_directories(_broadtrack PRIVATE ${Boost_INCLUDE_DIRS})
endif()

# The vendored sources target OpenCV 4; vendor_compat.hpp bridges the handful of
# declarations OpenCV 5 relocated. See the header for the details.
if(MSVC)
    target_compile_options(_broadtrack PRIVATE /bigobj /FI"${CMAKE_CURRENT_SOURCE_DIR}/vendor_compat.hpp")
else()
    target_compile_options(_broadtrack PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/vendor_compat.hpp")
endif()
