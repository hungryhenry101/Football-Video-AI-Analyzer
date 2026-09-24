#!/usr/bin/env python
"""Build the BroadTrack C++ extension and drop it into ``core/broadtrack/``.

Usage
-----
    python scripts/build_native.py                 # build with this interpreter
    python scripts/build_native.py --python /path/to/python
    python scripts/build_native.py --clean
    python scripts/build_native.py --debug

The extension is deliberately not built through ``setup.py``: it needs CMake to
find Ceres and OpenCV, which is far more reliable through CMake's own
``find_package`` machinery than through a hand-written ``Extension``.

Requirements (any package manager):
  * a C++17 compiler
  * CMake >= 3.18
  * pybind11 (``pip install pybind11``)
  * OpenCV 4 with the core, imgproc and video modules
  * Ceres Solver >= 2.1 (2.1 is the first release with ``SubsetManifold``)
  * RapidJSON headers
  * Boost headers
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NATIVE_DIR = REPO_ROOT / "core" / "BroadTrack" / "python"
# NOTE: the package cannot be called `broadtrack`: macOS and Windows have
# case-insensitive filesystems, so `core/broadtrack/` and the vendored C++
# source tree `core/BroadTrack/` would be the same directory.
PACKAGE_DIR = REPO_ROOT / "core" / "broadtrack_calib"
DEFAULT_BUILD_DIR = NATIVE_DIR / "build"

#: Dependencies built locally by scripts/build_minimal_*.py, if those have run.
MINIMAL_CERES_DIR = REPO_ROOT / "core" / "BroadTrack" / "third_party" / "ceres"
MINIMAL_OPENCV_DIR = REPO_ROOT / "core" / "BroadTrack" / "third_party" / "opencv"


def run(command: list[str], **kwargs) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"$ {printable}", flush=True)
    subprocess.run(command, check=True, **kwargs)


def pybind11_cmake_dir(python: str) -> str:
    result = subprocess.run(
        [python, "-c", "import pybind11; print(pybind11.get_cmake_dir())"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def minimal_ceres_cmake_dir() -> Path | None:
    """The CMake package directory of a locally built minimal Ceres, if any."""
    candidate = MINIMAL_CERES_DIR / "lib" / "cmake" / "Ceres"
    return candidate if (candidate / "CeresConfig.cmake").is_file() else None


def minimal_opencv_cmake_dir() -> Path | None:
    """The CMake package directory of a locally built minimal OpenCV, if any."""
    candidate = MINIMAL_OPENCV_DIR / "lib" / "cmake" / "opencv4"
    return candidate if (candidate / "OpenCVConfig.cmake").is_file() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", default=sys.executable, help="interpreter to build the module for")
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR), help="CMake build directory")
    parser.add_argument("--clean", action="store_true", help="remove the build directory first")
    parser.add_argument("--debug", action="store_true", help="build with CMAKE_BUILD_TYPE=Debug")
    parser.add_argument("-j", "--jobs", type=int, default=0, help="parallel build jobs (0 = all cores)")
    parser.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        metavar="VAR=VALUE",
        help="extra -D argument forwarded to CMake (repeatable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_dir = Path(args.build_dir).expanduser().resolve()
    python = args.python

    if not Path(python).exists() and shutil.which(python) is None:
        print(f"error: interpreter not found: {python}", file=sys.stderr)
        return 2

    try:
        cmake_dir = pybind11_cmake_dir(python)
    except subprocess.CalledProcessError:
        print(f"error: {python} has no pybind11. Install it with:\n  {python} -m pip install pybind11", file=sys.stderr)
        return 2

    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    configure = [
        "cmake",
        "-S", str(NATIVE_DIR),
        "-B", str(build_dir),
        f"-Dpybind11_DIR={cmake_dir}",
        f"-DBROADTRACK_PYTHON_EXECUTABLE={python}",
        f"-DCMAKE_BUILD_TYPE={'Debug' if args.debug else 'Release'}",
    ]

    local_ceres = minimal_ceres_cmake_dir()
    if local_ceres is not None:
        configure.append(f"-DCeres_DIR={local_ceres}")
    else:
        print(
            "note: no minimal Ceres found; falling back to the system Ceres.\n"
            "      If the import then dies with 'OMP: Error #15' or\n"
            "      \"flag 'flagfile' was defined more than once\", run:\n"
            "          python scripts/build_minimal_ceres.py",
            file=sys.stderr,
        )

    local_opencv = minimal_opencv_cmake_dir()
    if local_opencv is not None:
        configure.append(f"-DOpenCV_DIR={local_opencv}")
    else:
        print(
            "note: no minimal OpenCV found; falling back to the system OpenCV.\n"
            "      Homebrew's and conda-forge's OpenCV both pull a second OpenMP\n"
            "      runtime, which makes libomp abort the interpreter on import\n"
            "      ('OMP: Error #15'). Run this to avoid that:\n"
            "          python scripts/build_minimal_opencv.py",
            file=sys.stderr,
        )

    configure += [f"-D{arg}" for arg in args.cmake_arg]
    run(configure)

    build = ["cmake", "--build", str(build_dir), "--config", "Debug" if args.debug else "Release"]
    if args.jobs:
        build += ["-j", str(args.jobs)]
    else:
        build += ["-j"]
    run(build)

    # MSVC puts the module in a per-configuration subdirectory.
    search_dirs = [build_dir, build_dir / "Release", build_dir / "Debug"]
    modules = [
        module
        for directory in search_dirs
        if directory.is_dir()
        for pattern in ("_broadtrack*.so", "_broadtrack*.pyd", "_broadtrack*.dll")
        for module in sorted(directory.glob(pattern))
    ]
    if not modules:
        print(f"error: no extension module found under {build_dir}", file=sys.stderr)
        return 1

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("_broadtrack*.so", "_broadtrack*.pyd", "_broadtrack*.dll"):
        for stale in PACKAGE_DIR.glob(pattern):
            stale.unlink()

    installed = []
    for module in modules:
        destination = PACKAGE_DIR / module.name
        shutil.copy2(module, destination)
        installed.append(destination)

    for destination in installed:
        print(f"installed {destination.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
