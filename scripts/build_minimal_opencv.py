#!/usr/bin/env python
"""Build a minimal, self-contained OpenCV for the BroadTrack extension.

Usage
-----
    python scripts/build_minimal_opencv.py
    python scripts/build_minimal_opencv.py --force
    python scripts/build_minimal_opencv.py --source-dir /path/to/opencv-4.14.0

Why this exists
---------------
Same reason as scripts/build_minimal_ceres.py: one process, one OpenMP runtime.

Every packaged OpenCV on this machine drags a second one in.

* Homebrew's ``opencv`` and ``opencv@4`` both link ``libopenblas``, which links
  Homebrew's libomp (``/opt/homebrew/opt/libomp/lib/libomp.dylib``).
* conda-forge's ``libopencv`` lists ``llvm-openmp`` as a direct dependency --
  the build has OpenMP switched on -- so forcing a pthreads BLAS does not help
  either.

PyTorch's wheel ships its own OpenMP and that one cannot be removed, so the
second copy has to go. PyTorch 2.14 stamps its copy
``/opt/llvm-openmp/lib/libomp.dylib``; up to 2.9 it stamped it
``/opt/homebrew/opt/libomp/lib/libomp.dylib``, which happened to match Homebrew's
and let dyld collapse the two into one image. With 2.14 the names no longer
collide and libomp aborts the interpreter:

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
    initialized.

This build produces OpenCV 4 with ``WITH_OPENMP=OFF`` and ``WITH_LAPACK=OFF``,
built **static**, and limited to the three modules the extension uses
(``core``, ``imgproc``, ``video``). Linked into the module, it leaves the
extension with no OpenCV, BLAS or OpenMP dynamic dependency at all.

The trade-off is real: this is a source build of OpenCV and takes appreciably
longer than the Ceres one. Run it once; the result is cached under
``core/BroadTrack/third_party/opencv/`` and picked up by scripts/build_native.py.
If your platform's OpenCV is known to be free of OpenMP and BLAS -- or if you
simply do not care about the second runtime -- skip this and let the build use
the system OpenCV.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PREFIX = REPO_ROOT / "core" / "BroadTrack" / "third_party" / "opencv"

#: Matches the version the extension was validated against (Homebrew opencv@4).
#: OpenCV 5 works too -- see core/BroadTrack/python/vendor_compat.hpp -- but 4 is
#: what the vendored BroadTrack sources were written for.
OPENCV_VERSION = "4.14.0"
OPENCV_URL = f"https://github.com/opencv/opencv/archive/refs/tags/{OPENCV_VERSION}.tar.gz"

#: Only what the extension calls into. Notably no imgcodecs: the C++ side never
#: reads or writes an image, that happens in Python through cv2.
BUILD_MODULES = "core,imgproc,video"

#: Libraries that must not appear in the finished build. Checked after install.
FORBIDDEN_CAPSULES = ("libomp", "libopenblas", "libcblas", "liblapack", "libglog", "libgflags")


def run(command: list[str], **kwargs) -> None:
    print(f"$ {' '.join(str(part) for part in command)}", flush=True)
    subprocess.run(command, check=True, **kwargs)


def already_built() -> bool:
    return (INSTALL_PREFIX / "lib" / "cmake" / "opencv4" / "OpenCVConfig.cmake").is_file()


def download(destination: Path) -> Path:
    print(f"downloading OpenCV {OPENCV_VERSION} (~90 MB) ...", flush=True)
    with urllib.request.urlopen(OPENCV_URL, timeout=600) as response:  # noqa: S310 (fixed https URL)
        destination.write_bytes(response.read())
    return destination


def extract(archive: Path, into: Path) -> Path:
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile() or m.isdir()]
        try:
            tar.extractall(into, members=members, filter="data")
        except TypeError:  # `filter` is Python >= 3.12
            tar.extractall(into, members=members)
    roots = [p for p in into.iterdir() if p.is_dir()]
    if len(roots) != 1 or not (roots[0] / "CMakeLists.txt").is_file():
        raise RuntimeError(f"unexpected archive layout under {into}")
    return roots[0]


def cmake_arguments(source_dir: Path, build_dir: Path, stage: Path) -> list[str]:
    # Everything optional is off. Besides shrinking the build, this stops CMake
    # from downloading the 3rdparty bundles (ippicv, ade, protobuf, ...) that
    # OpenCV fetches at configure time when the matching feature is enabled.
    disabled_features = [
        "OPENMP", "LAPACK", "EIGEN", "IPP", "TBB", "OPENCL", "OPENCLAMDBLAS", "OPENCLAMDFFT",
        "CUDA", "CUFFT", "CUBLAS", "QUIRC", "FFMPEG", "GSTREAMER", "V4L", "1394", "JASPER",
        "OPENEXR", "OPENJPEG", "PNG", "JPEG", "TIFF", "WEBP", "PROTOBUF", "GTK", "QT", "VTK",
        "ITT", "ADE", "GDCM", "IMGCODEC_HDR", "IMGCODEC_SUNRASTER", "IMGCODEC_PXM",
        "IMGCODEC_PFM", "ANDROID_MEDIANDK", "XINE", "AVFOUNDATION", "GPHOTO2", "DC1394",
    ]
    return [
        "cmake",
        "-S", str(source_dir),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={stage}",
        # Static: linked into the extension module, so no OpenCV dylib is left
        # as a runtime dependency -- and therefore no BLAS and no OpenMP either.
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        f"-DBUILD_LIST={BUILD_MODULES}",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_DOCS=OFF",
        "-DBUILD_opencv_apps=OFF",
        "-DBUILD_JAVA=OFF",
        "-DBUILD_opencv_python2=OFF",
        "-DBUILD_opencv_python3=OFF",
        "-DBUILD_opencv_world=OFF",
        "-DOPENCV_GENERATE_PKGCONFIG=OFF",
        "-DOPENCV_ENABLE_NONFREE=OFF",
        "-DCV_TRACE=OFF",
        "-DWITH_DEBUG_INFO=OFF",
    ] + [f"-DWITH_{feature}=OFF" for feature in disabled_features]


def verify(prefix: Path) -> None:
    """Fail if anything we were trying to avoid ended up in the install tree."""
    offenders = []
    for pattern in ("*.dylib", "*.so", "*.a"):
        for path in prefix.rglob(pattern):
            name = path.name.lower()
            if any(capsule in name for capsule in FORBIDDEN_CAPSULES):
                offenders.append(path.name)
    if offenders:
        raise RuntimeError(f"unexpected libraries in the OpenCV install: {sorted(set(offenders))}")

    installed = sorted(p.name for p in (prefix / "lib").glob("*.a"))
    if not installed:
        raise RuntimeError("no static OpenCV archives were installed")
    print(f"static archives: {', '.join(installed)}")


def swap_into_place(staged: Path) -> None:
    """Replace INSTALL_PREFIX with a freshly staged tree.

    Staged rather than installed in place so a failed build leaves the previous
    install untouched.
    """
    if INSTALL_PREFIX.exists():
        shutil.rmtree(INSTALL_PREFIX)
    INSTALL_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(INSTALL_PREFIX))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", help="use an existing OpenCV source tree instead of downloading")
    parser.add_argument("--force", action="store_true", help="rebuild even if an install already exists")
    parser.add_argument("-j", "--jobs", type=int, default=0, help="parallel build jobs (0 = all cores)")
    args = parser.parse_args()

    if already_built() and not args.force:
        print(f"already built at {INSTALL_PREFIX.relative_to(REPO_ROOT)} (use --force to rebuild)")
        return 0

    with tempfile.TemporaryDirectory(prefix="opencv-build-") as scratch:
        scratch_path = Path(scratch)
        if args.source_dir:
            source_dir = Path(args.source_dir).expanduser().resolve()
            if not (source_dir / "CMakeLists.txt").is_file():
                print(f"error: {source_dir} does not look like an OpenCV source tree", file=sys.stderr)
                return 2
        else:
            archive = download(scratch_path / f"opencv-{OPENCV_VERSION}.tar.gz")
            source_dir = extract(archive, scratch_path / "src")

        stage = scratch_path / "stage"
        build_dir = scratch_path / "build"
        run(cmake_arguments(source_dir, build_dir, stage))
        jobs = args.jobs or (os.cpu_count() or 2)
        run(["cmake", "--build", str(build_dir), "--target", "install", "-j", str(jobs)])

        verify(stage)
        swap_into_place(stage)
    print(f"\ninstalled minimal OpenCV {OPENCV_VERSION} -> {INSTALL_PREFIX.relative_to(REPO_ROOT)}")
    print("scripts/build_native.py will pick it up automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
