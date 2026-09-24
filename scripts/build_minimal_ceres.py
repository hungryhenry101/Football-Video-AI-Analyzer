#!/usr/bin/env python
"""Build a minimal, self-contained Ceres Solver for the BroadTrack extension.

Usage
-----
    python scripts/build_minimal_ceres.py            # download, build, install
    python scripts/build_minimal_ceres.py --force    # rebuild even if present
    python scripts/build_minimal_ceres.py --source-dir /path/to/ceres-solver-2.2.0

Why this exists
---------------
A stock Ceres (Homebrew, apt, conda-forge) pulls in SuiteSparse/CHOLMOD and
OpenBLAS, both of which link an OpenMP runtime, and glog, which links gflags.
Inside a Python process that already runs PyTorch that is two problems:

* PyTorch's wheel ships its own OpenMP. Two OpenMP runtimes make libomp abort
  the interpreter outright -- ``OMP: Error #15: Initializing libomp.dylib, but
  found libomp.dylib already initialized.`` It used to be survivable because
  PyTorch's bundled libomp was stamped with the same install name as Homebrew's
  (both ``/opt/homebrew/opt/libomp/lib/libomp.dylib``), so dyld reused a single
  copy. PyTorch 2.14 stamps it ``/opt/llvm-openmp/lib/libomp.dylib`` instead, so
  the names no longer collide and both copies load.
* glog's gflags dependency can resolve to a *different* gflags build than the
  one glog itself links, and the two registries fuse through dyld's flat lookup:
  ``ERROR: flag 'flagfile' was defined more than once``.

This build removes both dependencies rather than papering over them. BroadTrack
solves one eight-parameter block against a few hundred line-segment residuals,
so SuiteSparse, LAPACK and multi-threaded BLAS buy nothing measurable; Eigen's
sparse Cholesky is more than enough. Ceres is built **static**, so it is linked
into the extension module and no libceres, libglog, libgflags, libopenblas or
libomp ever enters the process.

The result is installed under ``core/BroadTrack/third_party/ceres/``, which is
gitignored, and picked up automatically by ``scripts/build_native.py``.
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
INSTALL_PREFIX = REPO_ROOT / "core" / "BroadTrack" / "third_party" / "ceres"

CERES_VERSION = "2.2.0"
CERES_URL = f"https://github.com/ceres-solver/ceres-solver/archive/refs/tags/{CERES_VERSION}.tar.gz"

#: Ceres 2.2.0 predates Eigen 5 and refuses to configure against it. These are
#: the two upstream commits that fix that (the same pair Homebrew's ceres-solver
#: formula applies). Without them, CMake stops at
#:     Could not find a configuration file for package "Eigen3" that is
#:     compatible with requested version "3.3"
#: 2.2.0 is still the newest Ceres release, so there is no version to upgrade to.
CERES_PATCHES = (
    # Raise minimum required Eigen version to 3.3.4
    "https://github.com/ceres-solver/ceres-solver/commit/f0720aeb84ec7bb479fe3618b30fa54981baf8fd.patch",
    # Support Eigen3 5.0.0
    "https://github.com/ceres-solver/ceres-solver/commit/f9b7b6651b108136a16df44d91fb31735645f5a7.patch?full_index=1",
)

#: Applied in order; the second rewrites the line the first introduces.
PATCHED_MARKER = "3.3.4...5"

#: Silence Ceres' VLOG tracing in the bundled miniglog.
#:
#: miniglog's native VLOG has no level gate at all -- the macro expands straight
#: to a MessageLogger, and unlike real glog it reads neither GLOG_v nor
#: GLOG_minloglevel (Ceres' own comment: "Currently, VLOG is always on"). Every
#: solve therefore prints a dozen lines of iteration trace to stderr, which at
#: one solve per video frame is unusable. MAX_LOG_LEVEL cannot fix it either: it
#: compares magnitudes, so anything low enough to drop VLOG(1) also drops
#: LOG(WARNING) and LOG(ERROR).
#:
#: The substitutions below turn VLOG into the same never-evaluated form the
#: header already uses for its own DLOG, so LOG/CHECK/ERROR still work and only
#: the tracing goes. Pass --keep-ceres-vlog to build without them.
#:
#: The header defines VLOG twice, in an #ifdef MAX_LOG_LEVEL branch and an #else
#: branch, and MAX_LOG_LEVEL is defined in Ceres' own build, so the first branch
#: is the live one. Both are rewritten here -- patching only the #else left the
#: tracing fully intact, which is how this was found.
MINIGLOG_PATH = "internal/ceres/miniglog/glog/logging.h"
_MINIGLOG_NOOP = '#  define VLOG{n}(n{s}) true ? (void) 0 : LoggerVoidify() & \\\n      MessageLogger((char *)__FILE__, __LINE__, "native", n).stream()'
MINIGLOG_VLOG_SUBSTITUTIONS = (
    # #ifdef MAX_LOG_LEVEL branch (the one that is actually compiled)
    ("#  define VLOG(n) LOG_IF(n, n <= MAX_LOG_LEVEL)", _MINIGLOG_NOOP.format(n="", s="")),
    (
        "#  define VLOG_IF(n, condition) LOG_IF(n, (n <= MAX_LOG_LEVEL) && condition)",
        _MINIGLOG_NOOP.format(n="_IF", s=", condition"),
    ),
    # #else branch
    (
        '#  define VLOG(n) MessageLogger((char *)__FILE__, __LINE__, "native", n).stream()    // NOLINT',
        _MINIGLOG_NOOP.format(n="", s=""),
    ),
    (
        "#  define VLOG_IF(n, condition) LOG_IF(n, condition)",
        _MINIGLOG_NOOP.format(n="_IF", s=", condition"),
    ),
    ("#  define VLOG_IS_ON(x) (1)", "#  define VLOG_IS_ON(x) (0)"),
    ("#  define VLOG_IS_ON(x) (x <= MAX_LOG_LEVEL)", "#  define VLOG_IS_ON(x) (0)"),
)

#: Libraries that must not appear in the finished build. Checked after install.
FORBIDDEN_CAPSULES = ("libomp", "libglog", "libgflags", "libopenblas", "libcholmod", "libspqr")


def run(command: list[str], **kwargs) -> None:
    print(f"$ {' '.join(str(part) for part in command)}", flush=True)
    subprocess.run(command, check=True, **kwargs)


def already_built() -> bool:
    return (INSTALL_PREFIX / "lib" / "cmake" / "Ceres" / "CeresConfig.cmake").is_file()


def download(destination: Path) -> Path:
    print(f"downloading Ceres {CERES_VERSION} ...", flush=True)
    with urllib.request.urlopen(CERES_URL, timeout=120) as response:  # noqa: S310 (fixed https URL)
        destination.write_bytes(response.read())
    return destination


def extract(archive: Path, into: Path) -> Path:
    with tarfile.open(archive) as tar:
        # The tarball is fetched over https from a fixed GitHub tag; still, only
        # pull out regular files and directories so a malformed archive cannot
        # write outside `into`.
        members = [m for m in tar.getmembers() if m.isfile() or m.isdir()]
        try:
            tar.extractall(into, members=members, filter="data")
        except TypeError:  # `filter` is Python >= 3.12
            tar.extractall(into, members=members)
    roots = [p for p in into.iterdir() if p.is_dir()]
    if len(roots) != 1 or not (roots[0] / "CMakeLists.txt").is_file():
        raise RuntimeError(f"unexpected archive layout under {into}")
    return roots[0]


def apply_patches(source_dir: Path, scratch: Path) -> None:
    """Apply the Eigen 5 compatibility patches to a freshly extracted tree."""
    for index, url in enumerate(CERES_PATCHES, start=1):
        patch_file = scratch / f"eigen5-{index}.patch"
        print(f"fetching patch {index}/{len(CERES_PATCHES)} ...", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (fixed https URLs)
            patch_file.write_bytes(response.read())

        # The commits were authored against a later master, so their doc churn
        # (docs/source/installation.rst) does not match the 2.2.0 tarball. Only
        # the code hunks matter; Homebrew's formula trims the same way.
        if shutil.which("git"):
            command = ["git", "apply", "--verbose", "--exclude=docs/*", str(patch_file)]
        elif shutil.which("patch"):
            command = ["patch", "-p1", "--batch", "--forward", "--exclude=docs/*", "--input", str(patch_file)]
        else:
            raise RuntimeError("applying the Ceres Eigen 5 patches needs either `git` or `patch` on PATH")
        run(command, cwd=source_dir)

    if PATCHED_MARKER not in (source_dir / "CMakeLists.txt").read_text():
        raise RuntimeError("the Eigen 5 patches did not apply as expected")


def silence_miniglog_vlog(source_dir: Path) -> None:
    """Turn the bundled miniglog's VLOG macros into no-ops. See the note above."""
    header = source_dir / MINIGLOG_PATH
    text = header.read_text()
    for old, new in MINIGLOG_VLOG_SUBSTITUTIONS:
        if old not in text:
            raise RuntimeError(f"{MINIGLOG_PATH}: expected to find {old!r} to rewrite")
        text = text.replace(old, new, 1)
    header.write_text(text)
    print(f"patched {MINIGLOG_PATH}: VLOG tracing disabled", flush=True)


def cmake_arguments(source_dir: Path, build_dir: Path, stage: Path) -> list[str]:
    return [
        "cmake",
        "-S", str(source_dir),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={stage}",
        # Static: the archive is linked straight into the extension module, so
        # nothing of Ceres is left as a runtime dependency to collide with.
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DBUILD_TESTING=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_BENCHMARKS=OFF",
        # The point of the exercise.
        "-DUSE_OPENMP=OFF",
        "-DSUITESPARSE=OFF",
        "-DLAPACK=OFF",
        "-DCXSPARSE=OFF",
        # Graph partitioning, only worth it for large problems; leaving it on
        # makes the exported Ceres target drag in the system METIS.
        "-DMETIS=OFF",
        "-DEIGENSPARSE=ON",
        # Ceres' bundled logger, so glog (and therefore gflags) stays out.
        "-DMINIGLOG=ON",
    ]


def verify(prefix: Path) -> None:
    """Fail if anything we were trying to avoid ended up in the install tree."""
    offenders = []
    for path in list(prefix.rglob("*.dylib")) + list(prefix.rglob("*.so")) + list(prefix.rglob("*.a")):
        name = path.name.lower()
        if any(capsule in name for capsule in FORBIDDEN_CAPSULES):
            offenders.append(path.name)
    if offenders:
        raise RuntimeError(f"unexpected libraries in the Ceres install: {offenders}")


def swap_into_place(staged: Path) -> None:
    """Replace INSTALL_PREFIX with a freshly staged tree.

    Staged rather than installed in place so that a failed or interrupted build
    leaves the previous install untouched -- an earlier version of this script
    deleted the prefix first under --force, and a build error then left the
    project with no Ceres at all and the extension silently linked against the
    system one.
    """
    if INSTALL_PREFIX.exists():
        shutil.rmtree(INSTALL_PREFIX)
    INSTALL_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(INSTALL_PREFIX))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", help="use an existing Ceres source tree instead of downloading")
    parser.add_argument("--force", action="store_true", help="rebuild even if an install already exists")
    parser.add_argument("-j", "--jobs", type=int, default=0, help="parallel build jobs (0 = all cores)")
    parser.add_argument("--keep-ceres-vlog", action="store_true",
                        help="do not silence miniglog VLOG tracing (noisy: ~15 lines per solve)")
    args = parser.parse_args()

    if already_built() and not args.force:
        print(f"already built at {INSTALL_PREFIX.relative_to(REPO_ROOT)} (use --force to rebuild)")
        return 0

    with tempfile.TemporaryDirectory(prefix="ceres-build-") as scratch:
        scratch_path = Path(scratch)
        if args.source_dir:
            source_dir = Path(args.source_dir).expanduser().resolve()
            if not (source_dir / "CMakeLists.txt").is_file():
                print(f"error: {source_dir} does not look like a Ceres source tree", file=sys.stderr)
                return 2
        else:
            archive = download(scratch_path / f"ceres-{CERES_VERSION}.tar.gz")
            source_dir = extract(archive, scratch_path / "src")
            apply_patches(source_dir, scratch_path)

        if not args.keep_ceres_vlog:
            silence_miniglog_vlog(source_dir)

        stage = scratch_path / "stage"
        build_dir = scratch_path / "build"
        run(cmake_arguments(source_dir, build_dir, stage))
        jobs = args.jobs or (os.cpu_count() or 2)
        run(["cmake", "--build", str(build_dir), "--target", "install", "-j", str(jobs)])

        verify(stage)
        swap_into_place(stage)

    print(f"\ninstalled minimal Ceres {CERES_VERSION} -> {INSTALL_PREFIX.relative_to(REPO_ROOT)}")
    print("scripts/build_native.py will pick it up automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
