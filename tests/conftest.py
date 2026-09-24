"""Shared fixtures for the BroadTrack calibration tests.

Two kinds of test live here. The geometry and label-map tests are free: they
need nothing but the compiled extension. The end-to-end ones need BroadTrack's
TorchScript detectors and a frame sequence, both of which are large and
gitignored, so they skip cleanly when the files are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BROADTRACK_DIR = REPO_ROOT / "core" / "BroadTrack"
KEYPOINT_WEIGHTS = REPO_ROOT / "models" / "nbjw_keypoint_model.pt"
LINE_WEIGHTS = REPO_ROOT / "models" / "tvcalib_model.pt"
FRAMES_DIR = BROADTRACK_DIR / "frames"
REFERENCE_JSON = BROADTRACK_DIR / "out" / "anchored_pos.json"

#: The sequence the reference C++ binary was run on, and the scores it recorded.
#: 1920x1080, which is also the tracker's internal reference resolution.
FRAME_COUNT = 20


def _have_models() -> bool:
    return KEYPOINT_WEIGHTS.is_file() and LINE_WEIGHTS.is_file()


def _have_sequence() -> bool:
    return FRAMES_DIR.is_dir() and any(FRAMES_DIR.glob("*.jpg"))


requires_models = pytest.mark.skipif(
    not _have_models(), reason="BroadTrack TorchScript models are not downloaded"
)
requires_sequence = pytest.mark.skipif(
    not (_have_models() and _have_sequence()), reason="BroadTrack models or frames/ are missing"
)


@pytest.fixture(scope="session")
def extension():
    from core.broadtrack_calib import _broadtrack

    return _broadtrack


@pytest.fixture(scope="session")
def calib():
    """One BroadTrackCalib shared across the sequence tests (model load is slow)."""
    from core.broadtrack_calib import BroadTrackCalib

    return BroadTrackCalib(
        weights_kp=str(KEYPOINT_WEIGHTS),
        weights_line=str(LINE_WEIGHTS),
        device="cpu",
        width=1920,
        height=1080,
    )


@pytest.fixture(scope="session")
def frames():
    import cv2

    paths = sorted(FRAMES_DIR.glob("*.jpg"))[:FRAME_COUNT]
    return [(path.name, cv2.imread(str(path))) for path in paths]


@pytest.fixture(scope="session")
def reference():
    if not REFERENCE_JSON.is_file():
        pytest.skip("reference calibration json is missing")
    return json.loads(REFERENCE_JSON.read_text())
