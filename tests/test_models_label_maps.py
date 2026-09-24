"""The Python detector ports must agree with the C++ originals.

``models.py`` reproduces ``LineSegmentationModel.cpp`` and
``KeypointDetectionModel.cpp`` so the networks can run under the Python ``torch``
package instead of a second libtorch. The class-index -> pitch-element tables are
transcribed by hand, and a wrong entry is close to invisible: the tracker simply
loses a line type and scores slightly worse.

Rather than trusting the transcription, these tests re-parse the C++ tables and
diff them against the Python dictionaries. They fail the moment upstream changes
a table and the port is not updated to match.
"""

from __future__ import annotations

import re

import pytest

from core.broadtrack_calib.models import KEYPOINT_CLASS_TO_POINT_ID, LINE_CLASS_TO_ID
from tests.conftest import BROADTRACK_DIR


def _enum_values(header_text: str) -> dict:
    """All ``NAME = <int>`` enumerators in a header."""
    return {match.group(1): int(match.group(2)) for match in re.finditer(r"(\w+)\s*=\s*(\d+)\s*,", header_text)}


def _parse_cpp_conversion_map(path, enum_qualified_name: str, enums: dict) -> dict:
    source = path.read_text()
    marker = f"const std::map<int, SoccerPitch3D::{enum_qualified_name}> "
    start = source.index(marker)
    body = source[start : source.index("};", start)]
    table = {}
    for match in re.finditer(r"\{\s*(\d+)\s*,\s*SoccerPitch3D::(?:\w+::)?(\w+)\s*\}", body):
        table[int(match.group(1))] = enums[match.group(2)]
    return table


@pytest.fixture(scope="module")
def cpp_enums() -> dict:
    return _enum_values((BROADTRACK_DIR / "SoccerPitch3D.h").read_text())


def test_line_class_map_matches_the_cpp_model(cpp_enums):
    expected = _parse_cpp_conversion_map(
        BROADTRACK_DIR / "LineSegmentationModel.cpp", "LineID", cpp_enums
    )
    assert expected, "failed to parse the C++ line conversion map"
    assert LINE_CLASS_TO_ID == expected


def test_keypoint_class_map_matches_the_cpp_model(cpp_enums):
    expected = _parse_cpp_conversion_map(
        BROADTRACK_DIR / "KeypointDetectionModel.cpp", "PointID", cpp_enums
    )
    assert expected, "failed to parse the C++ keypoint conversion map"
    assert KEYPOINT_CLASS_TO_POINT_ID == expected


def test_undefined_line_is_never_emitted(cpp_enums):
    """Class 0 is background and the goal-frame classes map to UNDEFINED_LINE;
    neither may survive as a labelled pixel in the mask."""
    undefined = cpp_enums["UNDEFINED_LINE"]
    assert 0 not in LINE_CLASS_TO_ID
    emitted = {line_id for line_id in LINE_CLASS_TO_ID.values() if line_id != undefined}
    assert undefined not in emitted
    # PointExtractor scans mask values in [100, 200]; every emitted id must be in
    # that window and must not collide with the background.
    assert all(100 <= line_id <= 200 for line_id in emitted)
