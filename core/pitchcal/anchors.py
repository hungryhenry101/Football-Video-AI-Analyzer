"""Anchor files: one known camera for one or more frames of a video.

An anchor is the human-guidance seam of the pipeline.  It can come from the
click tool (``pitchcal.annotate``), from the optional BroadTrack bootstrap, or
from an existing venue profile.  Everything downstream consumes only
``f, R, C, pitch`` -- no model or dataset dependency at run time.

World frame: metres, origin at the pitch centre, x along the length, y across
the width, z up.  Camera frame: x right, y down, z forward.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .geometry import PinholeCamera


@dataclass
class Anchor:
    frame: int
    width: int
    height: int
    f: float
    R: np.ndarray
    C: np.ndarray
    source: str = "unknown"
    score: float = 0.0
    k1: float = 0.0
    time_sec: float = 0.0
    points: List[Dict[str, Any]] = field(default_factory=list)
    lines: List[Dict[str, Any]] = field(default_factory=list)
    line_hit: float = 0.0
    line_cov: float = 0.0
    H: Optional[np.ndarray] = None

    def camera(self, work_width: Optional[int] = None, work_height: Optional[int] = None) -> PinholeCamera:
        """Camera rescaled to a working resolution (intrinsics scale, pose stays)."""
        w = int(work_width or self.width)
        h = int(work_height or self.height)
        sx = w / float(self.width)
        sy = h / float(self.height)
        if abs(sx - sy) > 1e-6:
            raise ValueError("work and native aspect ratios differ")
        return PinholeCamera(self.f * sx, w / 2.0, h / 2.0,
                             np.asarray(self.R, float).copy(),
                             np.asarray(self.C, float).copy())

    def homography(self, work_width: Optional[int] = None,
                   work_height: Optional[int] = None) -> np.ndarray:
        """Ground-plane world->image homography at the requested resolution.

        A manual anchor can store H directly: a homography is all the BEV and
        the tracker need, and it avoids the two-fold camera-pose ambiguity of
        decomposing a point homography.
        """
        if self.H is not None:
            H = np.asarray(self.H, float).copy()
            w = int(work_width or self.width)
            h = int(work_height or self.height)
            sx, sy = w / float(self.width), h / float(self.height)
            S = np.diag([sx, sy, 1.0])
            H = S @ H
            if abs(H[2, 2]) > 1e-12:
                H /= H[2, 2]
            return H
        return self.camera(work_width, work_height).homography()


@dataclass
class AnchorFile:
    video: str
    pitch: tuple[float, float]
    anchors: List[Anchor]
    path: Optional[str] = None

    def nearest(self, frame: int, tol: int = 90) -> Optional[Anchor]:
        if not self.anchors:
            return None
        anchor = min(self.anchors, key=lambda a: abs(a.frame - frame))
        return anchor if abs(anchor.frame - frame) <= tol else None

    def at_or_before(self, frame: int) -> Optional[Anchor]:
        candidates = [a for a in self.anchors if a.frame <= frame]
        return max(candidates, key=lambda a: a.frame) if candidates else None


def load_anchor_file(path: str | Path) -> AnchorFile:
    data = json.loads(Path(path).read_text())
    pitch = tuple(float(v) for v in data.get("pitch", [105.0, 68.0]))
    anchors: List[Anchor] = []
    for a in data.get("anchors", []):
        anchors.append(Anchor(
            frame=int(a["frame"]),
            width=int(a.get("width", 1920)),
            height=int(a.get("height", 1080)),
            f=float(a.get("f", 0.0)),
            R=np.asarray(a.get("R", np.zeros((3, 3))), float),
            C=np.asarray(a.get("C", np.zeros(3)), float),
            source=str(a.get("source", data.get("source", "unknown"))),
            score=float(a.get("score", 0.0)),
            k1=float(a.get("k1", 0.0)),
            time_sec=float(a.get("time_sec", 0.0)),
            points=list(a.get("points", [])),
            lines=list(a.get("lines", [])),
            line_hit=float(a.get("line_hit", 0.0)),
            line_cov=float(a.get("line_cov", 0.0)),
            H=(np.asarray(a["H"], float) if a.get("H") is not None else None),
        ))
    anchors.sort(key=lambda a: a.frame)
    return AnchorFile(video=str(data.get("video", "")), pitch=pitch, anchors=anchors, path=str(path))


def save_anchor_file(path: str | Path, video: str, pitch: tuple[float, float],
                     anchors: List[Anchor], source: str = "manual") -> None:
    data = {
        "version": 1,
        "video": str(video),
        "pitch": [float(pitch[0]), float(pitch[1])],
        "source": source,
        "anchors": [{
            "frame": int(a.frame),
            "time_sec": float(a.time_sec),
            "width": int(a.width),
            "height": int(a.height),
            **({"f": float(a.f), "R": np.asarray(a.R, float).tolist(),
                "C": np.asarray(a.C, float).tolist()} if a.H is None else {}),
            "k1": float(a.k1),
            "source": a.source,
            "score": float(a.score),
            "line_hit": float(a.line_hit),
            "line_cov": float(a.line_cov),
            "points": a.points,
            "lines": a.lines,
            **({"H": np.asarray(a.H, float).tolist()} if a.H is not None else {}),
        } for a in anchors],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def anchor_from_venue_profile(path: str | Path, pitch_override: Optional[tuple[float, float]] = None) -> AnchorFile:
    """Read the reference project's venue JSON (z-down) into our z-up format."""
    data = json.loads(Path(path).read_text())
    pitch = tuple(float(v) for v in data.get("pitch", {}).values()) if isinstance(data.get("pitch"), dict) else (105.0, 68.0)
    if pitch_override:
        pitch = pitch_override
    anchors = []
    for a in data.get("anchors", []):
        R_ref = np.asarray(a["R"], float)
        C_ref = np.asarray(a["t"], float)
        R = R_ref @ np.diag([1.0, 1.0, -1.0])
        C = np.array([C_ref[0], C_ref[1], -C_ref[2]], float)
        anchors.append(Anchor(
            frame=int(a["frame"]),
            width=int(a.get("frame_width", 1920)),
            height=int(a.get("frame_height", 1080)),
            f=float(np.asarray(a["K"], float)[0, 0]),
            R=R, C=C,
            source="venue-profile",
            score=float(a.get("line_iou", a.get("score", 0.0))),
            k1=float(a.get("k1", 0.0)),
            time_sec=float(a.get("time_sec", 0.0)),
            points=list(a.get("points", [])),
            lines=list(a.get("lines", [])),
            line_hit=float(a.get("line_hit", 0.0)),
            line_cov=float(a.get("line_cov", 0.0)),
            H=(np.asarray(a["H"], float) if a.get("H") is not None else None),
        ))
    anchors.sort(key=lambda a: a.frame)
    return AnchorFile(video=str(data.get("video", "")), pitch=pitch, anchors=anchors, path=str(path))
