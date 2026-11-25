"""Point-click and white-line tracing tool for manual pitch anchors.

Usage:
    python -m pitchcal.cli annotate --video vids/U15-bj.mp4 --frame 5844 \
        --out configs/anchors/u15_manual.json --pitch 105x68

    python -m pitchcal.cli annotate --video vids/U15-bj.mp4 --frame 7000 \
        --mode lines --out configs/anchors/u15_lines.json --pitch 105x68

Line mode is the practical choice for central views: pick a template line from
the panel and drag along the visible white paint.  With ``--append`` the
previous anchor initializes the line solve, so it carries the far-field scale
while the traced lines snap the current frame onto the paint.

Points and lines can be combined on the same frame: mark the landmarks in point
mode, press ``m`` to trace lines, then press ``s``.  Both lists are kept across
mode switches and both are given to the solver -- the lines constrain the near
field along their length, the clicks pin the far-field scale (and a
well-conditioned click seed is never discarded in favour of a lines-only fit).

After a solve the evidence that does *not* agree with the result is shown in
red: a click RANSAC rejected, or a traced line whose median distance to the
projected template is far above the rest.  The same list is printed to the
console.  Nothing is deleted automatically -- re-click / undo and solve again.

For a hard venue (for example the Beijing low-contrast PTZ match), use
``--append --step 1`` and press ``a`` after each anchor: the tool saves the
current anchor and jumps forward one second, carrying the previous anchor's
pitch overlay as a visual guide.

Keys
----
m            toggle point / line mode (clicks and lines are kept)
n / p        next / previous landmark or template line
1..9         jump to landmark / line
left click   point mode: record the selected landmark
drag         line mode: trace the selected white line, release to commit
click        line mode: place a vertex (two clicks finish a straight line;
             for the centre circle click several points around the arc)
f / Enter    line mode: finish the current click-built line
right click  undo the last click / the current or last traced line
u            undo (mode-specific)
s            solve and show the projected pitch + global paint support;
             outliers (clicked points and traced lines) are painted red
g            snap the solved anchor onto the observed paint (gated; opt-in)
w            save anchor JSON (press twice if the solve was not accepted)
a            save anchor, then advance by --step seconds
k            skip this step: advance by --step seconds without saving
, / .        previous / next video frame (skip a bad frame)
r            clear all clicks and traced lines
q / ESC      quit
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .anchors import Anchor, load_anchor_file, save_anchor_file
from .geometry import PitchTemplate, project_polyline_segments
from .segments import paint_support, refine_homography
from .solve import (line_metrics_from_homography, point_layout_problem,
                    solve_homography_from_clicks, solve_homography_from_lines)

WINDOW_TITLE = "PitchCal - annotate"
# The pitch reference panel sits left of the video in one window.  The whole
# canvas is scaled down to fit these limits so both stay fully on screen (the
# built-in 13" display is only 1280 points wide).
PANEL_W = 320
WINDOW_MAX_W = 1280
WINDOW_MAX_H = 780

# Ordered from the easiest / most common points.  A human does not need all of
# them; 6-8 spread over the pitch are ideal.
#
# The first four are deliberately *not* collinear.  A homography needs four
# points in general position; the old order (centre mark, circle top, circle
# bottom, circle left) has three of those on the halfway line, so a human who
# stopped after four clicks got an arbitrary member of a one-parameter
# family -- a silently wrong anchor.  `point_layout_problem` now rejects such
# a set, and this order makes the common four-click case well posed.
RECOMMENDED = [
    "center_mark",
    "center_circle_top",
    "center_circle_left",
    "left_penalty_top_inner",
    "right_penalty_top_inner",
    "center_circle_right",
    "center_circle_bottom",
    "halfway_top_touch",
    "halfway_bottom_touch",
    "left_corner_top",
    "right_corner_top",
    "left_penalty_bottom_inner",
    "right_penalty_bottom_inner",
    "left_goal_area_top_inner",
    "right_goal_area_top_inner",
    "left_corner_bottom",
    "right_corner_bottom",
    # Line/arc intersections.  Unlike the centre-circle left/right landmarks,
    # these are projected *intersection points*: the visible junction of two
    # painted lines is the image of the world junction, so a click on it does
    # not inherit the "visible extreme of the projected ellipse is not the
    # projected world extreme" error.  They are only usable when both lines
    # are visible, which is why they come after the easy points.
    "left_goal_area_bottom_inner",
    "right_goal_area_bottom_inner",
    "left_penalty_top_goal_line",
    "right_penalty_top_goal_line",
    "left_penalty_bottom_goal_line",
    "right_penalty_bottom_goal_line",
    "left_goal_area_top_goal_line",
    "right_goal_area_top_goal_line",
    "left_goal_area_bottom_goal_line",
    "right_goal_area_bottom_goal_line",
    "left_penalty_arc_top",
    "right_penalty_arc_top",
    "left_penalty_arc_bottom",
    "right_penalty_arc_bottom",
]

# Ordered for central / tactical views: the lines that are visible when the
# camera is parked near the halfway line come first.  A human only needs the
# lines they can actually see; 3-6 are plenty.
RECOMMENDED_LINES = [
    "halfway_line",
    "center_circle",
    "touchline_top",
    "touchline_bottom",
    "left_penalty_front",
    "right_penalty_front",
    "left_penalty_top",
    "left_penalty_bottom",
    "right_penalty_top",
    "right_penalty_bottom",
    "left_goal_area_front",
    "right_goal_area_front",
    "left_goal_area_top",
    "left_goal_area_bottom",
    "right_goal_area_top",
    "right_goal_area_bottom",
    "goal_line_left",
    "goal_line_right",
    "left_penalty_arc",
    "right_penalty_arc",
    "corner_tl",
    "corner_tr",
    "corner_br",
    "corner_bl",
]

CURVED_LINES = {
    "center_circle", "left_penalty_arc", "right_penalty_arc",
    "corner_tl", "corner_tr", "corner_br", "corner_bl",
}


def _text_panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.62) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


class ClickAnnotator:
    def __init__(self, video: str, frame_idx: int, out: str,
                 pitch: Tuple[float, float] = (105.0, 68.0),
                 append: bool = False, step_sec: float = 2.0,
                 view_w: int = 1280, view_h: int = 720,
                 mode: str = "points"):
        self.video = video
        self.out = out
        self.template = PitchTemplate(*pitch)
        self.pitch = pitch
        self.append = append
        self.view_w, self.view_h = view_w, view_h
        self.mode = mode if mode in ("points", "lines") else "points"
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {video}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        self.step_frames = max(1, int(round(float(step_sec) * self.fps)))
        self.step_sec = float(step_sec)
        self.existing: List[Anchor] = []
        if append and Path(out).exists():
            self.existing = load_anchor_file(out).anchors
        self.message = ""
        self.points: List[dict] = []
        self.lines: List[dict] = []
        self.idx = 0
        self.line_names = list(RECOMMENDED_LINES)
        self.line_index = 0
        self.current_line: Optional[dict] = None
        self.drag_last_native: Optional[Tuple[float, float]] = None
        self.dragging = False
        self.click_mode = False
        self.press_display: Optional[Tuple[int, int]] = None
        self.press_native: Optional[Tuple[float, float]] = None
        self.moved_since_press = False
        self.H: Optional[np.ndarray] = None
        self.solve_info = None
        # Solved-evidence diagnostics the human can act on: which clicks the
        # RANSAC/residual test does not explain, and which traced lines the
        # solved template does not explain.  Cached because the line metrics
        # are too expensive to recompute in every draw().
        self.outlier_points: List[dict] = []
        self.outlier_lines: List[dict] = []
        # Signature of a rejected solve the human has already been warned
        # about.  A second `w` on the same solve saves it; any change to the
        # frame/clicks/lines produces a new signature and asks again.
        self.pending_save: Optional[tuple] = None
        self.panel_w = PANEL_W
        self.cursor = (view_w // 2, view_h // 2)
        self.canvas_scale = 1.0
        self._load_frame(int(frame_idx))

    # ------------------------------------------------------------------ frame
    def _load_frame(self, frame_idx: int) -> None:
        if self.total_frames:
            frame_idx = int(np.clip(frame_idx, 0, self.total_frames - 1))
        cap = cv2.VideoCapture(self.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"cannot read frame {frame_idx}")
        self.frame_idx = int(frame_idx)
        self.frame = frame
        self.native_h, self.native_w = frame.shape[:2]
        self.scale = min(self.view_w / self.native_w, self.view_h / self.native_h)
        self.disp_w = int(round(self.native_w * self.scale))
        self.disp_h = int(round(self.native_h * self.scale))
        self.display = cv2.resize(frame, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
        self.cursor = (self.disp_w // 2, self.disp_h // 2)
        self.points = []
        self.lines = []
        self.idx = 0
        self.line_index = 0
        self.current_line = None
        self.drag_last_native = None
        self.dragging = False
        self.click_mode = False
        self.press_display = None
        self.press_native = None
        self.moved_since_press = False
        self.H = None
        self.solve_info = None
        self.outlier_points = []
        self.outlier_lines = []
        self.previous_H = self._previous_h(self.frame_idx)
        print(f"frame {self.frame_idx} (t={self.frame_idx / max(self.fps,1e-9):.2f}s)")

    def _previous_h(self, frame_idx: int) -> Optional[np.ndarray]:
        if not self.existing:
            return None
        usable = [a for a in self.existing if a.frame <= frame_idx
                  and (a.H is not None or a.f > 0)]
        if usable:
            a = max(usable, key=lambda x: x.frame)
        else:
            a = min(self.existing, key=lambda x: abs(x.frame - frame_idx))
        if a.H is not None:
            return np.asarray(a.H, float)
        if a.f > 0:
            try:
                return a.homography(self.native_w, self.native_h)
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------ actions
    def select(self, i: int) -> None:
        self.idx = int(np.clip(i, 0, len(RECOMMENDED) - 1))

    def add_click(self, x_disp: int, y_disp: int) -> None:
        if self.idx >= len(RECOMMENDED):
            return
        name = RECOMMENDED[self.idx]
        x = float(np.clip(x_disp / self.scale, 0, self.native_w - 1))
        y = float(np.clip(y_disp / self.scale, 0, self.native_h - 1))
        world = self.template.landmarks[name]
        self.points = [p for p in self.points if p["name"] != name]
        self.points.append({"name": name, "world": [float(world[0]), float(world[1])],
                            "pixel": [x, y]})
        self._clear_outliers()
        self.idx = min(self.idx + 1, len(RECOMMENDED) - 1)
        self.message = f"marked {name}"
        problem = self._layout_problem()
        if problem is not None:
            # Tell the human immediately, not after a failed solve.
            self.message = f"layout will not solve: {problem}"

    def _layout_problem(self) -> Optional[str]:
        if len(self.points) < 4:
            return None
        world = np.asarray([p["world"] for p in self.points], float)
        image = np.asarray([p["pixel"] for p in self.points], float)
        return point_layout_problem(world, image)

    # ------------------------------------------------------------- outliers
    def _clear_outliers(self) -> None:
        """Annotations changed, so the previous solve's diagnostics are stale."""
        self.outlier_points = []
        self.outlier_lines = []

    def _point_diagnostics(self) -> List[dict]:
        """Clicks the current homography does not explain.

        A point-only solve carries an authoritative RANSAC mask.  A joint
        point+line solve does not: there the clicks are soft guidance, so a
        click is reported only when its residual is far above the robust
        median of the other clicks.
        """
        if self.H is None or not self.points:
            return []
        world = np.asarray([p["world"] for p in self.points], float)
        image = np.asarray([p["pixel"] for p in self.points], float)
        hom = np.column_stack([world, np.ones(len(world))])
        q = (np.asarray(self.H, float) @ hom.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = q[:, :2] / q[:, 2:3]
        err = np.linalg.norm(uv - image, axis=1)
        err = np.where(np.isfinite(err), err, np.inf)
        info = self.solve_info or {}
        if info.get("point_layout_problem"):
            # The clicks were rejected as a degenerate layout before solving;
            # marking them as outliers would blame the wrong thing.
            return []
        mask = info.get("inlier_mask")
        if mask is not None and len(mask) == len(err):
            bad = ~np.asarray(mask, bool)
        else:
            median = float(np.median(err)) if len(err) else 0.0
            bad = err > max(12.0, 3.0 * median)
        return [{
            "index": int(i),
            "name": str(self.points[i].get("name", f"point {i + 1}")),
            "error_px": float(err[i]),
        } for i in np.flatnonzero(bad)]

    def _update_outliers(self) -> None:
        """Recompute both outlier lists against the H that was just solved."""
        self.outlier_points = self._point_diagnostics()
        self.outlier_lines = []
        if self.H is not None and self.lines:
            try:
                metrics = line_metrics_from_homography(
                    self.H, self.lines, self.template, self.native_w, self.native_h)
                self.outlier_lines = list(metrics.get("outlier_lines", []))
                # A paint snap moves H after the solve; keep the displayed
                # line diagnostics consistent with the H that is now shown.
                info = self.solve_info or {}
                if "line_median_px" in info:
                    for key in ("line_median_px", "line_hit", "line_coverage",
                                "per_line", "worst_line", "worst_line_px",
                                "line_points", "outlier_lines",
                                "outlier_threshold_px"):
                        if key in metrics:
                            info[key] = metrics[key]
            except Exception:
                self.outlier_lines = []
        self._print_outliers()

    def _print_outliers(self) -> None:
        if self.outlier_points:
            print("outlier click(s):")
            for o in self.outlier_points:
                print(f"  {o['index'] + 1:2d}. {o['name']:28s} {o['error_px']:6.1f} px")
        elif len(self.points) >= 6:
            print("outlier clicks: none")
        elif self.points:
            print(f"outlier clicks: not checked (only {len(self.points)} click(s); "
                  "add a 6th for redundancy)")
        if self.outlier_lines:
            print("outlier line(s):")
            for o in self.outlier_lines:
                print(f"  {o['name']:26s} median {o['median_px']:6.1f} px  "
                      f"p90 {o['p90_px']:6.1f} px  hit {o['hit']:.2f}  "
                      f"n={o['points']}")
        elif self.lines:
            print("outlier lines: none")

    def _outlier_text(self) -> str:
        """One-line summary for the on-screen status message."""
        bits = []
        for o in self.outlier_points:
            bits.append(f"click {o['name']} {o['error_px']:.0f}px")
        for o in self.outlier_lines:
            bits.append(f"line {o['name']} {o['median_px']:.0f}px")
        if bits:
            return "  outliers: " + ", ".join(bits)
        if self.lines:
            return "  no outliers"
        if len(self.points) >= 6:
            return "  no click outliers"
        if self.points:
            return "  no outlier check (add a 6th click)"
        return ""

    def _outlier_count_text(self) -> str:
        n = len(self.outlier_points) + len(self.outlier_lines)
        return f"  !{n} outlier(s) in red" if n else ""

    def undo(self) -> None:
        if self.points:
            removed = self.points.pop()
            self._clear_outliers()
            self.message = f"undo {removed['name']}"
            # Like line mode: undo returns the selector to the landmark/line
            # that was just removed, so the next click re-marks it.
            if removed["name"] in RECOMMENDED:
                self.idx = RECOMMENDED.index(removed["name"])

    # ------------------------------------------------------------------ lines
    def select_line(self, i: int) -> None:
        if self.current_line is not None:
            self.commit_pending_line()
        self.line_index = int(np.clip(i, 0, len(self.line_names) - 1))

    def toggle_mode(self) -> None:
        if self.current_line is not None:
            self.commit_pending_line()
        self.mode = "lines" if self.mode == "points" else "points"
        self.message = ("line mode: drag or click vertices"
                        if self.mode == "lines" else "point mode: click landmarks")

    def _native_from_disp(self, x_disp: int, y_disp: int) -> Tuple[float, float]:
        return (float(np.clip(x_disp / self.scale, 0, self.native_w - 1)),
                float(np.clip(y_disp / self.scale, 0, self.native_h - 1)))

    def begin_line(self, x_disp: int, y_disp: int) -> None:
        x, y = self._native_from_disp(x_disp, y_disp)
        if self.current_line is not None:
            if self.click_mode:
                # A second/third click adds a vertex to the current polyline.
                self._add_click_vertex(x, y)
                return
            self.cancel_line()
        name = self.line_names[self.line_index]
        self.current_line = {"name": name, "pixels": [[x, y]]}
        self.press_display = (int(x_disp), int(y_disp))
        self.press_native = (x, y)
        self.moved_since_press = False
        self.drag_last_native = (x, y)
        self.dragging = True
        self.click_mode = False
        self.message = f"tracing {name}"

    def _add_click_vertex(self, x_native: float, y_native: float) -> None:
        if self.current_line is None:
            return
        pts = self.current_line["pixels"]
        last = np.asarray(pts[-1], float)
        if math.hypot(x_native - last[0], y_native - last[1]) < 1.5:
            return
        name = str(self.current_line["name"])
        pts.append([x_native, y_native])
        if name not in CURVED_LINES:
            # Two clicks define a straight line; commit it immediately.
            self._commit_current_line()
            return
        self.message = f"{name}: {len(pts)} point(s); click more, f finishes"

    def extend_line(self, x_disp: int, y_disp: int) -> None:
        if self.current_line is None or not self.dragging or self.click_mode:
            return
        x, y = self._native_from_disp(x_disp, y_disp)
        if self.press_display is not None:
            if math.hypot(x_disp - self.press_display[0],
                          y_disp - self.press_display[1]) > 3.0:
                self.moved_since_press = True
        last = self.drag_last_native
        if last is not None:
            dist = math.hypot(x - last[0], y - last[1])
            if dist < 1.5:
                return
        else:
            dist = 0.0
        name = str(self.current_line.get("name", ""))
        if name not in CURVED_LINES:
            # A straight template line stays straight while dragging: the live
            # preview is always the segment from the initial press to the
            # current cursor.  No freehand bends to clean up on release.
            start = self.current_line["pixels"][0]
            self.current_line["pixels"] = [start, [x, y]]
            self.drag_last_native = (x, y)
            return
        if last is not None:
            n = max(1, int(dist / 2.0))
            for k in range(1, n + 1):
                t = k / n
                self.current_line["pixels"].append(
                    [last[0] + t * (x - last[0]), last[1] + t * (y - last[1])])
        else:
            self.current_line["pixels"].append([x, y])
        self.drag_last_native = (x, y)

    def _simplify_line(self, name: str, pts: np.ndarray,
                       from_clicks: bool = False) -> np.ndarray:
        """Reduce mouse jitter without changing the line the human traced."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        if name in CURVED_LINES:
            if from_clicks:
                # Human-placed vertices are independent observations on the
                # arc.  Do not join/resample them into chord samples.
                return pts
            approx = cv2.approxPolyDP(
                pts.reshape(-1, 1, 2).astype(np.float32), 1.5, False).reshape(-1, 2)
            return approx.astype(float) if len(approx) >= 2 else pts
        # Straight template line: total-least-squares fit, keep its two extremes.
        centre = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centre)
        direction = vt[0]
        t = (pts - centre) @ direction
        return np.stack([centre + float(t.min()) * direction,
                         centre + float(t.max()) * direction], axis=0)

    def end_line(self) -> None:
        if self.current_line is None:
            self.dragging = False
            return
        if self.click_mode:
            # Vertex clicks are added on mouse-down; mouse-up does nothing.
            return
        if self.moved_since_press:
            self._commit_current_line()
            return
        # No meaningful movement: this was a click.  Keep the line alive and
        # collect vertices (centre circle) or wait for the end point (straight).
        name = str(self.current_line["name"])
        self.current_line["pixels"] = self.current_line["pixels"][:1]
        self.dragging = False
        self.drag_last_native = None
        self.click_mode = True
        self.message = ("click the end point" if name not in CURVED_LINES
                        else f"{name}: click points on the arc; f finishes")

    def _commit_current_line(self) -> bool:
        if self.current_line is None:
            return False
        name = str(self.current_line["name"])
        pts = np.asarray(self.current_line["pixels"], float)
        from_clicks = self.click_mode
        self.current_line = None
        self.dragging = False
        self.click_mode = False
        self.drag_last_native = None
        self.press_display = None
        self.press_native = None
        self.moved_since_press = False
        if len(pts) < 2:
            self.message = ("line needs another point" if from_clicks
                            else "line too short; drag along the white line")
            return False
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if length < 4.0:
            self.message = "line too short"
            return False
        simplified = self._simplify_line(name, pts, from_clicks=from_clicks)
        self.lines.append({"name": name, "pixels": simplified.tolist()})
        self._clear_outliers()
        self.line_index = min(self.line_index + 1, len(self.line_names) - 1)
        self.message = (f"traced {name} ({len(simplified)} pts); "
                        f"n/p select the next line, s solve")
        return True

    def finish_line(self) -> None:
        if self.current_line is None:
            return
        if len(self.current_line.get("pixels", [])) < 2:
            self.cancel_line()
            self.message = "line needs at least 2 points"
            return
        self._commit_current_line()

    def commit_pending_line(self) -> None:
        """Commit an in-progress line so s/w/a never silently drop vertices."""
        if self.current_line is None:
            return
        if len(self.current_line.get("pixels", [])) >= 2:
            self._commit_current_line()
        else:
            self.cancel_line()

    def cancel_line(self) -> None:
        if self.current_line is not None:
            self.current_line = None
            self.message = "cancelled line"
        self.dragging = False
        self.click_mode = False
        self.drag_last_native = None
        self.press_display = None
        self.press_native = None
        self.moved_since_press = False

    def undo_line(self) -> None:
        if self.current_line is not None:
            self.cancel_line()
            return
        if self.lines:
            removed = self.lines.pop()
            self._clear_outliers()
            self.message = f"undo {removed['name']}"
            if removed["name"] in self.line_names:
                self.line_index = self.line_names.index(removed["name"])

    def _report_line_solve(self) -> None:
        info = self.solve_info or {}
        median = float(info.get("line_median_px", float("nan")))
        hit = float(info.get("line_hit", 0.0))
        worst = info.get("worst_line", "?")
        worst_px = float(info.get("worst_line_px", 0.0))
        accepted = bool(info.get("accepted", True))
        self.message = (f"lines: median {median:.1f}px  hit {hit:.2f}  "
                        f"worst {worst} {worst_px:.1f}px"
                        + self._paint_text()
                        + self._outlier_text()
                        + ("" if accepted else "  NOT ACCEPTED")
                        + ("  [extrapolated]"
                           if info.get("method") == "lines-only-multistart" else "")
                        + ("  [seed rejected]"
                           if info.get("point_seed_rejected") else "")
                        + ("  [click layout degenerate; clicks ignored]"
                           if info.get("point_layout_problem") else ""))
        print(f"line solve ({info.get('method', '?')}): median {median:.2f} px, "
              f"hit {hit:.2f}, coverage {info.get('line_coverage', 0):.2f}, "
              f"accepted={accepted}"
              + (f", paint hit {info['paint_hit']:.2f} "
                 f"(median {info.get('paint_median_px', 99):.1f} px)"
                 if "paint_hit" in info else ""))
        for name, m in (info.get("per_line") or {}).items():
            print(f"  {name:24s} median {m.get('median_px', float('nan')):6.1f} px  "
                  f"p90 {m.get('p90_px', float('nan')):6.1f} px  "
                  f"hit {m.get('hit', 0):.2f}  n={m.get('points', 0)}")
        if info.get("point_before_px") is not None:
            print(f"  point seed: {info['point_before_px']:.1f} px before, "
                  f"{info.get('point_after_px', float('nan')):.1f} px after"
                  + ("  (points are guidance only)" if info.get("point_warning") else ""))
        if not accepted:
            cand = info.get("per_line_candidate") or {}
            if cand:
                print("  rejected candidate:")
                for name, m in cand.items():
                    print(f"    {name:22s} median {m.get('median_px', float('nan')):6.1f} px  "
                          f"hit {m.get('hit', 0):.2f}  n={m.get('points', 0)}")
            if info.get("line_median_candidate_px") is not None:
                print(f"  candidate median {info['line_median_candidate_px']:.2f} px "
                      f"(rejected)")
            print("  hint: the worst line is usually a wrong top/bottom label, "
                  "a non-pitch line, or a wrong --pitch size.")
        if info.get("warning"):
            print(f"  warning: {info['warning']}")

    def solve(self) -> None:
        self.commit_pending_line()
        if self.lines:
            self.message = f"solving {len(self.lines)} traced line(s)..."
            try:
                # Show the "solving" message before the from-scratch camera
                # search (which can take a few seconds).
                cv2.imshow(WINDOW_TITLE, self._compose())
                cv2.waitKey(1)
            except Exception:
                pass
            try:
                self.H, self.solve_info = solve_homography_from_lines(
                    self.lines, self.template, self.native_w, self.native_h,
                    points=self.points, init_H=self.previous_H,
                    allow_from_scratch=True)
                self._attach_paint_metrics()
                self._update_outliers()
                self._report_line_solve()
                return
            except Exception as exc:
                self.solve_info = {"error": str(exc)}
                self.H = None
                self._clear_outliers()
                if len(self.points) >= 4:
                    # A rough point seed can rescue a central view where the
                    # lines alone do not constrain the far field.
                    world = np.asarray([p["world"] for p in self.points], float)
                    image = np.asarray([p["pixel"] for p in self.points], float)
                    try:
                        self.H, self.solve_info = solve_homography_from_clicks(world, image)
                        self._attach_paint_metrics()
                        self._update_outliers()
                        self.message = ("line solve failed; used points: "
                                        f"median {self.solve_info['median_reproj_px']:.1f} px"
                                        + self._paint_text() + self._outlier_text())
                        return
                    except Exception:
                        pass
                self.message = f"line solve failed: {exc}"
                return
        if len(self.points) < 4:
            self.message = "need at least 4 clicks (or drag white lines in line mode)"
            return
        world = np.asarray([p["world"] for p in self.points], float)
        image = np.asarray([p["pixel"] for p in self.points], float)
        try:
            self.H, self.solve_info = solve_homography_from_clicks(world, image)
            self._attach_paint_metrics()
            self._update_outliers()
            self.message = (f"solved: median {self.solve_info['median_reproj_px']:.1f} px "
                            f"({self.solve_info['used_points']}/{len(self.points)} inliers)"
                            + self._paint_text() + self._outlier_text())
        except Exception as exc:
            self.solve_info = {"error": str(exc)}
            self.H = None
            self._clear_outliers()
            self.message = f"solve failed: {exc}"

    def _attach_paint_metrics(self) -> None:
        """Measure the anchor against all visible paint, not just the traces."""
        if self.H is None:
            return
        try:
            metrics = paint_support(self.frame, self.H, self.template,
                                    work_width=min(960, self.native_w))
        except Exception as exc:
            metrics = {"paint_error": str(exc)}
        self.solve_info = {**(self.solve_info or {}), **metrics}

    def _snap_to_paint(self) -> bool:
        """Small absolute correction toward the observed paint, strictly gated.

        This is the same ``refine_homography`` the pipeline runs every ~0.8 s.
        It is only applied here when the global paint support improves and the
        human's own evidence (traced lines, clicked points) does not get
        materially worse; otherwise the solved anchor is left untouched.  It
        rescues anchors whose clicks are a few pixels off without inventing a
        different far field.
        """
        info = dict(self.solve_info or {})
        if self.H is None or info.get("accepted") is False:
            return False
        if int(info.get("paint_segments", 0)) < 12:
            return False
        try:
            H_corr, corr = refine_homography(
                self.H, self.frame, self.template, None,
                gate_px=200.0, max_nfev=100)
            if H_corr is None or not corr.get("accepted"):
                return False
            after = paint_support(self.frame, H_corr, self.template,
                                  work_width=min(960, self.native_w))
        except Exception:
            return False
        paint_before = float(info.get("paint_hit", 0.0))
        paint_med_before = float(info.get("paint_median_px", 99.0))
        better = (after["paint_hit"] >= paint_before + 0.03
                  or after["paint_median_px"] <= paint_med_before * 0.8)
        if not better:
            return False
        if self.lines and "line_median_px" in info:
            m = line_metrics_from_homography(
                H_corr, self.lines, self.template, self.native_w, self.native_h)
            if (m["line_median_px"] > max(5.0, float(info["line_median_px"]) * 1.5)
                    or m["worst_line_px"] > max(12.0, float(info.get("worst_line_px", 0.0)) * 1.5)):
                return False
        if self.points:
            world = np.asarray([p["world"] for p in self.points], float)
            image = np.asarray([p["pixel"] for p in self.points], float)
            uv = cv2.perspectiveTransform(world[None].astype(np.float32), H_corr)[0]
            med = float(np.median(np.linalg.norm(uv - image, axis=1)))
            ref = info.get("point_after_px") or info.get("median_reproj_px")
            if ref is not None and med > max(12.0, float(ref) * 2.0):
                return False
        self.H = H_corr
        info.pop("inlier_mask", None)  # the RANSAC mask belongs to the old H
        self.solve_info = {
            **info, **after,
            "paint_refine": {
                "hit_before": paint_before, "hit_after": after["paint_hit"],
                "median_before_px": paint_med_before,
                "median_after_px": after["paint_median_px"],
            },
        }
        return True

    def snap(self) -> None:
        """Opt-in: pull the solved anchor toward the observed paint."""
        if self.H is None:
            self.solve()
        if self.H is None:
            return
        if self._snap_to_paint():
            self._update_outliers()
            r = (self.solve_info or {}).get("paint_refine", {})
            self.message = (f"paint-refined: hit {r.get('hit_before', 0):.2f}->"
                            f"{r.get('hit_after', 0):.2f}  median "
                            f"{r.get('median_before_px', 99):.0f}->"
                            f"{r.get('median_after_px', 99):.0f} px  — press w to save"
                            + self._outlier_text())
        else:
            self.message = ("paint snap not accepted (no clear improvement, or "
                            "it would move the clicks/lines too far); anchor unchanged")
        print(self.message)

    def _paint_text(self) -> str:
        info = self.solve_info or {}
        if "paint_hit" not in info:
            return ""
        return (f"  paint {info['paint_hit']:.2f}"
                f" (med {info.get('paint_median_px', 99):.0f} px)"
                + (" [paint-refined]" if info.get("paint_refine") else ""))

    def save(self) -> bool:
        self.commit_pending_line()
        if self.H is None:
            self.solve()
        if self.H is None:
            return False
        info = self.solve_info or {}
        # A rejected / extrapolated solve used to be saved silently, and the
        # pipeline then trusted it.  Ask once (press w again) instead.
        risky = (info.get("accepted") is False
                 or info.get("method") == "lines-only-multistart"
                 or info.get("point_seed_rejected"))
        signature = (self.frame_idx, np.round(np.asarray(self.H, float), 6).tobytes())
        if risky and self.pending_save != signature:
            self.pending_save = signature
            self.message = ("solve NOT ACCEPTED"
                            + self._paint_text()
                            + " — press w again to save anyway")
            print("warning: solve is not accepted ("
                  + str(info.get("warning") or info.get("reason") or "poor fit")
                  + "); press w again to save it anyway")
            return False
        self.pending_save = None
        has_line_metric = "line_median_px" in info
        line_score = (1.0 - min(float(info.get("line_median_px", 20.0)) / 20.0, 1.0)
                      if has_line_metric else 0.0)
        worst_score = (1.0 - min(float(info.get("worst_line_px", 20.0)) / 30.0, 1.0)
                       if has_line_metric else 0.0)
        if "median_reproj_px" in info:
            point_err = float(info["median_reproj_px"])
        elif info.get("point_after_px") is not None:
            point_err = float(info["point_after_px"])
        else:
            point_err = None
        point_score = (1.0 - min(point_err / 20.0, 1.0)
                       if point_err is not None else 0.0)
        if self.lines and self.points:
            source = "manual+lines"
        elif self.lines:
            source = "manual-lines"
        else:
            source = "manual"
        # The score is a review hint, not a calibration metric: include the
        # worst traced line and the global paint support so a fit that only
        # explains the long lines cannot score highly.
        parts = [point_score] if point_err is not None else []
        if has_line_metric:
            parts += [line_score, worst_score]
        if "paint_hit" in info:
            parts.append(float(info["paint_hit"]))
        score = float(np.mean(parts)) if parts else 0.0
        anchors = []
        if self.append and Path(self.out).exists():
            anchors = [a for a in load_anchor_file(self.out).anchors if a.frame != self.frame_idx]
        elif not self.append:
            anchors = [a for a in self.existing if a.frame != self.frame_idx]
        anchors.append(Anchor(
            frame=self.frame_idx, width=self.native_w, height=self.native_h,
            f=0.0, R=np.zeros((3, 3)), C=np.zeros(3),
            source=source,
            score=float(score),
            time_sec=self.frame_idx / max(self.fps, 1e-9),
            points=list(self.points),
            lines=list(self.lines),
            line_hit=float(info.get("line_hit", 0.0)),
            line_cov=float(info.get("line_coverage", 0.0)),
            H=self.H.copy(),
        ))
        anchors.sort(key=lambda a: a.frame)
        save_anchor_file(self.out, self.video, self.pitch, anchors, source=source)
        self.existing = anchors
        self.message = f"saved {len(anchors)} anchor(s) -> {Path(self.out).name}"
        detail = (f"{len(self.lines)} line(s), {len(self.points)} click(s)"
                  if self.lines else f"{len(self.points)} clicks")
        print(f"saved {self.out}  ({detail}, score {score:.2f})")
        return True

    def save_and_next(self) -> None:
        if not self.save():
            if self.pending_save is None:
                self.message = "solve/save failed; press s to inspect"
            return
        target = self.frame_idx + self.step_frames
        self._load_frame(target)
        self.message = (f"anchor saved; now frame {self.frame_idx} "
                        f"(+{self.step_sec:g}s). Previous anchor shown in gray.")

    def skip_step(self) -> None:
        """Move on to the next anchor step without saving the current frame.

        Handy when the current step is blurred, cut away or otherwise
        unmarkable: the anchor sequence stays aligned to ``--step`` while no
        bad anchor is written.
        """
        self._load_frame(self.frame_idx + self.step_frames)
        print(f"skipped step; frame {self.frame_idx} "
              f"(t={self.frame_idx / max(self.fps, 1e-9):.2f}s), nothing saved")
        self.message = (f"skipped this step; now frame {self.frame_idx} "
                        f"(+{self.step_sec:g}s). Nothing saved.")

    def jump_frame(self, direction: int) -> None:
        """Step one video frame, e.g. to skip a blurred/transition frame."""
        self._load_frame(self.frame_idx + direction)
        self.message = f"frame {self.frame_idx} (t={self.frame_idx / max(self.fps,1e-9):.2f}s)"

    # ------------------------------------------------------------------ drawing
    def _pitch_panel(self) -> np.ndarray:
        if self.mode == "lines":
            return self._line_panel()
        w, h = self.panel_w, self.disp_h
        panel = np.full((h, w, 3), 24, np.uint8)
        cv2.putText(panel, "PITCH LANDMARKS", (14, 26), cv2.FONT_HERSHEY_DUPLEX,
                    0.52, (230, 230, 230), 1, cv2.LINE_AA)
        current = RECOMMENDED[self.idx]
        cv2.putText(panel, f"POINT {self.idx + 1}/{len(RECOMMENDED)}", (14, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(panel, current, (14, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(panel, "green = current point   orange = clicked",
                    (14, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1, cv2.LINE_AA)

        # Schematic, scaled to the space left between the header and the hints.
        top = 120
        bottom = max(top + 60, h - 118)
        avail_w, avail_h = w - 28, bottom - top
        s = min(avail_w / self.template.length, avail_h / self.template.width)
        ox = (w - self.template.length * s) / 2
        oy = top + (avail_h - self.template.width * s) / 2

        def to_px(x, y):
            return (int(round(ox + (x + self.template.hx) * s)),
                    int(round(oy + (y + self.template.hy) * s)))
        for name, pts in self.template.polylines():
            px = np.array([to_px(x, y) for x, y in pts], np.int32)
            cv2.polylines(panel, [px], False, (190, 190, 190), 1, cv2.LINE_AA)
        clicked = {p["name"] for p in self.points}
        bad_points = {o["name"] for o in self.outlier_points}
        for name, (x, y) in self.template.landmarks.items():
            px = to_px(x, y)
            if name == current:
                cv2.circle(panel, px, 6, (80, 255, 80), -1, cv2.LINE_AA)
                cv2.putText(panel, name, (px[0] + 8, px[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 255, 80), 1, cv2.LINE_AA)
            elif name in bad_points:
                cv2.circle(panel, px, 4, (60, 60, 255), -1, cv2.LINE_AA)
                cv2.putText(panel, "!", (px[0] + 5, px[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (60, 60, 255), 1, cv2.LINE_AA)
            elif name in clicked:
                cv2.circle(panel, px, 4, (60, 190, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(panel, px, 2, (110, 110, 110), -1)
        hints = [
            "n/p  next/prev point",
            "click  mark   u undo   s solve",
            "g paint-snap   w save   a save+next",
            "k skip   r reset",
            "m line mode (clicks and lines are kept)",
            f"frame {self.frame_idx}  clicks {len(self.points)}"
            f"  lines {len(self.lines)}",
            (f"{len(self.lines)} line(s) also used by the solve"
             if self.lines else "trace lines in m mode for far-field support"),
        ]
        for j, line in enumerate(hints):
            cv2.putText(panel, line, (14, h - 121 + j * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 190, 190), 1, cv2.LINE_AA)
        return panel

    def _line_panel(self) -> np.ndarray:
        w, h = self.panel_w, self.disp_h
        panel = np.full((h, w, 3), 24, np.uint8)
        cv2.putText(panel, "PITCH LINES", (14, 26), cv2.FONT_HERSHEY_DUPLEX,
                    0.52, (230, 230, 230), 1, cv2.LINE_AA)
        current = self.line_names[self.line_index]
        cv2.putText(panel, f"LINE {self.line_index + 1}/{len(self.line_names)}", (14, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(panel, current, (14, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(panel, "green = current   orange = traced",
                    (14, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(panel, "magenta = click vertices (f finishes)",
                    (14, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 120, 255), 1, cv2.LINE_AA)

        schem_top, schem_h = 120, 110
        avail_w = w - 28
        s = min(avail_w / self.template.length, schem_h / self.template.width)
        ox = (w - self.template.length * s) / 2
        oy = schem_top + (schem_h - self.template.width * s) / 2

        def to_px(x, y):
            return (int(round(ox + (x + self.template.hx) * s)),
                    int(round(oy + (y + self.template.hy) * s)))

        for name, pts in self.template.polylines():
            px = np.array([to_px(x, y) for x, y in pts], np.int32)
            if name == current:
                cv2.polylines(panel, [px], False, (80, 255, 80), 2, cv2.LINE_AA)
            else:
                cv2.polylines(panel, [px], False, (170, 170, 170), 1, cv2.LINE_AA)

        traced = {str(entry.get("name", "")) for entry in self.lines}
        bad = {o["name"] for o in self.outlier_lines}
        tracing = self.current_line["name"] if self.current_line else None
        list_top = schem_top + schem_h + 14
        list_bottom = h - 118
        row_h = 14
        max_rows = max(1, int((list_bottom - list_top) // row_h))
        start = max(0, min(self.line_index - max_rows // 2,
                           len(self.line_names) - max_rows))
        for row in range(max_rows):
            i = start + row
            if i >= len(self.line_names):
                break
            name = self.line_names[i]
            if name in bad:
                prefix, color = "!", (60, 60, 255)
            elif name == current:
                prefix, color = ">", (80, 255, 80)
            elif name == tracing:
                prefix, color = "~", (255, 255, 0)
            elif name in traced:
                prefix, color = "x", (60, 190, 255)
            else:
                prefix, color = " ", (165, 165, 165)
            cv2.putText(panel, f"{prefix} {name}", (14, list_top + row * row_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

        hints = [
            "n/p next/prev line   1-9 jump",
            "drag trace   click add vertex",
            "f finish line   u undo",
            "s solve   g paint-snap   w save",
            "a save+next   k skip step",
            "m point mode (clicks and lines are kept)",
            f"frame {self.frame_idx}  pitch {self.template.length:.0f}x"
            f"{self.template.width:.0f}  lines {len(self.lines)}"
            f"  clicks {len(self.points)}",
        ]
        for j, line in enumerate(hints):
            cv2.putText(panel, line, (14, h - 121 + j * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 190, 190), 1, cv2.LINE_AA)
        return panel

    def _draw_polyline(self, img: np.ndarray, H: np.ndarray, pts: np.ndarray,
                       color: tuple, thickness: int) -> None:
        """Draw a pitch line, skipping anything behind the camera.

        ``project_polyline_segments`` drops points with w <= 0 and clips each
        visible segment to the frame, so a slightly extrapolated anchor can no
        longer mirror a template line over the horizon and paint it in the
        stands.
        """
        Hs = np.diag([self.scale, self.scale, 1.0]) @ H
        h, w = img.shape[:2]
        for (x0, y0), (x1, y1) in project_polyline_segments(Hs, pts, w, h):
            cv2.line(img, (int(round(x0)), int(round(y0))),
                     (int(round(x1)), int(round(y1))),
                     color, thickness, cv2.LINE_AA)

    def _draw(self) -> np.ndarray:
        img = self.display.copy()
        # Previous anchor (the one we are trying to replace) in gray.
        if self.previous_H is not None:
            for _, pts in self.template.polylines():
                self._draw_polyline(img, self.previous_H, pts, (145, 145, 145), 1)
        bad_points = {o["index"]: o for o in self.outlier_points}
        for k, p in enumerate(self.points):
            u = int(round(p["pixel"][0] * self.scale))
            v = int(round(p["pixel"][1] * self.scale))
            outlier = bad_points.get(k)
            if outlier is not None:
                # The solve could not explain this click: mark it so it can be
                # re-clicked or removed without re-reading the console.
                cv2.circle(img, (u, v), 7, (60, 60, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(img, (u, v), (60, 60, 255), cv2.MARKER_CROSS,
                               13, 2, cv2.LINE_AA)
                cv2.putText(img, f"! {outlier['name']} {outlier['error_px']:.0f}px",
                            (u + 9, v - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.40, (60, 60, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (u, v), 5, (0, 165, 255), -1, cv2.LINE_AA)
                cv2.putText(img, str(k + 1), (u + 7, v - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1,
                            cv2.LINE_AA)
        # Manually traced white lines (orange) and the one being dragged (cyan).
        # A line its own solve could not explain is red.
        bad_lines = {o["name"]: o for o in self.outlier_lines}
        for entry in self.lines:
            pts = np.asarray(entry.get("pixels", []), float) * self.scale
            if len(pts) < 2:
                continue
            name = str(entry.get("name", ""))
            outlier = bad_lines.get(name)
            color = (60, 60, 255) if outlier is not None else (0, 165, 255)
            px = np.round(pts).astype(np.int32)
            cv2.polylines(img, [px], False, color, 2, cv2.LINE_AA)
            lx, ly = int(px[0, 0]) + 4, int(px[0, 1]) - 4
            label = (f"! {name} {outlier['median_px']:.0f}px"
                     if outlier is not None else name)
            cv2.putText(img, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        # Draw the solved template last of the static layers: a correct green
        # model line must stay visible where it crosses the orange traces.
        # The in-progress line (cyan/magenta) is still drawn above it so the
        # human can see what they are currently tracing.
        if self.H is not None:
            for _, pts in self.template.polylines():
                self._draw_polyline(img, self.H, pts, (0, 255, 0), 2)
        # Redraw an outlier trace above the green template: the point of the
        # report is to see the disagreement, so the model must not hide it.
        for entry in self.lines:
            name = str(entry.get("name", ""))
            outlier = bad_lines.get(name)
            if outlier is None:
                continue
            pts = np.asarray(entry.get("pixels", []), float) * self.scale
            if len(pts) < 2:
                continue
            px = np.round(pts).astype(np.int32)
            cv2.polylines(img, [px], False, (60, 60, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"! {name} {outlier['median_px']:.0f}px",
                        (int(px[0, 0]) + 4, int(px[0, 1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 60, 255), 1,
                        cv2.LINE_AA)
        if self.current_line is not None:
            pts = np.asarray(self.current_line.get("pixels", []), float) * self.scale
            # Cyan = freehand drag, magenta = click-vertex mode (not yet
            # committed).  Both turn orange once committed.
            color = (255, 0, 255) if self.click_mode else (255, 255, 0)
            if len(pts) >= 2:
                cv2.polylines(img, [np.round(pts).astype(np.int32)], False,
                              color, 2, cv2.LINE_AA)
            if self.click_mode:
                # Show the clicked vertices and a rubber band to the cursor so
                # a centre-circle arc can be placed with a handful of clicks.
                for p in pts:
                    cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 3,
                               color, -1, cv2.LINE_AA)
                if len(pts) >= 1 and (0 <= self.cursor[0] < self.disp_w
                                      and 0 <= self.cursor[1] < self.disp_h):
                    last = pts[-1]
                    cv2.line(img, (int(round(last[0])), int(round(last[1]))),
                             self.cursor, color, 1, cv2.LINE_AA)
        # Top status panel: the current line/landmark is the largest text on
        # screen, which is the thing a human needs while marking.
        _text_panel(img, 0, 0, img.shape[1], 104, 0.64)
        if self.mode == "lines":
            current = self.line_names[self.line_index]
            active_name = (str(self.current_line["name"])
                           if self.current_line is not None else current)
            if self.click_mode:
                hint = ("click the end point   ·   u cancel"
                        if active_name not in CURVED_LINES
                        else "click points on the arc; f finishes   ·   u cancel")
            elif self.dragging:
                hint = "drag along this white line; release to commit"
            else:
                hint = "drag along this white line, or click to place vertices"
            cv2.putText(img, f"LINE {self.line_index + 1}/{len(self.line_names)}  ·  {current}",
                        (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.78, (90, 255, 120), 1, cv2.LINE_AA)
            cv2.putText(img, hint,
                        (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(img, f"frame {self.frame_idx}  pitch {self.template.length:.0f}x"
                             f"{self.template.width:.0f}  t={self.frame_idx / max(self.fps,1e-9):6.2f}s  "
                             f"lines {len(self.lines)}  clicks {len(self.points)}  anchors {len(self.existing)}"
                             + self._outlier_count_text(),
                        (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)
        else:
            current = RECOMMENDED[self.idx]
            cv2.putText(img, f"POINT {self.idx + 1}/{len(RECOMMENDED)}  ·  {current}",
                        (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.78, (90, 255, 120), 1, cv2.LINE_AA)
            cv2.putText(img, "click this landmark on the frame   ·   n advances without marking",
                        (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(img, f"frame {self.frame_idx}  pitch {self.template.length:.0f}x"
                             f"{self.template.width:.0f}  t={self.frame_idx / max(self.fps,1e-9):6.2f}s  "
                             f"clicks {len(self.points)}  lines {len(self.lines)}  anchors {len(self.existing)}"
                             + self._outlier_count_text(),
                        (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)
        if self.solve_info:
            if "error" in self.solve_info:
                msg = f"SOLVE FAILED: {self.solve_info['error']}"
                color = (80, 80, 255)
            elif "line_median_px" in self.solve_info:
                info = self.solve_info
                accepted = bool(info.get("accepted", True))
                msg = (f"lines: median {info['line_median_px']:.1f}px  "
                       f"hit {info.get('line_hit', 0):.2f}  "
                       f"worst {info.get('worst_line', '?')} "
                       f"{info.get('worst_line_px', 0):.1f}px"
                       + ("" if accepted else "   NOT ACCEPTED"))
                color = (0, 255, 0) if accepted else (0, 200, 255)
            else:
                msg = (f"solve: median {self.solve_info['median_reproj_px']:.1f}px  "
                       f"max {self.solve_info['max_reproj_px']:.1f}px  "
                       f"inliers {self.solve_info['used_points']}/{len(self.points)}")
                color = (0, 255, 0)
            cv2.putText(img, msg, (16, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        if self.message:
            cv2.putText(img, self.message, (16, img.shape[0] - 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
        # Bottom key hints, always visible.
        _text_panel(img, 0, img.shape[0] - 58, img.shape[1], img.shape[0], 0.68)
        if self.mode == "lines":
            cv2.putText(img, "drag trace / click vertices   n/p next line   1-9 jump   f finish   u undo",
                        (14, img.shape[0] - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(img, "s solve   g paint-snap   w save   a save+next   k skip   m point mode   q quit",
                        (14, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "n/p next/prev   1-9 jump   click mark   right/u undo",
                        (14, img.shape[0] - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(img, "s solve   g paint-snap   w save   a save+next   k skip   r reset   q quit",
                        (14, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        # Magnifier at the cursor.
        mx, my = self.cursor
        r = 55
        x0, x1 = max(0, mx - r), min(self.disp_w, mx + r)
        y0, y1 = max(0, my - r), min(self.disp_h, my + r)
        if x1 > x0 and y1 > y0:
            patch = img[y0:y1, x0:x1]
            mag = cv2.resize(patch, (210, 210), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(mag, (98, 98), (112, 112), (0, 255, 0), 1)
            img[-220:-10, -220:-10] = mag
            cv2.rectangle(img, (img.shape[1] - 221, img.shape[0] - 221),
                          (img.shape[1] - 9, img.shape[0] - 9), (0, 255, 0), 1)
        return img

    # ------------------------------------------------------------------ main loop
    def _compose(self) -> np.ndarray:
        """One window: pitch reference panel on the left, video on the right.

        They never overlap, so the panel cannot cover a landmark that is being
        marked.  The canvas is scaled down only as far as needed to fit the
        screen; ``canvas_scale`` maps mouse coordinates back.
        """
        panel = self._pitch_panel()
        video = self._draw()
        h = max(panel.shape[0], video.shape[0])
        canvas = np.full((h, panel.shape[1] + video.shape[1], 3), 18, np.uint8)
        canvas[:panel.shape[0], :panel.shape[1]] = panel
        canvas[:video.shape[0], panel.shape[1]:] = video
        s = min(1.0, WINDOW_MAX_W / canvas.shape[1], WINDOW_MAX_H / canvas.shape[0])
        if s < 1.0:
            size = (max(1, int(round(canvas.shape[1] * s))),
                    max(1, int(round(canvas.shape[0] * s))))
            canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
        self.canvas_scale = canvas.shape[1] / (panel.shape[1] + video.shape[1])
        return canvas

    def run(self) -> None:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_TITLE, self._on_mouse)
        if self.mode == "lines":
            print("Mode: lines.  Drag along a white pitch line to trace it.")
            print("Lines:")
            for i, name in enumerate(self.line_names):
                print(f"  {i+1:2d}. {name}")
        else:
            print("Mode: points.  Click the named landmarks.")
            print("Landmarks:")
            for i, name in enumerate(RECOMMENDED):
                print(f"  {i+1:2d}. {name}")
        print("Keys: n/p next/prev, 1-9 jump, u undo, s solve, w save, "
              "a save+next, k skip step, m toggle mode, ,/. frame, r reset, q quit")
        print("Clicks and traced lines are both kept and both used by s; "
              "unexplained clicks/lines are shown in red after a solve.")
        while True:
            cv2.imshow(WINDOW_TITLE, self._compose())
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                self.toggle_mode()
            elif key == ord("n"):
                if self.mode == "lines":
                    self.select_line(self.line_index + 1)
                else:
                    self.select(self.idx + 1)
            elif key == ord("p"):
                if self.mode == "lines":
                    self.select_line(self.line_index - 1)
                else:
                    self.select(self.idx - 1)
            elif ord("1") <= key <= ord("9"):
                if self.mode == "lines":
                    self.select_line(key - ord("1"))
                else:
                    self.select(key - ord("1"))
            elif key in (ord("f"), 13):
                if self.mode == "lines":
                    self.finish_line()
            elif key == ord("u"):
                if self.mode == "lines":
                    self.undo_line()
                else:
                    self.undo()
            elif key == ord("g"):
                self.snap()
            elif key == ord("s"):
                self.solve()
                if (self.solve_info and "error" not in self.solve_info
                        and "line_median_px" not in self.solve_info):
                    print(f"solve: median {self.solve_info['median_reproj_px']:.2f} px, "
                          f"max {self.solve_info['max_reproj_px']:.2f} px, "
                          f"inliers {self.solve_info['used_points']}/{len(self.points)}")
            elif key == ord("w"):
                self.save()
            elif key == ord("a"):
                self.save_and_next()
            elif key == ord("k"):
                self.skip_step()
            elif key == ord("."):
                self.jump_frame(+1)
            elif key == ord(","):
                self.jump_frame(-1)
            elif key == ord("r"):
                self.cancel_line()
                self.points.clear()
                self.lines.clear()
                self.H = None
                self.solve_info = None
                self._clear_outliers()
                self.line_index = 0
                self.message = "cleared"
        cv2.destroyAllWindows()

    def _on_mouse(self, event, x, y, flags, param):
        s = self.canvas_scale if self.canvas_scale > 0 else 1.0
        vx = int(round(x / s)) - self.panel_w
        vy = int(round(y / s))
        over_video = 0 <= vx < self.disp_w and 0 <= vy < self.disp_h
        if over_video:
            self.cursor = (vx, vy)
        if event == cv2.EVENT_LBUTTONDOWN:
            if over_video:
                if self.mode == "lines":
                    self.begin_line(vx, vy)
                else:
                    self.add_click(vx, vy)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mode == "lines":
                if self.dragging and (flags & cv2.EVENT_FLAG_LBUTTON):
                    self.extend_line(vx, vy)
                elif self.dragging:
                    # LBUTTONUP can be delivered outside the OpenCV window (or
                    # lost entirely).  Finish the drag as soon as a move arrives
                    # with the button up, so the preview cannot stay cyan.
                    self.end_line()
        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode == "lines":
                self.end_line()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == "lines":
                self.undo_line()
            elif over_video:
                self.undo()


def run_annotator(video: str, frame: int, out: str, pitch: Tuple[float, float],
                  append: bool = False, step_sec: float = 2.0,
                  mode: str = "points") -> None:
    ClickAnnotator(video, frame, out, pitch, append=append, step_sec=step_sec,
                   mode=mode).run()
