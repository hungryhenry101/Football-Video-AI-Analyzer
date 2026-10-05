"""Team assignment over a real video, with the colour statistics behind it.

From the repo root:

    python demos/team_assigner_demo.py
    python demos/team_assigner_demo.py --video input_vids/test1.mp4 --max-frames 200
    python demos/team_assigner_demo.py --save          # write output/*.mp4, no windows
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import cv2
import numpy as np
import torch

from core.broadtrack_calib import BroadTrackCalib
from core.player_tracker import PlayerTracker
from core.projection_utils import pixel_to_ground
from core.team_assigner import TeamAssigner, lab_to_bgr

PLAYER_MODEL = "models/yolo11m-seg.pt"
KEYPOINT_WEIGHTS = "models/nbjw_keypoint_model.pt"
LINE_WEIGHTS = "models/tvcalib_model.pt"

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 300


def swatch(lab, size=44):
    """A square of the BGR that a mean Lab colour corresponds to."""
    return np.full((size, size, 3), lab_to_bgr(lab), np.uint8)


def team_means(assigner):
    """{team label: mean Lab of the players currently on that team}."""
    grouped = {}
    for tid, assignment in assigner.assignments.items():
        if assignment["team"] is not None:
            grouped.setdefault(assignment["team"], []).append(assigner.colours[tid])
    return {team: np.mean(cols, axis=0) for team, cols in grouped.items()}


def draw_cam(frame, dets, assigner):
    for det in dets:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        colour = assigner.draw_colour(det["id"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Bottom-centre is the ground contact point, as on the BEV pane.
        cv2.circle(frame, ((x1 + x2) // 2, y2), 4, colour, -1)

        assignment = assigner.assignments.get(det["id"])
        tag = ""
        if assignment is not None:
            role = assignment["role"]
            tag = "" if role == "player" else f" {role[:3].upper()}"
            if assignment["team"] is not None:
                tag = f" T{assignment['team']}{tag}"
        cv2.putText(frame, f"#{det['id']}{tag}", (x1, max(14, y1 - 6)), FONT, 0.55, colour, 2)
    return frame


def draw_panel(assigner, observations, frame_idx, height):
    panel = np.full((height, PANEL_W, 3), 24, np.uint8)
    cv2.putText(panel, f"frame {frame_idx}", (12, 26), FONT, 0.6, (235, 235, 235), 1)

    y = 46
    for team, mean in sorted(team_means(assigner).items()):
        panel[y : y + 44, 12:56] = swatch(mean)
        count = sum(1 for a in assigner.assignments.values() if a["team"] == team)
        cv2.putText(panel, f"TEAM_{'AB'[team]}  n={count}", (66, y + 28), FONT, 0.5, (235, 235, 235), 1)
        cv2.putText(panel, f"L{mean[0]:.0f} a{mean[1]:.0f} b{mean[2]:.0f}", (66, y + 44), FONT, 0.4, (150, 150, 150), 1)
        y += 60

    y += 6
    cv2.putText(panel, "player  obs   sd", (12, y), FONT, 0.45, (170, 170, 170), 1)
    y += 12
    for tid in sorted(assigner.colours):
        if y > height - 20:
            cv2.putText(panel, "...", (12, y), FONT, 0.5, (150, 150, 150), 1)
            break
        samples = observations.get(tid, [])
        sd = float(np.std(np.array(samples, dtype=np.float32))) if len(samples) > 1 else 0.0
        cv2.rectangle(panel, (12, y + 2), (26, y + 14), assigner.draw_colour(tid), -1)
        cv2.putText(panel, f"#{tid:<4d} {len(samples):>4d}  {sd:5.1f}", (34, y + 13), FONT, 0.42, (225, 225, 225), 1)
        y += 19
    return panel


def report(assigner, observations):
    print("\nper-player colour stability")
    print(f"{'id':>5} {'obs':>5} {'mean L':>8} {'mean a':>8} {'mean b':>8} {'per-frame sd':>13}  team")
    print("-" * 66)
    for tid in sorted(assigner.colours):
        samples = np.array(observations.get(tid, []), dtype=np.float32)
        sd = float(np.std(samples)) if len(samples) > 1 else 0.0
        assignment = assigner.assignments.get(tid, {})
        team = assignment.get("team")
        team = f"TEAM_{'AB'[team]}" if team is not None else assignment.get("role", "?").upper()
        mean = assigner.colours[tid]
        print(f"{tid:>5} {len(samples):>5} {mean[0]:>8.1f} {mean[1]:>8.1f} "
              f"{mean[2]:>8.1f} {sd:>13.1f}  {team}")

    means = team_means(assigner)
    sds = [float(np.std(np.array(observations[t], dtype=np.float32)))
           for t in observations if len(observations[t]) > 1]
    if len(means) < 2 or not sds:
        print("\nnot enough of a sample to compare against a team separation")
        return

    centres = np.stack(list(means.values()))
    separation = float(np.linalg.norm(centres[0] - centres[1]))
    median_sd = float(np.median(sds))
    median_obs = int(np.median([len(observations[t]) for t in observations]))
    print(f"\nbetween-team distance      {separation:6.1f} Lab units")
    print(f"median per-frame sd        {median_sd:6.1f} Lab units")
    print(f"after {median_obs} frames of averaging {median_sd / np.sqrt(max(median_obs, 1)):6.1f}"
          "   (noise falls ~1/sqrt(N))")
    if median_sd > separation:
        print("-> raw frames overlap; clustering them one by one would be a coin toss.")



def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", default="input_vids/test2.mp4")
    parser.add_argument("--max-frames", type=int, default=300, help="0 for the whole video")
    parser.add_argument("--save", action="store_true", help="write output/*.mp4 instead of showing windows")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"could not open {args.video}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)   # as main.py does
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(args.max_frames, total) if args.max_frames else total

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"{args.video}: {total} frames at {width}x{height}, {limit} to process, on {device}")

    calib_engine = BroadTrackCalib(
        weights_kp=KEYPOINT_WEIGHTS, weights_line=LINE_WEIGHTS,
        device=device, width=width, height=height,
    )
    player_tracker = PlayerTracker(PLAYER_MODEL, device)
    assigner = TeamAssigner()

    bev_template = calib_engine.create_bev_template()
    writer_cam = writer_bev = None
    if args.save:
        os.makedirs("output", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_cam = cv2.VideoWriter("output/team_cam.mp4", fourcc, fps, (width + PANEL_W, height))
        writer_bev = cv2.VideoWriter("output/team_bev.mp4", fourcc, fps,
                                     (bev_template.shape[1], bev_template.shape[0]))

    observations: dict[int, list] = {}
    for idx in range(limit):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height))

        dets = player_tracker.update(frame)
        calib = calib_engine.estimate(frame, boxes=[d["bbox"] for d in dets])
        K, R, t = (calib["K"], calib["R"], calib["t"]) if calib is not None else (None, None, None)

        # Called for the statistics table only: update() computes the same
        # estimate internally, so this is a deliberate 2x on a step that is
        # cheap next to the detector and the solver. The alternative is reading
        # TeamAssigner's private accumulator, which the demo has no business doing.
        for det in dets:
            sample = assigner.identify_jersey_colour(det, frame)
            if sample is not None:
                observations.setdefault(det["id"], []).append(sample)

        assigner.update(frame, dets, K, R, t)

        cam = draw_cam(frame, dets, assigner)
        panel = draw_panel(assigner, observations, idx, cam.shape[0])
        combined = np.hstack([cam, panel])

        bev = bev_template.copy()
        if K is not None:
            for det in dets:
                x1, _, x2, y2 = det["bbox"]
                point = pixel_to_ground((x1 + x2) / 2, y2, K, R, t)
                if point is not None:
                    px, py = calib_engine.world_to_bev_px(point[0], point[1])
                    cv2.circle(bev, (px, py), 5, assigner.draw_colour(det["id"]), -1)

        if writer_cam is not None:
            writer_cam.write(combined)
            writer_bev.write(bev)
        cv2.imshow("cam", combined)
        cv2.imshow("bev", bev)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    for writer in (writer_cam, writer_bev):
        if writer is not None:
            writer.release()
    cv2.destroyAllWindows()
    if args.save:
        print("wrote output/team_cam.mp4 and output/team_bev.mp4")

    report(assigner, observations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
