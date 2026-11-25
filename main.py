from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
from tqdm import tqdm
import torch

from core.anchor_bootstrap import homography_from_camera, load_manual_bootstrap
from core.broadtrack_calib import BroadTrackCalib, PitchGeometry
from core.log import (
    clear_frame_context,
    configure_logging,
    get_logger,
    parse_module_levels,
    set_frame_context,
)
from core.pitch_tracker import PitchHomographyTracker, PitchTrackConfig
from core.pitchcal.geometry import project_polyline_segments
from core.player_tracker import PlayerTracker
from core.player_position import PlayerPositionFilter
from core.ball_tracker import OCCLUDED, BallDetector, BallTracker, BallTrackerConfig
from core.team_assigner import TeamAssigner

logger = get_logger(__name__)

BALL_MODEL_PATH = "models/yolo-sn-ball.pt"
PLAYER_MODEL_PATH = "models/yolo11n-seg.pt"

# BroadTrack's TorchScript detectors. See core/BroadTrack/README.md;
BROADTRACK_KP_WEIGHTS = "models/nbjw_keypoint_model.pt"
BROADTRACK_LINE_WEIGHTS = "models/tvcalib_model.pt"

DEFAULT_PITCH = (105.0, 68.0)
PENALTY_SPOT_FROM_GOAL_LINE = 11.0


def parse_pitch(text: str) -> tuple[float, float]:
    """'90x64' / '90×64' / '90,64' -> (90.0, 64.0)."""
    cleaned = text.lower().replace("×", "x").replace(",", "x").replace(" ", "")
    parts = cleaned.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("pitch must look like 105x68")
    return float(parts[0]), float(parts[1])


def static_marks(pitch_length: float) -> tuple:
    """Penalty spots and the centre mark, in world metres (z-down frame)."""
    spot = pitch_length / 2.0 - PENALTY_SPOT_FROM_GOAL_LINE
    return ((0.0, 0.0), (spot, 0.0), (-spot, 0.0))


def draw_homography_lines(img, homography, wireframe, color=(0, 255, 0), thickness=2):
    """Project the pitch wireframe onto ``img`` through a ground homography."""
    height, width = img.shape[:2]
    homography = homography.reshape(3, 3).astype(float)
    for polyline in wireframe:
        points = polyline.reshape(-1, 3)[:, :2]
        for start, end in project_polyline_segments(homography, points, width, height):
            cv2.line(
                img,
                (int(round(start[0])), int(round(start[1]))),
                (int(round(end[0])), int(round(end[1]))),
                color, thickness,
            )
    return img


def all_finite(*arrays) -> bool:
    """Whether every array is present and finite (used before seeding a camera)."""
    for array in arrays:
        if array is None or not np.isfinite(array).all():
            return False
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="input_vids/test1.mp4")
    parser.add_argument("--max-frames", type=int, default=0, help="0 for the whole video")
    parser.add_argument("--start-frame", type=int, default=0, help="first frame to process")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--no-display", action="store_true", help="run headless")
    parser.add_argument(
        "--anchor", default=None,
        help="manual anchor JSON from `python -m core.pitchcal.cli annotate`; "
             "seeds the calibration from a human-marked frame",
    )
    parser.add_argument(
        "--anchor-free-camera", action="store_true",
        help="do not pin the camera centre when BroadTrack is used as a "
             "keyframe; default is a fixed PTZ camera",
    )
    parser.add_argument(
        "--pitch", default=None,
        help="pitch size LENGTHxWIDTH in metres (default 105x68, or the size "
             "stored in --anchor)",
    )
    parser.add_argument(
        "--calib-backend", choices=["pitchcal", "broadtrack"], default="pitchcal",
        help="pitchcal: LK tracking + sparse white-line evidence (default); "
             "broadtrack: semantic line-segmentation model + Ceres every frame",
    )
    parser.add_argument(
        "--broadtrack-every", type=int, default=0, metavar="N",
        help="with --calib-backend pitchcal, run BroadTrack as an absolute "
             "keyframe every N frames (0 = only anchors/bootstrap)",
    )
    parser.add_argument(
        "--line-every", type=int, default=4,
        help="frames between pitchcal white-line evidence measurements",
    )
    parser.add_argument(
        "--line-correct-every", type=int, default=20,
        help="frames between gated homography corrections (0 = off)",
    )
    parser.add_argument(
        "--player-interval", type=float, default=0.15,
        help="seconds between YOLO player detections (default 0.15)",
    )

    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="raise the log level: -v = DEBUG, -vv = TRACE",
    )
    logging_group.add_argument(
        "--log-level", default=None,
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="global log level (default INFO, or FOOTBALL_LOG_LEVEL)",
    )
    logging_group.add_argument(
        "--log-module", action="append", default=[], metavar="NAME=LEVEL",
        help="override one module's level, e.g. ball_tracker=DEBUG (repeatable)",
    )
    logging_group.add_argument(
        "--log-file", default=None,
        help="also write to this file (DEBUG by default, or FOOTBALL_LOG_FILE)",
    )
    logging_group.add_argument(
        "--log-file-level", default=None,
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="file handler level (default DEBUG, or FOOTBALL_LOG_FILE_LEVEL)",
    )
    logging_group.add_argument(
        "--no-log-color", action="store_true", help="disable ANSI colours in the console log",
    )
    return parser.parse_args()


def configure_logging_from_args(args):
    if args.log_level is not None:
        level = args.log_level
    elif args.verbose >= 2:
        level = "TRACE"
    elif args.verbose == 1:
        level = "DEBUG"
    else:
        level = None  # configure_logging falls back to the environment / INFO

    configure_logging(
        level=level,
        log_file=args.log_file,
        file_level=args.log_file_level,
        modules=parse_module_levels(args.log_module) if args.log_module else None,
        color=False if args.no_log_color else None,
    )


def main():
    args = parse_args()
    configure_logging_from_args(args)
    display = not args.no_display and bool(os.environ.get("DISPLAY"))

    # VIDEO PROCESSING
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, min(int(args.start_frame), max(0, total_frames - 1)))
    end_frame = total_frames
    if args.max_frames:
        end_frame = min(total_frames, start_frame + args.max_frames)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    limit = max(0, end_frame - start_frame)

    # A manual anchor also carries the pitch size. --pitch overrides it; with
    # no anchor, --pitch (or the 105x68 default) is used directly.
    pitch_override = parse_pitch(args.pitch) if args.pitch else None
    bootstrap = None
    if args.anchor:
        bootstrap = load_manual_bootstrap(args.anchor, width, height, pitch=pitch_override)
        pitch_length, pitch_width = bootstrap.pitch
    else:
        pitch_length, pitch_width = pitch_override or DEFAULT_PITCH

    player_interval = max(1.0 / fps, float(args.player_interval))
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(
        "video=%s frames=%d..%d/%d %dx%d @ %.2f fps, pitch=%.0fx%.0f m, "
        "calib=%s, players every %.2fs, device=%s, display=%s",
        args.video, start_frame, end_frame, total_frames, width, height, fps,
        pitch_length, pitch_width, args.calib_backend, player_interval,
        device, "on" if display else "off",
    )
    if bootstrap is not None:
        logger.info(
            "manual bootstrap: %s (%d anchor frame(s): %s)",
            bootstrap.path, len(bootstrap.anchors),
            ", ".join(str(a.frame) for a in bootstrap.anchors),
        )

    # CORE init
    geometry = PitchGeometry(pitch_length, pitch_width)
    bev_template = geometry.create_bev_template()
    bev_canvas_h, bev_canvas_w = bev_template.shape[:2]

    ball_detector = BallDetector(BALL_MODEL_PATH, device)
    ball_tracker = BallTracker(
        fps=fps,
        config=BallTrackerConfig(pitch_length=pitch_length, pitch_width=pitch_width),
        static_marks=static_marks(pitch_length),
    )
    player_tracker = PlayerTracker(PLAYER_MODEL_PATH, device)
    player_positions = PlayerPositionFilter(pitch_length=pitch_length, pitch_width=pitch_width)
    team_assigner = TeamAssigner(pitch_length=pitch_length)

    # BroadTrack is loaded lazily: with an anchor and no periodic keyframes the
    # pitchcal backend never needs its detectors at all.
    calib_engine = None

    def broadtrack() -> BroadTrackCalib:
        nonlocal calib_engine
        if calib_engine is None:
            calib_engine = BroadTrackCalib(
                weights_kp=BROADTRACK_KP_WEIGHTS,
                weights_line=BROADTRACK_LINE_WEIGHTS,
                device=device,
                width=width,
                height=height,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
            )
        return calib_engine

    fix_position = not args.anchor_free_camera
    pitch_config = PitchTrackConfig(
        line_every=max(0, int(args.line_every)),
        correct_every=max(0, int(args.line_correct_every)),
        sift_keyframe_every=max(2, int(round(fps * 0.5))),
    )
    pitch_tracker = None

    def apply_anchor(anchor, current_frame: int) -> None:
        nonlocal pitch_tracker
        homography = homography_from_camera(anchor.K, anchor.R, anchor.t)
        principal = (float(anchor.K[0, 2]), float(anchor.K[1, 2]))
        focal = float(anchor.K[0, 0])
        if args.calib_backend == "pitchcal":
            if pitch_tracker is None:
                pitch_tracker = PitchHomographyTracker(
                    homography, width, height, (pitch_length, pitch_width),
                    principal, focal, pitch_config,
                )
            else:
                pitch_tracker.reset(homography, principal, focal)
        if args.calib_backend == "broadtrack" or args.broadtrack_every > 0:
            broadtrack().set_camera(
                anchor.K, anchor.R, anchor.t, anchor.k1,
                frame_width=anchor.width, frame_height=anchor.height,
                fix_position=fix_position,
            )
        logger.info(
            "manual anchor from frame %d applied at frame %d "
            "(f=%.1f, principal %.1f,%.1f, source=%s, score=%.2f)",
            anchor.frame, current_frame, focal, principal[0], principal[1],
            anchor.source, anchor.score,
        )

    pending_anchors = []
    if bootstrap is not None:
        anchor = bootstrap.at_or_before(start_frame)
        if anchor is None:
            logger.info(
                "first manual anchor is at frame %d, after the start; BroadTrack "
                "bootstraps itself until then", bootstrap.anchors[0].frame,
            )
        else:
            if anchor.frame < start_frame:
                logger.warning(
                    "starting %d frame(s) after the latest anchor (frame %d); "
                    "the camera may have moved since it was marked",
                    start_frame - anchor.frame, anchor.frame,
                )
            apply_anchor(anchor, start_frame)
        pending_anchors = [a for a in bootstrap.anchors if a.frame > start_frame]

    if args.calib_backend == "pitchcal":
        logger.info(
            "pitchcal calibration: line evidence every %d frame(s), correction "
            "every %s, BroadTrack keyframe every %s",
            args.line_every,
            args.line_correct_every or "never",
            args.broadtrack_every or "never",
        )
    elif args.broadtrack_every:
        logger.warning("--broadtrack-every is ignored with --calib-backend broadtrack")

    # Video Output
    os.makedirs(args.out_dir, exist_ok=True)
    logger.debug("writing cam.mp4 (%dx%d) and bev.mp4 (%dx%d) to %s",
                 width, height, bev_canvas_w, bev_canvas_h, args.out_dir)
    out_cam = cv2.VideoWriter(os.path.join(args.out_dir, "cam.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    out_bev = cv2.VideoWriter(os.path.join(args.out_dir, "bev.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (bev_canvas_w, bev_canvas_h))

    if display:
        cv2.namedWindow("cam", cv2.WINDOW_NORMAL)
        cv2.namedWindow("bev", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cam", width, height)
        cv2.resizeWindow("bev", bev_canvas_w, bev_canvas_h)
        cv2.moveWindow("cam", 50, 50)
        cv2.moveWindow("bev", 50 + width + 20, 50)

    calib_was_valid = True
    processed = 0
    last_player_dets = []
    last_player_boxes = []
    last_on_pitch = {}
    last_homography = None
    last_K = last_R = last_t = None
    player_accumulator = player_interval  # detect on the first frame
    started = time.perf_counter()

    try:
        for i in tqdm(range(limit)):
            ret, frame = cap.read()
            if not ret:
                logger.error("failed to read frame %d", start_frame + i)
                break
            frame_idx = start_frame + i
            frame = cv2.resize(frame, (width, height))
            set_frame_context(frame_idx)
            cam_canva = frame.copy()
            bev_canva = bev_template.copy()

            # A new manual anchor replaces the tracker state at its exact frame.
            anchor_applied = False
            while pending_anchors and pending_anchors[0].frame <= frame_idx:
                anchor = pending_anchors.pop(0)
                apply_anchor(anchor, frame_idx)
                anchor_applied = True

            # Player detection is sparse; skipped frames reuse the last boxes.
            # BroadTrack's flow rejection and pitchcal's feature/line masks only
            # need approximate player regions, so a 0.15 s cache is fine.
            player_accumulator += 1.0 / fps
            player_fresh = player_accumulator >= player_interval
            if player_fresh:
                player_accumulator -= player_interval
                player_dets = player_tracker.update(frame)
                player_boxes = [d["bbox"] for d in player_dets]
                last_player_dets = player_dets
                last_player_boxes = player_boxes
            else:
                player_dets = last_player_dets
                player_boxes = last_player_boxes

            # ------------------------------------------------------ calibration
            homography = None
            K = R = t = None
            calib_valid = False
            calib_score = 0.0
            calib_reinit = False
            calib_frames_lost = 0
            line_hit = 0.0

            if args.calib_backend == "broadtrack":
                engine = broadtrack()
                new_calib = engine.estimate(frame, boxes=player_boxes)
                if new_calib is not None:
                    K, R, t = new_calib["K"], new_calib["R"], new_calib["t"]
                    homography = homography_from_camera(K, R, t)
                    calib_valid = True
                    calib_score = float(new_calib["score"])
                    calib_reinit = bool(new_calib["reinit"])
                    calib_frames_lost = int(new_calib["frames_lost"])
            else:
                if pitch_tracker is None:
                    # No anchor yet: use BroadTrack once to get an absolute
                    # camera, then carry it with LK.
                    new_calib = broadtrack().estimate(frame, boxes=player_boxes)
                    if new_calib is not None:
                        K, R, t = new_calib["K"], new_calib["R"], new_calib["t"]
                        homography = homography_from_camera(K, R, t)
                        principal = (float(K[0, 2]), float(K[1, 2]))
                        pitch_tracker = PitchHomographyTracker(
                            homography, width, height, (pitch_length, pitch_width),
                            principal, float(K[0, 0]), pitch_config,
                        )
                        calib_valid = True
                        calib_score = float(new_calib["score"])
                        calib_reinit = bool(new_calib["reinit"])
                        calib_frames_lost = int(new_calib["frames_lost"])
                        logger.info("pitchcal tracker seeded from BroadTrack at frame %d", frame_idx)
                else:
                    tracked = pitch_tracker.update(frame, boxes=player_boxes)
                    homography = tracked.homography
                    K, R, t = tracked.K, tracked.R, tracked.t
                    calib_valid = bool(tracked.valid and not tracked.cut)
                    calib_score = float(tracked.confidence)
                    calib_reinit = bool(tracked.cut)
                    calib_frames_lost = int(tracked.lost_frames)
                    line_hit = tracked.line_hit
                    # Optional absolute keyframe. It is seeded from the tracked
                    # camera, then becomes the new reference for the LK tracker.
                    if (args.broadtrack_every > 0 and i > 0 and not anchor_applied
                            and i % args.broadtrack_every == 0):
                        engine = broadtrack()
                        if all_finite(K, R, t):
                            engine.set_camera(
                                K, R, t, 0.0,
                                frame_width=width, frame_height=height,
                                fix_position=fix_position,
                            )
                        keyframe = engine.estimate(frame, boxes=player_boxes)
                        if keyframe is not None:
                            K, R, t = keyframe["K"], keyframe["R"], keyframe["t"]
                            homography = homography_from_camera(K, R, t)
                            principal = (float(K[0, 2]), float(K[1, 2]))
                            pitch_tracker.reset(homography, principal, float(K[0, 0]))
                            calib_valid = True
                            calib_score = float(keyframe["score"])
                            calib_reinit = bool(keyframe["reinit"])
                            calib_frames_lost = int(keyframe["frames_lost"])
                            line_hit = 0.0
                            logger.info(
                                "BroadTrack keyframe at frame %d (score %.2f, f=%.1f)",
                                frame_idx, calib_score, float(K[0, 0]),
                            )

            if homography is not None:
                last_homography, last_K, last_R, last_t = homography, K, R, t
            elif last_homography is not None:
                homography, K, R, t = last_homography, last_K, last_R, last_t

            if calib_valid and not calib_was_valid:
                logger.info("calibration recovered at frame %d", frame_idx)
            elif not calib_valid and calib_was_valid:
                logger.warning(
                    "calibration lost at frame %d; %s", frame_idx,
                    "using the last tracked homography" if homography is not None
                    else "skipping projections until it recovers",
                )
            calib_was_valid = calib_valid

            # ------------------------------------------------------- projection
            if homography is not None:
                if player_fresh:
                    on_pitch_pos, _ = player_positions.update(player_dets, H=homography)
                    team_assigner.update(frame, player_dets, on_pitch_pos)
                    last_on_pitch = on_pitch_pos
                else:
                    on_pitch_pos = last_on_pitch
            else:
                on_pitch_pos = {}

            for tid, (x, y) in on_pitch_pos.items():
                px, py = geometry.world_to_bev_px(x, y)
                cv2.circle(bev_canva, (px, py), 4, team_assigner.draw_colour(tid), -1)

            ball_dets = ball_detector.detect(frame) if homography is not None else []
            if homography is not None:
                has_camera = K is not None and R is not None and t is not None
                ball_candidates = ball_detector.get_candidates(
                    ball_dets, H=homography,
                    K=K if has_camera else None,
                    R=R if has_camera else None,
                    t=t if has_camera else None,
                )
            else:
                ball_candidates = []
            prediction = ball_tracker.process_frame(
                ball_candidates,
                player_boxes=player_boxes,
                frame_idx=frame_idx,
                calib_score=calib_score,
                calib_reinit=calib_reinit,
                calib_frames_lost=calib_frames_lost,
                calib_valid=calib_valid,
            )
            cv2.putText(
                bev_canva,
                f"{ball_tracker.state} m{ball_tracker.miss_count} r{ball_tracker.reject_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
            )
            if prediction is not None:
                px, py = geometry.world_to_bev_px(prediction[0], prediction[1])
                if ball_tracker.state == OCCLUDED:
                    cv2.circle(bev_canva, (px, py), 5, (0, 255, 255), 2)  # hollow
                else:
                    cv2.circle(bev_canva, (px, py), 5, (0, 255, 255), -1)  # yellow

            logger.debug(
                "frame: %d players on pitch, %d ball candidate(s); ball state=%s "
                "pred=%s; calib valid=%s score=%.2f hit=%.2f",
                len(on_pitch_pos), len(ball_candidates), ball_tracker.state,
                "-" if prediction is None else f"({prediction[0]:.2f}, {prediction[1]:.2f})",
                calib_valid, calib_score, line_hit,
            )
            processed += 1

            # Camera Plane
            if homography is not None:
                draw_homography_lines(cam_canva, homography, geometry.wireframe,
                                      color=(0, 255, 0), thickness=2)
            if player_fresh:
                player_tracker.draw_tracks(cam_canva, player_dets)
            if ball_dets and ball_dets[0].boxes is not None:
                ball_boxes = ball_dets[0].boxes.xyxy.cpu().numpy()
                for box in ball_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(cam_canva, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red

            # Calibration quality: line-IoU (BroadTrack) or tracking confidence
            # combined with the white-paint line hit (pitchcal).  The first
            # frame after an anchor has no frame-to-frame measurement yet, so
            # show its line hit instead of a misleading 0.00.
            if args.calib_backend == "pitchcal":
                quality = (f"calib {calib_score:.2f}" if calib_score > 0.0
                           else "anchor")
                if line_hit > 0.0:
                    quality += f" hit {line_hit:.2f}"
            else:
                quality = f"calib {calib_score:.2f}"
            cv2.putText(cam_canva, quality, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            out_cam.write(cam_canva)
            out_bev.write(bev_canva)

            if display:
                cv2.imshow("cam", cam_canva)
                cv2.imshow("bev", bev_canva)
            delay = max(1, int(1000 / fps))
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
    finally:
        clear_frame_context()
        out_cam.release()
        out_bev.release()
        cap.release()
        if display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    logger.info(
        "processed %d frame(s) in %.1fs (%.2f fps)",
        processed, elapsed, processed / elapsed if elapsed > 0 else 0.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
