"""Annotator entry point for the manual calibration bootstrap.

Only the interactive anchor marking is exposed here.  The original pitchcal
project also ships a standalone LK/SIFT tracker, BEV renderer and HTML report;
this repository does not use those, because BroadTrack is the calibration
backend and ``main.py`` renders the BEV.

    python -m core.pitchcal.cli annotate \
        --video input_vids/test2.mp4 --frame 0 \
        --out config/anchors/test2.json --pitch 105x68

Then run the analyzer with the anchor:

    python main.py --video input_vids/test2.mp4 --anchor config/anchors/test2.json
"""

from __future__ import annotations

import argparse

from .anchors import load_anchor_file


def _parse_pitch(text: str) -> tuple[float, float]:
    cleaned = text.lower().replace("×", "x").replace(",", "x").replace(" ", "")
    parts = cleaned.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("pitch must look like 105x68")
    return float(parts[0]), float(parts[1])


def cmd_annotate(args) -> None:
    from .annotate import run_annotator

    pitch = _parse_pitch(args.pitch)
    if args.append and args.out:
        # Keep the pitch already recorded in the file when appending.
        try:
            pitch = load_anchor_file(args.out).pitch
        except (OSError, ValueError):
            pass
    run_annotator(
        args.video,
        int(args.frame),
        args.out,
        pitch,
        append=args.append,
        step_sec=args.step,
        mode=args.mode,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m core.pitchcal.cli",
        description="Manually mark a pitch anchor for BroadTrack calibration.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    annotate = sub.add_parser(
        "annotate",
        help="click landmarks or trace white lines on one frame",
    )
    annotate.add_argument("--video", required=True)
    annotate.add_argument("--frame", type=int, required=True)
    annotate.add_argument("--out", required=True, help="anchor JSON to write")
    annotate.add_argument("--pitch", default="105x68")
    annotate.add_argument(
        "--append", action="store_true",
        help="add to an existing anchor file instead of replacing it",
    )
    annotate.add_argument(
        "--step", type=float, default=2.0,
        help="seconds advanced by 'a' (save and next anchor)",
    )
    annotate.add_argument(
        "--mode", choices=["points", "lines"], default="points",
        help="start in point-click or white-line tracing mode (toggle with 'm')",
    )
    annotate.set_defaults(func=cmd_annotate)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
