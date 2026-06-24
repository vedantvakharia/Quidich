"""CLI entry point for Phase 2 per-camera tracking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_estimation.cricket.p2_config import load_p2_config  # noqa: E402
from pose_estimation.cricket.p2_runner import run_p2_tracking  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2 per-camera tracking")
    parser.add_argument("--input-dir", required=True, help="dir of P1 cam_XX.jsonl predictions")
    parser.add_argument("--output-dir", required=True, help="dir for P2 cam_XX.jsonl outputs")
    parser.add_argument("--delivery-id", required=True)
    parser.add_argument("--config", default=None, help="optional p2_tracking.yaml")
    parser.add_argument("--camera", action="append", default=None, help="restrict to camera(s)")
    parser.add_argument("--expected-frames", type=int, default=600)
    parser.add_argument("--max-workers", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_p2_config(args.config)
    results = run_p2_tracking(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        delivery_id=args.delivery_id,
        config=config,
        cameras=args.camera,
        expected_frames=args.expected_frames,
        max_workers=args.max_workers,
    )
    failed = []
    for cam in sorted(results):
        status, summary, error = results[cam]
        if status == "ok":
            print(f"{cam}: ok  frames={summary.get('frames_read')} "
                  f"confirmed={summary.get('confirmed_tracks')}", flush=True)
        else:
            failed.append(cam)
            print(f"{cam}: FAILED  {error}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
