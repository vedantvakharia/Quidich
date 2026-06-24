#!/usr/bin/env python3
"""Run Phase 1 person detection + 2D pose inference on DS-001 cricket frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_estimation.cricket.p1_runner import P1RunConfig, run_phase1_delivery
from pose_estimation.cricket.p1_yolo import YOLOPoseAdapter, load_model_config


DEFAULT_MODEL_CONFIG = ROOT / "configs" / "model_envs.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", default="drive")
    parser.add_argument("--delivery-id", required=True)
    parser.add_argument("--model-id", default="yolo26x_pose")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution if CUDA is unavailable")
    parser.add_argument("--inference-mode", choices=["full_frame", "crops"], default="full_frame")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--cameras", nargs="+", default=None, help="Optional camera ids, e.g. cam_01 cam_07")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--no-half", dest="half", action="store_false", help="Disable FP16 inference on CUDA")
    parser.set_defaults(half=True)
    parser.add_argument(
        "--no-preload-full-frame",
        dest="preload_full_frame",
        action="store_false",
        help="Disable camera-level decode/resize preload for full-frame inference",
    )
    parser.set_defaults(preload_full_frame=True)
    parser.add_argument(
        "--resize-long-side",
        type=int,
        default=None,
        help="Resize decoded full frames to this long side before inference; defaults to --imgsz",
    )
    parser.add_argument("--no-progress", dest="show_progress", action="store_false")
    parser.set_defaults(show_progress=True)
    return parser.parse_args()


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(device: str, *, allow_cpu: bool) -> str:
    if device != "auto":
        if device == "cpu" and not allow_cpu:
            raise SystemExit("CPU execution requested without --allow-cpu")
        if device.startswith("cuda") and not cuda_available():
            raise SystemExit("CUDA device requested but torch.cuda.is_available() is false in this env")
        return device
    if cuda_available():
        return "cuda:0"
    if not allow_cpu:
        raise SystemExit("CUDA is unavailable in this env. Re-run with --allow-cpu only for debugging.")
    return "cpu"


def resolve_model_paths(model_config: dict) -> dict:
    resolved = dict(model_config)
    for key in ["model_name", "checkpoint"]:
        value = resolved.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            resolved[key] = str((ROOT / path).resolve())
    return resolved


def main() -> int:
    args = parse_args()
    if args.model_id != "yolo26x_pose":
        raise SystemExit(
            "run_cricket_p1_inference.py currently supports yolo26x_pose for full delivery inference. "
            "Use smoke/shortlist tooling for other model candidates."
        )
    drive_root = (ROOT / args.drive_root).resolve() if not Path(args.drive_root).is_absolute() else Path(args.drive_root)
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else ROOT / "benchmarks" / "runs" / args.run_id
    )
    model_config = resolve_model_paths(load_model_config(args.model_id, args.model_config))
    device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    half = args.half and device != "cpu"
    resize_long_side = args.resize_long_side if args.resize_long_side is not None else args.imgsz
    if args.inference_mode == "crops":
        input_mode = "opencv_crop_array"
    elif args.preload_full_frame and resize_long_side and resize_long_side > 0:
        input_mode = "opencv_resized_preload"
    elif args.preload_full_frame:
        input_mode = "opencv_preload"
    else:
        input_mode = "opencv_array_batch"
    print(
        "Phase 1 runtime: "
        f"device={device} imgsz={args.imgsz} batch_size={args.batch_size} "
        f"half={half} input_mode={input_mode} resize_long_side={resize_long_side}",
        flush=True,
    )
    adapter = YOLOPoseAdapter(
        model_id=args.model_id,
        model_config=model_config,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=half,
    )
    metrics = run_phase1_delivery(
        P1RunConfig(
            drive_root=drive_root,
            delivery_id=args.delivery_id,
            run_id=args.run_id,
            run_dir=run_dir,
            model_id=args.model_id,
            device=device,
            inference_mode=args.inference_mode,
            frame_limit=args.frame_limit,
            start_index=args.start_index,
            cameras=args.cameras,
            nms_iou_threshold=args.nms_iou_threshold,
            batch_size=args.batch_size,
            resume=args.resume,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            half=half,
            show_progress=args.show_progress,
            preload_full_frame=args.preload_full_frame,
            resize_long_side=resize_long_side,
        ),
        adapter,
    )
    print(f"Wrote Phase 1 run to {run_dir}")
    print(
        f"status={metrics['summary']['status']} "
        f"records={metrics['summary']['records_written']} "
        f"players={metrics['summary']['total_players_detected']} "
        f"failed_frames={metrics['summary']['failed_frames']}"
    )
    return 0 if metrics["summary"]["status"] in {"pass", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
