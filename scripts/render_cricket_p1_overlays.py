#!/usr/bin/env python3
"""Render visual QA overlays for cricket Phase 1 prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_estimation.cricket.dataset import resolve_delivery_camera_dirs
from pose_estimation.cricket.p1_outputs import COCO_17_EDGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", default="drive")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--sample-every", type=int, default=60)
    parser.add_argument("--max-per-camera", type=int, default=20)
    parser.add_argument("--keypoint-threshold", type=float, default=0.2)
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def draw_record(image, record: dict, *, keypoint_threshold: float) -> None:
    for player_index, player in enumerate(record.get("players", []), start=1):
        x, y, w, h = [int(round(value)) for value in player["bbox_xywh_px"]]
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image,
            f"P? {player.get('track_confidence', 0):.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        keypoints = player["pose_2d"]["keypoints_px"]
        confidence = player["pose_2d"]["confidence"]
        for left, right in COCO_17_EDGES:
            if confidence[left] < keypoint_threshold or confidence[right] < keypoint_threshold:
                continue
            p1 = tuple(int(round(value)) for value in keypoints[left])
            p2 = tuple(int(round(value)) for value in keypoints[right])
            cv2.line(image, p1, p2, (255, 180, 0), 2)
        for point, score in zip(keypoints, confidence):
            if score < keypoint_threshold:
                continue
            center = tuple(int(round(value)) for value in point)
            cv2.circle(image, center, 3, (0, 0, 255), -1)
        cv2.putText(
            image,
            str(player_index),
            (x + 4, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    drive_root = Path(args.drive_root)
    if not drive_root.is_absolute():
        drive_root = ROOT / drive_root
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    delivery_id = manifest["delivery_id"]
    run_id = manifest["run_id"]
    artifact_dir = (
        Path(args.artifact_dir)
        if args.artifact_dir
        else ROOT / "benchmarks" / "artifacts" / run_id / "overlays"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    camera_dirs = resolve_delivery_camera_dirs(drive_root, delivery_id)

    written = []
    for prediction_path in sorted((run_dir / "predictions").glob("cam_*.jsonl")):
        camera_id = prediction_path.stem
        camera_dir = camera_dirs.get(camera_id)
        if camera_dir is None:
            continue
        camera_output_dir = artifact_dir / camera_id
        camera_output_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for index, record in enumerate(read_jsonl(prediction_path)):
            if index % args.sample_every != 0:
                continue
            if count >= args.max_per_camera:
                break
            image_path = camera_dir / record["frame_name"]
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            draw_record(image, record, keypoint_threshold=args.keypoint_threshold)
            cv2.putText(
                image,
                f"{camera_id} frame={record['frame_index']} detections={len(record['players'])}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            output_path = camera_output_dir / f"{camera_id}_{record['frame_index']}.jpg"
            cv2.imwrite(str(output_path), image)
            written.append(str(output_path))
            count += 1

    manifest_payload = {
        "schema_version": "cricket_phase1_visual_qa/v1",
        "run_id": run_id,
        "delivery_id": delivery_id,
        "artifact_dir": str(artifact_dir),
        "sample_every": args.sample_every,
        "max_per_camera": args.max_per_camera,
        "overlay_count": len(written),
        "sample_overlays": written[:20],
    }
    with (run_dir / "visual_qa_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {len(written)} overlays under {artifact_dir}")
    print(f"Wrote {run_dir / 'visual_qa_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

