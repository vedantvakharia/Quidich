"""Phase 2 JSONL I/O, retroactive write buffer, and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from pose_estimation.cricket.contract import validate_group1_frame
from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_pose_vector import build_pose_vector
from pose_estimation.cricket.p2_tracker import CameraTracker, Detection


def read_p1_frames(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def frame_to_detections(record: dict, config: P2Config) -> list[Detection]:
    detections: list[Detection] = []
    for player in record.get("players", []):
        pose_block = player.get("pose_2d", {})
        pose = build_pose_vector(
            pose_block.get("keypoints_px", []),
            pose_block.get("confidence", []),
            player.get("bbox_xywh_px", [0, 0, 0, 0]),
            config,
        )
        detections.append(
            Detection(
                bbox_xywh=list(player.get("bbox_xywh_px", [0, 0, 0, 0])),
                pose=pose,
                confidence=float(player.get("track_confidence") or 0.0),
                player=player,
            )
        )
    return detections


def track_camera_file(
    input_path: str | Path,
    output_path: str | Path,
    diagnostics_path: str | Path,
    camera_id: str,
    delivery_id: str,
    config: P2Config,
    expected_frames: int = 600,
) -> dict:
    output_path = Path(output_path)
    diagnostics_path = Path(diagnostics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = CameraTracker(camera_id, config)
    # Buffer holds (record) until its tentative tracks resolve; players are stamped in place,
    # so we can flush after `tentative_confirm_window` frames of look-ahead.
    buffer: list[dict] = []
    frames_read = 0

    with output_path.open("w", encoding="utf-8") as out:
        def flush(up_to: int) -> None:
            while len(buffer) > up_to:
                record = buffer.pop(0)
                validate_group1_frame(record)
                out.write(json.dumps(record, sort_keys=True) + "\n")

        # Drive the tracker with a per-camera ordinal (0, 1, 2, ...) so confirmation-window and
        # dormancy math are independent of P1's absolute frame_index (which may be large/sparse).
        # The output record keeps its own true frame_index untouched.
        for ordinal, record in enumerate(read_p1_frames(input_path)):
            frames_read += 1
            detections = frame_to_detections(record, config)
            tracker.update(detections, frame_index=ordinal)
            buffer.append(record)
            flush(config.tentative_confirm_window)  # keep a look-ahead window buffered

        tracker.finalize()
        flush(0)  # drain remaining buffer at EOF

    diagnostics = {
        "camera_id": camera_id,
        "delivery_id": delivery_id,
        "status": "ok",
        "error": None,
        "frames_expected": expected_frames,
        "frames_read": frames_read,
        **tracker.diagnostics,
    }
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return diagnostics
