"""Phase 1 output conversion and JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pose_estimation.cricket.contract import (
    KEYPOINT_COUNT,
    SCHEMA_VERSION,
    SKELETON,
    validate_group1_frame,
)


COCO_17_EDGES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def clip_xyxy(
    bbox_xyxy: Iterable[float],
    *,
    width: int,
    height: int,
) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def xyxy_to_xywh(bbox_xyxy: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def normalize_xywh(bbox_xywh: Iterable[float], *, width: int, height: int) -> list[float]:
    x, y, w, h = [float(value) for value in bbox_xywh]
    return [x / width, y / height, w / width, h / height]


def normalize_keypoints(keypoints_px: list[list[float]], *, width: int, height: int) -> list[list[float]]:
    return [[float(x) / width, float(y) / height] for x, y in keypoints_px]


def coerce_coco17_keypoints(
    keypoints: Iterable[Iterable[float]],
    confidence: Iterable[float] | None = None,
) -> tuple[list[list[float]], list[float]]:
    keypoint_rows = [list(row) for row in keypoints]
    confidence_values = list(confidence) if confidence is not None else None
    output_points: list[list[float]] = []
    output_confidence: list[float] = []

    for index in range(KEYPOINT_COUNT):
        row = keypoint_rows[index] if index < len(keypoint_rows) else []
        x = float(row[0]) if len(row) >= 1 and np.isfinite(row[0]) else 0.0
        y = float(row[1]) if len(row) >= 2 and np.isfinite(row[1]) else 0.0
        if confidence_values is not None and index < len(confidence_values):
            score = confidence_values[index]
        elif len(row) >= 3:
            score = row[2]
        else:
            score = 0.0
        output_points.append([x, y])
        output_confidence.append(float(score) if np.isfinite(score) else 0.0)
    return output_points, output_confidence


def bbox_iou_xyxy(left: Iterable[float], right: Iterable[float]) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


def nms_predictions(predictions: list[dict[str, Any]], *, iou_threshold: float = 0.6) -> list[dict[str, Any]]:
    ordered = sorted(predictions, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    for prediction in ordered:
        if all(bbox_iou_xyxy(prediction["bbox_xyxy"], existing["bbox_xyxy"]) < iou_threshold for existing in kept):
            kept.append(prediction)
    return kept


def offset_prediction(prediction: dict[str, Any], *, x_offset: float, y_offset: float) -> dict[str, Any]:
    shifted = dict(prediction)
    x1, y1, x2, y2 = [float(value) for value in shifted["bbox_xyxy"]]
    shifted["bbox_xyxy"] = [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset]
    shifted["keypoints"] = [
        [float(point[0]) + x_offset, float(point[1]) + y_offset]
        for point in shifted.get("keypoints", [])
    ]
    return shifted


def scale_prediction(prediction: dict[str, Any], *, x_scale: float, y_scale: float) -> dict[str, Any]:
    scaled = dict(prediction)
    x1, y1, x2, y2 = [float(value) for value in scaled["bbox_xyxy"]]
    scaled["bbox_xyxy"] = [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale]
    scaled["keypoints"] = [
        [float(point[0]) * x_scale, float(point[1]) * y_scale]
        for point in scaled.get("keypoints", [])
    ]
    return scaled


def prediction_to_player(
    prediction: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    bbox_xyxy = clip_xyxy(prediction["bbox_xyxy"], width=image_width, height=image_height)
    bbox_xywh = xyxy_to_xywh(bbox_xyxy)
    if bbox_xywh[2] <= 0 or bbox_xywh[3] <= 0:
        return None
    keypoints_px, keypoint_confidence = coerce_coco17_keypoints(
        prediction.get("keypoints", []),
        prediction.get("keypoint_confidence"),
    )
    keypoints_px = [
        [
            min(max(float(x), 0.0), float(image_width)),
            min(max(float(y), 0.0), float(image_height)),
        ]
        for x, y in keypoints_px
    ]
    return {
        "global_player_id": None,
        "local_track_id": None,
        "role": "unknown",
        "bbox_xywh_px": bbox_xywh,
        "bbox_xywh_norm": normalize_xywh(bbox_xywh, width=image_width, height=image_height),
        "track_confidence": float(prediction.get("score") or 0.0),
        "pose_2d": {
            "skeleton": SKELETON,
            "keypoints_px": keypoints_px,
            "keypoints_norm": normalize_keypoints(keypoints_px, width=image_width, height=image_height),
            "confidence": keypoint_confidence,
        },
        "pose_3d": None,
    }


def build_phase1_frame_record(
    *,
    match_id: str,
    delivery_id: str,
    camera_id: str,
    frame_index: int,
    frame_name: str,
    image_width: int,
    image_height: int,
    predictions: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    players = []
    for prediction in predictions:
        player = prediction_to_player(
            prediction,
            image_width=image_width,
            image_height=image_height,
        )
        if player is not None:
            players.append(player)
    record = {
        "schema_version": SCHEMA_VERSION,
        "match_id": match_id,
        "delivery_id": delivery_id,
        "camera_id": camera_id,
        "frame_index": int(frame_index),
        "frame_name": frame_name,
        "players": players,
    }
    if metadata:
        record["metadata"] = metadata
    validate_group1_frame(record, final_handoff=False)
    return record


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
