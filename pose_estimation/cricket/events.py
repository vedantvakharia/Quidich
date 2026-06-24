"""Inspection helpers for existing ball event pipeline artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


REQUIRED_EVENT_SUFFIXES = [
    "_2D.json",
    "_2D_cleaned.json",
    "_3D.json",
    "_3D_cleaned.json",
    "_3D_trimmed.json",
    "_3D_unreal.json",
    "_reprojection.json",
]


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value == value]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": float(mean(finite)),
        "median": float(median(finite)),
        "p95": percentile(finite, 0.95),
        "max": float(max(finite)),
    }


def expected_artifact_paths(delivery_dir: Path) -> dict[str, Path]:
    stem = delivery_dir.name
    return {suffix: delivery_dir / f"{stem}{suffix}" for suffix in REQUIRED_EVENT_SUFFIXES}


def detection_class_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    class_names: Counter[str] = Counter()
    missing_class_name = 0
    detection_count = 0
    nonempty_camera_frames = 0

    for frame in payload.get("frames", []):
        for camera in frame.get("cameras", []):
            detections = camera.get("detections", [])
            if detections:
                nonempty_camera_frames += 1
            for detection in detections:
                detection_count += 1
                class_name = detection.get("class_name")
                if class_name is None:
                    missing_class_name += 1
                    class_names["__missing__"] += 1
                else:
                    class_names[str(class_name)] += 1

    return {
        "frame_count": len(payload.get("frames", [])),
        "detection_count": detection_count,
        "nonempty_camera_frames": nonempty_camera_frames,
        "class_names": dict(sorted(class_names.items())),
        "missing_class_name_count": missing_class_name,
        "ball_only": set(class_names) <= {"ball", "__missing__"},
    }


def reprojection_errors(path: Path) -> list[float]:
    payload = read_json(path)
    errors = []
    for frame in payload.get("frames", []):
        for camera in frame.get("cameras", []):
            for reprojection in camera.get("reprojection", []):
                error = reprojection.get("error")
                if isinstance(error, (int, float)):
                    errors.append(float(error))
    return errors


def inspect_events_pipeline(drive_root: str | Path) -> dict[str, Any]:
    """Inspect existing ball pipeline artifacts and reprojection statistics."""

    drive_root = Path(drive_root)
    events_root = drive_root / "dataset" / "events-data"
    errors: list[str] = []
    warnings: list[str] = []
    deliveries = []
    all_reprojection_errors: list[float] = []
    all_classes: Counter[str] = Counter()

    if not events_root.exists():
        return {
            "events_root": str(events_root),
            "summary": {"status": "missing"},
            "deliveries": [],
            "errors": [f"events root does not exist: {events_root}"],
            "warnings": [],
        }

    for delivery_dir in sorted(path for path in events_root.iterdir() if path.is_dir()):
        artifacts = expected_artifact_paths(delivery_dir)
        missing = [suffix for suffix, path in artifacts.items() if not path.exists()]
        present = [suffix for suffix, path in artifacts.items() if path.exists()]
        if missing:
            errors.append(f"{delivery_dir.name} missing artifacts {missing}")

        detection_summary = {}
        if artifacts["_2D.json"].exists():
            detection_summary = detection_class_summary(artifacts["_2D.json"])
            all_classes.update(detection_summary.get("class_names", {}))
            if detection_summary.get("missing_class_name_count", 0):
                warnings.append(
                    f"{delivery_dir.name} has detections without class_name"
                )

        reprojection_stats = numeric_stats([])
        if artifacts["_reprojection.json"].exists():
            values = reprojection_errors(artifacts["_reprojection.json"])
            all_reprojection_errors.extend(values)
            reprojection_stats = numeric_stats(values)

        deliveries.append(
            {
                "delivery_id": delivery_dir.name,
                "present_artifacts": present,
                "missing_artifacts": missing,
                "detection_summary": detection_summary,
                "reprojection_error_px": reprojection_stats,
            }
        )

    summary = {
        "status": "pass" if not errors else "fail",
        "delivery_count": len(deliveries),
        "deliveries_with_full_artifact_chain": sum(
            1 for delivery in deliveries if not delivery["missing_artifacts"]
        ),
        "class_names": dict(sorted(all_classes.items())),
        "all_reprojection_error_px": numeric_stats(all_reprojection_errors),
        "error_count": len(errors),
        "warning_count": len(warnings),
    }

    return {
        "events_root": str(events_root),
        "required_artifact_suffixes": REQUIRED_EVENT_SUFFIXES,
        "summary": summary,
        "deliveries": deliveries,
        "errors": errors,
        "warnings": warnings,
    }

