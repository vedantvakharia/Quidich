"""Calibration loading, validation, projection, and reprojection checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


CALIBRATION_RELATIVE_DIR = Path("dataset/calibration-data/CCPL080626/calibration_data")
IMAGE_WIDTH = 2560
IMAGE_HEIGHT = 1440
CAMERA_KEYS = [f"C{index:02d}" for index in range(1, 8)]


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
    finite = [float(value) for value in values if np.isfinite(value)]
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


def project_world_to_pixel(point_xyz: list[float] | np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    point = np.asarray([*np.asarray(point_xyz, dtype=float), 1.0], dtype=float)
    projected = np.asarray(projection_matrix, dtype=float) @ point
    if abs(float(projected[2])) < 1e-12:
        return np.asarray([np.nan, np.nan], dtype=float)
    return projected[:2] / projected[2]


def pixel_to_normalized(pixel_xy: np.ndarray) -> np.ndarray:
    return np.asarray([pixel_xy[0] / IMAGE_WIDTH, pixel_xy[1] / IMAGE_HEIGHT], dtype=float)


def load_survey_points(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                rows.append(
                    {
                        "name": row[0],
                        "point_world_m": [float(row[1]), float(row[2]), float(row[3])],
                    }
                )
            except ValueError:
                continue
    return rows


def load_projection_matrices(calibration_dir: str | Path) -> dict[str, np.ndarray]:
    payload = read_json(Path(calibration_dir) / "Bundle_Adjusted_extrinsics.json")
    matrices = payload.get("projection_matrices", {})
    return {camera_id: np.asarray(matrix, dtype=float) for camera_id, matrix in matrices.items()}


def validate_matrix_shapes(
    intrinsics: dict[str, Any],
    projection_matrices: dict[str, np.ndarray],
) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []

    for camera_id in CAMERA_KEYS:
        if camera_id not in intrinsics:
            errors.append(f"missing intrinsics for {camera_id}")
            continue
        matrix = np.asarray(intrinsics[camera_id].get("camera_matrix"), dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            errors.append(f"invalid 3x3 intrinsic matrix for {camera_id}")

    for camera_id in CAMERA_KEYS:
        matrix = projection_matrices.get(camera_id)
        if matrix is None:
            errors.append(f"missing projection matrix for {camera_id}")
            continue
        if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
            errors.append(f"invalid 3x4 projection matrix for {camera_id}")

    extra_projection_keys = sorted(set(projection_matrices) - set(CAMERA_KEYS))
    if extra_projection_keys:
        warnings.append(f"extra projection matrix keys present: {extra_projection_keys}")

    return errors, warnings


def survey_projection_report(
    survey_points: list[dict[str, Any]],
    projection_matrices: dict[str, np.ndarray],
) -> dict[str, Any]:
    per_camera = {}
    for camera_id, projection_matrix in sorted(projection_matrices.items()):
        finite = 0
        in_frame = 0
        samples = []
        for point in survey_points:
            pixel = project_world_to_pixel(point["point_world_m"], projection_matrix)
            is_finite = bool(np.isfinite(pixel).all())
            if is_finite:
                finite += 1
            is_in_frame = bool(
                is_finite
                and 0 <= pixel[0] < IMAGE_WIDTH
                and 0 <= pixel[1] < IMAGE_HEIGHT
            )
            if is_in_frame:
                in_frame += 1
            if len(samples) < 5:
                samples.append(
                    {
                        "name": point["name"],
                        "pixel_xy": pixel.tolist() if is_finite else [None, None],
                        "in_frame": is_in_frame,
                    }
                )
        per_camera[camera_id] = {
            "survey_point_count": len(survey_points),
            "finite_projection_count": finite,
            "in_frame_projection_count": in_frame,
            "sample": samples,
        }
    return per_camera


def compare_ball_reprojection(drive_root: str | Path) -> dict[str, Any]:
    """Compare projected 3D ball points with existing reprojection coordinates."""

    drive_root = Path(drive_root)
    calibration_dir = drive_root / CALIBRATION_RELATIVE_DIR
    projection_matrices = load_projection_matrices(calibration_dir)
    events_root = drive_root / "dataset" / "events-data"
    errors = []
    warnings = []
    deliveries = []
    delta_errors = []
    stored_errors = []

    if not events_root.exists():
        return {
            "summary": {"status": "missing"},
            "deliveries": [],
            "errors": [f"events root does not exist: {events_root}"],
            "warnings": [],
        }

    for delivery_dir in sorted(path for path in events_root.iterdir() if path.is_dir()):
        stem = delivery_dir.name
        path_3d = delivery_dir / f"{stem}_3D.json"
        path_reprojection = delivery_dir / f"{stem}_reprojection.json"
        if not path_3d.exists() or not path_reprojection.exists():
            warnings.append(f"{stem} lacks 3D or reprojection artifact")
            continue

        payload_3d = read_json(path_3d)
        payload_reprojection = read_json(path_reprojection)
        frames_3d = payload_3d.get("frames", {})
        delivery_deltas = []
        delivery_stored = []
        comparison_count = 0

        for frame in payload_reprojection.get("frames", []):
            frame_id = str(frame.get("frame_id"))
            point = frames_3d.get(frame_id)
            if point is None:
                continue
            for camera in frame.get("cameras", []):
                camera_key = f"C{int(camera.get('camera_id')):02d}"
                projection_matrix = projection_matrices.get(camera_key)
                if projection_matrix is None:
                    continue
                projected_pixel = project_world_to_pixel(point, projection_matrix)
                if not np.isfinite(projected_pixel).all():
                    continue
                projected_norm = pixel_to_normalized(projected_pixel)
                for reprojection in camera.get("reprojection", []):
                    coords = reprojection.get("coords")
                    if not coords or len(coords) < 2:
                        continue
                    stored_norm = np.asarray(coords[:2], dtype=float)
                    delta = (projected_norm - stored_norm) * np.asarray(
                        [IMAGE_WIDTH, IMAGE_HEIGHT],
                        dtype=float,
                    )
                    delta_px = float(np.linalg.norm(delta))
                    delivery_deltas.append(delta_px)
                    delta_errors.append(delta_px)
                    stored_error = reprojection.get("error")
                    if isinstance(stored_error, (int, float)):
                        delivery_stored.append(float(stored_error))
                        stored_errors.append(float(stored_error))
                    comparison_count += 1

        if comparison_count == 0:
            warnings.append(f"{stem} produced no reprojection comparisons")
        deliveries.append(
            {
                "delivery_id": stem,
                "comparison_count": comparison_count,
                "projected_vs_stored_reprojection_delta_px": numeric_stats(delivery_deltas),
                "stored_ball_reprojection_error_px": numeric_stats(delivery_stored),
            }
        )

    delta_stats = numeric_stats(delta_errors)
    stored_stats = numeric_stats(stored_errors)
    if delta_stats.get("p95") is not None and float(delta_stats["p95"]) > 10.0:
        warnings.append(
            "projected-vs-stored reprojection deltas exceed 10 px at p95; "
            "stored reprojection coordinates likely include crop/normalization "
            "conventions for some cameras. Use stored_ball_reprojection_error_px "
            "as the existing ball-pipeline quality signal."
        )

    return {
        "summary": {
            "status": "pass" if not errors else "fail",
            "delivery_count": len(deliveries),
            "comparison_count": sum(delivery["comparison_count"] for delivery in deliveries),
            "projected_vs_stored_reprojection_delta_px": delta_stats,
            "stored_ball_reprojection_error_px": stored_stats,
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        "deliveries": deliveries,
        "errors": errors,
        "warnings": warnings,
    }


def audit_calibration(drive_root: str | Path) -> dict[str, Any]:
    """Load and validate calibration files for Phase 0 readiness."""

    drive_root = Path(drive_root)
    calibration_dir = drive_root / CALIBRATION_RELATIVE_DIR
    errors: list[str] = []
    warnings: list[str] = []
    required_files = [
        "Bundle_Adjusted_intrinsics.json",
        "Bundle_Adjusted_extrinsics.json",
        "camera_calibration_config.json",
        "pitch_calibration_config.json",
        "crop_mech.json",
        "CPL08626_coord_aligned.csv",
    ]

    for filename in required_files:
        if not (calibration_dir / filename).exists():
            errors.append(f"missing calibration file: {filename}")

    if errors:
        return {
            "calibration_dir": str(calibration_dir),
            "summary": {"status": "fail", "error_count": len(errors)},
            "errors": errors,
            "warnings": warnings,
        }

    intrinsics = read_json(calibration_dir / "Bundle_Adjusted_intrinsics.json")
    projection_matrices = load_projection_matrices(calibration_dir)
    matrix_errors, matrix_warnings = validate_matrix_shapes(intrinsics, projection_matrices)
    errors.extend(matrix_errors)
    warnings.extend(matrix_warnings)

    survey_points = load_survey_points(calibration_dir / "CPL08626_coord_aligned.csv")
    if not survey_points:
        errors.append("no surveyed reference points loaded")

    survey_report = survey_projection_report(survey_points, projection_matrices)
    ball_report = compare_ball_reprojection(drive_root)
    errors.extend(ball_report.get("errors", []))
    warnings.extend(ball_report.get("warnings", []))

    summary = {
        "status": "pass" if not errors else "fail",
        "camera_count": len(projection_matrices),
        "survey_point_count": len(survey_points),
        "ball_reprojection_comparison_count": ball_report["summary"].get("comparison_count", 0),
        "ball_reprojection_delta_p95_px": ball_report["summary"]
        .get("projected_vs_stored_reprojection_delta_px", {})
        .get("p95"),
        "stored_ball_reprojection_p95_px": ball_report["summary"]
        .get("stored_ball_reprojection_error_px", {})
        .get("p95"),
        "error_count": len(errors),
        "warning_count": len(warnings),
    }

    return {
        "calibration_dir": str(calibration_dir),
        "image_size_px": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "summary": summary,
        "survey_projection": survey_report,
        "ball_reprojection": ball_report,
        "errors": errors,
        "warnings": warnings,
    }
