"""Dataset discovery and validation for the internal cricket frame layout."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2


EXPECTED_CAMERA_GROUPS = {
    "bt_01": ["01", "04"],
    "bt_02": ["02", "05", "07"],
    "bt_03": ["03", "06"],
}

EXPECTED_CAMERA_IDS = [f"{index:02d}" for index in range(1, 8)]
FRAME_RE = re.compile(r"^frame_camera(?P<camera>\d{2})_(?P<frame>\d+)\.jpg$")


def camera_label(camera_id: str) -> str:
    return f"cam_{int(camera_id):02d}"


def parse_frame_id(path: Path) -> int | None:
    match = FRAME_RE.match(path.name)
    if not match:
        return None
    return int(match.group("frame"))


def read_image_dimensions(path: Path) -> list[int] | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return [int(width), int(height)]


def _sample_dimensions(frames: list[Path]) -> dict[str, list[int] | None]:
    if not frames:
        return {"first": None, "middle": None, "last": None}
    indices = [0, len(frames) // 2, len(frames) - 1]
    labels = ["first", "middle", "last"]
    return {
        label: read_image_dimensions(frames[index])
        for label, index in zip(labels, indices)
    }


def _camera_record(
    *,
    capture_group: str,
    camera_dir: Path,
    inspect_dimensions: bool,
) -> dict[str, Any]:
    camera_id = camera_dir.name.replace("camera", "")
    frames = sorted(camera_dir.glob("*.jpg"))
    parsed_frame_ids = [parse_frame_id(frame) for frame in frames]
    invalid_names = [frame.name for frame, frame_id in zip(frames, parsed_frame_ids) if frame_id is None]
    frame_ids = sorted(frame_id for frame_id in parsed_frame_ids if frame_id is not None)

    return {
        "capture_group": capture_group,
        "camera_id": camera_label(camera_id),
        "camera_number": camera_id,
        "path": str(camera_dir),
        "frame_count": len(frames),
        "valid_frame_name_count": len(frame_ids),
        "invalid_frame_names": invalid_names[:20],
        "first_frame_id": frame_ids[0] if frame_ids else None,
        "last_frame_id": frame_ids[-1] if frame_ids else None,
        "frame_ids": frame_ids,
        "image_dimensions": _sample_dimensions(frames) if inspect_dimensions else {},
    }


def discover_dataset(
    drive_root: str | Path,
    *,
    expected_frame_count: int = 600,
    inspect_dimensions: bool = True,
) -> dict[str, Any]:
    """Discover deliveries, camera folders, frame counts, and sync mismatches."""

    drive_root = Path(drive_root)
    dataset_root = drive_root / "dataset"
    errors: list[str] = []
    warnings: list[str] = []
    delivery_map: dict[str, dict[str, Any]] = {}

    if not dataset_root.exists():
        return {
            "dataset_root": str(dataset_root),
            "summary": {"status": "missing"},
            "deliveries": [],
            "errors": [f"dataset root does not exist: {dataset_root}"],
            "warnings": [],
        }

    for capture_group, expected_cameras in EXPECTED_CAMERA_GROUPS.items():
        group_root = dataset_root / capture_group
        if not group_root.exists():
            errors.append(f"missing capture group: {group_root}")
            continue
        for delivery_dir in sorted(path for path in group_root.iterdir() if path.is_dir()):
            delivery = delivery_map.setdefault(
                delivery_dir.name,
                {
                    "delivery_id": delivery_dir.name,
                    "cameras": [],
                    "errors": [],
                    "warnings": [],
                },
            )
            seen_camera_numbers = []
            for camera_dir in sorted(path for path in delivery_dir.iterdir() if path.is_dir()):
                if not camera_dir.name.startswith("camera"):
                    continue
                camera_number = camera_dir.name.replace("camera", "")
                seen_camera_numbers.append(camera_number)
                record = _camera_record(
                    capture_group=capture_group,
                    camera_dir=camera_dir,
                    inspect_dimensions=inspect_dimensions,
                )
                delivery["cameras"].append(record)
                if record["frame_count"] != expected_frame_count:
                    delivery["errors"].append(
                        f"{delivery_dir.name}/{camera_dir.name} has "
                        f"{record['frame_count']} frames, expected {expected_frame_count}"
                    )
                if record["invalid_frame_names"]:
                    delivery["errors"].append(
                        f"{delivery_dir.name}/{camera_dir.name} has invalid frame names"
                    )
            missing_in_group = sorted(set(expected_cameras) - set(seen_camera_numbers))
            if missing_in_group:
                delivery["errors"].append(
                    f"{capture_group}/{delivery_dir.name} missing cameras {missing_in_group}"
                )

    deliveries = []
    for delivery in sorted(delivery_map.values(), key=lambda item: item["delivery_id"]):
        cameras = sorted(delivery["cameras"], key=lambda item: item["camera_id"])
        delivery["cameras"] = cameras
        camera_ids = [camera["camera_id"] for camera in cameras]
        missing_cameras = [
            camera_label(camera_id)
            for camera_id in EXPECTED_CAMERA_IDS
            if camera_label(camera_id) not in camera_ids
        ]
        if missing_cameras:
            delivery["errors"].append(
                f"{delivery['delivery_id']} missing expected cameras {missing_cameras}"
            )

        frame_sets = {
            camera["camera_id"]: set(camera["frame_ids"])
            for camera in cameras
            if camera["frame_ids"]
        }
        union = set().union(*frame_sets.values()) if frame_sets else set()
        intersection = set.intersection(*frame_sets.values()) if frame_sets else set()
        sync_mismatches = {}
        for camera_id, frame_ids in frame_sets.items():
            missing = sorted(union - frame_ids)
            extra = sorted(frame_ids - intersection)
            if missing or extra:
                sync_mismatches[camera_id] = {
                    "missing_from_camera_count": len(missing),
                    "extra_vs_common_count": len(extra),
                    "missing_from_camera_sample": missing[:20],
                    "extra_vs_common_sample": extra[:20],
                }
        delivery["sync"] = {
            "union_frame_count": len(union),
            "common_frame_count": len(intersection),
            "camera_frame_sets_identical": not sync_mismatches,
            "mismatches": sync_mismatches,
        }
        if sync_mismatches:
            delivery["warnings"].append("camera frame id sets are not identical")
        deliveries.append(delivery)

    delivery_errors = [error for delivery in deliveries for error in delivery["errors"]]
    delivery_warnings = [warning for delivery in deliveries for warning in delivery["warnings"]]
    errors.extend(delivery_errors)
    warnings.extend(delivery_warnings)

    summary = {
        "status": "pass" if not errors else "fail",
        "delivery_count": len(deliveries),
        "camera_folder_count": sum(len(delivery["cameras"]) for delivery in deliveries),
        "expected_frame_count_per_camera": expected_frame_count,
        "deliveries_with_all_cameras": sum(
            1 for delivery in deliveries if len(delivery["cameras"]) == len(EXPECTED_CAMERA_IDS)
        ),
        "deliveries_with_sync_mismatch": sum(
            1 for delivery in deliveries if not delivery["sync"]["camera_frame_sets_identical"]
        ),
        "error_count": len(errors),
        "warning_count": len(warnings),
    }

    compact_deliveries = []
    for delivery in deliveries:
        compact_cameras = []
        for camera in delivery["cameras"]:
            compact = dict(camera)
            compact.pop("frame_ids", None)
            compact_cameras.append(compact)
        compact_delivery = dict(delivery)
        compact_delivery["cameras"] = compact_cameras
        compact_deliveries.append(compact_delivery)

    return {
        "dataset_root": str(dataset_root),
        "expected_camera_groups": EXPECTED_CAMERA_GROUPS,
        "expected_camera_ids": [camera_label(camera_id) for camera_id in EXPECTED_CAMERA_IDS],
        "summary": summary,
        "deliveries": compact_deliveries,
        "errors": errors,
        "warnings": warnings,
    }


def resolve_delivery_camera_dirs(
    drive_root: str | Path,
    delivery_id: str,
) -> dict[str, Path]:
    """Return full-frame camera folders for one delivery keyed by cam_XX."""

    drive_root = Path(drive_root)
    dataset_root = drive_root / "dataset"
    camera_dirs: dict[str, Path] = {}
    for capture_group, camera_numbers in EXPECTED_CAMERA_GROUPS.items():
        delivery_root = dataset_root / capture_group / delivery_id
        for camera_number in camera_numbers:
            camera_dir = delivery_root / f"camera{camera_number}"
            if camera_dir.exists():
                camera_dirs[camera_label(camera_number)] = camera_dir
    return dict(sorted(camera_dirs.items()))


def frame_paths_for_camera(camera_dir: str | Path) -> list[Path]:
    """Return valid JPG frame paths sorted by absolute frame id."""

    paths = [path for path in Path(camera_dir).glob("*.jpg") if parse_frame_id(path) is not None]
    return sorted(paths, key=lambda path: parse_frame_id(path) or -1)
