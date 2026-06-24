import json
import subprocess
import sys
from pathlib import Path

import pytest

from pose_estimation.cricket.calibration import audit_calibration
from pose_estimation.cricket.contract import (
    example_group1_frame,
    validate_group1_frame,
)
from pose_estimation.cricket.dataset import discover_dataset
from pose_estimation.cricket.events import inspect_events_pipeline


CAMERA_GROUPS = {
    "bt_01": ["01", "04"],
    "bt_02": ["02", "05", "07"],
    "bt_03": ["03", "06"],
}


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_frame_dataset(root: Path, *, frame_ids_by_camera: dict[str, int] | None = None):
    frame_ids_by_camera = frame_ids_by_camera or {}
    for group, cameras in CAMERA_GROUPS.items():
        for camera in cameras:
            frame_id = frame_ids_by_camera.get(camera, 1)
            path = (
                root
                / "dataset"
                / group
                / "DELIVERY_001"
                / f"camera{camera}"
                / f"frame_camera{camera}_{frame_id:06d}.jpg"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-a-real-image")


def make_calibration(root: Path):
    calibration_dir = root / "dataset" / "calibration-data" / "CCPL080626" / "calibration_data"
    projection = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    intrinsics = {
        f"C{index:02d}": {
            "camera_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        }
        for index in range(1, 8)
    }
    extrinsics = {
        "camera_locations": {f"C{index:02d}": [0.0, 0.0, 0.0] for index in range(1, 8)},
        "projection_matrices": {f"C{index:02d}": projection for index in range(1, 8)},
    }
    write_json(calibration_dir / "Bundle_Adjusted_intrinsics.json", intrinsics)
    write_json(calibration_dir / "Bundle_Adjusted_extrinsics.json", extrinsics)
    write_json(calibration_dir / "camera_calibration_config.json", {"camera_parameters": []})
    write_json(calibration_dir / "pitch_calibration_config.json", {"origin": [0.0, 0.0, 0.0]})
    write_json(calibration_dir / "crop_mech.json", {"01": [[0, 0, 10, 10]]})
    (calibration_dir / "CPL08626_coord_aligned.csv").write_text(
        "pt,256.0,144.0,1.0\n",
        encoding="utf-8",
    )


def make_events(root: Path):
    delivery_dir = root / "dataset" / "events-data" / "DELIVERY_001_V0"
    stem = delivery_dir.name
    base_2d = {
        "ball_id": stem,
        "frames": [
            {
                "frame_id": 1,
                "cameras": [
                    {
                        "camera_id": 1,
                        "detections": [
                            {
                                "coords": [0.1, 0.1, 0.01, 0.01],
                                "confidence_score": 0.9,
                                "class_id": 0,
                                "class_name": "ball",
                                "track_id": 1,
                            }
                        ],
                    }
                ],
            }
        ],
        "selected_track_ids": {"1": 1},
    }
    write_json(delivery_dir / f"{stem}_2D.json", base_2d)
    write_json(delivery_dir / f"{stem}_2D_cleaned.json", base_2d)
    for suffix in ["_3D", "_3D_cleaned", "_3D_trimmed", "_3D_unreal"]:
        write_json(delivery_dir / f"{stem}{suffix}.json", {"ball_id": stem, "frames": {"1": [256.0, 144.0, 1.0]}})
    write_json(
        delivery_dir / f"{stem}_reprojection.json",
        {
            "ball_id": stem,
            "frames": [
                {
                    "frame_id": 1,
                    "cameras": [
                        {
                            "camera_id": 1,
                            "reprojection": [{"coords": [0.1, 0.1], "error": 0.0}],
                        }
                    ],
                }
            ],
        },
    )


def make_phase0_fixture(root: Path):
    make_frame_dataset(root)
    make_calibration(root)
    make_events(root)


def test_dataset_discovery_validates_camera_groups_and_frame_counts(tmp_path: Path):
    make_frame_dataset(tmp_path)
    report = discover_dataset(tmp_path, expected_frame_count=1, inspect_dimensions=False)
    assert report["summary"]["status"] == "pass"
    assert report["summary"]["delivery_count"] == 1
    assert report["summary"]["deliveries_with_all_cameras"] == 1


def test_dataset_discovery_reports_frame_id_mismatches(tmp_path: Path):
    make_frame_dataset(tmp_path, frame_ids_by_camera={"07": 2})
    report = discover_dataset(tmp_path, expected_frame_count=1, inspect_dimensions=False)
    delivery = report["deliveries"][0]
    assert delivery["sync"]["camera_frame_sets_identical"] is False
    assert "cam_07" in delivery["sync"]["mismatches"]


def test_events_pipeline_reports_artifacts_and_reprojection_stats(tmp_path: Path):
    make_events(tmp_path)
    report = inspect_events_pipeline(tmp_path)
    assert report["summary"]["status"] == "pass"
    assert report["summary"]["class_names"] == {"ball": 1}
    assert report["summary"]["all_reprojection_error_px"]["count"] == 1


def test_calibration_audit_projects_and_compares_ball_reprojection(tmp_path: Path):
    make_calibration(tmp_path)
    make_events(tmp_path)
    report = audit_calibration(tmp_path)
    assert report["summary"]["status"] == "pass"
    assert report["summary"]["ball_reprojection_comparison_count"] == 1
    assert report["summary"]["ball_reprojection_delta_p95_px"] == 0.0


def test_contract_validation_accepts_final_and_rejects_bad_role():
    record = example_group1_frame(final_handoff=True)
    validate_group1_frame(record, final_handoff=True)

    record["players"][0]["role"] = "captain"
    with pytest.raises(ValueError, match="invalid role"):
        validate_group1_frame(record, final_handoff=True)


def test_contract_validation_requires_global_id_for_final_handoff():
    record = example_group1_frame(final_handoff=False)
    with pytest.raises(ValueError, match="global_player_id is required"):
        validate_group1_frame(record, final_handoff=True)


def test_phase0_cli_writes_compact_evidence(tmp_path: Path):
    drive_root = tmp_path / "drive"
    output_dir = tmp_path / "run"
    make_phase0_fixture(drive_root)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/phase0_audit.py",
            "--drive-root",
            str(drive_root),
            "--run-id",
            "phase0-test",
            "--output-dir",
            str(output_dir),
            "--expected-frame-count",
            "1",
            "--skip-image-dimensions",
            "--fail-on-internal-errors",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout
    assert (output_dir / "phase0_readiness.json").exists()
    assert (output_dir / "dataset_inventory.json").exists()
    readiness = json.loads((output_dir / "phase0_readiness.json").read_text())
    assert readiness["internal_status"] == "pass"
    assert readiness["external_status"] == "blocked"

