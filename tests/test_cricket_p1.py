import json
from pathlib import Path

import cv2
import numpy as np

from pose_estimation.cricket.contract import validate_group1_frame
from pose_estimation.cricket.p1_outputs import (
    build_phase1_frame_record,
    nms_predictions,
    offset_prediction,
    scale_prediction,
    xyxy_to_xywh,
)
from pose_estimation.cricket.p1_runner import P1RunConfig, run_phase1_delivery


class FakeAdapter:
    model_id = "fake_pose"

    def predict(self, image):
        return [
            {
                "bbox_xyxy": [10.0, 20.0, 50.0, 100.0],
                "score": 0.8,
                "keypoints": [[12.0 + index, 22.0 + index] for index in range(17)],
                "keypoint_confidence": [0.9 for _ in range(17)],
            }
        ]


def write_image(path: Path, width: int = 128, height: int = 96):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def make_tiny_delivery(root: Path, delivery_id: str = "DELIVERY_001", frame_count: int = 1):
    groups = {
        "bt_01": ["01", "04"],
        "bt_02": ["02", "05", "07"],
        "bt_03": ["03", "06"],
    }
    for group, cameras in groups.items():
        for camera in cameras:
            for frame_id in range(1, frame_count + 1):
                write_image(
                    root
                    / "dataset"
                    / group
                    / delivery_id
                    / f"camera{camera}"
                    / f"frame_camera{camera}_{frame_id:06d}.jpg"
                )


def test_xyxy_to_xywh_conversion():
    assert xyxy_to_xywh([10, 20, 50, 100]) == [10.0, 20.0, 40.0, 80.0]


def test_phase1_record_validates_empty_and_populated_detections():
    record = build_phase1_frame_record(
        match_id="CCPL080626",
        delivery_id="DELIVERY_001",
        camera_id="cam_01",
        frame_index=1,
        frame_name="frame_camera01_000001.jpg",
        image_width=128,
        image_height=96,
        predictions=FakeAdapter().predict(None),
    )
    validate_group1_frame(record, final_handoff=False)
    assert record["players"][0]["global_player_id"] is None
    assert record["players"][0]["role"] == "unknown"
    assert record["players"][0]["pose_3d"] is None

    empty = build_phase1_frame_record(
        match_id="CCPL080626",
        delivery_id="DELIVERY_001",
        camera_id="cam_01",
        frame_index=1,
        frame_name="frame_camera01_000001.jpg",
        image_width=128,
        image_height=96,
        predictions=[],
    )
    validate_group1_frame(empty, final_handoff=False)
    assert empty["players"] == []


def test_crop_offset_and_nms():
    prediction = FakeAdapter().predict(None)[0]
    shifted = offset_prediction(prediction, x_offset=100, y_offset=200)
    assert shifted["bbox_xyxy"] == [110.0, 220.0, 150.0, 300.0]
    assert shifted["keypoints"][0] == [112.0, 222.0]

    duplicate = dict(shifted)
    duplicate["score"] = 0.5
    kept = nms_predictions([shifted, duplicate], iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0]["score"] == 0.8


def test_scale_prediction_restores_original_coordinates():
    prediction = FakeAdapter().predict(None)[0]
    scaled = scale_prediction(prediction, x_scale=4.0, y_scale=4.0)
    assert scaled["bbox_xyxy"] == [40.0, 80.0, 200.0, 400.0]
    assert scaled["keypoints"][0] == [48.0, 88.0]


def test_phase1_runner_processes_tiny_delivery(tmp_path: Path):
    drive_root = tmp_path / "drive"
    run_dir = tmp_path / "run"
    make_tiny_delivery(drive_root)
    metrics = run_phase1_delivery(
        P1RunConfig(
            drive_root=drive_root,
            delivery_id="DELIVERY_001",
            run_id="p1-test",
            run_dir=run_dir,
            model_id="fake_pose",
            device="cpu",
            frame_limit=1,
            cameras=["cam_01"],
            imgsz=640,
            conf=0.25,
            iou=0.7,
            half=False,
            show_progress=False,
            preload_full_frame=True,
            resize_long_side=64,
        ),
        FakeAdapter(),
    )

    assert metrics["summary"]["status"] == "pass"
    assert metrics["summary"]["records_written"] == 1
    assert metrics["input_mode"] == "opencv_resized_preload"
    assert metrics["imgsz"] == 640
    assert metrics["summary"]["decode_latency"]["count"] == 1
    prediction_path = run_dir / "predictions" / "cam_01.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    assert len(rows) == 1
    validate_group1_frame(rows[0], final_handoff=False)
    assert rows[0]["metadata"]["input_mode"] == "opencv_resized_preload"
    assert rows[0]["metadata"]["inference_image_size_px"] == [64, 48]
    assert rows[0]["metadata"]["coordinate_scale_xy"] == [2.0, 2.0]


def test_phase1_runner_appends_from_partial_resume(tmp_path: Path):
    drive_root = tmp_path / "drive"
    run_dir = tmp_path / "run"
    make_tiny_delivery(drive_root, frame_count=3)

    base_config = dict(
        drive_root=drive_root,
        delivery_id="DELIVERY_001",
        run_id="p1-test-resume",
        run_dir=run_dir,
        model_id="fake_pose",
        device="cpu",
        cameras=["cam_01"],
        imgsz=640,
        conf=0.25,
        iou=0.7,
        half=False,
        show_progress=False,
        preload_full_frame=True,
        resize_long_side=64,
    )
    first_metrics = run_phase1_delivery(
        P1RunConfig(
            **base_config,
            frame_limit=1,
        ),
        FakeAdapter(),
    )
    assert first_metrics["summary"]["records_written"] == 1

    prediction_path = run_dir / "predictions" / "cam_01.jsonl"
    first_line = prediction_path.read_text().splitlines()[0]

    second_metrics = run_phase1_delivery(
        P1RunConfig(
            **base_config,
            frame_limit=3,
        ),
        FakeAdapter(),
    )

    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    assert len(rows) == 3
    assert prediction_path.read_text().splitlines()[0] == first_line
    assert [row["frame_name"] for row in rows] == [
        "frame_camera01_000001.jpg",
        "frame_camera01_000002.jpg",
        "frame_camera01_000003.jpg",
    ]
    assert second_metrics["summary"]["records_written"] == 3
    assert second_metrics["summary"]["records_reused"] == 1
    assert second_metrics["summary"]["records_written_this_run"] == 2
    camera_metrics = second_metrics["per_camera"]["cam_01"]
    assert camera_metrics["records_reused"] == 1
    assert camera_metrics["records_written_this_run"] == 2
    assert camera_metrics["append_mode"] is True
