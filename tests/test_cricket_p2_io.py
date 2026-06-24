# tests/test_cricket_p2_io.py
import json
from pathlib import Path

from pose_estimation.cricket.contract import validate_group1_frame
from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_io import (
    frame_to_detections, read_p1_frames, track_camera_file,
)

CFG = P2Config()


def _player(cx, cy, conf=0.8):
    kp = [[cx + i, cy + 2 * i] for i in range(17)]
    return {
        "global_player_id": None, "local_track_id": None, "role": "unknown",
        "bbox_xywh_px": [cx - 20.0, cy - 40.0, 40.0, 80.0],
        "bbox_xywh_norm": [0.0, 0.0, 0.02, 0.06],
        "track_confidence": conf,
        "pose_2d": {"skeleton": "coco_17", "keypoints_px": kp,
                     "keypoints_norm": [[x / 2560, y / 1440] for x, y in kp],
                     "confidence": [0.9] * 17},
        "pose_3d": None,
    }


def _frame(frame_index, players):
    return {
        "schema_version": "g1_player_frame/v0", "match_id": "CCPL080626",
        "delivery_id": "CCPL080626M1_1_14_1", "camera_id": "cam_01",
        "frame_index": frame_index,
        "frame_name": f"frame_camera01_{frame_index:09d}.jpg",
        "players": players,
    }


def _write_jsonl(path: Path, frames):
    with path.open("w", encoding="utf-8") as h:
        for f in frames:
            h.write(json.dumps(f) + "\n")


def test_read_p1_frames_round_trips(tmp_path: Path):
    path = tmp_path / "cam_01.jsonl"
    _write_jsonl(path, [_frame(i, [_player(100, 200)]) for i in range(3)])
    frames = list(read_p1_frames(path))
    assert len(frames) == 3
    assert frames[0]["frame_index"] == 0


def test_frame_to_detections_builds_pose(tmp_path: Path):
    dets = frame_to_detections(_frame(0, [_player(100, 200)]), CFG)
    assert len(dets) == 1
    assert dets[0].pose.defined


def test_track_camera_file_fills_ids_and_validates(tmp_path: Path):
    in_path = tmp_path / "cam_01.jsonl"
    out_path = tmp_path / "out" / "cam_01.jsonl"
    diag_path = tmp_path / "out" / "diagnostics" / "cam_01.json"
    _write_jsonl(in_path, [_frame(i, [_player(100 + 5 * i, 200)]) for i in range(8)])

    diag = track_camera_file(in_path, out_path, diag_path, "cam_01",
                             "CCPL080626M1_1_14_1", CFG, expected_frames=8)

    out_frames = list(read_p1_frames(out_path))
    assert len(out_frames) == 8  # one output line per input frame
    for fr in out_frames:
        validate_group1_frame(fr)
    ids = {p["local_track_id"] for fr in out_frames for p in fr["players"] if p["local_track_id"]}
    assert len(ids) == 1  # single stable track
    assert diag["frames_read"] == 8
    assert diag_path.exists()


def test_output_line_count_equals_input_even_with_dropped_tracks(tmp_path: Path):
    in_path = tmp_path / "cam_01.jsonl"
    out_path = tmp_path / "out" / "cam_01.jsonl"
    diag_path = tmp_path / "out" / "diag" / "cam_01.json"
    # one isolated low-conf detection that never confirms -> id stays null, line still emitted
    frames = [_frame(0, [_player(100, 200, conf=0.2)])] + [_frame(i, []) for i in range(1, 4)]
    _write_jsonl(in_path, frames)
    track_camera_file(in_path, out_path, diag_path, "cam_01",
                      "CCPL080626M1_1_14_1", CFG, expected_frames=4)
    out_frames = list(read_p1_frames(out_path))
    assert len(out_frames) == 4
    assert out_frames[0]["players"][0]["local_track_id"] is None
