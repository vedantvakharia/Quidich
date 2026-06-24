# tests/test_cricket_p2_runner.py
import json
from pathlib import Path

from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_runner import run_p2_tracking, track_one_camera

CFG = P2Config()


def _player(cx):
    kp = [[cx + i, 200 + 2 * i] for i in range(17)]
    return {
        "global_player_id": None, "local_track_id": None, "role": "unknown",
        "bbox_xywh_px": [cx - 20.0, 160.0, 40.0, 80.0],
        "bbox_xywh_norm": [0.0, 0.0, 0.02, 0.06], "track_confidence": 0.8,
        "pose_2d": {"skeleton": "coco_17", "keypoints_px": kp,
                     "keypoints_norm": [[x / 2560, y / 1440] for x, y in kp],
                     "confidence": [0.9] * 17},
        "pose_3d": None,
    }


def _frame(cam, idx, cx):
    return {
        "schema_version": "g1_player_frame/v0", "match_id": "CCPL080626",
        "delivery_id": "CCPL080626M1_1_14_1", "camera_id": cam, "frame_index": idx,
        "frame_name": f"frame_{cam}_{idx:09d}.jpg", "players": [_player(cx)],
    }


def _make_input(in_dir: Path, cam: str):
    in_dir.mkdir(parents=True, exist_ok=True)
    with (in_dir / f"{cam}.jsonl").open("w", encoding="utf-8") as h:
        for i in range(6):
            h.write(json.dumps(_frame(cam, i, 100 + 5 * i)) + "\n")


def test_track_one_camera_worker_succeeds(tmp_path: Path):
    in_dir = tmp_path / "p1" / "predictions"
    _make_input(in_dir, "cam_01")
    out_dir = tmp_path / "p2"
    cam, status, summary, err = track_one_camera((
        "cam_01", in_dir / "cam_01.jsonl", out_dir / "cam_01.jsonl",
        out_dir / "diagnostics" / "cam_01.json", "CCPL080626M1_1_14_1", CFG, 6,
    ))
    assert status == "ok"
    assert err is None
    assert summary["frames_read"] == 6


def test_run_p2_tracking_processes_multiple_cameras(tmp_path: Path):
    in_dir = tmp_path / "p1" / "predictions"
    for cam in ("cam_01", "cam_02"):
        _make_input(in_dir, cam)
    out_dir = tmp_path / "p2"
    results = run_p2_tracking(in_dir, out_dir, "CCPL080626M1_1_14_1", CFG,
                              expected_frames=6, max_workers=2)
    assert set(results) == {"cam_01", "cam_02"}
    for cam in ("cam_01", "cam_02"):
        assert results[cam][0] == "ok"
        assert (out_dir / f"{cam}.jsonl").exists()


def test_failed_camera_is_reported_not_raised(tmp_path: Path):
    in_dir = tmp_path / "p1" / "predictions"
    in_dir.mkdir(parents=True, exist_ok=True)
    (in_dir / "cam_01.jsonl").write_text("{ not valid json\n", encoding="utf-8")
    out_dir = tmp_path / "p2"
    results = run_p2_tracking(in_dir, out_dir, "CCPL080626M1_1_14_1", CFG,
                              expected_frames=1, max_workers=1)
    assert results["cam_01"][0] == "failed"
    assert results["cam_01"][2] is not None  # error string present
