# tests/test_cricket_p2_cli.py
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_cricket_p2_tracking", ROOT / "scripts" / "run_cricket_p2_tracking.py"
)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


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


def test_arg_parser_has_required_options():
    parser = cli.build_arg_parser()
    args = parser.parse_args([
        "--input-dir", "outputs/p1/predictions", "--output-dir", "outputs/p2",
        "--delivery-id", "D1",
    ])
    assert args.input_dir == "outputs/p1/predictions"
    assert args.delivery_id == "D1"


def test_main_runs_end_to_end_single_camera(tmp_path: Path):
    in_dir = tmp_path / "p1" / "predictions"
    in_dir.mkdir(parents=True, exist_ok=True)
    with (in_dir / "cam_01.jsonl").open("w", encoding="utf-8") as h:
        for i in range(6):
            rec = {
                "schema_version": "g1_player_frame/v0", "match_id": "CCPL080626",
                "delivery_id": "D1", "camera_id": "cam_01", "frame_index": i,
                "frame_name": f"frame_camera01_{i:09d}.jpg", "players": [_player(100 + 5 * i)],
            }
            h.write(json.dumps(rec) + "\n")
    out_dir = tmp_path / "p2"
    code = cli.main([
        "--input-dir", str(in_dir), "--output-dir", str(out_dir),
        "--delivery-id", "D1", "--camera", "cam_01", "--max-workers", "1",
        "--expected-frames", "6",
    ])
    assert code == 0
    assert (out_dir / "cam_01.jsonl").exists()
