# tests/test_cricket_p3_p4_contract.py
from __future__ import annotations
import pytest
from pose_estimation.cricket.contract import validate_group1_frame, SCHEMA_VERSION

def _base_frame(**overrides):
    kp_px = [[120.0 + i, 220.0 + i] for i in range(17)]
    kp_norm = [[x/2560, y/1440] for x, y in kp_px]
    player = {
        "global_player_id": "P001",
        "local_track_id": "cam_01_trk_0001",
        "role": "unknown",
        "bbox_xywh_px": [100.0, 200.0, 80.0, 240.0],
        "bbox_xywh_norm": [0.039, 0.138, 0.031, 0.166],
        "track_confidence": 0.85,
        "track_state": "confirmed",
        "single_camera": False,
        "pose_2d": {
            "skeleton": "coco_17",
            "keypoints_px": kp_px,
            "keypoints_norm": kp_norm,
            "confidence": [0.9] * 17,
        },
        "pose_3d": None,
    }
    player.update(overrides)
    return {
        "schema_version": SCHEMA_VERSION,
        "match_id": "CCPL080626",
        "delivery_id": "CCPL080626M1_1_14_1",
        "camera_id": "cam_01",
        "frame_index": 212334,
        "frame_name": "frame_camera01_000212334.jpg",
        "players": [player],
    }

def test_confirmed_player_validates():
    validate_group1_frame(_base_frame(), final_handoff=False)

def test_lost_player_allows_null_bbox_and_pose():
    """Fix 4.3: lost tracks cannot emit bbox or pose (no detection this frame)."""
    frame = _base_frame(
        track_state="lost",
        bbox_xywh_px=None,
        bbox_xywh_norm=None,
        pose_2d=None,
    )
    # Must not raise
    validate_group1_frame(frame, final_handoff=False)

def test_confirmed_player_requires_bbox():
    frame = _base_frame(track_state="confirmed", bbox_xywh_px=None)
    with pytest.raises(ValueError):
        validate_group1_frame(frame, final_handoff=False)

def test_track_confidence_non_negative():
    """Fix 4.1: confidence must not be negative."""
    frame = _base_frame(track_confidence=-0.1)
    with pytest.raises(ValueError):
        validate_group1_frame(frame, final_handoff=False)

def test_track_confidence_at_most_one():
    frame = _base_frame(track_confidence=1.01)
    with pytest.raises(ValueError):
        validate_group1_frame(frame, final_handoff=False)
