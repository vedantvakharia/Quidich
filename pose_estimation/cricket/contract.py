"""Group 1 player-output contract validation."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any


SCHEMA_VERSION = "g1_player_frame/v0"
SKELETON = "coco_17"
KEYPOINT_COUNT = 17
ROLE_VALUES = {
    "bowler",
    "striker",
    "non_striker",
    "wicketkeeper",
    "umpire",
    "fielder",
    "unknown",
}
CAMERA_RE = re.compile(r"^cam_0[1-7]$")


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_numeric_vector(
    value: Any,
    *,
    length: int,
    field_name: str,
    allow_none: bool = False,
) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field_name} must be a list of length {length}")
    for item in value:
        if not is_finite_number(item):
            raise ValueError(f"{field_name} contains a non-finite value")


def validate_points(
    value: Any,
    *,
    count: int,
    dims: int,
    field_name: str,
) -> None:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{field_name} must contain {count} points")
    for point in value:
        validate_numeric_vector(point, length=dims, field_name=f"{field_name} point")


def validate_pose_2d(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("pose_2d must be an object")
    if value.get("skeleton") != SKELETON:
        raise ValueError(f"pose_2d.skeleton must be {SKELETON}")
    validate_points(
        value.get("keypoints_px"),
        count=KEYPOINT_COUNT,
        dims=2,
        field_name="pose_2d.keypoints_px",
    )
    validate_points(
        value.get("keypoints_norm"),
        count=KEYPOINT_COUNT,
        dims=2,
        field_name="pose_2d.keypoints_norm",
    )
    validate_numeric_vector(
        value.get("confidence"),
        length=KEYPOINT_COUNT,
        field_name="pose_2d.confidence",
    )


def validate_pose_3d(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("pose_3d must be an object or null")
    validate_points(
        value.get("keypoints_world_m"),
        count=KEYPOINT_COUNT,
        dims=3,
        field_name="pose_3d.keypoints_world_m",
    )
    validate_numeric_vector(
        value.get("confidence"),
        length=KEYPOINT_COUNT,
        field_name="pose_3d.confidence",
    )
    validate_numeric_vector(
        value.get("mean_reprojection_error_px"),
        length=KEYPOINT_COUNT,
        field_name="pose_3d.mean_reprojection_error_px",
    )


def validate_player(player: Any, *, final_handoff: bool) -> None:
    if not isinstance(player, dict):
        raise ValueError("player must be an object")
    global_player_id = player.get("global_player_id")
    if final_handoff and not global_player_id:
        raise ValueError("global_player_id is required for final handoff")
    if global_player_id is not None and not isinstance(global_player_id, str):
        raise ValueError("global_player_id must be a string or null")
    role = player.get("role", "unknown")
    if role not in ROLE_VALUES:
        raise ValueError(f"invalid role: {role}")
    validate_numeric_vector(
        player.get("bbox_xywh_px"),
        length=4,
        field_name="bbox_xywh_px",
    )
    validate_numeric_vector(
        player.get("bbox_xywh_norm"),
        length=4,
        field_name="bbox_xywh_norm",
    )
    track_confidence = player.get("track_confidence")
    if track_confidence is not None and not is_finite_number(track_confidence):
        raise ValueError("track_confidence must be numeric or null")
    validate_pose_2d(player.get("pose_2d"))
    validate_pose_3d(player.get("pose_3d"))


def validate_group1_frame(record: dict[str, Any], *, final_handoff: bool = False) -> None:
    """Validate one Group 1 frame-level output record."""

    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(record.get("match_id"), str) or not record["match_id"]:
        raise ValueError("match_id is required")
    if not isinstance(record.get("delivery_id"), str) or not record["delivery_id"]:
        raise ValueError("delivery_id is required")
    camera_id = record.get("camera_id")
    if not isinstance(camera_id, str) or not CAMERA_RE.match(camera_id):
        raise ValueError("camera_id must be cam_01 through cam_07")
    if not isinstance(record.get("frame_index"), int):
        raise ValueError("frame_index must be an integer")
    if not isinstance(record.get("frame_name"), str) or not record["frame_name"]:
        raise ValueError("frame_name is required")
    players = record.get("players")
    if not isinstance(players, list):
        raise ValueError("players must be a list")
    for player in players:
        validate_player(player, final_handoff=final_handoff)


def example_group1_frame(*, final_handoff: bool = True) -> dict[str, Any]:
    keypoints_px = [[120.0 + index, 220.0 + index] for index in range(KEYPOINT_COUNT)]
    keypoints_norm = [[round(x / 2560.0, 6), round(y / 1440.0, 6)] for x, y in keypoints_px]
    keypoints_world = [
        [round(0.1 + index * 0.01, 3), round(8.2 + index * 0.01, 3), round(1.6, 3)]
        for index in range(KEYPOINT_COUNT)
    ]
    player = {
        "global_player_id": "P001" if final_handoff else None,
        "local_track_id": "cam_01_trk_0001",
        "role": "unknown",
        "bbox_xywh_px": [100.0, 200.0, 80.0, 240.0],
        "bbox_xywh_norm": [0.0390625, 0.1388889, 0.03125, 0.1666667],
        "track_confidence": 0.94,
        "pose_2d": {
            "skeleton": SKELETON,
            "keypoints_px": keypoints_px,
            "keypoints_norm": keypoints_norm,
            "confidence": [0.91 for _ in range(KEYPOINT_COUNT)],
        },
        "pose_3d": {
            "keypoints_world_m": keypoints_world,
            "confidence": [0.88 for _ in range(KEYPOINT_COUNT)],
            "mean_reprojection_error_px": [3.4 for _ in range(KEYPOINT_COUNT)],
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "match_id": "CCPL080626",
        "delivery_id": "CCPL080626M1_1_14_1",
        "camera_id": "cam_01",
        "frame_index": 212334,
        "frame_name": "frame_camera01_000212334.jpg",
        "players": [player],
    }


def contract_report() -> dict[str, Any]:
    final_example = example_group1_frame(final_handoff=True)
    intermediate_example = deepcopy(final_example)
    intermediate_example["players"][0]["global_player_id"] = None
    intermediate_example["players"][0]["pose_3d"] = None
    validate_group1_frame(final_example, final_handoff=True)
    validate_group1_frame(intermediate_example, final_handoff=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "skeleton": SKELETON,
        "keypoint_count": KEYPOINT_COUNT,
        "roles": sorted(ROLE_VALUES),
        "camera_ids": [f"cam_0{index}" for index in range(1, 8)],
        "canonical_coordinates": "full-frame pixels",
        "compatibility_coordinates": "normalized full-frame coordinates",
        "final_handoff_requires_global_player_id": True,
        "pose_3d_nullable_until_cross_camera_association": True,
        "real_player_names_in_scope": False,
        "valid_final_example": final_example,
        "valid_intermediate_example": intermediate_example,
        "summary": {"status": "pass"},
        "errors": [],
        "warnings": [],
    }

