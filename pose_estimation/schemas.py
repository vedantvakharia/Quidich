"""Data classes for calibrated cameras and UE-ready pose packets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole camera calibration using a 3x4 projection matrix."""

    camera_id: str
    projection_matrix: np.ndarray
    calibration_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.projection_matrix, dtype=float)
        if matrix.shape != (3, 4):
            raise ValueError("projection_matrix must have shape (3, 4)")
        object.__setattr__(self, "projection_matrix", matrix)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CameraCalibration":
        return cls(
            camera_id=payload["camera_id"],
            projection_matrix=np.asarray(payload["projection_matrix"], dtype=float),
            calibration_id=payload.get("calibration_id"),
            metadata=payload.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "calibration_id": self.calibration_id,
            "projection_matrix": self.projection_matrix.tolist(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PosePacket:
    """Versioned output packet for HyperView/UE consumers."""

    frame_id: str
    timestamp_ns: int
    player_id: str
    model_version: str
    calibration_id: str | None
    keypoints3d_world_m: list[list[float | None]]
    keypoints3d_ue_cm: list[list[float | None]]
    confidence: list[float]
    visibility: list[float | None] = field(default_factory=list)
    occlusion_tags: list[str] = field(default_factory=list)
    keypoints2d: dict[str, list[list[float | None]]] = field(default_factory=dict)
    schema_version: str = "cricket_pose_packet/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "player_id": self.player_id,
            "model_version": self.model_version,
            "calibration_id": self.calibration_id,
            "keypoints2d": self.keypoints2d,
            "keypoints3d_world_m": self.keypoints3d_world_m,
            "keypoints3d_ue_cm": self.keypoints3d_ue_cm,
            "confidence": self.confidence,
            "visibility": self.visibility,
            "occlusion_tags": self.occlusion_tags,
        }


def keypoints_array(payload: Any, dims: int = 3) -> np.ndarray:
    """Convert a keypoint payload to a float array with a fixed trailing size."""

    array = np.asarray(payload, dtype=float)
    if array.ndim != 2 or array.shape[1] < dims:
        raise ValueError(f"keypoints must have shape (N, >= {dims})")
    return array[:, :dims]

