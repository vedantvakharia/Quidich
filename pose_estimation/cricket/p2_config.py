"""Phase 2 per-camera tracking configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass(frozen=True)
class P2Config:
    # Stage thresholds
    stage1_confidence_threshold: float = 0.5
    stage2_confidence_min: float = 0.1
    cost_accept_threshold: float = 0.7
    lowconf_can_spawn: bool = True
    # Cost matrix weights
    iou_alpha: float = 0.6
    pose_beta: float = 0.4
    # Pose vector
    pose_keypoint_confidence_min: float = 0.3
    min_shared_keypoints: int = 6
    scale_min_frac_bbox_h: float = 0.05
    # Spatial / motion gating
    chi2_gate: float = 9.21
    gate_bbox_factor: float = 1.5
    gate_max_distance_px: float = 600.0
    v_max_px_per_frame: float = 120.0
    # Dormant re-ID
    pose_cosine_reid_threshold: float = 0.25
    reid_ambiguity_margin: float = 0.05
    dormant_max_frames: int = 60
    # Kalman stability
    kalman_cov_trace_max: float = 1.0e6
    # Track confirmation
    tentative_confirm_hits: int = 3
    tentative_confirm_window: int = 5
    # Gallery
    pose_gallery_size: int = 30
    gallery_repr: str = "medoid"


def load_p2_config(path: str | Path | None) -> P2Config:
    if path is None:
        return P2Config()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    type_hints = get_type_hints(P2Config)
    unknown = set(raw) - set(type_hints)
    if unknown:
        raise ValueError(f"unknown P2 config keys: {sorted(unknown)}")

    coerced: dict[str, Any] = {}
    for name, val in raw.items():
        field_type = type_hints[name]
        if field_type is float and not isinstance(val, float):
            coerced[name] = float(val)
        elif field_type is int and not isinstance(val, int):
            coerced[name] = int(val)
        elif field_type is bool and not isinstance(val, bool):
            coerced[name] = bool(val)
        else:
            coerced[name] = val
    return P2Config(**coerced)


