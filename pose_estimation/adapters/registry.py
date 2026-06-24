"""Adapter lookup by framework name."""

from __future__ import annotations

from pose_estimation.adapters.base import BaseAdapter


def get_adapter(framework: str) -> type[BaseAdapter]:
    if framework in {"mmpose", "mmpose_onnx", "ultralytics", "mediapipe", "sapiens2", "openpose"}:
        return BaseAdapter
    raise KeyError(f"Unknown adapter framework: {framework}")

