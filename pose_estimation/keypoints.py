"""Keypoint skeleton mapping utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = ROOT / "configs" / "keypoint_mappings.yaml"


def load_keypoint_mappings(path: str | Path = DEFAULT_MAPPING) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def map_keypoints(
    keypoints: list[list[float | None]] | np.ndarray,
    source_skeleton: str,
    target_skeleton: str = "coco_17",
    mapping_path: str | Path = DEFAULT_MAPPING,
) -> list[list[float | None]]:
    """Map an ``N x >=3`` keypoint array to the configured target skeleton."""

    mappings = load_keypoint_mappings(mapping_path)
    if target_skeleton != mappings.get("target_skeleton", target_skeleton):
        raise ValueError(f"Unsupported target skeleton: {target_skeleton}")
    source = mappings["source_to_coco_17"].get(source_skeleton)
    if source is None:
        raise KeyError(f"No mapping from {source_skeleton} to {target_skeleton}")

    array = np.asarray(keypoints, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("keypoints must have shape (N, >=3)")

    indices = source["source_indices"]
    output = np.full((len(indices), array.shape[1]), np.nan, dtype=float)
    for out_index, source_index in enumerate(indices):
        if source_index < array.shape[0]:
            output[out_index] = array[source_index]
    return _nullable(output[:, :3])


def _nullable(array: np.ndarray) -> list[list[float | None]]:
    result: list[list[float | None]] = []
    for row in array:
        result.append([None if not np.isfinite(value) else float(value) for value in row])
    return result

