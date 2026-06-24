"""Base classes for model-specific benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pose_estimation.keypoints import map_keypoints
from pose_estimation.predictions import PredictionRecord


@dataclass(frozen=True)
class AdapterResult:
    keypoints: list[list[float | None]]
    source_skeleton: str
    bbox_xyxy: list[float | None]
    score: float | None
    timing_ms: dict[str, float]
    metadata: dict[str, Any]


class BaseAdapter:
    framework = "base"

    def __init__(self, model_id: str, model_config: dict[str, Any]) -> None:
        self.model_id = model_id
        self.model_config = model_config

    def predict_image(self, image_path: str | Path) -> list[AdapterResult]:
        raise NotImplementedError


def build_prediction_record(
    *,
    run_id: str,
    model_id: str,
    dataset_id: str,
    sample_id: str,
    image_id: str,
    person_id: str,
    result: AdapterResult,
) -> PredictionRecord:
    return PredictionRecord(
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        sample_id=sample_id,
        image_id=image_id,
        person_id=person_id,
        source_skeleton=result.source_skeleton,
        target_skeletons={
            "native": result.keypoints,
            "coco_17": map_keypoints(result.keypoints, result.source_skeleton, "coco_17"),
        },
        bbox_xyxy=result.bbox_xyxy,
        score=result.score,
        timing_ms=result.timing_ms,
        metadata=result.metadata,
    )

