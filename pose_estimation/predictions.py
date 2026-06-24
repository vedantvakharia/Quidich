"""Unified prediction schema helpers for benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "pose_predictions/v1"


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    model_id: str
    dataset_id: str
    sample_id: str
    image_id: str
    person_id: str
    source_skeleton: str
    target_skeletons: dict[str, list[list[float | None]]]
    bbox_xyxy: list[float | None]
    score: float | None
    timing_ms: dict[str, float] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "sample_id": self.sample_id,
            "image_id": self.image_id,
            "person_id": self.person_id,
            "source_skeleton": self.source_skeleton,
            "target_skeletons": self.target_skeletons,
            "bbox_xyxy": self.bbox_xyxy,
            "score": self.score,
            "timing_ms": self.timing_ms,
            "metadata": self.metadata,
        }


def validate_prediction_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "run_id",
        "model_id",
        "dataset_id",
        "sample_id",
        "image_id",
        "person_id",
        "source_skeleton",
        "target_skeletons",
        "bbox_xyxy",
        "score",
        "timing_ms",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"prediction record missing fields: {', '.join(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported prediction schema: {record['schema_version']}")
    if "native" not in record["target_skeletons"]:
        raise ValueError("prediction record must include target_skeletons.native")

