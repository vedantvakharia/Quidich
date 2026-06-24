"""Model registry loading and filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_registry(path: str | Path = "configs/model_registry.yaml") -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or "models" not in payload:
        raise ValueError(f"{path} is not a valid model registry")
    return payload


def list_models(path: str | Path = "configs/model_registry.yaml", role: str | None = None) -> list[dict[str, Any]]:
    registry = load_registry(path)
    models = list(registry["models"])
    if role is not None:
        models = [model for model in models if model.get("role") == role]
    return models


def get_model(model_id: str, path: str | Path = "configs/model_registry.yaml") -> dict[str, Any]:
    for model in list_models(path):
        if model.get("id") == model_id:
            return model
    raise KeyError(f"Unknown model_id: {model_id}")

