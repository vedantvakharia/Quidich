"""Ultralytics YOLO pose adapter for cricket Phase 1 inference."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_model_config(model_id: str, config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    try:
        return config["models"][model_id]
    except KeyError as exc:
        raise ValueError(f"model_id not found in {config_path}: {model_id}") from exc


class YOLOPoseAdapter:
    """Thin wrapper around Ultralytics YOLO pose output."""

    def __init__(
        self,
        *,
        model_id: str,
        model_config: dict[str, Any],
        device: str = "cpu",
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.7,
        half: bool = False,
    ) -> None:
        from ultralytics import YOLO

        self.model_id = model_id
        self.model_config = model_config
        self.device = device
        self.ultralytics_device = self._ultralytics_device(device)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half
        model_path = model_config.get("model_name") or model_config.get("checkpoint")
        if model_path is None:
            raise ValueError(f"{model_id} has no model_name/checkpoint configured")
        self.model = YOLO(str(model_path))
        self._configure_torch_runtime()
        try:
            self.model.fuse()
        except Exception:
            pass

    @staticmethod
    def _ultralytics_device(device: str) -> str | int:
        if device.startswith("cuda:"):
            return int(device.split(":", 1)[1])
        if device.isdigit():
            return int(device)
        return device

    def _configure_torch_runtime(self) -> None:
        if not self.device.startswith("cuda"):
            return
        try:
            import torch

            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _inference_context(self):
        try:
            import torch

            return torch.inference_mode()
        except Exception:
            return nullcontext()

    def _result_to_predictions(self, result: Any) -> list[dict[str, Any]]:
        if result.boxes is None or result.keypoints is None:
            return []

        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.zeros(len(boxes_xyxy))
        keypoints_xy = result.keypoints.xy.detach().cpu().numpy()
        if getattr(result.keypoints, "conf", None) is not None:
            keypoint_conf = result.keypoints.conf.detach().cpu().numpy()
        elif getattr(result.keypoints, "data", None) is not None and result.keypoints.data.shape[-1] >= 3:
            keypoint_conf = result.keypoints.data.detach().cpu().numpy()[..., 2]
        else:
            keypoint_conf = np.ones(keypoints_xy.shape[:2], dtype=float)

        predictions = []
        for index, bbox in enumerate(boxes_xyxy):
            predictions.append(
                {
                    "bbox_xyxy": [float(value) for value in bbox.tolist()],
                    "score": float(scores[index]) if index < len(scores) else 0.0,
                    "keypoints": keypoints_xy[index].astype(float).tolist()
                    if index < len(keypoints_xy)
                    else [],
                    "keypoint_confidence": keypoint_conf[index].astype(float).tolist()
                    if index < len(keypoint_conf)
                    else [],
                    "model_id": self.model_id,
                }
            )
        return predictions

    def predict(self, image: str | Path | np.ndarray) -> list[dict[str, Any]]:
        return self.predict_batch([image], batch_size=1)[0]

    def predict_batch(
        self,
        images: list[str | Path | np.ndarray],
        *,
        batch_size: int = 8,
    ) -> list[list[dict[str, Any]]]:
        with self._inference_context():
            results = self.model(
                images,
                device=self.ultralytics_device,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                batch=batch_size,
                half=self.half,
            )
        return [self._result_to_predictions(result) for result in results]
