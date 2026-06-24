"""Reproducibility and CSV result helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


BENCHMARK_COLUMNS = [
    "model_id",
    "run_id",
    "benchmark_type",
    "dataset",
    "split",
    "skeleton",
    "num_keypoints",
    "precision",
    "backend",
    "hardware",
    "batch_size",
    "input_size",
    "cricket_2d_score",
    "occlusion_score",
    "latency_p50_ms",
    "latency_p95_ms",
    "fps_per_camera",
    "throughput_2cam_fps",
    "throughput_4cam_fps",
    "throughput_6cam_fps",
    "gpu_memory_mb",
    "jitter_score",
    "mean_reprojection_error_px",
    "mpjpe_mm",
    "p_mpjpe_mm",
    "dropped_track_rate",
    "integration_effort_score",
    "weighted_score",
    "passes_latency",
    "passes_2d",
    "passes_3d",
    "checkpoint_hash",
    "dataset_split_hash",
    "calibration_hash",
    "git_sha",
    "command",
    "notes",
    "created_at",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def git_sha(workdir: str | Path = ".") -> str | None:
    return command_output(["git", "-C", str(workdir), "rev-parse", "HEAD"])


def collect_environment(command: str | None = None, workdir: str | Path = ".") -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "git_sha": git_sha(workdir),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "cuda_visible_devices": command_output(["bash", "-lc", "printf '%s' \"${CUDA_VISIBLE_DEVICES:-}\""]),
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(str(Path(item)) for item in paths):
        digest.update(path.encode("utf-8"))
        digest.update(file_sha256(path).encode("utf-8"))
    return digest.hexdigest()


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def ensure_csv(path: str | Path, columns: list[str] = BENCHMARK_COLUMNS) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()


def append_result(path: str | Path, row: dict[str, Any], columns: list[str] = BENCHMARK_COLUMNS) -> None:
    ensure_csv(path, columns)
    normalized = {column: row.get(column, "") for column in columns}
    with Path(path).open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writerow(normalized)

