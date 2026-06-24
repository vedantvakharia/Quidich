"""Phase 2 parallel per-camera tracking orchestration (spawn-safe for win32)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_io import track_camera_file


def track_one_camera(args: tuple) -> tuple[str, str, dict, str | None]:
    (camera_id, input_path, output_path, diagnostics_path,
     delivery_id, config, expected_frames) = args
    try:
        diagnostics = track_camera_file(
            input_path, output_path, diagnostics_path,
            camera_id, delivery_id, config, expected_frames,
        )
        summary = {
            "frames_read": diagnostics["frames_read"],
            "confirmed_tracks": diagnostics["confirmed_tracks"],
            "total_tracks_spawned": diagnostics["total_tracks_spawned"],
        }
        return camera_id, "ok", summary, None
    except Exception as exc:  # noqa: BLE001 — surface per-camera failure, do not kill the pool
        return camera_id, "failed", {}, f"{type(exc).__name__}: {exc}"


def _discover_cameras(input_dir: Path, cameras: list[str] | None) -> list[str]:
    if cameras:
        return sorted(cameras)
    return sorted(p.stem for p in input_dir.glob("cam_*.jsonl"))


def run_p2_tracking(
    input_dir: str | Path,
    output_dir: str | Path,
    delivery_id: str,
    config: P2Config,
    cameras: list[str] | None = None,
    expected_frames: int = 600,
    max_workers: int | None = None,
) -> dict[str, tuple[str, dict, str | None]]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    camera_ids = _discover_cameras(input_dir, cameras)
    if not camera_ids:
        raise RuntimeError(f"no cam_XX.jsonl inputs found in {input_dir}")

    jobs = [
        (
            cam,
            input_dir / f"{cam}.jsonl",
            output_dir / f"{cam}.jsonl",
            output_dir / "diagnostics" / f"{cam}.json",
            delivery_id,
            config,
            expected_frames,
        )
        for cam in camera_ids
    ]
    workers = max_workers or min(7, os.cpu_count() or 1)

    results: dict[str, tuple[str, dict, str | None]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cam, status, summary, error in pool.map(track_one_camera, jobs):
            results[cam] = (status, summary, error)
    return results
