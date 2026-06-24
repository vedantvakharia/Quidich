"""Phase 1 cricket delivery inference runner."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Protocol

import cv2

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional outside the runtime env.
    tqdm = None

from pose_estimation.cricket.dataset import (
    frame_paths_for_camera,
    parse_frame_id,
    resolve_delivery_camera_dirs,
)
from pose_estimation.cricket.p1_outputs import (
    build_phase1_frame_record,
    nms_predictions,
    offset_prediction,
    scale_prediction,
)


class PoseAdapter(Protocol):
    model_id: str

    def predict(self, image: str | Path | Any) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class P1RunConfig:
    drive_root: Path
    delivery_id: str
    run_id: str
    run_dir: Path
    model_id: str
    device: str
    inference_mode: str = "full_frame"
    match_id: str = "CCPL080626"
    frame_limit: int | None = None
    start_index: int = 0
    cameras: list[str] | None = None
    crop_config_path: Path | None = None
    nms_iou_threshold: float = 0.6
    batch_size: int = 8
    resume: bool = True
    imgsz: int | None = None
    conf: float | None = None
    iou: float | None = None
    half: bool | None = None
    show_progress: bool = True
    preload_full_frame: bool = True
    resize_long_side: int | None = None


class NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def update(self, value: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def make_progress(*, total: int, desc: str, enabled: bool):
    if enabled and tqdm is not None and total > 0:
        return tqdm(total=total, desc=desc, unit="frame", dynamic_ncols=True)
    return NullProgress()


def configured_input_mode(config: P1RunConfig) -> str:
    if config.inference_mode == "crops":
        return "opencv_crop_array"
    if config.preload_full_frame:
        if config.resize_long_side and config.resize_long_side > 0:
            return "opencv_resized_preload"
        return "opencv_preload"
    return "opencv_array_batch"


@dataclass(frozen=True)
class DecodedFrame:
    path: Path
    image: Any
    original_size: tuple[int, int]
    input_size: tuple[int, int]
    scale_xy: tuple[float, float]
    decode_ms: float


@dataclass(frozen=True)
class ResumeState:
    existing_records: int
    existing_empty_frames: int
    existing_players: int
    is_complete: bool
    can_append: bool
    mismatch: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def latency_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "p50_ms": float(median(values)),
        "p95_ms": percentile(values, 0.95),
        "max_ms": float(max(values)),
    }


def git_sha(workdir: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workdir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def load_crop_windows(path: Path | None) -> dict[str, list[list[int]]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_crop_config_path(drive_root: Path) -> Path:
    return (
        drive_root
        / "dataset"
        / "calibration-data"
        / "CCPL080626"
        / "calibration_data"
        / "crop_mech.json"
    )


def select_frames(paths: list[Path], *, start_index: int, frame_limit: int | None) -> list[Path]:
    selected = paths[start_index:]
    if frame_limit is not None:
        selected = selected[:frame_limit]
    return selected


def read_batch_images(
    frame_paths: list[Path],
    *,
    resize_long_side: int | None = None,
) -> tuple[list[DecodedFrame], list[dict[str, Any]], float]:
    """Decode a batch with OpenCV before inference.

    Passing file paths directly to Ultralytics is substantially slower for this
    dataset because the high-level path loader becomes the bottleneck. OpenCV
    decode keeps the runner in control and still preserves frame names/ids.
    """

    start = time.perf_counter()
    decoded_frames: list[DecodedFrame] = []
    read_failures: list[dict[str, Any]] = []
    for frame_path in frame_paths:
        frame_start = time.perf_counter()
        frame = cv2.imread(str(frame_path))
        if frame is None:
            read_failures.append(
                {"frame_name": frame_path.name, "error": "cv2.imread failed"}
            )
            continue
        height, width = frame.shape[:2]
        input_frame = frame
        input_width = int(width)
        input_height = int(height)
        if resize_long_side and resize_long_side > 0:
            scale = float(resize_long_side) / float(max(width, height))
            if scale > 0 and scale != 1.0:
                input_width = max(1, int(round(width * scale)))
                input_height = max(1, int(round(height * scale)))
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                input_frame = cv2.resize(
                    frame,
                    (input_width, input_height),
                    interpolation=interpolation,
                )
        decoded_frames.append(
            DecodedFrame(
                path=frame_path,
                image=input_frame,
                original_size=(int(width), int(height)),
                input_size=(input_width, input_height),
                scale_xy=(float(width) / input_width, float(height) / input_height),
                decode_ms=(time.perf_counter() - frame_start) * 1000,
            )
        )
    decode_ms = (time.perf_counter() - start) * 1000
    return decoded_frames, read_failures, decode_ms


def batched(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def inspect_existing_jsonl(path: Path, expected_frames: list[Path]) -> ResumeState:
    if not path.exists():
        return ResumeState(
            existing_records=0,
            existing_empty_frames=0,
            existing_players=0,
            is_complete=False,
            can_append=True,
        )

    records = 0
    empty_frames = 0
    players = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                return ResumeState(
                    existing_records=records,
                    existing_empty_frames=empty_frames,
                    existing_players=players,
                    is_complete=False,
                    can_append=False,
                    mismatch=f"invalid JSONL at record {records + 1}: {exc}",
                )
            if records >= len(expected_frames):
                return ResumeState(
                    existing_records=records + 1,
                    existing_empty_frames=empty_frames,
                    existing_players=players,
                    is_complete=False,
                    can_append=False,
                    mismatch=(
                        f"existing JSONL has more records than selected frames "
                        f"({records + 1} > {len(expected_frames)})"
                    ),
                )
            expected_frame = expected_frames[records]
            if row.get("frame_name") != expected_frame.name:
                return ResumeState(
                    existing_records=records,
                    existing_empty_frames=empty_frames,
                    existing_players=players,
                    is_complete=False,
                    can_append=False,
                    mismatch=(
                        f"record {records + 1} frame_name={row.get('frame_name')!r} "
                        f"does not match expected {expected_frame.name!r}"
                    ),
                )
            expected_index = parse_frame_id(expected_frame)
            if expected_index is not None and row.get("frame_index") != expected_index:
                return ResumeState(
                    existing_records=records,
                    existing_empty_frames=empty_frames,
                    existing_players=players,
                    is_complete=False,
                    can_append=False,
                    mismatch=(
                        f"record {records + 1} frame_index={row.get('frame_index')!r} "
                        f"does not match expected {expected_index!r}"
                    ),
                )
            records += 1
            player_count = len(row.get("players", []))
            players += player_count
            if player_count == 0:
                empty_frames += 1
    return ResumeState(
        existing_records=records,
        existing_empty_frames=empty_frames,
        existing_players=players,
        is_complete=records == len(expected_frames),
        can_append=True,
    )


def ensure_jsonl_append_boundary(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) != b"\n":
            handle.write(b"\n")


def predict_full_frame_batch(
    adapter: PoseAdapter,
    images: list[Any],
    *,
    batch_size: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, float]]:
    start = time.perf_counter()
    if hasattr(adapter, "predict_batch"):
        predictions = adapter.predict_batch(images, batch_size=batch_size)  # type: ignore[attr-defined]
    else:
        predictions = [adapter.predict(image) for image in images]
    return predictions, {"inference_ms": (time.perf_counter() - start) * 1000}


def is_cuda_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "cuda out of memory" in text or "outofmemoryerror" in text


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def predict_crops(
    adapter: PoseAdapter,
    frame,
    *,
    crop_windows: list[list[int]],
    nms_iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    predictions: list[dict[str, Any]] = []
    inference_ms = 0.0
    for crop in crop_windows:
        x1, y1, x2, y2 = [int(value) for value in crop]
        image_crop = frame[y1:y2, x1:x2]
        if image_crop.size == 0:
            continue
        start = time.perf_counter()
        crop_predictions = adapter.predict(image_crop)
        inference_ms += (time.perf_counter() - start) * 1000
        predictions.extend(
            offset_prediction(prediction, x_offset=x1, y_offset=y1)
            for prediction in crop_predictions
        )
    before_nms = len(predictions)
    predictions = nms_predictions(predictions, iou_threshold=nms_iou_threshold)
    return predictions, {
        "inference_ms": inference_ms,
        "crop_count": len(crop_windows),
        "detections_before_nms": before_nms,
    }


def run_phase1_delivery(config: P1RunConfig, adapter: PoseAdapter) -> dict[str, Any]:
    """Run Phase 1 inference for one delivery and write JSONL predictions."""

    config.run_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = config.run_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    crop_windows_by_camera = load_crop_windows(
        config.crop_config_path or default_crop_config_path(config.drive_root)
    )

    camera_dirs = resolve_delivery_camera_dirs(config.drive_root, config.delivery_id)
    if config.cameras:
        wanted = set(config.cameras)
        camera_dirs = {camera_id: path for camera_id, path in camera_dirs.items() if camera_id in wanted}
    if not camera_dirs:
        raise RuntimeError(f"no camera directories found for {config.delivery_id}")

    selected_frames_by_camera = {
        camera_id: select_frames(
            frame_paths_for_camera(camera_dir),
            start_index=config.start_index,
            frame_limit=config.frame_limit,
        )
        for camera_id, camera_dir in camera_dirs.items()
    }
    resume_state_by_camera: dict[str, ResumeState] = {}
    progress_total = 0
    for camera_id, frames in selected_frames_by_camera.items():
        prediction_path = prediction_dir / f"{camera_id}.jsonl"
        if config.resume:
            resume_state = inspect_existing_jsonl(prediction_path, frames)
            if not resume_state.can_append:
                raise RuntimeError(
                    f"cannot resume {prediction_path}: {resume_state.mismatch}. "
                    "Use --no-resume to overwrite this prediction file."
                )
        else:
            resume_state = ResumeState(
                existing_records=0,
                existing_empty_frames=0,
                existing_players=0,
                is_complete=False,
                can_append=True,
            )
        resume_state_by_camera[camera_id] = resume_state
        progress_total += max(0, len(frames) - resume_state.existing_records)

    run_start = time.perf_counter()
    per_camera: dict[str, Any] = {}
    all_total_latencies: list[float] = []
    all_inference_latencies: list[float] = []
    all_decode_latencies: list[float] = []
    failures: list[dict[str, Any]] = []
    batch_fallbacks: list[dict[str, Any]] = []
    total_records = 0
    total_records_written_this_run = 0
    total_records_reused = 0
    total_players = 0
    progress = make_progress(
        total=progress_total,
        desc=f"P1 {config.delivery_id}",
        enabled=config.show_progress,
    )

    for camera_id, camera_dir in camera_dirs.items():
        frames = selected_frames_by_camera[camera_id]
        camera_number = camera_id.replace("cam_", "")
        camera_prediction_path = prediction_dir / f"{camera_id}.jsonl"
        camera_latencies: list[float] = []
        camera_inference_latencies: list[float] = []
        camera_decode_latencies: list[float] = []
        player_counts: list[int] = []
        empty_frames = 0
        resume_state = resume_state_by_camera[camera_id]
        resume_offset = resume_state.existing_records if config.resume else 0
        frames_to_process = frames[resume_offset:]

        if config.resume and resume_state.is_complete:
            per_camera[camera_id] = {
                "camera_dir": str(camera_dir),
                "prediction_jsonl": str(camera_prediction_path),
                "frames_selected": len(frames),
                "frames_to_process": 0,
                "records_written": resume_state.existing_records,
                "records_written_this_run": 0,
                "records_reused": resume_state.existing_records,
                "empty_detection_frames": resume_state.existing_empty_frames,
                "total_players_detected": resume_state.existing_players,
                "players_per_frame_mean": (resume_state.existing_players / resume_state.existing_records)
                if resume_state.existing_records
                else 0.0,
                "total_latency": latency_stats([]),
                "inference_latency": latency_stats([]),
                "decode_latency": latency_stats([]),
                "resumed_from_existing": True,
                "skipped_complete_camera": True,
            }
            total_records += resume_state.existing_records
            total_records_reused += resume_state.existing_records
            total_players += resume_state.existing_players
            continue

        if resume_offset > 0:
            ensure_jsonl_append_boundary(camera_prediction_path)
        open_mode = "a" if config.resume and resume_offset > 0 else "w"
        total_records += resume_offset
        total_records_reused += resume_offset
        total_players += resume_state.existing_players

        with camera_prediction_path.open(open_mode, encoding="utf-8") as handle:
            if config.inference_mode == "full_frame":
                resize_long_side = config.resize_long_side if config.resize_long_side and config.resize_long_side > 0 else None
                input_mode = (
                    "opencv_resized_preload"
                    if config.preload_full_frame and resize_long_side
                    else "opencv_preload"
                    if config.preload_full_frame
                    else "opencv_array_batch"
                )
                decoded_frames: list[DecodedFrame] = []
                if config.preload_full_frame:
                    preload_progress = make_progress(
                        total=len(frames_to_process),
                        desc=f"Preload {camera_id}",
                        enabled=config.show_progress,
                    )
                    for batch_paths in batched(frames_to_process, max(1, config.batch_size)):
                        decoded_batch, read_failures, _ = read_batch_images(
                            batch_paths,
                            resize_long_side=resize_long_side,
                        )
                        decoded_frames.extend(decoded_batch)
                        for decoded in decoded_batch:
                            camera_decode_latencies.append(decoded.decode_ms)
                            all_decode_latencies.append(decoded.decode_ms)
                        for failure in read_failures:
                            failures.append({"camera_id": camera_id, **failure})
                        if read_failures:
                            progress.update(len(read_failures))
                        preload_progress.update(len(batch_paths))
                    preload_progress.close()

                index = 0
                active_batch_size = max(1, config.batch_size)
                frame_source = decoded_frames if config.preload_full_frame else frames_to_process
                while index < len(frame_source):
                    frame_start = time.perf_counter()
                    if config.preload_full_frame:
                        decoded_batch = frame_source[index : index + active_batch_size]
                    else:
                        batch_paths = frame_source[index : index + active_batch_size]
                        decoded_batch, read_failures, _ = read_batch_images(
                            batch_paths,
                            resize_long_side=resize_long_side,
                        )
                        for decoded in decoded_batch:
                            camera_decode_latencies.append(decoded.decode_ms)
                            all_decode_latencies.append(decoded.decode_ms)
                        for failure in read_failures:
                            failures.append({"camera_id": camera_id, **failure})
                        if not decoded_batch:
                            progress.update(len(batch_paths))
                            index += len(batch_paths)
                            continue
                    batch_images = [decoded.image for decoded in decoded_batch]
                    try:
                        batch_predictions, timing = predict_full_frame_batch(
                            adapter,
                            batch_images,
                            batch_size=active_batch_size,
                        )
                    except Exception as exc:
                        if is_cuda_oom(exc) and active_batch_size > 1:
                            failed_batch_size = active_batch_size
                            active_batch_size = max(1, active_batch_size // 2)
                            clear_cuda_cache()
                            batch_fallbacks.append(
                                {
                                    "camera_id": camera_id,
                                    "frame_name": decoded_batch[0].path.name if decoded_batch else None,
                                    "from_batch_size": failed_batch_size,
                                    "to_batch_size": active_batch_size,
                                    "reason": "cuda_oom",
                                }
                            )
                            continue
                        for decoded in decoded_batch:
                            failures.append({"camera_id": camera_id, "frame_name": decoded.path.name, "error": str(exc)})
                        progress.update(len(decoded_batch))
                        index += len(decoded_batch)
                        continue
                    per_frame_inference_ms = float(timing.get("inference_ms", 0.0)) / max(1, len(decoded_batch))
                    for decoded, predictions in zip(decoded_batch, batch_predictions):
                        image_width, image_height = decoded.original_size
                        input_width, input_height = decoded.input_size
                        x_scale, y_scale = decoded.scale_xy
                        scaled_predictions = [
                            scale_prediction(prediction, x_scale=x_scale, y_scale=y_scale)
                            for prediction in predictions
                        ]
                        frame_id = parse_frame_id(decoded.path)
                        if frame_id is None:
                            failures.append({"camera_id": camera_id, "frame_name": decoded.path.name, "error": "invalid frame id"})
                            continue
                        record = build_phase1_frame_record(
                            match_id=config.match_id,
                            delivery_id=config.delivery_id,
                            camera_id=camera_id,
                            frame_index=frame_id,
                            frame_name=decoded.path.name,
                            image_width=image_width,
                            image_height=image_height,
                            predictions=scaled_predictions,
                            metadata={
                                "model_id": config.model_id,
                                "run_id": config.run_id,
                                "inference_mode": config.inference_mode,
                                "image_size_px": [image_width, image_height],
                                "inference_image_size_px": [input_width, input_height],
                                "coordinate_scale_xy": [x_scale, y_scale],
                                "batch_size_requested": config.batch_size,
                                "batch_size_effective": active_batch_size,
                                "input_mode": input_mode,
                                "imgsz": config.imgsz,
                                "resize_long_side": resize_long_side,
                                "conf": config.conf,
                                "iou": config.iou,
                                "half": config.half,
                            },
                        )
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                        total_records += 1
                        total_records_written_this_run += 1
                        player_count = len(record["players"])
                        total_players += player_count
                        player_counts.append(player_count)
                        if player_count == 0:
                            empty_frames += 1
                        per_frame_total_ms = decoded.decode_ms + per_frame_inference_ms
                        camera_latencies.append(per_frame_total_ms)
                        camera_inference_latencies.append(per_frame_inference_ms)
                        all_total_latencies.append(per_frame_total_ms)
                        all_inference_latencies.append(per_frame_inference_ms)
                    progress.update(len(decoded_batch))
                    elapsed = max(time.perf_counter() - run_start, 1e-9)
                    progress.set_postfix(
                        camera=camera_id,
                        batch=active_batch_size,
                        fps=f"{total_records / elapsed:.2f}",
                        players=total_players,
                    )
                    index += len(decoded_batch)
                per_camera[camera_id] = {
                    "camera_dir": str(camera_dir),
                    "prediction_jsonl": str(camera_prediction_path),
                    "frames_selected": len(frames),
                    "frames_to_process": len(frames_to_process),
                    "records_written": resume_offset + len(player_counts),
                    "records_written_this_run": len(player_counts),
                    "records_reused": resume_offset,
                    "empty_detection_frames": resume_state.existing_empty_frames + empty_frames,
                    "total_players_detected": resume_state.existing_players + sum(player_counts),
                    "players_per_frame_mean": (
                        (resume_state.existing_players + sum(player_counts))
                        / (resume_offset + len(player_counts))
                    )
                    if (resume_offset + len(player_counts))
                    else 0.0,
                    "total_latency": latency_stats(camera_latencies),
                    "inference_latency": latency_stats(camera_inference_latencies),
                    "decode_latency": latency_stats(camera_decode_latencies),
                    "resumed_from_existing": resume_offset > 0,
                    "skipped_complete_camera": False,
                    "append_mode": open_mode == "a",
                }
                continue

            for frame_path in frames_to_process:
                frame_start = time.perf_counter()
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    failures.append({"camera_id": camera_id, "frame_name": frame_path.name, "error": "cv2.imread failed"})
                    progress.update(1)
                    continue
                image_height, image_width = frame.shape[:2]
                frame_id = parse_frame_id(frame_path)
                if frame_id is None:
                    failures.append({"camera_id": camera_id, "frame_name": frame_path.name, "error": "invalid frame id"})
                    progress.update(1)
                    continue

                try:
                    if config.inference_mode == "crops":
                        crop_windows = crop_windows_by_camera.get(camera_number, [])
                        if not crop_windows:
                            crop_windows = [[0, 0, image_width, image_height]]
                        predictions, timing = predict_crops(
                            adapter,
                            frame,
                            crop_windows=crop_windows,
                            nms_iou_threshold=config.nms_iou_threshold,
                        )
                    else:
                        raise ValueError(f"unsupported inference_mode: {config.inference_mode}")

                    record = build_phase1_frame_record(
                        match_id=config.match_id,
                        delivery_id=config.delivery_id,
                        camera_id=camera_id,
                        frame_index=frame_id,
                        frame_name=frame_path.name,
                        image_width=image_width,
                        image_height=image_height,
                        predictions=predictions,
                        metadata={
                            "model_id": config.model_id,
                            "run_id": config.run_id,
                            "inference_mode": config.inference_mode,
                            "image_size_px": [image_width, image_height],
                            "batch_size_requested": config.batch_size,
                            "input_mode": "opencv_crop_array",
                            "imgsz": config.imgsz,
                            "conf": config.conf,
                            "iou": config.iou,
                            "half": config.half,
                        },
                    )
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    total_records += 1
                    total_records_written_this_run += 1
                    player_count = len(record["players"])
                    total_players += player_count
                    player_counts.append(player_count)
                    if player_count == 0:
                        empty_frames += 1
                    total_ms = (time.perf_counter() - frame_start) * 1000
                    inference_ms = float(timing.get("inference_ms", 0.0))
                    camera_latencies.append(total_ms)
                    camera_inference_latencies.append(inference_ms)
                    all_total_latencies.append(total_ms)
                    all_inference_latencies.append(inference_ms)
                except Exception as exc:
                    failures.append({"camera_id": camera_id, "frame_name": frame_path.name, "error": str(exc)})
                progress.update(1)
                elapsed = max(time.perf_counter() - run_start, 1e-9)
                progress.set_postfix(
                    camera=camera_id,
                    batch=1,
                    fps=f"{total_records / elapsed:.2f}",
                    players=total_players,
                )

        per_camera[camera_id] = {
            "camera_dir": str(camera_dir),
            "prediction_jsonl": str(camera_prediction_path),
            "frames_selected": len(frames),
            "frames_to_process": len(frames_to_process),
            "records_written": resume_offset + len(player_counts),
            "records_written_this_run": len(player_counts),
            "records_reused": resume_offset,
            "empty_detection_frames": resume_state.existing_empty_frames + empty_frames,
            "total_players_detected": resume_state.existing_players + sum(player_counts),
            "players_per_frame_mean": (
                (resume_state.existing_players + sum(player_counts))
                / (resume_offset + len(player_counts))
            )
            if (resume_offset + len(player_counts))
            else 0.0,
            "total_latency": latency_stats(camera_latencies),
            "inference_latency": latency_stats(camera_inference_latencies),
            "decode_latency": latency_stats(camera_decode_latencies),
            "resumed_from_existing": resume_offset > 0,
            "skipped_complete_camera": False,
            "append_mode": open_mode == "a",
        }

    progress.close()
    wall_clock_s = time.perf_counter() - run_start
    input_mode = configured_input_mode(config)
    metrics = {
        "schema_version": "cricket_phase1_metrics/v1",
        "run_id": config.run_id,
        "created_at": utc_now(),
        "delivery_id": config.delivery_id,
        "match_id": config.match_id,
        "model_id": config.model_id,
        "device": config.device,
        "inference_mode": config.inference_mode,
        "batch_size": config.batch_size,
        "imgsz": config.imgsz,
        "conf": config.conf,
        "iou": config.iou,
        "half": config.half,
        "input_mode": input_mode,
        "preload_full_frame": config.preload_full_frame,
        "resize_long_side": config.resize_long_side,
        "git_sha": git_sha(Path(__file__).resolve().parents[2]),
        "summary": {
            "camera_count": len(per_camera),
            "records_written": total_records,
            "records_written_this_run": total_records_written_this_run,
            "records_reused": total_records_reused,
            "total_players_detected": total_players,
            "failed_frames": len(failures),
            "wall_clock_s": wall_clock_s,
            "fps_overall": (total_records / wall_clock_s) if wall_clock_s > 0 else None,
            "total_latency": latency_stats(all_total_latencies),
            "inference_latency": latency_stats(all_inference_latencies),
            "decode_latency": latency_stats(all_decode_latencies),
            "status": "pass" if not failures else "partial",
        },
        "per_camera": per_camera,
        "failures": failures[:200],
        "batch_fallbacks": batch_fallbacks,
    }
    with (config.run_dir / "p1_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (config.run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": "cricket_phase1_run/v1",
                "run_id": config.run_id,
                "created_at": metrics["created_at"],
                "drive_root": str(config.drive_root),
                "delivery_id": config.delivery_id,
                "model_id": config.model_id,
                "device": config.device,
                "inference_mode": config.inference_mode,
                "batch_size": config.batch_size,
                "imgsz": config.imgsz,
                "conf": config.conf,
                "iou": config.iou,
                "half": config.half,
                "input_mode": input_mode,
                "preload_full_frame": config.preload_full_frame,
                "resize_long_side": config.resize_long_side,
                "prediction_dir": str(prediction_dir),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return metrics
