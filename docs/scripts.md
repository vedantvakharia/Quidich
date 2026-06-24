# Scripts reference

Every script in `scripts/`, what it's for, and how to call it. Most work goes through
the master CLI (`benchmark.py`); the rest are here for debugging and one-off jobs.

Quick map:

| Script | When | Purpose |
| ------ | ---- | ------- |
| [`benchmark.py`](#benchmarkpy) | constantly | Master CLI: `prepare`/`smoke`/`run`/`aggregate`/`report`. |
| [`setup_model_envs.py`](#setup_model_envspy) | via `prepare` | Create per-model Conda envs + download weights. |
| [`download_coco_keypoints.py`](#download_coco_keypointspy) | via `prepare` | Download + validate COCO 2017 val keypoints. |
| [`setup_openpose.py`](#setup_openposepy) | for OpenPose only | Clone + build the CMU OpenPose binary. |
| [`check_assets.py`](#check_assetspy) | before runs | Report which model checkpoints are present. |
| [`check_environment.py`](#check_environmentpy) | when debugging setup | Print local packages, binaries, and GPU. |
| [`smoke_model_envs.py`](#smoke_model_envspy) | via `smoke` | Orchestrate one-image readiness checks. |
| [`run_model_smoke.py`](#run_model_smokepy) | inside each env | Per-model smoke implementation. |
| [`run_yolo_coco_benchmark.py`](#run_yolo_coco_benchmarkpy) | via `run` | Full end-to-end YOLO/COCO-17 benchmark runner. |
| [`run_mmpose_coco_benchmark.py`](#run_mmpose_coco_benchmarkpy) | via `run` | Full top-down MMPose/COCO-17 runner (RTMW, RTMPose). |
| [`audit_repo.py`](#audit_repopy) | before every commit | Fail if local/generated artifacts are tracked. |
| [`sync_model_store.py`](#sync_model_storepy) | after asset changes | Regenerate per-model metadata/README/checksums. |
| [`benchmark_models.py`](#benchmark_modelspy) | rarely | Write placeholder/manual benchmark manifest rows. |
| [`score_models.py`](#score_modelspy) | rarely | Rank rows with the weighted selection formula. |
| [`validate_results.py`](#validate_resultspy) | rarely | Check rows against acceptance thresholds. |
| [`triangulate_predictions.py`](#triangulate_predictionspy) | 3D path | Multi-view 2D → triangulated 3D. |
| [`export_ue_packets.py`](#export_ue_packetspy) | 3D path | Triangulated 3D → Unreal Engine pose packets. |

---

## The pipeline scripts

### `benchmark.py`

The master CLI. Five subcommands, each covered in depth in [workflow.md](workflow.md).

- `prepare` — set up envs + download weights and datasets (delegates to
  `setup_model_envs.py`, `download_coco_keypoints.py`, optionally `setup_openpose.py`).
- `smoke` — one-image readiness checks (delegates to `smoke_model_envs.py`).
- `run` — full dataset benchmark into an immutable run folder (delegates to
  `run_yolo_coco_benchmark.py` or `run_mmpose_coco_benchmark.py` per model).
- `aggregate` — flatten all committed run metrics into `results/aggregate_metrics.csv`.
- `report` — render that CSV to `benchmarks/reports/aggregate/index.html`.

**Reads:** `configs/*.yaml`, committed run folders. **Writes:** run folders under
`benchmarks/runs/`, the aggregate CSV, and the HTML report. Full flag list per
subcommand is in [workflow.md](workflow.md).

### `setup_model_envs.py`

Creates one Conda environment per model from the install profile in
`configs/model_envs.yaml`, then optionally downloads that model's weights (from direct
URLs, with fallbacks, or Hugging Face). Writes an install stamp under
`.model_env_stamps/` so re-runs skip already-installed envs.

```bash
python3 scripts/setup_model_envs.py --models all --download-assets
python3 scripts/setup_model_envs.py --models sapiens2_1b_pose --download-assets --download-large-assets --skip-envs
```

| Flag | Meaning |
| ---- | ------- |
| `--models <ids>` | Which models (omit for all). |
| `--download-assets` | Fetch configured (non-large) assets. |
| `--download-large-assets` | Also fetch assets marked `large` (e.g. Sapiens2). |
| `--skip-envs` / `--skip-assets` | Do only downloads / only env creation. |
| `--force-install` | Reinstall packages even if a stamp exists. |
| `--dry-run` | Print commands without running. |

**Reads:** `configs/model_envs.yaml`. **Writes:** Conda envs, `models/<id>/weights/`,
`.model_env_stamps/`.

### `download_coco_keypoints.py`

Downloads, extracts, and validates the COCO 2017 validation keypoint set (annotations
+ 5000 images), then writes a manifest. The first reproducible 2D benchmark dataset.

```bash
python3 scripts/download_coco_keypoints.py --remove-archives
```

| Flag | Meaning |
| ---- | ------- |
| `--root` | Target dir (default `data/raw/coco`). |
| `--skip-images` | Annotations only. |
| `--force` | Re-download existing archives. |
| `--remove-archives` | Delete `.zip` files after extraction (keeps images/annotations). |

**Writes:** `data/raw/coco/{val2017/, annotations/, coco17_val2017_manifest.json}`.

### `setup_openpose.py`

Clones the official CMU OpenPose repo, applies an OpenBLAS build fix for this Ubuntu
setup, configures with CMake, and builds the `openpose.bin` binary. Only needed for the
`openpose_body25` model. The laptop build is CPU-only; build with `--gpu-mode CUDA` on
benchmark machines.

```bash
python3 scripts/setup_openpose.py --gpu-mode CPU_ONLY --jobs 2
```

| Flag | Meaning |
| ---- | ------- |
| `--gpu-mode CPU_ONLY\|CUDA\|OPENCL` | Build acceleration backend. |
| `--repo-dir` / `--build-dir` | Clone/build locations (default under `external/openpose`). |
| `--jobs` | Build parallelism. |
| `--dry-run` / `--skip-build` | Print only / clone+configure but don't compile. |

**Writes:** `external/openpose/` (local, never committed).

### `smoke_model_envs.py`

Orchestrates smoke checks across selected models: for each, it shells into the model's
Conda env and runs `run_model_smoke.py`, then records the status. Invoked by
`benchmark.py smoke`.

| Flag | Meaning |
| ---- | ------- |
| `--models`, `--device`, `--allow-heavy` | As in `benchmark.py smoke`. |
| `--results` | Smoke CSV path (default `results/smoke_results.csv`). |
| `--runs-dir` | Where per-model smoke manifests go. |

**Writes:** `results/smoke_results.csv` (local scratch) and
`benchmarks/runs/<smoke_run_id>/<model>_smoke.json`.

### `run_model_smoke.py`

The per-model smoke *implementation*, run **inside** a model's Conda env (you rarely
call it directly). One framework-specific path each for MMPose, DWPose, Ultralytics,
MediaPipe, ViTPose, Sapiens2, and OpenPose. Prints a JSON result with status, instance
count, keypoints, and latency.

| Flag | Meaning |
| ---- | ------- |
| `--model` | Required model id. |
| `--image` | Override the smoke image. |
| `--device`, `--allow-heavy` | As above. |

### `run_yolo_coco_benchmark.py`

The **end-to-end** full benchmark runner: batched, resumable YOLO pose inference over
COCO-17 (YOLO does its own person detection), then COCO OKS AP/AR evaluation and latency
percentiles. Invoked by `benchmark.py run` for `yolo26x_pose`. Records
`eval_protocol: end_to_end`.

### `run_mmpose_coco_benchmark.py`

The **top-down** full benchmark runner for the MMPose whole-body models (`rtmw_l`,
`rtmw_x`, `rtmpose_l_wholebody`). For each COCO image it runs MMPose `inference_topdown`
on the **ground-truth person boxes**, reduces the native 133 keypoints to COCO-17 via
[`configs/keypoint_mappings.yaml`](configuration.md#keypoint_mappingsyaml), then scores
with COCO OKS AP/AR. Records `eval_protocol: topdown_gt_bbox`.

> Because it uses GT boxes, its AP is **not comparable** to the end-to-end YOLO numbers —
> see the protocol note in [models.md](models.md#evaluation-protocols).

Both runners share image selection, resumability, and COCO OKS evaluation via
**`pose_estimation/coco_keypoint_eval.py`**. Each writes metrics JSON into the **run**
folder, raw predictions/logs into the **artifact** folder, and a resumable
`progress.json`. **Reads:** model checkpoint, COCO images + annotations. **Writes:**
`<run_dir>/metrics/<model>__<dataset>.json` plus `<artifact_dir>/predictions/…` and
`<artifact_dir>/logs/…`.

---

## Hygiene & maintenance

### `audit_repo.py`

**Run before every commit.** Checks `git ls-files` against a list of forbidden patterns
(weights, checksums, datasets, raw artifacts, derived `results/*.csv` and
`benchmarks/reports/*`) and fails if any were tracked by mistake. Placeholders and
`README.md` files are allowed.

```bash
python3 scripts/audit_repo.py --fail
```

### `check_assets.py`

Reports, per model, whether the required checkpoints exist locally (path, size, source)
— without downloading anything. Use it after `prepare` and before `run`.

```bash
python3 scripts/check_assets.py --models all --fail-missing
```

`--json` for machine-readable output; `--fail-missing` to exit non-zero on any gap.

### `check_environment.py`

No flags. Prints your Python version, key binaries (`nvidia-smi`, `nvcc`, `cmake`,
`ffmpeg`, …), key Python packages (torch, onnxruntime, mmpose, ultralytics, …), and GPU
info. First thing to run when setup misbehaves.

### `sync_model_store.py`

No flags. Regenerates the per-model store from `configs/model_registry.yaml` +
`configs/model_envs.yaml`: writes `models/<id>/model.yaml`, `README.md`, checksum JSON
(from local weights), and `.gitkeep` placeholders. Run after assets change so the
tracked metadata stays in sync. The `model.yaml`/`README.md` it writes are committed;
the checksums are local.

---

## Manual / expert scripts

These predate the run-folder pipeline and are kept for manual matrices and the 3D path.
Their CSV outputs are generated files under `results/` and don't replace run folders.

### `benchmark_models.py`

Writes reproducible benchmark **manifests** with placeholder metric rows (a scaffold for
hand-entered or not-yet-automated results).

| Flag | Meaning |
| ---- | ------- |
| `--models`, `--registry` | Models / registry path. |
| `--dataset`, `--split`, `--backend`, `--hardware` | Labels for the rows. |
| `--results`, `--runs-dir` | Output CSV / manifest dir. |
| `--dry-run` | Write placeholders only. |

### `score_models.py`

Ranks benchmark rows with the project's weighted selection formula (accuracy 0.40,
occlusion 0.25, latency 0.20, jitter 0.10, integration 0.05 — from
`benchmark_protocol.yaml`).

```bash
python3 scripts/score_models.py --input results/manual_benchmark_matrix.csv --output results/model_ranking.csv
```

### `validate_results.py`

Checks populated rows against acceptance thresholds (p95 latency ≤ 200 ms, reprojection
≤ 10 px, MPJPE ≤ 25 mm) and exits non-zero on any failure.

```bash
python3 scripts/validate_results.py --input results/manual_benchmark_matrix.csv
```

### `triangulate_predictions.py`

The 3D path, step 1: takes multi-view 2D keypoint predictions (JSONL) plus camera
calibration and triangulates 3D world coordinates with RANSAC outlier rejection.

| Flag | Meaning |
| ---- | ------- |
| `--predictions` | 2D JSONL (frame/camera/player/keypoints). |
| `--calibration` | Camera projection matrices JSON. |
| `--output` | Triangulated 3D JSONL. |
| `--reprojection-threshold-px` | RANSAC outlier threshold (default 10). |
| `--min-views` | Minimum cameras per 3D point (default 2). |

### `export_ue_packets.py`

The 3D path, step 2: converts triangulated 3D JSONL into Unreal Engine-ready pose
packets (JSONL), tagging each with a model version.

| Flag | Meaning |
| ---- | ------- |
| `--input` | Triangulated 3D JSONL. |
| `--output` | UE packet JSONL. |
| `--model-version` | Version label embedded in packets. |
