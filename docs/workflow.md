# The benchmarking workflow

`scripts/benchmark.py` is the master entry point. It has five stages:

```
prepare → smoke → run → aggregate → report
```

You almost always go through `benchmark.py`. The lower-level scripts it calls are
documented in [scripts.md](scripts.md) and are useful for debugging, but benchmark
evidence should flow through the run folders this command creates.

A full pass, with the recommended asset check wedged in:

```bash
python3 scripts/benchmark.py prepare --models all --datasets coco17_val2017
python3 scripts/check_assets.py --models all --fail-missing
python3 scripts/benchmark.py smoke --models all
python3 scripts/benchmark.py run --models yolo26x_pose --datasets coco17_val2017 --limit 100 --batch-size 8
python3 scripts/benchmark.py aggregate
python3 scripts/benchmark.py report
```

---

## 1. prepare — set up envs, weights, and datasets

`prepare` does two things: creates each model's Conda env + downloads its weights
(via `setup_model_envs.py`), and downloads requested datasets (COCO via
`download_coco_keypoints.py`).

```bash
python3 scripts/benchmark.py prepare --models all --datasets coco17_val2017
```

Useful flags:

| Flag | Why |
| ---- | --- |
| `--models <ids> \| all` | Limit setup to specific models while iterating. |
| `--datasets <ids>` | Datasets to fetch. Automated today: `coco17_val2017`. |
| `--download-large-assets` | Also pull multi-GB assets (Sapiens2-1B + its DETR detector). Off by default so you don't download 6 GB by accident. |
| `--remove-archives` | Delete dataset `.zip` files after extraction (keeps images/annotations). |
| `--build-openpose` | Clone/configure/build the CMU OpenPose binary (only relevant with `openpose_body25`). |
| `--dry-run` | Print what would happen without doing it. Great for a first look. |

Downloads-only (no Conda work) is sometimes what you want on a machine that already
has the envs:

```bash
python3 scripts/setup_model_envs.py --models all --download-assets --download-large-assets --skip-envs
python3 scripts/download_coco_keypoints.py --remove-archives
```

**Output:** local-only `models/<id>/weights/` and `data/raw/`. Nothing committed.

## 2. smoke — prove each model is wired up

Smoke pushes **one image** through each model. It validates the env, config, and
checkpoint and confirms the model produces keypoints. It is intentionally tiny and is
**not benchmark evidence** — use it to catch setup problems before you spend time on a
full dataset.

```bash
python3 scripts/benchmark.py smoke --models all
python3 scripts/benchmark.py smoke --models openpose_body25 --device cpu
python3 scripts/benchmark.py smoke --models sapiens2_1b_pose --allow-heavy --device cuda:0
```

| Flag | Why |
| ---- | --- |
| `--device <cuda:0\|cpu>` | Where to run. OpenPose's laptop build is CPU-only. |
| `--allow-heavy` | Permit the heavyweight smoke path (Sapiens2-1B actually runs inference instead of just checking assets). |

**Reading the status** (written to `results/smoke_results.csv` and a per-model manifest
under `benchmarks/runs/<smoke_run_id>/`):

| Status | Meaning |
| ------ | ------- |
| `ok` | Loaded and produced keypoints. |
| `ready_heavy_skipped` | Assets/env ready; heavy inference skipped (no `--allow-heavy`). |
| `ready_runtime_limited` | Assets/env ready, but this machine can't run full inference (e.g. Sapiens2 with no CUDA in its env). Passing on a laptop. |
| `missing_assets` | A required checkpoint/file is absent — go back to `prepare`/`check_assets`. |
| `error` | Something failed; the manifest has details. |

## 3. run — the actual benchmark

`run` evaluates models on datasets and writes a compact, immutable **run folder**.

```bash
python3 scripts/benchmark.py run \
  --models yolo26x_pose \
  --datasets coco17_val2017 \
  --limit 100 \
  --batch-size 8 \
  --device cuda:0 \
  --imgsz 640 --conf 0.001 --iou 0.7
```

What to know:

- **Four models have a full benchmark adapter on `coco17_val2017`:** `yolo26x_pose`
  (end-to-end) and `rtmw_l` / `rtmw_x` / `rtmpose_l_wholebody` (top-down, GT boxes). Any
  other model/dataset still produces a run folder, but its metrics file is marked
  `adapter_pending` (or `dataset_missing`) rather than holding scores. The two protocols
  aren't directly comparable — see the readiness matrix and the protocol note in
  [models.md](models.md).
- **`--limit`** controls how many images are evaluated. Use a small number (e.g. 100)
  to verify the pipeline quickly; omit it for the full dataset. The limit is recorded
  in the run id (`scope-n100` vs `scope-full`) so a partial run is never mistaken for a
  full one.
- **The run id is generated** from models, datasets, scope, and a UTC timestamp. Pass
  `--run-id` only when you want a fixed name for a publication/handoff.
- **Resumable:** re-running the same command with the same `--run-id` continues the
  remaining images. `--no-resume` clears that model/dataset's local artifacts and
  starts fresh.

| Flag | Why |
| ---- | --- |
| `--limit N` / `--start-index N` | Evaluate N images / skip the first N. |
| `--batch-size`, `--imgsz`, `--conf`, `--iou` | Inference settings for the YOLO runner. |
| `--device` | `cuda:0` or `cpu`. |
| `--precision`, `--backend` | Recorded in the manifest (intended precision/backend). |
| `--run-id` | Fixed run-folder name (otherwise auto-generated). |
| `--no-resume` | Discard prior artifacts for this run/model/dataset and restart. |
| `--dry-run` | Write manifests/placeholders only. |

**Output:** `benchmarks/runs/<run_id>/` (committed) holds manifests + metrics JSON +
a summary HTML. Raw predictions and per-image logs go to local-only
`benchmarks/artifacts/<run_id>/`. Full layout: [results-format.md](results-format.md).

## 4 & 5. aggregate + report — the comparison view

`aggregate` scans every committed run folder under `benchmarks/runs/*/metrics/*.json`
and flattens them into one CSV. `report` renders that CSV as an HTML table.

```bash
python3 scripts/benchmark.py aggregate   # -> results/aggregate_metrics.csv
python3 scripts/benchmark.py report      # -> benchmarks/reports/aggregate/index.html
```

Both outputs are **derived and local** — one row per run, pure function of the run
folders. You run them to preview your own numbers, but you **don't commit them**:
on `main`, CI regenerates and publishes them so the shared view is always current and
nobody hand-merges a CSV. The full rationale and team flow is in
[collaboration.md](collaboration.md).

---

## Expert / manual scripts

These are not part of the main pipeline but exist for specific jobs (all detailed in
[scripts.md](scripts.md)):

- `benchmark_models.py` — write placeholder/hand-entered benchmark manifest rows.
- `score_models.py` — rank rows with the project's weighted selection formula.
- `validate_results.py` — check rows against latency/reprojection/MPJPE thresholds.
- `triangulate_predictions.py` / `export_ue_packets.py` — the 3D path: turn multi-view
  2D predictions into triangulated 3D, then into Unreal Engine pose packets.
