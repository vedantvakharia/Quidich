# Getting started

This guide takes you from a fresh checkout to your first real benchmark result.
Every step ends with a **✓ Check** so you know it worked before moving on.

If you want the *why* behind any of this, read [concepts.md](concepts.md) alongside it.

## Prerequisites

You need:

- **Linux with an NVIDIA GPU** for real runs. (You can do most of the setup and the
  YOLO path on CPU, but speed numbers are only meaningful on GPU — see the note in the
  [README](../README.md).)
- **Conda** (Miniconda/Anaconda) on your `PATH`. Each model lives in its own Conda
  env; the tooling creates them for you.
- **Python 3** for the orchestrator itself, plus the small dependency set in
  [requirements.txt](../requirements.txt): `pip install -r requirements.txt`.
- Build tools only if you intend to run **OpenPose** (a C++/Caffe build) — `cmake`,
  a C++ compiler, and BLAS. Skip this unless you need OpenPose.

Quick environment probe (binaries, Python packages, GPU):

```bash
python3 scripts/check_environment.py
```

**✓ Check:** it prints your Python version and, if a GPU is visible, your CUDA device
under `nvidia-smi`. Missing entries here tell you what to install.

## Step 0 — Confirm the orchestrator runs

```bash
python3 -m pytest -q
python3 scripts/audit_repo.py --fail
```

**✓ Check:** tests pass and the audit prints `Repository hygiene audit passed`. You're
working from a clean, consistent checkout.

## Step 1 — Prepare environments and the dataset

This creates the per-model Conda envs, downloads model weights, and downloads COCO.
It's the longest step (weights + COCO are several GB).

```bash
# Everything (envs + assets + COCO). Add --download-large-assets for Sapiens2.
python3 scripts/benchmark.py prepare --models all --datasets coco17_val2017
```

Doing a lot the first time? Two smaller, faster paths:

```bash
# Just the YOLO path, to get to a result quickly:
python3 scripts/benchmark.py prepare --models yolo26x_pose --datasets coco17_val2017

# See what would run, without doing it:
python3 scripts/benchmark.py prepare --models all --datasets coco17_val2017 --dry-run
```

Weights land in local-only `models/<id>/weights/`; COCO lands in `data/raw/coco/`.
Neither is committed.

**✓ Check:** `data/raw/coco/val2017/` contains ~5000 `.jpg` files and
`data/raw/coco/annotations/person_keypoints_val2017.json` exists.

> Downloads can fail on some networks (notably `download.openmmlab.com`). That's
> expected and recoverable — see [troubleshooting.md](troubleshooting.md).

## Step 2 — Confirm assets are present

```bash
python3 scripts/check_assets.py --models all --fail-missing
```

**✓ Check:** exit code 0 and a table showing each required checkpoint as present. If a
model is missing weights, this tells you exactly which file and where it should go.

## Step 3 — Smoke test (one image per model)

Smoke pushes a single image through each model to prove the env, config, and
checkpoint load and produce keypoints. It is **not** a benchmark — it's a 30-second
"is this wired up?" check.

```bash
# One model first is the fastest signal:
python3 scripts/benchmark.py smoke --models yolo26x_pose

# Then everything:
python3 scripts/benchmark.py smoke --models all
```

Some models have device/size caveats:

```bash
python3 scripts/benchmark.py smoke --models openpose_body25 --device cpu
python3 scripts/benchmark.py smoke --models sapiens2_1b_pose --allow-heavy --device cuda:0
```

**✓ Check:** each model reports a passing status — `ok`, or `ready_heavy_skipped` /
`ready_runtime_limited` for the heavy models on a laptop. Results are logged to
`results/smoke_results.csv` (local scratch) and a per-model manifest under
`benchmarks/runs/<smoke_run_id>/`. A non-passing status points you at
[troubleshooting.md](troubleshooting.md).

## Step 4 — Run a real benchmark

Four models are benchmark-ready on `coco17_val2017`: **`yolo26x_pose`** (end-to-end) and
**`rtmw_l` / `rtmw_x` / `rtmpose_l_wholebody`** (top-down with GT boxes — see the
protocol note in [models.md](models.md)). We'll use `yolo26x_pose` here. Start small with
`--limit` to confirm the whole pipeline, then do the full set.

```bash
# A fast 100-image sanity run:
python3 scripts/benchmark.py run --models yolo26x_pose --datasets coco17_val2017 --limit 100 --batch-size 8

# The full COCO val (5000 images):
python3 scripts/benchmark.py run --models yolo26x_pose --datasets coco17_val2017 --batch-size 8
```

The run id is generated for you (timestamped, readable). If a run stops partway,
re-running the **same command with the same `--run-id`** resumes the remaining images
— it doesn't start over (details in [results-format.md](results-format.md)).

**✓ Check:** the command prints `Wrote immutable benchmark run: benchmarks/runs/<run_id>`
and that folder contains `run_manifest.json`, `hardware.json`, `software.json`, and
`metrics/yolo26x_pose__coco17_val2017.json` with real `coco_oks_ap` / latency numbers.

> Run a model without a benchmark adapter yet (anything outside the four benchmark-ready
> models) and the run still succeeds, but its metrics file is marked `adapter_pending`
> instead of holding scores. That's expected — see [models.md](models.md).

## Step 5 — Preview the comparison

`aggregate` collects every run folder's metrics into one CSV; `report` renders it as
HTML. On `main`, **CI does this for everyone** — but running it locally is the way to
preview your own numbers.

```bash
python3 scripts/benchmark.py aggregate
python3 scripts/benchmark.py report
```

**✓ Check:** `results/aggregate_metrics.csv` has a row per run, and
`benchmarks/reports/aggregate/index.html` opens as a table in your browser. Both files
are local/derived — you do **not** commit them (see [collaboration.md](collaboration.md)).

## You're done — now what?

- Ready to share results with the team? → [collaboration.md](collaboration.md)
- Want to benchmark a model that isn't here yet? → [adding-a-model.md](adding-a-model.md)
- Curious what each flag/script does? → [scripts.md](scripts.md)
