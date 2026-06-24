# Results format

A benchmark writes results in **two layers**: compact evidence you commit, and bulky raw
artifacts you keep local. This page is the reference for both, plus the run-id scheme,
resumability, and the hardware/software metadata every run records.

## The two layers

```text
benchmarks/runs/<run_id>/              # COMMITTED — compact, immutable evidence
  run_manifest.json                    #   command, models, datasets, precision, schema
  hardware.json                        #   GPU/CPU/RAM of the machine
  software.json                        #   package versions + git SHA
  model_manifest.json                  #   model identity/checkpoint info
  dataset_manifest.json                #   dataset paths + annotation hash
  metrics/
    <model_id>__<dataset_id>.json      #   THE scores (+ eval image ids)
  visualizations/
    summary.html                       #   per-run HTML summary

benchmarks/artifacts/<run_id>/         # LOCAL ONLY — ignored by git
  predictions/
    <model_id>__<dataset_id>.jsonl             # raw per-instance keypoints
    <model_id>__<dataset_id>.coco_keypoints.json
  logs/
    <model_id>__<dataset_id>.progress.json     # resumability state
    <model_id>__<dataset_id>.latency.jsonl     # per-image timings
```

Commit a `benchmarks/runs/<run_id>/` folder when it holds real evidence worth reviewing
or merging. Never commit `benchmarks/artifacts/<run_id>/` — it's the raw prediction/log
dump, too large and unnecessary for cross-machine comparison. The split is enforced by
[`audit_repo.py`](scripts.md#audit_repopy).

## The metrics JSON

`metrics/<model_id>__<dataset_id>.json` is the heart of a run. It carries the model/
dataset identity, a `status`, the `metrics` block (the actual numbers — COCO OKS AP/AR,
latency percentiles, FPS, detection rate), and a `dataset` block with the annotation
hash and the list of evaluated image ids. (That image-id list is most of the file's
size; the numbers themselves are a few dozen fields.)

The `metrics` block also records **`eval_protocol`**: `end_to_end` (the model did its
own detection, e.g. `yolo26x_pose`) or `topdown_gt_bbox` (the model was given
ground-truth person boxes, e.g. the MMPose models). AP from the two protocols is **not
directly comparable** — filter on `eval_protocol` before ranking. See
[models.md](models.md#evaluation-protocols).

`status` tells you what kind of result it is:

| status | meaning |
| ------ | ------- |
| `ok` | A real, complete benchmark. |
| `adapter_pending` | No full adapter for this model/dataset yet — placeholder, no scores. |
| `dataset_missing` | The dataset isn't present locally. |
| `runner_failed` | The runner errored; placeholder written. |
| `dry_run` | Written by `--dry-run`. |

## Derived aggregates (not committed)

`aggregate` + `report` turn the committed run folders into:

```text
results/aggregate_metrics.csv          # one row per run (derived)
results/model_ranking.csv              # score_models.py output (derived)
benchmarks/reports/aggregate/index.html# the CSV rendered as a table (derived)
```

These are **generated, not authored** — you run them to preview locally, but CI builds
and publishes the shared copies on `main`. Don't commit them. See
[collaboration.md](collaboration.md#why-ci-owns-the-aggregates).

## Run ids

Generated ids are readable and unique:

```text
models-<model_ids>__bench-<dataset_ids>__scope-<scope>__<YYYYMMDDTHHMMSSZ>
```

```text
models-yolo26x_pose__bench-coco17_val2017__scope-full5000__20260609T215042Z
models-yolo26x_pose__bench-coco17_val2017__scope-n100__20260610T061500Z
```

`scope` encodes the evaluation extent (`full`, `n100`, `start200-n100`) so a partial run
is never mistaken for a full one. Pass `--run-id` only for a fixed publication/handoff
name.

## Resumability

The full runners write `benchmarks/artifacts/<run_id>/logs/<...>.progress.json`. Reusing
the same `--run-id` resumes the missing images **as long as the local artifact directory
is still present** — e.g. a run that stopped at 80/100 processes only the remaining 20.
Pass `--no-resume` to clear that model/dataset's local predictions, latency logs,
progress, and metrics before starting again.

The benchmark-ready full-run paths are `yolo26x_pose`, `rtmw_l`, `rtmw_x`, and
`rtmpose_l_wholebody` on `coco17_val2017`.

## Hardware & software metadata

Every run records, in `hardware.json` / `software.json` / `run_manifest.json`:

- timestamp, command, git SHA (when available)
- OS / platform
- Python executable and package list
- GPU name, driver, memory; CUDA visibility
- model checkpoint hash, dataset annotation hash

This is non-negotiable because **a laptop functional check and an A100 performance run
are not directly comparable** without this context. The helpers live in
`pose_estimation/hardware/`. Always include this metadata when comparing results across
machines.
