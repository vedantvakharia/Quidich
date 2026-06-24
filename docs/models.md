# Models

The canonical list is [`configs/model_registry.yaml`](configuration.md#model_registryyaml)
(identity) plus [`configs/model_envs.yaml`](configuration.md#model_envsyaml) (envs and
assets). This page is the human view: what's in the registry, how ready each model is,
and the model-specific quirks worth knowing before you spend time on one.

## Smoke adapters vs. benchmark adapters

These are different things, and a model can have one without the other:

- A *smoke adapter* loads the model in its Conda env and runs one image, confirming the
  environment, config, and checkpoint are wired up. Implemented per framework in
  [`scripts/run_model_smoke.py`](scripts.md#run_model_smokepy). All 10 models have one.
- A *benchmark adapter* runs a full dataset and produces real scores (COCO OKS AP/AR,
  latency). Implemented as a `scripts/run_*_coco_benchmark.py` runner and enabled per
  model via the `benchmark_runner` field in `configs/model_envs.yaml`. A model with a
  smoke adapter but no benchmark adapter still runs under `benchmark.py run`, but writes
  an `adapter_pending` placeholder instead of metrics.

So readiness has two levels: smoke-ready (all 10) and benchmark-ready (the four below).
Downloaded assets (`check_assets.py`) are a separate precondition for both.

Benchmark-ready today: `yolo26x_pose`, `rtmw_l`, `rtmw_x`, `rtmpose_l_wholebody`.

## Readiness matrix

| Model | Framework | Native skeleton | Smoke | Benchmark (COCO-17) | Notes |
| ----- | --------- | --------------- | ----- | ------------------- | ----- |
| `yolo26x_pose` | Ultralytics | COCO-17 | yes | yes (end-to-end) | Detects and poses in one pass. |
| `rtmw_l` | MMPose | WholeBody-133 | yes | yes (top-down, GT bbox) | |
| `rtmw_x` | MMPose | WholeBody-133 | yes | yes (top-down, GT bbox) | Heaviest RTMW. |
| `rtmpose_l_wholebody` | MMPose | WholeBody-133 | yes | yes (top-down, GT bbox) | |
| `rtmo_l` | MMPose | COCO-17 | yes | pending | One-stage multi-person. |
| `dwpose_l_384` | MMPose/ONNX | WholeBody-133 | yes | pending | ONNX runtime; see DWPose note. |
| `vitpose_h` | MMPose v1 | COCO-17 | yes | pending | See ViTPose note. |
| `sapiens2_1b_pose` | Sapiens2 | WholeBody-308 | heavy | pending | Offline teacher; see Sapiens2 note. |
| `mediapipe_blazepose_heavy` | MediaPipe | MediaPipe-33 | yes | pending | Single-person reference. |
| `openpose_body25` | CMU OpenPose | BODY_25 | yes (CPU) | pending | C++/Caffe build; see OpenPose note. |

"Pending" means smoke works but `run` writes an `adapter_pending` metrics file rather
than real scores. The recipe to add one is in [adding-a-model.md](adding-a-model.md).

### Evaluation protocols

The benchmark-ready models do not all measure the same thing, and each run records its
protocol in `metrics.eval_protocol`:

- `end_to_end` (`yolo26x_pose`): the model does its own person detection and pose. This
  is the realistic deployment number.
- `topdown_gt_bbox` (`rtmw_l`, `rtmw_x`, `rtmpose_l_wholebody`): the model is given the
  ground-truth person boxes and only predicts keypoints. This isolates pose quality but
  removes all detection error, so the AP is inflated relative to end-to-end and is not
  directly comparable to `yolo26x_pose`.

Compare like with like by filtering on `eval_protocol` in the aggregate CSV. A fair
end-to-end comparison for the MMPose models would require pairing them with a person
detector, which is tracked as future work.

## The model store layout

Each model owns a directory; metadata is tracked, binaries are local:

```text
models/<model_id>/
  model.yaml     # generated metadata (tracked)
  README.md      # generated description (tracked)
  weights/       # checkpoints (LOCAL, ignored)
  configs/       # model-specific config (e.g. openpose_body25)
  checksums/     # hashes of local weights (LOCAL, ignored)
```

`model.yaml`, `README.md`, and `checksums/` are produced by
[`sync_model_store.py`](scripts.md#sync_model_storepy) — run it after assets change.
`external/` is only for upstream source clones; if an upstream repo hard-codes a
checkpoint path, use a compatibility symlink back into `models/<id>/weights/` rather
than duplicating the file or reviving the old top-level `checkpoints/` folder.

## Model-specific notes

### OpenPose (`openpose_body25`)

The canonical CMU OpenPose BODY_25 baseline — a C++/Caffe application, heavier to set up
than the Python models. The repo builds a **CPU-only** validation binary at
`external/openpose/build/examples/openpose/openpose.bin` via
[`setup_openpose.py`](scripts.md#setup_openposepy):

```bash
python3 scripts/setup_openpose.py --gpu-mode CPU_ONLY --jobs 2
python3 scripts/benchmark.py smoke --models openpose_body25 --device cpu
```

CPU-only is for **functional validation only** — never read speed numbers off it.
On benchmark machines, rebuild with `--gpu-mode CUDA`. Weights come from the official
CMU host first, with a Hugging Face mirror (`gaijingeek/openpose-models`) only as a
documented fallback because the CMU host is often unreachable.

### ViTPose-H (`vitpose_h`)

Targets the **MMPose v1** ViTPose-H COCO 256×192 config/checkpoint (a direct OpenMMLab
download), not the legacy standalone ViTPose distribution, because both config and
checkpoint are deterministic and script-downloadable. The benchmark identity is still
ViTPose-H; the runtime adapter is MMPose v1. If you have an old env built from the
legacy profile, rebuild/force-install it before smoking.

### Sapiens2-1B (`sapiens2_1b_pose`)

An **offline teacher candidate**, not a real-time production model. It needs a large
pose checkpoint plus a DETR detector (both `large: true`, so
`--download-large-assets`). On a laptop, smoke can legitimately return
`ready_runtime_limited`: assets and Python wiring are good, but full CUDA inference
isn't launched. That's a **passing** laptop result — run the actual 1B inference on a
bigger GPU with adequate VRAM.

### DWPose (`dwpose_l_384`)

Runs via ONNX. Upstream DWPose code expects its ONNX files inside the cloned repo; the
canonical files live in `models/dwpose_l_384/weights/` and compatibility symlinks point
the upstream `ckpts/` folder back to the store.

## Adding a new model

See [adding-a-model.md](adding-a-model.md) for the end-to-end recipe (registry entry →
env profile → assets → smoke → full adapter).
