# Troubleshooting

Common failures and what to do. When in doubt, start with
`python3 scripts/check_environment.py` to see what's actually installed.

## Downloads

### `download.openmmlab.com` times out

The OpenMMLab host is unreliable on some networks — this is the single most common
setup snag. It affects the `.pth` checkpoints for RTMW, RTMO, and ViTPose. Options:

- Retry — it's often transient.
- Download the file from another network or a verified mirror, then drop it at the exact
  `path` listed for that asset in
  [`configs/model_envs.yaml`](configuration.md#model_envsyaml).
- Re-check with `python3 scripts/check_assets.py --models <id> --fail-missing`.

Some assets ship with a `fallback_urls` mirror in the config (OpenPose, RTMW-X) that the
downloader tries automatically.

### COCO download size

COCO val2017 is ~1 GB of images, plus a transient ~241 MB annotation zip. Use
`--remove-archives` to delete the `.zip` files after extraction (images/annotations are
kept).

## Source-control

### Checkpoints / weights show as untracked

**That's expected and correct.** `models/*/weights/` and `models/*/checksums/` are
ignored so multi-GB binaries and machine-specific checksums never get committed. The
metadata (`model.yaml`, `README.md`) is what's tracked. See
[collaboration.md](collaboration.md).

### `audit_repo.py` reports tracked artifacts

It found a weight, dataset, raw artifact, or derived file (`results/*.csv`,
`benchmarks/reports/*`) that was committed by mistake. Un-track it:
`git rm --cached <path>` (the file stays on disk), then commit. `.gitignore` should keep
it out going forward.

## Models

### Smoke passes but the benchmark is empty / says `adapter_pending`

Smoke is a one-image readiness check, not a benchmark. A model with no benchmark
adapter still produces a run folder, but its metrics file is marked `adapter_pending`
instead of holding scores. Four models are benchmark-ready on `coco17_val2017`
(`yolo26x_pose`, `rtmw_l`, `rtmw_x`, `rtmpose_l_wholebody`) — everything else is
smoke-only for now. See the readiness matrix in [models.md](models.md) and the recipe in
[adding-a-model.md](adding-a-model.md).

### OpenPose runtime missing (`missing_runtime`)

The CMU OpenPose binary isn't built. Build the CPU-only validation binary:

```bash
python3 scripts/setup_model_envs.py --models openpose_body25 --skip-assets
python3 scripts/setup_openpose.py --gpu-mode CPU_ONLY --jobs 2
python3 scripts/benchmark.py smoke --models openpose_body25 --device cpu
```

The expected binary path is `external/openpose/build/examples/openpose/openpose.bin`.

### OpenPose: "Atlas not found"

Bundled Caffe defaults to the Atlas BLAS library, but this Ubuntu machine has OpenBLAS
dev libraries instead. [`setup_openpose.py`](scripts.md#setup_openposepy) patches the
CMake config to pass `-DBLAS=Open` — so use the script rather than running CMake by hand
from a fresh clone.

### Sapiens2 says `ready_runtime_limited`

A **passing** laptop result: the Sapiens2 package, config, detector, and checkpoint are
all present, but full inference wasn't launched because CUDA isn't visible inside the
Sapiens env. Run the actual 1B inference on a larger GPU with adequate VRAM. See the
Sapiens2 note in [models.md](models.md#sapiens2-1b-sapiens2_1b_pose).

### ViTPose env errors after an upgrade

`vitpose_h` now targets MMPose v1. If its Conda env was built from the older legacy
ViTPose profile, rebuild or force-install it:
`python3 scripts/setup_model_envs.py --models vitpose_h --force-install`.

### DWPose can't find its ONNX files

Upstream DWPose expects the ONNX files inside its cloned repo. The canonical files live
in `models/dwpose_l_384/weights/`; compatibility symlinks point the upstream `ckpts/`
folder back to the store. Don't recreate the old top-level `checkpoints/` folder.

## Hardware / speed sanity

If speed numbers look implausible, check the run's `hardware.json` / `software.json` —
laptop and A100 numbers aren't comparable, CPU-only OpenPose timings are validation-only,
and precision/backend/batch-size all move latency. [metrics.md](metrics.md) lists what
has to match for two speed numbers to be comparable.
