# Datasets

Dataset definitions live in [`configs/datasets.yaml`](configuration.md#datasetsyaml).
Everything is ultimately compared in **COCO-17**: whatever a dataset's native skeleton,
model outputs are reduced to COCO-17 via
[`configs/keypoint_mappings.yaml`](configuration.md#keypoint_mappingsyaml) before
keypoint metrics are computed.

## The tiered suite (and what's automated)

The plan is a ladder from "easy public baseline" to "real cricket." Only the first rung
is automated today; the rest are registered so the framework is ready for them.

| Dataset | Purpose | Status |
| ------- | ------- | ------ |
| `coco17_val2017` | First public 2D body-keypoint baseline | automated downloader + runner |
| `coco_wholebody_val` | Native whole-body (133) evaluation | registered, manual data |
| `ochuman_val` | Occlusion robustness (overlapping people) | registered, manual data |
| `crowdpose_val` | Crowded multi-person scenes | registered, manual data |
| `human36m`, `mpi_inf_3dhp` | 3D pose (MPJPE/PA-MPJPE) where licensing permits | registered, manual data |
| `cricket_internal` | The real target: calibration + occlusion labels | future, needs locked splits |

"Manual data" means the dataset is defined in config (paths, expected files, eval
metric) but you supply the files yourself; there's no automated downloader yet.

## Why this ladder

Each rung measures something COCO-17 can't:

- **COCO-17** is a clean, reproducible starting point — but it says nothing about
  whole-body accuracy, occlusion, crowds, or 3D.
- **COCO-WholeBody** is the native test for the whole-body models (RTMW, RTMPose-WB,
  DWPose) — report body/foot/face/hand AP there, plus reduced COCO-17 for comparison.
- **OCHuman / CrowdPose** stress occlusion and player overlap, which is exactly what
  cricket equipment and fielders create.
- **Human3.6M / MPI-INF-3DHP** are 3D references for the triangulation path.
- **`cricket_internal`** is the final domain benchmark — added once the public-dataset
  framework is stable, and it needs locked splits, camera calibration, and occlusion
  tags before it's meaningful.

## COCO-17 — the one you'll actually use

A standard 17-joint 2D person-keypoint benchmark (5000 val images). Good first
reproducible baseline; **not** sufficient on its own for cricket, whole-body, occlusion,
or 3D claims.

Download + validate (≈1 GB of images, plus a transient ≈241 MB annotation zip):

```bash
python3 scripts/download_coco_keypoints.py --remove-archives
```

This lands `data/raw/coco/{val2017/, annotations/, coco17_val2017_manifest.json}` — all
local, none committed. `--remove-archives` deletes the downloaded `.zip` files after
extraction (it keeps the images and annotations).
