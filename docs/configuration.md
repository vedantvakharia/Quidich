# Configuration reference

Everything the framework does is driven from five YAML files in `configs/`. They are
the source of truth — committed, small, and the thing you edit to add a model or
dataset. This page explains each file's purpose and shows an annotated slice.

| File | Answers the question |
| ---- | -------------------- |
| [`model_registry.yaml`](#model_registryyaml) | *Which models exist, and what are they?* |
| [`model_envs.yaml`](#model_envsyaml) | *How do I install and feed each model?* |
| [`datasets.yaml`](#datasetsyaml) | *Which datasets exist, and where are they?* |
| [`keypoint_mappings.yaml`](#keypoint_mappingsyaml) | *How do I reduce any skeleton to COCO-17?* |
| [`benchmark_protocol.yaml`](#benchmark_protocolyaml) | *What do we measure, and what's a pass?* |

`model_registry.yaml` defines model **identity**; `model_envs.yaml` defines model
**operations**. They're keyed by the same `id`/model key, and together fully describe a
model. To add a model you touch both — see [adding-a-model.md](adding-a-model.md).

---

## `model_registry.yaml`

The catalog of models: identity, skeleton, role, and research metadata. One entry per
model under `models:`.

```yaml
models:
  - id: yolo26x_pose                 # stable key used everywhere (CLI, paths, configs)
    name: Ultralytics YOLO26x-pose   # human-readable name
    family: Ultralytics YOLO Pose
    role: fast_body17_fallback       # why it's here (candidate/teacher/reference/baseline)
    framework: ultralytics           # which adapter/runtime drives it
    skeleton: coco_17                # native output skeleton
    input_size: [640, 640]
    checkpoint: models/yolo26x_pose/weights/yolo26x-pose.pt
    export_targets: [onnx, tensorrt, openvino, coreml, tflite]
    license: AGPL-3.0 or enterprise license required for closed products
    status: verify_availability      # maturity/notes flag
    expected_strengths: [Production-friendly API and export path, Good fast baseline]
    expected_risks: [Body-only COCO-17, License must be checked for commercial deployment]
    sources: [https://docs.ultralytics.com/tasks/pose]
```

There's also a `default_target` block at the top (the production target: `nvidia_gpu`,
`fp16`, `coco_17`, and the latency/reprojection/MPJPE budgets).

## `model_envs.yaml`

How each model is **installed and run**: its Conda env, install profile, smoke image,
checkpoint, and downloadable assets. Two sections:

**`profiles:`** — reusable install recipes (one per ecosystem). Models reference a
profile by name so several models share an environment recipe.

```yaml
profiles:
  ultralytics_pose:
    python: "3.10"
    install:
      conda:
        channels: [pytorch, nvidia]
        packages: [pytorch==2.1.2, torchvision==0.16.2, pytorch-cuda=12.1, ...]
      pip:
        - ultralytics onnx onnxruntime-gpu pandas tqdm opencv-python<4.12 pycocotools
```

**`models:`** — per-model operational config.

```yaml
models:
  yolo26x_pose:
    env_name: cricket-yolo26x-pose   # the Conda env created for this model
    profile: ultralytics_pose        # which install recipe to use
    smoke_profile: ultralytics       # which smoke implementation in run_model_smoke.py
    checkpoint: models/yolo26x_pose/weights/yolo26x-pose.pt
    smoke_image: external/mmpose/.../human-pose.jpeg

  rtmw_l:
    env_name: cricket-rtmw-l
    profile: mmpose_v1
    smoke_profile: mmpose
    config: external/mmpose/configs/.../rtmw-l_8xb320-270e_cocktail14-384x288.py
    checkpoint: models/rtmw_l/weights/rtmw-dw-x-l_...pth
    assets:                          # what setup_model_envs.py downloads
      - kind: url                    # url | hf (file) | hf_repo (whole repo)
        url: https://download.openmmlab.com/.../rtmw-dw-x-l_...pth
        path: models/rtmw_l/weights/rtmw-dw-x-l_...pth
        required_for_smoke: true     # check_assets/smoke treat it as mandatory
```

Asset entries can carry `fallback_urls` (tried if the primary host fails — OpenPose
uses this) and `large: true` (only downloaded with `--download-large-assets`, e.g.
Sapiens2).

## `datasets.yaml`

The benchmark datasets: local paths, expected files, target skeleton, eval metric, and
download URLs. One entry per dataset under `datasets:`.

```yaml
datasets:
  coco17_val2017:
    name: COCO 2017 validation keypoints
    task: 2d_keypoint_detection
    target_skeleton: coco_17              # everything is compared in COCO-17
    root: data/raw/coco
    images: data/raw/coco/val2017
    annotation_file: data/raw/coco/annotations/person_keypoints_val2017.json
    keypoint_mapping: configs/keypoint_mappings.yaml
    eval_metric: COCO OKS AP/AR
    urls:
      images: http://images.cocodataset.org/zips/val2017.zip
      annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    expected: { images: 5000, annotation_file: person_keypoints_val2017.json }
```

Several datasets are **registered but not yet automated** (COCO-WholeBody, OCHuman,
CrowdPose, Human3.6M, MPI-INF-3DHP, and the internal `cricket_internal`). Only
`coco17_val2017` has an automated downloader and a full runner today — see
[datasets.md](datasets.md).

## `keypoint_mappings.yaml`

Defines the canonical **COCO-17** target skeleton (joint names + skeleton edges) and,
for every native skeleton, the index list that reduces it to COCO-17. This is what lets
a 133-keypoint whole-body model and a 25-keypoint OpenPose model be compared on the
same 17 joints.

```yaml
target_skeleton: coco_17
coco_17:
  keypoints: [nose, left_eye, right_eye, ..., left_ankle, right_ankle]   # the 17, in order
  skeleton_edges: [[nose, left_eye], [left_shoulder, right_shoulder], ...]

source_to_coco_17:
  coco_wholebody_133:        # body joints are the first 17 → identity slice
    source_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
  mediapipe_33:              # remap by semantic joint name
    source_indices: [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
  openpose_body25:           # BODY_25 → COCO-17 (neck/foot kept for native metrics only)
    source_indices: [0, 15, 16, 17, 18, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
```

**Rule:** every model must emit reduced `coco_17` keypoints through this mapping so the
comparison column is apples-to-apples. Native metrics (whole-body, BODY_25) can be
reported *in addition* but never *instead*.

## `benchmark_protocol.yaml`

The measurement contract: where results go, what counts as acceptable, how models are
scored, and the full metric inventory.

```yaml
result_storage:
  run_root: benchmarks/runs
  report_root: benchmarks/reports
  aggregate_csv: results/aggregate_metrics.csv
  immutable_per_run_files: true

acceptance_thresholds:                 # used by validate_results.py
  p95_end_to_end_latency_ms: 200
  mean_2d_reprojection_error_px: 10
  mean_3d_keypoint_error_mm: 25
  max_dropped_track_rate: 0.05

selection_weights:                     # used by score_models.py
  cricket_2d_accuracy: 0.40
  occlusion_robustness: 0.25
  latency: 0.20
  stability_jitter: 0.10
  integration_effort: 0.05

speed_protocol: { batch_size: 1, warmup_frames: 100, latency_percentiles: [50, 90, 95, 99], ... }
metric_inventory: { two_dimensional: [...], three_dimensional: [...], speed: [...], robustness: [...] }
```

The metric inventory here is the master list; [metrics.md](metrics.md) explains the
metrics in prose.
