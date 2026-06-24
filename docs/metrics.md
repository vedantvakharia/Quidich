# Metrics

The full metric inventory is defined in
[`configs/benchmark_protocol.yaml`](configuration.md#benchmark_protocolyaml). This page
explains the groups in prose. The acceptance thresholds and selection weights that turn
metrics into pass/fail and rankings are in
[configuration.md](configuration.md#benchmark_protocolyaml).

## The golden rule: always report reduced COCO-17

Whatever a model's native skeleton, it **must** emit reduced `coco_17` keypoints via
[`configs/keypoint_mappings.yaml`](configuration.md#keypoint_mappingsyaml). The COCO-17
column is the apples-to-apples comparison across all models. Native-skeleton metrics
(whole-body, BODY_25, …) are reported **in addition**, never instead.

## 2D metrics

- **COCO OKS AP / AR** — the headline accuracy numbers (Object Keypoint Similarity,
  averaged over thresholds; the YOLO/COCO runner reports `ap`, `ap50`, `ap75`,
  medium/large variants, and `ar`).
- **PCK@0.05 / 0.10 / 0.20** — fraction of keypoints within a normalized distance.
- **mean pixel error**, **normalized mean error**, **per-keypoint error**.
- **detection rate**, **missing keypoint rate**, **confidence calibration error**.

## Native-capability metrics

Model-family specific, reported where dataset annotations support them:

- **body / foot / face / hand AP** (whole-body models on COCO-WholeBody).
- **native skeleton completeness** and **reduced COCO-17 score**.

Whole-body models report body/foot/face/hand; body-only models report completeness
against their own skeleton plus the reduced COCO-17 column.

## 3D metrics

For the triangulation path (multi-view → 3D):

- **MPJPE / PA-MPJPE** (mean per-joint position error, raw and Procrustes-aligned).
- **PCK3D**, **AUC**, **acceleration error**.
- **mean reprojection error (px)**, **triangulation success rate**, **per-joint 3D error**.

## Speed metrics

- **cold start**, **model load time**.
- **preprocess / inference / postprocess / end-to-end latency**.
- **p50 / p90 / p95 / p99 latency**, **FPS per camera**.
- **GPU/CPU memory peak**, **GPU utilization** where available.

> **Speed numbers are only comparable when their context matches.** Hardware, driver,
> CUDA/runtime, precision, backend, batch size, input size, and warmup policy all
> change the result. Every run records this in its manifests
> (see [results-format.md](results-format.md)) — which is exactly why a laptop number
> and an A100 number can't be compared without that context.

## Robustness metrics

Aimed at the cricket target: occlusion / motion-blur / low-light bucket scores,
multi-person crowd failure rate, temporal jitter, and dropped-track rate. The cricket
occlusion tags (gloves, helmet, pads, bat, ball, other players, …) are listed in
`benchmark_protocol.yaml`.
