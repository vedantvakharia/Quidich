# Reducing noise and improving model outputs

Picking a model is the starting point, not the goal. Off-the-shelf 2D keypoints are
noisy: they jitter frame to frame, drop out under occlusion, and produce false or
swapped detections in crowded scenes. The production value comes from improving those
outputs: filtering bad detections, fusing multiple cameras, smoothing over time, and
fine-tuning on the target domain. This page explains how the repo supports that work.

The benchmark harness is the measurement tool. The improvement work is a loop:

```
baseline run  ->  apply an intervention  ->  re-measure  ->  compare
                  (filter / fuse /            (same metrics)   (did jitter and
                   smooth / fine-tune)                          error go down?)
```

Every intervention should be proven, not assumed. Run a baseline, apply the change,
run again, and compare the committed numbers. That is what the immutable run folders,
aggregation, and report exist for.

## What the repo gives you

### 1. Measuring noise and quality

Implemented and tested in [`pose_estimation/metrics.py`](../pose_estimation/metrics.py):

- `temporal_jitter(sequence_xyz, fps)`: mean frame-to-frame joint displacement. The core
  "how noisy is this" number.
- `reprojection_error(...)`: how far a triangulated 3D point lands from its 2D
  observations. Low reprojection error means consistent multi-view geometry.
- `mpjpe` / `p_mpjpe`: 3D position error against ground truth (raw and Procrustes-aligned).
- `weighted_model_score(...)`: the selection score that already weights
  `stability_jitter` (0.10) alongside accuracy, so a less jittery model scores higher.

The full robustness inventory (occlusion, motion-blur, low-light buckets, dropped-track
rate, temporal jitter) and the acceptance thresholds (reprojection <= 10 px, MPJPE
<= 25 mm) live in [`configs/benchmark_protocol.yaml`](configuration.md#benchmark_protocolyaml).

### 2. Reducing noise

Implemented and tested in [`pose_estimation/triangulation.py`](../pose_estimation/triangulation.py):

- **Multi-view geometric denoising.** `ransac_triangulate_point(...)` and
  `triangulate_skeleton_ransac(...)` triangulate each joint with pairwise RANSAC and
  re-fit on inlier views, rejecting any 2D observation whose reprojection error exceeds
  `reprojection_threshold_px`. This removes per-camera detection noise using the geometry
  of the other cameras. Exposed end to end by
  [`scripts/triangulate_predictions.py`](scripts.md#triangulate_predictionspy)
  (`--reprojection-threshold-px`, `--min-views`).
- **Temporal smoothing.** `confidence_ema_smooth(sequence_xyz, confidences, alpha)`
  applies confidence-aware exponential smoothing to a `T x J x 3` sequence: high-confidence
  frames update quickly, low-confidence frames lean on the previous estimate, and missing
  joints are carried forward. This is the direct lever on `temporal_jitter`.

### 3. Making the models better

- **Teachers for pseudo-labeling and distillation.** `vitpose_h` (accuracy teacher) and
  `sapiens2_1b_pose` (offline teacher) are in the registry precisely so you can generate
  high-quality labels on hard cricket frames and distill them into a faster student.
- **Domain fine-tuning.** `fine_tuning_augmentations` in
  [`configs/benchmark_protocol.yaml`](configuration.md#benchmark_protocolyaml) lists the
  augmentations aimed at cricket noise (equipment-occlusion overlay, motion blur, partial
  crops, low-confidence masking). Fine-tune, then re-benchmark to prove the gain.

## Where denoising sits in the pipeline

For multi-camera cricket production the path is:

```
2D keypoints (per camera)
   -> per-view confidence gating / filtering
   -> multi-view triangulation  (RANSAC, geometric outlier rejection)  [triangulation.py]
   -> temporal smoothing         (confidence-aware EMA)                 [triangulation.py]
   -> 3D pose
   -> Unreal Engine export                                             [export_ue_packets.py]
```

Each arrow is a place to measure jitter and reprojection error before and after, and to
try a better method.

## What exists vs. what you will build

Honest status, so nobody assumes more than is there:

- **Built and tested:** the triangulation, RANSAC outlier rejection, confidence-aware
  smoothing, and the jitter / reprojection / MPJPE metrics above
  ([`tests/test_triangulation.py`](../tests/test_triangulation.py),
  [`tests/test_metrics.py`](../tests/test_metrics.py)).
- **Your work:** wiring these into an end-to-end "denoise then re-benchmark" flow on
  cricket multi-view video, and improving the methods themselves (better filters, learned
  smoothing, occlusion handling, fine-tuned students). The temporal and multi-view metrics
  need video sequences with calibration, which the 2D COCO still-image benchmarks do not
  have; that is the `cricket_internal` dataset and calibration described in
  [datasets.md](datasets.md). The 2D COCO benchmarks give you the per-frame model-quality
  baseline to build on.

## A concrete improvement loop

Multi-view predictions in hand, triangulate with two outlier thresholds and compare the
reprojection error and triangulation success rate:

```bash
python3 scripts/triangulate_predictions.py \
  --predictions preds.jsonl --calibration cameras.json \
  --output tri_loose.jsonl --reprojection-threshold-px 12

python3 scripts/triangulate_predictions.py \
  --predictions preds.jsonl --calibration cameras.json \
  --output tri_tight.jsonl --reprojection-threshold-px 6
```

Then, in a short script, load a sequence, measure `temporal_jitter` before smoothing,
apply `confidence_ema_smooth`, and measure again:

```python
from pose_estimation.metrics import temporal_jitter
from pose_estimation.triangulation import confidence_ema_smooth

before = temporal_jitter(sequence_xyz, fps=50)
smoothed = confidence_ema_smooth(sequence_xyz, confidences, alpha=0.65)
after = temporal_jitter(smoothed, fps=50)
print("jitter:", before, "->", after)
```

Sweep `alpha` and the reprojection threshold, keep what lowers jitter without hurting
accuracy, and record the result as a run so the team can compare. See
[metrics.md](metrics.md) for the metric definitions and [workflow.md](workflow.md) for the
run/aggregate/report loop.
