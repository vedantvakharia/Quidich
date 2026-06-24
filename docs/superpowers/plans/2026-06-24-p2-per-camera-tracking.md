# P2 — Per-Camera Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Link P1 per-frame detections into stable per-camera tracklets by filling `local_track_id` on each player record, producing `outputs/p2/cam_XX.jsonl` plus per-camera diagnostics.

**Architecture:** A stateful per-camera tracker (BoT-SORT-style Kalman motion core + two-stage ByteTrack association) with the appearance branch replaced by a pose-configuration gallery. Lost tracks enter a spatially-gated dormant gallery for re-ID. Each camera is tracked independently; all 7 run in parallel processes, each doing its own disk I/O. Source of truth is the design spec `docs/superpowers/specs/2026-06-23-p2-per-camera-tracking-design.md`.

**Tech Stack:** Python 3.11+, numpy 2.x, scipy (`scipy.optimize.linear_sum_assignment`), PyYAML, pytest. No new third-party deps.

## Global Constraints

- Input path: `outputs/p1/predictions/cam_XX.jsonl` (P1 runner writes under a `predictions/` subdir).
- Schema is `g1_player_frame/v0` (`pose_estimation/cricket/contract.py`); every output line must still pass `validate_group1_frame`. P2 only writes `local_track_id`; `global_player_id`, `role`, `pose_3d` stay untouched (`null`).
- Tracking operates entirely in **pixel space**: `bbox_xywh_px`, `pose_2d.keypoints_px`, `pose_2d.confidence`. Normalised fields pass through unchanged.
- `local_track_id` is camera-prefixed `cam_XX_trk_XXXX` (e.g. `cam_01_trk_0001`), matching the canonical example in `contract.py`. Counter is per-camera, monotonic, starts at `0001`, never reused within a delivery.
- `stage2_confidence_min` (default `0.1`) MUST stay ≥ the P1 inference `--conf` (now `0.1`), or the low-conf band is starved.
- Platform is **win32**: `ProcessPoolExecutor` uses `spawn`; entry point guarded by `if __name__ == "__main__":`; worker fn + config must be top-level and picklable (no closures/lambdas).
- Cost matrices must always be finite — no NaN reaches `linear_sum_assignment`. Degenerate poses → cost `1.0`; degenerate scales → floored.
- Follow existing module style: `from __future__ import annotations`, type hints, focused files, `pose_estimation/cricket/` package, tests in `tests/test_cricket_p2_*.py`.
- COCO-17 keypoint indices: 5=L-shoulder, 6=R-shoulder, 11=L-hip, 12=R-hip, 13=L-knee, 14=R-knee, 15=L-ankle, 16=R-ankle.

---

### Task 1: P2 config dataclass + YAML loader

**Files:**
- Create: `pose_estimation/cricket/p2_config.py`
- Create: `configs/p2_tracking.yaml`
- Test: `tests/test_cricket_p2_config.py`

**Interfaces:**
- Consumes: nothing (foundation task).
- Produces:
  - `P2Config` — frozen dataclass with every tunable from spec §9 (field names below).
  - `load_p2_config(path: str | Path | None) -> P2Config` — `None` returns defaults; a YAML file overrides only the keys it contains; unknown keys raise `ValueError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cricket_p2_config.py
from pathlib import Path

import pytest

from pose_estimation.cricket.p2_config import P2Config, load_p2_config


def test_defaults_match_spec():
    cfg = load_p2_config(None)
    assert cfg.stage1_confidence_threshold == 0.5
    assert cfg.stage2_confidence_min == 0.1
    assert cfg.cost_accept_threshold == 0.7
    assert cfg.iou_alpha == 0.6
    assert cfg.pose_beta == 0.4
    assert cfg.min_shared_keypoints == 6
    assert cfg.dormant_max_frames == 60
    assert cfg.gallery_repr == "medoid"


def test_yaml_overrides_only_named_keys(tmp_path: Path):
    path = tmp_path / "p2.yaml"
    path.write_text("dormant_max_frames: 90\nv_max_px_per_frame: 200\n", encoding="utf-8")
    cfg = load_p2_config(path)
    assert cfg.dormant_max_frames == 90
    assert cfg.v_max_px_per_frame == 200
    assert cfg.iou_alpha == 0.6  # untouched default


def test_unknown_key_raises(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("nonsense_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_p2_config(path)


def test_stage2_floor_must_not_undercut_intent():
    cfg = load_p2_config(None)
    # coupling guard from spec §5 / Global Constraints
    assert cfg.stage2_confidence_min <= cfg.stage1_confidence_threshold
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cricket_p2_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pose_estimation.cricket.p2_config'`

- [ ] **Step 3: Write minimal implementation**

```python
# pose_estimation/cricket/p2_config.py
"""Phase 2 per-camera tracking configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class P2Config:
    # Stage thresholds
    stage1_confidence_threshold: float = 0.5
    stage2_confidence_min: float = 0.1
    cost_accept_threshold: float = 0.7
    lowconf_can_spawn: bool = True
    # Cost matrix weights
    iou_alpha: float = 0.6
    pose_beta: float = 0.4
    # Pose vector
    pose_keypoint_confidence_min: float = 0.3
    min_shared_keypoints: int = 6
    scale_min_frac_bbox_h: float = 0.05
    # Spatial / motion gating
    chi2_gate: float = 9.21
    gate_bbox_factor: float = 1.5
    gate_max_distance_px: float = 600.0
    v_max_px_per_frame: float = 120.0
    # Dormant re-ID
    pose_cosine_reid_threshold: float = 0.25
    reid_ambiguity_margin: float = 0.05
    dormant_max_frames: int = 60
    # Kalman stability
    kalman_cov_trace_max: float = 1.0e6
    # Track confirmation
    tentative_confirm_hits: int = 3
    tentative_confirm_window: int = 5
    # Gallery
    pose_gallery_size: int = 30
    gallery_repr: str = "medoid"


def load_p2_config(path: str | Path | None) -> P2Config:
    if path is None:
        return P2Config()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    known = {f.name for f in fields(P2Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown P2 config keys: {sorted(unknown)}")
    return P2Config(**raw)
```

```yaml
# configs/p2_tracking.yaml
# Stage thresholds
stage1_confidence_threshold: 0.5
stage2_confidence_min: 0.1          # matches P1 inference --conf (0.1); MUST stay >= P1 --conf
cost_accept_threshold: 0.7
lowconf_can_spawn: true

# Cost matrix weights
iou_alpha: 0.6
pose_beta: 0.4

# Pose vector
pose_keypoint_confidence_min: 0.3
min_shared_keypoints: 6
scale_min_frac_bbox_h: 0.05

# Spatial / motion gating
chi2_gate: 9.21
gate_bbox_factor: 1.5
gate_max_distance_px: 600
v_max_px_per_frame: 120

# Dormant re-ID
pose_cosine_reid_threshold: 0.25
reid_ambiguity_margin: 0.05
dormant_max_frames: 60

# Kalman stability
kalman_cov_trace_max: 1.0e6

# Track confirmation
tentative_confirm_hits: 3
tentative_confirm_window: 5

# Gallery
pose_gallery_size: 30
gallery_repr: medoid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cricket_p2_config.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add pose_estimation/cricket/p2_config.py configs/p2_tracking.yaml tests/test_cricket_p2_config.py
git commit -m "feat(p2): config dataclass and YAML loader"
```

---

### Task 2: Pose vector representation + masked cosine distance

**Files:**
- Create: `pose_estimation/cricket/p2_pose_vector.py`
- Test: `tests/test_cricket_p2_pose_vector.py`

**Interfaces:**
- Consumes: `P2Config` (Task 1) for `pose_keypoint_confidence_min`, `scale_min_frac_bbox_h`, `min_shared_keypoints`.
- Produces:
  - `PoseVector` — frozen dataclass: `vector: np.ndarray (34,)`, `mask: np.ndarray (17,) bool`, `confidence: np.ndarray (17,)`, `defined: bool`.
  - `build_pose_vector(keypoints_px: list[list[float]], confidence: list[float], bbox_xywh_px: list[float], config: P2Config) -> PoseVector`.
  - `masked_weighted_cosine(a: PoseVector, b: PoseVector, *, min_shared_keypoints: int) -> float` — returns cost in `[0.0, 2.0]`, never NaN; `1.0` when undefined / too few shared / zero-norm.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cricket_p2_pose_vector.py
import numpy as np

from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_pose_vector import (
    build_pose_vector,
    masked_weighted_cosine,
)

CFG = P2Config()


def _kp(value=10.0):
    return [[value + i, value + 2 * i] for i in range(17)]


def test_identical_poses_have_zero_cost():
    kp = _kp()
    conf = [0.9] * 17
    bbox = [0.0, 0.0, 100.0, 200.0]
    a = build_pose_vector(kp, conf, bbox, CFG)
    b = build_pose_vector(kp, conf, bbox, CFG)
    assert a.defined and b.defined
    assert masked_weighted_cosine(a, b, min_shared_keypoints=CFG.min_shared_keypoints) < 1e-6


def test_low_confidence_keypoints_are_masked_out():
    kp = _kp()
    conf = [0.9] * 17
    conf[9] = 0.1  # below 0.3 -> invalid
    a = build_pose_vector(kp, conf, [0, 0, 100, 200], CFG)
    assert a.mask[9] == False
    assert a.mask[5] == True


def test_undefined_when_no_root_or_shoulders():
    kp = _kp()
    conf = [0.9] * 17
    for i in (5, 6, 11, 12):
        conf[i] = 0.0  # kill both hips and both shoulders
    a = build_pose_vector(kp, conf, [0, 0, 100, 200], CFG)
    assert a.defined is False


def test_too_few_shared_keypoints_returns_unit_cost():
    kp = _kp()
    conf_a = [0.9] * 17
    conf_b = [0.9] * 17
    # only keypoints 11,12 shared-valid (fewer than min_shared_keypoints=6)
    for i in range(17):
        if i not in (11, 12):
            conf_b[i] = 0.0
    a = build_pose_vector(kp, conf_a, [0, 0, 100, 200], CFG)
    b = build_pose_vector(kp, conf_b, [0, 0, 100, 200], CFG)
    assert masked_weighted_cosine(a, b, min_shared_keypoints=6) == 1.0


def test_cost_is_always_finite_and_bounded():
    kp = _kp()
    a = build_pose_vector(kp, [0.9] * 17, [0, 0, 100, 200], CFG)
    flipped = [[-x, -y] for x, y in kp]
    b = build_pose_vector(flipped, [0.9] * 17, [0, 0, 100, 200], CFG)
    cost = masked_weighted_cosine(a, b, min_shared_keypoints=6)
    assert np.isfinite(cost)
    assert 0.0 <= cost <= 2.0


def test_degenerate_scale_is_floored_not_divide_by_zero():
    # all keypoints collapse to one pixel -> every anchor length ~0
    kp = [[50.0, 50.0] for _ in range(17)]
    a = build_pose_vector(kp, [0.9] * 17, [0, 0, 100, 200], CFG)
    assert np.all(np.isfinite(a.vector))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cricket_p2_pose_vector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pose_estimation.cricket.p2_pose_vector'`

- [ ] **Step 3: Write minimal implementation**

```python
# pose_estimation/cricket/p2_pose_vector.py
"""Phase 2 pose-vector representation and masked confidence-weighted cosine distance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pose_estimation.cricket.p2_config import P2Config

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
KEYPOINT_COUNT = 17

# Scale cascade: (endpoint_a, endpoint_b) pairs tried in order; bbox diagonal is last resort.
_SCALE_ANCHORS = [(L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP)]


@dataclass(frozen=True)
class PoseVector:
    vector: np.ndarray      # (34,) float — [x0, y0, ..., x16, y16], root-relative & scaled
    mask: np.ndarray        # (17,) bool — per-keypoint validity
    confidence: np.ndarray  # (17,) float
    defined: bool           # False when no valid root anchor exists


def _select_root(kp: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    if valid[L_HIP] and valid[R_HIP]:
        return (kp[L_HIP] + kp[R_HIP]) / 2.0
    if valid[L_HIP]:
        return kp[L_HIP]
    if valid[R_HIP]:
        return kp[R_HIP]
    if valid[L_SHOULDER] and valid[R_SHOULDER]:
        return (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2.0
    if valid[L_SHOULDER]:
        return kp[L_SHOULDER]
    if valid[R_SHOULDER]:
        return kp[R_SHOULDER]
    return None


def _select_scale(kp: np.ndarray, valid: np.ndarray, bbox_xywh: list[float], scale_min: float) -> float:
    for a, b in _SCALE_ANCHORS:
        if valid[a] and valid[b]:
            length = float(np.linalg.norm(kp[a] - kp[b]))
            if length > scale_min:
                return max(length, scale_min)
    diag = float(np.hypot(bbox_xywh[2], bbox_xywh[3]))
    return max(diag, scale_min)


def build_pose_vector(
    keypoints_px: list[list[float]],
    confidence: list[float],
    bbox_xywh_px: list[float],
    config: P2Config,
) -> PoseVector:
    kp = np.asarray(keypoints_px, dtype=float).reshape(KEYPOINT_COUNT, 2)
    conf = np.asarray(confidence, dtype=float).reshape(KEYPOINT_COUNT)
    valid = conf >= config.pose_keypoint_confidence_min

    root = _select_root(kp, valid)
    if root is None:
        return PoseVector(
            vector=np.zeros(2 * KEYPOINT_COUNT),
            mask=np.zeros(KEYPOINT_COUNT, dtype=bool),
            confidence=conf,
            defined=False,
        )

    scale_min = config.scale_min_frac_bbox_h * float(bbox_xywh_px[3])
    scale_min = max(scale_min, 1e-6)
    scale = _select_scale(kp, valid, bbox_xywh_px, scale_min)

    relative = (kp - root) / scale
    return PoseVector(
        vector=relative.reshape(-1),
        mask=valid,
        confidence=conf,
        defined=True,
    )


def masked_weighted_cosine(a: PoseVector, b: PoseVector, *, min_shared_keypoints: int) -> float:
    if not a.defined or not b.defined:
        return 1.0
    shared = a.mask & b.mask
    if int(shared.sum()) < min_shared_keypoints:
        return 1.0
    idx = np.where(shared)[0]
    weights = np.minimum(a.confidence[idx], b.confidence[idx])  # (k,)
    av = a.vector.reshape(KEYPOINT_COUNT, 2)[idx]               # (k, 2)
    bv = b.vector.reshape(KEYPOINT_COUNT, 2)[idx]
    dot = float(np.sum(weights * np.sum(av * bv, axis=1)))
    norm_a = float(np.sqrt(np.sum(weights * np.sum(av * av, axis=1))))
    norm_b = float(np.sqrt(np.sum(weights * np.sum(bv * bv, axis=1))))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 1.0
    cosine = dot / (norm_a * norm_b)
    cosine = max(-1.0, min(1.0, cosine))
    return 1.0 - cosine
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cricket_p2_pose_vector.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add pose_estimation/cricket/p2_pose_vector.py tests/test_cricket_p2_pose_vector.py
git commit -m "feat(p2): pose vector + masked confidence-weighted cosine distance"
```

---

### Task 3: Constant-velocity Kalman filter + gating

**Files:**
- Create: `pose_estimation/cricket/p2_kalman.py`
- Test: `tests/test_cricket_p2_kalman.py`

**Interfaces:**
- Consumes: nothing beyond numpy.
- Produces `KalmanBoxTracker` with:
  - `__init__(self, bbox_xywh: list[float])` — seeds state from first detection.
  - `predict(self) -> None` — advance one frame (call once per frame before matching).
  - `update(self, bbox_xywh: list[float]) -> None` — correct with a matched detection.
  - `predicted_bbox(self) -> np.ndarray` — current `[cx, cy, w, h]` → returns `xywh` (top-left + w/h) for IoU.
  - `center(self) -> np.ndarray` — `(cx, cy)`.
  - `bbox_height(self) -> float`.
  - `position_cov_trace(self) -> float` — trace of the 2×2 position covariance.
  - `gating_distance_sq(self, center_xy: np.ndarray) -> float` — squared Mahalanobis distance of a point to predicted centre.
  - `inflate_process_noise(self, factor: float) -> None` — grow process noise while dormant.
  - `velocity(self) -> np.ndarray` — `(vcx, vcy)`.
  - `reseed(self, bbox_xywh: list[float], keep_velocity: np.ndarray) -> None` — re-init position from detection, retain prior velocity.

State vector: `[cx, cy, w, h, vcx, vcy, vw, vh]`. Measurement: `[cx, cy, w, h]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cricket_p2_kalman.py
import numpy as np

from pose_estimation.cricket.p2_kalman import KalmanBoxTracker


def _xywh_topleft(cx, cy, w, h):
    return np.array([cx - w / 2, cy - h / 2, w, h])


def test_predicted_bbox_matches_seed_before_motion():
    kf = KalmanBoxTracker([100.0, 200.0, 40.0, 80.0])  # top-left xywh
    kf.predict()
    bbox = kf.predicted_bbox()
    assert np.allclose(bbox, [100.0, 200.0, 40.0, 80.0], atol=1.0)


def test_constant_velocity_extrapolates_forward():
    kf = KalmanBoxTracker([100.0, 200.0, 40.0, 80.0])
    for step in range(1, 6):
        kf.predict()
        kf.update([100.0 + 10 * step, 200.0, 40.0, 80.0])  # moving +10px/frame in x
    kf.predict()
    cx, _ = kf.center()
    assert cx > 150.0  # extrapolated beyond last observation


def test_gating_distance_grows_when_dormant():
    kf = KalmanBoxTracker([100.0, 200.0, 40.0, 80.0])
    kf.predict()
    near = kf.gating_distance_sq(np.array([105.0, 205.0]))
    kf.inflate_process_noise(10.0)
    kf.predict()
    relaxed = kf.gating_distance_sq(np.array([105.0, 205.0]))
    assert relaxed < near  # bigger covariance -> smaller Mahalanobis distance


def test_cov_trace_is_finite_and_positive():
    kf = KalmanBoxTracker([100.0, 200.0, 40.0, 80.0])
    kf.predict()
    assert np.isfinite(kf.position_cov_trace())
    assert kf.position_cov_trace() > 0.0


def test_reseed_retains_velocity():
    kf = KalmanBoxTracker([100.0, 200.0, 40.0, 80.0])
    for step in range(1, 4):
        kf.predict()
        kf.update([100.0 + 10 * step, 200.0, 40.0, 80.0])
    v_before = kf.velocity().copy()
    kf.reseed([300.0, 200.0, 40.0, 80.0], keep_velocity=v_before)
    assert np.allclose(kf.velocity(), v_before)
    assert np.allclose(kf.center(), [300.0, 200.0], atol=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cricket_p2_kalman.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pose_estimation.cricket.p2_kalman'`

- [ ] **Step 3: Write minimal implementation**

```python
# pose_estimation/cricket/p2_kalman.py
"""Constant-velocity Kalman filter for bbox tracking (centre + size state)."""

from __future__ import annotations

import numpy as np

# State: [cx, cy, w, h, vcx, vcy, vw, vh]; measurement: [cx, cy, w, h]
_NDIM = 4


class KalmanBoxTracker:
    def __init__(self, bbox_xywh: list[float]) -> None:
        cx, cy, w, h = self._to_cxcywh(bbox_xywh)
        self._x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=float)

        self._F = np.eye(8)
        for i in range(_NDIM):
            self._F[i, i + _NDIM] = 1.0  # x += v each step

        self._H = np.zeros((_NDIM, 8))
        self._H[:_NDIM, :_NDIM] = np.eye(_NDIM)

        self._P = np.eye(8) * 10.0
        self._P[4:, 4:] *= 1000.0  # high initial velocity uncertainty
        self._q = 1.0   # process-noise scale (inflated while dormant)
        self._r = 1.0   # measurement-noise scale

    @staticmethod
    def _to_cxcywh(bbox_xywh: list[float]) -> tuple[float, float, float, float]:
        x, y, w, h = [float(v) for v in bbox_xywh]
        return x + w / 2.0, y + h / 2.0, w, h

    def _Q(self) -> np.ndarray:
        q = np.eye(8)
        q[:_NDIM, :_NDIM] *= self._q
        q[_NDIM:, _NDIM:] *= self._q * 0.01
        return q

    def _R(self) -> np.ndarray:
        return np.eye(_NDIM) * self._r

    def predict(self) -> None:
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q()

    def update(self, bbox_xywh: list[float]) -> None:
        z = np.array(self._to_cxcywh(bbox_xywh), dtype=float)
        S = self._H @ self._P @ self._H.T + self._R()
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ (z - self._H @ self._x)
        self._P = (np.eye(8) - K @ self._H) @ self._P

    def predicted_bbox(self) -> np.ndarray:
        cx, cy, w, h = self._x[:_NDIM]
        return np.array([cx - w / 2.0, cy - h / 2.0, w, h])

    def center(self) -> np.ndarray:
        return self._x[:2].copy()

    def velocity(self) -> np.ndarray:
        return self._x[4:6].copy()

    def bbox_height(self) -> float:
        return float(self._x[3])

    def position_cov_trace(self) -> float:
        return float(np.trace(self._P[:2, :2]))

    def gating_distance_sq(self, center_xy: np.ndarray) -> float:
        S = self._P[:2, :2] + np.eye(2) * self._r
        diff = np.asarray(center_xy, dtype=float) - self._x[:2]
        return float(diff @ np.linalg.inv(S) @ diff)

    def inflate_process_noise(self, factor: float) -> None:
        self._q *= factor

    def reseed(self, bbox_xywh: list[float], keep_velocity: np.ndarray) -> None:
        cx, cy, w, h = self._to_cxcywh(bbox_xywh)
        self._x = np.array(
            [cx, cy, w, h, float(keep_velocity[0]), float(keep_velocity[1]), 0.0, 0.0],
            dtype=float,
        )
        self._P = np.eye(8) * 10.0
        self._P[4:, 4:] *= 1000.0
        self._q = 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cricket_p2_kalman.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add pose_estimation/cricket/p2_kalman.py tests/test_cricket_p2_kalman.py
git commit -m "feat(p2): constant-velocity Kalman filter with Mahalanobis gating"
```

---

### Task 4: Track lifecycle + pose gallery (medoid)

**Files:**
- Create: `pose_estimation/cricket/p2_track.py`
- Test: `tests/test_cricket_p2_track.py`

**Interfaces:**
- Consumes: `P2Config` (Task 1), `PoseVector` + `masked_weighted_cosine` (Task 2), `KalmanBoxTracker` (Task 3).
- Produces:
  - Module constants `TENTATIVE = "tentative"`, `CONFIRMED = "confirmed"`, `DORMANT = "dormant"`, `DELETED = "deleted"`.
  - `Track` class:
    - `__init__(self, id_num: int, camera_id: str, bbox_xywh, pose: PoseVector, is_lowconf: bool, config: P2Config, frame_index: int)`.
    - `state: str` (one of the constants).
    - `local_track_id` property → `f"{camera_id}_trk_{id_num:04d}"`.
    - `register_hit(self, bbox_xywh, pose: PoseVector, confidence: float, frame_index: int) -> None` — Kalman update + gallery append + bookkeeping; resets miss counter; flips DORMANT→CONFIRMED.
    - `mark_missed(self, frame_index: int) -> None` — Kalman predict-only, increments dormancy, inflates process noise; CONFIRMED→DORMANT on first miss.
    - `maybe_confirm(self) -> bool` — returns True and promotes when confirmation rule met.
    - `should_delete(self) -> bool` — dormant expiry OR covariance explosion.
    - `gallery_repr(self) -> PoseVector | None` — medoid of stored vectors (or mean when `gallery_repr == "mean"`).
    - `reachability_radius(self) -> float` — `v_max·frames_dormant + gate_bbox_factor·bbox_height`.
    - Public fields used by the tracker: `frames_since_update: int`, `hits: int`, `highconf_hits: int`, `is_lowconf: bool`, `kalman: KalmanBoxTracker`, `max_cov_trace: float`, `gap_count: int`, `max_gap_frames: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cricket_p2_track.py
import numpy as np

from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_pose_vector import build_pose_vector
from pose_estimation.cricket.p2_track import (
    CONFIRMED, DELETED, DORMANT, TENTATIVE, Track,
)

CFG = P2Config()


def _pose(shift=0.0):
    kp = [[10.0 + i + shift, 12.0 + 2 * i] for i in range(17)]
    return build_pose_vector(kp, [0.9] * 17, [0, 0, 40, 80], CFG)


def _new_track(is_lowconf=False, num=1):
    return Track(num, "cam_01", [100, 200, 40, 80], _pose(), is_lowconf, CFG, frame_index=0)


def test_local_track_id_is_camera_prefixed():
    t = _new_track(num=7)
    assert t.local_track_id == "cam_01_trk_0007"


def test_highconf_track_confirms_after_three_hits_in_window():
    t = _new_track()
    assert t.state == TENTATIVE
    for f in range(1, 3):
        t.register_hit([100, 200, 40, 80], _pose(), 0.8, frame_index=f)
        t.maybe_confirm()
    assert t.state == CONFIRMED  # 3 hits (spawn + 2) within window


def test_lowconf_track_needs_a_highconf_hit():
    t = _new_track(is_lowconf=True)
    for f in range(1, 3):
        t.register_hit([100, 200, 40, 80], _pose(), 0.2, frame_index=f)  # all low-conf
        t.maybe_confirm()
    assert t.state == TENTATIVE  # never got a >0.5 hit
    t.register_hit([100, 200, 40, 80], _pose(), 0.8, frame_index=3)
    t.maybe_confirm()
    assert t.state == CONFIRMED


def test_confirmed_goes_dormant_then_deletes_after_max_frames():
    t = _new_track()
    for f in range(1, 3):
        t.register_hit([100, 200, 40, 80], _pose(), 0.8, frame_index=f)
        t.maybe_confirm()
    assert t.state == CONFIRMED
    t.mark_missed(frame_index=3)
    assert t.state == DORMANT
    for f in range(4, 4 + CFG.dormant_max_frames + 1):
        t.mark_missed(frame_index=f)
    assert t.should_delete() is True


def test_reachability_radius_grows_with_dormancy():
    t = _new_track()
    for f in range(1, 3):
        t.register_hit([100, 200, 40, 80], _pose(), 0.8, frame_index=f)
        t.maybe_confirm()
    t.mark_missed(frame_index=3)
    r1 = t.reachability_radius()
    t.mark_missed(frame_index=4)
    assert t.reachability_radius() > r1


def test_gallery_medoid_is_a_stored_vector():
    t = _new_track()
    for f in range(1, 5):
        t.register_hit([100, 200, 40, 80], _pose(shift=f), 0.8, frame_index=f)
    repr_vec = t.gallery_repr()
    assert repr_vec is not None and repr_vec.defined


def test_lowconf_tentative_expires_without_highconf_hit():
    t = _new_track(is_lowconf=True)
    # keeps matching low-conf detections every frame but never lands a >0.5 hit
    for f in range(1, CFG.tentative_confirm_window + 1):
        t.register_hit([100, 200, 40, 80], _pose(), 0.2, frame_index=f)
        t.maybe_confirm()
    assert t.state == TENTATIVE
    assert t.tentative_expired(CFG.tentative_confirm_window) is True


def test_flush_id_backfills_tentative_frames_on_confirmation():
    t = _new_track()
    spawn_player = {"local_track_id": None}
    t.record_player(spawn_player)              # frame 0, still TENTATIVE
    t.flush_id()
    assert spawn_player["local_track_id"] is None  # not stamped while tentative

    next_player = {"local_track_id": None}
    for f in (1, 2):
        t.register_hit([100, 200, 40, 80], _pose(), 0.8, frame_index=f)
    t.record_player(next_player)
    t.maybe_confirm()                          # now CONFIRMED
    t.flush_id()                               # back-fills BOTH the buffered spawn frame and now
    assert spawn_player["local_track_id"] == "cam_01_trk_0001"
    assert next_player["local_track_id"] == "cam_01_trk_0001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cricket_p2_track.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pose_estimation.cricket.p2_track'`

- [ ] **Step 3: Write minimal implementation**

```python
# pose_estimation/cricket/p2_track.py
"""Phase 2 track object: lifecycle state machine + pose gallery."""

from __future__ import annotations

from collections import deque

import numpy as np

from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_kalman import KalmanBoxTracker
from pose_estimation.cricket.p2_pose_vector import PoseVector, masked_weighted_cosine

TENTATIVE = "tentative"
CONFIRMED = "confirmed"
DORMANT = "dormant"
DELETED = "deleted"


class Track:
    def __init__(
        self,
        id_num: int,
        camera_id: str,
        bbox_xywh: list[float],
        pose: PoseVector,
        is_lowconf: bool,
        config: P2Config,
        frame_index: int,
    ) -> None:
        self._id_num = id_num
        self._camera_id = camera_id
        self._config = config
        self.is_lowconf = is_lowconf
        self.state = TENTATIVE

        self.kalman = KalmanBoxTracker(bbox_xywh)
        self._gallery: deque[PoseVector] = deque(maxlen=config.pose_gallery_size)
        if pose.defined:
            self._gallery.append(pose)

        self.hits = 1
        self.highconf_hits = 0
        self.frames_since_update = 0
        self._spawn_frame = frame_index
        self._last_frame = frame_index

        self.max_cov_trace = self.kalman.position_cov_trace()
        self.gap_count = 0
        self.max_gap_frames = 0
        self._current_gap = 0

        # Every player dict this track has matched, in frame order, for retroactive ID back-fill.
        self.assigned_players: list[dict] = []
        self._stamped_upto = 0

    @property
    def local_track_id(self) -> str:
        return f"{self._camera_id}_trk_{self._id_num:04d}"

    def record_player(self, player: dict) -> None:
        """Remember a matched player dict so its `local_track_id` can be filled on confirmation."""
        self.assigned_players.append(player)

    def flush_id(self) -> None:
        """Stamp `local_track_id` onto any not-yet-stamped matched players (no-op until CONFIRMED).

        Back-fills the tentative frames retroactively the moment the track promotes (spec §6).
        Idempotent: only players past `_stamped_upto` are written.
        """
        if self.state != CONFIRMED:
            return
        tid = self.local_track_id
        for player in self.assigned_players[self._stamped_upto:]:
            player["local_track_id"] = tid
        self._stamped_upto = len(self.assigned_players)

    def register_hit(self, bbox_xywh, pose: PoseVector, confidence: float, frame_index: int) -> None:
        self.kalman.update(bbox_xywh)
        if pose.defined:
            self._gallery.append(pose)
        self.hits += 1
        if confidence > self._config.stage1_confidence_threshold:
            self.highconf_hits += 1
        if self._current_gap > 0:
            self.gap_count += 1
            self.max_gap_frames = max(self.max_gap_frames, self._current_gap)
            self._current_gap = 0
        self.frames_since_update = 0
        self._last_frame = frame_index
        self.max_cov_trace = max(self.max_cov_trace, self.kalman.position_cov_trace())
        if self.state == DORMANT:
            self.state = CONFIRMED

    def mark_missed(self, frame_index: int) -> None:
        # NB: the tracker calls kalman.predict() once per frame for every track at the top of
        # update(); mark_missed must NOT predict again (that would double-advance the state).
        self.kalman.inflate_process_noise(1.5)
        self.frames_since_update += 1
        self._current_gap += 1
        self.max_cov_trace = max(self.max_cov_trace, self.kalman.position_cov_trace())
        if self.state == CONFIRMED:
            self.state = DORMANT

    def maybe_confirm(self, *, ignore_window: bool = False) -> bool:
        """Promote TENTATIVE→CONFIRMED when the confirmation rule is met.

        `ignore_window=True` is used at end-of-stream (spec §6): the window is closed early, so a
        track already meeting the hit count is promoted even if 5 frames have not elapsed.
        """
        if self.state != TENTATIVE:
            return False
        within_window = ignore_window or (
            (self._last_frame - self._spawn_frame) < self._config.tentative_confirm_window
        )
        enough_hits = self.hits >= self._config.tentative_confirm_hits
        highconf_ok = (not self.is_lowconf) or self.highconf_hits >= 1
        if within_window and enough_hits and highconf_ok:
            self.state = CONFIRMED
            return True
        return False

    def tentative_expired(self, frame_index: int) -> bool:
        """A still-TENTATIVE track whose confirmation window has fully elapsed (spec §5/§6).

        Catches both the silent-rejection case and a low-conf tentative that keeps matching
        low-conf detections but never lands a >0.5 hit within the window.
        """
        return (
            self.state == TENTATIVE
            and (frame_index - self._spawn_frame) >= self._config.tentative_confirm_window
        )

    def should_delete(self) -> bool:
        if self.kalman.position_cov_trace() > self._config.kalman_cov_trace_max:
            return True
        if self.state == DORMANT and self.frames_since_update > self._config.dormant_max_frames:
            return True
        return False

    def reachability_radius(self) -> float:
        return (
            self._config.v_max_px_per_frame * self.frames_since_update
            + self._config.gate_bbox_factor * self.kalman.bbox_height()
        )

    def gallery_repr(self) -> PoseVector | None:
        members = [v for v in self._gallery if v.defined]
        if not members:
            return None
        if len(members) == 1 or self._config.gallery_repr != "medoid":
            return members[0]
        best_idx, best_cost = 0, float("inf")
        for i, vi in enumerate(members):
            total = sum(
                masked_weighted_cosine(vi, vj, min_shared_keypoints=self._config.min_shared_keypoints)
                for j, vj in enumerate(members)
                if j != i
            )
            if total < best_cost:
                best_idx, best_cost = i, total
        return members[best_idx]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cricket_p2_track.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add pose_estimation/cricket/p2_track.py tests/test_cricket_p2_track.py
git commit -m "feat(p2): track lifecycle state machine and medoid pose gallery"
```

---

### Task 5: Per-camera tracker (two-stage association + dormant re-ID)

**Files:**
- Create: `pose_estimation/cricket/p2_tracker.py`
- Test: `tests/test_cricket_p2_tracker.py`

**Interfaces:**
- Consumes: `P2Config`, `build_pose_vector`/`masked_weighted_cosine`/`PoseVector`, `Track` + state constants, `KalmanBoxTracker`.
- Produces:
  - `Detection` — frozen dataclass: `bbox_xywh: list[float]`, `pose: PoseVector`, `confidence: float`, `player: dict` (reference to the original P1 player dict so the tracker can stamp `local_track_id`).
  - `iou_xywh(a: list[float], b: list[float]) -> float`.
  - `CameraTracker`:
    - `__init__(self, camera_id: str, config: P2Config)`.
    - `update(self, detections: list[Detection], frame_index: int) -> None` — runs one frame: predict, Stage 1, Stage 2, dormant re-ID, spawning, confirmation, deletion; stamps `local_track_id` on matched confirmed-track players.
    - `diagnostics` counters dict (incrementally maintained): keys `total_tracks_spawned`, `lowconf_tracks_spawned`, `confirmed_tracks`, `tentative_rejected`, `dormant_reidentified`, `dormant_reid_ambiguous`, `dormant_deleted`, `pose_undefined_count`, `pose_skipped_low_overlap`, `kalman_cov_explosions`, `id_switches_estimated`, `frames_with_unmatched_detections`.
    - `tracks: list[Track]` (live tracks) and `finalize() -> None` (closes confirmation windows at EOF; counts `tentative_unresolved_at_eof`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cricket_p2_tracker.py
import numpy as np

from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_pose_vector import build_pose_vector
from pose_estimation.cricket.p2_tracker import CameraTracker, Detection, iou_xywh

CFG = P2Config()


def _pose(shift=0.0):
    kp = [[10.0 + i + shift, 12.0 + 2 * i] for i in range(17)]
    return build_pose_vector(kp, [0.9] * 17, [0, 0, 40, 80], CFG)


def _det(x, y, conf=0.8, player=None):
    bbox = [float(x), float(y), 40.0, 80.0]
    return Detection(bbox_xywh=bbox, pose=_pose(), confidence=conf, player=player if player is not None else {})


def test_iou_identical_boxes_is_one():
    assert abs(iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9


def test_iou_disjoint_boxes_is_zero():
    assert iou_xywh([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0


def test_single_object_gets_one_stable_id():
    tracker = CameraTracker("cam_01", CFG)
    players = []
    for f in range(6):
        p = {"local_track_id": None, "track_confidence": 0.8}
        players.append(p)
        tracker.update([_det(100 + 5 * f, 200, player=p)], frame_index=f)
    tracker.finalize()
    # every frame back-filled (incl. the pre-confirmation tentative frames) with one stable id
    assert all(p["local_track_id"] == "cam_01_trk_0001" for p in players)


def test_two_separated_objects_get_distinct_ids():
    tracker = CameraTracker("cam_01", CFG)
    left, right = [], []
    for f in range(6):
        pl = {"local_track_id": None, "track_confidence": 0.8}
        pr = {"local_track_id": None, "track_confidence": 0.8}
        left.append(pl)
        right.append(pr)
        tracker.update([_det(100, 200, player=pl), _det(900, 200, player=pr)], frame_index=f)
    lid = {p["local_track_id"] for p in left if p["local_track_id"]}
    rid = {p["local_track_id"] for p in right if p["local_track_id"]}
    assert len(lid) == 1 and len(rid) == 1
    assert lid != rid


def test_unmatched_low_conf_does_not_crash_and_counts():
    tracker = CameraTracker("cam_01", CFG)
    p = {"local_track_id": None, "track_confidence": 0.2}
    tracker.update([_det(100, 200, conf=0.2, player=p)], frame_index=0)
    tracker.finalize()
    assert tracker.diagnostics["total_tracks_spawned"] >= 1


def test_dormant_reid_recovers_same_id_after_short_gap():
    tracker = CameraTracker("cam_01", CFG)
    players = []
    for f in range(5):
        p = {"local_track_id": None, "track_confidence": 0.8}
        players.append(p)
        tracker.update([_det(100, 200, player=p)], frame_index=f)
    first_id = players[-1]["local_track_id"]
    for f in range(5, 8):  # 3-frame gap, no detections
        tracker.update([], frame_index=f)
    p = {"local_track_id": None, "track_confidence": 0.8}
    tracker.update([_det(110, 205, player=p)], frame_index=8)
    assert p["local_track_id"] == first_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cricket_p2_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pose_estimation.cricket.p2_tracker'`

- [ ] **Step 3: Write minimal implementation**

```python
# pose_estimation/cricket/p2_tracker.py
"""Phase 2 per-camera tracker: two-stage association + spatially-gated dormant re-ID."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_pose_vector import PoseVector, masked_weighted_cosine
from pose_estimation.cricket.p2_track import CONFIRMED, DORMANT, TENTATIVE, Track

_LARGE = 1e6  # sentinel "no match" cost; never NaN/inf into linear_sum_assignment


@dataclass(frozen=True)
class Detection:
    bbox_xywh: list[float]
    pose: PoseVector
    confidence: float
    player: dict


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def _new_diagnostics() -> dict[str, int]:
    return {
        "total_tracks_spawned": 0,
        "lowconf_tracks_spawned": 0,
        "confirmed_tracks": 0,
        "tentative_rejected": 0,
        "tentative_unresolved_at_eof": 0,
        "dormant_reidentified": 0,
        "dormant_reid_ambiguous": 0,
        "dormant_deleted": 0,
        "pose_undefined_count": 0,
        "pose_skipped_low_overlap": 0,
        "kalman_cov_explosions": 0,
        "id_switches_estimated": 0,
        "frames_with_unmatched_detections": 0,
    }


class CameraTracker:
    def __init__(self, camera_id: str, config: P2Config) -> None:
        self.camera_id = camera_id
        self.config = config
        self.tracks: list[Track] = []
        self.diagnostics = _new_diagnostics()
        self._next_id = 1
        self._prev_match: dict[int, str] = {}  # detection-centre key -> track id, for id-switch heuristic

    # ---- association helpers -------------------------------------------------
    def _match(self, dets: list[Detection], tracks: list[Track], use_pose: bool) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        if not dets or not tracks:
            return [], set(range(len(dets))), set(range(len(tracks)))
        cost = np.full((len(dets), len(tracks)), _LARGE, dtype=float)
        for di, d in enumerate(dets):
            for ti, t in enumerate(tracks):
                pred = t.kalman.predicted_bbox()
                iou = iou_xywh(d.bbox_xywh, list(pred))
                # scale-adaptive gate: exclude only if no overlap AND beyond reach
                if iou == 0.0:
                    center = np.array([d.bbox_xywh[0] + d.bbox_xywh[2] / 2,
                                       d.bbox_xywh[1] + d.bbox_xywh[3] / 2])
                    maha_ok = t.kalman.gating_distance_sq(center) <= self.config.chi2_gate
                    floor = self.config.gate_bbox_factor * t.kalman.bbox_height()
                    dist = float(np.linalg.norm(center - t.kalman.center()))
                    within = (maha_ok or dist <= floor) and dist <= self.config.gate_max_distance_px
                    if not within:
                        continue
                iou_cost = 1.0 - iou
                if use_pose:
                    repr_vec = t.gallery_repr()
                    pose_cost = (
                        masked_weighted_cosine(d.pose, repr_vec, min_shared_keypoints=self.config.min_shared_keypoints)
                        if repr_vec is not None else 1.0
                    )
                    if repr_vec is None or not d.pose.defined or pose_cost >= 1.0:
                        if repr_vec is not None and d.pose.defined and pose_cost >= 1.0:
                            self.diagnostics["pose_skipped_low_overlap"] += 1
                        c = iou_cost  # IoU alone (pose unavailable / no overlap)
                    else:
                        a, b = self.config.iou_alpha, self.config.pose_beta
                        c = (a * iou_cost + b * pose_cost) / (a + b)
                else:
                    c = iou_cost
                if c <= self.config.cost_accept_threshold:
                    cost[di, ti] = c
        rows, cols = linear_sum_assignment(cost)
        matches, um_d, um_t = [], set(range(len(dets))), set(range(len(tracks)))
        for r, c in zip(rows, cols):
            if cost[r, c] < _LARGE:
                matches.append((r, c))
                um_d.discard(r)
                um_t.discard(c)
        return matches, um_d, um_t

    def _spawn(self, det: Detection, frame_index: int) -> Track:
        # Track.__init__ already counts this detection as the first hit and seeds the gallery,
        # so we do NOT also call register_hit; we only record the player for retroactive back-fill.
        track = Track(self._next_id, self.camera_id, det.bbox_xywh, det.pose,
                      is_lowconf=det.confidence <= self.config.stage1_confidence_threshold,
                      config=self.config, frame_index=frame_index)
        track.record_player(det.player)
        self._next_id += 1
        self.tracks.append(track)
        self.diagnostics["total_tracks_spawned"] += 1
        if track.is_lowconf:
            self.diagnostics["lowconf_tracks_spawned"] += 1
        return track

    def _try_dormant_reid(self, det: Detection, frame_index: int) -> Track | None:
        center = np.array([det.bbox_xywh[0] + det.bbox_xywh[2] / 2,
                           det.bbox_xywh[1] + det.bbox_xywh[3] / 2])
        candidates = []
        for t in self.tracks:
            if t.state != DORMANT:
                continue
            if float(np.linalg.norm(center - t.kalman.center())) <= t.reachability_radius():
                repr_vec = t.gallery_repr()
                if repr_vec is None or not det.pose.defined:
                    continue
                cost = masked_weighted_cosine(det.pose, repr_vec, min_shared_keypoints=self.config.min_shared_keypoints)
                if cost < self.config.pose_cosine_reid_threshold:
                    candidates.append((cost, t))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        if len(candidates) >= 2 and (candidates[1][0] - candidates[0][0]) < self.config.reid_ambiguity_margin:
            self.diagnostics["dormant_reid_ambiguous"] += 1
            return None
        cost, track = candidates[0]
        prior_v = track.kalman.velocity()
        track.kalman.reseed(det.bbox_xywh, keep_velocity=prior_v)
        self.diagnostics["dormant_reidentified"] += 1
        return track

    # ---- per-frame entry point ----------------------------------------------
    def update(self, detections: list[Detection], frame_index: int) -> None:
        for t in self.tracks:
            t.kalman.predict()

        high = [d for d in detections if d.confidence > self.config.stage1_confidence_threshold]
        low = [d for d in detections
               if self.config.stage2_confidence_min <= d.confidence <= self.config.stage1_confidence_threshold]
        for d in detections:
            if not d.pose.defined:
                self.diagnostics["pose_undefined_count"] += 1

        hit: set[int] = set()  # id() of every track that matched/spawned this frame
        active = [t for t in self.tracks if t.state in (CONFIRMED, TENTATIVE)]

        # Stage 1: high-conf vs active, IoU + pose
        m1, um_d1, um_t1 = self._match(high, active, use_pose=True)
        for di, ti in m1:
            self._apply_hit(active[ti], high[di], frame_index)
            hit.add(id(active[ti]))

        # Stage 2: low-conf vs still-unmatched active, IoU only
        remaining = [active[i] for i in um_t1]
        m2, um_d2, _ = self._match(low, remaining, use_pose=False)
        for di, ti in m2:
            self._apply_hit(remaining[ti], low[di], frame_index)
            hit.add(id(remaining[ti]))

        unmatched_high = [high[i] for i in um_d1]
        if unmatched_high or um_d2:
            self.diagnostics["frames_with_unmatched_detections"] += 1

        # unmatched high-conf: dormant re-ID, else spawn a new TENTATIVE
        for d in unmatched_high:
            track = self._try_dormant_reid(d, frame_index)
            if track is not None:
                self._apply_hit(track, d, frame_index)  # re-ID: genuine new observation
            else:
                track = self._spawn(d, frame_index)      # spawn already counts the detection
            hit.add(id(track))

        # unmatched low-conf: optionally spawn a LOW-CONF TENTATIVE
        if self.config.lowconf_can_spawn:
            for i in um_d2:
                hit.add(id(self._spawn(low[i], frame_index)))

        # age every live track that was NOT hit this frame (incl. DORMANT — they advance via the
        # top-of-frame predict, mark_missed only updates counters/state, never re-predicts)
        for t in self.tracks:
            if id(t) not in hit and t.state in (CONFIRMED, TENTATIVE, DORMANT):
                t.mark_missed(frame_index)

        self._promote_and_prune(frame_index)
        for t in self.tracks:
            t.flush_id()  # retroactively back-fill local_track_id on newly-confirmed tracks

    def _apply_hit(self, track: Track, det: Detection, frame_index: int) -> None:
        track.register_hit(det.bbox_xywh, det.pose, det.confidence, frame_index)
        track.record_player(det.player)

    def _promote_and_prune(self, frame_index: int) -> None:
        survivors: list[Track] = []
        for t in self.tracks:
            if t.maybe_confirm():
                self.diagnostics["confirmed_tracks"] += 1
            if t.kalman.position_cov_trace() > self.config.kalman_cov_trace_max:
                self.diagnostics["kalman_cov_explosions"] += 1
                continue  # force-delete
            if t.tentative_expired(frame_index):
                self.diagnostics["tentative_rejected"] += 1
                continue  # never confirmed within the window (incl. low-conf w/o high-conf hit)
            if t.should_delete():
                if t.state == DORMANT:
                    self.diagnostics["dormant_deleted"] += 1
                elif t.state == TENTATIVE:
                    self.diagnostics["tentative_rejected"] += 1
                continue
            survivors.append(t)
        self.tracks = survivors

    def finalize(self) -> None:
        for t in self.tracks:
            if t.state == TENTATIVE:
                if t.maybe_confirm(ignore_window=True):
                    self.diagnostics["confirmed_tracks"] += 1
                else:
                    self.diagnostics["tentative_unresolved_at_eof"] += 1
        for t in self.tracks:
            t.flush_id()  # drain any EOF-confirmed tracks' buffered players
```

> **Note on `maybe_confirm` at EOF:** the within-window check uses `frames < tentative_confirm_window`. At EOF the window is closed early — `finalize()` calls `maybe_confirm()` which still promotes if the hit count is already met. This matches spec §6 "End-of-stream finalisation".

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cricket_p2_tracker.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add pose_estimation/cricket/p2_tracker.py tests/test_cricket_p2_tracker.py
git commit -m "feat(p2): two-stage per-camera tracker with dormant re-ID"
```

---

### Task 6: JSONL I/O + retroactive buffer + diagnostics writer

**Files:**
- Create: `pose_estimation/cricket/p2_io.py`
- Test: `tests/test_cricket_p2_io.py`

**Interfaces:**
- Consumes: `P2Config`, `CameraTracker` + `Detection`, `build_pose_vector`, `validate_group1_frame` (`contract.py`).
- Produces:
  - `read_p1_frames(path: str | Path) -> Iterator[dict]` — yields parsed P1 frame records in file order.
  - `frame_to_detections(record: dict, config: P2Config) -> list[Detection]` — builds `Detection` per player, attaching the live player dict.
  - `track_camera_file(input_path, output_path, diagnostics_path, camera_id, delivery_id, config, expected_frames=600) -> dict` — full single-camera pipeline: reads frames in order, runs `CameraTracker`, writes one output line per input frame (with `local_track_id` filled), validates each line, writes the diagnostic JSON, returns the diagnostics dict. Handles the 5-frame retroactive fill: a frame is held back until its tentative tracks resolve (or `tentative_confirm_window` frames pass), then flushed in order; all buffers drained at EOF.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cricket_p2_io.py
import json
from pathlib import Path

from pose_estimation.cricket.contract import validate_group1_frame
from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_io import (
    frame_to_detections, read_p1_frames, track_camera_file,
)

CFG = P2Config()


def _player(cx, cy, conf=0.8):
    kp = [[cx + i, cy + 2 * i] for i in range(17)]
    return {
        "global_player_id": None, "local_track_id": None, "role": "unknown",
        "bbox_xywh_px": [cx - 20.0, cy - 40.0, 40.0, 80.0],
        "bbox_xywh_norm": [0.0, 0.0, 0.02, 0.06],
        "track_confidence": conf,
        "pose_2d": {"skeleton": "coco_17", "keypoints_px": kp,
                     "keypoints_norm": [[x / 2560, y / 1440] for x, y in kp],
                     "confidence": [0.9] * 17},
        "pose_3d": None,
    }


def _frame(frame_index, players):
    return {
        "schema_version": "g1_player_frame/v0", "match_id": "CCPL080626",
        "delivery_id": "CCPL080626M1_1_14_1", "camera_id": "cam_01",
        "frame_index": frame_index,
        "frame_name": f"frame_camera01_{frame_index:09d}.jpg",
        "players": players,
    }


def _write_jsonl(path: Path, frames):
    with path.open("w", encoding="utf-8") as h:
        for f in frames:
            h.write(json.dumps(f) + "\n")


def test_read_p1_frames_round_trips(tmp_path: Path):
    path = tmp_path / "cam_01.jsonl"
    _write_jsonl(path, [_frame(i, [_player(100, 200)]) for i in range(3)])
    frames = list(read_p1_frames(path))
    assert len(frames) == 3
    assert frames[0]["frame_index"] == 0


def test_frame_to_detections_builds_pose(tmp_path: Path):
    dets = frame_to_detections(_frame(0, [_player(100, 200)]), CFG)
    assert len(dets) == 1
    assert dets[0].pose.defined


def test_track_camera_file_fills_ids_and_validates(tmp_path: Path):
    in_path = tmp_path / "cam_01.jsonl"
    out_path = tmp_path / "out" / "cam_01.jsonl"
    diag_path = tmp_path / "out" / "diagnostics" / "cam_01.json"
    _write_jsonl(in_path, [_frame(i, [_player(100 + 5 * i, 200)]) for i in range(8)])

    diag = track_camera_file(in_path, out_path, diag_path, "cam_01",
                             "CCPL080626M1_1_14_1", CFG, expected_frames=8)

    out_frames = list(read_p1_frames(out_path))
    assert len(out_frames) == 8  # one output line per input frame
    for fr in out_frames:
        validate_group1_frame(fr)
    ids = {p["local_track_id"] for fr in out_frames for p in fr["players"] if p["local_track_id"]}
    assert len(ids) == 1  # single stable track
    assert diag["frames_read"] == 8
    assert diag_path.exists()


def test_output_line_count_equals_input_even_with_dropped_tracks(tmp_path: Path):
    in_path = tmp_path / "cam_01.jsonl"
    out_path = tmp_path / "out" / "cam_01.jsonl"
    diag_path = tmp_path / "out" / "diag" / "cam_01.json"
    # one isolated low-conf detection that never confirms -> id stays null, line still emitted
    frames = [_frame(0, [_player(100, 200, conf=0.2)])] + [_frame(i, []) for i in range(1, 4)]
    _write_jsonl(in_path, frames)
    track_camera_file(in_path, out_path, diag_path, "cam_01",
                      "CCPL080626M1_1_14_1", CFG, expected_frames=4)
    out_frames = list(read_p1_frames(out_path))
    assert len(out_frames) == 4
    assert out_frames[0]["players"][0]["local_track_id"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cricket_p2_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pose_estimation.cricket.p2_io'`

- [ ] **Step 3: Write minimal implementation**

```python
# pose_estimation/cricket/p2_io.py
"""Phase 2 JSONL I/O, retroactive write buffer, and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from pose_estimation.cricket.contract import validate_group1_frame
from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_pose_vector import build_pose_vector
from pose_estimation.cricket.p2_tracker import CameraTracker, Detection


def read_p1_frames(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def frame_to_detections(record: dict, config: P2Config) -> list[Detection]:
    detections: list[Detection] = []
    for player in record.get("players", []):
        pose_block = player.get("pose_2d", {})
        pose = build_pose_vector(
            pose_block.get("keypoints_px", []),
            pose_block.get("confidence", []),
            player.get("bbox_xywh_px", [0, 0, 0, 0]),
            config,
        )
        detections.append(
            Detection(
                bbox_xywh=list(player.get("bbox_xywh_px", [0, 0, 0, 0])),
                pose=pose,
                confidence=float(player.get("track_confidence") or 0.0),
                player=player,
            )
        )
    return detections


def track_camera_file(
    input_path: str | Path,
    output_path: str | Path,
    diagnostics_path: str | Path,
    camera_id: str,
    delivery_id: str,
    config: P2Config,
    expected_frames: int = 600,
) -> dict:
    output_path = Path(output_path)
    diagnostics_path = Path(diagnostics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = CameraTracker(camera_id, config)
    # Buffer holds (record) until its tentative tracks resolve; players are stamped in place,
    # so we can flush after `tentative_confirm_window` frames of look-ahead.
    buffer: list[dict] = []
    frames_read = 0

    with output_path.open("w", encoding="utf-8") as out:
        def flush(up_to: int) -> None:
            while len(buffer) > up_to:
                record = buffer.pop(0)
                validate_group1_frame(record)
                out.write(json.dumps(record, sort_keys=True) + "\n")

        # Drive the tracker with a per-camera ordinal (0, 1, 2, ...) so confirmation-window and
        # dormancy math are independent of P1's absolute frame_index (which may be large/sparse).
        # The output record keeps its own true frame_index untouched.
        for ordinal, record in enumerate(read_p1_frames(input_path)):
            frames_read += 1
            detections = frame_to_detections(record, config)
            tracker.update(detections, frame_index=ordinal)
            buffer.append(record)
            flush(config.tentative_confirm_window)  # keep a look-ahead window buffered

        tracker.finalize()
        flush(0)  # drain remaining buffer at EOF

    diagnostics = {
        "camera_id": camera_id,
        "delivery_id": delivery_id,
        "status": "ok",
        "error": None,
        "frames_expected": expected_frames,
        "frames_read": frames_read,
        **tracker.diagnostics,
    }
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return diagnostics
```

> **Buffer correctness:** the player dicts are shared by reference between the buffered records
> and each track's `assigned_players` list. When a tentative track confirms, `Track.flush_id()`
> (called inside `CameraTracker.update`/`finalize`) writes `local_track_id` onto every player it
> ever matched — including the earlier ones still sitting in `buffer` — before those records are
> serialised. The `tentative_confirm_window` look-ahead guarantees a track has seen its full
> confirmation window (so it has either confirmed and back-filled, or been rejected) before its
> first frame leaves the buffer. Tracks that never confirm leave their players' `local_track_id`
> as `null`, which is valid per the contract.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cricket_p2_io.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add pose_estimation/cricket/p2_io.py tests/test_cricket_p2_io.py
git commit -m "feat(p2): JSONL I/O, retroactive buffer, and diagnostics writer"
```

---

### Task 7: Parallel runner across all cameras

**Files:**
- Create: `pose_estimation/cricket/p2_runner.py`
- Test: `tests/test_cricket_p2_runner.py`

**Interfaces:**
- Consumes: `P2Config`, `track_camera_file` (Task 6).
- Produces:
  - `track_one_camera(args: tuple) -> tuple[str, str, dict, str | None]` — top-level, picklable worker. `args = (camera_id, input_path, output_path, diagnostics_path, delivery_id, config, expected_frames)`. Returns `(camera_id, status, summary_counts, error)`; catches exceptions and returns `status="failed"` with the error string (so one bad camera does not kill the pool).
  - `run_p2_tracking(input_dir, output_dir, delivery_id, config, cameras=None, expected_frames=600, max_workers=None) -> dict` — discovers `cam_XX.jsonl` files (or uses `cameras`), runs them via `ProcessPoolExecutor` (spawn-safe), returns `{camera_id: (status, summary, error)}`. `max_workers` defaults to `min(7, os.cpu_count())`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cricket_p2_runner.py
import json
from pathlib import Path

from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_runner import run_p2_tracking, track_one_camera

CFG = P2Config()


def _player(cx):
    kp = [[cx + i, 200 + 2 * i] for i in range(17)]
    return {
        "global_player_id": None, "local_track_id": None, "role": "unknown",
        "bbox_xywh_px": [cx - 20.0, 160.0, 40.0, 80.0],
        "bbox_xywh_norm": [0.0, 0.0, 0.02, 0.06], "track_confidence": 0.8,
        "pose_2d": {"skeleton": "coco_17", "keypoints_px": kp,
                     "keypoints_norm": [[x / 2560, y / 1440] for x, y in kp],
                     "confidence": [0.9] * 17},
        "pose_3d": None,
    }


def _frame(cam, idx, cx):
    return {
        "schema_version": "g1_player_frame/v0", "match_id": "CCPL080626",
        "delivery_id": "CCPL080626M1_1_14_1", "camera_id": cam, "frame_index": idx,
        "frame_name": f"frame_{cam}_{idx:09d}.jpg", "players": [_player(cx)],
    }


def _make_input(in_dir: Path, cam: str):
    in_dir.mkdir(parents=True, exist_ok=True)
    with (in_dir / f"{cam}.jsonl").open("w", encoding="utf-8") as h:
        for i in range(6):
            h.write(json.dumps(_frame(cam, i, 100 + 5 * i)) + "\n")


def test_track_one_camera_worker_succeeds(tmp_path: Path):
    in_dir = tmp_path / "p1" / "predictions"
    _make_input(in_dir, "cam_01")
    out_dir = tmp_path / "p2"
    cam, status, summary, err = track_one_camera((
        "cam_01", in_dir / "cam_01.jsonl", out_dir / "cam_01.jsonl",
        out_dir / "diagnostics" / "cam_01.json", "CCPL080626M1_1_14_1", CFG, 6,
    ))
    assert status == "ok"
    assert err is None
    assert summary["frames_read"] == 6


def test_run_p2_tracking_processes_multiple_cameras(tmp_path: Path):
    in_dir = tmp_path / "p1" / "predictions"
    for cam in ("cam_01", "cam_02"):
        _make_input(in_dir, cam)
    out_dir = tmp_path / "p2"
    results = run_p2_tracking(in_dir, out_dir, "CCPL080626M1_1_14_1", CFG,
                              expected_frames=6, max_workers=2)
    assert set(results) == {"cam_01", "cam_02"}
    for cam in ("cam_01", "cam_02"):
        assert results[cam][0] == "ok"
        assert (out_dir / f"{cam}.jsonl").exists()


def test_failed_camera_is_reported_not_raised(tmp_path: Path):
    in_dir = tmp_path / "p1" / "predictions"
    in_dir.mkdir(parents=True, exist_ok=True)
    (in_dir / "cam_01.jsonl").write_text("{ not valid json\n", encoding="utf-8")
    out_dir = tmp_path / "p2"
    results = run_p2_tracking(in_dir, out_dir, "CCPL080626M1_1_14_1", CFG,
                              expected_frames=1, max_workers=1)
    assert results["cam_01"][0] == "failed"
    assert results["cam_01"][2] is not None  # error string present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cricket_p2_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pose_estimation.cricket.p2_runner'`

- [ ] **Step 3: Write minimal implementation**

```python
# pose_estimation/cricket/p2_runner.py
"""Phase 2 parallel per-camera tracking orchestration (spawn-safe for win32)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_io import track_camera_file


def track_one_camera(args: tuple) -> tuple[str, str, dict, str | None]:
    (camera_id, input_path, output_path, diagnostics_path,
     delivery_id, config, expected_frames) = args
    try:
        diagnostics = track_camera_file(
            input_path, output_path, diagnostics_path,
            camera_id, delivery_id, config, expected_frames,
        )
        summary = {
            "frames_read": diagnostics["frames_read"],
            "confirmed_tracks": diagnostics["confirmed_tracks"],
            "total_tracks_spawned": diagnostics["total_tracks_spawned"],
        }
        return camera_id, "ok", summary, None
    except Exception as exc:  # noqa: BLE001 — surface per-camera failure, do not kill the pool
        return camera_id, "failed", {}, f"{type(exc).__name__}: {exc}"


def _discover_cameras(input_dir: Path, cameras: list[str] | None) -> list[str]:
    if cameras:
        return sorted(cameras)
    return sorted(p.stem for p in input_dir.glob("cam_*.jsonl"))


def run_p2_tracking(
    input_dir: str | Path,
    output_dir: str | Path,
    delivery_id: str,
    config: P2Config,
    cameras: list[str] | None = None,
    expected_frames: int = 600,
    max_workers: int | None = None,
) -> dict[str, tuple[str, dict, str | None]]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    camera_ids = _discover_cameras(input_dir, cameras)
    if not camera_ids:
        raise RuntimeError(f"no cam_XX.jsonl inputs found in {input_dir}")

    jobs = [
        (
            cam,
            input_dir / f"{cam}.jsonl",
            output_dir / f"{cam}.jsonl",
            output_dir / "diagnostics" / f"{cam}.json",
            delivery_id,
            config,
            expected_frames,
        )
        for cam in camera_ids
    ]
    workers = max_workers or min(7, os.cpu_count() or 1)

    results: dict[str, tuple[str, dict, str | None]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cam, status, summary, error in pool.map(track_one_camera, jobs):
            results[cam] = (status, summary, error)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cricket_p2_runner.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pose_estimation/cricket/p2_runner.py tests/test_cricket_p2_runner.py
git commit -m "feat(p2): spawn-safe parallel per-camera runner"
```

---

### Task 8: CLI entry point

**Files:**
- Create: `scripts/run_cricket_p2_tracking.py`
- Test: `tests/test_cricket_p2_cli.py`

**Interfaces:**
- Consumes: `run_p2_tracking`, `load_p2_config`.
- Produces: a CLI matching spec §8 (realised under `scripts/` to follow the repo's `scripts/run_cricket_*` convention; this is the concrete form of the spec's `tools/track_p2.py`):
  `python scripts/run_cricket_p2_tracking.py --input-dir outputs/p1/predictions --output-dir outputs/p2 --delivery-id <id> [--config configs/p2_tracking.yaml] [--camera cam_01] [--max-workers N]`.
  - `build_arg_parser() -> argparse.ArgumentParser` (testable without invoking main).
  - `main(argv: list[str] | None = None) -> int` — returns `0` when all cameras `ok`, `1` if any failed; guarded by `if __name__ == "__main__":`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cricket_p2_cli.py
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_cricket_p2_tracking", ROOT / "scripts" / "run_cricket_p2_tracking.py"
)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def _player(cx):
    kp = [[cx + i, 200 + 2 * i] for i in range(17)]
    return {
        "global_player_id": None, "local_track_id": None, "role": "unknown",
        "bbox_xywh_px": [cx - 20.0, 160.0, 40.0, 80.0],
        "bbox_xywh_norm": [0.0, 0.0, 0.02, 0.06], "track_confidence": 0.8,
        "pose_2d": {"skeleton": "coco_17", "keypoints_px": kp,
                     "keypoints_norm": [[x / 2560, y / 1440] for x, y in kp],
                     "confidence": [0.9] * 17},
        "pose_3d": None,
    }


def test_arg_parser_has_required_options():
    parser = cli.build_arg_parser()
    args = parser.parse_args([
        "--input-dir", "outputs/p1/predictions", "--output-dir", "outputs/p2",
        "--delivery-id", "D1",
    ])
    assert args.input_dir == "outputs/p1/predictions"
    assert args.delivery_id == "D1"


def test_main_runs_end_to_end_single_camera(tmp_path: Path):
    in_dir = tmp_path / "p1" / "predictions"
    in_dir.mkdir(parents=True, exist_ok=True)
    with (in_dir / "cam_01.jsonl").open("w", encoding="utf-8") as h:
        for i in range(6):
            rec = {
                "schema_version": "g1_player_frame/v0", "match_id": "CCPL080626",
                "delivery_id": "D1", "camera_id": "cam_01", "frame_index": i,
                "frame_name": f"frame_camera01_{i:09d}.jpg", "players": [_player(100 + 5 * i)],
            }
            h.write(json.dumps(rec) + "\n")
    out_dir = tmp_path / "p2"
    code = cli.main([
        "--input-dir", str(in_dir), "--output-dir", str(out_dir),
        "--delivery-id", "D1", "--camera", "cam_01", "--max-workers", "1",
        "--expected-frames", "6",
    ])
    assert code == 0
    assert (out_dir / "cam_01.jsonl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cricket_p2_cli.py -v`
Expected: FAIL — `FileNotFoundError` / spec load error (script does not exist yet)

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/run_cricket_p2_tracking.py
"""CLI entry point for Phase 2 per-camera tracking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_estimation.cricket.p2_config import load_p2_config  # noqa: E402
from pose_estimation.cricket.p2_runner import run_p2_tracking  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2 per-camera tracking")
    parser.add_argument("--input-dir", required=True, help="dir of P1 cam_XX.jsonl predictions")
    parser.add_argument("--output-dir", required=True, help="dir for P2 cam_XX.jsonl outputs")
    parser.add_argument("--delivery-id", required=True)
    parser.add_argument("--config", default=None, help="optional p2_tracking.yaml")
    parser.add_argument("--camera", action="append", default=None, help="restrict to camera(s)")
    parser.add_argument("--expected-frames", type=int, default=600)
    parser.add_argument("--max-workers", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_p2_config(args.config)
    results = run_p2_tracking(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        delivery_id=args.delivery_id,
        config=config,
        cameras=args.camera,
        expected_frames=args.expected_frames,
        max_workers=args.max_workers,
    )
    failed = []
    for cam in sorted(results):
        status, summary, error = results[cam]
        if status == "ok":
            print(f"{cam}: ok  frames={summary.get('frames_read')} "
                  f"confirmed={summary.get('confirmed_tracks')}", flush=True)
        else:
            failed.append(cam)
            print(f"{cam}: FAILED  {error}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cricket_p2_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full P2 suite + commit**

Run: `python -m pytest tests/test_cricket_p2_config.py tests/test_cricket_p2_pose_vector.py tests/test_cricket_p2_kalman.py tests/test_cricket_p2_track.py tests/test_cricket_p2_tracker.py tests/test_cricket_p2_io.py tests/test_cricket_p2_runner.py tests/test_cricket_p2_cli.py -v`
Expected: PASS (all P2 tests green)

```bash
git add scripts/run_cricket_p2_tracking.py tests/test_cricket_p2_cli.py
git commit -m "feat(p2): CLI entry point for per-camera tracking"
```

---

## Diagnostic Field Coverage Note

Spec §10 lists a richer diagnostic payload than the counters wired up in Tasks 5–6. **Fully
computed** counters: `total_tracks_spawned`, `lowconf_tracks_spawned`, `confirmed_tracks`,
`tentative_rejected`, `tentative_unresolved_at_eof`, `dormant_reidentified`,
`dormant_reid_ambiguous`, `dormant_deleted`, `pose_undefined_count`,
`pose_skipped_low_overlap`, `kalman_cov_explosions`, `frames_with_unmatched_detections`,
`frames_expected`, `frames_read`, `status`, `error` — these cover the counters P7 metrics 2 & 4
depend on.

**Stub/deferred** (present in the dict for schema stability but left at `0` / omitted, since
they are reporting niceties not required by the exit criteria §11):
- `id_switches_estimated` — initialised to `0`; the spec §10 heuristic (Stage-1 best-match track
  changed between consecutive frames while prior-track IoU > 0.3) is a post-hoc analysis better
  computed in P7 from the emitted JSONL than inline here. Wire it later in `_match` if needed.
- `scale_floor_hits` — requires `build_pose_vector` to report when the scale floor fired; add a
  `scale_floored: bool` to `PoseVector` and count it, if desired.
- `stage2_discarded` — only meaningful when `lowconf_can_spawn=false`; increment in the low-conf
  branch of `update()` under that flag.
- `per_track` block (`length_frames`, `gap_count`, `max_gap_frames`, `max_cov_trace`) — the
  `Track` object already records `gap_count`, `max_gap_frames`, `max_cov_trace`; aggregate them
  in `track_camera_file` over surviving + deleted tracks when this block is wanted.

None of these change any interface, so they can be added without disturbing Tasks 1–8.

## Exit Criteria (from spec §11)

- [ ] `local_track_id` filled for all confirmed-track detections across a full delivery (verify on a real `outputs/p1/predictions/` delivery once P1 has been run with `--conf 0.1`).
- [ ] Per-camera diagnostic logs produced under `outputs/p2/diagnostics/`.
- [ ] Single-camera debug mode verified: `python scripts/run_cricket_p2_tracking.py --input-dir outputs/p1/predictions --output-dir outputs/p2 --delivery-id <id> --camera cam_01`.
- [ ] Full parallel run completes with all 7 cameras `ok`.
