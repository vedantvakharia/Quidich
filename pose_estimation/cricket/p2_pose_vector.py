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
