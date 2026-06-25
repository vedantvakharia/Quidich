"""P3 offline precomputation: F matrices, degeneracy flags, calibration stats."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from pose_estimation.cricket.p3_geometry import (
    compute_fundamental_matrix, compute_right_epipole, huber_cost,
    reprojection_error_px, triangulate_dlt, parallax_angle_deg, parallax_weight,
    bbox_bottom_center_px,
)

# Degenerate if epipole inside image OR ray-angle between cameras < threshold
_BASELINE_ANGLE_DEGEN_DEG = 20.0
# Huber delta calibrated as 90th-pct fine score of true matches (scaled below by stats)
_HUBER_DELTA_FALLBACK = 5.0


@dataclass
class PairGeometry:
    cam_id_a: str
    cam_id_b: str
    F: np.ndarray          # 3x3 fundamental matrix (x_B^T F x_A = 0)
    is_degenerate: bool
    w_epi: float           # weight for epipolar residual in fine score
    w_tri: float           # weight for triangulation residual (parallax-adjusted at runtime)
    huber_delta: float     # Huber transition point (calibrated from survey points)


@dataclass
class CalibrationStats:
    mu_fine_score: float    # mean fine score for known-correct matches on survey points
    sigma_fine_score: float # std of fine score for known-correct matches


@dataclass
class PrecomputedGeometry:
    pairs: dict[tuple[str, str], PairGeometry]
    camera_centers: dict[str, np.ndarray]
    stats: CalibrationStats


def _camera_center_from_P(P: np.ndarray) -> np.ndarray:
    """Extract 3D camera center (right null vector of P)."""
    _, _, Vt = np.linalg.svd(P)
    C_h = Vt[-1]
    return C_h[:3] / (C_h[3] + 1e-12)


def _is_degenerate(F: np.ndarray, C_a: np.ndarray, C_b: np.ndarray,
                   image_wh: tuple[int, int]) -> bool:
    """Degenerate if the right epipole is inside the image OR the baseline angle is small."""
    W, H = image_wh
    e2 = compute_right_epipole(F)
    epipole_in_image = (np.isfinite(e2).all() and
                        0.0 <= e2[0] <= W and 0.0 <= e2[1] <= H)
    baseline = C_b - C_a
    bl_norm = np.linalg.norm(baseline)
    if bl_norm < 1e-9:
        return True
    # Angle between vectors from midpoint to each camera
    mid = (C_a + C_b) / 2.0
    r_a = C_a - mid; r_b = C_b - mid
    na, nb = np.linalg.norm(r_a), np.linalg.norm(r_b)
    if na < 1e-9 or nb < 1e-9:
        small_baseline = True
    else:
        cos_a = float(np.clip((r_a / na) @ (r_b / nb), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_a)))
        small_baseline = angle_deg < _BASELINE_ANGLE_DEGEN_DEG
    return bool(epipole_in_image or small_baseline)


def _pair_weights(is_degenerate: bool, C_a: np.ndarray, C_b: np.ndarray,
                  X_ref: np.ndarray) -> tuple[float, float]:
    """Returns (w_epi, w_tri). Epipolar weight zero for degenerate pairs."""
    if is_degenerate:
        return 0.0, 1.0
    # Base weights: both reliable
    return 0.6, 0.4


def _compute_calibration_stats(
    proj_matrices: dict[str, np.ndarray],
    camera_centers: dict[str, np.ndarray],
    survey_points: list[dict[str, Any]],
    pairs: dict[tuple[str, str], PairGeometry],
) -> CalibrationStats:
    """Compute mu/sigma of fine scores on known-correct survey-point matches."""
    fine_scores: list[float] = []
    for (cid_a, cid_b), pg in pairs.items():
        P_a = proj_matrices[cid_a]
        P_b = proj_matrices[cid_b]
        for sp in survey_points:
            X_true = np.asarray(sp["point_world_m"], float)
            # Project to pixel coords in each camera
            def proj(X, P):
                h = P @ np.append(X, 1.0)
                return h[:2] / h[2] if abs(h[2]) > 1e-12 else None
            x_a = proj(X_true, P_a)
            x_b = proj(X_true, P_b)
            if x_a is None or x_b is None:
                continue
            # Triangulation residual
            X_tri = triangulate_dlt(x_a, P_a, x_b, P_b)
            if not np.isfinite(X_tri).all():
                continue
            r_tri = (reprojection_error_px(X_tri, P_a, x_a) +
                     reprojection_error_px(X_tri, P_b, x_b))
            # Parallax-adjusted triangulation weight
            par_deg = parallax_angle_deg(camera_centers[cid_a], camera_centers[cid_b], X_tri)
            pw = parallax_weight(par_deg)
            if pg.is_degenerate or pw < 0.1:
                fine = r_tri  # only tri for degenerate pairs
            else:
                from pose_estimation.cricket.p3_geometry import sampson_distance
                r_epi = sampson_distance(x_a, pg.F, x_b)
                fine = pg.w_epi * r_epi + pg.w_tri * pw * r_tri
            fine_scores.append(fine)
    if len(fine_scores) < 2:
        return CalibrationStats(mu_fine_score=1.0, sigma_fine_score=1.0)
    arr = np.asarray(fine_scores)
    return CalibrationStats(
        mu_fine_score=float(np.mean(arr)),
        sigma_fine_score=float(max(np.std(arr), 1e-3)),
    )


def build_precomputed_geometry(
    projection_matrices: dict[str, np.ndarray],
    camera_centers: dict[str, np.ndarray],
    survey_points: list[dict[str, Any]],
    image_wh: tuple[int, int] = (2560, 1440),
) -> PrecomputedGeometry:
    """Build all per-pair geometry, degeneracy flags, and calibration stats offline."""
    cam_ids = sorted(projection_matrices.keys())
    pairs: dict[tuple[str, str], PairGeometry] = {}

    # Use first survey point centroid as reference world point for pair analysis
    if survey_points:
        X_ref = np.mean([sp["point_world_m"] for sp in survey_points], axis=0)
    else:
        X_ref = np.zeros(3)

    for cid_a, cid_b in combinations(cam_ids, 2):
        P_a = projection_matrices[cid_a]
        P_b = projection_matrices[cid_b]
        C_a = camera_centers.get(cid_a, _camera_center_from_P(P_a))
        C_b = camera_centers.get(cid_b, _camera_center_from_P(P_b))
        F = compute_fundamental_matrix(P_a, P_b)
        degen = _is_degenerate(F, C_a, C_b, image_wh)
        w_epi, w_tri = _pair_weights(degen, C_a, C_b, X_ref)
        pairs[(cid_a, cid_b)] = PairGeometry(
            cam_id_a=cid_a, cam_id_b=cid_b,
            F=F, is_degenerate=degen,
            w_epi=w_epi, w_tri=w_tri,
            huber_delta=_HUBER_DELTA_FALLBACK,
        )

    stats = _compute_calibration_stats(projection_matrices, camera_centers, survey_points, pairs)

    # Update Huber deltas using empirical 90th-percentile of correct-match scores
    delta_calibrated = float(stats.mu_fine_score + 1.645 * stats.sigma_fine_score)
    for pg in pairs.values():
        object.__setattr__(pg, "huber_delta", delta_calibrated)  # PairGeometry is mutable

    return PrecomputedGeometry(pairs=pairs, camera_centers=camera_centers, stats=stats)
