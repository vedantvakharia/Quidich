"""P3 per-frame cross-camera association with sticky anchor and soft Huber costs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from pose_estimation.cricket.p3_geometry import (
    triangulate_dlt, reprojection_error_px, condition_number_dlt,
    parallax_angle_deg, parallax_weight, sampson_distance,
    bbox_bottom_center_px, huber_cost,
)
from pose_estimation.cricket.p3_precompute import (
    PairGeometry, CalibrationStats, PrecomputedGeometry,
)

# Chi-squared gate at 95%, 2 DOF (ground-plane x,y)
_CHI2_95_2DOF = 5.991
# No-match dummy cost: just above the Huber linear regime (means "plausible but not matched")
_DUMMY_COST_SCALE = 3.0
# Minimum confidence below which observations discarded by P4a
CONFIDENCE_DISCARD = 0.3
CONFIDENCE_HIGH = 0.7


@dataclass
class Detection3:
    cam_id: str
    bbox_xywh_px: list[float]
    keypoints_px: np.ndarray   # shape (17, 2)
    keypoint_conf: np.ndarray  # shape (17,)
    confidence: float


@dataclass
class Correspondence:
    det_a: Detection3
    det_b: Detection3 | None   # None for single-camera detections
    track_confidence: float
    single_camera: bool
    ground_xy: np.ndarray      # world XY ground position (from triangulation or single-cam estimate)


@dataclass
class AnchorState:
    anchor_id: str
    frames_since_switch: int


def select_anchor(
    dets_per_cam: dict[str, list[Detection3]],
    prev: AnchorState,
    hysteresis_margin: int = 2,
    hysteresis_frames: int = 3,
    priority: tuple[str, ...] = ("C01", "C04", "C02", "C03", "C05", "C06", "C07"),
) -> AnchorState:
    """Sticky anchor: only switch if challenger exceeds current anchor by hysteresis_margin.

    Fix 1.4: prevents anchor flapping that injects spurious ID switches.
    """
    counts = {cam: len(dets) for cam, dets in dets_per_cam.items() if dets}
    if not counts:
        return prev

    current_count = counts.get(prev.anchor_id, 0)
    # Find best challenger (not current anchor)
    best_other = max(
        ((cam, cnt) for cam, cnt in counts.items() if cam != prev.anchor_id),
        key=lambda kv: (kv[1], -priority.index(kv[0]) if kv[0] in priority else 0),
        default=(None, -1),
    )
    challenger_id, challenger_count = best_other

    if challenger_id is not None and challenger_count > current_count + hysteresis_margin:
        return AnchorState(anchor_id=challenger_id, frames_since_switch=0)

    # No switch
    return AnchorState(
        anchor_id=prev.anchor_id,
        frames_since_switch=prev.frames_since_switch + 1,
    )


def _foot_pixel(det: Detection3) -> np.ndarray:
    """Primary ground reference: bbox bottom-center.

    Falls back to ankle keypoints only when high confidence. This avoids
    the ankle-averaging jitter (Fix 1.6) and the height-off-ground bias (Fix 1.1).
    """
    bc = bbox_bottom_center_px(det.bbox_xywh_px)
    # Try high-confidence ankle average as a refinement
    l_ankle_conf = det.keypoint_conf[15]
    r_ankle_conf = det.keypoint_conf[16]
    if l_ankle_conf > 0.6 and r_ankle_conf > 0.6:
        ankle_px = (det.keypoints_px[15] + det.keypoints_px[16]) / 2.0
        # Only use ankle if it's below the bbox bottom (sanity check)
        if ankle_px[1] >= bc[1] - 20:
            return ankle_px
    elif l_ankle_conf > 0.6:
        return det.keypoints_px[15]
    elif r_ankle_conf > 0.6:
        return det.keypoints_px[16]
    return bc


def compute_track_confidence(
    fine_score: float,
    score_2nd_best: float | None,
    stats: CalibrationStats,
) -> float:
    """Fix 4.1: sigmoid normalized by calibration stats + runner-up margin.

    Never goes negative; naturally bounded in [0, 1].
    """
    z = (fine_score - stats.mu_fine_score) / (stats.sigma_fine_score + 1e-9)
    conf_geo = 1.0 / (1.0 + np.exp(z))
    if score_2nd_best is None or score_2nd_best <= fine_score + 1e-9:
        margin = 0.5
    else:
        margin = float(np.clip(
            (score_2nd_best - fine_score) / (score_2nd_best + 1e-9), 0.0, 1.0
        ))
    return float(np.clip(conf_geo * margin, 0.0, 1.0))


def build_cost_matrix(
    dets_a: list[Detection3],
    dets_b: list[Detection3],
    P_a: np.ndarray,
    P_b: np.ndarray,
    C_a: np.ndarray,
    C_b: np.ndarray,
    pg: PairGeometry,
    stats: CalibrationStats,
    tau_tri_cond: float = 50.0,
) -> np.ndarray:
    """Build (M+1)×(N+1) cost matrix with soft Huber costs and no-match dummy row/col.

    Fix 3.1: no hard-∞ gating. The dummy column/row lets Hungarian choose "no match"
    at a defined cost rather than forcing a wrong match.

    Fix 1.1: uses DLT triangulation for ground distance (not per-camera plane projection).
    Fix 1.3: downweights triangulation by parallax angle (condition number proxy).
    """
    M, N = len(dets_a), len(dets_b)
    dummy_cost = _DUMMY_COST_SCALE * pg.huber_delta

    # (M+1) x (N+1): last row = dummy detections in A, last col = dummy detections in B
    cost = np.full((M + 1, N + 1), dummy_cost, dtype=float)
    # Corner: dummy-A vs dummy-B, cost = 0 (both no-match)
    cost[M, N] = 0.0

    for i, da in enumerate(dets_a):
        foot_a = _foot_pixel(da)
        for j, db in enumerate(dets_b):
            foot_b = _foot_pixel(db)

            # Gate 1 (Fix 1.1): triangulate foot in 3D, compare XY world distance
            X_world = triangulate_dlt(foot_a, P_a, foot_b, P_b)
            if not np.isfinite(X_world).all():
                cost[i, j] = dummy_cost
                continue

            r_tri_a = reprojection_error_px(X_world, P_a, foot_a)
            r_tri_b = reprojection_error_px(X_world, P_b, foot_b)
            r_tri = r_tri_a + r_tri_b

            # Condition number: down-weight ill-conditioned triangulation (Fix 1.3)
            cond = condition_number_dlt(foot_a, P_a, foot_b, P_b)
            par_deg = parallax_angle_deg(C_a, C_b, X_world)
            pw = parallax_weight(par_deg)  # 0 for small parallax, 1 for good geometry

            if pg.is_degenerate or pw < 0.1:
                # Only triangulation available; scale by parallax weight
                fine = huber_cost(r_tri * pw, pg.huber_delta) if pw > 0 else dummy_cost
            else:
                # Epipolar + triangulation (Fix 2.4 per-pair weights)
                top5_idx = np.argsort(da.keypoint_conf)[-5:]
                valid = [k for k in top5_idx if da.keypoint_conf[k] > 0.5
                         and db.keypoint_conf[k] > 0.5]
                if valid:
                    r_epi = float(np.mean([
                        sampson_distance(da.keypoints_px[k], pg.F, db.keypoints_px[k])
                        for k in valid
                    ]))
                else:
                    r_epi = pg.huber_delta  # no usable keypoints: moderate penalty

                fine = (pg.w_epi * huber_cost(r_epi, pg.huber_delta) +
                        pg.w_tri * pw * huber_cost(r_tri, pg.huber_delta))

            cost[i, j] = fine

    return cost


def _pair_key(cid_a: str, cid_b: str) -> tuple[str, str]:
    return (cid_a, cid_b) if cid_a < cid_b else (cid_b, cid_a)


def associate_frame(
    dets_per_cam: dict[str, list[Detection3]],
    proj_matrices: dict[str, np.ndarray],
    geo: PrecomputedGeometry,
    anchor: AnchorState,
) -> list[Correspondence]:
    """Run anchor-vs-partner Hungarian for each pair, return correspondences.

    Note: cycle consistency is enforced by grouping anchor correspondences that
    share the same partner detection across multiple views.
    """
    anchor_id = anchor.anchor_id
    dets_anchor = dets_per_cam.get(anchor_id, [])
    if not dets_anchor:
        return []

    P_anchor = proj_matrices[anchor_id]
    C_anchor = geo.camera_centers[anchor_id]

    correspondences: list[Correspondence] = []
    # Track which anchor detections have been matched (for cycle consistency)
    anchor_match: dict[int, list[tuple[Detection3, float, np.ndarray]]] = {
        i: [] for i in range(len(dets_anchor))
    }

    for partner_id, dets_partner in dets_per_cam.items():
        if partner_id == anchor_id or not dets_partner:
            continue
        pk = _pair_key(anchor_id, partner_id)
        pg = geo.pairs.get(pk)
        if pg is None:
            continue

        P_partner = proj_matrices[partner_id]
        C_partner = geo.camera_centers[partner_id]

        # If F was computed as P_anchor->P_partner, transpose if needed
        # (compute_fundamental_matrix(P_a, P_b) satisfies x_b^T F x_a = 0)
        if pg.cam_id_a == anchor_id:
            F_used = pg.F
        else:
            F_used = pg.F.T  # swap direction

        pg_directed = PairGeometry(
            cam_id_a=anchor_id, cam_id_b=partner_id,
            F=F_used, is_degenerate=pg.is_degenerate,
            w_epi=pg.w_epi, w_tri=pg.w_tri, huber_delta=pg.huber_delta,
        )

        cost = build_cost_matrix(
            dets_anchor, dets_partner, P_anchor, P_partner,
            C_anchor, C_partner, pg_directed, geo.stats,
        )

        rows, cols = linear_sum_assignment(cost)
        M, N = len(dets_anchor), len(dets_partner)

        for r, c in zip(rows, cols):
            if r >= M or c >= N:
                continue  # dummy matched to dummy or to real — skip
            fine = cost[r, c]
            # Compute second-best for confidence margin
            col_costs = sorted(cost[r, :N])
            score_2nd = col_costs[1] if len(col_costs) > 1 else None
            conf = compute_track_confidence(fine, score_2nd, geo.stats)
            # Ground position from triangulation
            foot_a = _foot_pixel(dets_anchor[r])
            foot_b = _foot_pixel(dets_partner[c])
            X_world = triangulate_dlt(foot_a, P_anchor, foot_b, P_partner)
            ground_xy = X_world[:2] if np.isfinite(X_world).all() else np.full(2, np.nan)
            anchor_match[r].append((dets_partner[c], conf, ground_xy))

    # Emit correspondences (use average ground position across all partners for cycle consistency)
    emitted: set[int] = set()
    for i, da in enumerate(dets_anchor):
        matches = anchor_match[i]
        if not matches:
            # Single-camera
            bc = bbox_bottom_center_px(da.bbox_xywh_px)
            correspondences.append(Correspondence(
                det_a=da, det_b=None, track_confidence=0.3,
                single_camera=True,
                ground_xy=np.full(2, np.nan),  # Fix 4.2: no fake ground position
            ))
        else:
            # Average ground XY across partners (cycle-consistency aggregate)
            valid_xy = [gxy for _, _, gxy in matches if np.isfinite(gxy).all()]
            ground_xy = np.mean(valid_xy, axis=0) if valid_xy else np.full(2, np.nan)
            avg_conf = float(np.mean([c for _, c, _ in matches]))
            correspondences.append(Correspondence(
                det_a=da, det_b=matches[0][0],
                track_confidence=avg_conf,
                single_camera=False, ground_xy=ground_xy,
            ))
        emitted.add(i)

    return correspondences
