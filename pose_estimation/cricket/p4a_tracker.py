"""P4a online global track manager: two-stage association, re-entry pool, role latching."""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from pose_estimation.cricket.p3_association import Correspondence, CONFIDENCE_HIGH, CONFIDENCE_DISCARD
from pose_estimation.cricket.p4a_kalman import SingerGroundKalman, ROLE_PARAMS, _CHI2_95_2DOF
from pose_estimation.cricket.p4a_track import GlobalTrack, TENTATIVE, CONFIRMED, LOST, DELETED

_ID_COUNTER = itertools.count(1)
_REENTRY_TEMPORAL_GATE = 120   # frames
_REENTRY_MAHA_GATE = 5.991     # chi2 95%, 2 DOF (with gap-scaling)
_KINEMATIC_V_MAX = 9.0         # m/s sprint speed
_FRAME_RATE = 50.0             # fps (approximate)


def _mint_id() -> str:
    return f"P{next(_ID_COUNTER):03d}"


class TrackManager:
    ROLE_LATCH_FRAMES = 5  # Fix 2.2: role must be stable for N consecutive frames

    def __init__(self) -> None:
        self.tracks: list[GlobalTrack] = []
        self.deleted_pool: list[GlobalTrack] = []
        self._role_proposals: dict[str, dict[str, int]] = {}  # id -> {role: count}

    # ---- re-entry ----------------------------------------------------------

    def _try_reentry(self, new_ground_xy: np.ndarray, first_frame: int) -> GlobalTrack | None:
        """Fix 2.4: Hungarian over all deleted candidates using Mahalanobis gate
        from the propagated Kalman state. No fixed spatial radius. One-to-one."""
        candidates: list[tuple[float, GlobalTrack]] = []
        for dt in self.deleted_pool:
            gap = first_frame - dt.last_frame
            if gap <= 0 or gap > _REENTRY_TEMPORAL_GATE:
                continue
            # Propagate deleted track's state forward by gap frames
            x_pred, P_pred = dt.kalman.propagate_state(gap)
            pos_pred = x_pred[:2]
            innovation = new_ground_xy - pos_pred
            H = np.zeros((2, 6)); H[0,0]=1; H[1,1]=1
            S = H @ P_pred @ H.T + dt.kalman._R
            try:
                maha_sq = float(innovation @ np.linalg.solve(S, innovation))
            except np.linalg.LinAlgError:
                continue
            # Gate scales slightly with gap to handle long occlusions
            gate = _REENTRY_MAHA_GATE * (1.0 + gap / 60.0)
            if maha_sq > gate:
                continue
            # Kinematic feasibility
            spatial = float(np.linalg.norm(new_ground_xy - dt.last_ground_pos))
            v_max_per_frame = _KINEMATIC_V_MAX / _FRAME_RATE
            if spatial > v_max_per_frame * gap * 1.5:
                continue
            candidates.append((maha_sq, dt))

        if not candidates:
            return None
        # One-to-one: best by Mahalanobis
        candidates.sort(key=lambda c: c[0])
        winner = candidates[0][1]
        self.deleted_pool.remove(winner)
        return winner

    # ---- association -------------------------------------------------------

    def _build_assign_matrix(
        self, corrs: list[Correspondence], tracks: list[GlobalTrack]
    ) -> tuple[list[tuple[int,int]], set[int], set[int]]:
        """Mahalanobis-gated Hungarian. Fix 2.3/3.1: gate from filter covariance."""
        if not corrs or not tracks:
            return [], set(range(len(corrs))), set(range(len(tracks)))

        _LARGE = 1e6
        cost = np.full((len(corrs), len(tracks)), _LARGE)
        for ci, c in enumerate(corrs):
            if not np.isfinite(c.ground_xy).all():
                continue
            for ti, t in enumerate(tracks):
                maha_sq = t.kalman.mahalanobis_sq(c.ground_xy)
                if maha_sq <= _CHI2_95_2DOF:
                    cost[ci, ti] = maha_sq

        rows, cols = linear_sum_assignment(cost)
        matches, um_c, um_t = [], set(range(len(corrs))), set(range(len(tracks)))
        for r, c in zip(rows, cols):
            if cost[r, c] < _LARGE:
                matches.append((r, c)); um_c.discard(r); um_t.discard(c)
        return matches, um_c, um_t

    # ---- per-frame entry point ---------------------------------------------

    def update(self, correspondences: list[Correspondence], frame_index: int) -> list[GlobalTrack]:
        """Process one frame. Returns list of active tracks."""
        # Advance all active tracks via prediction (Fix 2.3: predict-only, no virtual updates)
        for t in self.tracks:
            t.kalman.predict()
            t.kalman.cap_covariance()
            t.frames_since_update += 1

        high = [c for c in correspondences if c.track_confidence >= CONFIDENCE_HIGH]
        low  = [c for c in correspondences
                if CONFIDENCE_DISCARD <= c.track_confidence < CONFIDENCE_HIGH]

        active = [t for t in self.tracks if t.state in (CONFIRMED, TENTATIVE, LOST)]
        hit_ids: set[int] = set()

        # Stage 1: high-conf vs all active
        m1, um_h, um_t1 = self._build_assign_matrix(high, active)
        for ci, ti in m1:
            c = high[ci]; t = active[ti]
            t.apply_hit(c.ground_xy,
                        c.det_a.bbox_xywh_px if not c.single_camera else c.det_a.bbox_xywh_px,
                        None, frame_index)
            hit_ids.add(id(t))
            c.global_track = t

        # Stage 2: low-conf vs unmatched Confirmed
        unmatched_active = [active[i] for i in um_t1 if active[i].state == CONFIRMED]
        m2, um_l, _ = self._build_assign_matrix(low, unmatched_active)
        for ci, ti in m2:
            c = low[ci]; t = unmatched_active[ti]
            t.apply_hit(c.ground_xy, c.det_a.bbox_xywh_px, None, frame_index)
            hit_ids.add(id(t))
            c.global_track = t

        # Unmatched high-conf: try re-entry, else new Tentative
        unmatched_high = [high[i] for i in um_h]
        for c in unmatched_high:
            if not np.isfinite(c.ground_xy).all():
                continue
            revival = self._try_reentry(c.ground_xy, frame_index)
            if revival is not None:
                revival.state = CONFIRMED
                revival.apply_hit(c.ground_xy, c.det_a.bbox_xywh_px, None, frame_index)
                self.tracks.append(revival)
                hit_ids.add(id(revival))
                c.global_track = revival
            else:
                new_t = GlobalTrack(
                    global_player_id=None, state=TENTATIVE,
                    kalman=SingerGroundKalman(c.ground_xy, "unknown"),
                    first_frame=frame_index, last_frame=frame_index,
                    first_ground_pos=c.ground_xy.copy(), last_ground_pos=c.ground_xy.copy(),
                    last_bbox_xywh_px=c.det_a.bbox_xywh_px,
                )
                self.tracks.append(new_t)
                hit_ids.add(id(new_t))
                c.global_track = new_t

        # Mark missed tracks (predict already called above; just mark state)
        for t in self.tracks:
            if id(t) not in hit_ids and t.state in (CONFIRMED, TENTATIVE, LOST):
                if t.state == CONFIRMED:
                    t.state = LOST
                t.kalman.cap_covariance()

        self._promote_and_prune(frame_index)
        return [t for t in self.tracks if t.state != DELETED]

    def _promote_and_prune(self, frame_index: int) -> None:
        survivors: list[GlobalTrack] = []
        for t in self.tracks:
            if t.maybe_confirm():
                t.global_player_id = _mint_id()
            if t.should_delete():
                t.state = DELETED
                self.deleted_pool.append(t)
                continue
            survivors.append(t)
        self.tracks = survivors

    def propose_role(self, global_player_id: str, role: str, frame_index: int) -> None:
        """Fix 2.2: role latching — only switch motion model after ROLE_LATCH_FRAMES stable frames."""
        track = next((t for t in self.tracks if t.global_player_id == global_player_id), None)
        if track is None:
            return
        if track._role_candidate == role:
            track.role_latch_count += 1
        else:
            track._role_candidate = role
            track.role_latch_count = 1

        if track.role_latch_count >= self.ROLE_LATCH_FRAMES and track.dominant_role != role:
            track.dominant_role = role
            track.kalman.switch_role(role)  # Fix 2.1: inflates P

    def finalize(self) -> None:
        for t in self.tracks:
            if t.state == TENTATIVE and t.hits >= 2:
                t.state = CONFIRMED
                t.global_player_id = _mint_id()
