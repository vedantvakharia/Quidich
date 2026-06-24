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
