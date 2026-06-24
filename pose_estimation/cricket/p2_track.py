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
