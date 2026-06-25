"""P4a global track dataclass and lifecycle states."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pose_estimation.cricket.p4a_kalman import SingerGroundKalman

TENTATIVE = "tentative"
CONFIRMED = "confirmed"
LOST = "lost"
DELETED = "deleted"

_CONFIRM_HITS = 3
_LOST_WINDOW = 30
_BOWLER_LOST_WINDOW = 60
_CHI2_95_2DOF = 5.991


@dataclass
class GlobalTrack:
    global_player_id: str | None
    state: str
    kalman: SingerGroundKalman
    first_frame: int
    last_frame: int
    first_ground_pos: np.ndarray
    last_ground_pos: np.ndarray
    dominant_role: str = "unknown"
    role_latch_count: int = 0
    _role_candidate: str = field(default="unknown", repr=False)
    hits: int = 1
    frames_since_update: int = 0
    last_bbox_xywh_px: list[float] | None = None
    last_pose_2d: dict | None = None

    def mark_missed(self) -> None:
        self.frames_since_update += 1
        if self.state == CONFIRMED:
            self.state = LOST
        self.kalman.predict()
        self.kalman.cap_covariance()  # Fix 2.3: cap P, don't feed virtual measurements

    def apply_hit(self, ground_xy: np.ndarray, bbox_xywh_px: list[float] | None,
                  pose_2d: dict | None, frame_index: int) -> None:
        self.kalman.update(ground_xy)
        self.last_ground_pos = ground_xy.copy()
        self.last_frame = frame_index
        self.frames_since_update = 0
        self.hits += 1
        self.last_bbox_xywh_px = bbox_xywh_px
        self.last_pose_2d = pose_2d
        if self.state == LOST:
            self.state = CONFIRMED

    def maybe_confirm(self) -> bool:
        if self.state == TENTATIVE and self.hits >= _CONFIRM_HITS:
            self.state = CONFIRMED
            return True
        return False

    def should_delete(self) -> bool:
        window = _BOWLER_LOST_WINDOW if self.dominant_role == "bowler" else _LOST_WINDOW
        return self.state == LOST and self.frames_since_update > window

    def velocity_toward_crease(self, crease_y: float = 0.0) -> bool:
        """True if bowler's velocity direction points toward the crease (y=0)."""
        vy = float(self.kalman.velocity_xy[1])
        pos_y = float(self.kalman.pos_world_xy[1])
        return (pos_y > crease_y and vy < -0.1) or (pos_y < crease_y and vy > 0.1)
