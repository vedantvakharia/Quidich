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
