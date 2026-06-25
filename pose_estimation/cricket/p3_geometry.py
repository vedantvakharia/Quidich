"""P3 geometry primitives: DLT triangulation, epipolar, degeneracy, cost functions."""
from __future__ import annotations

import numpy as np


def triangulate_dlt(x1_px: np.ndarray, P1: np.ndarray,
                    x2_px: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """DLT triangulation. Returns [X,Y,Z] world coords.

    x1_px, x2_px: pixel coords [u, v] (not homogeneous).
    P1, P2: 3x4 projection matrices.
    """
    A = np.array([
        x1_px[0] * P1[2] - P1[0],
        x1_px[1] * P1[2] - P1[1],
        x2_px[0] * P2[2] - P2[0],
        x2_px[1] * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def reprojection_error_px(X_world: np.ndarray, P: np.ndarray,
                          x_obs_px: np.ndarray) -> float:
    """Pixel reprojection error: distance from observed pixel to projected point."""
    h = P @ np.append(np.asarray(X_world, float), 1.0)
    if abs(h[2]) < 1e-12:
        return np.inf
    proj = h[:2] / h[2]
    return float(np.linalg.norm(proj - np.asarray(x_obs_px, float)))


def condition_number_dlt(x1_px: np.ndarray, P1: np.ndarray,
                         x2_px: np.ndarray, P2: np.ndarray) -> float:
    """Condition number of the DLT linear system. Higher = worse-conditioned triangulation."""
    A = np.array([
        x1_px[0] * P1[2] - P1[0],
        x1_px[1] * P1[2] - P1[1],
        x2_px[0] * P2[2] - P2[0],
        x2_px[1] * P2[2] - P2[1],
    ])
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / (s[-1] + 1e-12))


def parallax_angle_deg(C1_world: np.ndarray, C2_world: np.ndarray,
                       X_world: np.ndarray) -> float:
    """Angle (degrees) between the two camera rays at the 3D point."""
    r1 = np.asarray(C1_world) - np.asarray(X_world)
    r2 = np.asarray(C2_world) - np.asarray(X_world)
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = float(np.clip((r1 / n1) @ (r2 / n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def compute_fundamental_matrix(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Compute F such that x2^T F x1 = 0 from two 3x4 projection matrices."""
    _, _, Vt = np.linalg.svd(P1)
    C1_h = Vt[-1]
    C1 = C1_h[:3] / (C1_h[3] + 1e-12)

    e2 = P2 @ np.append(C1, 1.0)
    e2_cross = np.array([
        [0.0,    -e2[2],  e2[1]],
        [e2[2],   0.0,   -e2[0]],
        [-e2[1],  e2[0],  0.0  ],
    ])
    F = e2_cross @ P2 @ np.linalg.pinv(P1)
    norm = np.linalg.norm(F)
    return F / (norm + 1e-12)


def compute_right_epipole(F: np.ndarray) -> np.ndarray:
    """Right epipole e2: F^T e2 = 0. Returns dehomogenized [u, v] pixel coords."""
    _, _, Vt = np.linalg.svd(F.T)
    e = Vt[-1]
    if abs(e[2]) < 1e-12:
        return np.array([np.inf, np.inf])
    return e[:2] / e[2]


def sampson_distance(x1_px: np.ndarray, F: np.ndarray, x2_px: np.ndarray) -> float:
    """Symmetric Sampson distance (approximation of geometric epipolar error, in px^2)."""
    x1h = np.array([x1_px[0], x1_px[1], 1.0])
    x2h = np.array([x2_px[0], x2_px[1], 1.0])
    Fx1 = F @ x1h
    Ftx2 = F.T @ x2h
    num = float(x2h @ Fx1) ** 2
    denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2
    return num / (denom + 1e-12)


def bbox_bottom_center_px(bbox_xywh_px: list[float]) -> np.ndarray:
    """Returns [u, v] pixel coords of the bbox bottom-center (ground contact reference)."""
    x, y, w, h = bbox_xywh_px
    return np.array([x + w / 2.0, y + h], dtype=float)


def huber_cost(r: float, delta: float) -> float:
    """Huber cost: quadratic for r < delta, linear beyond. Continuous at delta."""
    if r <= delta:
        return r ** 2 / (2.0 * delta)
    return r - delta / 2.0


def parallax_weight(parallax_deg: float, min_deg: float = 10.0,
                    full_deg: float = 25.0) -> float:
    """Weight for triangulation reliability. 0 below min_deg, 1 above full_deg, linear between."""
    if parallax_deg <= min_deg:
        return 0.0
    if parallax_deg >= full_deg:
        return 1.0
    return (parallax_deg - min_deg) / (full_deg - min_deg)
