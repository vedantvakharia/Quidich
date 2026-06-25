"""Tests for P3 geometry primitives."""
from __future__ import annotations

import numpy as np
import pytest

from pose_estimation.cricket.p3_geometry import (
    bbox_bottom_center_px,
    compute_fundamental_matrix,
    compute_right_epipole,
    condition_number_dlt,
    huber_cost,
    parallax_angle_deg,
    parallax_weight,
    reprojection_error_px,
    sampson_distance,
    triangulate_dlt,
)


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = eye - target
    z /= np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    if abs(z @ up) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def _make_cameras():
    """Two cameras looking at origin, roughly 90 degrees apart."""
    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    C1 = np.array([5.0, 0.0, 2.0])
    R1 = _look_at(C1, np.zeros(3))
    t1 = -R1 @ C1
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])

    C2 = np.array([0.0, 5.0, 2.0])
    R2 = _look_at(C2, np.zeros(3))
    t2 = -R2 @ C2
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])
    return P1, P2, C1, C2


def _project(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    h = P @ np.append(X, 1.0)
    return h[:2] / h[2]


def test_triangulate_dlt_round_trip():
    P1, P2, _, _ = _make_cameras()
    X_true = np.array([0.5, 0.3, 0.0])
    x1 = _project(X_true, P1)
    x2 = _project(X_true, P2)
    X_est = triangulate_dlt(x1, P1, x2, P2)
    assert np.allclose(X_est, X_true, atol=1e-4)


def test_reprojection_error_zero_for_exact():
    P1, _, _, _ = _make_cameras()
    X = np.array([0.5, 0.3, 0.0])
    x1 = _project(X, P1)
    err = reprojection_error_px(X, P1, x1)
    assert err < 1e-4


def test_condition_number_perpendicular_better_than_collinear():
    """Condition number is LOWER (better) for perpendicular cameras than near-collinear."""
    P1, P2, C1, C2 = _make_cameras()
    X = np.array([0.5, 0.3, 0.0])
    x1 = _project(X, P1)
    x2 = _project(X, P2)
    cond_good = condition_number_dlt(x1, P1, x2, P2)

    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    C_close = np.array([5.0, 0.01, 2.0])  # nearly identical to C1 -> near-collinear
    R_close = _look_at(C_close, np.zeros(3))
    t_close = -R_close @ C_close
    P_close = K @ np.hstack([R_close, t_close.reshape(3, 1)])
    x_close = _project(X, P_close)
    cond_bad = condition_number_dlt(x1, P1, x_close, P_close)

    assert cond_good < cond_bad


def test_parallax_angle_90_degrees():
    _, _, C1, C2 = _make_cameras()
    X = np.zeros(3)
    angle = parallax_angle_deg(C1, C2, X)
    assert 80.0 < angle < 100.0


def test_fundamental_matrix_epipolar_constraint():
    P1, P2, _, _ = _make_cameras()
    F = compute_fundamental_matrix(P1, P2)
    X = np.array([0.5, 0.3, 0.0])
    x1 = _project(X, P1)
    x2 = _project(X, P2)
    x1h = np.append(x1, 1.0)
    x2h = np.append(x2, 1.0)
    assert abs(x2h @ F @ x1h) < 1e-4


def test_sampson_distance_small_for_true_match():
    P1, P2, _, _ = _make_cameras()
    F = compute_fundamental_matrix(P1, P2)
    X = np.array([0.5, 0.3, 0.0])
    x1 = _project(X, P1)
    x2 = _project(X, P2)
    sd = sampson_distance(x1, F, x2)
    assert sd < 1e-4


def test_epipole_outside_image_for_perpendicular_cameras():
    P1, P2, _, _ = _make_cameras()
    F = compute_fundamental_matrix(P1, P2)
    e2 = compute_right_epipole(F)
    W, H = 1280.0, 720.0
    outside = not (0.0 <= e2[0] <= W and 0.0 <= e2[1] <= H)
    assert outside


def test_bbox_bottom_center():
    bc = bbox_bottom_center_px([100.0, 200.0, 80.0, 240.0])
    assert np.allclose(bc, [140.0, 440.0])


def test_huber_cost_quadratic_near_zero():
    assert abs(huber_cost(0.0, 5.0)) < 1e-12
    assert abs(huber_cost(1.0, 5.0) - 0.1) < 1e-9


def test_huber_cost_linear_beyond_delta():
    assert abs(huber_cost(10.0, 5.0) - 7.5) < 1e-9
    assert abs(huber_cost(5.0, 5.0) - 2.5) < 1e-9


def test_parallax_weight_zero_below_min():
    assert parallax_weight(5.0, min_deg=10.0) == 0.0


def test_parallax_weight_full_above_threshold():
    assert parallax_weight(30.0, min_deg=10.0) == 1.0
