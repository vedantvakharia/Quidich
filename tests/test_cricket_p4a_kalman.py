# tests/test_cricket_p4a_kalman.py
from __future__ import annotations
import numpy as np
import pytest
from pose_estimation.cricket.p4a_kalman import SingerGroundKalman, ROLE_PARAMS

def test_predict_increases_covariance():
    k = SingerGroundKalman(np.array([0.0, 0.0]), "fielder")
    P_before = k._P.copy()
    k.predict()
    assert np.trace(k._P) > np.trace(P_before)

def test_update_decreases_uncertainty():
    k = SingerGroundKalman(np.array([0.0, 0.0]), "fielder")
    k.predict()
    P_before = k._P.copy()
    k.update(np.array([0.1, 0.1]))
    assert np.trace(k._P) < np.trace(P_before)

def test_mahalanobis_zero_at_prediction():
    k = SingerGroundKalman(np.array([1.0, 2.0]), "fielder")
    k.predict()
    pred = k.pos_world_xy
    # Mahalanobis at exactly the predicted position should be very small
    maha = k.mahalanobis_sq(pred)
    assert maha < 1e-6

def test_switch_role_inflates_covariance():
    k = SingerGroundKalman(np.array([0.0, 0.0]), "wicketkeeper")
    # Run many updates to shrink covariance
    for _ in range(20):
        k.predict(); k.update(np.array([0.0, 0.0]))
    P_before = np.trace(k._P)
    k.switch_role("bowler")
    P_after = np.trace(k._P)
    assert P_after > P_before  # Fix 2.1: P must inflate on model switch

def test_switch_role_recomputes_F_Q():
    k = SingerGroundKalman(np.array([0.0, 0.0]), "wicketkeeper")
    F_before = k._F.copy()
    k.switch_role("bowler")
    # bowler has higher alpha -> different F
    assert not np.allclose(k._F, F_before)

def test_cap_covariance_limits_growth():
    k = SingerGroundKalman(np.array([0.0, 0.0]), "fielder")
    for _ in range(200):
        k.predict()  # no updates: covariance grows
    k.cap_covariance(max_pos_var=25.0)
    assert k._P[0, 0] <= 25.0 + 1e-9
    assert k._P[1, 1] <= 25.0 + 1e-9

def test_propagate_state_does_not_mutate():
    k = SingerGroundKalman(np.array([1.0, 2.0]), "fielder")
    x_orig = k.pos_world_xy.copy()
    k.propagate_state(10)
    assert np.allclose(k.pos_world_xy, x_orig)  # original state unchanged

def test_striker_role_has_reasonable_Q():
    # Fix 2.1: striker must not have near-zero Q (they sprint between wickets)
    params = ROLE_PARAMS["striker"]
    assert params.sigma_a > 0.1  # not "near-zero" — must handle sprints

def test_all_roles_defined():
    for role in ("bowler", "striker", "non_striker", "wicketkeeper", "umpire", "fielder", "unknown"):
        assert role in ROLE_PARAMS
