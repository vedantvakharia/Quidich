# tests/test_cricket_p3_association.py
from __future__ import annotations
import numpy as np
import pytest
from pose_estimation.cricket.p3_association import (
    AnchorState, select_anchor, build_cost_matrix, compute_track_confidence, Detection3,
)
from pose_estimation.cricket.p3_precompute import PairGeometry
from pose_estimation.cricket.p3_geometry import compute_fundamental_matrix

def _dummy_det(cam_id, conf=0.8, bbox=(100,100,80,200), kp=None):
    if kp is None:
        kp = np.zeros((17, 2))
        kp[:] = [140, 300]  # all at bbox center
    return Detection3(cam_id=cam_id, bbox_xywh_px=list(bbox),
                      keypoints_px=kp, keypoint_conf=np.full(17, conf),
                      confidence=conf)

def test_select_anchor_returns_max_detection_camera():
    dets = {"C01": [_dummy_det("C01"), _dummy_det("C01")],
            "C02": [_dummy_det("C02")]}
    state = AnchorState(anchor_id="C02", frames_since_switch=0)
    new_state = select_anchor(dets, state, hysteresis_margin=1, hysteresis_frames=3)
    # C01 has 2 detections vs C02 has 1, margin=1: C01 > C02+1 is false (2 > 1+1 = 2 is false)
    # Hysteresis: don't switch unless challenger exceeds by margin
    assert new_state.anchor_id == "C02"  # stays on C02; challenger not strictly > margin

def test_select_anchor_switches_when_margin_exceeded():
    dets = {"C01": [_dummy_det("C01"), _dummy_det("C01"), _dummy_det("C01")],
            "C02": [_dummy_det("C02")]}
    state = AnchorState(anchor_id="C02", frames_since_switch=10)
    new_state = select_anchor(dets, state, hysteresis_margin=1, hysteresis_frames=3)
    # C01 has 3, C02 has 1; 3 > 1+1 = true -> switch
    assert new_state.anchor_id == "C01"

def test_compute_track_confidence_in_range():
    from pose_estimation.cricket.p3_precompute import CalibrationStats
    stats = CalibrationStats(mu_fine_score=2.0, sigma_fine_score=1.0)
    conf = compute_track_confidence(fine_score=1.0, score_2nd_best=5.0, stats=stats)
    assert 0.0 <= conf <= 1.0

def test_compute_track_confidence_never_negative():
    from pose_estimation.cricket.p3_precompute import CalibrationStats
    stats = CalibrationStats(mu_fine_score=1.0, sigma_fine_score=0.5)
    # fine_score much larger than mu — would go negative under old linear formula
    conf = compute_track_confidence(fine_score=100.0, score_2nd_best=200.0, stats=stats)
    assert conf >= 0.0

def test_compute_track_confidence_low_when_ambiguous():
    from pose_estimation.cricket.p3_precompute import CalibrationStats
    stats = CalibrationStats(mu_fine_score=2.0, sigma_fine_score=1.0)
    # Near-tied runner-up -> low confidence
    conf_ambiguous = compute_track_confidence(1.0, 1.001, stats)
    conf_clear = compute_track_confidence(1.0, 10.0, stats)
    assert conf_ambiguous < conf_clear

def test_cost_matrix_has_dummy_column():
    """Cost matrix has N+1 columns (last = no-match dummy) and M+1 rows."""
    P1 = np.eye(3, 4); P1[0,0]=800; P1[1,1]=800; P1[0,2]=640; P1[1,2]=360
    P2 = P1.copy(); P2[0,3] = 5.0  # translate slightly
    F = compute_fundamental_matrix(P1, P2)
    pg = PairGeometry("C01","C02", F, is_degenerate=False,
                      w_epi=0.6, w_tri=0.4, huber_delta=5.0)
    from pose_estimation.cricket.p3_precompute import CalibrationStats
    stats = CalibrationStats(mu_fine_score=2.0, sigma_fine_score=1.0)
    dets_a = [_dummy_det("C01"), _dummy_det("C01")]
    dets_b = [_dummy_det("C02"), _dummy_det("C02"), _dummy_det("C02")]
    C_a = np.array([0.,0.,5.]); C_b = np.array([5.,0.,5.])
    cost = build_cost_matrix(dets_a, dets_b, P1, P2, C_a, C_b, pg, stats, tau_tri_cond=50.0)
    # Shape: (M+1) x (N+1) = 3 x 4
    assert cost.shape == (3, 4)
    assert np.isfinite(cost).all(), "No inf/nan; soft costs only"
