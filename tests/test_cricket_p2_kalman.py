# tests/test_cricket_p2_kalman.py
import numpy as np

from pose_estimation.cricket.p2_kalman import KalmanBoxTracker


def _xywh_topleft(cx, cy, w, h):
    return np.array([cx - w / 2, cy - h / 2, w, h])


def test_predicted_bbox_matches_seed_before_motion():
    kf = KalmanBoxTracker([100.0, 200.0, 40.0, 80.0])  # top-left xywh
    kf.predict()
    bbox = kf.predicted_bbox()
    assert np.allclose(bbox, [100.0, 200.0, 40.0, 80.0], atol=1.0)


def test_constant_velocity_extrapolates_forward():
    kf = KalmanBoxTracker([100.0, 200.0, 40.0, 80.0])
    for step in range(1, 6):
        kf.predict()
        kf.update([100.0 + 10 * step, 200.0, 40.0, 80.0])  # moving +10px/frame in x
    kf.predict()
    cx, _ = kf.center()
    assert cx > 150.0  # extrapolated beyond last observation


def test_gating_distance_grows_when_dormant():
    kf = KalmanBoxTracker([100.0, 200.0, 40.0, 80.0])
    kf.predict()
    near = kf.gating_distance_sq(np.array([105.0, 205.0]))
    kf.inflate_process_noise(10.0)
    kf.predict()
    relaxed = kf.gating_distance_sq(np.array([105.0, 205.0]))
    assert relaxed < near  # bigger covariance -> smaller Mahalanobis distance


def test_cov_trace_is_finite_and_positive():
    kf = KalmanBoxTracker([100.0, 200.0, 40.0, 80.0])
    kf.predict()
    assert np.isfinite(kf.position_cov_trace())
    assert kf.position_cov_trace() > 0.0


def test_reseed_retains_velocity():
    kf = KalmanBoxTracker([100.0, 200.0, 40.0, 80.0])
    for step in range(1, 4):
        kf.predict()
        kf.update([100.0 + 10 * step, 200.0, 40.0, 80.0])
    v_before = kf.velocity().copy()
    kf.reseed([300.0, 200.0, 40.0, 80.0], keep_velocity=v_before)
    assert np.allclose(kf.velocity(), v_before)
    assert np.allclose(kf.center(), [320.0, 240.0], atol=1.0)  # cx = 300+40/2=320, cy = 200+80/2=240
