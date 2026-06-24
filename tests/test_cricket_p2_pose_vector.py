# tests/test_cricket_p2_pose_vector.py
import numpy as np

from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_pose_vector import (
    build_pose_vector,
    masked_weighted_cosine,
)

CFG = P2Config()


def _kp(value=10.0):
    return [[value + i, value + 2 * i] for i in range(17)]


def test_identical_poses_have_zero_cost():
    kp = _kp()
    conf = [0.9] * 17
    bbox = [0.0, 0.0, 100.0, 200.0]
    a = build_pose_vector(kp, conf, bbox, CFG)
    b = build_pose_vector(kp, conf, bbox, CFG)
    assert a.defined and b.defined
    assert masked_weighted_cosine(a, b, min_shared_keypoints=CFG.min_shared_keypoints) < 1e-6


def test_low_confidence_keypoints_are_masked_out():
    kp = _kp()
    conf = [0.9] * 17
    conf[9] = 0.1  # below 0.3 -> invalid
    a = build_pose_vector(kp, conf, [0, 0, 100, 200], CFG)
    assert a.mask[9] == False
    assert a.mask[5] == True


def test_undefined_when_no_root_or_shoulders():
    kp = _kp()
    conf = [0.9] * 17
    for i in (5, 6, 11, 12):
        conf[i] = 0.0  # kill both hips and both shoulders
    a = build_pose_vector(kp, conf, [0, 0, 100, 200], CFG)
    assert a.defined is False


def test_too_few_shared_keypoints_returns_unit_cost():
    kp = _kp()
    conf_a = [0.9] * 17
    conf_b = [0.9] * 17
    # only keypoints 11,12 shared-valid (fewer than min_shared_keypoints=6)
    for i in range(17):
        if i not in (11, 12):
            conf_b[i] = 0.0
    a = build_pose_vector(kp, conf_a, [0, 0, 100, 200], CFG)
    b = build_pose_vector(kp, conf_b, [0, 0, 100, 200], CFG)
    assert masked_weighted_cosine(a, b, min_shared_keypoints=6) == 1.0


def test_cost_is_always_finite_and_bounded():
    kp = _kp()
    a = build_pose_vector(kp, [0.9] * 17, [0, 0, 100, 200], CFG)
    flipped = [[-x, -y] for x, y in kp]
    b = build_pose_vector(flipped, [0.9] * 17, [0, 0, 100, 200], CFG)
    cost = masked_weighted_cosine(a, b, min_shared_keypoints=6)
    assert np.isfinite(cost)
    assert 0.0 <= cost <= 2.0


def test_degenerate_scale_is_floored_not_divide_by_zero():
    # all keypoints collapse to one pixel -> every anchor length ~0
    kp = [[50.0, 50.0] for _ in range(17)]
    a = build_pose_vector(kp, [0.9] * 17, [0, 0, 100, 200], CFG)
    assert np.all(np.isfinite(a.vector))
