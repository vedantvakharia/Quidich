# tests/test_cricket_p2_config.py
from pathlib import Path

import pytest

from pose_estimation.cricket.p2_config import P2Config, load_p2_config


def test_defaults_match_spec():
    cfg = load_p2_config(None)
    assert cfg.stage1_confidence_threshold == 0.5
    assert cfg.stage2_confidence_min == 0.1
    assert cfg.cost_accept_threshold == 0.7
    assert cfg.iou_alpha == 0.6
    assert cfg.pose_beta == 0.4
    assert cfg.min_shared_keypoints == 6
    assert cfg.dormant_max_frames == 60
    assert cfg.gallery_repr == "medoid"


def test_yaml_overrides_only_named_keys(tmp_path: Path):
    path = tmp_path / "p2.yaml"
    path.write_text("dormant_max_frames: 90\nv_max_px_per_frame: 200\n", encoding="utf-8")
    cfg = load_p2_config(path)
    assert cfg.dormant_max_frames == 90
    assert cfg.v_max_px_per_frame == 200
    assert cfg.iou_alpha == 0.6  # untouched default


def test_unknown_key_raises(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("nonsense_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_p2_config(path)


def test_stage2_floor_must_not_undercut_intent():
    cfg = load_p2_config(None)
    # coupling guard from spec §5 / Global Constraints
    assert cfg.stage2_confidence_min <= cfg.stage1_confidence_threshold


def test_yaml_type_coercion(tmp_path: Path):
    path = tmp_path / "coerce.yaml"
    path.write_text(
        "kalman_cov_trace_max: '1.0e6'\n"
        "dormant_max_frames: '45'\n"
        "stage1_confidence_threshold: 0.6\n",
        encoding="utf-8"
    )
    cfg = load_p2_config(path)
    assert isinstance(cfg.kalman_cov_trace_max, float)
    assert cfg.kalman_cov_trace_max == 1.0e6
    assert isinstance(cfg.dormant_max_frames, int)
    assert cfg.dormant_max_frames == 45

