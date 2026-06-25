# tests/test_cricket_p3_precompute.py
from __future__ import annotations
import numpy as np
import pytest
from pose_estimation.cricket.p3_precompute import (
    build_precomputed_geometry, PairGeometry, PrecomputedGeometry,
)

def _make_projection(eye, target, K=None):
    if K is None:
        K = np.array([[800,0,640],[0,800,360],[0,0,1]], dtype=float)
    z = eye - target; z /= np.linalg.norm(z)
    x = np.cross([0,1,0], z)
    if np.linalg.norm(x) < 1e-9: x = np.cross([1,0,0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ eye
    return K @ np.hstack([R, t.reshape(3,1)])

def _make_system():
    cameras = {
        "C01": np.array([5.0, 0.0, 2.0]),   # end-on
        "C02": np.array([0.0, 5.0, 2.0]),   # side
        "C04": np.array([-5.0, 0.0, 2.0]),  # opposing end-on (C01 <-> C04 degenerate)
    }
    P = {cid: _make_projection(pos, np.zeros(3)) for cid, pos in cameras.items()}
    # One survey point at origin
    survey = [{"name": "origin", "point_world_m": [0.0, 0.0, 0.0]}]
    return P, cameras, survey

def test_build_creates_all_pairs():
    P, centers, survey = _make_system()
    geo = build_precomputed_geometry(P, centers, survey, image_wh=(1280, 720))
    # 3 cameras -> 3 pairs
    assert len(geo.pairs) == 3

def test_opposing_pair_is_degenerate():
    P, centers, survey = _make_system()
    geo = build_precomputed_geometry(P, centers, survey, image_wh=(1280, 720))
    pair = geo.pairs[("C01", "C04")]
    assert pair.is_degenerate is True
    assert pair.w_epi == 0.0
    assert pair.w_tri == 1.0

def test_perpendicular_pair_not_degenerate():
    P, centers, survey = _make_system()
    geo = build_precomputed_geometry(P, centers, survey, image_wh=(1280, 720))
    pair = geo.pairs[("C01", "C02")]
    assert pair.is_degenerate is False
    assert pair.w_epi > 0.0

def test_calibration_stats_finite():
    P, centers, survey = _make_system()
    geo = build_precomputed_geometry(P, centers, survey, image_wh=(1280, 720))
    assert np.isfinite(geo.stats.mu_fine_score)
    assert np.isfinite(geo.stats.sigma_fine_score)
    assert geo.stats.sigma_fine_score > 0.0

def test_camera_centers_stored():
    P, centers, survey = _make_system()
    geo = build_precomputed_geometry(P, centers, survey, image_wh=(1280, 720))
    assert "C01" in geo.camera_centers
    assert np.allclose(geo.camera_centers["C01"], centers["C01"], atol=0.1)
