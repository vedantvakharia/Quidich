# tests/test_cricket_p4b_flow.py
from __future__ import annotations
import numpy as np
import pytest
from pose_estimation.cricket.p4b_flow import (
    Segment, build_flow_graph, solve_flow, extract_segments,
)

def _seg(sid, start, end, pos_first, pos_last, role="unknown"):
    return Segment(
        seg_id=sid, global_player_id=f"P{sid:03d}",
        start_frame=start, end_frame=end,
        first_ground_pos=np.asarray(pos_first, float),
        last_ground_pos=np.asarray(pos_last, float),
        dominant_role=role,
        exit_velocity=np.array([0.0, 0.0]),
    )

def test_obvious_chain_gets_linked():
    """Segment A ends at t=10 at pos [0,0]; B starts at t=12 at pos [0.2,0]. Should link."""
    segs = [
        _seg(1, 0, 10, [0.0, 0.0], [0.0, 0.0]),
        _seg(2, 12, 20, [0.2, 0.0], [1.0, 0.0]),
    ]
    G = build_flow_graph(segs, w_t=0.1, w_s=1.0, w_r=100, v_max_mps=9.0, frame_rate=50.0)
    links = solve_flow(G)
    assert links.get(1) == 2  # seg 1 -> seg 2

def test_time_overlap_not_linked():
    """Overlapping segments must never be linked."""
    segs = [
        _seg(1, 0, 15, [0.0, 0.0], [0.0, 0.0]),
        _seg(2, 10, 20, [0.2, 0.0], [1.0, 0.0]),  # overlaps with seg 1
    ]
    G = build_flow_graph(segs, w_t=0.1, w_s=1.0, w_r=100, v_max_mps=9.0, frame_rate=50.0)
    links = solve_flow(G)
    assert links.get(1) != 2  # must not link

def test_kinematic_impossibility_not_linked():
    """Segments too far apart in space/time to be the same player must not link."""
    segs = [
        _seg(1, 0, 10, [0.0, 0.0], [0.0, 0.0]),
        _seg(2, 11, 20, [50.0, 0.0], [51.0, 0.0]),  # 50m away in 1 frame — impossible
    ]
    G = build_flow_graph(segs, w_t=0.1, w_s=1.0, w_r=100, v_max_mps=9.0, frame_rate=50.0)
    links = solve_flow(G)
    assert links.get(1) != 2

def test_role_mismatch_penalized():
    """Bowler->wicketkeeper link should be disfavored vs bowler->bowler link."""
    segs = [
        _seg(1, 0, 10, [0.0, 0.0], [0.0, 0.0], role="bowler"),
        _seg(2, 12, 20, [0.2, 0.0], [1.0, 0.0], role="wicketkeeper"),
        _seg(3, 12, 20, [0.3, 0.0], [1.0, 0.0], role="bowler"),
    ]
    G = build_flow_graph(segs, w_t=0.1, w_s=1.0, w_r=100, v_max_mps=9.0, frame_rate=50.0)
    links = solve_flow(G)
    assert links.get(1) == 3  # bowler prefers bowler over wicketkeeper

def test_single_solver_handles_large_graph():
    """Fix 3.2: one solver for all graph sizes. Verify >50 segments handled."""
    segs = [_seg(i, i*20, i*20+15, [float(i)*0.1, 0.0], [float(i)*0.1, 0.0])
            for i in range(60)]
    G = build_flow_graph(segs, w_t=0.1, w_s=1.0, w_r=100, v_max_mps=9.0, frame_rate=50.0)
    links = solve_flow(G)  # must not raise; uses same solver regardless of size
    assert isinstance(links, dict)
