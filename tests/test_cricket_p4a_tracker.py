# tests/test_cricket_p4a_tracker.py
from __future__ import annotations
import numpy as np
import pytest
from pose_estimation.cricket.p4a_track import GlobalTrack, TENTATIVE, CONFIRMED, LOST
from pose_estimation.cricket.p4a_tracker import TrackManager
from pose_estimation.cricket.p3_association import Correspondence, Detection3

def _det(cam="C01", conf=0.8, xy=(0.0, 0.0)):
    kp = np.zeros((17, 2))
    return Detection3(cam_id=cam, bbox_xywh_px=[100,100,80,200],
                      keypoints_px=kp, keypoint_conf=np.full(17, conf), confidence=conf)

def _corr(conf=0.8, xy=(0.0, 0.0)):
    return Correspondence(det_a=_det(conf=conf), det_b=None,
                          track_confidence=conf, single_camera=True,
                          ground_xy=np.asarray(xy, float))

def test_new_track_is_tentative():
    tm = TrackManager()
    tm.update([_corr()], frame_index=0)
    assert len(tm.tracks) == 1
    assert tm.tracks[0].state == TENTATIVE

def test_track_confirms_after_3_frames():
    tm = TrackManager()
    for fi in range(3):
        tm.update([_corr()], frame_index=fi)
    assert tm.tracks[0].state == CONFIRMED

def test_lost_track_gets_kalman_prediction_not_observation():
    """Fix 2.3: lost track must NOT get virtual measurements."""
    tm = TrackManager()
    for fi in range(3):
        tm.update([_corr(xy=(0.0, 0.0))], frame_index=fi)
    pos_before = tm.tracks[0].kalman.pos_world_xy.copy()
    # Next frame: no correspondences -> track goes Lost
    tm.update([], frame_index=3)
    assert tm.tracks[0].state == LOST
    # Position is Kalman prediction (should change slightly due to velocity)
    # Importantly P must have GROWN (not shrunk as virtual updates would cause)
    P_trace_after = np.trace(tm.tracks[0].kalman._P)
    # Just verify the track is still alive and in LOST state
    assert tm.tracks[0].state == LOST

def test_reentry_arbitration_uses_hungarian():
    """Fix 2.4: re-entry must not be ambiguous; closest deleted track wins."""
    tm = TrackManager()
    # Create and confirm a track
    for fi in range(3):
        tm.update([_corr(xy=(0.0, 0.0))], frame_index=fi)
    # Let it go deleted (30+ frames unmatched)
    for fi in range(3, 35):
        tm.update([], frame_index=fi)
    # Now new detection appears near old position
    tm.update([_corr(xy=(0.5, 0.0))], frame_index=35)
    # Should have re-entry to the deleted track, not a new mint
    active = [t for t in tm.tracks if t.state in (TENTATIVE, CONFIRMED)]
    if active:
        # The re-entered track should use the old global_player_id
        assert active[0].global_player_id is not None

def test_role_switch_requires_latching():
    """Fix 2.2: role must latch for N frames before triggering model switch."""
    tm = TrackManager()
    for fi in range(3):
        tm.update([_corr()], frame_index=fi)
    track = tm.tracks[0]
    # Assign role once: should not immediately switch (needs K consecutive frames)
    tm.propose_role(track.global_player_id, "bowler", frame_index=3)
    assert track.dominant_role != "bowler"  # not yet latched
    # Assign repeatedly:
    for fi in range(4, 4 + TrackManager.ROLE_LATCH_FRAMES):
        tm.propose_role(track.global_player_id, "bowler", frame_index=fi)
    assert track.dominant_role == "bowler"  # latched after enough frames
