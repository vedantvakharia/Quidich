# tests/test_cricket_p2_track.py
import numpy as np

from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_pose_vector import build_pose_vector
from pose_estimation.cricket.p2_track import (
    CONFIRMED, DELETED, DORMANT, TENTATIVE, Track,
)

CFG = P2Config()


def _pose(shift=0.0):
    kp = [[10.0 + i + shift, 12.0 + 2 * i] for i in range(17)]
    return build_pose_vector(kp, [0.9] * 17, [0, 0, 40, 80], CFG)


def _new_track(is_lowconf=False, num=1):
    return Track(num, "cam_01", [100, 200, 40, 80], _pose(), is_lowconf, CFG, frame_index=0)


def test_local_track_id_is_camera_prefixed():
    t = _new_track(num=7)
    assert t.local_track_id == "cam_01_trk_0007"


def test_highconf_track_confirms_after_three_hits_in_window():
    t = _new_track()
    assert t.state == TENTATIVE
    for f in range(1, 3):
        t.register_hit([100, 200, 40, 80], _pose(), 0.8, frame_index=f)
        t.maybe_confirm()
    assert t.state == CONFIRMED  # 3 hits (spawn + 2) within window


def test_lowconf_track_needs_a_highconf_hit():
    t = _new_track(is_lowconf=True)
    for f in range(1, 3):
        t.register_hit([100, 200, 40, 80], _pose(), 0.2, frame_index=f)  # all low-conf
        t.maybe_confirm()
    assert t.state == TENTATIVE  # never got a >0.5 hit
    t.register_hit([100, 200, 40, 80], _pose(), 0.8, frame_index=3)
    t.maybe_confirm()
    assert t.state == CONFIRMED


def test_confirmed_goes_dormant_then_deletes_after_max_frames():
    t = _new_track()
    for f in range(1, 3):
        t.register_hit([100, 200, 40, 80], _pose(), 0.8, frame_index=f)
        t.maybe_confirm()
    assert t.state == CONFIRMED
    t.mark_missed(frame_index=3)
    assert t.state == DORMANT
    for f in range(4, 4 + CFG.dormant_max_frames + 1):
        t.mark_missed(frame_index=f)
    assert t.should_delete() is True


def test_reachability_radius_grows_with_dormancy():
    t = _new_track()
    for f in range(1, 3):
        t.register_hit([100, 200, 40, 80], _pose(), 0.8, frame_index=f)
        t.maybe_confirm()
    t.mark_missed(frame_index=3)
    r1 = t.reachability_radius()
    t.mark_missed(frame_index=4)
    assert t.reachability_radius() > r1


def test_gallery_medoid_is_a_stored_vector():
    t = _new_track()
    for f in range(1, 5):
        t.register_hit([100, 200, 40, 80], _pose(shift=f), 0.8, frame_index=f)
    repr_vec = t.gallery_repr()
    assert repr_vec is not None and repr_vec.defined


def test_lowconf_tentative_expires_without_highconf_hit():
    t = _new_track(is_lowconf=True)
    # keeps matching low-conf detections every frame but never lands a >0.5 hit
    for f in range(1, CFG.tentative_confirm_window + 1):
        t.register_hit([100, 200, 40, 80], _pose(), 0.2, frame_index=f)
        t.maybe_confirm()
    assert t.state == TENTATIVE
    assert t.tentative_expired(CFG.tentative_confirm_window) is True


def test_flush_id_backfills_tentative_frames_on_confirmation():
    t = _new_track()
    spawn_player = {"local_track_id": None}
    t.record_player(spawn_player)              # frame 0, still TENTATIVE
    t.flush_id()
    assert spawn_player["local_track_id"] is None  # not stamped while tentative

    next_player = {"local_track_id": None}
    for f in (1, 2):
        t.register_hit([100, 200, 40, 80], _pose(), 0.8, frame_index=f)
    t.record_player(next_player)
    t.maybe_confirm()                          # now CONFIRMED
    t.flush_id()                               # back-fills BOTH the buffered spawn frame and now
    assert spawn_player["local_track_id"] == "cam_01_trk_0001"
    assert next_player["local_track_id"] == "cam_01_trk_0001"
