# tests/test_cricket_p2_tracker.py
import numpy as np

from pose_estimation.cricket.p2_config import P2Config
from pose_estimation.cricket.p2_pose_vector import build_pose_vector
from pose_estimation.cricket.p2_tracker import CameraTracker, Detection, iou_xywh

CFG = P2Config()


def _pose(shift=0.0):
    kp = [[10.0 + i + shift, 12.0 + 2 * i] for i in range(17)]
    return build_pose_vector(kp, [0.9] * 17, [0, 0, 40, 80], CFG)


def _det(x, y, conf=0.8, player=None):
    bbox = [float(x), float(y), 40.0, 80.0]
    return Detection(bbox_xywh=bbox, pose=_pose(), confidence=conf, player=player if player is not None else {})


def test_iou_identical_boxes_is_one():
    assert abs(iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9


def test_iou_disjoint_boxes_is_zero():
    assert iou_xywh([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0


def test_single_object_gets_one_stable_id():
    tracker = CameraTracker("cam_01", CFG)
    players = []
    for f in range(6):
        p = {"local_track_id": None, "track_confidence": 0.8}
        players.append(p)
        tracker.update([_det(100 + 5 * f, 200, player=p)], frame_index=f)
    tracker.finalize()
    # every frame back-filled (incl. the pre-confirmation tentative frames) with one stable id
    assert all(p["local_track_id"] == "cam_01_trk_0001" for p in players)


def test_two_separated_objects_get_distinct_ids():
    tracker = CameraTracker("cam_01", CFG)
    left, right = [], []
    for f in range(6):
        pl = {"local_track_id": None, "track_confidence": 0.8}
        pr = {"local_track_id": None, "track_confidence": 0.8}
        left.append(pl)
        right.append(pr)
        tracker.update([_det(100, 200, player=pl), _det(900, 200, player=pr)], frame_index=f)
    lid = {p["local_track_id"] for p in left if p["local_track_id"]}
    rid = {p["local_track_id"] for p in right if p["local_track_id"]}
    assert len(lid) == 1 and len(rid) == 1
    assert lid != rid


def test_unmatched_low_conf_does_not_crash_and_counts():
    tracker = CameraTracker("cam_01", CFG)
    p = {"local_track_id": None, "track_confidence": 0.2}
    tracker.update([_det(100, 200, conf=0.2, player=p)], frame_index=0)
    tracker.finalize()
    assert tracker.diagnostics["total_tracks_spawned"] >= 1


def test_dormant_reid_recovers_same_id_after_short_gap():
    tracker = CameraTracker("cam_01", CFG)
    players = []
    for f in range(5):
        p = {"local_track_id": None, "track_confidence": 0.8}
        players.append(p)
        tracker.update([_det(100, 200, player=p)], frame_index=f)
    first_id = players[-1]["local_track_id"]
    for f in range(5, 8):  # 3-frame gap, no detections
        tracker.update([], frame_index=f)
    p = {"local_track_id": None, "track_confidence": 0.8}
    tracker.update([_det(110, 205, player=p)], frame_index=8)
    assert p["local_track_id"] == first_id
