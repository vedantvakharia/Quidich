import sys, json
from pathlib import Path
sys.path.insert(0, str(Path("C:/Users/intel/Downloads/Quidich")))
from pose_estimation.cricket.calibration import load_projection_matrices, load_survey_points, get_calibration_dir
from pose_estimation.cricket.p3_precompute import build_precomputed_geometry, _camera_center_from_P
from pose_estimation.cricket.p3_association import select_anchor, associate_frame, AnchorState, Detection3
import numpy as np

p2_dir = Path("C:/Users/intel/Downloads/Quidich/outputs/p2")
calibration_dir = get_calibration_dir(Path("C:/Users/intel/Downloads/Quidich"))
proj_raw = load_projection_matrices(calibration_dir)
proj = {f"cam_{int(k[1:]):02d}": v for k, v in proj_raw.items()}
centers = {k: _camera_center_from_P(v) for k, v in proj.items()}
sp = load_survey_points(calibration_dir / "CPL08626_coord_aligned.csv")
geo = build_precomputed_geometry(proj, centers, sp)
print("Dummy cost scale factor:", geo.stats.mu_fine_score, geo.stats.sigma_fine_score)

cams_in_frame = {}
for p2_file in sorted(p2_dir.glob("cam_*.jsonl")):
    cam_id = p2_file.stem
    with open(p2_file) as f:
        for line in f:
            rec = json.loads(line)
            if rec["frame_index"] == 212334:
                cams_in_frame[cam_id] = rec
                break

dets_per_cam = {}
for cam_id, rec in cams_in_frame.items():
    dets = []
    for p in rec.get("players", []):
        kp = np.array(p["pose_2d"]["keypoints_px"])
        conf = np.array(p["pose_2d"]["confidence"])
        dets.append(Detection3(cam_id, p["bbox_xywh_px"], kp, conf, p.get("track_confidence", 1.0)))
    dets_per_cam[cam_id] = dets

anchor = AnchorState("cam_01", 0)
print("Number of cameras:", len(dets_per_cam))
corrs = associate_frame(dets_per_cam, proj, geo, anchor)
print("Corrs:")
for c in corrs:
    print(f"  single_camera: {c.single_camera}, track_confidence: {c.track_confidence}")
