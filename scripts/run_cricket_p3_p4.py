"""CLI entry point for Phase 3 and Phase 4 cross-camera association and global tracking."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_estimation.cricket.calibration import (  # noqa: E402
    load_projection_matrices,
    load_survey_points,
    get_calibration_dir,
)
from pose_estimation.cricket.p3_precompute import build_precomputed_geometry  # noqa: E402
from pose_estimation.cricket.p3_association import (  # noqa: E402
    Detection3,
    AnchorState,
    select_anchor,
    associate_frame,
)
from pose_estimation.cricket.p4a_tracker import TrackManager  # noqa: E402
from pose_estimation.cricket.p4b_flow import (  # noqa: E402
    extract_segments,
    build_flow_graph,
    solve_flow,
    remap_ids,
)
from pose_estimation.cricket.contract import validate_group1_frame  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 & 4 association and global tracking")
    parser.add_argument("--p2-dir", required=True, help="dir of P2 cam_XX.jsonl tracklets")
    parser.add_argument("--output-dir", required=True, help="dir for final P4 cam_XX.jsonl outputs")
    parser.add_argument("--drive-root", default=str(ROOT), help="root of Quidich repository/drive")
    parser.add_argument("--delivery-id", required=True)
    parser.add_argument("--image-w", type=int, default=2560)
    parser.add_argument("--image-h", type=int, default=1440)
    return parser


def _camera_center_from_P(P: np.ndarray) -> np.ndarray:
    _, _, Vt = np.linalg.svd(P)
    C_h = Vt[-1]
    return C_h[:3] / (C_h[3] + 1e-12)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    p2_dir = Path(args.p2_dir)
    output_dir = Path(args.output_dir)
    drive_root = Path(args.drive_root)
    calibration_dir = get_calibration_dir(drive_root)

    print(f"Loading calibration from {calibration_dir}...", flush=True)
    projection_matrices_raw = load_projection_matrices(calibration_dir)
    # Map C01 -> cam_01, C02 -> cam_02, etc.
    projection_matrices = {}
    for k, matrix in projection_matrices_raw.items():
        try:
            cam_idx = int(k[1:])
            projection_matrices[f"cam_{cam_idx:02d}"] = matrix
        except ValueError:
            continue

    camera_centers = {cid: _camera_center_from_P(P) for cid, P in projection_matrices.items()}
    survey_points = load_survey_points(calibration_dir / "CPL08626_coord_aligned.csv")

    print("Building precomputed geometry primitives...", flush=True)
    geo = build_precomputed_geometry(
        projection_matrices, camera_centers, survey_points, image_wh=(args.image_w, args.image_h)
    )

    # Discover and load P2 files
    p2_files = sorted(p2_dir.glob("cam_*.jsonl"))
    if not p2_files:
        print(f"Error: no cam_*.jsonl files found in {p2_dir}", file=sys.stderr)
        return 1

    # Load all records and group by frame_index
    print("Reading Phase 2 tracking files...", flush=True)
    records_by_frame: dict[int, dict[str, dict]] = {}
    for p2_file in p2_files:
        cam_id = p2_file.stem
        with p2_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("delivery_id") != args.delivery_id:
                    continue
                frame_idx = record["frame_index"]
                if frame_idx not in records_by_frame:
                    records_by_frame[frame_idx] = {}
                records_by_frame[frame_idx][cam_id] = record

    sorted_frames = sorted(records_by_frame.keys())
    if not sorted_frames:
        print(f"No frames found for delivery {args.delivery_id}", file=sys.stderr)
        return 1

    print(f"Processing {len(sorted_frames)} frames...", flush=True)

    # Initialize tracking state
    anchor_state = AnchorState(anchor_id="cam_01", frames_since_switch=0)
    track_manager = TrackManager()

    # We will also keep a list of all records for post-processing flow remapping
    all_records = []

    for frame_idx in sorted_frames:
        cams_in_frame = records_by_frame[frame_idx]
        dets_per_cam: dict[str, list[Detection3]] = {}

        # 1. Gather detections and attach player dict references
        for cam_id, record in cams_in_frame.items():
            dets_per_cam[cam_id] = []
            all_records.append(record)
            for player in record.get("players", []):
                # Discard lost tracklets if they don't have bbox or keypoints
                if player.get("track_state") == "lost" or player.get("bbox_xywh_px") is None:
                    continue
                kp_px = np.array(player["pose_2d"]["keypoints_px"])
                kp_conf = np.array(player["pose_2d"]["confidence"])
                det = Detection3(
                    cam_id=cam_id,
                    bbox_xywh_px=player["bbox_xywh_px"],
                    keypoints_px=kp_px,
                    keypoint_conf=kp_conf,
                    confidence=player.get("track_confidence", 1.0),
                )
                # Store dynamic references
                det.player_ref = player
                dets_per_cam[cam_id].append(det)

        # 2. Select anchor and run association
        anchor_state = select_anchor(dets_per_cam, anchor_state)
        correspondences = associate_frame(dets_per_cam, projection_matrices, geo, anchor_state)

        # 3. Update global track manager
        track_manager.update(correspondences, frame_idx)

        # 4. Map global IDs and properties back to the player records in the frame
        for c in correspondences:
            gt = getattr(c, "global_track", None)
            # Assign global ID and confidence to all camera detections associated with this correspondence
            for det in getattr(c, "all_detections", []):
                player = getattr(det, "player_ref", None)
                if player is not None:
                    if gt is not None:
                        player["_gt_ref"] = gt  # Save reference for backfilling
                        player["global_player_id"] = gt.global_player_id
                        player["track_state"] = gt.state
                        player["track_confidence"] = c.track_confidence
                        player["single_camera"] = c.single_camera
                        # Save the ground position for P4b stitching
                        if np.isfinite(c.ground_xy).all():
                            player["_ground_xy"] = c.ground_xy.tolist()
                        else:
                            player["_ground_xy"] = None
                    else:
                        player["global_player_id"] = None
                        player["track_confidence"] = 0.3
                        player["single_camera"] = True
                        player["_ground_xy"] = None

    print("Finalizing global tracking...", flush=True)
    track_manager.finalize()

    # 5. Phase 4b: Min-cost-flow post-delivery correction
    print("Running Phase 4b min-cost flow stitching...", flush=True)
    segments = extract_segments(all_records)
    G = build_flow_graph(segments)
    links = solve_flow(G)
    switch_report = remap_ids(all_records, segments, links)
    print(f"Stitched {len(switch_report)} ID switches.", flush=True)

    # 6. Contract validation and write outputs
    print(f"Saving final P4 tracking outputs to {output_dir}...", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a copy of the switch report
    with (output_dir / "id_switch_report.json").open("w", encoding="utf-8") as f:
        json.dump(switch_report, f, indent=2)

    # Backfill global IDs for tentative frames of confirmed tracks
    print("Backfilling global IDs for tentative frames...", flush=True)
    for record in all_records:
        for player in record.get("players", []):
            gt = player.pop("_gt_ref", None)
            if gt is not None and gt.global_player_id is not None:
                player["global_player_id"] = gt.global_player_id

    # Group records back by camera
    records_by_cam: dict[str, list[dict]] = {}
    for record in all_records:
        cam_id = record["camera_id"]
        if cam_id not in records_by_cam:
            records_by_cam[cam_id] = []
        # Clean up internal helper fields before writing out
        for player in record.get("players", []):
            if "_ground_xy" in player:
                del player["_ground_xy"]
        
        # Validate schema contract
        validate_group1_frame(record, final_handoff=False)
        records_by_cam[cam_id].append(record)

    # Write each camera file
    for cam_id, records in records_by_cam.items():
        out_file = output_dir / f"{cam_id}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for record in sorted(records, key=lambda r: r["frame_index"]):
                f.write(json.dumps(record) + "\n")

    print("Phase 3 & 4 execution completed successfully!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
