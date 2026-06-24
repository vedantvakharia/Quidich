# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is **Group 1** of a cricket broadcast analytics system built on Quidich's DRS (Decision Review System) camera network. The mandate: build a robust multi-camera association and tracking layer that assigns stable anonymous IDs and cricket role labels across calibrated, synchronised DRS camera views.

## Dataset

- **Match:** CCPL080626, 8 deliveries ("balls")
- **Cameras:** 7 (camera01–07), grouped: `bt_01` (cams 1,4), `bt_02` (cams 2,5,7), `bt_03` (cams 3,6)
- **Frames:** 600 JPGs per camera per delivery @ 2560×1440
- **Frame filename format:** `frame_camera01_000212334.jpg` — the number is the absolute frame index used for sync across all cameras
- **Calibration files** (in `dataset/calibration-data/CCPL080626/calibration_data/`):
  - `Bundle_Adjusted_intrinsics.json` — per-camera `camera_matrix` (distortion ≈ 0)
  - `Bundle_Adjusted_extrinsics.json` — per-camera `camera_locations` + 3×4 `projection_matrices`
  - `pitch_calibration_config.json` — pitch/crease world coordinates
  - `CPL08626_coord_aligned.csv` — surveyed reference points

## Coordinate System

- **Origin:** geometric centre of the pitch
- **Y-axis:** along pitch length, positive toward Far End (FE)
- **X-axis:** perpendicular to pitch, positive toward right side of field
- **Z-axis:** upward (height)
- **Camera positions:** C1 behind Near End, C4 behind Far End, C2/C3 on left, C5/C6 on right, C7 bottom-right diagonal

## Existing Pipeline (ball-only — the template to extend)

The existing pipeline processes the ball only. Every detection in `*_2D.json` is `class_name:"ball"`. There is no person detection anywhere in the dataset yet. The geometry stages are reusable:

```
*_2D.json → *_2D_cleaned.json → *_3D.json → *_3D_cleaned.json → *_3D_trimmed.json → *_3D_unreal.json
```
Plus: `*_reprojection.json` (per-camera pixel error), `*_predicted_3D.json`, `*_speed.json`, `*_EVENTS.json`.

**Key data shapes** (in `dataset/events-data/CCPL080626M1_1_14_1_V0/`):

```jsonc
// *_2D.json — one detection per frame per camera
{ "ball_id": "...",
  "frames": [ { "frame_id": 212334,
                "cameras": [ { "camera_id": 3,
                               "detections": [ { "coords": [x,y,w,h],   // normalised
                                                 "confidence_score": 0.76,
                                                 "class_id": 0, "class_name": "ball",
                                                 "track_id": 1354 } ],
                               "frame_name": "frame_camera03_000212334.jpg",
                               "speed_kmh": null } ] } ],
  "selected_track_ids": { "3": 1354, "5": 1360, ... } }  // camera_id → chosen track

// *_3D.json — one world point per frame
{ "ball_id": "...", "frames": { "212385": [x,y,z], ... } }
```

## Target Output Contract (for Groups 2 & 3)

```jsonc
{ "camera_id": "cam_01", "frame_index": 12518,
  "players": [ { "global_player_id": "P001",    // stable anonymous ID
                 "role": "bowler",
                 "bbox": [x,y,w,h],
                 "track_confidence": 0.94,
                 "pose_2d": { "keypoints": [[x,y], ...],  // 17 keypoints
                              "confidence": [c, ...] },
                 "pose_3d": { "keypoints_world": [[x,y,z], ...] } } ] }
```

**Role enum:** `bowler · striker · non_striker · wicketkeeper · umpire · fielder · unknown`

## Implementation Phases

| Phase | Goal | Key output |
|---|---|---|
| P0 | Lock data contracts, verify calibration, map reuse | Frozen G2/G3 JSON schema |
| P1 | Per-camera person detection + 2D pose (17 keypoints) | bbox + keypoints per frame |
| P2 | Per-camera tracking (intra-camera stable `track_id`) | Tracklets per camera |
| P3 | Cross-camera association — geometry-first (triangulation, epipolar, ground-plane), appearance/pose as tie-breakers | Cross-camera correspondences |
| P4 | Global ID assignment + tracklet stitching across occlusion/exit-entry | `global_player_id` per player |
| P5 | Role classification via geometry rules + priors | `role` per player |
| P6 | Per-joint 3D reconstruction (confidence-weighted triangulation + noise chain) + Unreal export | `pose_3d`, FBX/USD |
| P7 | Validation on blind set, reports, handover | Metrics 1–6, handover docs |

## Cross-Camera Association Approach (P3)

Geometry takes priority over appearance because kits are identical. Cost matrix from:
1. **Triangulation/reprojection consistency** — triangulate via 3×4 projection matrices, reproject, measure pixel error (reuse `*_reprojection.json` check)
2. **Epipolar consistency** — keypoint in camera A must lie near its epipolar line in B
3. **Ground-plane test** — project foot/ankle to pitch plane; same world (x,y) → same person

Assignment: Hungarian algorithm. Tie-breakers when geometry is ambiguous: appearance/ReID embedding, pose-configuration similarity, temporal continuity, role priors.

## 3D Reconstruction Noise Chain (P6)

| Stage | Method | Removes |
|---|---|---|
| A | Confidence gating + RANSAC ray rejection | 2D jitter, outliers |
| B | Confidence-weighted triangulation | missing-view bias |
| C | Reprojection-error rejection (reuse existing) | bad associations |
| D | Temporal filter (One-Euro / Savitzky-Golay / Kalman+RTS) | residual jitter |
| E | Skeleton constraints (bone length, joint limits) | bone-length variation |
| F | IK retarget + foot-lock | foot-skate, occlusion gaps |
| G | Quaternion SLERP smoothing | rotation jitter |

## Open Blockers

- DS-001 access / DS-002 blind-subset readiness (needed before P1 / P7)
- Ground-truth owner + annotation tooling (metrics 1–3 unmeasurable without labels)
- Management validation targets for metrics 1–3 (no numeric threshold set yet)
- Frozen G2/G3 JSON contract (P6 export and downstream integration blocked)

## Key Reference Files

- `CORE_TASKS.md` — authoritative phase-by-phase task plan with citation legend
- `coordinate_conventions.md` — world coordinate and camera layout reference
- `dataset/calibration-data/` — all calibration JSONs and CSVs
- `dataset/events-data/` — existing ball pipeline artifacts (shapes to mirror)
- `00_Shared/Role_Event_Label_Schema.xlsx` — role enum and output field definitions
- `00_Shared/Validation_Results.xlsx` — metric definitions and targets
- `00_Shared/Open_Questions_and_TODOs.xlsm` — live blockers list
