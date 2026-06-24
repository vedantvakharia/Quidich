# Group 1 — CORE_TASKS

Technical task plan for **ReID, role classification, and stable tracks** (with triangulated 3D
pose and Unreal export). This is the deep-dive companion to
[TASKS_EXPLAINED.md](TASKS_EXPLAINED.md), which explains every term in plain language. Read that
first if a term here is unfamiliar.

Phases are ordered by **dependency**, not calendar. A rough map to the week plan
`[DOCS: docs/06_Group1_Week_By_Week_Plan.md]` is given per phase, but the week plan is not
binding here.

### Citation legend
- `[SRC: <file>]` — authoritative repo file (the problem/validation/handover sheets, the shared
  schema, or the dataset itself).
- `[DOCS: <file>]` — internal write-ups under `docs/`, `grp_1/`, or `Archived_Documentation/`.
  **Inferred, not official.**
- `[INFERRED]` — our reasoning; not stated anywhere in the repo.
- `[WEB: <ref>]` — external source.

### Conventions
- Models and methods are listed as a **candidate pool to benchmark on our own footage**, never
  as a fixed choice. We may add candidates not listed here.
- "Delivery" = one ball (a clip of 600 frames per camera) `[SRC: dataset/bt_*/]`.

---

## P0 — Understand the problem, fix the contract, plumb the data

**Goal.** Pin down exactly what Group 1 must output, lock the data shapes it consumes and
produces, and confirm what already exists vs. what must be built. Nothing downstream is
benchmarkable until the output contract and ground truth are settled.

**Mandate** `[SRC: Problem_Statement.xlsm, Objective]`
> Build a robust multi-camera association and tracking layer that assigns stable anonymous IDs
> and cricket role labels across calibrated, synchronised DRS camera views.

Identity is **anonymous** — `P001`, `P002`… are sufficient; real names are linked later.
`[SRC: 00_Shared/Programme_Brief.xlsm, ID requirement]` `[SRC: 00_Shared/Decision_Log.xlsm,
"Use stable anonymous IDs", 2026-06-08]`

**Required outputs (6)** `[SRC: Problem_Statement.xlsm, Outputs]`
1. ReID baseline (cross-camera association)
2. Role classifier
3. Stable tracklet JSON
4. ID-switch report
5. Role-accuracy report
6. Failure-case library

**Inputs available** `[SRC: dataset/]`
- 7 cameras `camera01–07`, one match `CCPL080626`, capture groups `bt_01`(1,4), `bt_02`(2,5,7),
  `bt_03`(3,6). 8 deliveries; 600 JPGs/camera @ 2560×1440; absolute frame number in the filename
  (`frame_camera01_000212334.jpg`) provides sync. `[SRC: dataset/bt_*/]`
- Calibration: `Bundle_Adjusted_intrinsics.json` (per-camera `camera_matrix`, distortion≈0),
  `Bundle_Adjusted_extrinsics.json` (per-camera `camera_locations` + 3×4 `projection_matrices`),
  `pitch_calibration_config.json` (pitch/crease world coords), `CPL08626_coord_aligned.csv`
  (surveyed reference points), `crop_mech.json`, `reference_frames/Camera0*.jpg`.
  `[SRC: dataset/calibration-data/CCPL080626/calibration_data/]`
- Existing **ball** pipeline artifacts per delivery (the template to extend), see below.
  `[SRC: dataset/events-data/]`

**What exists vs. what we build** `[SRC: dataset/events-data/]`
- The current detector outputs the **ball only** — every detection in `*_2D.json` is
  `class_name:"ball"` (no other class present). **There is no player detection, no player pose,
  no player tracker anywhere in the dataset.** Building per-camera player detection + pose +
  tracking is Group 1's job. `[SRC: Programme_Brief.xlsm, Player tracker availability]`
- The geometry chain already works for a single point and is reused unchanged:
  `*_2D.json → *_2D_cleaned.json → *_3D.json → *_3D_cleaned.json → *_3D_trimmed.json →
  *_3D_unreal.json`, plus `*_reprojection.json` (per-camera pixel error), `*_predicted_3D.json`,
  `*_speed.json`, `*_EVENTS.json`.

**Existing data shapes (to mirror)** `[SRC: dataset/events-data/CCPL080626M1_1_14_1_V0/]`
```jsonc
// *_2D.json
{ "ball_id": "...",
  "frames": [ { "frame_id": 212334,
                "cameras": [ { "camera_id": 3,
                               "detections": [ { "coords": [x,y,w,h],      // normalised
                                                 "confidence_score": 0.76,
                                                 "class_id": 0, "class_name": "ball",
                                                 "track_id": 1354 } ],
                               "frame_name": "frame_camera03_000212334.jpg",
                               "speed_kmh": null } ] } ],   // 600 frames
  "selected_track_ids": { "3": 1354, "5": 1360, ... } }     // camera_id -> chosen track
// *_3D.json
{ "ball_id": "...", "frames": { "212385": [x,y,z], ... } } // one point per frame
```

**Target output contract for Groups 2 & 3** `[SRC: 00_Shared/Role_Event_Label_Schema.xlsx,
role field]` `[DOCS: docs/03_...md §6]` — concrete layout `[INFERRED]`, to be frozen:
```jsonc
{ "camera_id": "cam_01", "frame_index": 12518,
  "players": [ { "global_player_id": "P001",   // Group 1 owns (stable anon)
                 "role": "bowler",              // Group 1 owns
                 "bbox": [x,y,w,h],
                 "track_confidence": 0.94,
                 "pose_2d": { "keypoints": [[x,y] x17], "confidence": [c x17] },
                 "pose_3d": { "keypoints_world": [[x,y,z] x17] } } ] }
```
Role enum: `bowler · striker · non_striker · wicketkeeper · umpire · fielder · unknown`
`[SRC: Role_Event_Label_Schema.xlsx]`.

**Tasks**
- [ ] Confirm DS-001 access and the blind subset DS-002. `[SRC: 00_Shared/Data_Catalogue.xlsx]`
      ⚠ open blocker `[SRC: 00_Shared/Open_Questions_and_TODOs.xlsm, Dataset access]`
- [ ] Confirm ground-truth owner + annotation tooling for IDs/roles/3D. ⚠ open blocker
      `[SRC: Open_Questions_and_TODOs.xlsm, Ground truth availability]`
- [ ] Freeze the exact G2/G3 JSON field set (meeting item). `[DOCS: docs/09_...md §10]`
- [ ] Load and sanity-check calibration: reproject a known surveyed point and confirm pixel
      error is in the same range the ball pipeline reports (~few px). `[SRC: dataset/.../*_reprojection.json]`
- [ ] Document the reuse map: which existing stages (2D-clean, 3D-clean, trim, reprojection,
      unreal export) extend from one point to many joints.

**Exit criteria.** Output contract frozen; calibration verified usable; data access + ground-
truth ownership resolved or escalated; reuse map written.

**Maps to.** Foundation for all deliverables. Week ≈ W1.

---

## P1 — Per-camera perception (player detection + 2D pose)

**Goal.** For every camera and frame, detect each person and estimate their 2D keypoints. This
2D front-end feeds everything downstream. `[INFERRED]`

**Depends on.** P0.

**Inputs.** Frames `[SRC: dataset/bt_*/]`; crop windows `[SRC: .../crop_mech.json]`.
**Output.** Per camera, per frame: `bbox` + 17 keypoints + 17 per-keypoint confidences,
mirroring the detection record in `*_2D.json` but `class_name:"person"` and one entry per
player. `[INFERRED]` extension of `[SRC: events-data *_2D.json]`.

**Candidate pool (variable — benchmark on our footage)** `[DOCS: docs/full.md]`
| Candidate | Type | Why considered |
|---|---|---|
| RTMPose-m / -l | 2D body, top-down | Real-time baseline |
| RTMO-l | 2D body, one-stage | Cost independent of player count; strong under crowding/occlusion |
| DWPose-l, RTMW-l/x | 2D whole-body | Hands/feet detail (wrist at release, foot for no-ball) |
| Sapiens | dense whole-body | Accuracy ceiling, offline |
| *(others)* | — | Pool is open; add as found `[WEB]` |

**Tasks**
- [ ] Stand up a person detector per camera (today only ball is detected).
- [ ] Run 2D pose on detections; record per-keypoint confidence.
- [ ] Benchmark candidates on our footage for accuracy vs. latency (define the harness; metrics
      AP, MPJPE-proxy, latency per `[DOCS: docs/full.md]`).
- [ ] Handle the tight DRS crops and 2560×1440 resolution. `[SRC: Problem_Statement.xlsm, Known risks]`

**Exit criteria.** Per-camera per-frame person bbox + 17 keypoints + confidences produced for at
least one full delivery across all 7 cameras; a benchmarked model shortlist with numbers.

**Maps to.** Enables all later phases. Week ≈ W1. `[DOCS: docs/06_...md W1]`

---

## P2 — Per-camera tracking (ReID across frames → tracklets)

**Goal.** Within a single camera, link a player's detections over time so the same person keeps
one local `track_id` through motion and brief occlusion. `[DOCS: reid_plan.md Step 2]`

**Depends on.** P1.

**Inputs.** P1 detections + keypoints. **Output.** Per-camera tracklets `trk_XXXX`, the
per-person analogue of the ball's `selected_track_ids`. `[DOCS: reid_plan.md Step 2]`

**Candidate pool (variable)**
| Candidate | Mechanism |
|---|---|
| DeepSORT | Appearance embedding + Kalman motion association `[DOCS: reid_plan.md]` |
| PipeTrack | Pipeline tracker on these DRS clips `[DOCS: grp_1/plan.md §4]` |
| Projection-matrix motion prior | Geometry-assisted motion gating `[DOCS: grp_1/plan.md §4]` |
| *(others, incl. ByteTrack/BoT-SORT class)* | `[WEB]` add as benchmarked |

**Tasks**
- [ ] Build per-camera tracking from P1 detections.
- [ ] Maintain ID through brief occlusion; mark gaps for later stitching (P4).
- [ ] Benchmark candidates; report **ID switches per delivery within a camera** and **track
      completeness**. `[SRC: Validation_Results.xlsx, metrics 2 & 4]`

**Exit criteria.** Stable per-camera tracklets for a full delivery; intra-camera ID-switch and
completeness numbers recorded.

**Maps to.** ID-switch report (partial), failure-case library (seed). Week ≈ W2.

---

## P3 — Cross-camera association (core, geometry-first)

**Goal.** Match the same physical person across the 7 calibrated views. **Geometry first, not
appearance**, because kits are identical. `[SRC: Problem_Statement.xlsm, Approach + Known risks]`
`[DOCS: docs/03_...md §3]`

**Depends on.** P2 (per-camera tracklets to associate).

**Inputs.** P2 tracklets + P1 keypoints; projection matrices + pitch plane
`[SRC: dataset/calibration-data/...]`. **Output.** Cross-camera correspondences between local
tracklets, with a geometric confidence per match. `[DOCS: reid_plan.md Step 3]`

**Method (candidate residuals, combined into a cost matrix)** `[DOCS: reid_plan.md Step 3]`
`[DOCS: docs/03_...md §4]`
- **Triangulation/reprojection consistency** — triangulate a candidate person across a camera
  pair via the 3×4 projection matrices, reproject, measure per-camera pixel error; low error ⇒
  same person. Reuses the existing reprojection check. `[SRC: events-data *_reprojection.json]`
- **Epipolar consistency** — a keypoint in camera A must lie near its epipolar line in B.
  `[DOCS: docs/03_...md §4a]`
- **Ground-plane test** — project each player's foot/ankle to the surveyed pitch plane; same
  world (x,y) ⇒ same person. `[SRC: pitch_calibration_config.json]` `[DOCS: reid_plan.md]`
- Solve camera-to-camera assignment from the combined cost (Hungarian). `[DOCS: reid_plan.md]`
- **Tie-breakers when geometry is ambiguous** (side-on overlap): appearance/ReID embedding,
  pose-configuration similarity, temporal continuity, role priors.
  `[SRC: Problem_Statement.xlsm, Approach]` `[DOCS: docs/03_...md §4c]`

**Tasks**
- [ ] Implement the projection/triangulation + epipolar + ground-plane residuals.
- [ ] Build the per-frame cross-camera cost matrix and assignment.
- [ ] Add tie-breakers; quantify how often each is needed.
- [ ] Report **cross-camera association accuracy** vs. labels and the **reprojection effect**.
      `[SRC: Validation_Results.xlsx, metrics 1 & 5]`

**Exit criteria.** Same player matched across all 7 views for a full delivery; association
accuracy measured against ground truth (once available).

**Maps to.** ReID baseline (deliverable 1). Week ≈ W3. ⚠ documented risks: similar kits,
occlusion, tight views, side-on overlap, late entry/exit, no full-field context.
`[SRC: Problem_Statement.xlsm, Known risks]`

---

## P4 — Global ID + tracklet stitching + temporal smoothing

**Goal.** Collapse per-camera tracklets and cross-camera matches into **one
`global_player_id` per physical person**, surviving the whole delivery and all views; repair ID
switches and bridge occlusion / exit-entry. `[DOCS: reid_plan.md Steps 4–5]` `[DOCS: docs/03_...md §3]`

**Depends on.** P3.

**Inputs.** P3 correspondences. **Output.** `global_player_id` assigned in every camera/frame;
stitched, gap-filled tracks. Feeds the contract JSON's `global_player_id`. `[SRC: Role_Event_Label_Schema.xlsx]`

**Tasks**
- [ ] Cluster agreeing cross-camera matches into one global ID. `[DOCS: reid_plan.md Step 4]`
- [ ] Stitch tracklets across occlusion and exit/entry; handle re-entry. `[DOCS: docs/03_...md §3]`
- [ ] Temporal smoothing to repair ID switches. `[DOCS: reid_plan.md Step 5]`
- [ ] Hand-assign `global_player_id` seeds on DS-001 as the **manual-ID bridge** so Groups 2/3
      can start before automation is ready; then switch them to automated IDs.
      `[DOCS: docs/09_...md §5]` `[SRC: Experiment_Log.xlsx, W1 "Reproduce/manual-ID baseline"]`
- [ ] Report **ID switches per delivery across cameras**. `[SRC: Validation_Results.xlsx, metric 2]`

**Exit criteria.** One stable global ID per player across the full delivery and all cameras;
cross-camera ID-switch count recorded; manual-ID seeds delivered to G2/G3.

**Maps to.** Stable tracklet JSON (deliverable 3), ID-switch report (deliverable 4). Week ≈ W4.

---

## P5 — Role classification

**Goal.** Label each global player with a cricket role.
`[SRC: Problem_Statement.xlsm, Approach + Success criteria]`

**Depends on.** P4 (needs a stable identity to attach a role to).

**Inputs.** Global tracks + 3D position + pitch/crease landmarks + motion. **Output.** `role` ∈
`{bowler, striker, non_striker, wicketkeeper, umpire, fielder, unknown}`.
`[SRC: Role_Event_Label_Schema.xlsx]`

**Method (rule-based + geometry + role priors)** `[SRC: Problem_Statement.xlsm, Approach]`;
decision tree `[INFERRED]` `[DOCS: docs/03_...md §5]`:
- behind striker's stumps ⇒ wicketkeeper; running in from far end + delivery stride ⇒ bowler;
  at a crease holding position ⇒ striker/non_striker (by facing); stationary in known umpire
  spots ⇒ umpire; else fielder; else unknown.
- Role priors (1 bowler, wk behind stumps, etc.) also disambiguate P3/P4. `[DOCS: docs/03_...md §3]`

**Tasks**
- [ ] Derive crease/pitch geometry features from calibration. `[SRC: pitch_calibration_config.json]`
- [ ] Implement the rule tree + priors; handle `unknown`.
- [ ] Confirm exact rules and `run_up_start`/event definitions with Harsh. ⚠
      `[SRC: 00_Shared/Annotation_Guide.xlsx]` `[INFERRED]`
- [ ] Report **role classification accuracy**. `[SRC: Validation_Results.xlsx, metric 3]`

**Exit criteria.** Roles assigned for a full delivery; role accuracy measured against labels.

**Maps to.** Role classifier (deliverable 2), role-accuracy report (deliverable 5). Week ≈ W4.

---

## P6 — 3D reconstruction, noise removal & export  *(extension beyond the documented contract)*

**Goal.** Turn associated multi-view 2D keypoints into a smooth, plausible 3D skeleton and
export it to both the G2/G3 JSON (`pose_3d.keypoints_world`) and an Unreal animation.

> **Scope note.** `[SRC: Problem_Statement.xlsm]` demands ReID + roles + stable tracks and uses
> triangulation/reprojection only as a **validation signal** `[SRC: Validation_Results.xlsx,
> metric 5]`. The full 3D-cleaning + Unreal pipeline below is an **extension**
> `[DOCS: grp_1/plan.md §8]` `[INFERRED]`, built on the existing single-point chain
> `[SRC: events-data]`. Treat it as such when prioritising against core ReID.

**Depends on.** P3–P4 (clean correspondences + stable IDs for per-joint multi-view triangulation).

**Inputs.** P1 keypoints + confidences, P3/P4 associations, projection matrices.
**Output.** `pose_3d.keypoints_world` (17 joints) per player per frame + an Unreal export.

**Noise sources** `[DOCS: grp_1/plan.md §7]` `[INFERRED]`: 2D keypoint jitter, triangulation
outliers (one bad ray), missing views/occlusion, reprojection error, bone-length variation,
foot-skate, rotation jitter. Each maps to a stage:

| Stage | Method (candidate pool, variable) | Removes |
|---|---|---|
| A | Confidence gating + RANSAC ray rejection | 2D jitter, outliers |
| B | Confidence-weighted triangulation (projection matrices) | missing-view bias |
| C | Reprojection-error rejection (reuse existing check) | bad associations/calib slack |
| D | Temporal filter: One-Euro / Savitzky-Golay / Kalman+RTS | residual jitter |
| E | Skeleton constraints (constant bone length, joint limits) | bone-length variation |
| F | IK retarget + foot-lock: HybrIK / NIKI / KinePose / MANIKIN / PLIKS / OpenSim IK / *others* | occlusion gaps, foot-skate |
| G | Quaternion SLERP rotation smoothing | rotation jitter |

`[DOCS: grp_1/plan.md §8]` `[INFERRED]`; stage C reuses `[SRC: events-data *_reprojection.json]`.

**Tasks**
- [ ] Per-joint confidence-weighted triangulation across the 7 views.
- [ ] Implement stages A–G as a configurable chain; benchmark the variable choices (filter D,
      IK solver F) on our footage. Pool is open.
- [ ] Export `pose_3d` into the frozen G2/G3 JSON (P0).
- [ ] Export an Unreal-ready rig (FBX / USD / Live Link), modelled on `*_3D_unreal.json`.
      `[INFERRED]` `[SRC: events-data *_3D_unreal.json]`
- [ ] **Bounded R&D only:** FMPose3D, SAM 3D, FreeMocap, OpenSim feasibility — time-boxed, not a
      commitment. `[SRC: Problem_Statement.xlsm, Approach]` `[SRC: Experiment_Log.xlsx, W4]`
      `[WEB: xiu-cs.github.io/FMPose3D, ai.meta.com/research/sam3d]`

**Exit criteria.** Smooth `pose_3d` exported for a full delivery; reprojection error at or below
current ball values; a renderable Unreal clip. Smoothness target is proposed, not set (P7).

**Maps to.** `pose_3d` field of the contract; demo material. Week ≈ spans W2–W5.

---

## P7 — Validation, reports & handover

**Goal.** Measure the pipeline on the blind subset and complete the documented deliverables and
handover. `[SRC: Validation_Results.xlsx]` `[SRC: Final_Handover.xlsx]`

**Depends on.** P3–P6 + ground truth.

**Metrics** `[SRC: Validation_Results.xlsx]`
| # | Metric | Target |
|---|---|---|
| 1 | Cross-camera association accuracy | ⚠ MANAGEMENT INPUT REQUIRED |
| 2 | ID switches per delivery | ⚠ MANAGEMENT INPUT REQUIRED |
| 3 | Role classification accuracy | ⚠ MANAGEMENT INPUT REQUIRED |
| 4 | Track completeness | (no target in sheet) |
| 5 | Reprojection effect (does association improve triangulation?) | qualitative |
| 6 | 3D smoothness (per-joint frame-to-frame jerk) | *proposed* `[INFERRED]` `[DOCS: grp_1/plan.md §10]` |

**Tasks**
- [ ] Run the full pipeline on blind subset DS-002. ⚠ needs ready DS-002 + labels.
      `[SRC: Data_Catalogue.xlsx]` `[SRC: Open_Questions_and_TODOs.xlsm]`
- [ ] Compute metrics 1–5 (+6 if adopted); fill `Validation_Results.xlsx`.
- [ ] Produce the **ID-switch report**, **role-accuracy report**, and **failure-case library**
      (before/after on the documented risks). `[SRC: Problem_Statement.xlsm, Outputs]`
- [ ] Complete all 9 sections of `Final_Handover.xlsx`: problem, method, datasets, best demo,
      measured results, failure cases, recommended next step, code handover (GitHub),
      OpenProject links. `[SRC: Final_Handover.xlsx]`
- [ ] One row per cycle in `Weekly_Demo_Log.xlsm`. `[SRC: 00_Shared/Weekly_Demo_Log.xlsm]`

**Exit criteria.** All 6 deliverables produced; validation sheet filled (subject to targets);
handover sheet complete. Week ≈ W5.

---

## Open blockers (cross-cutting ⚠)

| Blocker | Impact | Source |
|---|---|---|
| DS-001 access / DS-002 readiness | Can't start P1 / can't validate P7 | `[SRC: Open_Questions_and_TODOs.xlsm; Data_Catalogue.xlsx]` |
| Ground-truth owner + annotation tooling | No IDs/roles/3D labels ⇒ metrics 1–3 unmeasurable | `[SRC: Open_Questions_and_TODOs.xlsm; Validation_Results.xlsx]` |
| Management validation targets (1–3) | "Accurate" undefined until set | `[SRC: Validation_Results.xlsx]` |
| Frozen G2/G3 JSON contract | P6 export + downstream integration blocked | `[DOCS: docs/09_...md §10]` |
| R&D time-split (FMPose3D/SAM3D vs core ReID) | Unresolved scope risk | `[SRC: Open_Questions_and_TODOs.xlsm, FMPose3D/SAM3D scope]` |
| More data (extra matches for generalization) | One match limits robustness claims | `[INFERRED]` |

## Deliverable → phase map

| Deliverable / output | Produced in | Source |
|---|---|---|
| ReID baseline (cross-camera association) | P3 | `[SRC: Problem_Statement.xlsm, Outputs]` |
| Role classifier | P5 | `[SRC: Problem_Statement.xlsm, Outputs]` |
| Stable tracklet JSON (`global_player_id`) | P4 (+P6 for `pose_3d`) | `[SRC: Outputs; Role_Event_Label_Schema.xlsx]` |
| ID-switch report | P2→P4, final P7 | `[SRC: Problem_Statement.xlsm, Outputs]` |
| Role-accuracy report | P5, final P7 | `[SRC: Problem_Statement.xlsm, Outputs]` |
| Failure-case library | P2→P6, final P7 | `[SRC: Problem_Statement.xlsm, Outputs]` |
| `pose_3d.keypoints_world` (G2/G3 field) | P6 *(extension)* | `[SRC: Role_Event_Label_Schema.xlsx]` `[DOCS: grp_1/plan.md §8]` |
| Unreal animation export | P6 *(extension)* | `[INFERRED]` `[SRC: events-data *_3D_unreal.json]` |
| Validation results | P7 | `[SRC: Validation_Results.xlsx]` |
| Final handover doc + code | P7 | `[SRC: Final_Handover.xlsx]` |

---
*Team: Aksh (lead), Vedant, Anshul. `[SRC: Problem_Statement.xlsm, Management input]`*
