# Phase 3 & Phase 4 Design — Cross-Camera Association and Global ID Management

**Project:** Quidich Group 1 — Cricket Multi-Camera Player Tracking
**Team:** Aksh (lead), Vedant, Anshul
**Date:** 2026-06-19
**Status:** Approved for implementation

---

## 1. Architecture Overview

### 1.1 Inputs and Outputs

**Inputs (from P1 + P2):**
- Per-camera, per-frame: bounding boxes, 17-point 2D keypoints with per-keypoint confidence, stable intra-camera `track_id`
- Calibration: `Bundle_Adjusted_intrinsics.json`, `Bundle_Adjusted_extrinsics.json`, `pitch_calibration_config.json`, `CPL08626_coord_aligned.csv`

**Outputs:**
- Per-camera, per-frame: `global_player_id`, `track_confidence`, `track_state` (see Section 4.2 for full contract)
- Post-delivery: ID-switch report (deliverable 4), segment map for debugging

### 1.2 Pipeline Flow

```
[P2 Output]
Per-camera intra-camera tracklets
(bbox, 17 keypoints, track_id per frame per camera)
          │
          ▼
┌─────────────────────────────────────────────┐
│  PHASE 3 — Cross-Camera Association         │
│                                             │
│  Per frame:                                 │
│  1. Select dynamic anchor camera            │
│  2. Project foot keypoints to ground plane  │
│  3. Build cost matrix per anchor↔partner    │
│     pair using hybrid cascade:              │
│     Gate 1 (ground plane) →                 │
│     Gate 2 (epipolar, non-degenerate only)→ │
│     Fine score (epipolar + triangulation)   │
│  4. Hungarian assignment (6 problems)       │
│  5. Fallback pairwise for anchor-occluded   │
│                                             │
│  Output: per-frame cross-camera             │
│  correspondences + geometric confidence     │
└─────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  PHASE 4a — Online Track Management         │
│  (OC-SORT + role-aware Singer Kalman)       │
│                                             │
│  Per frame:                                 │
│  1. Two-stage ByteTrack-style association   │
│  2. Singer Kalman predict + OC-SORT update  │
│  3. Track lifecycle management              │
│  4. Re-entry matching against Deleted pool  │
│                                             │
│  Output: global_player_id per frame,        │
│  smooth ground trajectory, track states     │
└─────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  PHASE 4b — Post-Delivery Correction        │
│  (Min-Cost Flow)                            │
│                                             │
│  Once all 600 frames processed:             │
│  1. Extract tracklet segments from 4a       │
│  2. Build directed segment graph            │
│  3. Min-cost flow → optimal global chains   │
│  4. Remap corrected IDs into per-frame JSON │
│  5. Emit ID-switch report                   │
│                                             │
│  Output: corrected global_player_id,        │
│  ID-switch report, segment map              │
└─────────────────────────────────────────────┘
          │
          ▼
[Output Contract JSON — feeds P5, G2, G3]
```

### 1.3 Processing Mode

**Real-time, causal.** P4a runs frame-by-frame with no look-ahead. P4b is a bounded post-pass over the completed 600-frame delivery — runs in milliseconds, adds ~1 frame of latency after delivery end, and is still compatible with the real-time constraint at the delivery level.

### 1.4 Role Feedback Loop

Role information from P5 feeds back into P4a's Kalman motion models mid-delivery. Until P5 assigns a role, all tracks default to the `fielder` (Brownian) motion model.

---

## 2. Phase 3 — Cross-Camera Association

### 2.1 Offline Precomputation

Before any frame is processed, compute and cache:

- **All 21 fundamental matrices** `F_AB` (Fundamental Matrix) for every camera pair `(A, B)` from the 3×4 projection matrices using the standard relation `F = K_B^{-T} [t]_× R K_A^{-1}`. The fundamental matrix algebraically relates corresponding 2D points between two camera views (satisfying $x_B^T F_{AB} x_A = 0$), allowing keypoint matching validation via epipolar line constraints (Sampson distance).
- **Degenerate pair flags:** A pair is degenerate if cameras share a near-collinear baseline (C1↔C4, C2↔C5, C3↔C6 — directly opposing cameras). Detect offline by checking `rank(F_AB)` (which drops or becomes ill-conditioned in these setups). Because the epipolar lines coincide with the baseline or converge, Sampson distance calculations become numerical noise and lose their discriminating power; hence, these pairs skip epipolar gating.
- **Per-pair ground-plane thresholds `τ_gp`:** Calibrated by backprojecting known surveyed points from `CPL08626_coord_aligned.csv` and measuring projection error per camera. End-on cameras (C1, C4): `τ_gp ≈ 1.0m`. Side cameras (C2, C3, C5, C6): `τ_gp ≈ 1.5m` (shallower angle → larger projection error).
- **Per-pair fine-score weights** (see Section 2.4).

### 2.2 Dynamic Anchor Selection

At the start of each frame, select the anchor camera `A`:

```
A = argmax_cam  count(detections[cam][frame]  where confidence > 0.5)
```

Ties broken by fixed priority: C1 > C4 > C2 > C3 > C5 > C6 > C7. End-on cameras are preferred tie-breakers because they have the widest full-pitch coverage.

If `A` has zero detections, skip Phase 3 for this frame. All active tracks advance via Kalman prediction only.

### 2.3 Hybrid Cascade Cost Function

For each anchor↔partner pair `(A, B)`, build an `M × N` cost matrix where `M` = detections in `A`, `N` = detections in `B`.

**Gate 1 — Ground-plane distance (always applied):**

Project each detection's foot keypoint (ankle, averaged over left/right if both visible and confidence > 0.4) to the world ground plane via the surveyed pitch plane equation from `pitch_calibration_config.json`:

```
foot_world = backproject_to_plane(ankle_2d, P_camera, pitch_plane)
r_gp(i,j)  = ||foot_world_A_i  −  foot_world_B_j||₂  (Distance on ground plane in meters)
```

If `r_gp(i,j) > τ_gp` (per-pair threshold from Section 2.1): `cost(i,j) = ∞`. Hard prune. This is the most geometrically reliable residual and works identically for all camera pair types.

**Gate 2 — Epipolar consistency (non-degenerate pairs only):**

For non-degenerate pairs, compute the Sampson distance (symmetric epipolar error) over the top-5 highest-confidence keypoints (confidence > 0.5):

```
r_epi(i,j) = mean_k [ sampson_distance(kp_A_i_k, F_AB, kp_B_j_k) ]
```

If `r_epi(i,j) > τ_epi` (default 3px Sampson distance): `cost(i,j) = ∞`. Degenerate pairs skip this gate entirely — `F_AB` is near-singular and the residual is noise.

**Fine score — weighted combination over survivors:**

For cells that survived both gates, assign a fine score used by Hungarian:

```
cost(i,j) = w_epi · r_epi(i,j)  +  w_tri · r_tri(i,j)
```

where `r_tri` is the triangulation reprojection error, reusing the same DLT method as the existing ball pipeline:

```
X_world    = triangulate_DLT(foot_2d_A_i, P_A, foot_2d_B_j, P_B)
r_tri(i,j) = reprojection_error(X_world, P_A, foot_2d_A_i)
           + reprojection_error(X_world, P_B, foot_2d_B_j)
```

### 2.4 Per-Pair Weight Table

| Pair type | `w_epi` | `w_tri` | Notes |
|---|---|---|---|
| Perpendicular (e.g., C1↔C2) | 0.6 | 0.4 | Both residuals fully informative |
| Opposing end-on (C1↔C4) | 0.0 | 1.0 | Epipolar degenerate — skip Gate 2, `w_epi=0` |
| Opposing lateral (C2↔C5) | 0.0 | 1.0 | Epipolar degenerate |
| Diagonal (any↔C7) | 0.4 | 0.6 | Moderate epipolar geometry |

### 2.5 Assignment and Confidence

Solve each of the 6 anchor↔partner problems with the Hungarian algorithm. For each winning match `(i, j)`:

```
track_confidence = 1 − (fine_score(i,j) / fine_score_max)
```

where `fine_score_max` is calibrated from reprojection errors on the surveyed reference points.

**Fallback for anchor-occluded detections:** Any detection in a non-anchor camera unmatched by the anchor path enters a fallback pool. Run one additional Hungarian problem between the two highest-detection non-anchor cameras to rescue these. Matched fallback pairs are assigned `track_confidence` capped at 0.6.

**Single-camera detections:** If a player appears in only 1 camera, mark the correspondence as `single_camera = true` and contribute it to P4a as a lower-confidence observation (no `r_tri` available).

---

## 3. Phase 4a — Online Track Management

### 3.1 State Representation

Each global track maintains a Singer acceleration model state on the ground plane:

```
state = [x, y, vx, vy, ax, ay]
```

The Singer model treats acceleration as a first-order Markov process:
`da/dt = −α · a + w`, where `α` is the maneuver frequency and `w` is white noise. This handles stop-start cricket motion (bowler acceleration/deceleration, fielder sprints) better than constant velocity.

### 3.2 Role-Aware Process Noise

| Role | `Q` profile | `α` (maneuver freq.) |
|---|---|---|
| Bowler | High `Q_vy`, low `Q_vx` (directed run along Y-axis) | High — frequent acceleration changes |
| Wicketkeeper | Small isotropic | Low |
| Striker / Non-striker | Very small | Very low — near-stationary |
| Umpire | Very small | Very low |
| Fielder / Unknown (default) | Moderate isotropic | Medium |

Until P5 assigns a role, all tracks use the Fielder model. P5 role assignment triggers an in-place `Q`(uncertainty in the motion model) and `α` (maneuver frequency) swap without resetting the track state.

### 3.3 Two-Stage Association

Each frame receives high-confidence and low-confidence P3 correspondences:

**Stage 1 — High-confidence matches (confidence > 0.7):**
- Match against all `Confirmed` and `Lost` tracks
- Hungarian on ground-plane Euclidean distance between P3 ground position and Kalman-predicted position
- Gate: reject cells where distance > 1.5m

**Stage 2 — Low-confidence matches (confidence 0.3–0.7):**
- Match remaining unmatched detections against `Confirmed` tracks unmatched in Stage 1
- Same distance-gated Hungarian
- Detections below 0.3 confidence are discarded

### 3.4 Track Lifecycle

```
New detection, no track matched     →  Tentative
Tentative matched ≥ 3 frames        →  Confirmed
Confirmed unmatched 1 frame         →  Lost       (Kalman predicts forward)
Lost matched ≤ 30 frames later      →  Confirmed  (OC-SORT re-update applied)
Lost unmatched for 30 frames        →  Deleted
```

**Bowler exception:** Extend Lost window to 60 frames for tracks where `dominant_role = bowler` AND last known velocity vector points toward the crease. The bowler's run-up frequently takes them out of the anchor camera's view behind the umpire.

### 3.5 OC-SORT Virtual Trajectory Re-Update

When a `Lost` track at frame `t_lost` re-observes at frame `t_now`:

1. Linearly interpolate virtual ground positions from `last_observation(t_lost)` to `current_observation(t_now)` across the gap frames.
2. Feed each virtual position through the Kalman update step sequentially, as if those frames weren't missing.
3. Resume normal prediction from the corrected state.

This prevents the state drift that accumulates during a Lost window from causing a failed Stage 1 re-association.

### 3.6 Re-Entry After Deletion

When a new detection creates a `Tentative` track that reaches `Confirmed` (3 frames), before minting a new `global_player_id`:

```
candidates = Deleted tracks where:
  spatial_gap(deleted.last_ground_pos, new.first_ground_pos) < 3.0m
  AND temporal_gap(deleted.last_frame, new.first_frame) < 120 frames
  AND role_consistent(deleted.dominant_role, position_derived_role_prior)

if candidates: reassign deleted track's global_player_id to new track
else:          mint new global_player_id (P001, P002, ... incrementing)
```

`role_consistent` uses position-derived priors from `pitch_calibration_config.json`: detections near the stumps are consistent with `striker`/`non_striker`/`wicketkeeper`; detections near the boundary with `fielder`. This prevents a deleted `wicketkeeper` from being accidentally resurrected by a fielder running near the stumps.

---

## 4. Phase 4b — Post-Delivery Min-Cost Flow Correction

Runs once after frame 600. Input: per-frame `global_player_id` assignments from P4a.

### 4.1 Segment Extraction

A **segment** is a maximal contiguous run of frames where a given `global_player_id` was in `Confirmed` state. Each segment stores:
`{ global_player_id, start_frame, end_frame, first_ground_pos, last_ground_pos, dominant_role }`

### 4.2 Flow Graph Construction

```
Nodes:   source S,  sink T,  one node per segment endpoint
Edges:
  S → segment_i.start          cost = 0,  capacity = 1
  segment_i.end → T            cost = 0,  capacity = 1
  segment_i.end → segment_j.start  (link edge, if feasible):
    temporal_gap = frame_j.start − frame_i.end
    spatial_gap  = ||pos_j.first − pos_i.last||₂
    feasible if:  temporal_gap ∈ [1, 120]  AND  spatial_gap < 3.0m
    cost = w_t · temporal_gap + w_s · spatial_gap
           + w_r · role_mismatch_penalty(i, j)
```

`role_mismatch_penalty` is 0 if roles match or either is `unknown`; large constant if roles are incompatible (e.g., linking a `bowler` segment to a `wicketkeeper` segment).

### 4.3 Solver

Use `scipy.optimize.linear_sum_assignment` for small graphs (<50 segments) or `networkx.min_cost_flow` for larger. Each unit of flow traces one complete global identity chain across the delivery.

### 4.4 ID Remapping

Segments linked by flow that had different `global_player_id` values in P4a are merged: the earlier segment's ID wins. The per-frame output is patched in-place. Every detected merge is a corrected ID switch and is written directly into the ID-switch report.

---

## 5. Edge Cases

| # | Scenario | Mitigation |
|---|---|---|
| E1 | Anchor camera has 0 detections | Skip P3 this frame; all tracks advance via Kalman prediction (1-frame Lost) |
| E2 | Two players overlap in all cameras (same ground position) | Flag both matches as `low_confidence`; P4b min-cost flow resolves via temporal continuity |
| E3 | Bowler mid-run-up, briefly occluded | Lost window extended to 60 frames for `bowler` tracks with velocity vector toward crease |
| E4 | Side cameras: players stacked at same pixel X (side-on overlap) | `τ_gp` loosened to 1.5m; triangulation fine score relies on 3D depth to disambiguate |
| E5 | Player replaced mid-match (injury substitution) | Out of scope — does not occur within a single 600-frame delivery |
| E6 | Player visible in only 1 camera | Skip `r_tri`; mark `single_camera = true`; contribute as lower-confidence P4a observation |

---

## 6. Output Contract

Per-camera, per-frame JSON — mirrors existing ball pipeline structure:

```jsonc
{
  "delivery_id": "CCPL080626M1_1_14_1_V0",
  "frame_id": 212334,
  "cameras": [
    {
      "camera_id": 1,
      "frame_name": "frame_camera01_000212334.jpg",
      "players": [
        {
          "global_player_id": "P001",
          "role": "bowler",              // "unknown" until P5 runs
          "bbox": [x, y, w, h],         // normalised, from P1
          "track_confidence": 0.91,     // geometric confidence from P3, range 0–1
          "track_state": "confirmed",   // confirmed | lost | tentative
          "single_camera": false,        // true if P3 only saw this player in 1 camera
          "pose_2d": {
            "keypoints": [[x, y]],      // 17 keypoints, normalised
            "confidence": [c]           // per-keypoint confidence
          },
          "pose_3d": {
            "keypoints_world": null     // null until P6 runs
          }
        }
      ]
    }
  ]
}
```

**Field definitions:**
- `track_confidence`: `1 − (fine_score / fine_score_max)`, calibrated from surveyed reference points. Downstream consumers should threshold at ≥ 0.5 for high-reliability use.
- `track_state: "lost"`: position is a Kalman prediction, not an observation. Consumers treat as interpolated.
- `track_state: "tentative"`: track not yet confirmed (< 3 frames). Included in output but flagged as unreliable.

---

## 7. Dependency Map

| Phase | Depends on | Blocks |
|---|---|---|
| P3 | P2 (tracklets), calibration files | P4a |
| P4a | P3 correspondences | P4b, P5, P6 |
| P4b | P4a segments (post-delivery) | Final ID-switch report |
| P5 (role) | P4a global IDs, pitch geometry | P4a Q-model swap, P6 export |

---

## 8. Open Questions (to resolve before implementation)

1. **`τ_epi` value:** Default 3px Sampson distance — needs empirical validation on the first delivery before locking.
2. **Singer `α` per role:** Initial values are estimates; should be tuned against real trajectories once P1/P2 are running.
3. **Min-cost flow weights `w_t`, `w_s`, `w_r`:** Start with `w_t = 0.1` (frames), `w_s = 1.0` (metres), `w_r = 100` (hard penalty); tune from metric 2 results.
4. **Ankle keypoint fallback:** When both ankles are low-confidence (< 0.4), fall back to the centroid of `[left_knee, right_knee]` projected to ground plane. Needs validation that knee projection error stays within `τ_gp`.
5. **E2 overlap resolution in P4b:** When two segment pairs are both feasible link candidates for the same identity, the current cost function does not explicitly prefer the temporally closer one. Add a secondary sort by `temporal_gap` as a tie-breaker in the flow graph edge selection to ensure the nearest-in-time segment wins.
