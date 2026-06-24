# P2 — Per-Camera Tracking Design Spec

**Date:** 2026-06-23 (rev. 2026-06-24 — robustness pass: numerical stability, spatially-gated
dormant re-ID, low-conf bootstrap, scale-adaptive gating, EOF flush, IPC discipline, diagnostics;
rev. 2026-06-24b — P1-alignment pass: input path `outputs/p1/predictions/`, `stage2_confidence_min`
coupled to P1 `--conf`, explicit pixel-space input fields, camera-prefixed `local_track_id`)
**Phase:** P2 (depends on P1, feeds P3)
**Owner:** Group 1 (Aksh, Vedant, Anshul)
**Status:** Approved for implementation

---

## 1. Goal

For every camera, link per-frame detections produced by P1 into stable per-camera tracklets
by filling in `local_track_id` on each player record. No cross-camera association (that is P3).
No global IDs (that is P4). P2's sole job is intra-camera temporal linking.

---

## 2. Inputs & Outputs

**Input:**
- Per-camera JSONL from P1: `outputs/p1/predictions/cam_XX.jsonl` (the runner writes under a
  `predictions/` subdirectory — see `p1_runner.py`)
- One line per frame, schema `g1_player_frame/v0`
- Each player record has `local_track_id: null`, `global_player_id: null`
- Tracking consumes the **pixel-space** fields (`bbox_xywh_px`, `pose_2d.keypoints_px`,
  `pose_2d.confidence`); the parallel normalised fields are passed through untouched (§4)
- Frames arrive in ascending `frame_index` order

**Output:**
- Updated JSONL: `outputs/p2/cam_XX.jsonl`
- Same schema `g1_player_frame/v0`, all fields preserved
- `local_track_id` filled in for confirmed tracks, camera-prefixed to match the contract's
  canonical example in `contract.py` (e.g. `"cam_01_trk_0001"`)
- `local_track_id` remains `null` for detections that never confirm
- `global_player_id`, `role`, `pose_3d` remain `null` — P2 does not touch them
- Per-camera diagnostic JSON: `outputs/p2/diagnostics/cam_XX.json`

---

## 3. Architecture Overview

P2 is a **stateful per-camera tracker** built on BoT-SORT's Kalman motion core, with the
appearance branch replaced by a **pose-configuration gallery**. This design is deliberately
chosen because cricket kits are identical across players, making appearance embeddings
near-useless for intra-camera re-identification; pose shape (crouching wicketkeeper vs.
mid-run-up bowler) carries far more discriminative signal.

Association follows a two-stage ByteTrack-style scheme per frame:

1. **Stage 1** — high-confidence detections matched to active tracks via IoU + pose affinity
2. **Stage 2** — low-confidence detections matched to unmatched active tracks via IoU only

Lost tracks enter a **dormant gallery** for up to 60 frames. Re-entry re-ID is **spatially
gated first** (a reachability radius from the propagated last-known position) and **then**
discriminated by pose-gallery cosine distance — pose is never used without a spatial constraint,
since identical kits and shared crouch postures make pure-pose matching ambiguous across the
field (§5).

Each camera runs independently; all 7 cameras are processed in parallel, with each worker doing
its own disk I/O so no per-frame data crosses the process boundary (§8).

---

## 4. Pose Vector Representation

All pose-based costs operate on a normalised **pose vector** derived from P1's **pixel-space**
keypoints (`pose_2d.keypoints_px`), together with a parallel **validity mask** (one bool per
keypoint). Both are required — the mask is never discarded, because a zeroed coordinate is
geometrically indistinguishable from a joint that genuinely lies at the origin. All gating and
motion thresholds in §5 (`v_max_px_per_frame`, `gate_max_distance_px`, the bbox-height scale
floor) are likewise expressed in pixels and operate on `bbox_xywh_px`, so the whole tracker
lives in one consistent coordinate space.

A keypoint is **valid** iff its P1 confidence (`pose_2d.confidence[i]`) is
`≥ pose_keypoint_confidence_min` (0.3).

1. **Root selection (gated):** the root is the hip midpoint `(kp[11] + kp[12]) / 2`, but only
   if **both** hips are valid. If exactly one hip is valid, use that hip alone. If neither hip
   is valid, fall back to the shoulder midpoint `(kp[5] + kp[6]) / 2` (using whichever
   shoulders are valid). If no valid root anchor exists at all, the pose vector is marked
   **undefined** — pose cost is skipped for this detection and association falls back to IoU
   only (see §5). A single bad hip must not silently translate all 17 joints.
2. **Root-relative:** subtract the selected root from every keypoint.
3. **Scale normalisation (cascade with floor):** divide all coordinates by the first anchor in
   this cascade whose endpoints are both valid **and** whose length exceeds `scale_min_px`:
   1. left torso `‖kp[5] - kp[11]‖`
   2. right torso `‖kp[6] - kp[12]‖`
   3. shoulder width `‖kp[5] - kp[6]‖`
   4. hip width `‖kp[11] - kp[12]‖`
   5. bbox diagonal (always available — last-resort anchor)

   The chosen scale is floored: `scale = max(anchor_length, scale_min_px)`. This prevents
   division by zero/near-zero when an anchor is geometrically degenerate (e.g. shoulder and hip
   projecting to nearly the same pixel at oblique camera angles), which confidence alone does
   not catch. `scale_min_px` is expressed as a fraction of bbox height so it adapts to player
   scale and resolution.
4. **No hard zeroing.** Keypoints are *not* overwritten with zeros. Invalid keypoints keep
   their (unreliable) coordinates but are excluded from cost computation via the validity mask.
5. **Flatten** to a 34-dimensional vector `[x0, y0, x1, y1, ..., x16, y16]`, carried alongside
   its 17-element validity mask.

### Pose distance (confidence-aware, NaN-safe)

Pose cost between two vectors is a **masked, confidence-weighted cosine distance** computed
only over keypoints valid in **both** vectors:

- Let `S` = set of keypoints valid in both `a` and `b`. Per-dimension weight = `min(conf_a, conf_b)`.
- If `|S| < min_shared_keypoints` (default 6), return cost = 1.0 (treat as **no pose match** —
  never a NaN, never a spurious low cost from too little evidence).
- Otherwise compute weighted cosine over the 2·|S| coordinates of `S`. If either masked vector
  has (near-)zero norm, return cost = 1.0 rather than dividing.

This guarantees the cost matrix is always finite, so Hungarian assignment never receives NaN.

This vector is used in the Stage 1 cost matrix and the dormant re-ID gallery match.

---

## 5. Cost Matrix & Association

### Stage 1 — High-confidence detections (confidence > 0.5)

```
cost(d, t) = α · IoU_cost(d, t) + β · pose_cost(d, t)
```

- `IoU_cost(d, t)` = `1 - IoU(d.bbox, t.kalman_predicted_bbox)`
- `pose_cost(d, t)` = masked confidence-weighted cosine distance (§4) between `d`'s pose vector
  and `t`'s gallery representative (§7). **If `d`'s pose vector is undefined (no valid root) or
  `|S| < min_shared_keypoints`, set `β_eff = 0` for this pair and use IoU alone** — pose is
  never allowed to inject a fabricated cost.
- Default weights: `α = 0.6`, `β = 0.4` (tunable via config). When pose is unavailable for a
  pair, weights are renormalised so IoU carries the full cost.
- **Gate (scale-adaptive):** a pair is excluded from the cost matrix iff `IoU = 0` AND the
  Kalman-predicted centre is farther from the detection centre than the gate radius. The radius
  is **not a fixed pixel constant**; it is the larger of:
  - a **Mahalanobis gate** on the Kalman position covariance (`chi2_gate`, default 9.21 ≈ 99%
    for 2 DoF), which adapts to motion uncertainty and is the primary gate, and
  - a **size-relative floor** `gate_bbox_factor · track.bbox_height` (default 1.5×), which
    adapts to depth/resolution/player-size since bbox height is a depth proxy.

  `gate_max_distance_px` is retained only as an absolute hard cap (sanity clamp), not the
  primary mechanism.
- Hungarian assignment; pairs with `cost > 0.7` are rejected (no match)

### Stage 2 — Low-confidence detections (`stage2_confidence_min` ≤ confidence ≤ 0.5)

```
cost(d, t) = IoU_cost(d, t)
```

> **Cross-phase dependency (P1 `--conf`).** P1 only writes detections whose score is `≥ --conf`,
> which is set to **0.1** in `run_cricket_p1_inference.py` so the full `[0.1, 0.5]` low-conf band
> reaches the tracker. `stage2_confidence_min` **must stay ≥ P1's `--conf`** — if P1 is ever run
> with a higher `--conf`, anything below it never reaches the input and the LOW-CONF TENTATIVE
> bootstrap below is silently starved. The two values are coupled and must be chosen together.

- Only unmatched active tracks from Stage 1 participate
- Pose cost omitted (pose is unreliable at low confidence)
- Same 0.7 cost threshold
- Unmatched low-confidence detections **may spawn a LOW-CONF TENTATIVE track** (they no longer
  vanish). To avoid track spam from noise, a low-conf-seeded tentative track has a **stricter
  confirmation rule** than a high-conf one: it must accumulate at least one high-confidence
  (`> 0.5`) hit within the confirmation window in addition to the normal hit count, otherwise
  it is deleted. This recovers fielders that enter frame or emerge from occlusion at low
  confidence (common for small/distant players near the boundary at 2560×1440) without polluting
  the track set. Configurable via `lowconf_can_spawn` (default `true`).

### Dormant re-identification (track lost 1–60 frames)

When a new high-confidence detection is unmatched after both stages, it is tested against
dormant tracks before spawning a new track. **Dormant re-ID is spatially gated first, then
discriminated by pose** — pose alone is insufficient because 13+ players/officials wear
identical kits and frequently share crouch/ready postures, so a pure pose match would happily
bind a fielder entering frame-left to a dormant track last seen frame-right.

**Step 1 — reachability gate (mandatory).** A dormant track is only a *candidate* if the
detection centre lies within its reachability radius from the track's **last-known centre**
(propagated forward, see DORMANT in §6):

```
reach_radius = v_max_px_per_frame · frames_dormant + gate_bbox_factor · bbox_height
```

Dormant tracks outside this radius are not considered, regardless of pose similarity. This
alone eliminates the dominant cross-field false-match failure mode.

**Step 2 — pose discrimination among candidates only.**

```
cost(d, t_dormant) = masked_weighted_cosine(d.pose_vector, t_dormant.gallery_repr)   // §4, §7
```

- Skip any candidate whose pose comparison is undefined (`|S| < min_shared_keypoints`); if no
  candidate has a defined comparison, spawn a new TENTATIVE track.
- Accept the lowest-cost candidate if `cost < pose_cosine_reid_threshold` (0.25); otherwise
  spawn a new TENTATIVE track.
- **Ambiguity guard:** if two or more candidates pass the threshold and their costs differ by
  less than `reid_ambiguity_margin` (default 0.05), the match is **ambiguous** — do **not**
  guess; spawn a new TENTATIVE track and increment `dormant_reid_ambiguous` in diagnostics. A
  wrong re-ID is a hard ID switch; a fresh track is recoverable downstream.
- On successful re-ID the Kalman state is **re-seeded** from the detection bbox, but the prior
  velocity estimate is retained as the initial velocity (smooth handoff) rather than zeroed.

---

## 6. Track Lifecycle

```
TENTATIVE → CONFIRMED → DORMANT → [re-confirmed | DELETED]
```

### TENTATIVE
- Spawned when a high-confidence detection is unmatched by both stages and dormant re-ID, **or**
  from an unmatched low-confidence detection (LOW-CONF TENTATIVE, see §5 Stage 2)
- No `local_track_id` assigned yet
- Must achieve **3 matches within the first 5 frames** to be promoted to CONFIRMED. A LOW-CONF
  TENTATIVE additionally requires at least one `> 0.5` hit within the window.
- If not promoted within 5 frames, the track is deleted silently (no ID emitted)
- Records for tentative frames are **buffered** and flushed retroactively with the assigned
  ID once promoted; flushed with `null` if never promoted
- **End-of-stream finalisation:** when the input ends while a track is still TENTATIVE, the
  confirmation window is closed early. If the track already meets the hit count (and, for
  LOW-CONF tracks, the high-conf-hit requirement), it is **promoted and flushed with its ID**
  even though 5 frames have not elapsed; otherwise its buffered records are flushed with `null`.
  No buffer is left undrained at EOF. Tracks unresolved at EOF are counted in diagnostics
  (`tentative_unresolved_at_eof`).

### CONFIRMED
- Assigned a `local_track_id` of the form `cam_XX_trk_XXXX` (camera id + a per-camera monotonic
  counter starting at `0001`, e.g. `cam_01_trk_0001`) — this matches the canonical example in
  `contract.py`. The contract validator does not enforce a format, but emitting the
  camera-prefixed form keeps the id self-identifying for P3/P4.
- Kalman state updated every frame (predicted if no match, corrected if matched)
- Pose gallery updated on every matched detection
- Transitions to DORMANT on first unmatched frame

### DORMANT
- Kalman state is **propagated forward with inflated process noise** (constant-velocity, growing
  covariance) rather than hard-frozen. The point estimate is *not* trusted for matching, but the
  propagated **last-known centre** + `frames_dormant` provides the reachability radius used by
  the dormant re-ID spatial gate (§5). Covariance growth is capped (see below) to avoid overflow.
- **Covariance guard:** if the position covariance trace exceeds `kalman_cov_trace_max`, the
  track is force-deleted (the prediction is meaningless) and `kalman_cov_explosions` is
  incremented. This prevents a divergent filter from producing garbage gates.
- Pose gallery preserved
- Deleted after `dormant_max_frames` (60) consecutive unmatched frames
- Can be re-confirmed by a dormant re-ID match (transitions back to CONFIRMED); Kalman is
  re-seeded from the matching detection, retaining the propagated velocity (§5)

### DELETED
- Terminal state; `local_track_id` is **never reused** within a delivery
- **Scope of uniqueness:** the numeric counter is unique only within `(camera_id, delivery_id)`
  and resets per delivery. The emitted `cam_XX_trk_XXXX` string is therefore unique within a
  delivery but **not** across deliveries — P3/P4 must still key on the
  `(camera_id, delivery_id, local_track_id)` triple, not the `local_track_id` string alone.

---

## 7. Pose Gallery

Each CONFIRMED track maintains a **ring buffer of the last 30 pose vectors** (with their
validity masks and a per-vector mean confidence):

- Updated on every matched detection (append new vector, drop oldest if at capacity). Vectors
  whose pose is undefined (no valid root, §4) are **not** stored — the gallery never holds a
  fabricated vector.
- **Gallery representative = medoid, not mean.** The representative used by `pose_cost` and
  dormant re-ID is the buffer member with the lowest summed masked-cosine distance to the other
  members (the most central *real* pose). A flat mean of root-relative poses is geometrically
  unsound — averaging across a posture transition (run-up → delivery) yields a centroid that
  matches neither pose. The medoid is always a valid observed configuration.
- A rolling masked mean is still maintained for diagnostics only; it is recomputed from the
  buffer on a fixed cadence (not by running-sum subtraction) to avoid float drift over 600
  frames.
- Ring buffer size of 30 is intentional: recent pose history without contamination from stale
  postures earlier in the delivery.

---

## 8. I/O & Parallelism

**Processing model:**
- Frame-by-frame sequential read within each camera
- 5-frame write buffer for TENTATIVE retroactive fill (see §6), fully drained at EOF (§6)
- All 7 cameras run in parallel: `concurrent.futures.ProcessPoolExecutor(max_workers=N)`
- Single-camera debug mode: `python track.py --camera cam_01`

**IPC discipline (avoid pickle overhead):**
- Each worker **reads its own `outputs/p1/predictions/cam_XX.jsonl` and writes its own `outputs/p2/cam_XX.jsonl` +
  diagnostic JSON directly to disk.** Cameras are fully independent (§3, §13), so frame
  structures, pose vectors, and diagnostics are **never** marshalled back through the process
  pool. Workers return only a small status tuple `(camera_id, status, summary_counts, error)` —
  a few hundred bytes, not megabytes of per-frame data. This eliminates the large-result pickle
  / pipe-deadlock risk.
- `max_workers` defaults to `min(7, os.cpu_count())` to avoid oversubscribing CPU-bound work on
  boxes with fewer than 7 physical cores.

**Windows / `spawn` safety (this pipeline targets win32):**
- `ProcessPoolExecutor` uses the **spawn** start method on Windows — every worker re-imports the
  entry module. The entry point **must** be guarded by `if __name__ == "__main__":`, and the
  per-camera worker function plus the config object must be top-level and picklable (no
  closures/lambdas). The parent must call `future.result()` for every camera so worker
  exceptions (e.g. file-read errors) surface instead of being silently swallowed.

**Entry point:** (realised as `scripts/run_cricket_p2_tracking.py` to follow the repo's
established `scripts/run_cricket_*` convention)
```
python scripts/run_cricket_p2_tracking.py \
    --input-dir outputs/p1/predictions \
    --output-dir outputs/p2 \
    --delivery-id CCPL080626M1_1_14_1 \
    --config configs/p2_tracking.yaml \
    [--camera cam_01]   # optional, single-camera debug
```

---

## 9. Tunable Config (`configs/p2_tracking.yaml`)

```yaml
# Stage thresholds
stage1_confidence_threshold: 0.5
stage2_confidence_min: 0.1          # matches P1 inference --conf (now 0.1); MUST stay >= P1 --conf
                                    # so the [0.1, 0.5] low-conf band is actually populated (see §5 Stage 2)
cost_accept_threshold: 0.7
lowconf_can_spawn: true            # low-conf detections may seed LOW-CONF TENTATIVE tracks

# Cost matrix weights
iou_alpha: 0.6
pose_beta: 0.4                     # renormalised to IoU-only when pose is undefined for a pair

# Pose vector
pose_keypoint_confidence_min: 0.3
min_shared_keypoints: 6           # below this, pose cost = 1.0 (no match), never NaN
scale_min_frac_bbox_h: 0.05       # scale floor as fraction of bbox height (resolution-safe)

# Spatial / motion gating (scale- and motion-adaptive; replaces fixed-pixel gate)
chi2_gate: 9.21                   # Mahalanobis gate, 2 DoF ~99%
gate_bbox_factor: 1.5             # size-relative floor = factor * track bbox height
gate_max_distance_px: 600         # absolute hard cap / sanity clamp only
v_max_px_per_frame: 120           # reachability for dormant re-ID (per-camera override allowed)

# Dormant re-ID
pose_cosine_reid_threshold: 0.25
reid_ambiguity_margin: 0.05       # ambiguous candidates → new track, not a guess
dormant_max_frames: 60

# Kalman stability
kalman_cov_trace_max: 1.0e6       # force-delete + log on covariance explosion

# Track confirmation
tentative_confirm_hits: 3
tentative_confirm_window: 5

# Gallery
pose_gallery_size: 30
gallery_repr: medoid              # medoid | mean (medoid recommended)
```

---

## 10. Diagnostic Log (`outputs/p2/diagnostics/cam_XX.json`)

Emitted per camera per delivery. Feeds P7's ID-switch report and failure-case library.

```jsonc
{
  "camera_id": "cam_01",
  "delivery_id": "CCPL080626M1_1_14_1",
  "status": "ok",                        // ok | failed
  "error": null,                         // error string if status == failed
  "frames_expected": 600,
  "frames_read": 600,                    // mismatch ⇒ truncated/corrupt input

  "total_tracks_spawned": 24,            // TENTATIVE tracks created (incl. low-conf-seeded)
  "lowconf_tracks_spawned": 4,           // of which seeded by Stage-2 low-conf detections
  "confirmed_tracks": 18,                // promoted to CONFIRMED
  "tentative_rejected": 6,               // never confirmed
  "tentative_unresolved_at_eof": 1,      // still tentative when stream ended (§6)
  "dormant_reidentified": 3,             // successful dormant re-ID matches
  "dormant_reid_ambiguous": 1,           // passed threshold but within ambiguity margin → new track
  "dormant_deleted": 2,                  // expired without re-ID

  // id_switches_estimated heuristic (now DEFINED, reproducible):
  //   count of frames where a detection's best Stage-1 assignment changed the track it
  //   matched relative to the previous frame AND IoU with the prior track was still > 0.3
  //   (i.e. a plausible swap, not a genuine exit). Recorded so two implementers agree.
  "id_switches_estimated": 1,
  "frames_with_unmatched_detections": 12,

  // math / stability guards — how often the §4/§5/§6 safety paths fired:
  "pose_undefined_count": 7,             // detections with no valid root anchor
  "pose_skipped_low_overlap": 15,        // pose cost forced to 1.0 (|S| < min_shared_keypoints)
  "scale_floor_hits": 3,                 // degenerate torso anchor caught by scale_min floor
  "cost_rejections": 9,                  // pairs rejected by cost_accept_threshold
  "stage2_discarded": 5,                 // low-conf dets dropped (when lowconf_can_spawn=false)
  "kalman_cov_explosions": 0,            // tracks force-deleted on covariance blow-up

  "per_track": {
    "cam_01_trk_0001": {
      "length_frames": 580,
      "gap_count": 1,
      "max_gap_frames": 23,
      "max_cov_trace": 1240.5            // peak Kalman position covariance trace
    }
    // ...
  }
}
```

---

## 11. Exit Criteria

- `local_track_id` filled in for all confirmed-track detections across a full delivery (all
  600 frames, all 7 cameras)
- Per-camera diagnostic logs produced
- Intra-camera ID-switch count and track completeness numbers recorded (feeds P7 metrics 2 & 4)
- Single-camera debug mode verified on at least one delivery before parallel run

---

## 12. Open Questions

| # | Question | Impact |
|---|---|---|
| 1 | What is the actual maximum occlusion gap seen in the footage? | Sets `dormant_max_frames` empirically |
| 2 | Do any cameras have severe crop-exit/re-entry patterns (fielders fully leaving frame)? | May need per-camera `dormant_max_frames` override |
| 3 | Is torso height a stable normalisation anchor across all 7 camera angles? | Resolved in §4 via ordered scale cascade (torso → shoulder → hip → bbox) with a floor; remaining question is only the best *default* anchor per camera |
| 4 | What is a realistic `v_max_px_per_frame` per camera (near-end cams see fast pixel motion, wide diagonal C7 sees slow)? | Sets the dormant reachability gate; likely needs per-camera override |
| 5 | What `scale_min_frac_bbox_h` and `chi2_gate` values hold up empirically across the 7 angles? | Tune the degeneracy floor and Mahalanobis gate on a labelled delivery |

---

## 13. Dependencies & What P2 Does Not Do

| Concern | Owner |
|---|---|
| Cross-camera association | P3 |
| Global player ID assignment | P4 |
| Role classification | P5 |
| 3D pose reconstruction | P6 |
| Appearance-based ReID embeddings | Not used — kits are identical |
