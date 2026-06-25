"""P4b post-delivery correction via min-cost flow tracklet stitching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

# Fix 3.2: single solver for all graph sizes (max_flow_min_cost with node-split construction)
# Fix 3.3: kinematic feasibility + velocity continuity edge costs

_COST_SCALE = 1000  # scale floats to integers for networkx

_INCOMPATIBLE_ROLE_PAIRS = frozenset([
    frozenset(["bowler", "wicketkeeper"]),
    frozenset(["striker", "wicketkeeper"]),
    frozenset(["bowler", "striker"]),
    frozenset(["bowler", "non_striker"]),
    frozenset(["umpire", "bowler"]),
    frozenset(["umpire", "striker"]),
    frozenset(["umpire", "wicketkeeper"]),
])


@dataclass
class Segment:
    seg_id: int
    global_player_id: str
    start_frame: int
    end_frame: int
    first_ground_pos: np.ndarray
    last_ground_pos: np.ndarray
    dominant_role: str
    exit_velocity: np.ndarray  # velocity at end of segment (for continuity cost)


def _role_mismatch_penalty(role_i: str, role_j: str, w_r: float) -> float:
    if role_i == "unknown" or role_j == "unknown":
        return 0.0
    if frozenset([role_i, role_j]) in _INCOMPATIBLE_ROLE_PAIRS:
        return w_r
    if role_i == role_j:
        return 0.0
    return w_r * 0.3  # partial penalty for different but non-incompatible roles


def _velocity_continuity_cost(seg_i: Segment, seg_j: Segment, temporal_gap: int,
                               frame_rate: float) -> float:
    """Fix 3.3: cost for direction change between segment exit and link direction."""
    dt_s = temporal_gap / frame_rate
    if dt_s < 1e-9:
        return 1.0
    link_vec = seg_j.first_ground_pos - seg_i.last_ground_pos
    link_dist = np.linalg.norm(link_vec)
    if link_dist < 0.1:
        return 0.0  # no movement: no continuity cost
    exit_v = seg_i.exit_velocity
    ev_norm = np.linalg.norm(exit_v)
    if ev_norm < 0.01:
        return 0.0  # stationary: no direction penalty
    cos_sim = float(np.clip((exit_v / ev_norm) @ (link_vec / link_dist), -1.0, 1.0))
    # Penalty: 0 when perfectly aligned, 1 when opposed
    return (1.0 - cos_sim) / 2.0

def build_flow_graph(
    segments: list[Segment],
    w_t: float = 0.1,
    w_s: float = 1.0,
    w_r: float = 100.0,
    v_max_mps: float = 9.0,
    frame_rate: float = 50.0,
    temporal_gate_frames: int = 120,
) -> nx.DiGraph:
    """Build directed graph for min-cost tracklet stitching.

    Fix 3.2: single solver for all graph sizes.

    Graph structure (for use by solve_flow):
      - Nodes: one per segment (keyed by seg_id)
      - Edges: seg_i -> seg_j with 'link_cost' attribute for feasible transitions
      - Node attribute 'segment' stores the Segment object
      - Graph attribute 'new_traj_cost' stores per-edge no-link cost

    solve_flow uses scipy.optimize.linear_sum_assignment on this bipartite
    representation — equivalent to min-cost flow on a bipartite graph.
    """
    # No-link cost: cost to leave a segment unlinked (start new trajectory)
    # Link is preferred when link_cost < new_traj_cost
    _new_traj_cost = w_s * 0.5  # in original (unscaled) units

    G = nx.DiGraph()
    G.graph["new_traj_cost"] = _new_traj_cost
    G.graph["w_t"] = w_t
    G.graph["w_s"] = w_s
    G.graph["w_r"] = w_r
    G.graph["v_max_mps"] = v_max_mps
    G.graph["frame_rate"] = frame_rate
    G.graph["temporal_gate_frames"] = temporal_gate_frames

    for seg in segments:
        G.add_node(seg.seg_id, segment=seg)

    for seg_i in segments:
        for seg_j in segments:
            if seg_i.seg_id == seg_j.seg_id:
                continue
            temporal_gap = seg_j.start_frame - seg_i.end_frame
            if temporal_gap <= 0 or temporal_gap > temporal_gate_frames:
                continue  # Fix 3.3: no temporal overlap allowed

            spatial_gap = float(np.linalg.norm(
                seg_j.first_ground_pos - seg_i.last_ground_pos
            ))

            # Kinematic feasibility: Fix 3.3
            v_max_per_frame = v_max_mps / frame_rate
            if spatial_gap > v_max_per_frame * temporal_gap * 1.5:
                continue

            role_pen = _role_mismatch_penalty(seg_i.dominant_role, seg_j.dominant_role, w_r)
            vel_cost = _velocity_continuity_cost(seg_i, seg_j, temporal_gap, frame_rate)

            link_cost = (
                w_t * temporal_gap
                + w_s * spatial_gap
                + role_pen
                + 0.5 * vel_cost
            )
            G.add_edge(seg_i.seg_id, seg_j.seg_id, link_cost=link_cost)

    return G


def solve_flow(G: nx.DiGraph) -> dict[int, int]:
    """Fix 3.2: single solver (bipartite min-cost matching) for all graph sizes.

    Solves the tracklet stitching as a bipartite assignment problem:
    - Rows = segment tails (possible chain predecessors)
    - Cols = segment heads (possible chain successors) + 1 dummy column (no-link)
    - Cost = link_cost if feasible edge exists, else large (infeasible)
    - Dummy column cost = new_traj_cost (cost to not link)

    Returns dict mapping seg_id_i -> seg_id_j for linked pairs.
    """
    from scipy.optimize import linear_sum_assignment

    seg_ids = sorted(G.nodes)
    N = len(seg_ids)
    if N == 0:
        return {}

    new_traj_cost = G.graph.get("new_traj_cost", 1.0)
    _LARGE = new_traj_cost * 1000  # infeasible link cost (kinematic/overlap violation)

    id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}

    # Cost matrix: N rows (tails) x (N+1) cols (heads + dummy no-link)
    # col N = dummy (no link), cost = new_traj_cost
    cost = np.full((N, N + 1), _LARGE)
    cost[:, N] = new_traj_cost  # dummy column: cost to not link

    for i_id, j_id, data in G.edges(data=True):
        r = id_to_idx[i_id]
        c = id_to_idx[j_id]
        cost[r, c] = data["link_cost"]

    rows, cols = linear_sum_assignment(cost)

    links: dict[int, int] = {}
    for r, c in zip(rows, cols):
        if c < N:  # not the dummy column → solver preferred a real link
            i_id = seg_ids[r]
            j_id = seg_ids[c]
            if cost[r, c] < _LARGE:  # only if it's a feasible edge (not infeasible)
                links[i_id] = j_id
    return links


def extract_segments(
    per_frame_records: list[dict[str, Any]],
) -> list[Segment]:
    """Extract maximal confirmed-state runs per global_player_id from frame records."""
    # per_frame_records: list of per-frame output dicts with 'players' list
    from collections import defaultdict
    runs: dict[str, list[dict]] = defaultdict(list)
    for record in per_frame_records:
        for player in record.get("players", []):
            if player.get("track_state") == "confirmed" and player.get("global_player_id"):
                runs[player["global_player_id"]].append({
                    "frame": record["frame_index"],
                    "ground_xy": player.get("_ground_xy"),  # internal field
                    "role": player.get("role", "unknown"),
                })

    segments: list[Segment] = []
    seg_counter = 0
    for pid, frames in runs.items():
        if not frames:
            continue
        frames.sort(key=lambda f: f["frame"])
        # Split into maximal contiguous runs
        i = 0
        while i < len(frames):
            j = i
            while j + 1 < len(frames) and frames[j+1]["frame"] == frames[j]["frame"] + 1:
                j += 1
            run = frames[i:j+1]
            gxys = [np.asarray(f["ground_xy"]) for f in run if f["ground_xy"] is not None]
            if not gxys:
                i = j + 1; continue
            # Exit velocity: last two positions if available
            if len(gxys) >= 2:
                exit_v = gxys[-1] - gxys[-2]
            else:
                exit_v = np.zeros(2)
            from collections import Counter
            role_counts = Counter(f["role"] for f in run)
            dominant_role = role_counts.most_common(1)[0][0]
            segments.append(Segment(
                seg_id=seg_counter,
                global_player_id=pid,
                start_frame=run[0]["frame"],
                end_frame=run[-1]["frame"],
                first_ground_pos=gxys[0].copy(),
                last_ground_pos=gxys[-1].copy(),
                dominant_role=dominant_role,
                exit_velocity=exit_v,
            ))
            seg_counter += 1
            i = j + 1
    return segments


def remap_ids(
    per_frame_records: list[dict[str, Any]],
    segments: list[Segment],
    links: dict[int, int],
) -> list[dict[str, Any]]:
    """Apply flow links: merge chained segments to earliest ID. Return ID-switch report."""
    seg_by_id = {s.seg_id: s for s in segments}

    # Build chains: follow link pointers to find the root (earliest) segment
    def find_root(sid: int) -> str:
        seen = set()
        while sid in links and links[sid] != sid:
            if sid in seen: break
            seen.add(sid)
            # links[sid] is the successor; to find the root of the chain, go backward
            break
        return seg_by_id[sid].global_player_id

    # Build reverse map: for each segment in a chain, what is the root ID?
    # Chain: root -> ... -> tail. The root's ID wins.
    # Build successor chains, then walk backwards to find roots.
    pred: dict[int, int] = {}
    for i_id, j_id in links.items():
        pred[j_id] = i_id  # j's predecessor is i

    def get_chain_root(sid: int) -> int:
        while sid in pred:
            sid = pred[sid]
        return sid

    # Build remap: old_pid -> new_pid
    id_remap: dict[str, str] = {}
    switch_report: list[dict] = []
    for s in segments:
        root_sid = get_chain_root(s.seg_id)
        root_pid = seg_by_id[root_sid].global_player_id
        if root_pid != s.global_player_id:
            id_remap[s.global_player_id] = root_pid
            switch_report.append({
                "merged_id": s.global_player_id,
                "into_id": root_pid,
                "at_frame": s.start_frame,
            })

    # Patch per-frame records
    for record in per_frame_records:
        for player in record.get("players", []):
            old_id = player.get("global_player_id")
            if old_id and old_id in id_remap:
                player["global_player_id"] = id_remap[old_id]

    return switch_report
