"""Phase 3: dissect the plant's signalling and compare it to real plants.

Real plant electrical signalling (the thing we want to compare against): a local
wound (herbivory) triggers an electrical signal -- an action/variation potential
-- that **propagates systemically** at a finite speed through the vasculature and
**arms defense in distant, undamaged tissue**. We reproduce that experiment in
silico: grow a plant, deliver a single controlled wound to one leaf, and watch
the wound signal and defense spread across the graph.

Measured here:
* **propagation** -- wound signal / defense vs time, by graph hop-distance from
  the wound (the travelling wave);
* **speed / latency** -- when defense arms at each hop-distance (-> hops per step);
* **directionality** -- whether the response favours shoots or roots.

Controller-agnostic: pass any ``attach_fn`` (hand-designed or learned).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from phytonet.config import build_simulation
from phytonet.plant import NodeKind
from phytonet.scenarios import make_scenario


def hop_distances(plant, source: int) -> dict[int, int]:
    """BFS hop-distance from ``source`` to every reachable node."""
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for v in plant.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def wound_experiment(attach_fn, params=None, scenario: str = "baseline", seed: int = 0,
                     grow_days: int = 45, observe_days: int = 18, wound: float = 4.0):
    """Grow a plant, wound one peripheral leaf, and record the signal/defense
    spread across the *original* tissues (new growth is ignored for distances)."""
    sim = build_simulation(make_scenario(scenario, days=grow_days + observe_days,
                                         seed=seed, record_frames=False))
    attach_fn(sim, params)
    sim.run(grow_days)
    leaves = sim.plant.nodes_of(NodeKind.LEAF)
    if not leaves:
        return None
    # Wound the leaf farthest from the seed -- a peripheral site, like a herbivore.
    from_seed = hop_distances(sim.plant, min(sim.plant.nodes))
    target = max(leaves, key=lambda n: from_seed.get(n.id, 0))
    dist = hop_distances(sim.plant, target.id)
    kinds = {nid: n.kind.value for nid, n in sim.plant.nodes.items()}
    original = set(sim.plant.nodes)

    target.state["wound"] = target.state.get("wound", 0.0) + wound
    target.state["V"] = target.state.get("V", 0.0) + wound

    series = []
    for _ in range(observe_days):
        sim.step()
        snap = {nid: (sim.plant.nodes[nid].state.get("wound", 0.0),
                      sim.plant.nodes[nid].state.get("defense", 0.0))
                for nid in original if nid in sim.plant.nodes}
        series.append(snap)
    return {"dist": dist, "kinds": kinds, "series": series, "target": target.id}


def defense_by_distance(result) -> tuple[np.ndarray, list[int]]:
    """(time x distance) mean-defense array + the distance bins present."""
    dists = sorted({result["dist"][n] for n in result["dist"]
                    if n in result["series"][0]})
    mat = np.zeros((len(result["series"]), len(dists)))
    for t, snap in enumerate(result["series"]):
        for di, d in enumerate(dists):
            vals = [snap[n][1] for n in snap if result["dist"].get(n) == d]
            mat[t, di] = float(np.mean(vals)) if vals else 0.0
    return mat, dists


def propagation_speed(result, thresh: float = 0.05) -> dict:
    """Latency (steps) for defense to arm at each hop-distance, and a speed
    estimate (hops/step) from a linear fit of distance vs latency."""
    mat, dists = defense_by_distance(result)
    latency = {}
    for di, d in enumerate(dists):
        over = np.where(mat[:, di] > thresh)[0]
        if over.size:
            latency[d] = int(over[0])
    speed = float("nan")
    if len(latency) >= 2:
        ds = np.array(sorted(latency))
        ts = np.array([latency[d] for d in ds])
        if ts.std() > 1e-9:
            speed = float(np.polyfit(ts, ds, 1)[0])  # hops per step
    return {"latency": latency, "speed_hops_per_step": speed}


def calibrate_speed(hops_per_step: float, internode_mm: float = 10.0,
                    step_seconds: float = 5.0) -> dict:
    """Map the model's propagation speed (hops/step) to a physical speed (mm/s),
    given an internode length and the assumed real duration of one signalling
    step, and compare to measured plant action/variation potentials (~0.1-40
    mm/s). Note: the sim conflates the growth timestep with the signalling
    timestep -- ``step_seconds`` is the (separable) signalling timescale used here
    purely for calibration."""
    mm_per_s = hops_per_step * internode_mm / max(step_seconds, 1e-9)
    lo, hi = 0.1, 40.0   # real plant electrical-signal velocity range (mm/s)
    return {"mm_per_s": float(mm_per_s), "realistic_range_mm_per_s": (lo, hi),
            "in_range": lo <= mm_per_s <= hi,
            "assumptions": {"internode_mm": internode_mm, "step_seconds": step_seconds}}


def directionality(result) -> dict:
    """Peak defense reached in shoot vs root tissues (systemic reach by organ)."""
    peak = {}
    for nid in result["series"][0]:
        peak[nid] = max(snap[nid][1] for snap in result["series"] if nid in snap)
    shoot = [peak[n] for n in peak if result["kinds"][n] in ("stem", "leaf", "seed")]
    root = [peak[n] for n in peak if result["kinds"][n] == "root"]
    return {"shoot_peak_defense": float(np.mean(shoot)) if shoot else 0.0,
            "root_peak_defense": float(np.mean(root)) if root else 0.0}
