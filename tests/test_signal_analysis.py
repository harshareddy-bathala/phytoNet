"""Tests for the Phase-3 wound-propagation analysis."""

import numpy as np

from phytonet.plant import NodeKind, Plant
from phytonet.signaling import SignalingParams, attach_signaling
from phytonet.signal_analysis import (
    defense_by_distance, directionality, hop_distances, propagation_speed,
    wound_experiment,
)


def test_hop_distances_on_a_chain():
    p = Plant(dim=2)
    ids = [p.add_node(NodeKind.STEM, [0.0, float(i)]).id for i in range(4)]
    for a, b in zip(ids, ids[1:]):
        p.add_edge(a, b)
    d = hop_distances(p, ids[0])
    assert [d[i] for i in ids] == [0, 1, 2, 3]


def test_wound_propagates_systemically():
    res = wound_experiment(attach_signaling, SignalingParams(), scenario="baseline",
                           seed=0, grow_days=40, observe_days=14, wound=6.0)
    assert res is not None
    # Defense should appear beyond the wound site (distance >= 2) at some point.
    reached = [res["dist"][n] for snap in res["series"] for n, (_, d) in snap.items()
               if d > 0.05 and res["dist"].get(n, 0) >= 2]
    assert reached, "defense never propagated beyond the immediate neighbourhood"


def test_propagation_speed_and_directionality():
    res = wound_experiment(attach_signaling, SignalingParams(), scenario="baseline",
                           seed=0, grow_days=40, observe_days=16, wound=6.0)
    sp = propagation_speed(res)
    assert np.isnan(sp["speed_hops_per_step"]) or sp["speed_hops_per_step"] > 0
    mat, dists = defense_by_distance(res)
    assert mat.shape[0] == 16 and len(dists) >= 2
    d = directionality(res)
    assert d["shoot_peak_defense"] >= 0.0 and d["root_peak_defense"] >= 0.0


def test_calibrate_speed_maps_to_mm_per_s():
    from phytonet.signal_analysis import calibrate_speed
    cal = calibrate_speed(0.26, internode_mm=10.0, step_seconds=5.0)
    assert abs(cal["mm_per_s"] - 0.52) < 1e-6        # 0.26*10/5
    assert cal["in_range"] and cal["realistic_range_mm_per_s"] == (0.1, 40.0)
