"""Tests for the root-origin water-stress signal (root -> shoot signalling)."""

import numpy as np

from phytonet.config import SimConfig, build_simulation
from phytonet.plant import NodeKind
from phytonet.scenarios import make_scenario
from phytonet.signaling import attach_signaling


def test_stress_signal_travels_root_to_shoot_under_drought():
    sim = build_simulation(make_scenario("drought", days=45, seed=0, record_frames=False))
    attach_signaling(sim)
    sim.run(45)
    shoot_stress = [n.state.get("stress", 0.0) for n in sim.plant.nodes.values()
                    if n.kind in (NodeKind.STEM, NodeKind.LEAF)]
    # Stress originates in dry roots but should reach the shoots (systemic).
    assert max(shoot_stress, default=0.0) > 0.0


def test_no_stress_signal_when_well_watered():
    # Infinite reservoir -> roots never water-stressed -> no stress signal.
    sim = build_simulation(SimConfig(days=40, record_frames=False))
    attach_signaling(sim)
    sim.run(40)
    assert all(n.state.get("stress", 0.0) <= 1e-9 for n in sim.plant.nodes.values())
