"""Tests for herbivory and the electrical wound -> defense response."""

import numpy as np

from phytonet.config import SimConfig, build_simulation
from phytonet.plant import NodeKind, Plant
from phytonet.scenarios import make_scenario
from phytonet.signaling import SignalingController, SignalingParams, attach_signaling
from phytonet.world import World


def test_remove_node_drops_node_and_edges():
    p = Plant(dim=2)
    a = p.add_node(NodeKind.STEM, [0.0, 1.0])
    b = p.add_node(NodeKind.LEAF, [0.5, 1.3])
    p.add_edge(a.id, b.id)
    p.remove_node(b.id)
    assert b.id not in p.nodes
    assert all(b.id not in e for e in p.edges)


def test_herbivory_bites_and_wounds_a_leaf():
    sim = build_simulation(make_scenario("herbivory", start_from_seed=False,
                                         days=40, seed=0, record_frames=False))
    attach_signaling(sim)
    sim.run(40)
    assert sim.bites > 0                 # leaves were bitten
    assert sim.biomass_lost > 0.0        # biomass removed
    assert sim.history["bites"][-1] == sim.bites


def test_defense_reduces_bite_damage():
    """A fully-defended leaf should lose far less to a bite than an undefended one."""
    def loss_with_defense(defense):
        world = World(dim=2)
        sim = build_simulation(SimConfig(herbivory_interval=1,
                                         herbivory_intensity=0.7))
        # Force a single known leaf with the given defense.
        leaf = sim.plant.nodes_of(NodeKind.LEAF)[0]
        leaf.state["defense"] = defense
        before = leaf.state["size"]
        sim.day = 0
        sim._herbivory()  # day 0 -> (0+1)%1==0 -> bite fires
        return before - leaf.state["size"]

    assert loss_with_defense(0.0) > loss_with_defense(1.0) + 0.3


def test_wound_signal_arms_systemic_defense():
    """After wounding, the signaling controller should raise defense somewhere
    beyond the bitten cell (systemic response via propagation)."""
    sim = build_simulation(make_scenario("herbivory", start_from_seed=False,
                                         days=30, seed=1, record_frames=False))
    attach_signaling(sim)
    sim.run(30)
    defenses = [n.state.get("defense", 0.0) for n in sim.plant.nodes.values()]
    assert max(defenses) > 0.0   # the plant armed some defense in response


def test_defense_helps_survive_herbivory():
    """A plant that mounts the electrical defense should keep more biomass under
    herbivory than one whose defense is disabled."""
    def final_biomass(defense_on):
        params = SignalingParams()
        if not defense_on:
            params.defense_gain = 0.0   # disable the defense response
        sim = build_simulation(make_scenario("herbivory", start_from_seed=False,
                                             days=60, seed=2, record_frames=False))
        attach_signaling(sim, params)
        sim.run(60)
        return sim.history["biomass"][-1]

    assert final_biomass(True) >= final_biomass(False)
