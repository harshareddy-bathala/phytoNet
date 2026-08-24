"""Tests for physiology and the season loop.

The conservation test on transport is the important one: diffusion must only
*move* resources between tissues, never create or destroy them.
"""

import numpy as np

from phytonet import NodeKind, World, make_seedling
from phytonet.physiology import (
    PhysiologyParams,
    maintain,
    photosynthesize,
    transport,
    uptake,
)
from phytonet.simulation import Simulation


def test_photosynthesis_adds_sugar_to_leaves_above_ground():
    world = World(dim=2)
    plant = make_seedling(world)
    (leaf,) = plant.nodes_of(NodeKind.LEAF)
    before = leaf.state["sugar"]
    photosynthesize(plant, world, PhysiologyParams())
    assert leaf.state["sugar"] > before


def test_uptake_adds_water_to_roots_below_ground():
    world = World(dim=2)
    plant = make_seedling(world)
    roots = plant.nodes_of(NodeKind.ROOT)
    before = [r.state["water"] for r in roots]
    uptake(plant, world, PhysiologyParams())
    assert all(r.state["water"] > b for r, b in zip(roots, before))


def test_transport_conserves_total_sugar_and_water():
    world = World(dim=2)
    plant = make_seedling(world)
    # Seed some uneven resources so there is something to move.
    for i, node in enumerate(plant.nodes.values()):
        node.state["sugar"] = float(i)
        node.state["water"] = float(2 * i)
    total_sugar = sum(n.state["sugar"] for n in plant.nodes.values())
    total_water = sum(n.state["water"] for n in plant.nodes.values())

    for _ in range(10):
        transport(plant, PhysiologyParams())

    assert np.isclose(sum(n.state["sugar"] for n in plant.nodes.values()), total_sugar)
    assert np.isclose(sum(n.state["water"] for n in plant.nodes.values()), total_water)


def test_transport_moves_resources_toward_equalization():
    world = World(dim=2)
    plant = make_seedling(world)
    ids = sorted(plant.nodes)
    plant.nodes[ids[0]].state["sugar"] = 10.0
    plant.nodes[ids[-1]].state["sugar"] = 0.0
    gap_before = 10.0
    for _ in range(50):
        transport(plant, PhysiologyParams())
    sugars = [plant.nodes[i].state["sugar"] for i in ids]
    gap_after = max(sugars) - min(sugars)
    assert gap_after < gap_before


def test_maintenance_never_drives_sugar_negative():
    world = World(dim=2)
    plant = make_seedling(world)
    maintain(plant, PhysiologyParams(maintenance_rate=100.0))
    assert all(n.state["sugar"] >= 0.0 for n in plant.nodes.values())


def test_season_run_grows_the_plant():
    world = World(dim=2)
    plant = make_seedling(world)
    sim = Simulation(world, plant, record_frames=False)
    n_before = len(plant.nodes)
    sim.run(40)
    assert len(plant.nodes) > n_before
    # History has one entry per day, including the initial day 0 snapshot.
    assert len(sim.history["day"]) == 41
    assert sim.history["biomass"][-1] > sim.history["biomass"][0]


def test_simulation_records_frames_when_enabled():
    world = World(dim=2)
    plant = make_seedling(world)
    sim = Simulation(world, plant, record_frames=True)
    sim.run(5)
    assert len(sim.frames) == 6  # day 0 + 5 steps
    # Frames are independent snapshots, not aliases of the live plant.
    assert len(sim.frames[0].nodes) <= len(sim.frames[-1].nodes)
