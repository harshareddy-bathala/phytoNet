"""Structural tests for the plant graph and the seedling factory.

These need no screen, so they run anywhere -- this is exactly why the model is
kept separate from viz.
"""

import numpy as np
import pytest

from phytonet import NodeKind, World, make_seedling
from phytonet.plant import Plant


def test_seedling_structure():
    world = World(dim=2)
    plant = make_seedling(world, n_roots=3)

    # seed + stem + leaf + 3 roots = 6 tissues, and one edge per non-seed-to-seed
    # / stem-leaf link: seed-stem, stem-leaf, seed-root x3 = 5 edges.
    assert len(plant.nodes) == 6
    assert len(plant.edges) == 5
    kinds = sorted(n.kind.value for n in plant.nodes.values())
    assert kinds == ["leaf", "root", "root", "root", "seed", "stem"]


def test_seedling_root_count_is_configurable():
    world = World(dim=3)
    plant = make_seedling(world, n_roots=5)
    assert len(plant.nodes_of(NodeKind.ROOT)) == 5


def test_seedling_growing_tips_are_roots_and_stem():
    world = World(dim=2)
    plant = make_seedling(world)
    tip_kinds = {n.kind for n in plant.meristems()}
    assert tip_kinds == {NodeKind.ROOT, NodeKind.STEM}


def test_roots_below_ground_and_stem_above():
    world = World(dim=2)
    plant = make_seedling(world)
    roots = plant.nodes_of(NodeKind.ROOT)
    (stem,) = plant.nodes_of(NodeKind.STEM)
    assert roots and all(not world.is_above_ground(r.pos) for r in roots)
    assert world.is_above_ground(stem.pos)


def test_positions_have_world_dimension():
    for dim in (2, 3):
        world = World(dim=dim)
        plant = make_seedling(world, n_roots=3)
        assert plant.positions().shape == (6, dim)


def test_light_and_water_fields_respect_ground():
    world = World(dim=2, light_intensity=2.0, water_availability=3.0)
    above = np.array([0.0, 5.0])
    below = np.array([0.0, -2.0])
    assert world.light_at(above) == 2.0
    assert world.light_at(below) == 0.0
    assert world.water_at(below) == 3.0
    assert world.water_at(above) == 0.0


def test_add_edge_rejects_unknown_and_self_links():
    plant = Plant(dim=2)
    a = plant.add_node(NodeKind.STEM, [0.0, 0.0])
    with pytest.raises(ValueError):
        plant.add_edge(a.id, a.id)
    with pytest.raises(KeyError):
        plant.add_edge(a.id, 999)


def test_add_node_rejects_wrong_dimension():
    plant = Plant(dim=2)
    with pytest.raises(ValueError):
        plant.add_node(NodeKind.STEM, [0.0, 0.0, 0.0])
