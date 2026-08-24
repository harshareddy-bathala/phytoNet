"""Tests for the 3D world, tropism physics, and renderer dispatch."""

import numpy as np

from phytonet import NodeKind, World, make_seedling
from phytonet.config import SimConfig, build_simulation
from phytonet.plant import Node
from phytonet.policy import GrowthParams, GrowToLightPolicy
from phytonet import viz, viz3d
from phytonet.visualize import _backend


def test_world_light_direction_is_unit_and_points_at_sun():
    world = World(dim=3, light_pos=[10.0, 0.0, 5.0])
    d = world.light_direction(np.zeros(3))
    assert np.isclose(np.linalg.norm(d), 1.0)
    assert d[0] > 0  # sun is in the +x direction


def test_phototropism_bends_stem_toward_an_offcentre_sun():
    # No wobble so the direction is deterministic.
    world = World(dim=3, light_pos=[10.0, 10.0, 9.0])
    policy = GrowToLightPolicy(GrowthParams(wobble=0.0, phototropism=0.5, gravitropism=0.7))
    stem = Node(id=0, kind=NodeKind.STEM, pos=np.zeros(3))
    d = policy._direction(stem, world)
    assert d[0] > 0 and d[1] > 0   # leans toward the sun in x and y
    assert d[2] > 0                # still grows upward overall


def test_root_grows_downward_in_3d():
    world = World(dim=3)
    policy = GrowToLightPolicy(GrowthParams(wobble=0.0))
    root = Node(id=0, kind=NodeKind.ROOT, pos=np.zeros(3))
    d = policy._direction(root, world)
    assert d[2] < 0  # down the up-axis


def test_3d_season_grows_and_stays_3d():
    sim = build_simulation(SimConfig(dim=3, days=40, record_frames=False))
    n_before = len(sim.plant.nodes)
    sim.run(40)
    assert len(sim.plant.nodes) > n_before
    assert sim.plant.positions().shape[1] == 3


def test_3d_growth_spreads_laterally():
    """With phototropism + wobble, the plant should not be a straight 1D line --
    some tissue must end up off the central axis in x or y."""
    sim = build_simulation(SimConfig(dim=3, days=60, light_pos=(4.0, 4.0, 9.0),
                                     record_frames=False))
    sim.run(60)
    pos = sim.plant.positions()
    lateral_spread = np.ptp(pos[:, :2], axis=0)  # range in x and y
    assert lateral_spread.max() > 0.5


def test_renderer_dispatch_by_dimension():
    assert _backend(World(dim=2)) is viz
    assert _backend(World(dim=3)) is viz3d
