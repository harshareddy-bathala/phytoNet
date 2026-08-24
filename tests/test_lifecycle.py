"""Tests for spatial heterogeneity and the lifecycle (reproduction + death)."""

import numpy as np

from phytonet.config import SimConfig, build_simulation
from phytonet.env import PhytoNetEnv
from phytonet.scenarios import make_scenario
from phytonet.world import World


# -- spatial heterogeneity -------------------------------------------------
def test_light_gradient_makes_one_side_brighter():
    w = World(dim=2, light_intensity=1.0, light_gradient=0.4)
    assert w.light_at(np.array([2.0, 5.0])) > w.light_at(np.array([-2.0, 5.0]))
    # Zero gradient reproduces the uniform field.
    flat = World(dim=2, light_intensity=1.0)
    assert flat.light_at(np.array([2.0, 5.0])) == flat.light_at(np.array([-2.0, 5.0]))


def test_water_source_concentrates_water():
    w = World(dim=2, water_availability=1.0, water_source_x=-3.0, water_width=2.0)
    near = w.water_at(np.array([-3.0, -1.0]))   # at the source
    far = w.water_at(np.array([3.0, -1.0]))      # far away
    assert near > far


def test_patchy_scenario_runs():
    sim = build_simulation(make_scenario("patchy", days=40, record_frames=False))
    sim.run(40)
    assert sim.history["biomass"][-1] > sim.history["biomass"][0]


# -- lifecycle: reproduction ----------------------------------------------
def test_reproduction_sets_seeds():
    sim = build_simulation(make_scenario("lifecycle", days=50, record_frames=False))
    sim.run(50)
    assert sim.seeds > 0
    assert sim.history["seeds"][-1] == sim.seeds


# -- lifecycle: starvation death ------------------------------------------
def test_plant_starves_to_death_without_light():
    # No light -> no sugar -> starvation; should die within a few days.
    sim = build_simulation(SimConfig(light_intensity=0.0, starve_limit=3,
                                     days=20, record_frames=False))
    sim.run(10)
    assert not sim.alive


def test_env_terminates_on_death_with_penalty():
    cfg = SimConfig(light_intensity=0.0, starve_limit=3)
    env = PhytoNetEnv(cfg, max_days=30, randomize=False)
    env.reset(seed=0)
    terminated = False
    for _ in range(30):
        _, reward, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated  # died before truncation


def test_lifecycle_excluded_from_random_pool():
    from phytonet.scenarios import RANDOM_SCENARIOS
    assert "lifecycle" not in RANDOM_SCENARIOS
    assert "patchy" in RANDOM_SCENARIOS


# -- start from seed / germination ----------------------------------------
def test_make_seedling_ungerminated_is_just_a_seed():
    from phytonet.plant import NodeKind
    w = World(dim=2)
    from phytonet.plant import make_seedling
    p = make_seedling(w, n_roots=3, germinated=False)
    assert len(p.nodes) == 1
    assert p.nodes_of(NodeKind.SEED) and not p.nodes_of(NodeKind.LEAF)


def test_start_from_seed_germinates_then_grows():
    from phytonet.plant import NodeKind
    sim = build_simulation(SimConfig(start_from_seed=True, days=25, record_frames=False))
    # Day 0: a lone seed, no leaf.
    assert len(sim.plant.nodes) == 1
    assert not sim.plant.nodes_of(NodeKind.LEAF)
    sim.step()  # germinates: stem + roots appear (still no leaf yet)
    assert sim.plant.nodes_of(NodeKind.STEM)
    assert sim.plant.nodes_of(NodeKind.ROOT)
    sim.run(25)
    assert len(sim.plant.nodes) > 5            # grew a body
    assert sim.plant.nodes_of(NodeKind.LEAF)   # leaves emerged from stem growth
