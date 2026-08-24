"""Tests for plant-plant competition (shared light + water)."""

import numpy as np

from phytonet.competition import Arena, make_arena, make_controller
from phytonet.plant import NodeKind, Plant
from phytonet.world import World


def test_make_arena_spaces_plants_apart():
    arena = make_arena([("fixed", "a"), ("fixed", "b")], spacing=2.5)
    xa = np.mean([n.pos[0] for n in arena.entries[0]["plant"].nodes.values()])
    xb = np.mean([n.pos[0] for n in arena.entries[1]["plant"].nodes.values()])
    assert abs(xa - xb) > 1.5


def test_cross_plant_shading_reduces_sugar():
    """A leaf shaded by a taller leaf nearby gains less sugar than the taller one."""
    w = World(dim=2, light_intensity=1.0)

    def leaf_plant(x, h, size):
        p = Plant(dim=2)
        n = p.add_node(NodeKind.LEAF, [x, h])
        n.state["size"] = size
        return p

    low, high = leaf_plant(0.0, 4.0, 1.0), leaf_plant(0.0, 8.0, 2.0)
    arena = Arena(w, [{"plant": low, "controller": None, "label": "low"},
                      {"plant": high, "controller": None, "label": "high"}],
                  shade_k=0.6, rain_interval=0, evaporation=0.0)
    arena._photosynthesis()
    s_low = next(iter(low.nodes.values())).state["sugar"]
    s_high = next(iter(high.nodes.values())).state["sugar"]
    assert s_low < s_high            # the shaded (lower) leaf made less sugar


def test_competition_reduces_biomass():
    solo = make_arena([("bioelectric", "x")], spacing=1.4)
    solo.run(60)
    duo = make_arena([("bioelectric", "x"), ("fixed", "y")], spacing=1.4)
    duo.run(60)
    assert duo.biomass()["x"] <= solo.biomass()["x"] + 1e-6


def test_symmetric_controllers_nearly_tie():
    """Identical controllers should finish within a growth event of each other --
    outcomes are strategy-determined, not dominated by a first-mover/order effect
    (fair shared uptake). They aren't bit-identical because the two share an RNG
    seed at mirrored positions, so their wobble isn't perfectly mirror-symmetric."""
    arena = make_arena([("fixed", "a"), ("fixed", "b")], spacing=1.4)
    arena.run(60)
    b = arena.biomass()
    assert abs(b["a"] - b["b"]) < 1.5


def test_arena_runs_and_logs_history():
    arena = make_arena([("bioelectric", "a"), ("learned", "b")], spacing=1.4)
    arena.run(20)
    assert len(arena.history["a"]) == 20 and len(arena.history["b"]) == 20
    assert all(np.isfinite(v) for v in arena.biomass().values())


# -- competitive co-evolution ---------------------------------------------
def test_competitive_fitness_and_coevolution_run():
    from phytonet.competition import competitive_fitness, coevolve_competitive
    from phytonet.signaling import SignalingParams
    f = competitive_fitness(SignalingParams().to_vector(), days=30, seeds=range(1))
    assert np.isfinite(f)
    best, res = coevolve_competitive(iterations=3, population=8, days=30)
    assert len(res.history) == 3 and best.to_vector().shape == (15,)


def test_selfplay_coevolution_runs():
    from phytonet.competition import coevolve_selfplay
    best, history = coevolve_selfplay(iterations=3, population=8, days=30)
    assert len(history) == 3 and best.to_vector().shape == (15,)
