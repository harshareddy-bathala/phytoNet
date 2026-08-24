"""Step 3 tests: thickening sink and the config builder.

These pin down the behaviour that stops resources accumulating unbounded.
"""

from phytonet import NodeKind, World, make_seedling
from phytonet.config import SimConfig, build_simulation
from phytonet.policy import GrowthParams, GrowToLightPolicy


def test_thicken_increases_size_and_spends_resources():
    world = World(dim=2)
    plant = make_seedling(world)
    node = next(iter(plant.nodes.values()))
    node.state["sugar"] = 100.0
    node.state["water"] = 100.0
    size_before = node.state["size"]

    policy = GrowToLightPolicy(GrowthParams())
    policy._thicken(plant)

    assert node.state["size"] > size_before
    assert node.state["sugar"] < 100.0
    assert node.state["water"] < 100.0


def test_thicken_keeps_a_buffer_and_respects_max_size():
    world = World(dim=2)
    plant = make_seedling(world)
    p = GrowthParams(thicken_buffer=2.0, thicken_cost=0.5, max_size=2.0)
    policy = GrowToLightPolicy(p)

    # A node with resources at the buffer must not thicken (nothing to spare).
    node = next(iter(plant.nodes.values()))
    node.state["sugar"] = p.thicken_buffer
    node.state["water"] = p.thicken_buffer
    policy._thicken(plant)
    assert node.state["size"] == 1.0  # unchanged

    # A well-fed node thickens but never past max_size.
    node.state["sugar"] = 1000.0
    node.state["water"] = 1000.0
    for _ in range(100):
        policy._thicken(plant)
    assert node.state["size"] == p.max_size


def test_feedback_inhibition_keeps_per_tissue_load_bounded():
    """The headline Step-3 fix: no tissue accumulates resources without limit.

    Feedback inhibition stops a leaf/root producing once its load nears the cap,
    so over a long season every node stays at or below that cap (small slack for
    resources arriving by diffusion)."""
    cfg = SimConfig(days=120, record_frames=False)
    sim = build_simulation(cfg)
    sim.run(120)
    cap = cfg.physiology.sugar_cap  # sugar_cap == water_cap by default
    slack = 1.5
    for node in sim.plant.nodes.values():
        assert node.state["sugar"] <= cap + slack
        assert node.state["water"] <= cap + slack


def test_resources_track_biomass_not_runaway():
    """Pooled resources should scale with plant size, not explode while biomass
    flatlines (the pre-Step-3 pathology). We check the sugar-per-biomass ratio
    stays modest from mid-season to end."""
    sim = build_simulation(SimConfig(days=80, record_frames=False))
    sim.run(80)
    h = sim.history
    ratio_mid = h["total_sugar"][40] / h["biomass"][40]
    ratio_end = h["total_sugar"][-1] / h["biomass"][-1]
    assert ratio_end < 2 * ratio_mid  # bounded, not diverging


def test_thickening_increases_biomass_vs_no_sink():
    with_sink = build_simulation(SimConfig(days=60, record_frames=False))
    with_sink.run(60)
    no_sink = build_simulation(
        SimConfig(
            days=60,
            record_frames=False,
            growth=GrowthParams(thicken_cost=1e9, thicken_buffer=1e9),
        )
    )
    no_sink.run(60)
    assert with_sink.history["biomass"][-1] >= no_sink.history["biomass"][-1]


def test_build_simulation_applies_config():
    cfg = SimConfig(dim=2, light_intensity=0.3, water_availability=0.4, seed=7)
    sim = build_simulation(cfg)
    assert sim.world.light_intensity == 0.3
    assert sim.world.water_availability == 0.4
    assert len(sim.plant.nodes_of(NodeKind.LEAF)) == 1  # fresh seedling
