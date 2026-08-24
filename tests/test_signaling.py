"""Tests for the decentralized electrical-signaling controller."""

import numpy as np

from phytonet.config import SimConfig, build_simulation
from phytonet.plant import NodeKind, Plant
from phytonet.signaling import SignalingController, SignalingParams, attach_signaling
from phytonet.world import World


def _two_node_plant():
    p = Plant(dim=2)
    a = p.add_node(NodeKind.STEM, [0.0, 1.0])
    b = p.add_node(NodeKind.STEM, [0.0, 2.0])
    p.add_edge(a.id, b.id)
    return p, a, b


def test_potential_propagates_along_edges():
    """A depolarised cell raises its neighbour's potential (electrotonic coupling)."""
    world = World(dim=2)
    plant, a, b = _two_node_plant()
    a.state["V"], b.state["V"] = 5.0, 0.0
    # Isolate coupling: no leak / stimulus, no firing.
    params = SignalingParams(leak=0.0, k_light=0.0, k_water=0.0,
                             coupling=0.3, threshold=100.0)
    ctrl = SignalingController(policy=None, params=params)
    ctrl._electrical_step(plant, world)
    assert b.state["V"] > 0.0           # neighbour got excited
    assert a.state["V"] < 5.0           # source shared some charge


def test_action_potential_fires_and_resets():
    world = World(dim=2)
    plant, a, b = _two_node_plant()
    a.state["V"], b.state["V"] = 3.0, 0.0
    params = SignalingParams(leak=0.0, k_light=0.0, k_water=0.0, coupling=0.0,
                             threshold=2.5, ap_strength=1.0)
    ctrl = SignalingController(policy=None, params=params)
    fired = ctrl._electrical_step(plant, world)
    assert a.id in fired
    assert np.isclose(a.state["V"], params.v_rest)        # reset after firing
    assert np.isclose(b.state["V"], params.ap_strength)   # spike delivered


def test_params_vector_roundtrip():
    p = SignalingParams(leak=0.3, coupling=0.4, threshold=2.0)
    q = SignalingParams.from_vector(p.to_vector())
    assert np.isclose(q.leak, 0.3) and np.isclose(q.coupling, 0.4)
    assert np.isclose(q.threshold, 2.0)


def test_signaling_grows_plant_from_seed():
    sim = build_simulation(SimConfig(start_from_seed=True, days=50, record_frames=False))
    ctrl = attach_signaling(sim)
    assert sim.grow_fn == ctrl.step      # the electrical controller drives growth
    sim.run(50)
    assert len(sim.plant.nodes) > 5                 # grew a body via signals
    assert sim.plant.nodes_of(NodeKind.LEAF)        # leaves emerged
    # Cells carry an electrical state.
    assert any("V" in n.state for n in sim.plant.nodes.values())


def test_homogeneous_rule_no_central_state():
    """The controller holds only the shared parameters -- no per-node weights,
    no central network state."""
    ctrl = SignalingController(policy=None, params=SignalingParams())
    assert ctrl.p.to_vector().shape == (15,)   # incl. evolved wound-propagation


# -- evolving the rule (CEM) ----------------------------------------------
def test_cem_optimize_finds_a_target():
    """Generic CEM should climb a simple concave score toward its optimum."""
    from phytonet.learn import cem_optimize
    target = np.array([1.0, 2.0, 0.5])
    res = cem_optimize(lambda x: -float(np.sum((x - target) ** 2)),
                       x0=np.zeros(3), iterations=30, population=40,
                       init_std=1.2, seed=0)
    assert np.allclose(res.best_theta, target, atol=0.5)


def test_evaluate_params_is_finite():
    from phytonet.signaling import evaluate_params, SignalingParams
    v = SignalingParams().to_vector()
    score = evaluate_params(v, scenarios=["baseline", "patchy"], seeds=range(1), days=25)
    assert np.isfinite(score) and score > 0


def test_evolution_improves_or_matches_hand_designed():
    from phytonet.signaling import evolve_signaling, evaluate_params, SignalingParams
    scen, seeds, days = ["baseline", "drought"], range(1), 30
    hand = evaluate_params(SignalingParams().to_vector(), scen, seeds, days)
    best, result = evolve_signaling(scenarios=scen, seeds=seeds, days=days,
                                    iterations=4, population=10)
    evolved = evaluate_params(best.to_vector(), scen, seeds, days)
    assert len(result.history) == 4
    assert evolved >= hand - 1.0   # learning shouldn't make it clearly worse


def test_signal_substeps_propagate_wound_further_per_day():
    """More signalling sub-steps per growth day -> the wound signal reaches a
    distant tissue further in a single day (faster signalling timescale)."""
    from phytonet.plant import NodeKind, Plant
    from phytonet.policy import GrowToLightPolicy
    from phytonet.world import World

    def chain():
        p = Plant(dim=2)
        ids = [p.add_node(NodeKind.STEM, [0.0, float(i)]).id for i in range(5)]
        for a, b in zip(ids, ids[1:]):
            p.add_edge(a, b)
        p.nodes[ids[0]].state["wound"] = 5.0
        return p, ids

    w = World(dim=2)
    p1, i1 = chain(); SignalingController(GrowToLightPolicy(seed=0), signal_substeps=1).step(p1, w)
    p4, i4 = chain(); SignalingController(GrowToLightPolicy(seed=0), signal_substeps=4).step(p4, w)
    assert p4.nodes[i4[4]].state.get("wound", 0.0) > p1.nodes[i1[4]].state.get("wound", 0.0)
