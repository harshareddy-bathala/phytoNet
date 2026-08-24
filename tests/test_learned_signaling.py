"""Tests for the Phase-2 learned-signal-content controller."""

import numpy as np

from phytonet.config import SimConfig, build_simulation
from phytonet.learned_signaling import (
    N_CHANNELS, SENSOR_DIM, LearnedSignalingController, LearnedSignalingParams,
    attach_learned_signaling, channel_sensor_correlation, evaluate_learned,
    evolve_learned,
)
from phytonet.plant import NodeKind


def test_param_pack_unpack_shapes():
    p = LearnedSignalingParams.default()
    W_in, W_c, leak, W_out = p.unpack()
    assert W_in.shape == (N_CHANNELS, SENSOR_DIM)
    assert W_c.shape == (N_CHANNELS, N_CHANNELS)
    assert leak.shape == (N_CHANNELS,)
    assert W_out.shape == (4, N_CHANNELS)
    assert np.allclose(LearnedSignalingParams.from_vector(p.to_vector()).theta, p.theta)


def test_default_rule_grows_a_plant():
    sim = build_simulation(SimConfig(start_from_seed=True, days=50, record_frames=False))
    attach_learned_signaling(sim)
    sim.run(50)
    assert len(sim.plant.nodes) > 5
    assert sim.plant.nodes_of(NodeKind.LEAF)
    # Every tissue carries a multi-channel signal.
    assert all(len(n.state.get("sig", [])) == N_CHANNELS
               for n in sim.plant.nodes.values())


def test_evaluate_learned_is_finite():
    v = LearnedSignalingParams.default().to_vector()
    score = evaluate_learned(v, ["baseline", "patchy"], seeds=range(1), days=25)
    assert np.isfinite(score) and score > 0


def test_evolution_runs_and_improves_or_matches():
    scen, seeds, days = ["baseline", "patchy"], range(1), 30
    hand = evaluate_learned(LearnedSignalingParams.default().to_vector(), scen, seeds, days)
    best, result = evolve_learned(scenarios=scen, seeds=seeds, days=days,
                                  iterations=4, population=12)
    evolved = evaluate_learned(best.to_vector(), scen, seeds, days)
    assert len(result.history) == 4
    assert evolved >= hand - 1.0


def test_channel_correlation_is_in_range():
    corr = channel_sensor_correlation(LearnedSignalingParams.default(),
                                      scenario="patchy", seed=0, days=25)
    assert isinstance(corr, dict)
    assert all(-1.0 <= v <= 1.0 for v in corr.values())
