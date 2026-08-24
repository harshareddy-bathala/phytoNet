"""Benchmark harness: evaluate controllers across scenarios with seed variance.

Turns single-seed anecdotes into a mean ± std table. The cheap controllers
(random / fixed allocations) need no training, so this runs instantly and
quantifies the **headroom** in each scenario -- how much the choice of strategy
actually changes the outcome. (A scenario where every controller ties has no
learnable signal; one with a wide spread does.)

Trained controllers (PPO / GNN) can be dropped in via ``eval_policy`` with a
closure over a trained agent, but multi-seed *training* is slow until the
graph-batching speedup lands, so this module focuses on the fixed baselines.
"""

from __future__ import annotations

import numpy as np

from phytonet.config import build_simulation
from phytonet.env import PhytoNetEnv
from phytonet.scenarios import make_scenario


# -- a common outcome metric, so every paradigm is comparable --------------
def outcome(sim, repro_weight: float = 3.0) -> float:
    """Lifetime outcome of a finished simulation: final biomass + reproduction.
    Outcome-based (not tied to any controller's reward shaping), so MLP, GNN,
    bioelectric and fixed policies are measured on exactly the same yardstick."""
    return float(sim.history["biomass"][-1] + repro_weight * sim.seeds)


def eval_global_outcome(action_fn, scenario, seeds, days: int = 80) -> np.ndarray:
    """Outcome of a global-action controller (``action_fn(obs) -> action``)."""
    out = []
    for s in seeds:
        env = PhytoNetEnv(make_scenario(scenario), max_days=days, randomize=False)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            obs, _, te, tr, _ = env.step(action_fn(obs))
            done = te or tr
        out.append(outcome(env.sim))
    return np.array(out)


def eval_gnn_outcome(agent, scenario, seeds, days: int = 80) -> np.ndarray:
    """Outcome of the per-tissue GNN controller (greedy)."""
    out = []
    for s in seeds:
        env = PhytoNetEnv(make_scenario(scenario), max_days=days,
                          randomize=False, per_tissue=True)
        env.reset(seed=s)
        feats, edge = env.graph_observation()
        done = False
        while not done:
            _, te, tr, _ = env.step_local(agent.act_greedy(feats, edge))
            feats, edge = env.graph_observation()
            done = te or tr
        out.append(outcome(env.sim))
    return np.array(out)


def eval_signaling_outcome(params, scenario, seeds, days: int = 80) -> np.ndarray:
    """Outcome of the decentralized electrical-signaling controller."""
    from phytonet.signaling import attach_signaling
    out = []
    for s in seeds:
        sim = build_simulation(make_scenario(scenario, days=days, seed=s,
                                             record_frames=False))
        attach_signaling(sim, params)
        sim.run(days)
        out.append(outcome(sim))
    return np.array(out)


def eval_learned_outcome(params, scenario, seeds, days: int = 80) -> np.ndarray:
    """Outcome of the Phase-2 learned-signal-content controller."""
    from phytonet.learned_signaling import attach_learned_signaling
    out = []
    for s in seeds:
        sim = build_simulation(make_scenario(scenario, days=days, seed=s,
                                             record_frames=False))
        attach_learned_signaling(sim, params)
        sim.run(days)
        out.append(outcome(sim))
    return np.array(out)


def eval_policy(scenario: str, action_fn, seeds, days: int = 80,
                per_tissue: bool = False) -> np.ndarray:
    """Episode returns of ``action_fn`` on ``scenario`` over ``seeds``.

    For global controllers ``action_fn(env) -> action``; for per-tissue ones
    ``action_fn(env, feats, edge) -> node_actions`` with ``per_tissue=True``.
    """
    env = PhytoNetEnv(make_scenario(scenario), max_days=days,
                      randomize=False, per_tissue=per_tissue)
    returns = []
    for s in seeds:
        env.reset(seed=s)
        total, done = 0.0, False
        if per_tissue:
            feats, edge = env.graph_observation()
            while not done:
                reward, te, tr, _ = env.step_local(action_fn(env, feats, edge))
                feats, edge = env.graph_observation()
                total += reward
                done = te or tr
        else:
            while not done:
                _, reward, te, tr, _ = env.step(action_fn(env))
                total += reward
                done = te or tr
        returns.append(total)
    return np.array(returns, dtype=float)


# Fixed global-allocation controllers (no training needed).
FIXED_CONTROLLERS = {
    "random": lambda env: env.action_space.sample(),
    "balanced": lambda env: np.zeros(3, dtype=np.float32),
    "shoot-heavy": lambda env: np.array([1, -1, -1], dtype=np.float32),
    "root-heavy": lambda env: np.array([-1, 1, -1], dtype=np.float32),
}


def benchmark_table(scenarios, seeds, controllers=None, days: int = 80):
    """Return ``{controller: {scenario: (mean, std)}}`` over ``seeds``."""
    controllers = controllers or FIXED_CONTROLLERS
    table = {}
    for name, fn in controllers.items():
        table[name] = {}
        for sc in scenarios:
            r = eval_policy(sc, fn, seeds, days=days)
            table[name][sc] = (float(r.mean()), float(r.std()))
    return table


def format_table(table, scenarios) -> str:
    """Pretty-print a benchmark table as mean±std."""
    width = max(len(n) for n in table) + 2
    header = " " * width + "".join(f"{sc:>16s}" for sc in scenarios)
    lines = [header]
    for name, row in table.items():
        cells = "".join(f"{row[sc][0]:7.1f}+/-{row[sc][1]:<4.1f} " for sc in scenarios)
        lines.append(f"{name:<{width}s}{cells}")
    return "\n".join(lines)
