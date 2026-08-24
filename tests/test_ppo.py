"""Tests for the PPO implementation.

Skipped entirely if torch isn't installed (it's an optional `rl` extra). Kept
fast: tiny networks and short training runs. We check the machinery and that a
short run produces finite returns and a policy at least competitive with random.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phytonet.config import SimConfig
from phytonet.env import PhytoNetEnv
from phytonet.ppo import Agent, PPOConfig, evaluate_agent, train_ppo


def _fixed_env(days=30):
    return PhytoNetEnv(SimConfig(), max_days=days, randomize=False)


def test_agent_forward_shapes():
    agent = Agent(obs_dim=9, act_dim=3, hidden=16)
    x = torch.zeros((5, 9))
    action, logprob, entropy, value = agent.get_action_and_value(x)
    assert action.shape == (5, 3)
    assert logprob.shape == (5,)
    assert entropy.shape == (5,)
    assert value.shape == (5,)
    assert agent.get_value(x).shape == (5,)


def test_act_greedy_shape_and_range():
    agent = Agent(9, 3, hidden=16)
    a = agent.act_greedy(np.zeros(9, dtype=np.float32))
    assert a.shape == (3,)
    assert np.all(np.isfinite(a))


def test_train_ppo_runs_and_logs_history():
    env = _fixed_env(days=30)
    cfg = PPOConfig(total_timesteps=1200, n_steps=300, n_epochs=2,
                    n_minibatches=2, hidden=16, seed=0)
    agent, history = train_ppo(env, cfg)
    assert len(history["timestep"]) == cfg.total_timesteps // cfg.n_steps
    # Episodes (30 steps) finish within each 300-step rollout, so returns logged.
    assert all(np.isfinite(r) for r in history["ep_return"])


def test_ppo_is_not_worse_than_random_after_training():
    """A modest run should leave PPO at least on par with random actions.

    Lenient (deep RL is noisy on tiny budgets); we just guard against a broken
    update that makes the policy clearly worse."""
    env = _fixed_env(days=40)
    cfg = PPOConfig(total_timesteps=8000, n_steps=800, n_epochs=4,
                    n_minibatches=4, hidden=32, seed=0)
    agent, _ = train_ppo(env, cfg)

    trained = evaluate_agent(_fixed_env(40), agent, episodes=4, base_seed=123)

    rng = np.random.default_rng(0)
    renv = _fixed_env(40)
    totals = []
    for e in range(4):
        renv.reset(seed=123 + e)
        total, done = 0.0, False
        while not done:
            _, r, term, trunc, _ = renv.step(renv.action_space.sample())
            total += r
            done = term or trunc
        totals.append(total)
    random_score = float(np.mean(totals))

    assert trained >= random_score - 2.0
