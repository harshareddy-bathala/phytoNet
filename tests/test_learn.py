"""Tests for the numpy-only CEM learner.

Kept fast: tiny populations and short seasons. We check the machinery works and
that learning produces a sensible, positive-growth policy."""

import numpy as np

from phytonet.config import SimConfig
from phytonet.env import PhytoNetEnv
from phytonet.learn import LinearPolicy, evaluate, rollout, train_cem


def _fixed_env(days=20):
    return PhytoNetEnv(SimConfig(), max_days=days, randomize=False)


def test_linear_policy_shapes():
    pol = LinearPolicy(obs_dim=9, act_dim=3)
    assert pol.n_params == 3 * 9 + 3
    action = pol.act(np.zeros(9))
    assert action.shape == (3,)
    assert np.all(np.abs(action) <= 1.0)  # tanh-bounded


def test_rollout_returns_finite_reward():
    env = _fixed_env()
    pol = LinearPolicy(env.observation_space.shape[0], env.action_space.shape[0])
    r = rollout(env, pol, seed=0)
    assert np.isfinite(r)


def test_cem_runs_and_finds_a_growing_policy():
    env = _fixed_env(days=20)
    result = train_cem(env, iterations=3, population=8, eval_episodes=1, seed=0)
    assert len(result.history) == 3
    assert result.best_theta is not None
    assert result.best_score > 0.0  # the learned plant actually grew

    # The best learned policy should be at least as good as a do-nothing policy.
    zero = LinearPolicy(env.observation_space.shape[0], env.action_space.shape[0])
    assert result.best_score >= evaluate(env, zero, episodes=1, base_seed=0) - 1e-6
