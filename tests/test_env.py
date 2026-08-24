"""Tests for the Gymnasium wrapper.

The headline test runs gymnasium's own ``check_env``, which verifies the env
obeys the Gym API (space shapes/dtypes, reset/step signatures, etc.).
"""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from phytonet.config import SimConfig
from phytonet.env import PhytoNetEnv


def test_passes_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env

    env = PhytoNetEnv(SimConfig(days=200), max_days=30)
    check_env(env, skip_render_check=True)
    env.close()


def test_reset_returns_obs_in_space():
    env = PhytoNetEnv(max_days=20)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["day"] == 0


def test_episode_runs_to_truncation():
    env = PhytoNetEnv(max_days=15)
    env.reset(seed=1)
    steps = 0
    truncated = False
    while not truncated:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
    assert steps == 15
    assert info["day"] == 15


def test_action_allocates_toward_shoots():
    """An action favouring index 0 should give shoots the largest budget share."""
    env = PhytoNetEnv(max_days=10)
    env.reset(seed=0)
    env.step(np.array([1.0, -1.0, -1.0], dtype=np.float32))
    p = env.sim.policy.p
    assert p.shoot_alloc > p.root_alloc and p.shoot_alloc > p.thicken_alloc


def test_action_allocates_toward_roots():
    env = PhytoNetEnv(max_days=10)
    env.reset(seed=0)
    env.step(np.array([-1.0, 1.0, -1.0], dtype=np.float32))
    p = env.sim.policy.p
    assert p.root_alloc > p.shoot_alloc and p.root_alloc > p.thicken_alloc


def test_reward_tracks_biomass_growth():
    env = PhytoNetEnv(SimConfig(), max_days=30, randomize=False)
    env.reset(seed=0)
    total = 0.0
    for _ in range(30):
        _, reward, _, truncated, _ = env.step(np.array([1.0, 0.5, 0.2], dtype=np.float32))
        total += reward
        if truncated:
            break
    # Net reward over a season should be clearly positive (the plant grew).
    assert total > 0.0


def test_randomized_episodes_vary_scenario():
    env = PhytoNetEnv(max_days=10, randomize=True)
    names = set()
    for s in range(12):
        _, info = env.reset(seed=s)
        names.add(info["scenario"])
    # Over several resets we should see more than one scenario.
    assert len(names) > 1
