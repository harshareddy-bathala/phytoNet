"""A tiny, dependency-free learner so you can *watch the plant learn*.

This is NOT the deep-RL step (that's PPO + a GNN policy, which needs torch). It
is a deliberately simple **Cross-Entropy Method (CEM)** that optimises a small
*linear* policy mapping the env observation to the 3 growth actions. CEM is a
black-box optimiser: sample policies from a Gaussian, keep the best-performing
"elite" few, refit the Gaussian to them, repeat. Reward goes up over iterations
-- genuine learning, just lightweight enough to run on a laptop in seconds.

The policy here is linear (``action = tanh(W @ obs + b)``). It is exactly the
kind of thing the GNN replaces later: same env, same reward, smarter policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from phytonet.env import PhytoNetEnv


@dataclass
class LinearPolicy:
    """action = tanh(W @ obs + b), flattened params for CEM to optimise."""

    obs_dim: int
    act_dim: int
    theta: np.ndarray | None = None

    def __post_init__(self):
        if self.theta is None:
            self.theta = np.zeros(self.n_params)

    @property
    def n_params(self) -> int:
        return self.act_dim * self.obs_dim + self.act_dim

    def act(self, obs: np.ndarray) -> np.ndarray:
        W = self.theta[: self.act_dim * self.obs_dim].reshape(self.act_dim, self.obs_dim)
        b = self.theta[self.act_dim * self.obs_dim :]
        return np.tanh(W @ obs + b)


def rollout(env: PhytoNetEnv, policy: LinearPolicy, seed: int) -> float:
    """Run one episode, return total reward (final biomass, essentially)."""
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
        total += reward
        done = terminated or truncated
    return total


def evaluate(env: PhytoNetEnv, policy: LinearPolicy, episodes: int, base_seed: int) -> float:
    """Mean reward over several episodes (several conditions, if randomised)."""
    return float(np.mean([rollout(env, policy, base_seed + e) for e in range(episodes)]))


@dataclass
class CEMResult:
    history: list[float] = field(default_factory=list)  # best mean-reward per iter
    best_theta: np.ndarray | None = None
    best_score: float = -np.inf


def cem_optimize(score_fn, x0: np.ndarray, iterations: int = 12, population: int = 20,
                 elite_frac: float = 0.25, init_std: float = 0.4, seed: int = 0,
                 lo: float | None = None, hi: float | None = None) -> CEMResult:
    """Generic Cross-Entropy Method over a parameter vector.

    ``score_fn(vector) -> float`` (higher is better). Samples a population from a
    Gaussian, keeps the best ``elite_frac``, refits the Gaussian to them, repeats.
    Records the score of the running mean each iteration (clipped to [lo, hi] if
    given). Used to *evolve* the shared signaling rule -- no gradients, no network.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float)
    mean = x0.copy()
    std = np.full(x0.shape, init_std)
    n_elite = max(2, int(population * elite_frac))

    def clamp(v):
        return np.clip(v, lo, hi) if lo is not None or hi is not None else v

    result = CEMResult()
    for _ in range(iterations):
        samples = clamp(rng.normal(mean, std, size=(population, x0.shape[0])))
        scores = np.array([score_fn(s) for s in samples])
        elite = samples[np.argsort(scores)[-n_elite:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-3
        mscore = float(score_fn(clamp(mean)))
        result.history.append(mscore)
        if mscore > result.best_score:
            result.best_score = mscore
            result.best_theta = clamp(mean).copy()
    return result


def train_cem(
    env: PhytoNetEnv,
    iterations: int = 20,
    population: int = 24,
    elite_frac: float = 0.25,
    eval_episodes: int = 3,
    init_std: float = 0.5,
    seed: int = 0,
) -> CEMResult:
    """Optimise a LinearPolicy on ``env`` with the Cross-Entropy Method."""
    rng = np.random.default_rng(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    probe = LinearPolicy(obs_dim, act_dim)
    mean = np.zeros(probe.n_params)
    std = np.full(probe.n_params, init_std)
    n_elite = max(2, int(population * elite_frac))

    result = CEMResult()
    for it in range(iterations):
        samples = rng.normal(mean, std, size=(population, probe.n_params))
        # Fair selection: every candidate this iteration faces the SAME episodes
        # (same seed set). Vary the set across iterations so we don't overfit one.
        train_seed = 7 * it
        scores = np.array([
            evaluate(env, LinearPolicy(obs_dim, act_dim, theta), eval_episodes, train_seed)
            for theta in samples
        ])
        elite = samples[np.argsort(scores)[-n_elite:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-3

        # Learning curve: the current mean policy on a FIXED eval set, so the
        # numbers are comparable across iterations (and to the baseline).
        score = evaluate(env, LinearPolicy(obs_dim, act_dim, mean.copy()),
                         eval_episodes, base_seed=0)
        result.history.append(score)
        if score > result.best_score:
            result.best_score = score
            result.best_theta = mean.copy()
    return result
