"""Proper deep RL: a compact, readable PPO for the PhytoNet env.

This is written in the spirit of CleanRL -- a single, inspectable file rather
than a framework -- because the project has to become a paper and every line
should be auditable. It is continuous-action PPO with a Gaussian policy, GAE
advantages, and the clipped surrogate objective.

Pieces, in plain terms:

* **Agent** -- an actor-critic MLP. The actor outputs a Gaussian *mean* per
  action dim plus a learned (state-independent) log-std; the critic estimates
  the value of a state. Actions are sampled from that Gaussian and the env
  clips them to its [-1, 1] box (so no tanh-squashing bookkeeping needed).
* **Rollout** -- run the (single) env for ``n_steps``, storing transitions.
* **GAE** -- turn rewards + value estimates into advantages and returns.
* **Update** -- several epochs of minibatch SGD on the clipped policy loss +
  value loss - entropy bonus.

It's a single environment (no vectorisation) for clarity; our env is tiny so
this trains in well under a minute. Vectorising later is a drop-in speedup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
except ModuleNotFoundError as exc:  # pragma: no cover - helpful message
    raise ModuleNotFoundError(
        "PhytoNet's PPO needs PyTorch. Install the CPU build with:\n"
        '  pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu'
    ) from exc


@dataclass
class PPOConfig:
    total_timesteps: int = 60_000
    n_steps: int = 1024          # transitions collected per policy update
    n_epochs: int = 10           # SGD passes over each rollout
    n_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    hidden: int = 64
    norm_adv: bool = True
    seed: int = 0


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0):
    """Orthogonal init -- a standard PPO detail that stabilises early training."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """Actor-critic with separate MLP heads sharing nothing (simple + robust)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        # State-independent log-std (a learned exploration scale per action dim).
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, self.critic(x).squeeze(-1)

    @torch.no_grad()
    def act_greedy(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action (the distribution mean) -- for evaluation."""
        x = torch.as_tensor(obs, dtype=torch.float32)
        return self.actor_mean(x).numpy()


def evaluate_agent(env, agent: Agent, episodes: int = 5, base_seed: int = 0) -> float:
    """Mean total reward of the greedy policy over several episodes."""
    totals = []
    for e in range(episodes):
        obs, _ = env.reset(seed=base_seed + e)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act_greedy(obs))
            total += reward
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


def train_ppo(env, config: PPOConfig | None = None):
    """Train a PPO ``Agent`` on ``env``. Returns (agent, history).

    ``history`` has ``timestep`` and ``ep_return`` (mean episodic return of the
    episodes that finished within each rollout) for plotting a learning curve.
    """
    cfg = config or PPOConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = Agent(obs_dim, act_dim, cfg.hidden)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    # Rollout storage.
    obs_buf = torch.zeros((cfg.n_steps, obs_dim))
    act_buf = torch.zeros((cfg.n_steps, act_dim))
    logp_buf = torch.zeros(cfg.n_steps)
    rew_buf = torch.zeros(cfg.n_steps)
    done_buf = torch.zeros(cfg.n_steps)
    val_buf = torch.zeros(cfg.n_steps)

    history = {"timestep": [], "ep_return": []}
    minibatch_size = cfg.n_steps // cfg.n_minibatches

    next_obs_np, _ = env.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32)
    next_done = 0.0
    ep_return = 0.0
    global_step = 0
    num_iterations = max(1, cfg.total_timesteps // cfg.n_steps)

    for _ in range(num_iterations):
        finished_returns: list[float] = []
        # -- collect a rollout ------------------------------------------------
        for t in range(cfg.n_steps):
            global_step += 1
            obs_buf[t] = next_obs
            done_buf[t] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            act_buf[t] = action
            logp_buf[t] = logprob
            val_buf[t] = value

            next_obs_np, reward, terminated, truncated, _ = env.step(action.numpy())
            done = bool(terminated or truncated)
            rew_buf[t] = float(reward)
            ep_return += float(reward)

            if done:
                finished_returns.append(ep_return)
                ep_return = 0.0
                next_obs_np, _ = env.reset()  # randomised scenarios keep varying
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32)
            next_done = 1.0 if done else 0.0

        # -- GAE advantages and returns --------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros(cfg.n_steps)
            last_gae = 0.0
            for t in reversed(range(cfg.n_steps)):
                if t == cfg.n_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_val = next_value
                else:
                    next_nonterminal = 1.0 - done_buf[t + 1]
                    next_val = val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * next_val * next_nonterminal - val_buf[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + val_buf

        # -- PPO update (minibatch SGD over several epochs) ------------------
        idx = np.arange(cfg.n_steps)
        for _ in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, cfg.n_steps, minibatch_size):
                mb = idx[start : start + minibatch_size]
                _, new_logp, entropy, new_val = agent.get_action_and_value(
                    obs_buf[mb], act_buf[mb]
                )
                ratio = torch.exp(new_logp - logp_buf[mb])

                mb_adv = advantages[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped policy loss.
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_loss = 0.5 * ((new_val - returns[mb]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        mean_ret = float(np.mean(finished_returns)) if finished_returns else float("nan")
        history["timestep"].append(global_step)
        history["ep_return"].append(mean_ret)

    return agent, history
