"""PPO for the GNN graph policy.

Same algorithm as ``ppo.py`` (clipped surrogate + GAE), but the policy is the
``GNNActorCritic`` and the observation is the per-tissue graph from
``env.graph_observation()``. Because the plant grows, each step's graph has a
different number of nodes, so we store graphs in a list and run the GNN per
sample during the update (no fixed-size obs tensor). Graphs are tiny, so this is
plenty fast on CPU.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PhytoNet's GNN-PPO needs PyTorch. Install the CPU build with:\n"
        '  pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu'
    ) from exc

from phytonet.gnn import GNNActorCritic, GNNNodeActorCritic
from phytonet.ppo import PPOConfig


def evaluate_gnn(env, agent: GNNActorCritic, episodes: int = 8, base_seed: int = 0) -> float:
    totals = []
    for e in range(episodes):
        env.reset(seed=base_seed + e)
        feats, edge = env.graph_observation()
        total, done = 0.0, False
        while not done:
            action = agent.act_greedy(feats, edge)
            _, reward, terminated, truncated, _ = env.step(action)
            feats, edge = env.graph_observation()
            total += reward
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


def train_ppo_gnn(env, config: PPOConfig | None = None, hidden: int = 48, rounds: int = 3):
    """Train a ``GNNActorCritic`` on ``env`` with PPO. Returns (agent, history)."""
    cfg = config or PPOConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    act_dim = env.action_space.shape[0]
    env.reset(seed=cfg.seed)
    feats0, _ = env.graph_observation()
    node_dim = feats0.shape[1]
    agent = GNNActorCritic(node_dim, act_dim, hidden=hidden, rounds=rounds)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    # Per-step storage. Graphs are kept as numpy (variable node count).
    feats_buf: list[np.ndarray] = [None] * cfg.n_steps
    edge_buf: list[np.ndarray] = [None] * cfg.n_steps
    act_buf = torch.zeros((cfg.n_steps, act_dim))
    logp_buf = torch.zeros(cfg.n_steps)
    rew_buf = torch.zeros(cfg.n_steps)
    done_buf = torch.zeros(cfg.n_steps)
    val_buf = torch.zeros(cfg.n_steps)

    history = {"timestep": [], "ep_return": []}
    minibatch_size = cfg.n_steps // cfg.n_minibatches

    feats, edge = env.graph_observation()
    next_done = 0.0
    ep_return = 0.0
    global_step = 0
    num_iterations = max(1, cfg.total_timesteps // cfg.n_steps)

    for _ in range(num_iterations):
        finished: list[float] = []
        for t in range(cfg.n_steps):
            global_step += 1
            feats_buf[t] = feats
            edge_buf[t] = edge
            done_buf[t] = next_done

            ft = torch.as_tensor(feats, dtype=torch.float32)
            et = torch.as_tensor(edge, dtype=torch.long)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(ft, et)
            act_buf[t] = action
            logp_buf[t] = logprob
            val_buf[t] = value

            _, reward, terminated, truncated, _ = env.step(action.numpy())
            done = bool(terminated or truncated)
            rew_buf[t] = float(reward)
            ep_return += float(reward)
            if done:
                finished.append(ep_return)
                ep_return = 0.0
                env.reset()
            feats, edge = env.graph_observation()
            next_done = 1.0 if done else 0.0

        # GAE.
        with torch.no_grad():
            next_value = agent.get_value(torch.as_tensor(feats, dtype=torch.float32),
                                         torch.as_tensor(edge, dtype=torch.long))
            advantages = torch.zeros(cfg.n_steps)
            last_gae = 0.0
            for t in reversed(range(cfg.n_steps)):
                if t == cfg.n_steps - 1:
                    nonterminal, next_val = 1.0 - next_done, next_value
                else:
                    nonterminal, next_val = 1.0 - done_buf[t + 1], val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * next_val * nonterminal - val_buf[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + val_buf

        # PPO update -- run the GNN per stored graph (sizes differ).
        idx = np.arange(cfg.n_steps)
        for _ in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, cfg.n_steps, minibatch_size):
                mb = idx[start:start + minibatch_size]
                new_logp, ent, vals = [], [], []
                for t in mb:
                    ft = torch.as_tensor(feats_buf[t], dtype=torch.float32)
                    et = torch.as_tensor(edge_buf[t], dtype=torch.long)
                    _, lp, en, v = agent.get_action_and_value(ft, et, act_buf[t])
                    new_logp.append(lp); ent.append(en); vals.append(v)
                new_logp = torch.stack(new_logp)
                ent = torch.stack(ent)
                vals = torch.stack(vals)

                ratio = torch.exp(new_logp - logp_buf[mb])
                mb_adv = advantages[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((vals - returns[mb]) ** 2).mean()
                loss = pg_loss - cfg.ent_coef * ent.mean() + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        history["timestep"].append(global_step)
        history["ep_return"].append(
            float(np.mean(finished)) if finished else float("nan"))

    return agent, history


def evaluate_gnn_local(env, agent: GNNNodeActorCritic, episodes: int = 8,
                       base_seed: int = 0) -> float:
    """Mean return of the greedy per-tissue policy. ``env`` must be per_tissue."""
    totals = []
    for e in range(episodes):
        env.reset(seed=base_seed + e)
        feats, edge = env.graph_observation()
        total, done = 0.0, False
        while not done:
            actions = agent.act_greedy(feats, edge)         # [N, A]
            reward, terminated, truncated, _ = env.step_local(actions)
            feats, edge = env.graph_observation()
            total += reward
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


def train_ppo_gnn_local(env, config: PPOConfig | None = None,
                        hidden: int = 48, rounds: int = 3, use_messages: bool = True):
    """PPO for the per-tissue GNN controller. ``env`` must be ``per_tissue=True``.

    Each step the policy emits one action per tissue; the joint action's log-prob
    is the sum over tissues. Graphs (and per-node actions) vary in size, so both
    are stored as lists and replayed through the GNN during the update."""
    assert getattr(env, "per_tissue", False), "env must be created with per_tissue=True"
    cfg = config or PPOConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    act_dim = env.NODE_ACTION_DIM
    env.reset(seed=cfg.seed)
    feats, edge = env.graph_observation()
    agent = GNNNodeActorCritic(feats.shape[1], act_dim, hidden=hidden, rounds=rounds,
                               use_messages=use_messages)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    feats_buf: list = [None] * cfg.n_steps
    edge_buf: list = [None] * cfg.n_steps
    act_buf: list = [None] * cfg.n_steps          # per-step [N, A] numpy
    logp_buf = torch.zeros(cfg.n_steps)
    rew_buf = torch.zeros(cfg.n_steps)
    done_buf = torch.zeros(cfg.n_steps)
    val_buf = torch.zeros(cfg.n_steps)

    history = {"timestep": [], "ep_return": []}
    minibatch_size = cfg.n_steps // cfg.n_minibatches
    next_done = 0.0
    ep_return = 0.0
    global_step = 0
    num_iterations = max(1, cfg.total_timesteps // cfg.n_steps)

    for _ in range(num_iterations):
        finished: list[float] = []
        for t in range(cfg.n_steps):
            global_step += 1
            feats_buf[t], edge_buf[t], done_buf[t] = feats, edge, next_done
            ft = torch.as_tensor(feats, dtype=torch.float32)
            et = torch.as_tensor(edge, dtype=torch.long)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(ft, et)
            act_buf[t] = action.numpy()
            logp_buf[t] = logprob
            val_buf[t] = value

            reward, terminated, truncated, _ = env.step_local(action.numpy())
            done = bool(terminated or truncated)
            rew_buf[t] = reward
            ep_return += reward
            if done:
                finished.append(ep_return)
                ep_return = 0.0
                env.reset()
            feats, edge = env.graph_observation()
            next_done = 1.0 if done else 0.0

        with torch.no_grad():
            next_value = agent.get_value(torch.as_tensor(feats, dtype=torch.float32),
                                         torch.as_tensor(edge, dtype=torch.long))
            advantages = torch.zeros(cfg.n_steps)
            last_gae = 0.0
            for t in reversed(range(cfg.n_steps)):
                if t == cfg.n_steps - 1:
                    nonterminal, next_val = 1.0 - next_done, next_value
                else:
                    nonterminal, next_val = 1.0 - done_buf[t + 1], val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * next_val * nonterminal - val_buf[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + val_buf

        idx = np.arange(cfg.n_steps)
        for _ in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, cfg.n_steps, minibatch_size):
                mb = idx[start:start + minibatch_size]
                new_logp, ent, vals = [], [], []
                for t in mb:
                    ft = torch.as_tensor(feats_buf[t], dtype=torch.float32)
                    et = torch.as_tensor(edge_buf[t], dtype=torch.long)
                    at = torch.as_tensor(act_buf[t], dtype=torch.float32)
                    _, lp, en, v = agent.get_action_and_value(ft, et, at)
                    new_logp.append(lp); ent.append(en); vals.append(v)
                new_logp = torch.stack(new_logp)
                ent = torch.stack(ent)
                vals = torch.stack(vals)

                ratio = torch.exp(new_logp - logp_buf[mb])
                mb_adv = advantages[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((vals - returns[mb]) ** 2).mean()
                loss = pg_loss - cfg.ent_coef * ent.mean() + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        history["timestep"].append(global_step)
        history["ep_return"].append(
            float(np.mean(finished)) if finished else float("nan"))

    return agent, history
