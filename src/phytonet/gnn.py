"""A GNN actor-critic with learnable inter-tissue messages.

This is the heart of PhytoNet's research question. The plant body is a graph of
tissues; this network lets each tissue (node) read its local features, then
**exchange learned messages with its neighbours** over several rounds, and from
the resulting representation decide how to act. The message functions are
learned end-to-end by RL -- so the inter-tissue signals are *discovered* to
serve survival/growth, exactly the thing we want to compare against real plant
signaling.

Message passing is hand-written in plain PyTorch (no PyTorch Geometric) so every
step of the communication is explicit and inspectable -- and so it installs
cleanly. (PyG could replace ``MessagePassing`` later behind the same interface.)

One round, for a directed edge j -> i:
    m_{j->i} = msg_mlp([h_i, h_j])            # the learned message
    a_i      = mean_j m_{j->i}                 # aggregate neighbour messages
    h_i      = h_i + upd_mlp([h_i, a_i])       # update (residual)

After K rounds we mean-pool node states into a graph embedding and read out a
policy (action distribution) and a value. This increment keeps the *global*
allocation action (so it plugs into the existing PPO and is directly comparable
to the MLP baseline); per-tissue actions are the next increment.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PhytoNet's GNN needs PyTorch. Install the CPU build with:\n"
        '  pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu'
    ) from exc


def _mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class MessagePassing(nn.Module):
    """One round of learned message passing over an undirected graph.

    ``edge_index`` is ``[2, E]`` with rows (source, target); we already include
    both directions, so a message flows source -> target and is aggregated
    (mean) at the target."""

    def __init__(self, hidden: int):
        super().__init__()
        self.msg_mlp = _mlp([2 * hidden, hidden, hidden])   # the learned message
        self.upd_mlp = _mlp([2 * hidden, hidden, hidden])   # the state update

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        if edge_index.shape[1] == 0:
            agg = torch.zeros_like(h)
        else:
            src, tgt = edge_index[0], edge_index[1]
            messages = self.msg_mlp(torch.cat([h[tgt], h[src]], dim=-1))
            agg = torch.zeros_like(h).index_add_(0, tgt, messages)
            deg = torch.zeros(n, device=h.device).index_add_(
                0, tgt, torch.ones(tgt.shape[0], device=h.device))
            agg = agg / deg.clamp(min=1.0).unsqueeze(-1)
        return h + self.upd_mlp(torch.cat([h, agg], dim=-1))  # residual update


class GNNActorCritic(nn.Module):
    """Encode tissues -> K rounds of learned messages -> pool -> policy + value."""

    def __init__(self, node_dim: int, act_dim: int, hidden: int = 48, rounds: int = 3):
        super().__init__()
        self.encoder = _mlp([node_dim, hidden, hidden])
        self.layers = nn.ModuleList([MessagePassing(hidden) for _ in range(rounds)])
        self.actor_mean = _mlp([hidden, hidden, act_dim])
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        self.critic = _mlp([hidden, hidden, 1])

    def embed(self, feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run the GNN and mean-pool to a single graph embedding."""
        h = self.encoder(feats)
        for layer in self.layers:
            h = layer(h, edge_index)
        return h.mean(dim=0)  # graph-level readout

    def get_action_and_value(self, feats, edge_index, action=None):
        g = self.embed(feats, edge_index)
        mean = self.actor_mean(g)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(g).squeeze(-1)
        return action, logprob, entropy, value

    def get_value(self, feats, edge_index) -> torch.Tensor:
        return self.critic(self.embed(feats, edge_index)).squeeze(-1)

    @torch.no_grad()
    def act_greedy(self, feats: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        f = torch.as_tensor(feats, dtype=torch.float32)
        e = torch.as_tensor(edge_index, dtype=torch.long)
        return self.actor_mean(self.embed(f, e)).numpy()

    @torch.no_grad()
    def messages(self, feats: np.ndarray, edge_index: np.ndarray):
        """Expose the learned messages on each edge (for analysis/figures).

        Returns ``(edge_index, message_norms)`` -- how strongly each tissue
        signals its neighbour in the first round. This is the artifact the
        thesis inspects: what did the plant learn to communicate?"""
        f = torch.as_tensor(feats, dtype=torch.float32)
        e = torch.as_tensor(edge_index, dtype=torch.long)
        h = self.encoder(f)
        if e.shape[1] == 0:
            return edge_index, np.zeros(0, dtype=np.float32)
        src, tgt = e[0], e[1]
        msg = self.layers[0].msg_mlp(torch.cat([h[tgt], h[src]], dim=-1))
        return edge_index, msg.norm(dim=-1).numpy()


class GNNNodeActorCritic(nn.Module):
    """Per-tissue actor-critic: every node gets its own action distribution
    (the distributed controller), the value is graph-level.

    Increment 2: each tissue, after exchanging learned messages with neighbours,
    decides its *own* growth action (e.g. [grow, thicken]); the joint action is
    all node actions and its log-prob is their sum. This is what lets the plant
    do what a single global knob can't -- grow roots in one place and shoots in
    another at the same time."""

    def __init__(self, node_dim: int, act_dim: int, hidden: int = 48, rounds: int = 3,
                 use_messages: bool = True):
        super().__init__()
        self.encoder = _mlp([node_dim, hidden, hidden])
        self.layers = nn.ModuleList([MessagePassing(hidden) for _ in range(rounds)])
        self.actor_mean = _mlp([hidden, hidden, act_dim])   # per-node action mean
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        self.critic = _mlp([hidden, hidden, 1])             # graph-level value
        # Ablation switch: when False, the same layers run but with NO edges, so
        # tissues never exchange messages (each updates from itself only). Same
        # capacity, no communication -- isolates whether signaling actually helps.
        self.use_messages = use_messages

    def embed_nodes(self, feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(feats)
        if not self.use_messages:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=h.device)
        for layer in self.layers:
            h = layer(h, edge_index)
        return h  # [N, hidden] -- per-tissue representation (message-informed)

    def get_action_and_value(self, feats, edge_index, action=None):
        h = self.embed_nodes(feats, edge_index)
        mean = self.actor_mean(h)                  # [N, A]
        std = torch.exp(self.actor_logstd)         # [A]
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()                 # [N, A]
        logprob = dist.log_prob(action).sum()      # joint over nodes & dims
        entropy = dist.entropy().sum()
        value = self.critic(h.mean(dim=0)).squeeze(-1)
        return action, logprob, entropy, value

    def get_value(self, feats, edge_index) -> torch.Tensor:
        h = self.embed_nodes(feats, edge_index)
        return self.critic(h.mean(dim=0)).squeeze(-1)

    @torch.no_grad()
    def act_greedy(self, feats: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        f = torch.as_tensor(feats, dtype=torch.float32)
        e = torch.as_tensor(edge_index, dtype=torch.long)
        return self.actor_mean(self.embed_nodes(f, e)).numpy()  # [N, A]

    @torch.no_grad()
    def messages(self, feats: np.ndarray, edge_index: np.ndarray):
        f = torch.as_tensor(feats, dtype=torch.float32)
        e = torch.as_tensor(edge_index, dtype=torch.long)
        h = self.encoder(f)
        if e.shape[1] == 0:
            return edge_index, np.zeros(0, dtype=np.float32)
        src, tgt = e[0], e[1]
        msg = self.layers[0].msg_mlp(torch.cat([h[tgt], h[src]], dim=-1))
        return edge_index, msg.norm(dim=-1).numpy()
