"""Tests for the per-tissue graph observation and the GNN policy.

Skipped if torch isn't installed. Kept fast: tiny networks and short runs.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phytonet.config import SimConfig
from phytonet.env import NODE_FEATURE_DIM, PhytoNetEnv
from phytonet.gnn import GNNActorCritic, MessagePassing
from phytonet.ppo import PPOConfig
from phytonet.ppo_gnn import evaluate_gnn, train_ppo_gnn


def _fixed_env(days=30):
    return PhytoNetEnv(SimConfig(), max_days=days, randomize=False)


def test_graph_observation_shapes_match_plant():
    env = _fixed_env()
    env.reset(seed=0)
    feats, edge = env.graph_observation()
    n = len(env.sim.plant.nodes)
    assert feats.shape == (n, NODE_FEATURE_DIM)
    assert edge.shape[0] == 2
    assert edge.shape[1] == 2 * len(env.sim.plant.edges)  # both directions
    assert edge.max() < n  # indices in range


def test_kind_one_hot_is_set():
    env = _fixed_env()
    env.reset(seed=0)
    feats, _ = env.graph_observation()
    # Each row has exactly one kind bit set among the first 4 columns.
    assert np.all(feats[:, :4].sum(axis=1) == 1.0)


def test_message_passing_propagates_neighbour_info():
    torch.manual_seed(0)
    mp = MessagePassing(hidden=8)
    h = torch.randn(3, 8)
    # A connected graph vs no edges should give different updates.
    connected = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    empty = torch.zeros((2, 0), dtype=torch.long)
    out_connected = mp(h, connected)
    out_empty = mp(h, empty)
    assert not torch.allclose(out_connected, out_empty)
    assert out_connected.shape == h.shape


def test_gnn_actor_critic_forward_shapes():
    torch.manual_seed(0)
    agent = GNNActorCritic(node_dim=NODE_FEATURE_DIM, act_dim=3, hidden=16, rounds=2)
    feats = torch.randn(5, NODE_FEATURE_DIM)
    edge = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    action, logprob, entropy, value = agent.get_action_and_value(feats, edge)
    assert action.shape == (3,)
    assert logprob.shape == () and entropy.shape == () and value.shape == ()


def test_messages_expose_one_norm_per_directed_edge():
    env = _fixed_env()
    env.reset(seed=0)
    feats, edge = env.graph_observation()
    agent = GNNActorCritic(NODE_FEATURE_DIM, 3, hidden=16, rounds=2)
    _, norms = agent.messages(feats, edge)
    assert norms.shape[0] == edge.shape[1]
    assert np.all(norms >= 0)


def test_train_ppo_gnn_runs_and_is_not_worse_than_random():
    env = _fixed_env(days=30)
    cfg = PPOConfig(total_timesteps=900, n_steps=300, n_epochs=2,
                    n_minibatches=2, seed=0)
    agent, history = train_ppo_gnn(env, cfg, hidden=16, rounds=2)
    assert len(history["timestep"]) == cfg.total_timesteps // cfg.n_steps

    trained = evaluate_gnn(_fixed_env(30), agent, episodes=3, base_seed=50)

    renv = _fixed_env(30)
    totals = []
    for e in range(3):
        renv.reset(seed=50 + e)
        total, done = 0.0, False
        while not done:
            _, r, te, tr, _ = renv.step(renv.action_space.sample())
            total += r
            done = te or tr
        totals.append(total)
    random_score = float(np.mean(totals))
    assert trained >= random_score - 3.0  # lenient: just guard against a broken update


# -- increment 2: per-tissue actions --------------------------------------
def _local_env(days=30):
    return PhytoNetEnv(SimConfig(), max_days=days, randomize=False, per_tissue=True)


def test_node_actor_critic_forward_shapes():
    from phytonet.gnn import GNNNodeActorCritic
    torch.manual_seed(0)
    agent = GNNNodeActorCritic(node_dim=NODE_FEATURE_DIM, act_dim=2, hidden=16, rounds=2)
    feats = torch.randn(5, NODE_FEATURE_DIM)
    edge = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    action, logprob, entropy, value = agent.get_action_and_value(feats, edge)
    assert action.shape == (5, 2)              # one action per tissue
    assert logprob.shape == () and value.shape == ()  # joint logprob, graph value


def test_step_local_grows_the_plant():
    env = _local_env(days=25)
    env.reset(seed=0)
    n0 = len(env.sim.plant.nodes)
    for _ in range(25):
        feats, edge = env.graph_observation()
        # All tissues choose to grow + thicken (positive logits).
        actions = np.ones((feats.shape[0], env.NODE_ACTION_DIM), dtype=np.float32)
        reward, term, trunc, info = env.step_local(actions)
        if trunc:
            break
    assert len(env.sim.plant.nodes) > n0


def test_negative_actions_do_not_grow():
    env = _local_env(days=15)
    env.reset(seed=0)
    n0 = len(env.sim.plant.nodes)
    for _ in range(15):
        feats, edge = env.graph_observation()
        actions = -np.ones((feats.shape[0], env.NODE_ACTION_DIM), dtype=np.float32)
        _, _, trunc, _ = env.step_local(actions)
        if trunc:
            break
    assert len(env.sim.plant.nodes) == n0  # nobody chose to grow


def test_train_ppo_gnn_local_runs():
    from phytonet.ppo_gnn import evaluate_gnn_local, train_ppo_gnn_local
    env = _local_env(days=30)
    cfg = PPOConfig(total_timesteps=900, n_steps=300, n_epochs=2,
                    n_minibatches=2, seed=0)
    agent, history = train_ppo_gnn_local(env, cfg, hidden=16, rounds=2)
    assert len(history["timestep"]) == cfg.total_timesteps // cfg.n_steps
    score = evaluate_gnn_local(_local_env(30), agent, episodes=2, base_seed=7)
    assert np.isfinite(score)
