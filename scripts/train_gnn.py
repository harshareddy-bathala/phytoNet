"""Train the GNN graph policy (learnable inter-tissue messages) with PPO.

    python scripts/train_gnn.py                      # randomised conditions
    python scripts/train_gnn.py --scenario drought
    python scripts/train_gnn.py --compare-mlp --save-curve gnn.png

Reports GNN vs random / balanced (and optionally the MLP baseline), and prints a
breakdown of the *learned messages* by tissue type -- what the plant learned to
communicate. Needs torch.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from phytonet.env import PhytoNetEnv
from phytonet.ppo import PPOConfig
from phytonet.ppo_gnn import evaluate_gnn, train_ppo_gnn
from phytonet.scenarios import SCENARIOS, make_scenario

_KIND = ["seed", "root", "stem", "leaf"]


def _baseline(env, action_fn, episodes=8, base_seed=10_000):
    totals = []
    for e in range(episodes):
        env.reset(seed=base_seed + e)
        total, done = 0.0, False
        while not done:
            _, r, te, tr, _ = env.step(action_fn(env))
            total += r
            done = te or tr
        totals.append(total)
    return float(np.mean(totals))


def analyze_messages(env, agent, seed=0):
    """Average learned message strength by (sender kind -> receiver kind)."""
    env.reset(seed=seed)
    feats, edge = env.graph_observation()
    buckets = defaultdict(list)
    done = False
    while not done:
        _, norms = agent.messages(feats, edge)
        kinds = feats[:, :4].argmax(axis=1)
        for k in range(edge.shape[1]):
            src, tgt = edge[0, k], edge[1, k]
            buckets[(_KIND[kinds[src]], _KIND[kinds[tgt]])].append(float(norms[k]))
        action = agent.act_greedy(feats, edge)
        _, _, te, tr, _ = env.step(action)
        feats, edge = env.graph_observation()
        done = te or tr
    return {pair: float(np.mean(v)) for pair, v in buckets.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Train the PhytoNet GNN policy with PPO.")
    p.add_argument("--scenario", choices=sorted(SCENARIOS))
    p.add_argument("--timesteps", type=int, default=30_000)
    p.add_argument("--max-days", type=int, default=80)
    p.add_argument("--compare-mlp", action="store_true", help="also train the MLP baseline")
    p.add_argument("--save-curve", metavar="PATH")
    p.add_argument("--save-model", metavar="PATH")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    if args.scenario:
        make = lambda: PhytoNetEnv(make_scenario(args.scenario), max_days=args.max_days,
                                   randomize=False)
        where = f"scenario={args.scenario}"
    else:
        make = lambda: PhytoNetEnv(max_days=args.max_days, randomize=True)
        where = "randomised conditions"

    const = lambda vec: (lambda env: np.array(vec, dtype=np.float32))
    base_random = _baseline(make(), lambda e: e.action_space.sample())
    base_balanced = _baseline(make(), const([0, 0, 0]))

    cfg = PPOConfig(total_timesteps=args.timesteps, n_steps=512, n_epochs=4, seed=0)
    print(f"Training GNN policy on {where} for {args.timesteps} steps ...")
    agent, history = train_ppo_gnn(make(), cfg, hidden=48, rounds=3)
    gnn_score = evaluate_gnn(make(), agent, episodes=8, base_seed=10_000)

    mlp_score = None
    if args.compare_mlp:
        from phytonet.ppo import train_ppo, evaluate_agent
        print("Training MLP baseline for comparison ...")
        mlp_agent, _ = train_ppo(make(), PPOConfig(total_timesteps=args.timesteps, seed=0))
        mlp_score = evaluate_agent(make(), mlp_agent, episodes=8, base_seed=10_000)

    print("\nMean episodic return (~ final biomass):")
    print(f"  random actions : {base_random:.2f}")
    print(f"  fixed balanced : {base_balanced:.2f}")
    if mlp_score is not None:
        print(f"  MLP policy     : {mlp_score:.2f}")
    print(f"  GNN policy     : {gnn_score:.2f}")

    print("\nLearned messages (mean strength, sender -> receiver):")
    msgs = analyze_messages(make(), agent, seed=0)
    for pair, v in sorted(msgs.items(), key=lambda kv: -kv[1]):
        print(f"  {pair[0]:5s} -> {pair[1]:5s} : {v:.3f}")

    if args.save_model:
        import torch
        torch.save(agent.state_dict(), args.save_model)
        print(f"\nSaved {args.save_model}")

    ts, ret = np.array(history["timestep"]), np.array(history["ep_return"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, ret, color="#2f6b2a", lw=1.5, label="GNN (training)")
    ax.axhline(base_random, ls="--", color="#999", label="random baseline")
    if mlp_score is not None:
        ax.axhline(mlp_score, ls=":", color="#1f77b4", label="MLP (final)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episodic return (~ final biomass)")
    ax.set_title(f"PhytoNet GNN learning curve — {where}")
    ax.legend(); ax.grid(alpha=0.3)
    if args.save_curve:
        fig.savefig(args.save_curve, dpi=120, bbox_inches="tight")
        print(f"Saved {args.save_curve}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
