"""Audit the fairness of the neural baseline (PPO): is it *adequately trained*?

A reviewer's first attack on "brain-free beats neural" is "your PPO was
under-trained." This sweeps the training budget and plots the learned policy's
return vs timesteps against fixed references. The expected, fair picture: PPO
climbs from random-init, **beats a naive fixed policy** (so it genuinely learns),
**converges** (so more training wouldn't help), and lands at the balanced ceiling
— which is why it only *ties* random (random allocation averages to balanced).

    python scripts/baseline_audit.py --budgets 3000 10000 25000 50000 --outdir figures
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phytonet.env import PhytoNetEnv
from phytonet.ppo import PPOConfig, evaluate_agent, train_ppo


def _fixed_return(action_vec, episodes=8, base_seed=10_000, max_days=80):
    env = PhytoNetEnv(max_days=max_days, randomize=True)
    totals = []
    for e in range(episodes):
        env.reset(seed=base_seed + e)
        total, done = 0.0, False
        while not done:
            _, r, te, tr, _ = env.step(np.array(action_vec, np.float32))
            total += r
            done = te or tr
        totals.append(total)
    return float(np.mean(totals))


def main() -> None:
    p = argparse.ArgumentParser(description="PPO baseline-fairness audit.")
    p.add_argument("--budgets", type=int, nargs="+", default=[3000, 10000, 25000, 50000])
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--max-days", type=int, default=80)
    p.add_argument("--outdir", default="figures")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    mk = lambda: PhytoNetEnv(max_days=args.max_days, randomize=True)

    balanced = _fixed_return([0, 0, 0], args.eval_episodes, max_days=args.max_days)
    shoot = _fixed_return([1, -1, -1], args.eval_episodes, max_days=args.max_days)
    rng = np.random.default_rng(0)
    random_ret = np.mean([_fixed_return(rng.uniform(-1, 1, 3), 1, 10_000 + i, args.max_days)
                          for i in range(args.eval_episodes)])

    ppo_returns = []
    for b in args.budgets:
        print(f"  training PPO for {b} steps ...")
        agent, _ = train_ppo(mk(), PPOConfig(total_timesteps=b, seed=0))
        ppo_returns.append(evaluate_agent(mk(), agent, episodes=args.eval_episodes,
                                          base_seed=10_000))

    print("\nPPO return vs training budget:")
    for b, r in zip(args.budgets, ppo_returns):
        print(f"   {b:7d} steps : {r:.2f}")
    print(f"   references -> balanced {balanced:.2f} | shoot-heavy {shoot:.2f} | "
          f"random {random_ret:.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(args.budgets, ppo_returns, "o-", color="#1f77b4", label="PPO (greedy)")
    ax.axhline(balanced, ls="--", color="#2f9e44", label="balanced (ceiling)")
    ax.axhline(shoot, ls=":", color="#c0392b", label="shoot-heavy (naive fixed)")
    ax.axhline(random_ret, ls="-.", color="#999", label="random")
    ax.set_xlabel("training timesteps"); ax.set_ylabel("episodic return (greedy)")
    ax.set_title("Baseline-fairness audit: PPO learns, beats naive, converges")
    ax.legend(fontsize=8)
    path = os.path.join(args.outdir, "baseline_audit.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved {path}")


if __name__ == "__main__":
    main()
