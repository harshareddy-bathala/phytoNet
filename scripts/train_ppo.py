"""Train PhytoNet with PPO (proper deep RL) and plot the learning curve.

    python scripts/train_ppo.py                       # randomised conditions
    python scripts/train_ppo.py --scenario drought    # one fixed condition
    python scripts/train_ppo.py --timesteps 100000 --save-curve ppo.png --save-model ppo.pt

Needs PyTorch (CPU is fine):
    pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from phytonet.env import PhytoNetEnv
from phytonet.ppo import PPOConfig, evaluate_agent, train_ppo
from phytonet.scenarios import SCENARIOS, make_scenario


def _baseline(env, action_fn, episodes: int = 8, base_seed: int = 0) -> float:
    """Mean return of a fixed behaviour (action_fn(env) -> action)."""
    totals = []
    for e in range(episodes):
        env.reset(seed=base_seed + e)
        total, done = 0.0, False
        while not done:
            _, r, term, trunc, _ = env.step(action_fn(env))
            total += r
            done = term or trunc
        totals.append(total)
    return float(np.mean(totals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhytoNet with PPO.")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS),
                        help="fix one condition (default: randomise across all)")
    parser.add_argument("--timesteps", type=int, default=60_000)
    parser.add_argument("--max-days", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-curve", metavar="PATH")
    parser.add_argument("--save-model", metavar="PATH")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.scenario:
        make = lambda: PhytoNetEnv(make_scenario(args.scenario),
                                   max_days=args.max_days, randomize=False)
        where = f"scenario={args.scenario}"
    else:
        make = lambda: PhytoNetEnv(max_days=args.max_days, randomize=True)
        where = "randomised conditions"

    ev, sd = args.eval_episodes, 10_000
    # Fixed-policy baselines for context. Random averages to ~balanced, so it's a
    # strong baseline; a naive mis-allocation (shoot-heavy) shows the real gap.
    const = lambda vec: (lambda env: np.array(vec, dtype=np.float32))
    base_random = _baseline(make(), lambda e: e.action_space.sample(), ev, sd)
    base_balanced = _baseline(make(), const([0, 0, 0]), ev, sd)
    base_shoot = _baseline(make(), const([1, -1, -1]), ev, sd)
    base_root = _baseline(make(), const([-1, 1, -1]), ev, sd)

    print(f"Training PPO on {where} for {args.timesteps} steps ...")
    cfg = PPOConfig(total_timesteps=args.timesteps, seed=args.seed)
    agent, history = train_ppo(make(), cfg)

    trained = evaluate_agent(make(), agent, episodes=ev, base_seed=sd)
    print("\nMean episodic return (~ final biomass):")
    print(f"  fixed shoot-heavy : {base_shoot:.2f}")
    print(f"  fixed root-heavy  : {base_root:.2f}")
    print(f"  fixed balanced    : {base_balanced:.2f}")
    print(f"  random actions    : {base_random:.2f}")
    print(f"  PPO (learned)     : {trained:.2f}   "
          f"(+{trained - base_shoot:.2f} vs naive shoot-heavy)")
    base = base_random

    if args.save_model:
        import torch
        torch.save(agent.state_dict(), args.save_model)
        print(f"Saved {args.save_model}")

    ts = np.array(history["timestep"])
    ret = np.array(history["ep_return"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, ret, color="#3a7d34", lw=1.5, label="PPO (training)")
    ax.axhline(base, ls="--", color="#999", label="random-action baseline")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episodic return (≈ final biomass)")
    ax.set_title(f"PhytoNet PPO learning curve — {where}")
    ax.legend()
    ax.grid(alpha=0.3)
    if args.save_curve:
        fig.savefig(args.save_curve, dpi=120, bbox_inches="tight")
        print(f"Saved {args.save_curve}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
