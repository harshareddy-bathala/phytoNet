"""Watch PhytoNet learn: train a tiny linear policy with the Cross-Entropy Method.

    python scripts/train_cem.py                     # train, show the learning curve
    python scripts/train_cem.py --iterations 30 --save-curve curve.png
    python scripts/train_cem.py --scenario drought  # learn in one fixed condition

By default episodes are randomised across scenarios (drought/shade/monsoon/...),
so the policy learns to grow well *across conditions*. This is the lightweight,
numpy-only stand-in for the eventual PPO + GNN training.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from phytonet.env import PhytoNetEnv
from phytonet.learn import LinearPolicy, evaluate, train_cem
from phytonet.scenarios import SCENARIOS, make_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhytoNet with CEM.")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS),
                        help="fix one condition (default: randomise across all)")
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--save-curve", metavar="PATH", help="save the learning curve PNG")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.scenario:
        env = PhytoNetEnv(make_scenario(args.scenario), max_days=args.days, randomize=False)
        where = f"scenario={args.scenario}"
    else:
        env = PhytoNetEnv(max_days=args.days, randomize=True)
        where = "randomised conditions"

    # Baseline: a do-nothing (zero) policy, for reference.
    zero = LinearPolicy(env.observation_space.shape[0], env.action_space.shape[0])
    base_score = evaluate(env, zero, episodes=5, base_seed=0)

    print(f"Training CEM on {where} ...")
    result = train_cem(env, iterations=args.iterations, population=args.population)
    print(f"Baseline (zero policy) reward : {base_score:.2f}")
    print(f"Best learned reward          : {result.best_score:.2f}")
    print(f"Improvement                  : {result.best_score - base_score:+.2f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(result.history) + 1), result.history, marker="o", color="#3a7d34")
    ax.axhline(base_score, ls="--", color="#999", label="zero-policy baseline")
    ax.set_xlabel("CEM iteration")
    ax.set_ylabel("best episode reward (≈ final biomass)")
    ax.set_title(f"PhytoNet learning curve — {where}")
    ax.legend()
    ax.grid(alpha=0.3)
    if args.save_curve:
        fig.savefig(args.save_curve, dpi=120, bbox_inches="tight")
        print(f"Saved {args.save_curve}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
