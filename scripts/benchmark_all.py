"""Cross-paradigm benchmark: every controller on one common metric.

Trains/evolves each paradigm, then scores them all by the same outcome (final
biomass + reproduction) across scenarios x seeds, and draws one comparison
figure. The headline question: does the brain-free **bioelectric** controller
rival the **neural** ones (MLP/GNN)?

    python scripts/benchmark_all.py --outdir figures
    python scripts/benchmark_all.py --ppo-steps 30000 --gnn-steps 22000 --evo-iters 12
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from phytonet.benchmark import (
    eval_global_outcome, eval_gnn_outcome, eval_signaling_outcome,
)
from phytonet.env import PhytoNetEnv
from phytonet.ppo import PPOConfig, train_ppo
from phytonet.ppo_gnn import train_ppo_gnn_local
from phytonet.signaling import evolve_signaling

SCENARIOS = ["baseline", "drought", "shade", "patchy", "monsoon"]


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-paradigm PhytoNet benchmark.")
    p.add_argument("--days", type=int, default=80)
    p.add_argument("--eval-seeds", type=int, default=4)
    p.add_argument("--ppo-steps", type=int, default=24000)
    p.add_argument("--gnn-steps", type=int, default=18000)
    p.add_argument("--evo-iters", type=int, default=10)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    seeds = range(args.eval_seeds)
    days = args.days

    print("Training MLP-PPO ...")
    mlp, _ = train_ppo(PhytoNetEnv(max_days=days, randomize=True),
                       PPOConfig(total_timesteps=args.ppo_steps, seed=0))
    print("Training per-tissue GNN ...")
    gnn, _ = train_ppo_gnn_local(
        PhytoNetEnv(max_days=days, randomize=True, per_tissue=True),
        PPOConfig(total_timesteps=args.gnn_steps, n_steps=512, n_epochs=4, seed=0))
    print("Evolving bioelectric signaling rule ...")
    sig, _ = evolve_signaling(days=days, iterations=args.evo_iters, population=18)

    # Every controller as an outcome-vector function over a scenario.
    controllers = {
        "random":      lambda sc: eval_global_outcome(
                            lambda o: np.random.uniform(-1, 1, 3).astype(np.float32),
                            sc, seeds, days),
        "balanced":    lambda sc: eval_global_outcome(
                            lambda o: np.zeros(3, np.float32), sc, seeds, days),
        "MLP-PPO":     lambda sc: eval_global_outcome(
                            lambda o: mlp.act_greedy(o), sc, seeds, days),
        "GNN (tissue)": lambda sc: eval_gnn_outcome(gnn, sc, seeds, days),
        "bioelectric": lambda sc: eval_signaling_outcome(sig, sc, seeds, days),
    }

    table = {name: {sc: fn(sc) for sc in SCENARIOS} for name, fn in controllers.items()}

    # Print the table (mean +/- std).
    print("\nOutcome (final biomass + reproduction), mean over "
          f"{args.eval_seeds} seeds:\n")
    head = f"{'controller':14s}" + "".join(f"{sc:>10s}" for sc in SCENARIOS) + f"{'OVERALL':>10s}"
    print(head)
    overall = {}
    for name, row in table.items():
        means = [row[sc].mean() for sc in SCENARIOS]
        overall[name] = float(np.mean(means))
        cells = "".join(f"{m:10.1f}" for m in means) + f"{overall[name]:10.1f}"
        print(f"{name:14s}{cells}")

    # Figure: grouped bars per scenario + an OVERALL group.
    groups = SCENARIOS + ["OVERALL"]
    x = np.arange(len(groups))
    names = list(controllers)
    w = 0.16
    colors = {"random": "#bbb", "balanced": "#888", "MLP-PPO": "#1f77b4",
              "GNN (tissue)": "#9467bd", "bioelectric": "#2f9e44"}
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(names):
        vals = [table[name][sc].mean() for sc in SCENARIOS] + [overall[name]]
        errs = [table[name][sc].std() for sc in SCENARIOS] + [0.0]
        ax.bar(x + (i - 2) * w, vals, w, yerr=errs, capsize=2,
               label=name, color=colors[name])
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("outcome (biomass + reproduction)")
    ax.set_title("PhytoNet — cross-paradigm benchmark (brain-free vs neural)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    path = os.path.join(args.outdir, "benchmark_all.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nSaved {path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
