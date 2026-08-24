"""Validate the simulator against known plant biology (functional equilibrium).

    python scripts/validate.py --outdir figures

Reports the root:shoot ratio by scenario and checks the balanced-growth
prediction (more roots when water-limited, more shoots when light-limited),
which the model reproduces *emergently* from physiology + resource costs.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from phytonet.signaling import attach_signaling
from phytonet.validation import functional_equilibrium


def main() -> None:
    p = argparse.ArgumentParser(description="PhytoNet biological validation.")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--days", type=int, default=70)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    scenarios = ["drought", "baseline", "shade"]
    fixed = functional_equilibrium(range(args.seeds), args.days, attach=None,
                                   scenarios=scenarios)
    bio = functional_equilibrium(range(args.seeds), args.days, attach=attach_signaling,
                                 scenarios=scenarios)

    print("Root:shoot ratio (mean +/- std) -- functional-equilibrium check")
    print("Prediction: drought > baseline > shade\n")
    print(f"  {'scenario':10s}{'fixed':>16s}{'bioelectric':>16s}")
    for sc in scenarios:
        print(f"  {sc:10s}{fixed[sc][0]:8.2f}+/-{fixed[sc][1]:<4.2f}"
              f"{bio[sc][0]:10.2f}+/-{bio[sc][1]:<4.2f}")
    ok = fixed["drought"][0] > fixed["baseline"][0] > fixed["shade"][0]
    print(f"\n  Reproduces functional equilibrium (drought>baseline>shade): {ok}")

    x = np.arange(len(scenarios)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, [fixed[s][0] for s in scenarios], w,
           yerr=[fixed[s][1] for s in scenarios], capsize=3, label="fixed", color="#888")
    ax.bar(x + w / 2, [bio[s][0] for s in scenarios], w,
           yerr=[bio[s][1] for s in scenarios], capsize=3, label="bioelectric", color="#2f9e44")
    ax.set_xticks(x); ax.set_xticklabels(scenarios)
    ax.set_ylabel("root : shoot ratio")
    ax.set_title("Validation: functional-equilibrium response\n"
                 "(more roots when water-limited, fewer when light-limited)")
    ax.legend()
    path = os.path.join(args.outdir, "validation_functional_equilibrium.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved {path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
