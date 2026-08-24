"""Side-by-side comparison of controllers -- fast, no training.

Generates two figures so you can test the paradigms against each other locally:
  * compare/performance.png -- outcome (biomass + reproduction) per scenario,
    balanced vs bioelectric vs learned, with error bars;
  * compare/morphology.png  -- the actual plants each controller grows on one
    scenario, rendered side by side.

    python scripts/compare.py                          # opens both windows
    python scripts/compare.py --scenario drought --no-show
    python scripts/compare.py --scenarios baseline drought herbivory patchy
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from phytonet.benchmark import (
    eval_global_outcome, eval_learned_outcome, eval_signaling_outcome,
)
from phytonet.config import build_simulation
from phytonet.learned_signaling import LearnedSignalingParams, attach_learned_signaling
from phytonet.scenarios import SCENARIOS, make_scenario
from phytonet.signaling import SignalingParams, attach_signaling
from phytonet.visualize import render_frame

_COLORS = {"balanced": "#888", "bioelectric": "#2f9e44", "learned": "#7a3fb0"}


def performance_fig(scenarios, seeds, days, path):
    methods = {
        "balanced": lambda sc: eval_global_outcome(
            lambda o: np.zeros(3, np.float32), sc, seeds, days),
        "bioelectric": lambda sc: eval_signaling_outcome(SignalingParams(), sc, seeds, days),
        "learned": lambda sc: eval_learned_outcome(
            LearnedSignalingParams.default(), sc, seeds, days),
    }
    x = np.arange(len(scenarios)); w = 0.26
    fig, ax = plt.subplots(figsize=(1.6 * len(scenarios) + 3, 4.5))
    for i, (name, fn) in enumerate(methods.items()):
        vals = [fn(sc) for sc in scenarios]
        means = [v.mean() for v in vals]
        errs = [v.std() for v in vals]
        ax.bar(x + (i - 1) * w, means, w, yerr=errs, capsize=2,
               label=name, color=_COLORS[name])
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylabel("outcome (biomass + reproduction)")
    ax.set_title("Controller comparison (hand / default params, no training)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def morphology_fig(scenario, days, seed, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    def grow(attach):
        sim = build_simulation(make_scenario(scenario, dim=2, days=days, seed=seed))
        if attach:
            attach(sim)
        sim.run(days)
        return sim

    panels = [("fixed (allocation)", None),
              ("bioelectric (electrical)", attach_signaling),
              ("learned (evolved-content)", attach_learned_signaling)]
    for ax, (title, attach) in zip(axes, panels):
        sim = grow(attach)
        render_frame(sim.world, sim.plant, ax, title=title)
    fig.suptitle(f"Plants grown by each controller — scenario: {scenario}")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description="Side-by-side controller comparison.")
    p.add_argument("--scenarios", nargs="+", default=["baseline", "drought", "shade",
                   "patchy", "herbivory"], choices=sorted(SCENARIOS))
    p.add_argument("--scenario", default="patchy", choices=sorted(SCENARIOS),
                   help="scenario for the morphology figure")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--days", type=int, default=70)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="figures/compare")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    performance_fig(args.scenarios, range(args.seeds), args.days,
                    os.path.join(args.outdir, "performance.png"))
    morphology_fig(args.scenario, args.days, args.seed,
                   os.path.join(args.outdir, "morphology.png"))
    print(f"Saved {args.outdir}/performance.png and {args.outdir}/morphology.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
