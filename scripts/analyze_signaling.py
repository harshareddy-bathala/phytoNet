"""Characterise the plant's learned electrical signalling (the research question).

    python scripts/analyze_signaling.py --outdir figures
    python scripts/analyze_signaling.py --evolve   # evolve the rule first, then analyse

Produces:
  * signal_firing.png   -- firing rate + firing~stimulus correlation by tissue
  * signal_traces.png   -- membrane potential over time for sample tissues
Plus a printed summary comparing the learned signalling to real plant signalling.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from phytonet.analysis import (
    firing_rate_by_kind, firing_stimulus_correlation, run_and_log, summarize,
)
from phytonet.signaling import SignalingParams, evolve_signaling


def firing_figure(log, path):
    fr = firing_rate_by_kind(log)
    fc = firing_stimulus_correlation(log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    kinds = ["seed", "root", "stem", "leaf"]
    ax1.bar(kinds, [fr[k] for k in kinds],
            color=["#5a3a1a", "#7a5230", "#4a7d34", "#5bbf52"])
    ax1.set_ylabel("action-potential firing rate")
    ax1.set_title("Which tissues signal most")
    items = list(fc.items())
    ax2.bar([k for k, _ in items], [v for _, v in items], color="#7a3fb0")
    ax2.axhline(0, color="#999", lw=0.8)
    ax2.set_ylabel("correlation")
    ax2.set_title("Does firing encode the local stimulus?")
    ax2.set_ylim(-1, 1)
    fig.suptitle("PhytoNet — learned electrical signalling")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def trace_figure(log, path):
    """Membrane potential over time for one representative tissue of each kind."""
    a = log.as_arrays()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"root": "#7a5230", "stem": "#4a7d34", "leaf": "#5bbf52", "seed": "#5a3a1a"}
    for kind, color in colors.items():
        mask = a["kind"] == kind
        if not mask.any():
            continue
        days = a["day"][mask]
        V = a["V"][mask]
        # Average V across tissues of this kind per day (population signal).
        uniq = np.unique(days)
        meanV = [V[days == d].mean() for d in uniq]
        ax.plot(uniq, meanV, color=color, label=kind, lw=1.8)
    ax.set_xlabel("day")
    ax.set_ylabel("mean membrane potential V")
    ax.set_title("PhytoNet — electrical activity over the season")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Analyse PhytoNet's learned signalling.")
    p.add_argument("--scenario", default="patchy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--days", type=int, default=70)
    p.add_argument("--evolve", action="store_true", help="evolve the rule first")
    p.add_argument("--outdir", default="figures")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    params = SignalingParams()
    if args.evolve:
        print("Evolving the signalling rule first ...")
        params, _ = evolve_signaling(iterations=10, population=16, days=args.days)

    log = run_and_log(params, scenario=args.scenario, seed=args.seed, days=args.days)
    print(summarize(log))

    firing_figure(log, os.path.join(args.outdir, "signal_firing.png"))
    trace_figure(log, os.path.join(args.outdir, "signal_traces.png"))
    print(f"\nSaved figures to {args.outdir}/signal_firing.png, signal_traces.png")
    print("\nVs real plants: shoots/leaves are major signal sources (light, wounding) "
          "and signals propagate shoot->root (source->sink); compare the firing-rate "
          "and trace patterns above to that picture.")


if __name__ == "__main__":
    main()
