"""Generate explanatory figures of the current PhytoNet model.

    python scripts/make_figures.py --outdir figures

Produces:
  * plant_model_2d.png   -- the plant as a graph of tissues, end of season
  * growth_<scenario>.png-- growth + resources + soil moisture over a season
  * allocation.png       -- how each fixed shoot/root/girth strategy pays off
                            per scenario (the "what's the right strategy?" view)

These are read-only views of the *current* model (hand-coded balanced growth);
no learning involved.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from phytonet.config import SimConfig, build_simulation
from phytonet.env import PhytoNetEnv
from phytonet.scenarios import make_scenario
from phytonet.visualize import render_frame


def plant_model_fig(outdir: str, days: int = 50) -> str:
    """A labelled 2D render of the plant graph (tissues + vascular edges)."""
    sim = build_simulation(SimConfig(dim=2, days=days))
    sim.run(days)
    ax = render_frame(sim.world, sim.plant, title=f"Plant model — graph of tissues (day {days})")
    path = os.path.join(outdir, "plant_model_2d.png")
    ax.figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(ax.figure)
    return path


def growth_fig(scenario: str, outdir: str, days: int = 70) -> str:
    """Two stacked panels: structure (counts/biomass) and resources/moisture."""
    sim = build_simulation(make_scenario(scenario, dim=2, days=days))
    sim.run(days)
    h = sim.history
    day = h["day"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(day, h["biomass"], color="#2f6b2a", lw=2.5, label="biomass")
    ax1.plot(day, h["n_leaves"], color="#5bbf52", lw=1.5, ls="--", label="# leaves")
    ax1.plot(day, h["n_roots"], color="#7a5230", lw=1.5, ls="--", label="# roots")
    ax1.set_ylabel("count / biomass")
    ax1.set_title(f"PhytoNet growth — scenario: {scenario}")
    ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)

    ax2.plot(day, h["total_sugar"], color="#e0a800", lw=1.8, label="pooled sugar")
    ax2.plot(day, h["total_water"], color="#1f77b4", lw=1.8, label="pooled water")
    ax2.plot(day, np.array(h["moisture"]) * 20, color="#8a5a2b", lw=1.5, ls=":",
             label="soil moisture (×20)")
    rain_days = [d for d, r in zip(day, h["raining"]) if r]
    for rd in rain_days:
        ax2.axvline(rd, color="#2a6fdb", alpha=0.15)
    if rain_days:
        ax2.plot([], [], color="#2a6fdb", alpha=0.4, label="rain day")
    ax2.set_xlabel("day"); ax2.set_ylabel("amount")
    ax2.legend(loc="upper left"); ax2.grid(alpha=0.3)

    path = os.path.join(outdir, f"growth_{scenario}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _eval_const(scenario: str, act, seeds=range(8), days=80) -> float:
    env = PhytoNetEnv(make_scenario(scenario), max_days=days, randomize=False)
    tot = []
    for s in seeds:
        env.reset(seed=s); done = False; r = 0.0
        while not done:
            _, rw, te, tr, _ = env.step(np.array(act, dtype=np.float32)); r += rw
            done = te or tr
        tot.append(r)
    return float(np.mean(tot))


def allocation_fig(outdir: str) -> str:
    """Grouped bars: final biomass of fixed strategies, per scenario.

    This is the 'prediction' view -- which growth strategy is best where. A flat
    group means the choice doesn't matter (no signal); a varied group means it
    does (something to learn)."""
    strategies = {
        "shoot-heavy": [1, -1, -1],
        "balanced": [0, 0, 0],
        "root-heavy": [-1, 1, -1],
    }
    scenarios = ["baseline", "drought", "shade", "monsoon"]
    data = {name: [_eval_const(sc, act) for sc in scenarios]
            for name, act in strategies.items()}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(scenarios)); w = 0.26
    colors = {"shoot-heavy": "#5bbf52", "balanced": "#888", "root-heavy": "#7a5230"}
    for i, (name, vals) in enumerate(data.items()):
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=colors[name])
    ax.set_xticks(x); ax.set_xticklabels(scenarios)
    ax.set_ylabel("final biomass (mean of 8 seeds)")
    ax.set_title("Which growth strategy pays off? (fixed policies)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    path = os.path.join(outdir, "allocation.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PhytoNet explanatory figures.")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(plant_model_fig(args.outdir))
    for sc in ("baseline", "drought"):
        print(growth_fig(sc, args.outdir))
    print(allocation_fig(args.outdir))


if __name__ == "__main__":
    main()
