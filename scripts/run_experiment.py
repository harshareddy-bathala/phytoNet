"""Run the paper-grade PhytoNet experiment (Hydra-configured).

    python scripts/run_experiment.py                       # uses configs/experiment.yaml
    python scripts/run_experiment.py seeds=8 days=80
    python scripts/run_experiment.py wandb.enabled=true    # also log to Weights & Biases

Writes a tidy results CSV, a summary with 95% bootstrap CIs, a permutation-test
significance comparison, and a figure -- everything reproducible from the seeds.
"""

from __future__ import annotations

import os

import hydra
import matplotlib
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phytonet.experiments import (
    compare, run_experiment, summarize, write_csv,
)


def _figure(summ, methods, scenarios, cmp, cfg, path):
    x = np.arange(len(scenarios))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(1.7 * len(scenarios) + 3, 5))
    for i, m in enumerate(methods):
        means = [summ.get((m, sc), (np.nan,) * 3)[0] for sc in scenarios]
        los = [summ.get((m, sc), (np.nan,) * 3)[1] for sc in scenarios]
        his = [summ.get((m, sc), (np.nan,) * 3)[2] for sc in scenarios]
        err = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
        ax.bar(x + (i - len(methods) / 2) * w, means, w, yerr=err, capsize=2, label=m)
    # significance stars for the configured comparison
    for j, sc in enumerate(scenarios):
        if sc in cmp and cmp[sc].get("p_holm", 1.0) < 0.05:
            top = max(summ.get((m, sc), (0,))[0] for m in methods)
            ax.text(j, top * 1.04, "*", ha="center", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylabel("outcome (biomass + reproduction), 95% CI")
    ax.set_title(f"PhytoNet — multi-seed benchmark ({cfg.seeds} seeds)  "
                 f"* = {cfg.compare.a} vs {cfg.compare.b}, p<0.05")
    ax.legend(ncol=len(methods), fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
    methods, scenarios = list(cfg.methods), list(cfg.scenarios)
    seeds = range(cfg.seeds)
    outdir = to_absolute_path(cfg.outdir)
    os.makedirs(outdir, exist_ok=True)

    print(f"Running {len(methods)} methods x {len(scenarios)} scenarios x "
          f"{cfg.seeds} seeds ...")
    records = run_experiment(methods, scenarios, seeds, days=cfg.days,
                             evolve_iters=cfg.evolve_iters, evolve_pop=cfg.evolve_pop)

    csv_path = os.path.join(outdir, "results.csv")
    write_csv(records, csv_path)
    summ = summarize(records)
    cmp = compare(records, cfg.compare.a, cfg.compare.b, scenarios)

    print("\nMean outcome [95% CI] by method x scenario:")
    print(f"  {'method':20s}" + "".join(f"{sc:>16s}" for sc in scenarios))
    for m in methods:
        cells = "".join(f"{summ[(m, sc)][0]:6.1f}[{summ[(m, sc)][1]:4.1f},"
                        f"{summ[(m, sc)][2]:4.1f}]" for sc in scenarios)
        print(f"  {m:20s}{cells}")
    print(f"\nSignificance ({cfg.compare.a} vs {cfg.compare.b}); "
          f"d = Cohen's d; p_holm = Holm-corrected for {len(cmp)} comparisons:")
    for sc in scenarios:
        if sc in cmp:
            sig = "  *" if cmp[sc]["p_holm"] < 0.05 else ""
            print(f"  {sc:12s} diff={cmp[sc]['diff']:+6.2f}  d={cmp[sc]['d']:+5.2f}  "
                  f"p={cmp[sc]['p']:.3f}  p_holm={cmp[sc]['p_holm']:.3f}{sig}")

    # A markdown results table for the dissertation.
    from phytonet.experiments import results_markdown
    table_path = os.path.join(outdir, "results_table.md")
    with open(table_path, "w") as f:
        f.write(f"# PhytoNet results (mean [95% CI]), {cfg.seeds} seeds\n\n")
        f.write(results_markdown(summ, methods, scenarios) + "\n")
    fig_path = os.path.join(outdir, "experiment.png")
    _figure(summ, methods, scenarios, cmp, cfg, fig_path)
    print(f"\nSaved {csv_path}, {fig_path}, and {table_path}")

    if cfg.wandb.enabled:
        try:
            import wandb
            wandb.init(project=cfg.wandb.project,
                       config=OmegaConf.to_container(cfg, resolve=True))
            for (m, sc), (mean, lo, hi) in summ.items():
                wandb.log({f"{m}/{sc}/mean": mean})
            wandb.save(csv_path)
            wandb.finish()
            print("Logged to Weights & Biases.")
        except ImportError:
            print("wandb not installed -- skipping (pip install wandb). CSV is the "
                  "reproducible source of truth.")


if __name__ == "__main__":
    main()
