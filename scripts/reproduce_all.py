"""Regenerate every figure/result in the project from scratch.

    python scripts/reproduce_all.py            # fast figures (no neural training)
    python scripts/reproduce_all.py --full     # also the torch-heavy benchmarks

Each step is a separate script invocation, so a failure is isolated and reported.
Everything is deterministic given the seeds; outputs land in figures/ and results/.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PY = sys.executable

FAST = [
    ("validation (functional equilibrium)", ["scripts/validate.py", "--no-show"]),
    ("plant model + growth + strategy", ["scripts/make_figures.py", "--outdir", "figures"]),
    ("herbivory dose-response", ["scripts/herbivory_study.py", "--seeds", "8", "--no-show"]),
    ("controller comparison (side by side)", ["scripts/compare.py", "--no-show"]),
    ("learned-signal analysis", ["scripts/analyze_signaling.py", "--outdir", "figures"]),
    ("signalling vs real plants (Phase 3)", ["scripts/phase3_analysis.py",
                                             "--outdir", "figures", "--no-show"]),
    ("CEM learning curve", ["scripts/train_cem.py", "--iterations", "10", "--no-show",
                            "--save-curve", "figures/cem_curve.png"]),
]

FULL = [
    ("baseline-fairness audit (PPO)", ["scripts/baseline_audit.py",
                                       "--budgets", "3000", "10000", "25000"]),
    ("cross-paradigm benchmark", ["scripts/benchmark_all.py", "--no-show"]),
    ("paper experiment (multi-seed)", ["scripts/run_experiment.py", "seeds=5"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate all PhytoNet outputs.")
    ap.add_argument("--full", action="store_true",
                    help="also run the torch-heavy training benchmarks (slow)")
    args = ap.parse_args()
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    env = {**os.environ, "MPLBACKEND": "Agg"}

    steps = FAST + (FULL if args.full else [])
    ok = 0
    for name, cmd in steps:
        print(f"\n=== {name} ===\n    {' '.join(cmd)}")
        t0 = time.time()
        r = subprocess.run([PY, *cmd], env=env, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"    OK ({time.time() - t0:.0f}s)")
            ok += 1
        else:
            print(f"    FAILED (exit {r.returncode})\n{r.stdout[-400:]}\n{r.stderr[-400:]}")
    print(f"\n{ok}/{len(steps)} steps succeeded. Figures in figures/, results in results/.")
    if not args.full:
        print("(Run with --full to also regenerate the PPO/GNN benchmarks.)")


if __name__ == "__main__":
    main()
