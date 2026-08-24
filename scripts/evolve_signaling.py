"""Evolve the decentralized electrical-signaling rule with CEM, then benchmark it.

    python scripts/evolve_signaling.py
    python scripts/evolve_signaling.py --iterations 16 --save-curve evo.png

No backprop and no neural network: CEM tunes the ~9 shared parameters of the
interpretable cell rule so the *whole plant* discovers an electrical signaling
strategy that maximises fitness (biomass + reproduction, survival) across
scenarios. The plant learns what to signal -- with no brain.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from phytonet.signaling import SignalingParams, evolve_signaling, fitness

_SCENARIOS = ["baseline", "patchy", "drought", "shade", "lifecycle"]


def mean_fitness(params, seeds=range(3), days=60):
    return {sc: float(np.mean([fitness(params, sc, s, days) for s in seeds]))
            for sc in _SCENARIOS}


def main() -> None:
    p = argparse.ArgumentParser(description="Evolve the PhytoNet signaling rule.")
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--save-curve", metavar="PATH")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    hand = SignalingParams()
    print("Evolving the shared electrical rule with CEM (no network, no backprop)...")
    best, result = evolve_signaling(days=args.days, iterations=args.iterations,
                                    population=args.population)

    hand_fit = mean_fitness(hand)
    best_fit = mean_fitness(best)
    print("\nMean fitness by scenario (biomass + reproduction):")
    print(f"  {'scenario':12s}{'hand-designed':>15s}{'evolved':>12s}")
    for sc in _SCENARIOS:
        print(f"  {sc:12s}{hand_fit[sc]:15.1f}{best_fit[sc]:12.1f}")
    print(f"  {'OVERALL':12s}{np.mean(list(hand_fit.values())):15.1f}"
          f"{np.mean(list(best_fit.values())):12.1f}")

    print("\nEvolved rule parameters:")
    names = ["leak", "coupling", "threshold", "ap_strength", "k_light",
             "k_water", "grow_thresh", "thicken_thresh", "steer_gain"]
    for n, v in zip(names, best.to_vector()):
        print(f"  {n:14s}: {v:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(result.history) + 1), result.history, marker="o",
            color="#7a3fb0")
    ax.set_xlabel("CEM iteration")
    ax.set_ylabel("mean fitness across scenarios")
    ax.set_title("PhytoNet — evolving the electrical signaling rule")
    ax.grid(alpha=0.3)
    if args.save_curve:
        fig.savefig(args.save_curve, dpi=120, bbox_inches="tight")
        print(f"\nSaved {args.save_curve}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
