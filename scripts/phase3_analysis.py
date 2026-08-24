"""Phase 3: compare PhytoNet's learned signalling to real plant signalling.

Runs the controlled wound experiment and produces one figure:
  * the **systemic defense wave** (defense vs day, by hop-distance from the wound),
  * **latency vs distance** -> propagation speed,
  * **directionality** (shoot vs root response),
and prints a side-by-side comparison with real plant electrical signalling.

    python scripts/phase3_analysis.py --outdir figures
    python scripts/phase3_analysis.py --learned   # use the Phase-2 learned rule
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from phytonet.signal_analysis import (
    calibrate_speed, defense_by_distance, directionality, propagation_speed,
    wound_experiment,
)


def main() -> None:
    p = argparse.ArgumentParser(description="PhytoNet Phase-3 signalling analysis.")
    p.add_argument("--learned", action="store_true",
                   help="analyse the learned-content rule instead of the hand rule")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grow-days", type=int, default=45)
    p.add_argument("--observe-days", type=int, default=18)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.learned:
        from phytonet.learned_signaling import (
            LearnedSignalingParams, attach_learned_signaling)
        attach_fn, params, label = (attach_learned_signaling,
                                    LearnedSignalingParams.default(), "learned")
    else:
        from phytonet.signaling import SignalingParams, attach_signaling
        attach_fn, params, label = attach_signaling, SignalingParams(), "hand-designed"

    res = wound_experiment(attach_fn, params, scenario="baseline", seed=args.seed,
                           grow_days=args.grow_days, observe_days=args.observe_days,
                           wound=6.0)
    if res is None:
        print("no leaves to wound; try a longer --grow-days")
        return

    mat, dists = defense_by_distance(res)
    sp = propagation_speed(res)
    dr = directionality(res)
    days = np.arange(mat.shape[0])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(dists)))
    for di, d in enumerate(dists):
        ax1.plot(days, mat[:, di], color=cmap[di], label=f"{d} hops")
    ax1.set_xlabel("days after wound"); ax1.set_ylabel("mean defense")
    ax1.set_title("Systemic defense wave"); ax1.legend(fontsize=7, ncol=2)

    lat = sp["latency"]
    if lat:
        xs = sorted(lat); ys = [lat[d] for d in xs]
        ax2.plot(ys, xs, "o-", color="#7a3fb0")
    ax2.set_xlabel("latency (days to arm)"); ax2.set_ylabel("hop-distance from wound")
    ax2.set_title(f"Propagation\n~{sp['speed_hops_per_step']:.2f} hops/step")

    ax3.bar(["shoot", "root"], [dr["shoot_peak_defense"], dr["root_peak_defense"]],
            color=["#4a7d34", "#7a5230"])
    ax3.set_ylabel("peak defense"); ax3.set_title("Directionality")

    fig.suptitle(f"PhytoNet Phase 3 — wound signalling ({label} rule) vs real plants")
    fig.tight_layout()
    path = os.path.join(args.outdir, "phase3_signal.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved {path}")

    print("\n=== Learned signalling vs real plant signalling ===")
    print(f"  systemic spread : a single leaf wound arms defense out to "
          f"{max(sp['latency']) if sp['latency'] else 0} hops "
          f"(real plants: wound -> systemic defense in distant leaves)")
    cal = calibrate_speed(sp["speed_hops_per_step"] if sp["speed_hops_per_step"] == sp["speed_hops_per_step"] else 0.0)
    print(f"  finite speed    : ~{sp['speed_hops_per_step']:.2f} hops/step -> "
          f"~{cal['mm_per_s']:.2f} mm/s (assuming {cal['assumptions']['internode_mm']}mm "
          f"internodes, {cal['assumptions']['step_seconds']}s/step); real plant "
          f"action/variation potentials are ~0.1-40 mm/s -> in range: {cal['in_range']}")
    print(f"  directionality  : shoot {dr['shoot_peak_defense']:.2f} vs root "
          f"{dr['root_peak_defense']:.2f} -- a leaf wound favours foliar defense "
          f"(real plants: shoot-borne systemic wound signalling)")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
