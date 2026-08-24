"""Characterise the wound->defense result (the project's one significant win).

Turns a single p-value into a characterised phenomenon:
  * a **dose-response** curve -- biomass retained vs herbivory pressure, with the
    electrical defense ON vs OFF;
  * an **ablation** + **effect size** (Cohen's d) of the defense benefit;
all multi-seed with 95% bootstrap CIs.

    python scripts/herbivory_study.py --outdir figures --seeds 8
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from phytonet.config import build_simulation
from phytonet.experiments import bootstrap_ci, cohens_d, permutation_test
from phytonet.scenarios import make_scenario
from phytonet.signaling import SignalingParams, attach_signaling


def _final_biomass(intensity, defense_on, seed, days=80):
    params = SignalingParams()
    if not defense_on:
        params.defense_gain = 0.0          # ablate the defense response
    cfg = make_scenario("herbivory", days=days, seed=seed)
    cfg.herbivory_intensity = intensity
    sim = build_simulation(cfg)
    attach_signaling(sim, params)
    sim.run(days)
    return sim.history["biomass"][-1]


def main() -> None:
    p = argparse.ArgumentParser(description="Herbivory wound->defense study.")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--days", type=int, default=80)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    seeds = range(args.seeds)

    intensities = [0.0, 0.3, 0.5, 0.7, 0.9]
    on_m, on_lo, on_hi, off_m, off_lo, off_hi = ([] for _ in range(6))
    print("Dose-response: final biomass vs herbivory intensity (defense on/off)\n")
    print(f"  {'intensity':>9s}{'defense ON':>16s}{'defense OFF':>16s}{'d':>7s}{'p':>8s}")
    for it in intensities:
        on = np.array([_final_biomass(it, True, s, args.days) for s in seeds])
        off = np.array([_final_biomass(it, False, s, args.days) for s in seeds])
        m, lo, hi = bootstrap_ci(on); on_m.append(m); on_lo.append(lo); on_hi.append(hi)
        m, lo, hi = bootstrap_ci(off); off_m.append(m); off_lo.append(lo); off_hi.append(hi)
        d = cohens_d(on, off); pv = permutation_test(on, off)
        print(f"  {it:9.1f}{on.mean():16.1f}{off.mean():16.1f}{d:7.2f}{pv:8.3f}")

    # Headline number at the design pressure.
    on = np.array([_final_biomass(0.5, True, s, args.days) for s in seeds])
    off = np.array([_final_biomass(0.5, False, s, args.days) for s in seeds])
    print(f"\n  At intensity 0.5: defense benefit = {on.mean()-off.mean():+.2f} biomass, "
          f"Cohen's d = {cohens_d(on, off):.2f}, p = {permutation_test(on, off):.3f}")

    x = np.array(intensities)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.fill_between(x, on_lo, on_hi, color="#2f9e44", alpha=0.2)
    ax.plot(x, on_m, "o-", color="#2f9e44", label="defense ON (wound signal)")
    ax.fill_between(x, off_lo, off_hi, color="#c0392b", alpha=0.2)
    ax.plot(x, off_m, "s--", color="#c0392b", label="defense OFF")
    ax.set_xlabel("herbivory intensity (biomass per bite)")
    ax.set_ylabel("final biomass")
    ax.set_title("Wound->defense dose-response (95% CI, "
                 f"{args.seeds} seeds)")
    ax.legend(); ax.grid(alpha=0.3)
    path = os.path.join(args.outdir, "herbivory_dose_response.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved {path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
