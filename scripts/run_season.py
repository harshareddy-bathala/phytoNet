"""Season entry point (2D or 3D): grow a plant and visualize it.

    python scripts/run_season.py                       # 2D, animated window + plot
    python scripts/run_season.py --dim 3               # 3D, matplotlib viewer
    python scripts/run_season.py --dim 3 --scenario drought   # test a condition
    python scripts/run_season.py --dim 3 --sun 4 4 9   # 3D with an off-centre sun
    python scripts/run_season.py --dim 3 --save-gif season3d.gif --no-show

The plant is driven by the hand-coded grow-toward-light baseline (gravitropism +
phototropism) -- no learning. Pick a world with --scenario (baseline, drought,
monsoon, shade, bright) or tune individual knobs.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from phytonet.config import build_simulation
from phytonet.scenarios import SCENARIOS, make_scenario
from phytonet.visualize import animate, plot_resources, render_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Grow a PhytoNet plant over a season.")
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3), help="2D or 3D")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="baseline",
                        help="environmental conditions to test the plant in")
    parser.add_argument("--days", type=int, default=60, help="season length in steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="growth RNG seed (default: random each run, for variety)")
    parser.add_argument("--germinated", action="store_true",
                        help="start as a young seedling instead of a lone seed")
    parser.add_argument("--signaling", action="store_true",
                        help="grow via the decentralized electrical-signaling "
                             "controller; 3D nodes are coloured by membrane potential")
    parser.add_argument("--light", type=float, help="override scenario light intensity")
    parser.add_argument("--water", type=float, help="override scenario water availability")
    parser.add_argument("--sun", type=float, nargs="+", metavar="C",
                        help="sun position (dim coords); default high & centred")
    parser.add_argument("--substeps", type=int, default=6,
                        help="3D: in-between frames per day (higher = smoother/slower)")
    parser.add_argument("--fps", type=int, default=15, help="frames/sec when saving a GIF")
    parser.add_argument("--save-gif", metavar="PATH", help="save the animation as a GIF")
    parser.add_argument("--save-plot", metavar="PATH", help="save the resource plot PNG")
    parser.add_argument("--save-final", metavar="PATH", help="save the final frame PNG")
    parser.add_argument("--no-show", action="store_true", help="don't open windows")
    args = parser.parse_args()

    if args.sun is not None and len(args.sun) != args.dim:
        parser.error(f"--sun needs {args.dim} coordinates for dim={args.dim}")

    # Random seed per run (unless pinned) so repeated runs grow different plants.
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2**31))
    cfg = make_scenario(args.scenario, dim=args.dim, days=args.days, seed=seed,
                        start_from_seed=not args.germinated)
    if args.light is not None:
        cfg.light_intensity = args.light
    if args.water is not None:
        cfg.water_availability = args.water
    if args.sun is not None:
        cfg.light_pos = tuple(args.sun)
    sim = build_simulation(cfg)
    if args.signaling:
        from phytonet.signaling import attach_signaling
        attach_signaling(sim)
    sim.run(cfg.days)

    print(f"[{cfg.dim}D | {args.scenario}] After {cfg.days} days: "
          f"{sim.history['n_nodes'][-1]} nodes, "
          f"{sim.history['n_leaves'][-1]} leaves, {sim.history['n_roots'][-1]} roots, "
          f"biomass {sim.history['biomass'][-1]:.1f}, "
          f"soil {sim.history['moisture'][-1] * 100:.0f}%")

    anim = None
    if args.save_gif or not args.no_show:
        # Auto-orbit only for GIFs (you can't drag a GIF); interactive = free camera.
        anim = animate(sim.world, sim.frames, substeps=args.substeps, fps=args.fps,
                       env_series=sim.env_series(), orbit=bool(args.save_gif),
                       color_by="V" if args.signaling else "kind")
    if args.save_gif:
        anim.save(args.save_gif, writer="pillow", fps=getattr(anim, "_phyto_fps", args.fps))
        print(f"Saved {args.save_gif}")

    if args.save_final:
        ax = render_frame(sim.world, sim.frames[-1], title=f"PhytoNet — day {sim.day}")
        ax.figure.savefig(args.save_final, dpi=120, bbox_inches="tight")
        print(f"Saved {args.save_final}")

    if args.save_plot:
        ax = plot_resources(sim.history)
        ax.figure.savefig(args.save_plot, dpi=120, bbox_inches="tight")
        print(f"Saved {args.save_plot}")

    if not args.no_show:
        plot_resources(sim.history)
        plt.show()


if __name__ == "__main__":
    main()
