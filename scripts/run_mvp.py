"""Step 1 entry point: build a world + seedling and render one static frame.

Run from the project root after installing (``pip install -e .``):

    python scripts/run_mvp.py            # opens a window
    python scripts/run_mvp.py --save out.png   # also writes a PNG

No growth yet -- this just confirms the world/plant representation looks right.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from phytonet import World, make_seedling
from phytonet.viz import render_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a PhytoNet seedling.")
    parser.add_argument("--save", metavar="PATH", help="save the frame to this path")
    parser.add_argument("--no-show", action="store_true", help="don't open a window")
    args = parser.parse_args()

    world = World(dim=2)
    plant = make_seedling(world)

    print(f"World: dim={world.dim}, ground={world.ground_level}, "
          f"bounds={world.bounds_low.tolist()}..{world.bounds_high.tolist()}")
    print(f"Plant: {len(plant.nodes)} nodes, {len(plant.edges)} edges, "
          f"{len(plant.meristems())} growing tips")

    ax = render_frame(world, plant)
    if args.save:
        ax.figure.savefig(args.save, dpi=120, bbox_inches="tight")
        print(f"Saved {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
