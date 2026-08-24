"""Visualization layer (2D, matplotlib).

This is the *only* module that assumes a specific dimensionality. The core
model is N-dimensional; rendering is where we commit to a view. To support 3D
later, add a ``viz3d.py`` (e.g. Polyscope/Open3D) with the same ``render_frame``
signature -- nothing in the simulation needs to change.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes

from phytonet.plant import NodeKind, Plant
from phytonet.world import World

# Colors per tissue kind. Tweak freely -- this is presentation only.
_KIND_STYLE = {
    NodeKind.SEED: {"color": "#6b3f1d", "s": 90, "label": "seed"},
    NodeKind.ROOT: {"color": "#8a5a2b", "s": 50, "label": "root"},
    NodeKind.STEM: {"color": "#3a7d34", "s": 50, "label": "stem"},
    NodeKind.LEAF: {"color": "#4caf50", "s": 110, "label": "leaf"},
}


def render_frame(
    world: World, plant: Plant, ax: Axes | None = None, title: str = "PhytoNet seedling"
) -> Axes:
    """Draw the world (sky, soil, sun) and the plant graph onto ``ax``."""
    if world.dim != 2:
        raise NotImplementedError(
            f"render_frame draws 2D worlds; got dim={world.dim}. "
            "Add a 3D renderer (viz3d.py) for higher dimensions."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 7))

    lo, hi = world.bounds_low, world.bounds_high
    g = world.ground_level

    # Sky and soil bands.
    ax.axhspan(g, hi[1], facecolor="#cfe8ff", zorder=0)          # air
    ax.axhspan(lo[1], g, facecolor="#e8d8b0", zorder=0)          # soil
    ax.axhline(g, color="#7a6a3a", lw=1.2, zorder=1)             # ground line

    # Sun marker at the world's light position.
    sun = world.light_pos
    ax.scatter([sun[0]], [sun[1]], s=420, color="#ffd23f",
               edgecolors="#f0a000", zorder=2, label="light")

    # Vascular edges first, so nodes sit on top.
    for a, b in plant.edges:
        pa, pb = plant.nodes[a].pos, plant.nodes[b].pos
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#3a7d34", lw=2, zorder=3)

    # Nodes, grouped by kind so the legend has one entry each. Marker area scales
    # with tissue `size` so thickening (Step 3) is visible as fattening nodes.
    seen: set[NodeKind] = set()
    for node in plant.nodes.values():
        style = _KIND_STYLE[node.kind]
        label = style["label"] if node.kind not in seen else None
        seen.add(node.kind)
        ax.scatter([node.pos[0]], [node.pos[1]], color=style["color"],
                   s=style["s"] * node.state["size"], edgecolors="black",
                   linewidths=0.5, zorder=4, label=label)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("height")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    return ax


def animate(world: World, frames: list[Plant], interval_ms: int = 200) -> FuncAnimation:
    """Animate a list of plant snapshots (one per day) over the world.

    Returns the ``FuncAnimation`` so the caller can ``.save(...)`` or ``plt.show()``.
    """
    fig, ax = plt.subplots(figsize=(6, 7))

    def update(i: int):
        ax.clear()
        render_frame(world, frames[i], ax, title=f"PhytoNet — day {i}")
        return ax.collections + ax.lines

    return FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=False)


def plot_resources(history: dict, ax: Axes | None = None) -> Axes:
    """Plot biomass / node count / pooled sugar & water against day number."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    day = history["day"]
    ax.plot(day, history["biomass"], label="biomass (nodes)", color="#3a7d34", lw=2)
    ax.plot(day, history["n_leaves"], label="# leaves", color="#4caf50", lw=1.5, ls="--")
    ax.plot(day, history["total_sugar"], label="pooled sugar", color="#e0a800", lw=1.5)
    ax.plot(day, history["total_water"], label="pooled water", color="#1f77b4", lw=1.5)
    ax.set_xlabel("day")
    ax.set_ylabel("amount")
    ax.set_title("PhytoNet — resources over the season")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    return ax
