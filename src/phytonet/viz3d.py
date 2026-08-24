"""3D visualization (matplotlib mplot3d), tuned to look plant-like and animate
smoothly.

Two things make it read as a growing plant rather than a jumpy graph:

* **Tapered branches.** Edges are drawn with a line width that follows tissue
  ``size``, so the thickened base looks woody and twigs look thin. Leaves are
  translucent green discs.
* **Interpolated growth.** Between consecutive daily snapshots we insert
  ``substeps`` in-between frames: existing tissue eases toward its new position,
  and brand-new tissue *grows out* from its parent (size 0 -> full). This both
  slows the apparent passage of time and removes the popping.

Same public interface as ``viz.py`` (``render_frame``, ``animate``) so the
``visualize`` dispatcher can pick by dimension. A heavier viewer
(Polyscope/Open3D) could replace this module behind the same interface.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from phytonet.plant import Node, NodeKind, Plant
from phytonet.world import World

_LEAF_COLOR = "#5bbf52"
_STEM_COLOR = "#4a7d34"
_ROOT_COLOR = "#7a5230"
_SEED_COLOR = "#5a3a1a"


def _new_axes():
    fig = plt.figure(figsize=(7.5, 8))
    ax = fig.add_subplot(projection="3d")
    return fig, ax


# -- realistic-ish drawing -------------------------------------------------
def _edge_color(ka: NodeKind, kb: NodeKind) -> str:
    if NodeKind.ROOT in (ka, kb) or NodeKind.SEED in (ka, kb):
        return _ROOT_COLOR
    if NodeKind.LEAF in (ka, kb):
        return _LEAF_COLOR
    return _STEM_COLOR


def _soil_color(moisture: float):
    """Lerp dry tan -> wet brown so the ground visibly darkens when watered."""
    dry = np.array([0.85, 0.77, 0.59])   # #d9c596
    wet = np.array([0.42, 0.31, 0.17])   # #6b4f2a
    m = float(np.clip(moisture, 0.0, 1.0))
    return tuple((1 - m) * dry + m * wet)


def _draw_rain(ax, world: World) -> None:
    """A scatter of short blue dashes in the air to signal a watering/rain day."""
    lo, hi = world.bounds_low, world.bounds_high
    rng = np.random.default_rng(0)
    n = 40
    x = rng.uniform(lo[0], hi[0], n)
    y = rng.uniform(lo[1], hi[1], n)
    z = rng.uniform(world.ground_level, hi[2], n)
    ax.scatter(x, y, z, s=8, marker="|", color="#2a6fdb", alpha=0.6)


def _draw_nodes_by_potential(ax, plant: Plant, vmax: float = 2.5) -> None:
    """Colour every tissue by its membrane potential V (electrical state), so the
    propagating signals are visible: cool = at rest, hot = depolarised/firing."""
    nodes = list(plant.nodes.values())
    if not nodes:
        return
    pos = np.array([n.pos for n in nodes])
    V = np.array([n.state.get("V", 0.0) for n in nodes])
    sizes = np.array([40 * n.state["size"] for n in nodes])
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=V, cmap="plasma",
                    vmin=0.0, vmax=vmax, s=sizes, edgecolors="black",
                    linewidths=0.4, label="membrane V")
    return sc


def _draw_plant(ax, world: World, plant: Plant,
                moisture: float | None = None, raining: bool = False,
                color_by: str = "kind") -> None:
    lo, hi = world.bounds_low, world.bounds_high
    g = world.ground_level
    if moisture is None:
        moisture = world.moisture()

    # Sky-tinted panes + ground plane whose colour tracks soil moisture.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.82, 0.91, 1.0, 0.25))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax.grid(False)
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 2), np.linspace(lo[1], hi[1], 2))
    ax.plot_surface(xx, yy, np.full_like(xx, g), color=_soil_color(moisture), alpha=0.55)

    if raining:
        _draw_rain(ax, world)

    # Sun.
    sun = world.light_pos
    ax.scatter([sun[0]], [sun[1]], [sun[2]], s=500, color="#ffd23f",
               edgecolors="#f0a000", linewidths=1.0, label="light")

    # Branches as tapered tubes: one segment per edge, width follows tissue size.
    segments, widths, colors = [], [], []
    for a, b in plant.edges:
        na, nb = plant.nodes[a], plant.nodes[b]
        segments.append([na.pos, nb.pos])
        avg_size = 0.5 * (na.state["size"] + nb.state["size"])
        widths.append(1.2 + 2.2 * avg_size)
        colors.append(_edge_color(na.kind, nb.kind))
    if segments:
        ax.add_collection3d(Line3DCollection(segments, linewidths=widths,
                                             colors=colors, capstyle="round"))

    if color_by == "V":
        # Colour every tissue by membrane potential -> watch the signals travel.
        _draw_nodes_by_potential(ax, plant)
    else:
        # Leaves: prominent translucent discs. Other tissue: small joint dots.
        leaves = plant.nodes_of(NodeKind.LEAF)
        if leaves:
            lp = np.array([n.pos for n in leaves])
            ls = np.array([110 * n.state["size"] for n in leaves])
            ax.scatter(lp[:, 0], lp[:, 1], lp[:, 2], s=ls, color=_LEAF_COLOR,
                       alpha=0.85, edgecolors="#2f6b2a", linewidths=0.5, label="leaf")
        others = [n for n in plant.nodes.values() if n.kind != NodeKind.LEAF]
        if others:
            op = np.array([n.pos for n in others])
            oc = [_ROOT_COLOR if n.kind in (NodeKind.ROOT, NodeKind.SEED)
                  else _STEM_COLOR for n in others]
            os_ = [22 * n.state["size"] for n in others]
            ax.scatter(op[:, 0], op[:, 1], op[:, 2], s=os_, color=oc, alpha=0.9)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("height")


def render_frame(world: World, plant: Plant, ax=None, title: str = "PhytoNet (3D)",
                 moisture: float | None = None, raining: bool = False,
                 color_by: str = "kind"):
    if world.dim != 3:
        raise NotImplementedError(f"viz3d renders 3D worlds; got dim={world.dim}.")
    if ax is None:
        _, ax = _new_axes()
    _draw_plant(ax, world, plant, moisture=moisture, raining=raining, color_by=color_by)
    ax.set_title(title)
    ax.legend(loc="upper right")
    return ax


# -- smooth interpolation --------------------------------------------------
def _parent_in(prev: Plant, cur: Plant, nid: int) -> int | None:
    """The neighbour a new node grew out of (its parent present in ``prev``)."""
    for nb in cur.neighbors(nid):
        if nb in prev.nodes:
            return nb
    return None


def _lerp_plant(prev: Plant, cur: Plant, t: float) -> Plant:
    """A snapshot ``t`` of the way from ``prev`` to ``cur``.

    Nodes present in both ease between their states; nodes new in ``cur`` grow
    out from their parent's old position with size easing from 0.
    """
    smooth = t * t * (3 - 2 * t)  # smoothstep for gentle starts/stops
    out = Plant(dim=cur.dim)
    for nid, nc in cur.nodes.items():
        if nid in prev.nodes:
            npv = prev.nodes[nid]
            pos = (1 - t) * npv.pos + t * nc.pos
            size = (1 - t) * npv.state["size"] + t * nc.state["size"]
        else:
            parent = _parent_in(prev, cur, nid)
            start = prev.nodes[parent].pos if parent is not None else nc.pos
            pos = (1 - smooth) * start + smooth * nc.pos
            size = smooth * nc.state["size"]
        node = Node(id=nid, kind=nc.kind, pos=pos)
        node.state["size"] = max(size, 1e-3)
        node.state["V"] = nc.state.get("V", 0.0)  # carry electrical state for V-colouring
        out.nodes[nid] = node
    for a, b in cur.edges:
        if a in out.nodes and b in out.nodes:
            out.edges.add((a, b))
    return out


def _interpolate(frames: list[Plant], substeps: int) -> list[tuple[Plant, float]]:
    """Expand daily snapshots into (plant, day) pairs with in-between frames."""
    if substeps <= 1:
        return [(p, float(i)) for i, p in enumerate(frames)]
    out: list[tuple[Plant, float]] = []
    for i in range(len(frames) - 1):
        for s in range(substeps):
            t = s / substeps
            out.append((_lerp_plant(frames[i], frames[i + 1], t), i + t))
    out.append((frames[-1], float(len(frames) - 1)))
    return out


def animate(world: World, frames: list[Plant], interval_ms: int = 70,
            substeps: int = 6, fps: int = 15,
            env_series: list[dict] | None = None, orbit: bool = False,
            color_by: str = "kind") -> FuncAnimation:
    """Animate daily snapshots smoothly, with weather and a HUD.

    Plays through the season once and then **holds the final frame** (``repeat=
    False``) instead of looping back to day 0. The camera is **not** driven by the
    animation, so you can drag to rotate/zoom freely while it plays and after it
    finishes (your view is preserved across frames). Pass ``orbit=True`` for a
    slow auto-rotation (useful when saving a GIF, where you can't drag).

    ``env_series`` (one dict per day) supplies per-day soil moisture / rain.
    """
    fig, ax = _new_axes()
    states = _interpolate(frames, substeps)
    n = len(states)
    view = {"elev": 22.0, "azim": -70.0}  # remembered camera (user can change it)

    def env_at(day: float) -> dict:
        if not env_series:
            return {"moisture": world.moisture(), "raining": False}
        return env_series[min(int(round(day)), len(env_series) - 1)]

    def update(i: int):
        # Remember the user's current camera before clearing (clear resets it).
        if i > 0:
            view["elev"], view["azim"] = ax.elev, ax.azim
        ax.clear()
        plant, day = states[i]
        e = env_at(day)
        render_frame(world, plant, ax, title="PhytoNet — growing season",
                     moisture=e["moisture"], raining=e["raining"], color_by=color_by)
        biomass = sum(nd.state["size"] for nd in plant.nodes.values())
        done_tag = "  [end - paused]" if i == n - 1 else ""
        rain_tag = "  rain" if e["raining"] else ""
        ax.text2D(0.03, 0.91,
                  f"day {int(round(day))}\nbiomass {biomass:.1f}\n"
                  f"soil {e['moisture'] * 100:3.0f}%{rain_tag}{done_tag}",
                  transform=ax.transAxes, fontsize=11, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        if orbit:  # auto-rotate (for GIFs); otherwise keep the user's view
            view["azim"] = -70 + 35 * (i / max(n - 1, 1))
        ax.view_init(elev=view["elev"], azim=view["azim"])
        return []

    anim = FuncAnimation(fig, update, frames=n, interval=interval_ms,
                         blit=False, repeat=False)
    anim._phyto_fps = fps  # picked up by callers that save
    return anim
