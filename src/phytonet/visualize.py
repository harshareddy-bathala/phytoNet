"""Dimension-aware visualization front door.

Pick a renderer by ``world.dim`` so callers (scripts, notebooks) never have to.
2D -> ``viz``; 3D -> ``viz3d``. ``plot_resources`` is dimension-independent and
just re-exported from ``viz``.
"""

from __future__ import annotations

from phytonet import viz, viz3d
from phytonet.plant import Plant
from phytonet.world import World

plot_resources = viz.plot_resources


def _backend(world: World):
    if world.dim == 2:
        return viz
    if world.dim == 3:
        return viz3d
    raise NotImplementedError(
        f"No renderer for dim={world.dim}; only 2D and 3D are supported."
    )


def render_frame(world: World, plant: Plant, ax=None, title: str | None = None):
    backend = _backend(world)
    if title is None:
        return backend.render_frame(world, plant, ax)
    return backend.render_frame(world, plant, ax, title=title)


def animate(world: World, frames: list[Plant], **kwargs):
    """Animate snapshots with the dimension-appropriate backend.

    Extra kwargs (e.g. ``substeps``, ``fps``, ``interval_ms``) are forwarded;
    the 2D backend ignores ones it doesn't accept.
    """
    backend = _backend(world)
    if backend is viz:
        # 2D backend only knows interval_ms.
        return backend.animate(world, frames,
                               interval_ms=kwargs.get("interval_ms", 200))
    return backend.animate(world, frames, **kwargs)
