"""Resource physiology: how tissues make, share, and spend resources each step.

The MVP rules, in the order they run each step:

1. **Photosynthesis** -- leaves turn local light into sugar.
2. **Water uptake** -- roots turn local water into water-credit.
3. **Transport** -- sugar and water diffuse along vascular edges toward
   equalization. This is the hand-coded stand-in for plant *signaling /
   source-sink flow*; the GNN will later replace it with learned messages.
4. **Maintenance** -- every tissue burns a little sugar to stay alive.

Growth itself (spending resources to extend) lives in ``policy.py`` -- it is an
*action*, not physiology, so the learned policy can later take it over.

Units are arbitrary and consistent within the sim. Everything mutates node
``state`` in place; nothing here knows about dimensionality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phytonet.plant import NodeKind, Plant
from phytonet.world import World


@dataclass
class PhysiologyParams:
    photo_rate: float = 0.6        # sugar per unit light per leaf size per step
    uptake_rate: float = 0.6       # water per unit water-field per root size per step
    maintenance_rate: float = 0.02  # sugar burned per unit size per step
    diffusion_rate: float = 0.25   # fraction of a neighbour gap moved per step (0..0.5)
    # Feedback inhibition: production slows as a tissue's own load approaches a
    # cap, and stops at it. This is "sink limitation" -- a real phenomenon where
    # sugar accumulation down-regulates photosynthesis -- and it keeps pooled
    # resources bounded instead of accumulating forever.
    sugar_cap: float = 6.0         # leaf sugar load at which photosynthesis halts
    water_cap: float = 6.0         # root water load at which uptake halts
    # Self-shading: a leaf with many leaves above it gets less light. This makes
    # growing tall / spreading the canopy worthwhile, so "invest in shoots" is a
    # real strategy (and the right one when light is the bottleneck).
    shading: float = 0.8           # canopy attenuation strength (0 = no shading)


def _feedback(load: float, cap: float) -> float:
    """Linear down-regulation: full rate at load 0, zero at/above the cap."""
    return max(0.0, 1.0 - load / cap)


def photosynthesize(plant: Plant, world: World, p: PhysiologyParams) -> None:
    leaves = plant.nodes_of(NodeKind.LEAF)
    if not leaves:
        return
    up = world.up_axis
    heights = np.array([leaf.pos[up] for leaf in leaves])
    n = len(leaves)
    for leaf in leaves:
        # Exposure: fraction of leaves strictly above this one shades it.
        frac_above = float(np.sum(heights > leaf.pos[up])) / n
        exposure = np.exp(-p.shading * frac_above)
        gain = world.light_at(leaf.pos) * leaf.state["size"] * p.photo_rate * exposure
        leaf.state["sugar"] += gain * _feedback(leaf.state["sugar"], p.sugar_cap)


def uptake(plant: Plant, world: World, p: PhysiologyParams) -> None:
    drawn = 0.0
    for root in plant.nodes_of(NodeKind.ROOT):
        gain = world.water_at(root.pos) * root.state["size"] * p.uptake_rate
        gain *= _feedback(root.state["water"], p.water_cap)
        root.state["water"] += gain
        drawn += gain
    # Water moved from soil into the plant -- drain the reservoir (no-op if it
    # is the infinite default).
    world.deplete(drawn)


def maintain(plant: Plant, p: PhysiologyParams) -> None:
    for node in plant.nodes.values():
        cost = node.state["size"] * p.maintenance_rate
        node.state["sugar"] = max(0.0, node.state["sugar"] - cost)


def _diffuse(plant: Plant, key: str, rate: float) -> None:
    """Move ``rate`` of each edge's gap from the richer node to the poorer one.

    Flows are computed from the pre-step state and then applied, so the result
    is order-independent and conserves the total of ``key`` exactly.
    """
    deltas: dict[int, float] = {i: 0.0 for i in plant.nodes}
    for a, b in plant.edges:
        gap = plant.nodes[a].state[key] - plant.nodes[b].state[key]
        flow = rate * gap
        deltas[a] -= flow
        deltas[b] += flow
    for i, d in deltas.items():
        plant.nodes[i].state[key] += d


def transport(plant: Plant, p: PhysiologyParams) -> None:
    _diffuse(plant, "sugar", p.diffusion_rate)
    _diffuse(plant, "water", p.diffusion_rate)


def physiology_step(plant: Plant, world: World, p: PhysiologyParams) -> None:
    """Run one full physiology update (production -> transport -> maintenance)."""
    photosynthesize(plant, world, p)
    uptake(plant, world, p)
    transport(plant, p)
    maintain(plant, p)
