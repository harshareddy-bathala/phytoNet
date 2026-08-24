"""Validate that the simulator reproduces known plant biology it was *not*
explicitly programmed to do -- evidence that the abstraction is reasonable.

The headline check is the **functional-equilibrium** (balanced-growth) response:
real plants shift biomass toward the organ that captures the limiting resource —
more roots when water-limited (drought), more shoots when light-limited (shade).
If the model reproduces this from physiology + resource costs alone, that is
emergent validation, not a hard-coded rule.
"""

from __future__ import annotations

import numpy as np

from phytonet.config import build_simulation
from phytonet.plant import NodeKind
from phytonet.scenarios import make_scenario

_SHOOT = (NodeKind.STEM, NodeKind.LEAF, NodeKind.SEED)


def root_shoot_ratio(plant) -> float:
    root = sum(n.state["size"] for n in plant.nodes_of(NodeKind.ROOT))
    shoot = sum(n.state["size"] for n in plant.nodes.values() if n.kind in _SHOOT)
    return root / shoot if shoot > 0 else float("nan")


def morphometrics(scenario: str, attach=None, seed: int = 0, days: int = 70) -> dict:
    """Grow a plant and measure its morphology."""
    sim = build_simulation(make_scenario(scenario, days=days, seed=seed,
                                         record_frames=False))
    if attach is not None:
        attach(sim)
    sim.run(days)
    up = sim.world.up_axis
    heights = [n.pos[up] for n in sim.plant.nodes.values()]
    return {
        "root_shoot": root_shoot_ratio(sim.plant),
        "height": float(max(heights) - min(heights)),
        "n_leaves": len(sim.plant.nodes_of(NodeKind.LEAF)),
        "n_roots": len(sim.plant.nodes_of(NodeKind.ROOT)),
        "biomass": float(sim.history["biomass"][-1]),
    }


def functional_equilibrium(seeds=range(5), days: int = 70, attach=None,
                           scenarios=("baseline", "drought", "shade")) -> dict:
    """Mean root:shoot ratio per scenario (± std). Functional equilibrium predicts
    ``drought > baseline > shade`` (roots up when water-limited, down in shade)."""
    out = {}
    for sc in scenarios:
        vals = [morphometrics(sc, attach, s, days)["root_shoot"] for s in seeds]
        out[sc] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def reproduces_functional_equilibrium(seeds=range(5), days: int = 70, attach=None) -> bool:
    """True if the model shows the predicted ordering (more roots under drought
    than under shade)."""
    fe = functional_equilibrium(seeds, days, attach)
    return fe["drought"][0] > fe["shade"][0]
