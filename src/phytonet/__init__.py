"""PhytoNet: a simulated plant agent that learns its own internal signaling.

MVP stage: a plant *body* (a graph of tissues) living in a world with light and
soil water. No learning yet -- growth will be driven by a hand-coded baseline
policy in a later step.

Design rule for dimensionality
------------------------------
The core model (``world``, ``plant``, and later ``physiology``/``simulation``)
is written in N dimensions: positions are NumPy vectors of length ``world.dim``
and nothing hardcodes "2". Only the visualization layer assumes a specific
dimensionality. Migrating to 3D means constructing the world with ``dim=3`` and
adding a 3D renderer -- the simulation logic is untouched.
"""

from phytonet.plant import NodeKind, Plant, make_seedling
from phytonet.world import World

__all__ = ["World", "Plant", "NodeKind", "make_seedling"]
