"""The plant body: a graph of tissues.

A ``Plant`` is a set of nodes (tissues) connected by undirected edges (vascular
links). This is the substrate everything else builds on:

* the hand-coded policy (next step) reads node positions/kinds and grows tips;
* the physiology (next step) reads/writes node ``state`` (sugar, water, size);
* the GNN (much later) consumes exactly this graph -- the field names here are
  chosen so converting to a ``torch_geometric.data.Data`` object is mechanical:
  node features from ``state``, ``edge_index`` from ``edges``.

Positions are length-``dim`` NumPy vectors, so nothing here is tied to 2D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class NodeKind(str, Enum):
    """The type of a tissue. ``str`` mixin keeps it readable when printed/saved."""

    SEED = "seed"
    ROOT = "root"
    STEM = "stem"
    LEAF = "leaf"


def _default_state() -> dict[str, float]:
    # Resource bookkeeping lives here so physiology can mutate it in place.
    return {"sugar": 0.0, "water": 0.0, "size": 1.0, "age": 0.0}


@dataclass
class Node:
    """A single tissue.

    Attributes
    ----------
    id:
        Stable integer index, unique within a plant.
    kind:
        What kind of tissue this is (see :class:`NodeKind`).
    pos:
        Position as a length-``dim`` vector.
    state:
        Mutable resource/age bookkeeping. Kept as a dict (not fixed fields) so
        we can add quantities later without touching every call site.
    meristematic:
        Whether this node is an active growing tip. Tips are where new tissue
        is added by the growth policy.
    """

    id: int
    kind: NodeKind
    pos: np.ndarray
    state: dict[str, float] = field(default_factory=_default_state)
    meristematic: bool = False

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=float)


@dataclass
class Plant:
    """A graph of tissues. Nodes are stored by id; edges are unordered pairs."""

    dim: int
    nodes: dict[int, Node] = field(default_factory=dict)
    edges: set[tuple[int, int]] = field(default_factory=set)
    _next_id: int = 0

    def add_node(
        self,
        kind: NodeKind,
        pos,
        *,
        meristematic: bool = False,
        state: dict[str, float] | None = None,
    ) -> Node:
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (self.dim,):
            raise ValueError(f"pos must have shape ({self.dim},), got {pos.shape}")
        node = Node(
            id=self._next_id,
            kind=kind,
            pos=pos,
            meristematic=meristematic,
            state=state if state is not None else _default_state(),
        )
        self.nodes[node.id] = node
        self._next_id += 1
        return node

    def add_edge(self, a: int, b: int) -> None:
        if a == b:
            raise ValueError("a node cannot connect to itself")
        if a not in self.nodes or b not in self.nodes:
            raise KeyError(f"both nodes must exist before linking ({a}, {b})")
        self.edges.add((a, b) if a < b else (b, a))

    def remove_node(self, node_id: int) -> None:
        """Remove a node and its incident edges (e.g. a leaf eaten by a herbivore).

        Safe for degree-1 tissue like leaves; removing an interior node would
        orphan its subtree, so callers should only remove tips/leaves."""
        self.nodes.pop(node_id, None)
        self.edges = {(a, b) for (a, b) in self.edges
                      if a != node_id and b != node_id}

    def neighbors(self, node_id: int) -> list[int]:
        out: list[int] = []
        for a, b in self.edges:
            if a == node_id:
                out.append(b)
            elif b == node_id:
                out.append(a)
        return out

    def nodes_of(self, kind: NodeKind) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == kind]

    def meristems(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.meristematic]

    def positions(self) -> np.ndarray:
        """All node positions as an (n, dim) array, ordered by id."""
        return np.array([self.nodes[i].pos for i in sorted(self.nodes)])


def make_seedling(world, n_roots: int = 3, germinated: bool = True) -> Plant:
    """Build a starter plant.

    ``germinated=True`` (default) returns a young seedling -- a seed, a stem tip,
    a leaf, and ``n_roots`` root tips -- the back-compatible behaviour.

    ``germinated=False`` returns *just a seed* with stored reserves; it sprouts a
    stem and roots over the first simulated day (see ``GrowToLightPolicy.germinate``)
    and only grows leaves as the stem extends -- i.e. the plant truly starts at
    seed level, with no day-0 leaf.

    Geometry is ``(lateral, height)`` mapped onto the world's axes, so the same
    factory works in 2D or 3D.
    """

    plant = Plant(dim=world.dim)

    def place(lateral: float, height: float) -> np.ndarray:
        vec = np.zeros(world.dim)
        vec[0] = lateral
        vec[world.up_axis] = world.ground_level + height
        return vec

    seed = plant.add_node(NodeKind.SEED, place(0.0, 0.0))
    if not germinated:
        # A lone seed carrying stored reserves to fund germination + early growth.
        seed.state["sugar"] = 8.0
        seed.state["water"] = 8.0
        seed.state["germ_roots"] = float(n_roots)
        return plant
    stem = plant.add_node(NodeKind.STEM, place(0.0, 1.0), meristematic=True)
    leaf = plant.add_node(NodeKind.LEAF, place(0.6, 1.3))
    plant.add_edge(seed.id, stem.id)
    plant.add_edge(stem.id, leaf.id)

    # Fan the roots out symmetrically along the first lateral axis.
    offsets = [0.0] if n_roots == 1 else [
        (i / (n_roots - 1) - 0.5) * 2.0 for i in range(n_roots)
    ]
    for off in offsets:
        root = plant.add_node(NodeKind.ROOT, place(off, -1.0), meristematic=True)
        plant.add_edge(seed.id, root.id)
    return plant
