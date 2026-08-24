"""Growth policy: allocate a limited growth budget between shoots, roots, girth.

Each step the plant can fund only so many growth events (the *budget*, set by how
much sugar and water it has pooled). The policy splits that budget three ways --
extend shoots, extend roots, or thicken existing tissue -- according to
**allocation weights**. Because the three compete for one budget, the split is a
real trade-off: more roots means fewer shoots this step.

That trade-off is the whole point. Its best setting is *condition-dependent*:
under drought, water is the bottleneck so investing in roots (which unlock water)
pays; under shade, light/sugar is the bottleneck so investing in shoots/leaves
pays. A fixed split can't be optimal everywhere -- which is exactly what gives a
learned policy (PPO, later a GNN) something worth learning.

The hand-coded baseline uses equal weights. The Gym env overrides the weights
each step from the agent's action. Tropism *directions* (up/down, toward light)
remain fixed physics; the agent decides *how much* to grow where, not the angles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phytonet.plant import NodeKind, Plant
from phytonet.world import World


@dataclass
class GrowthParams:
    step_len: float = 0.7          # distance a tip extends per growth event
    # Asymmetric build costs -- the trade-off engine. Shoot/leaf expansion is
    # turgor-driven, so it costs mostly WATER; roots are built from imported
    # carbon, so they cost mostly SUGAR. Hence under water stress shoots get
    # expensive (favour roots) and under carbon/shade stress roots get expensive
    # (favour shoots): the best allocation depends on the world.
    shoot_sugar_cost: float = 0.3
    shoot_water_cost: float = 1.0
    root_sugar_cost: float = 1.0
    root_water_cost: float = 0.3
    wobble: float = 0.25           # lateral randomness in growth direction
    leaf_every: int = 2            # spawn a leaf every Nth stem extension
    branch_prob: float = 0.18      # chance a stem extension also forks a side branch
    root_branch_prob: float = 0.22  # chance a root extension forks (root system)
    max_meristems: int = 12        # cap active shoot tips
    max_root_meristems: int = 16   # cap active root tips
    # Budgeted allocation. Each step the plant funds up to `max_growth_per_step`
    # growth events (capped further by pooled resources), split between shoots /
    # roots / thickening in proportion to these weights. The agent sets them.
    max_growth_per_step: int = 6
    shoot_alloc: float = 1.0
    root_alloc: float = 1.0
    thicken_alloc: float = 1.0
    # Thickening: invest surplus (above a buffer) into girth (size). Gives `size`
    # a purpose -- bigger leaves photosynthesize more, bigger tissue costs more.
    thicken_buffer: float = 2.0    # resource level a node keeps before investing
    thicken_cost: float = 0.5      # sugar AND water spent per thickening event
    thicken_step: float = 0.2      # size added per thickening event
    max_size: float = 4.0          # cap on a single node's size
    # Tropisms ("physics"): growth *direction* is a blend of these cues.
    gravitropism: float = 0.7      # weight on growing up (shoot) / down (root)
    phototropism: float = 0.5      # weight on bending toward the sun (shoot only)
    steer_gain: float = 2.2        # how strongly a per-tissue steer action bends x


class GrowToLightPolicy:
    """Allocates the growth budget; extends tips toward light/water."""

    def __init__(self, params: GrowthParams | None = None, seed: int = 0):
        self.p = params or GrowthParams()
        self.rng = np.random.default_rng(seed)
        self._stem_growths = 0

    # -- geometry helpers ---------------------------------------------------
    def _random_horizontal(self, world: World) -> np.ndarray:
        """A random unit vector in the ground plane (zero on the up-axis)."""
        v = np.zeros(world.dim)
        for ax in range(world.dim):
            if ax != world.up_axis:
                v[ax] = self.rng.uniform(-1.0, 1.0)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def _direction(self, node, world: World, steer_x: float = 0.0) -> np.ndarray:
        """Unit growth direction: tropism blend + lateral wobble + optional steer.

        ``steer_x`` is the per-tissue steering action (along the heterogeneity
        axis x): positive heads toward +x, negative toward -x. With it, a tip can
        navigate toward local light/water -- something the fixed tropisms cannot."""
        up = world.up_vector()
        if node.kind == NodeKind.STEM:
            d = self.p.gravitropism * up
            d = d + self.p.phototropism * world.light_direction(node.pos)
        else:  # ROOT
            d = -self.p.gravitropism * up
        d = d + self.p.wobble * self._random_horizontal(world)
        d[0] += self.p.steer_gain * steer_x
        n = float(np.linalg.norm(d))
        return d / n if n > 0 else up

    def _clamp(self, pos: np.ndarray, world: World) -> np.ndarray:
        return np.clip(pos, world.bounds_low, world.bounds_high)

    def _extend(self, plant: Plant, world: World, tip, direction: np.ndarray):
        new_pos = self._clamp(tip.pos + self.p.step_len * direction, world)
        new = plant.add_node(tip.kind, new_pos, meristematic=True)
        plant.add_edge(tip.id, new.id)
        return new

    def _count(self, plant: Plant, kind: NodeKind) -> int:
        return sum(1 for n in plant.meristems() if n.kind == kind)

    def germinate(self, plant: Plant, world: World) -> None:
        """Sprout a stem + roots from a lone seed (no-op once germinated / grown).

        Lets the plant start at seed level: day 0 is just the seed; on the first
        day it sends up a stem tip and down a fan of root tips, funded by the
        seed's reserves. Leaves then appear as the stem extends."""
        if len(plant.nodes) != 1:
            return
        seed = next(iter(plant.nodes.values()))
        if seed.kind != NodeKind.SEED or seed.state.get("germinated"):
            return
        n_roots = int(seed.state.get("germ_roots", 3))
        up, g = world.up_axis, world.ground_level

        def place(lat: float, h: float) -> np.ndarray:
            v = np.zeros(world.dim)
            v[0] = lat
            v[up] = g + h
            return v

        stem = plant.add_node(NodeKind.STEM, place(0.0, 1.0), meristematic=True)
        plant.add_edge(seed.id, stem.id)
        offsets = [0.0] if n_roots == 1 else [
            (i / (n_roots - 1) - 0.5) * 2.0 for i in range(n_roots)
        ]
        for off in offsets:
            root = plant.add_node(NodeKind.ROOT, place(off, -1.0), meristematic=True)
            plant.add_edge(seed.id, root.id)
        seed.state["germinated"] = 1.0

    # -- budget + allocation ------------------------------------------------
    def _budget(self, plant: Plant) -> int:
        """How many growth events the plant can fund this step.

        Coupled to the *scarcer* pooled resource, so relieving the bottleneck
        actually unlocks more growth: in drought, building roots raises pooled
        water -> a bigger budget -> strictly more growth. That coupling is what
        makes the optimal allocation shift with conditions (rather than a flat
        plateau). Capped by ``max_growth_per_step``."""
        nodes = plant.nodes.values()
        pooled_sugar = sum(n.state["sugar"] for n in nodes)
        pooled_water = sum(n.state["water"] for n in nodes)
        return max(0, min(self.p.max_growth_per_step,
                          int(min(pooled_sugar, pooled_water))))

    def step(self, plant: Plant, world: World) -> None:
        w = np.array([self.p.shoot_alloc, self.p.root_alloc, self.p.thicken_alloc],
                     dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(3, 1 / 3)
        budget = self._budget(plant)
        n_shoot = int(round(budget * w[0]))
        n_root = int(round(budget * w[1]))
        n_thicken = max(0, budget - n_shoot - n_root)

        self._grow_shoots(plant, world, n_shoot)
        self._grow_roots(plant, world, n_root)
        self._thicken(plant, limit=n_thicken)

    # -- a single tip's growth (reused by global allocation AND per-tissue) --
    def extend_tip(self, plant: Plant, world: World, tip, steer_x: float = 0.0) -> bool:
        """Try to extend one meristem, paying its kind's asymmetric cost. Handles
        leaf spawning (shoots) and side-branch forking. Returns True if it grew.

        ``steer_x`` is the per-tissue steering action (0 for the hand-coded budget
        policy). This is the atomic growth action shared by both controllers."""
        if not tip.meristematic:
            return False
        if tip.kind == NodeKind.STEM:
            sc, wc = self.p.shoot_sugar_cost, self.p.shoot_water_cost
        else:
            sc, wc = self.p.root_sugar_cost, self.p.root_water_cost
        if tip.state["sugar"] < sc or tip.state["water"] < wc:
            return False
        tip.state["sugar"] -= sc
        tip.state["water"] -= wc
        new = self._extend(plant, world, tip, self._direction(tip, world, steer_x))
        tip.meristematic = False

        if tip.kind == NodeKind.STEM:
            self._stem_growths += 1
            if self._stem_growths % self.p.leaf_every == 0:
                leaf_pos = new.pos + 0.5 * self._random_horizontal(world)
                leaf = plant.add_node(NodeKind.LEAF, self._clamp(leaf_pos, world))
                plant.add_edge(new.id, leaf.id)
            if (self.rng.random() < self.p.branch_prob
                    and self._count(plant, NodeKind.STEM) < self.p.max_meristems):
                bd = self._direction(new, world) + 0.6 * self._random_horizontal(world)
                self._extend(plant, world, new, bd / np.linalg.norm(bd))
        else:
            if (self.rng.random() < self.p.root_branch_prob
                    and self._count(plant, NodeKind.ROOT) < self.p.max_root_meristems):
                bd = self._direction(new, world) + 0.8 * self._random_horizontal(world)
                self._extend(plant, world, new, bd / np.linalg.norm(bd))
        return True

    def thicken_one(self, node) -> bool:
        """Try to thicken a single node (invest surplus into size). Returns True
        if it thickened."""
        need = self.p.thicken_buffer + self.p.thicken_cost
        if (node.state["size"] < self.p.max_size
                and node.state["sugar"] >= need and node.state["water"] >= need):
            node.state["sugar"] -= self.p.thicken_cost
            node.state["water"] -= self.p.thicken_cost
            node.state["size"] = min(self.p.max_size,
                                     node.state["size"] + self.p.thicken_step)
            return True
        return False

    # -- the three things growth can spend the budget on --------------------
    def _grow_shoots(self, plant: Plant, world: World, n: int) -> None:
        grown = 0
        for tip in [t for t in plant.meristems() if t.kind == NodeKind.STEM]:
            if grown >= n:
                break
            if self.extend_tip(plant, world, tip):
                grown += 1

    def _grow_roots(self, plant: Plant, world: World, n: int) -> None:
        grown = 0
        for tip in [t for t in plant.meristems() if t.kind == NodeKind.ROOT]:
            if grown >= n:
                break
            if self.extend_tip(plant, world, tip):
                grown += 1

    def _thicken(self, plant: Plant, limit: int | None = None) -> int:
        """Invest surplus into node size. ``limit`` caps how many nodes thicken
        this call (``None`` = all eligible). Returns count."""
        count = 0
        for node in plant.nodes.values():
            if limit is not None and count >= limit:
                break
            if self.thicken_one(node):
                count += 1
        return count
