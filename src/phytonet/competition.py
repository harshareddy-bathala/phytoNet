"""Plant-plant competition: several plants in one shared world.

Unlike pure resource variability, competition poses a genuine *strategic* problem
-- a height/shading race for light and a tug-of-war over a shared soil-water
reservoir -- so it's the setting most likely to reward adaptive, distributed
control beyond the wound-defense case.

Each plant runs its own controller in a shared :class:`~phytonet.world.World`.
Light is attenuated by **any** taller leaf nearby (cross-plant shading, which
also subsumes self-shading), and every plant's roots draw from the *same*
reservoir. Fitness is each plant's own biomass.
"""

from __future__ import annotations

import numpy as np

from phytonet.physiology import (
    PhysiologyParams, _feedback, maintain, transport,
)
from phytonet.plant import NodeKind, make_seedling
from phytonet.policy import GrowthParams, GrowToLightPolicy
from phytonet.world import World

_SHADE_RADIUS = 2.0


def make_controller(kind: str):
    """A grow-phase controller (uniform ``.step(plant, world)`` interface)."""
    policy = GrowToLightPolicy(GrowthParams(), seed=0)
    if kind == "fixed":
        return policy                                  # hand-coded allocation
    if kind == "bioelectric":
        from phytonet.signaling import SignalingController, SignalingParams
        return SignalingController(policy, SignalingParams())
    if kind == "learned":
        from phytonet.learned_signaling import (
            LearnedSignalingController, LearnedSignalingParams)
        return LearnedSignalingController(policy, LearnedSignalingParams.default())
    raise ValueError(f"unknown controller kind {kind!r}")


def _shift(plant, dx: float) -> None:
    for n in plant.nodes.values():
        n.pos[0] += dx


class Arena:
    """Several plants competing for light + a shared soil-water reservoir."""

    def __init__(self, world: World, entries, phys: PhysiologyParams | None = None,
                 shade_k: float = 0.4, rain_interval: int = 5,
                 rain_amount: float = 9.0, evaporation: float = 0.3):
        self.world = world
        self.entries = entries          # [{plant, controller, label}, ...]
        self.phys = phys or PhysiologyParams()
        self.shade_k = shade_k
        self.rain_interval, self.rain_amount, self.evaporation = (
            rain_interval, rain_amount, evaporation)
        self.day = 0
        self.history = {e["label"]: [] for e in entries}

    def _photosynthesis(self) -> None:
        """Cross-plant shading: a leaf is shaded by any taller leaf within radius."""
        leaves = [lf for e in self.entries for lf in e["plant"].nodes_of(NodeKind.LEAF)]
        up = self.world.up_axis
        for leaf in leaves:
            shaded = sum(
                o.state["size"] for o in leaves
                if o is not leaf and o.pos[up] > leaf.pos[up]
                and abs(o.pos[0] - leaf.pos[0]) < _SHADE_RADIUS)
            exposure = float(np.exp(-self.shade_k * shaded))
            gain = (self.world.light_at(leaf.pos) * leaf.state["size"]
                    * self.phys.photo_rate * exposure)
            leaf.state["sugar"] += gain * _feedback(leaf.state["sugar"], self.phys.sugar_cap)

    def _shared_uptake(self) -> None:
        """All roots draw from the *same* pre-step soil moisture (fair), then the
        reservoir is depleted by the total -- no first-mover advantage."""
        total = 0.0
        for e in self.entries:                      # moisture unchanged until the end
            for root in e["plant"].nodes_of(NodeKind.ROOT):
                gain = (self.world.water_at(root.pos) * root.state["size"]
                        * self.phys.uptake_rate
                        * _feedback(root.state["water"], self.phys.water_cap))
                root.state["water"] += gain
                total += gain
        self.world.deplete(total)

    def step(self) -> None:
        if self.rain_interval > 0 and (self.day + 1) % self.rain_interval == 0:
            self.world.add_water(self.rain_amount)
        self.world.evaporate(self.evaporation)
        self._photosynthesis()
        self._shared_uptake()
        for e in self.entries:
            transport(e["plant"], self.phys)
            maintain(e["plant"], self.phys)
        for e in self.entries:
            e["controller"].step(e["plant"], self.world)
            for n in e["plant"].nodes.values():
                n.state["age"] += 1.0
        self.day += 1
        for e in self.entries:
            self.history[e["label"]].append(self.biomass()[e["label"]])

    def run(self, days: int) -> None:
        for _ in range(days):
            self.step()

    def biomass(self) -> dict[str, float]:
        return {e["label"]: float(sum(n.state["size"] for n in e["plant"].nodes.values()))
                for e in self.entries}


def competitive_fitness(theta, opponent: str = "fixed", spacing: float = 1.4,
                        shade_k: float = 0.6, seeds=range(2), days: int = 70) -> float:
    """Relative fitness (own biomass - opponent's) of a signalling rule when it
    *competes* in the arena against ``opponent``. Each pairing is run in both
    left/right arrangements to cancel any position bias."""
    import numpy as np
    from phytonet.policy import GrowToLightPolicy
    from phytonet.signaling import SignalingController, SignalingParams

    params = SignalingParams.from_vector(np.clip(theta, 0.0, 8.0))
    scores = []
    for s in seeds:
        for cand_off, opp_off in ((-spacing / 2, spacing / 2), (spacing / 2, -spacing / 2)):
            world = World(dim=2, water_capacity=55.0, light_intensity=1.3)
            cand = make_seedling(world, germinated=True); _shift(cand, cand_off)
            opp = make_seedling(world, germinated=True); _shift(opp, opp_off)
            arena = Arena(world, [
                {"plant": cand, "label": "cand",
                 "controller": SignalingController(GrowToLightPolicy(seed=0), params)},
                {"plant": opp, "label": "opp", "controller": make_controller(opponent)},
            ], shade_k=shade_k)
            arena.run(days)
            b = arena.biomass()
            scores.append(b["cand"] - b["opp"])
    return float(np.mean(scores))


def _duel(cand_params, opp_params, days: int = 70, spacing: float = 1.4,
          shade_k: float = 0.6, seeds=range(1)) -> float:
    """Relative biomass (candidate - opponent) of two *signalling* rules competing
    in the arena, both arrangements, averaged over seeds."""
    import numpy as np
    from phytonet.policy import GrowToLightPolicy
    from phytonet.signaling import SignalingController

    scores = []
    for _s in seeds:
        for c_off, o_off in ((-spacing / 2, spacing / 2), (spacing / 2, -spacing / 2)):
            world = World(dim=2, water_capacity=55.0, light_intensity=1.3)
            cp = make_seedling(world, germinated=True); _shift(cp, c_off)
            op = make_seedling(world, germinated=True); _shift(op, o_off)
            arena = Arena(world, [
                {"plant": cp, "label": "c",
                 "controller": SignalingController(GrowToLightPolicy(seed=0), cand_params)},
                {"plant": op, "label": "o",
                 "controller": SignalingController(GrowToLightPolicy(seed=0), opp_params)},
            ], shade_k=shade_k)
            arena.run(days)
            b = arena.biomass()
            scores.append(b["c"] - b["o"])
    return float(np.mean(scores))


def coevolve_selfplay(iterations: int = 12, population: int = 18, elite_frac: float = 0.25,
                      init_std: float = 0.4, seed: int = 0, days: int = 70):
    """Self-play co-evolution: each generation, candidates compete against the
    *current* population mean (a co-adapting opponent that strengthens as the
    population improves). Returns (best SignalingParams, history of best scores).

    Unlike best-response-to-fixed, both sides evolve; the natural endpoint is a
    competitive equilibrium (relative fitness -> 0)."""
    import numpy as np
    from phytonet.signaling import SignalingParams

    rng = np.random.default_rng(seed)
    x0 = SignalingParams().to_vector()
    mean, std = x0.copy(), np.full(x0.shape, init_std)
    n_elite = max(2, int(population * elite_frac))
    history = []
    for _ in range(iterations):
        opponent = SignalingParams.from_vector(np.clip(mean, 0.0, 8.0))   # current self
        samples = np.clip(rng.normal(mean, std, size=(population, x0.shape[0])), 0.0, 8.0)
        scores = np.array([_duel(SignalingParams.from_vector(t), opponent, days=days)
                           for t in samples])
        elite = samples[np.argsort(scores)[-n_elite:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-3
        history.append(float(scores.max()))
    return SignalingParams.from_vector(np.clip(mean, 0.0, 8.0)), history


def coevolve_competitive(opponent: str = "fixed", iterations: int = 12,
                         population: int = 18, seed: int = 0, days: int = 70):
    """Evolve (CEM) a signalling rule to *beat* ``opponent`` in head-to-head
    competition. Returns (best SignalingParams, CEM result)."""
    from phytonet.learn import cem_optimize
    from phytonet.signaling import SignalingParams
    x0 = SignalingParams().to_vector()
    res = cem_optimize(
        lambda v: competitive_fitness(v, opponent=opponent, days=days, seeds=range(2)),
        x0, iterations=iterations, population=population, seed=seed, lo=0.0, hi=8.0)
    return SignalingParams.from_vector(res.best_theta), res


def make_arena(kinds_labels, spacing: float = 2.5, water_capacity: float = 55.0,
               light_intensity: float = 1.3, dim: int = 2) -> Arena:
    """Build an arena from ``[(controller_kind, label), ...]``, plants spaced in x."""
    world = World(dim=dim, water_capacity=water_capacity, light_intensity=light_intensity)
    n = len(kinds_labels)
    offsets = (np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * spacing) if n > 1 else [0.0]
    entries = []
    for (kind, label), off in zip(kinds_labels, offsets):
        plant = make_seedling(world, germinated=True)
        _shift(plant, float(off))
        entries.append({"plant": plant, "controller": make_controller(kind),
                        "label": label})
    return Arena(world, entries)
