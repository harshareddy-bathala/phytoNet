"""The step loop: tie world + plant + physiology + policy together over a season.

Each simulated day, in order:

1. **Weather** -- rain refills the soil (on the watering schedule), then a little
   evaporation dries it. (Both are no-ops for an infinite-reservoir world.)
2. **Physiology** -- produce (light->sugar, water->root, draining the soil),
   transport, maintain.
3. **Growth** -- the policy extends tips and thickens tissue.
4. Ages advance; we record history (now including soil moisture and rain).

This is the seam the Gymnasium wrapper plugs into: ``step()`` is the env step,
``history`` feeds observations/rewards.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from phytonet.physiology import PhysiologyParams, physiology_step
from phytonet.plant import NodeKind, Plant
from phytonet.policy import GrowToLightPolicy
from phytonet.world import World


@dataclass
class Simulation:
    world: World
    plant: Plant
    physiology: PhysiologyParams = field(default_factory=PhysiologyParams)
    policy: GrowToLightPolicy = field(default_factory=GrowToLightPolicy)
    record_frames: bool = True
    # Weather. rain_interval<=0 disables rain; evaporation dries the soil daily.
    rain_interval: int = 0
    rain_amount: float = 0.0
    evaporation: float = 0.0
    # Optional override for the growth phase. If set, it is called instead of
    # ``policy.step`` -- this is how the per-tissue GNN controller plugs in.
    grow_fn: object = None
    # Lifecycle (opt-in; defaults keep the old behaviour). Reproduction lets a
    # mature, well-fed leaf set seed (fitness); starvation can kill the plant.
    reproduce: bool = False
    repro_age: float = 12.0      # min leaf age before it can set seed
    repro_sugar: float = 2.0     # min leaf sugar to set seed (transport keeps
                                 # per-leaf load modest, so this is ~steady state)
    repro_cost: float = 1.0      # sugar spent per seed
    starve_limit: int = 0        # consecutive sugar-starved days before death (0=off)
    # Herbivory: every `herbivory_interval` days a leaf is bitten. The bite both
    # removes biomass (less if the tissue is defended) AND injects a wound
    # depolarisation -- the stimulus for the plant's electrical defense signal.
    herbivory_interval: int = 0      # 0 = no herbivory
    herbivory_intensity: float = 0.5  # size removed per bite (before defense)
    herbivory_start: int = 0         # no bites before this day (let it establish)
    wound_depolarization: float = 3.0  # V injected into a bitten tissue
    leaf_eaten_size: float = 0.1     # below this a bitten leaf is fully eaten
    # Seasons: ambient light declines over the episode (a growing season ending
    # in autumn). Non-stationary conditions -- a fixed strategy can't be optimal
    # throughout, so adaptive signalling has room to win.
    season_amplitude: float = 0.0    # 0 = no season; light fades to (1-amp) of base
    season_floor: float = 0.25       # light never drops below this fraction of base
    season_length: int = 0           # days over which the decline plays out

    day: int = 0
    seeds: int = 0
    alive: bool = True
    bites: int = 0
    biomass_lost: float = 0.0
    history: dict[str, list[float]] = field(default_factory=dict)
    frames: list[Plant] = field(default_factory=list)
    _raining: bool = False
    _starve: int = 0

    _KEYS = ("day", "n_nodes", "n_leaves", "n_roots", "total_sugar",
             "total_water", "biomass", "moisture", "raining", "seeds", "alive",
             "bites", "biomass_lost", "light")

    def __post_init__(self) -> None:
        self._base_light = self.world.light_intensity   # for seasonal modulation
        for key in self._KEYS:
            self.history.setdefault(key, [])
        self._record()  # capture the initial state as day 0

    def _record(self) -> None:
        nodes = self.plant.nodes.values()
        self.history["day"].append(self.day)
        self.history["n_nodes"].append(len(self.plant.nodes))
        self.history["n_leaves"].append(len(self.plant.nodes_of(NodeKind.LEAF)))
        self.history["n_roots"].append(len(self.plant.nodes_of(NodeKind.ROOT)))
        self.history["total_sugar"].append(sum(n.state["sugar"] for n in nodes))
        self.history["total_water"].append(sum(n.state["water"] for n in nodes))
        self.history["biomass"].append(sum(n.state["size"] for n in nodes))
        self.history["moisture"].append(self.world.moisture())
        self.history["raining"].append(1.0 if self._raining else 0.0)
        self.history["seeds"].append(float(self.seeds))
        self.history["alive"].append(1.0 if self.alive else 0.0)
        self.history["bites"].append(float(self.bites))
        self.history["biomass_lost"].append(self.biomass_lost)
        self.history["light"].append(self.world.light_intensity)
        if self.record_frames:
            self.frames.append(copy.deepcopy(self.plant))

    def _reproduce(self) -> None:
        """Mature, well-fed leaves set seed once (consuming sugar) -> fitness."""
        for leaf in self.plant.nodes_of(NodeKind.LEAF):
            if (not leaf.state.get("seeded")
                    and leaf.state["age"] >= self.repro_age
                    and leaf.state["sugar"] >= self.repro_sugar):
                leaf.state["sugar"] -= self.repro_cost
                leaf.state["seeded"] = 1.0
                self.seeds += 1

    def _check_starvation(self) -> None:
        if self.starve_limit <= 0:
            return
        total_sugar = sum(n.state["sugar"] for n in self.plant.nodes.values())
        self._starve = self._starve + 1 if total_sugar <= 1e-6 else 0
        if self._starve >= self.starve_limit:
            self.alive = False

    def _weather(self) -> None:
        self._raining = (
            self.rain_interval > 0 and (self.day + 1) % self.rain_interval == 0
        )
        if self._raining:
            self.world.add_water(self.rain_amount)
        self.world.evaporate(self.evaporation)

    def _herbivory(self) -> None:
        """A leaf is bitten on schedule: it loses biomass (less if defended) and
        receives a wound depolarisation -- the trigger for the defense signal."""
        if self.herbivory_interval <= 0 or (self.day + 1) < self.herbivory_start:
            return
        if (self.day + 1) % self.herbivory_interval:
            return
        leaves = self.plant.nodes_of(NodeKind.LEAF)
        if not leaves:
            return
        leaf = leaves[self.policy.rng.integers(len(leaves))]
        damage = self.herbivory_intensity * (1.0 - leaf.state.get("defense", 0.0))
        leaf.state["size"] = max(0.0, leaf.state["size"] - damage)
        # The bite triggers the wound signal (defense channel) and a visible
        # depolarisation of the membrane potential.
        leaf.state["wound"] = leaf.state.get("wound", 0.0) + self.wound_depolarization
        leaf.state["V"] = leaf.state.get("V", 0.0) + self.wound_depolarization
        self.bites += 1
        self.biomass_lost += damage
        if leaf.state["size"] <= self.leaf_eaten_size:
            self.plant.remove_node(leaf.id)  # fully eaten

    def _season(self) -> None:
        """Fade ambient light over the season (low at the end -- autumn)."""
        if self.season_amplitude <= 0 or self.season_length <= 0:
            return
        phase = min(1.0, self.day / self.season_length)
        factor = max(self.season_floor, 1.0 - self.season_amplitude * phase)
        self.world.light_intensity = self._base_light * factor

    def step(self) -> None:
        self._season()
        self._weather()
        self._herbivory()
        physiology_step(self.plant, self.world, self.physiology)
        self.policy.germinate(self.plant, self.world)  # no-op once sprouted
        (self.grow_fn or self.policy.step)(self.plant, self.world)
        if self.reproduce:
            self._reproduce()
        self._check_starvation()
        for node in self.plant.nodes.values():
            node.state["age"] += 1.0
        self.day += 1
        self._record()

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    def env_series(self) -> list[dict]:
        """Per-day environment state, for renderers that show weather over time."""
        return [
            {"day": int(self.history["day"][i]),
             "moisture": self.history["moisture"][i],
             "raining": bool(self.history["raining"][i])}
            for i in range(len(self.history["day"]))
        ]
