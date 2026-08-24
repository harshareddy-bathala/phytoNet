"""One place for all tunable knobs.

``SimConfig`` bundles the environment, physiology, and growth parameters plus
season settings, and ``build_simulation`` turns a config into a ready-to-run
:class:`~phytonet.simulation.Simulation`. Today the knobs are set in code / via
CLI flags; this dataclass is shaped so Hydra can populate it later without any
call site changing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from phytonet.physiology import PhysiologyParams
from phytonet.plant import make_seedling
from phytonet.policy import GrowthParams, GrowToLightPolicy
from phytonet.simulation import Simulation
from phytonet.world import World


@dataclass
class SimConfig:
    dim: int = 2
    days: int = 60
    seed: int = 0
    # Environment. Lower these to make resources scarce and growth supply-limited.
    light_intensity: float = 1.0
    water_availability: float = 1.0
    # Sun position (length-dim). None -> world default (high, centred). Move it
    # off-centre to see phototropic leaning, especially in 3D.
    light_pos: tuple[float, ...] | None = None
    # Soil water. water_capacity=None -> infinite reservoir (constant water).
    # Set it finite to model a soil that dries (evaporation) and is refilled by
    # rain every `rain_interval` days. This is what makes drought/monsoon real.
    water_capacity: float | None = None
    evaporation: float = 0.0
    rain_interval: int = 0
    rain_amount: float = 0.0
    n_roots: int = 3
    # Spatial heterogeneity (along x). Defaults = uniform fields.
    light_gradient: float = 0.0
    water_source_x: float | None = None
    water_width: float = 3.0
    # Lifecycle (opt-in).
    reproduce: bool = False
    starve_limit: int = 0
    start_from_seed: bool = False  # True -> day 0 is a lone seed that germinates
    herbivory_interval: int = 0    # bite a leaf every N days (0 = no herbivory)
    herbivory_intensity: float = 0.5
    herbivory_start: int = 0
    season_amplitude: float = 0.0  # >0 -> ambient light fades over the season
    season_floor: float = 0.25
    physiology: PhysiologyParams = field(default_factory=PhysiologyParams)
    growth: GrowthParams = field(default_factory=GrowthParams)
    record_frames: bool = True


def build_simulation(cfg: SimConfig) -> Simulation:
    world = World(
        dim=cfg.dim,
        light_intensity=cfg.light_intensity,
        water_availability=cfg.water_availability,
        light_pos=None if cfg.light_pos is None else list(cfg.light_pos),
        water_capacity=cfg.water_capacity,
        light_gradient=cfg.light_gradient,
        water_source_x=cfg.water_source_x,
        water_width=cfg.water_width,
    )
    plant = make_seedling(world, n_roots=cfg.n_roots,
                          germinated=not cfg.start_from_seed)
    policy = GrowToLightPolicy(cfg.growth, seed=cfg.seed)
    return Simulation(
        world=world,
        plant=plant,
        physiology=cfg.physiology,
        policy=policy,
        record_frames=cfg.record_frames,
        rain_interval=cfg.rain_interval,
        rain_amount=cfg.rain_amount,
        evaporation=cfg.evaporation,
        reproduce=cfg.reproduce,
        starve_limit=cfg.starve_limit,
        herbivory_interval=cfg.herbivory_interval,
        herbivory_intensity=cfg.herbivory_intensity,
        herbivory_start=cfg.herbivory_start,
        season_amplitude=cfg.season_amplitude,
        season_floor=cfg.season_floor,
        season_length=cfg.days,   # the decline plays out over the episode
    )
