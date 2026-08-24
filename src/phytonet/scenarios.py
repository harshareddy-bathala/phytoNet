"""Named environmental conditions to test a plant against.

Each scenario is a :class:`~phytonet.config.SimConfig` describing a different
world -- plenty / drought / monsoon / shade / bright. This is the "test the
plant in different conditions" knob: pick one on the CLI (``--scenario drought``)
or sample one per episode in the RL env (domain randomization).

A drying soil only bites when ``water_capacity`` is finite, so the stress
scenarios set it; ``baseline`` keeps the infinite reservoir.
"""

from __future__ import annotations

from phytonet.config import SimConfig

# Factories (not shared instances) so callers can freely mutate what they get.
SCENARIOS: dict[str, callable] = {
    # Benign: infinite water, full sun. The original behaviour.
    "baseline": lambda: SimConfig(),
    # Frequent rain keeps a finite soil topped up -> lush growth.
    "monsoon": lambda: SimConfig(
        water_capacity=40.0, evaporation=0.4, rain_interval=4, rain_amount=14.0
    ),
    # Water-limited: low, slowly-refilled soil so WATER is the bottleneck.
    # With water-costly shoots, the winning move is to invest in (sugar-built)
    # roots, which fetch the scarce water.
    "drought": lambda: SimConfig(
        water_capacity=18.0, evaporation=0.7, rain_interval=14, rain_amount=7.0,
        water_availability=0.6,
    ),
    # Carbon-limited: dim light, ample water -> SUGAR is the bottleneck. With
    # sugar-costly roots, the winning move is to invest in (water-built) shoots/
    # leaves, which fetch the scarce carbon.
    "shade": lambda: SimConfig(
        light_intensity=0.4, water_capacity=60.0, evaporation=0.2,
        rain_interval=4, rain_amount=14.0,
    ),
    # Bright and well-watered.
    "bright": lambda: SimConfig(
        light_intensity=1.6, water_capacity=40.0, evaporation=0.4,
        rain_interval=5, rain_amount=12.0,
    ),
    # Spatially heterogeneous: brighter toward +x, soil water concentrated toward
    # -x. Now *where* a tissue sits decides its value, so the best action differs
    # tissue-to-tissue -- something a per-tissue/GNN policy can exploit but a
    # single global allocation cannot.
    "patchy": lambda: SimConfig(
        light_gradient=0.6, water_source_x=-2.5, water_width=2.0,
        water_availability=1.5,
    ),
    # Herbivory: a leaf is bitten every few days. Each bite wounds the tissue
    # (a depolarisation) and removes biomass -- a plant that propagates the wound
    # signal and arms systemic defense loses less. The canonical real plant
    # electrical signal, as a survival pressure.
    "herbivory": lambda: SimConfig(
        herbivory_interval=4, herbivory_intensity=0.5, herbivory_start=22,
        water_capacity=40.0, evaporation=0.3, rain_interval=5, rain_amount=10.0,
    ),
    # Seasonal: light is plentiful early and fades toward the end (autumn), with
    # reproduction on -- so *timing* matters (build + set seed before the light
    # goes). A fixed allocation can't be optimal across the whole season.
    "seasonal": lambda: SimConfig(
        season_amplitude=0.8, season_floor=0.2, reproduce=True,
        water_capacity=45.0, evaporation=0.3, rain_interval=5, rain_amount=11.0,
    ),
    # Full lifecycle: mature leaves set seed (fitness) and the plant can starve to
    # death -- survival + reproduction, not just biomass. (Death is enabled, so
    # this scenario is excluded from the RL randomisation pool below.)
    "lifecycle": lambda: SimConfig(
        reproduce=True, starve_limit=8, water_capacity=30.0, evaporation=0.5,
        rain_interval=10, rain_amount=9.0,
    ),
}

# Scenarios the RL env randomises over per episode. Excludes ``lifecycle``
# because its early-death termination would break fixed-length assumptions; it
# is used explicitly, not sampled.
RANDOM_SCENARIOS = ("baseline", "monsoon", "drought", "shade", "bright",
                    "patchy", "herbivory", "seasonal")


def make_scenario(name: str, **overrides) -> SimConfig:
    """Build a scenario config by name, applying any field overrides."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choices: {sorted(SCENARIOS)}")
    cfg = SCENARIOS[name]()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
