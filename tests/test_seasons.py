"""Tests for seasonal (time-varying light) environments."""

import numpy as np

from phytonet.config import SimConfig, build_simulation
from phytonet.scenarios import RANDOM_SCENARIOS, make_scenario


def test_light_fades_over_the_season():
    sim = build_simulation(SimConfig(season_amplitude=0.8, season_floor=0.2,
                                     days=60, record_frames=False))
    sim.run(60)
    light = sim.history["light"]
    assert light[0] > light[-1]                 # ambient light declined
    assert light[-1] >= 0.2 * light[0] - 1e-9   # but not below the floor


def test_no_season_keeps_light_constant():
    sim = build_simulation(SimConfig(days=40, record_frames=False))  # amplitude 0
    sim.run(40)
    assert np.allclose(sim.history["light"], sim.history["light"][0])


def test_seasonal_scenario_runs_and_reproduces_window():
    sim = build_simulation(make_scenario("seasonal", days=70, seed=0,
                                         record_frames=False))
    sim.run(70)
    assert sim.history["biomass"][-1] > sim.history["biomass"][0]
    assert sim.history["light"][0] > sim.history["light"][-1]


def test_seasonal_in_random_pool():
    assert "seasonal" in RANDOM_SCENARIOS
