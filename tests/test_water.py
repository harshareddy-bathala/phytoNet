"""Tests for the dynamic soil-water reservoir, weather, and scenarios."""

import numpy as np

from phytonet import World, make_seedling
from phytonet.config import SimConfig, build_simulation
from phytonet.physiology import PhysiologyParams, uptake
from phytonet.scenarios import SCENARIOS, make_scenario


def test_infinite_reservoir_is_constant_and_undepletable():
    world = World(dim=2, water_availability=2.0)  # capacity None -> infinite
    assert world.moisture() == 1.0
    below = np.array([0.0, -1.0])
    assert world.water_at(below) == 2.0
    world.deplete(100.0)
    world.evaporate(100.0)
    assert world.water_at(below) == 2.0  # unchanged


def test_finite_reservoir_depletes_evaporates_and_refills():
    world = World(dim=2, water_availability=1.0, water_capacity=10.0)
    assert world.soil_water == 10.0 and world.moisture() == 1.0
    world.deplete(4.0)
    assert np.isclose(world.soil_water, 6.0)
    world.evaporate(2.0)
    assert np.isclose(world.soil_water, 4.0)
    world.add_water(100.0)
    assert world.soil_water == 10.0  # capped at capacity


def test_water_availability_scales_with_moisture():
    world = World(dim=2, water_availability=2.0, water_capacity=10.0)
    below = np.array([0.0, -1.0])
    assert world.water_at(below) == 2.0   # full
    world.deplete(5.0)                     # half empty
    assert np.isclose(world.water_at(below), 1.0)


def test_uptake_drains_a_finite_reservoir():
    world = World(dim=2, water_capacity=10.0)
    plant = make_seedling(world)
    uptake(plant, world, PhysiologyParams())
    assert world.soil_water < 10.0


def test_drought_dries_out_more_than_monsoon():
    drought = build_simulation(make_scenario("drought", days=60, record_frames=False))
    monsoon = build_simulation(make_scenario("monsoon", days=60, record_frames=False))
    drought.run(60)
    monsoon.run(60)
    assert drought.world.moisture() < monsoon.world.moisture()


def test_rain_is_recorded_on_the_schedule():
    sim = build_simulation(make_scenario("monsoon", days=30, record_frames=False))
    sim.run(30)
    assert sum(sim.history["raining"]) > 0  # it rained at least once


def test_all_scenarios_build_and_run():
    for name in SCENARIOS:
        sim = build_simulation(make_scenario(name, days=20, record_frames=False))
        sim.run(20)
        assert sim.history["biomass"][-1] >= sim.history["biomass"][0]
