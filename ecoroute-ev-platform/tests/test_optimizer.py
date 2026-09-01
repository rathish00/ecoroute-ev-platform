import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecoroute.grid_game.nash_dispatch import ChargeRequest, MultiFleetNashDispatcher, StationSlotState
from ecoroute.optimizer.grid_dispatcher import ChargingStation, GridDispatcher, Vehicle


def test_dispatcher_respects_station_capacity():
    vehicles = [Vehicle(f"v{i}", state_of_charge_kwh=10, battery_soh=0.95) for i in range(5)]
    stations = [ChargingStation("s1", capacity=2, charger_type="ac_level2_11kw", queue_wait_min=1, electricity_price_per_kwh=0.1)]
    travel = {v.vehicle_id: {"s1": 1.0} for v in vehicles}

    dispatcher = GridDispatcher(travel_energy_kwh=travel)
    result = dispatcher.dispatch(vehicles, stations)

    assigned_to_s1 = [a for a in result.assignments if a.station_id == "s1"]
    assert len(assigned_to_s1) <= 2
    assert len(result.infeasible_vehicles) == 3


def test_dispatcher_blocks_low_soh_vehicle_from_fast_charger():
    vehicles = [
        Vehicle("healthy", state_of_charge_kwh=10, battery_soh=0.98),
        Vehicle("degraded", state_of_charge_kwh=10, battery_soh=0.50),
    ]
    stations = [
        ChargingStation("fast", capacity=2, charger_type="dc_fast_150kw", queue_wait_min=1, electricity_price_per_kwh=0.2),
    ]
    travel = {v.vehicle_id: {"fast": 1.0} for v in vehicles}

    dispatcher = GridDispatcher(travel_energy_kwh=travel)
    result = dispatcher.dispatch(vehicles, stations)

    assigned_ids = {a.vehicle_id for a in result.assignments}
    assert "healthy" in assigned_ids
    assert "degraded" not in assigned_ids
    assert "degraded" in result.infeasible_vehicles


def test_nash_dispatcher_converges_and_spreads_demand():
    stations = {
        "st_A": StationSlotState(base_price_per_kwh=0.15, capacity_kw=60, congestion_sensitivity=0.05),
    }
    requests = [
        ChargeRequest("fleetA", "v1", "st_A", preferred_slot=10, flexibility=2, energy_needed_kwh=40),
        ChargeRequest("fleetB", "v1", "st_A", preferred_slot=10, flexibility=2, energy_needed_kwh=40),
        ChargeRequest("fleetB", "v2", "st_A", preferred_slot=10, flexibility=2, energy_needed_kwh=40),
    ]
    dispatcher = MultiFleetNashDispatcher(stations)
    result = dispatcher.solve(requests)

    assert result.converged
    assigned_slots = {s.assigned_slot for s in result.settled}
    # With three agents all wanting slot 10 under a 60kW cap, equilibrium
    # should spread at least some of them to adjacent slots.
    assert len(assigned_slots) >= 2
