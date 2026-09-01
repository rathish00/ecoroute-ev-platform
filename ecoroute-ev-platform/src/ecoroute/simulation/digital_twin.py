"""
digital_twin.py
================
Fleet Telematics "Digital Twin" Simulation Engine.

Rather than validating the platform only against a static held-out test
set, this module runs a live-action simulation: N virtual EV vans drive
around the synthetic city graph delivering packages, consuming battery
according to the physics/ML energy model, and -- when their State of
Charge (SoC) drops below a critical threshold -- request a charging
assignment from the `GridDispatcher`. This produces a continuous stream of
telemetry events (`SimulationEvent`) that exercises the routing engine,
the battery model, and the optimizer together, the same way live fleet
traffic eventually would.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from ecoroute.ml.battery_model import SOHPenaltyMatrix
from ecoroute.optimizer.grid_dispatcher import ChargingStation, GridDispatcher, Vehicle as DispatchVehicle
from ecoroute.routing.router import EnergyAwareRouter, RouteContext
from ecoroute.utils.physics import VehicleProfile


@dataclass
class SimulationEvent:
    tick: int
    vehicle_id: str
    event_type: str  # "depart" | "arrive" | "drive" | "low_soc" | "charge_assigned" | "charging" | "charged"
    details: Dict = field(default_factory=dict)


@dataclass
class SimVehicle:
    vehicle_id: str
    profile: VehicleProfile
    node: Tuple[int, int]
    destination: Optional[Tuple[int, int]] = None
    route: List[Tuple[int, int]] = field(default_factory=list)
    route_idx: int = 0
    soc_kwh: float = 60.0
    battery_soh: float = 1.0
    status: str = "idle"  # idle | delivering | seeking_charge | charging
    charge_ticks_remaining: int = 0
    deliveries_completed: int = 0
    total_energy_kwh: float = 0.0
    total_distance_km: float = 0.0


class DigitalTwinSimulation:
    """
    Orchestrates a fleet of SimVehicle agents on a shared city graph,
    driving them via the EnergyAwareRouter, degrading their battery per
    the physics/ML cost model, and routing low-SoC vehicles through the
    GridDispatcher + SOH-Guard when they need to charge.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        n_vehicles: int = 20,
        stations: Optional[List[ChargingStation]] = None,
        low_soc_threshold_frac: float = 0.25,
        seed: int = 11,
    ):
        self.graph = graph
        self.rng = random.Random(seed)
        self.router = EnergyAwareRouter(graph)
        self.soh_guard = SOHPenaltyMatrix()
        self.low_soc_threshold_frac = low_soc_threshold_frac
        self.tick = 0
        self.events: List[SimulationEvent] = []

        self.stations = stations or self._default_stations()
        self.vehicles: List[SimVehicle] = [self._spawn_vehicle(i) for i in range(n_vehicles)]
        self._charging_occupancy: Dict[str, int] = {s.station_id: 0 for s in self.stations}

    def _default_stations(self) -> List[ChargingStation]:
        nodes = list(self.graph.nodes)
        chosen = self.rng.sample(nodes, k=min(6, len(nodes)))
        types = ["dc_fast_150kw", "dc_fast_50kw", "ac_level2_11kw", "ac_level2_11kw", "dc_fast_50kw", "ac_level2_11kw"]
        stations = []
        for i, node in enumerate(chosen):
            stations.append(
                ChargingStation(
                    station_id=f"station_{i}_{node}",
                    capacity=self.rng.choice([1, 2, 3]),
                    charger_type=types[i % len(types)],
                    queue_wait_min=self.rng.uniform(0, 20),
                    electricity_price_per_kwh=self.rng.uniform(0.10, 0.24),
                )
            )
        self._station_nodes = {s.station_id: node for s, node in zip(stations, chosen)}
        return stations

    def _spawn_vehicle(self, idx: int) -> SimVehicle:
        node = self.rng.choice(list(self.graph.nodes))
        profile = VehicleProfile()
        soh = self.rng.uniform(0.68, 1.0)
        soc_frac = self.rng.uniform(0.4, 1.0)
        return SimVehicle(
            vehicle_id=f"van_{idx:03d}",
            profile=profile,
            node=node,
            soc_kwh=soc_frac * profile.battery_capacity_kwh,
            battery_soh=soh,
        )

    def _log(self, vehicle_id: str, event_type: str, **details):
        self.events.append(SimulationEvent(tick=self.tick, vehicle_id=vehicle_id, event_type=event_type, details=details))

    def _assign_new_delivery(self, v: SimVehicle):
        nodes = list(self.graph.nodes)
        dest = self.rng.choice(nodes)
        while dest == v.node:
            dest = self.rng.choice(nodes)

        ctx = RouteContext(
            vehicle=v.profile,
            cargo_kg=self.rng.uniform(0, v.profile.max_cargo_kg),
            city_temperature_c=self.rng.uniform(10, 38),
            city_wind_kmh=self.rng.uniform(0, 25),
            battery_soh=v.battery_soh,
            gentle_mode=self.soh_guard.requires_gentle_routing(v.battery_soh),
        )
        try:
            result = self.router.shortest_energy_path(v.node, dest, ctx)
        except nx.NetworkXNoPath:
            return
        v.destination = dest
        v.route = result.path
        v.route_idx = 0
        v.status = "delivering"
        self._log(v.vehicle_id, "depart", destination=dest, planned_energy_kwh=result.total_energy_kwh)

    def _step_vehicle(self, v: SimVehicle):
        if v.status == "charging":
            v.charge_ticks_remaining -= 1
            if v.charge_ticks_remaining <= 0:
                v.status = "idle"
                v.soc_kwh = v.profile.battery_capacity_kwh * 0.9
                self._log(v.vehicle_id, "charged", soc_kwh=round(v.soc_kwh, 2))
            return

        if v.status == "seeking_charge":
            return  # waiting on dispatcher assignment this tick

        if v.status == "idle":
            self._assign_new_delivery(v)
            return

        if v.status == "delivering":
            if v.route_idx >= len(v.route) - 1:
                v.status = "idle"
                v.deliveries_completed += 1
                self._log(v.vehicle_id, "arrive", node=v.node, deliveries_completed=v.deliveries_completed)
                return

            u, nxt = v.route[v.route_idx], v.route[v.route_idx + 1]
            edge = self.graph.edges[u, nxt]
            ctx = RouteContext(vehicle=v.profile, battery_soh=v.battery_soh)
            energy = self.router.cost_fn(self.graph, u, nxt, edge, ctx)

            v.soc_kwh = max(0.0, v.soc_kwh - energy)
            v.total_energy_kwh += energy
            v.total_distance_km += edge["distance_km"]
            v.node = nxt
            v.route_idx += 1
            self._log(v.vehicle_id, "drive", node=v.node, soc_kwh=round(v.soc_kwh, 2), energy_kwh=round(energy, 4))

            soc_frac = v.soc_kwh / v.profile.battery_capacity_kwh
            if soc_frac <= self.low_soc_threshold_frac and v.status != "seeking_charge":
                v.status = "seeking_charge"
                self._log(v.vehicle_id, "low_soc", soc_frac=round(soc_frac, 3))

    def _run_charge_dispatch_round(self):
        seeking = [v for v in self.vehicles if v.status == "seeking_charge"]
        if not seeking:
            return

        dispatch_vehicles = [
            DispatchVehicle(vehicle_id=v.vehicle_id, state_of_charge_kwh=v.soc_kwh, battery_soh=v.battery_soh)
            for v in seeking
        ]

        travel_energy: Dict[str, Dict[str, float]] = {}
        for v in seeking:
            travel_energy[v.vehicle_id] = {}
            for s in self.stations:
                station_node = self._station_nodes[s.station_id]
                ctx = RouteContext(vehicle=v.profile, battery_soh=v.battery_soh)
                try:
                    result = self.router.shortest_energy_path(v.node, station_node, ctx)
                    travel_energy[v.vehicle_id][s.station_id] = result.total_energy_kwh
                except nx.NetworkXNoPath:
                    continue

        available_stations = [
            ChargingStation(
                station_id=s.station_id,
                capacity=max(0, s.capacity - self._charging_occupancy.get(s.station_id, 0)),
                charger_type=s.charger_type,
                queue_wait_min=s.queue_wait_min,
                electricity_price_per_kwh=s.electricity_price_per_kwh,
            )
            for s in self.stations
        ]

        dispatcher = GridDispatcher(travel_energy_kwh=travel_energy, soh_guard=self.soh_guard)
        result = dispatcher.dispatch(dispatch_vehicles, available_stations)

        by_vehicle = {v.vehicle_id: v for v in seeking}
        for assignment in result.assignments:
            v = by_vehicle[assignment.vehicle_id]
            v.status = "charging"
            v.charge_ticks_remaining = self.rng.randint(3, 8)
            self._charging_occupancy[assignment.station_id] = self._charging_occupancy.get(assignment.station_id, 0) + 1
            self._log(
                v.vehicle_id,
                "charge_assigned",
                station_id=assignment.station_id,
                cost=round(assignment.total_cost, 3),
                backend=result.backend,
            )

    def step(self):
        self.tick += 1
        for v in self.vehicles:
            self._step_vehicle(v)
        self._run_charge_dispatch_round()

    def run(self, n_ticks: int = 100):
        for _ in range(n_ticks):
            self.step()
        return self.summary()

    def summary(self) -> Dict:
        return {
            "ticks_run": self.tick,
            "vehicles": len(self.vehicles),
            "total_deliveries": sum(v.deliveries_completed for v in self.vehicles),
            "total_energy_kwh": round(sum(v.total_energy_kwh for v in self.vehicles), 2),
            "total_distance_km": round(sum(v.total_distance_km for v in self.vehicles), 2),
            "vehicles_currently_charging": sum(1 for v in self.vehicles if v.status == "charging"),
            "vehicles_seeking_charge": sum(1 for v in self.vehicles if v.status == "seeking_charge"),
            "avg_battery_soh": round(sum(v.battery_soh for v in self.vehicles) / len(self.vehicles), 3),
            "events_logged": len(self.events),
        }
