"""
router.py
=========
Energy-Aware Spatial Routing Engine (Pillar 1).

Wraps a `networkx.DiGraph` (see `graph_builder.py`) and computes shortest
paths where edge weight = predicted kWh cost rather than distance or time.
The per-edge energy cost is supplied by a pluggable `CostFunction` --
either the closed-form physics model (`physics.segment_energy_kwh`, fast,
zero-dependency) or a trained ML regressor (`ml.battery_model`, higher
fidelity). This lets the same routing code serve both a cold-start /
offline demo mode and a production ML-backed mode.

State-of-Health (SOH) awareness: when a vehicle's battery SOH drops below
`soh_fast_charge_threshold`, the router additionally exposes a "gentle
mode" cost function that penalizes high-slope / high-speed segments known
to draw high peak current, in favour of flatter, calmer routes -- feeding
directly into the SOH-aware dispatch decision made downstream by the
optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx

from ecoroute.utils.physics import VehicleProfile, segment_energy_kwh
from ecoroute.weather.h3_fusion import WeatherElevationFusionEngine

CostFunction = Callable[[nx.DiGraph, Tuple, Tuple, Dict, "RouteContext"], float]


@dataclass
class RouteContext:
    """Everything about *this specific trip* that the cost function needs."""

    vehicle: VehicleProfile
    cargo_kg: float = 0.0
    city_temperature_c: float = 28.0
    city_wind_kmh: float = 8.0
    city_wind_dir_deg: float = 90.0
    city_precip_mm_hr: float = 0.0
    battery_soh: float = 1.0
    gentle_mode: bool = False  # True => avoid high peak-current segments
    ml_model: Optional[object] = None  # trained BatteryDepletionModel, optional


def physics_cost_fn(g: nx.DiGraph, u: Tuple, v: Tuple, edge: Dict, ctx: RouteContext) -> float:
    fusion: WeatherElevationFusionEngine = g.graph.get("fusion_engine") or WeatherElevationFusionEngine()
    micro = fusion.fuse(
        g.nodes[u]["lat"],
        g.nodes[u]["lng"],
        ctx.city_temperature_c,
        ctx.city_wind_kmh,
        ctx.city_wind_dir_deg,
        ctx.city_precip_mm_hr,
    )
    headwind = fusion.headwind_component_kmh(micro, edge["heading_deg"])

    energy = segment_energy_kwh(
        vehicle=ctx.vehicle,
        distance_km=edge["distance_km"],
        avg_speed_kmh=edge["speed_limit_kmh"] * (1.0 - 0.5 * edge["base_traffic_index"]),
        slope_percent=edge["slope_percent"],
        temperature_c=micro.temperature_c,
        cargo_kg=ctx.cargo_kg,
        headwind_kmh=headwind,
        road_wetness=micro.road_wetness,
        traffic_index=edge["base_traffic_index"],
        battery_soh=ctx.battery_soh,
    )

    if ctx.gentle_mode:
        # SOH-Guard: penalize segments that would demand high peak current
        # (steep positive grade at speed) to protect a degraded battery.
        peak_current_risk = max(edge["slope_percent"], 0) * edge["speed_limit_kmh"] / 100.0
        energy *= 1.0 + 0.35 * peak_current_risk

    return max(energy, 1e-6)


def ml_cost_fn(g: nx.DiGraph, u: Tuple, v: Tuple, edge: Dict, ctx: RouteContext) -> float:
    """Cost function backed by a trained BatteryDepletionModel."""
    if ctx.ml_model is None:
        return physics_cost_fn(g, u, v, edge, ctx)

    fusion: WeatherElevationFusionEngine = g.graph.get("fusion_engine") or WeatherElevationFusionEngine()
    micro = fusion.fuse(
        g.nodes[u]["lat"],
        g.nodes[u]["lng"],
        ctx.city_temperature_c,
        ctx.city_wind_kmh,
        ctx.city_wind_dir_deg,
        ctx.city_precip_mm_hr,
    )
    headwind = fusion.headwind_component_kmh(micro, edge["heading_deg"])

    features = {
        "distance_km": edge["distance_km"],
        "avg_speed_kmh": edge["speed_limit_kmh"] * (1.0 - 0.5 * edge["base_traffic_index"]),
        "slope_percent": edge["slope_percent"],
        "temperature_c": micro.temperature_c,
        "cargo_kg": ctx.cargo_kg,
        "headwind_kmh": headwind,
        "road_wetness": micro.road_wetness,
        "traffic_index": edge["base_traffic_index"],
        "battery_soh": ctx.battery_soh,
    }
    energy = float(ctx.ml_model.predict_one(features))

    if ctx.gentle_mode:
        peak_current_risk = max(edge["slope_percent"], 0) * edge["speed_limit_kmh"] / 100.0
        energy *= 1.0 + 0.35 * peak_current_risk

    return max(energy, 1e-6)


@dataclass
class RouteResult:
    path: List[Tuple]
    total_energy_kwh: float
    total_distance_km: float
    segment_breakdown: List[Dict] = field(default_factory=list)


class EnergyAwareRouter:
    """Thin, testable wrapper around networkx shortest-path with a dynamic
    energy weight function."""

    def __init__(self, graph: nx.DiGraph, cost_fn: CostFunction = physics_cost_fn):
        self.graph = graph
        self.cost_fn = cost_fn

    def shortest_energy_path(self, origin: Tuple, destination: Tuple, ctx: RouteContext) -> RouteResult:
        def weight(u, v, edge_data):
            return self.cost_fn(self.graph, u, v, edge_data, ctx)

        path = nx.dijkstra_path(self.graph, origin, destination, weight=weight)
        total_energy = 0.0
        total_distance = 0.0
        breakdown = []
        for u, v in zip(path[:-1], path[1:]):
            edge = self.graph.edges[u, v]
            e_kwh = self.cost_fn(self.graph, u, v, edge, ctx)
            total_energy += e_kwh
            total_distance += edge["distance_km"]
            breakdown.append({"from": u, "to": v, "energy_kwh": round(e_kwh, 4), "distance_km": edge["distance_km"]})

        return RouteResult(
            path=path,
            total_energy_kwh=round(total_energy, 4),
            total_distance_km=round(total_distance, 3),
            segment_breakdown=breakdown,
        )

    def shortest_distance_path(self, origin: Tuple, destination: Tuple) -> RouteResult:
        """Baseline comparator: classic shortest-distance routing, for
        reporting how much energy the energy-aware router saves."""
        path = nx.dijkstra_path(self.graph, origin, destination, weight="distance_km")
        total_distance = sum(self.graph.edges[u, v]["distance_km"] for u, v in zip(path[:-1], path[1:]))
        return RouteResult(path=path, total_energy_kwh=-1.0, total_distance_km=round(total_distance, 3))
