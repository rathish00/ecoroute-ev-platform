"""
graph_builder.py
=================
Builds a synthetic (or OSM-derived) city road network as a `networkx.DiGraph`
where every edge carries the attributes needed to compute a dynamic
"energy cost": length, slope, speed limit, live traffic index, and a
reference to its fused micro-climate cell.

In production, `from_osm()` would ingest an OSMnx-extracted graph and
attach a DEM-derived slope + live traffic-API feed per edge. Here we ship a
deterministic synthetic-city generator (`from_synthetic_grid`) so the whole
pipeline runs end-to-end with zero external dependencies or API keys --
useful for demos, tests, and CI.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import networkx as nx

from ecoroute.weather.h3_fusion import WeatherElevationFusionEngine


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def from_synthetic_grid(
    rows: int = 12,
    cols: int = 12,
    lat0: float = 13.0827,
    lng0: float = 80.2707,
    cell_deg: float = 0.006,
    seed: int = 42,
    fusion_engine: Optional[WeatherElevationFusionEngine] = None,
) -> nx.DiGraph:
    """
    Generate a deterministic grid-city road network (Manhattan-style grid
    plus a handful of diagonal arterials) with realistic per-edge slope,
    speed-limit and elevation attributes. Coordinates default to a Chennai-
    sized bounding box but are purely illustrative.
    """
    rng = random.Random(seed)
    fusion = fusion_engine or WeatherElevationFusionEngine()
    g = nx.DiGraph()

    for r in range(rows):
        for c in range(cols):
            lat = lat0 + r * cell_deg
            lng = lng0 + c * cell_deg
            elevation = fusion._elevation_lookup(lat, lng)
            g.add_node((r, c), lat=lat, lng=lng, elevation_m=elevation)

    def add_road(n1, n2, road_class="local"):
        lat1, lng1 = g.nodes[n1]["lat"], g.nodes[n1]["lng"]
        lat2, lng2 = g.nodes[n2]["lat"], g.nodes[n2]["lng"]
        dist_km = max(_haversine_km(lat1, lng1, lat2, lng2), 0.05)
        elev1, elev2 = g.nodes[n1]["elevation_m"], g.nodes[n2]["elevation_m"]
        slope_percent = 100.0 * (elev2 - elev1) / (dist_km * 1000.0)

        speed_limit = {"arterial": 60, "local": 40}[road_class]
        base_traffic = rng.uniform(0.05, 0.55) if road_class == "local" else rng.uniform(0.15, 0.75)
        heading = math.degrees(math.atan2(lng2 - lng1, lat2 - lat1)) % 360

        g.add_edge(
            n1,
            n2,
            distance_km=round(dist_km, 4),
            slope_percent=round(slope_percent, 3),
            speed_limit_kmh=speed_limit,
            road_class=road_class,
            base_traffic_index=round(base_traffic, 3),
            heading_deg=round(heading, 2),
        )

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                road_class = "arterial" if r % 4 == 0 else "local"
                add_road((r, c), (r, c + 1), road_class)
                add_road((r, c + 1), (r, c), road_class)
            if r + 1 < rows:
                road_class = "arterial" if c % 4 == 0 else "local"
                add_road((r, c), (r + 1, c), road_class)
                add_road((r + 1, c), (r, c), road_class)
            # sparse diagonal arterials for extra route diversity
            if r + 1 < rows and c + 1 < cols and (r + c) % 5 == 0:
                add_road((r, c), (r + 1, c + 1), "arterial")
                add_road((r + 1, c + 1), (r, c), "arterial")

    g.graph["fusion_engine"] = fusion
    g.graph["grid_shape"] = (rows, cols)
    return g


def node_id(row: int, col: int):
    return (row, col)
