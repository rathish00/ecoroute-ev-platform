"""
h3_fusion.py
============
Spatio-Temporal Weather & Elevation Fusion Pipeline.

Most naive routing systems apply a single city-wide weather reading to every
road segment. In reality micro-climates exist: wind funnels differently
through a dense skyscraper canyon than across an open highway, and rain
pools in low-elevation valleys. This module tiles the service area into
hexagonal cells (Uber H3 when available, a deterministic offset-hex fallback
otherwise) and fuses per-cell weather + elevation signals that the routing
graph and the ML model both read from.

If the optional `h3` package is installed, real H3 indexing is used
(resolution 8, ~0.7 km^2 cells). Otherwise a lightweight pure-Python axial
hex grid with an identical interface is used, so the rest of the codebase
never needs to know which backend is active.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Tuple

try:
    import h3 as _h3

    H3_BACKEND = True
except ImportError:  # pragma: no cover - exercised in environments without h3
    H3_BACKEND = False


HEX_RESOLUTION = 8
_FALLBACK_CELL_SIZE_DEG = 0.0075  # ~ resolution-8-ish cell width


def latlng_to_cell(lat: float, lng: float) -> str:
    """Return a stable hex-cell id for a lat/lng point."""
    if H3_BACKEND:
        return _h3.latlng_to_cell(lat, lng, HEX_RESOLUTION)

    # Deterministic axial-hex fallback: snap to a hex-ish grid and hash it.
    row = round(lat / _FALLBACK_CELL_SIZE_DEG)
    col = round((lng - (row % 2) * _FALLBACK_CELL_SIZE_DEG / 2) / _FALLBACK_CELL_SIZE_DEG)
    key = f"{row}:{col}".encode()
    return "fx" + hashlib.md5(key).hexdigest()[:12]


@dataclass
class MicroClimate:
    """Fused environmental reading for a single hex cell."""

    cell_id: str
    temperature_c: float
    wind_speed_kmh: float
    wind_direction_deg: float
    precipitation_mm_hr: float
    road_wetness: float  # derived 0-1
    elevation_m: float


class WeatherElevationFusionEngine:
    """
    Builds and serves per-cell MicroClimate readings by cross-joining a
    coarse city-wide weather feed with local elevation + urban-canyon wind
    multipliers.

    In production this would pull from a live weather API + a DEM (Digital
    Elevation Model) raster; here `city_weather_feed` and `elevation_lookup`
    are injectable callables so the engine is fully testable offline.
    """

    def __init__(self, elevation_lookup=None, urban_density_lookup=None):
        self._elevation_lookup = elevation_lookup or self._default_elevation
        self._urban_density_lookup = urban_density_lookup or self._default_density
        self._cache: Dict[str, MicroClimate] = {}

    @staticmethod
    def _default_elevation(lat: float, lng: float) -> float:
        # Smooth synthetic terrain: a few overlapping sine hills, deterministic per-coordinate.
        return (
            80
            + 60 * math.sin(lat * 40) * math.cos(lng * 35)
            + 25 * math.sin(lat * 113 + lng * 71)
        )

    @staticmethod
    def _default_density(lat: float, lng: float) -> float:
        """0 = open highway, 1 = dense skyscraper canyon."""
        val = 0.5 + 0.5 * math.sin(lat * 97 + lng * 53)
        return min(max(val, 0.0), 1.0)

    def fuse(
        self,
        lat: float,
        lng: float,
        city_temperature_c: float,
        city_wind_kmh: float,
        city_wind_dir_deg: float,
        city_precip_mm_hr: float,
    ) -> MicroClimate:
        cell_id = latlng_to_cell(lat, lng)
        if cell_id in self._cache:
            return self._cache[cell_id]

        elevation = self._elevation_lookup(lat, lng)
        density = self._urban_density_lookup(lat, lng)

        # Urban canyons channel and accelerate wind locally (Venturi effect);
        # open highway exposes vehicles to the full city-wide wind reading.
        local_wind = city_wind_kmh * (1.0 + 0.4 * density)

        # Precipitation pools more in low-elevation cells relative to the
        # city's mean elevation -> higher road wetness.
        relative_low_ground = max(0.0, (110 - elevation) / 110)
        road_wetness = min(1.0, (city_precip_mm_hr / 15.0) * (0.6 + 0.4 * relative_low_ground))

        micro = MicroClimate(
            cell_id=cell_id,
            temperature_c=city_temperature_c - 0.0065 * max(elevation - 50, 0),  # lapse rate
            wind_speed_kmh=local_wind,
            wind_direction_deg=city_wind_dir_deg,
            precipitation_mm_hr=city_precip_mm_hr,
            road_wetness=road_wetness,
            elevation_m=elevation,
        )
        self._cache[cell_id] = micro
        return micro

    def headwind_component_kmh(self, micro: MicroClimate, heading_deg: float) -> float:
        """Project wind vector onto direction of travel; positive = headwind."""
        angle = math.radians(micro.wind_direction_deg - heading_deg)
        return micro.wind_speed_kmh * math.cos(angle)
