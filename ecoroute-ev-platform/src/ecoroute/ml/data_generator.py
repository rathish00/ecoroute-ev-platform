"""
data_generator.py
==================
Generates a physics-grounded synthetic telemetry dataset used to train the
Non-Linear Battery Depletion Predictor. Real deployments would replace this
with historical CAN-bus / fleet telematics logs; the schema here is designed
to match what a real telematics feed would provide, so swapping in real data
later requires no code changes downstream.
"""

from __future__ import annotations

import random
from typing import Optional

import pandas as pd

from ecoroute.utils.physics import VehicleProfile, segment_energy_kwh

FEATURE_COLUMNS = [
    "distance_km",
    "avg_speed_kmh",
    "slope_percent",
    "temperature_c",
    "cargo_kg",
    "headwind_kmh",
    "road_wetness",
    "traffic_index",
    "battery_soh",
]
TARGET_COLUMN = "energy_kwh"


def generate_telemetry(n_rows: int = 20000, seed: int = 7, vehicle: Optional[VehicleProfile] = None) -> pd.DataFrame:
    """Sample plausible driving-segment conditions and label them with the
    ground-truth physics energy cost (plus small sensor noise), producing a
    supervised-learning dataset for the regressor."""
    rng = random.Random(seed)
    vehicle = vehicle or VehicleProfile()
    rows = []

    for _ in range(n_rows):
        distance_km = rng.uniform(0.1, 6.0)
        avg_speed_kmh = rng.uniform(8, 95)
        slope_percent = rng.uniform(-12, 12)
        temperature_c = rng.uniform(-8, 45)
        cargo_kg = rng.uniform(0, vehicle.max_cargo_kg)
        headwind_kmh = rng.uniform(-25, 25)
        road_wetness = rng.uniform(0, 1)
        traffic_index = rng.uniform(0, 1)
        battery_soh = rng.uniform(0.65, 1.0)

        true_energy = segment_energy_kwh(
            vehicle=vehicle,
            distance_km=distance_km,
            avg_speed_kmh=avg_speed_kmh,
            slope_percent=slope_percent,
            temperature_c=temperature_c,
            cargo_kg=cargo_kg,
            headwind_kmh=headwind_kmh,
            road_wetness=road_wetness,
            traffic_index=traffic_index,
            battery_soh=battery_soh,
        )
        # +/- sensor & driver-behaviour noise (~4% relative std)
        noisy_energy = max(0.0, rng.gauss(true_energy, 0.04 * max(true_energy, 0.05)))

        rows.append(
            {
                "distance_km": round(distance_km, 4),
                "avg_speed_kmh": round(avg_speed_kmh, 2),
                "slope_percent": round(slope_percent, 3),
                "temperature_c": round(temperature_c, 2),
                "cargo_kg": round(cargo_kg, 1),
                "headwind_kmh": round(headwind_kmh, 2),
                "road_wetness": round(road_wetness, 3),
                "traffic_index": round(traffic_index, 3),
                "battery_soh": round(battery_soh, 3),
                "energy_kwh": round(noisy_energy, 5),
            }
        )

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS + [TARGET_COLUMN])


if __name__ == "__main__":
    df = generate_telemetry()
    print(df.describe())
