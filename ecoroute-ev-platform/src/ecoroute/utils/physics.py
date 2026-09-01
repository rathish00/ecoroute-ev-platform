"""
physics.py
==========
Ground-truth physics formulas used both to *generate* realistic synthetic
telemetry (for ML training) and to provide a fast analytical fallback for
energy-cost estimation when a trained model is not available.

The model follows the standard longitudinal vehicle dynamics equation:

    F_total = F_rolling + F_aerodynamic + F_gradient + F_inertial

    P = F_total * v / eta_drivetrain

Energy for a segment is P integrated over time, expressed in kWh.

None of this is meant to be a certified automotive simulator -- it is a
physically-plausible generator so that the downstream ML model has to learn
genuine non-linear structure (as it would from real fleet telemetry) rather
than pure noise.
"""

from __future__ import annotations

from dataclasses import dataclass

GRAVITY = 9.81  # m/s^2
AIR_DENSITY = 1.225  # kg/m^3 at sea level, 15C


@dataclass
class VehicleProfile:
    """Static physical parameters of a commercial EV van/truck."""

    mass_kg: float = 3200.0  # curb weight
    max_cargo_kg: float = 1200.0
    drag_coefficient: float = 0.36
    frontal_area_m2: float = 5.2
    rolling_resistance_coeff: float = 0.010
    drivetrain_efficiency: float = 0.88
    regen_efficiency: float = 0.55  # fraction of braking energy recovered
    battery_capacity_kwh: float = 82.0


def air_density(temperature_c: float) -> float:
    """Rough ideal-gas correction of air density for ambient temperature."""
    kelvin = temperature_c + 273.15
    return AIR_DENSITY * (288.15 / kelvin)


def rolling_resistance_force(vehicle: VehicleProfile, cargo_kg: float, road_wetness: float) -> float:
    total_mass = vehicle.mass_kg + cargo_kg
    # Wet/slick roads increase effective rolling resistance up to ~35%
    wetness_multiplier = 1.0 + 0.35 * road_wetness
    return vehicle.rolling_resistance_coeff * wetness_multiplier * total_mass * GRAVITY


def aerodynamic_force(vehicle: VehicleProfile, speed_mps: float, headwind_mps: float, temperature_c: float) -> float:
    relative_speed = max(speed_mps + headwind_mps, 0.0)
    rho = air_density(temperature_c)
    return 0.5 * rho * vehicle.drag_coefficient * vehicle.frontal_area_m2 * relative_speed**2


def gradient_force(vehicle: VehicleProfile, cargo_kg: float, slope_percent: float) -> float:
    total_mass = vehicle.mass_kg + cargo_kg
    # slope_percent -> approximate angle in radians for road grades (valid for |slope| < ~30%)
    import math

    theta = math.atan(slope_percent / 100.0)
    return total_mass * GRAVITY * math.sin(theta)


def segment_energy_kwh(
    vehicle: VehicleProfile,
    distance_km: float,
    avg_speed_kmh: float,
    slope_percent: float,
    temperature_c: float,
    cargo_kg: float = 0.0,
    headwind_kmh: float = 0.0,
    road_wetness: float = 0.0,
    traffic_index: float = 0.0,
    battery_soh: float = 1.0,
) -> float:
    """
    Estimate the kWh consumed traversing a road segment.

    traffic_index in [0, 1]: 0 = free flow, 1 = gridlock (adds stop/start
        inertial losses and idling HVAC draw).
    battery_soh in (0, 1]: degraded cells have higher internal resistance,
        modeled as an efficiency penalty.
    """
    speed_mps = max(avg_speed_kmh, 1.0) / 3.6
    headwind_mps = headwind_kmh / 3.6

    f_roll = rolling_resistance_force(vehicle, cargo_kg, road_wetness)
    f_aero = aerodynamic_force(vehicle, speed_mps, headwind_mps, temperature_c)
    f_grad = gradient_force(vehicle, cargo_kg, slope_percent)

    f_total = f_roll + f_aero + max(f_grad, 0.0)
    # Regenerative braking claws back part of the energy on descents
    regen_recovery = max(-f_grad, 0.0) * vehicle.regen_efficiency

    power_w = (f_total * speed_mps) / vehicle.drivetrain_efficiency
    time_h = distance_km / max(avg_speed_kmh, 1.0)
    energy_kwh = (power_w * time_h) / 1000.0
    energy_kwh -= (regen_recovery * speed_mps * time_h) / 1000.0

    # Stop-and-go traffic burns extra energy re-accelerating + climate control idling
    traffic_penalty = traffic_index * 0.18 * distance_km / 10.0
    # Cold-weather battery chemistry + cabin heating draw
    cold_penalty = max(0.0, (5.0 - temperature_c)) * 0.004 * distance_km

    energy_kwh += traffic_penalty + cold_penalty

    # Degraded batteries (low SOH) waste more energy as internal heat
    soh_penalty_factor = 1.0 + (1.0 - battery_soh) * 0.25
    energy_kwh *= soh_penalty_factor

    return max(energy_kwh, 0.0)
