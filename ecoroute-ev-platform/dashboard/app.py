"""
dashboard/app.py
=================
Streamlit operations dashboard for the Eco-Routing & Dynamic EV Charging
Logistics Engine. Provides three views mirroring the three pillars:

1. Route Explorer   -- compare energy-aware vs. shortest-distance routes.
2. Battery Model    -- train/inspect the non-linear depletion predictor.
3. Fleet Simulation -- run the digital-twin simulation and inspect KPIs.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st

from ecoroute.grid_game.nash_dispatch import ChargeRequest, MultiFleetNashDispatcher, StationSlotState
from ecoroute.ml.battery_model import BatteryDepletionModel, SOHPenaltyMatrix
from ecoroute.ml.data_generator import generate_telemetry
from ecoroute.routing.graph_builder import from_synthetic_grid
from ecoroute.routing.router import EnergyAwareRouter, RouteContext
from ecoroute.simulation.digital_twin import DigitalTwinSimulation
from ecoroute.utils.physics import VehicleProfile

st.set_page_config(page_title="EcoRoute EV Logistics Engine", layout="wide")

st.title("⚡ Eco-Routing & Dynamic EV Charging Logistics Engine")
st.caption("Energy-aware routing · Non-linear battery ML · Grid-aware charge dispatch · SOH-Guard · Multi-fleet game theory")


@st.cache_resource
def get_graph(rows: int, cols: int):
    return from_synthetic_grid(rows=rows, cols=cols)


@st.cache_resource
def get_trained_model(rows: int, seed: int):
    df = generate_telemetry(n_rows=rows, seed=seed)
    model = BatteryDepletionModel()
    metrics = model.fit(df)
    return model, metrics, df


tab_routing, tab_battery, tab_grid, tab_sim = st.tabs(
    ["📍 Route Explorer", "🔋 Battery Model", "⚡ Grid Dispatch & SOH", "🪟 Fleet Digital Twin"]
)

# ---------------------------------------------------------------------
with tab_routing:
    st.subheader("Energy-Aware Spatial Routing Engine")
    col1, col2 = st.columns(2)
    with col1:
        rows = st.slider("Grid rows", 4, 16, 8)
        cols = st.slider("Grid cols", 4, 16, 8)
        cargo_kg = st.slider("Cargo (kg)", 0, 1200, 300)
        soh = st.slider("Battery SOH", 0.5, 1.0, 0.9)
        temperature = st.slider("Ambient temperature (°C)", -10, 45, 22)
    with col2:
        origin_r = st.number_input("Origin row", 0, rows - 1, 0)
        origin_c = st.number_input("Origin col", 0, cols - 1, 0)
        dest_r = st.number_input("Destination row", 0, rows - 1, rows - 1)
        dest_c = st.number_input("Destination col", 0, cols - 1, cols - 1)

    graph = get_graph(rows, cols)
    router = EnergyAwareRouter(graph)
    ctx = RouteContext(
        vehicle=VehicleProfile(),
        cargo_kg=cargo_kg,
        city_temperature_c=temperature,
        battery_soh=soh,
        gentle_mode=SOHPenaltyMatrix().requires_gentle_routing(soh),
    )

    if st.button("Compute routes"):
        energy_result = router.shortest_energy_path((origin_r, origin_c), (dest_r, dest_c), ctx)
        distance_result = router.shortest_distance_path((origin_r, origin_c), (dest_r, dest_c))

        m1, m2, m3 = st.columns(3)
        m1.metric("Energy-optimal cost", f"{energy_result.total_energy_kwh} kWh")
        m2.metric("Energy-optimal distance", f"{energy_result.total_distance_km} km")
        m3.metric("Shortest-distance path length", f"{distance_result.total_distance_km} km")

        st.write("Energy-optimal path:", " → ".join(str(n) for n in energy_result.path))
        st.write("Shortest-distance path:", " → ".join(str(n) for n in distance_result.path))
        st.dataframe(pd.DataFrame(energy_result.segment_breakdown))

# ---------------------------------------------------------------------
with tab_battery:
    st.subheader("Non-Linear Battery Depletion Predictor")
    n_rows = st.slider("Synthetic training rows", 2000, 30000, 8000, step=1000)

    if st.button("Train model"):
        model, metrics, df = get_trained_model(n_rows, 7)
        c1, c2, c3 = st.columns(3)
        c1.metric("Backend", metrics["backend"])
        c2.metric("MAE (kWh)", f"{metrics['mae_kwh']:.4f}")
        c3.metric("R²", f"{metrics['r2']:.4f}")

        st.write("Feature importance:")
        importance_df = pd.DataFrame(model.feature_importance().items(), columns=["feature", "importance"])
        st.bar_chart(importance_df.set_index("feature"))

        st.write("Sample training rows:")
        st.dataframe(df.head(20))

# ---------------------------------------------------------------------
with tab_grid:
    st.subheader("SOH Penalty Matrix")
    soh_probe = st.slider("Probe SOH value", 0.5, 1.0, 0.8, key="soh_probe")
    guard = SOHPenaltyMatrix()
    penalties = guard.penalty_matrix(soh_probe)
    st.table(pd.DataFrame(penalties.items(), columns=["charger_type", "cost_penalty_multiplier"]))
    st.info(
        f"Gentle routing mode {'REQUIRED' if guard.requires_gentle_routing(soh_probe) else 'not required'} "
        f"at SOH={soh_probe}."
    )

    st.subheader("Multi-Fleet Nash Congestion Game")
    st.caption("Two competing fleets both want the same peak charging slot -- watch equilibrium spread demand.")
    if st.button("Run congestion game"):
        stations = {
            "st_A": StationSlotState(base_price_per_kwh=0.15, capacity_kw=60, congestion_sensitivity=0.05),
            "st_B": StationSlotState(base_price_per_kwh=0.17, capacity_kw=60, congestion_sensitivity=0.05),
        }
        requests = [
            ChargeRequest("fleetA", "v1", "st_A", preferred_slot=10, flexibility=2, energy_needed_kwh=40),
            ChargeRequest("fleetA", "v2", "st_A", preferred_slot=10, flexibility=2, energy_needed_kwh=40),
            ChargeRequest("fleetB", "v1", "st_A", preferred_slot=10, flexibility=3, energy_needed_kwh=35),
            ChargeRequest("fleetB", "v2", "st_A", preferred_slot=11, flexibility=1, energy_needed_kwh=35),
            ChargeRequest("fleetA", "v3", "st_B", preferred_slot=10, flexibility=2, energy_needed_kwh=30),
        ]
        dispatcher = MultiFleetNashDispatcher(stations)
        result = dispatcher.solve(requests)
        st.write(f"Converged: {result.converged} in {result.iterations} iterations")
        st.dataframe(pd.DataFrame([s.__dict__ for s in result.settled]))

# ---------------------------------------------------------------------
with tab_sim:
    st.subheader("Fleet Telematics Digital Twin")
    n_vehicles = st.slider("Fleet size", 5, 60, 20)
    n_ticks = st.slider("Simulation ticks", 20, 400, 150)

    if st.button("Run simulation"):
        graph = get_graph(10, 10)
        sim = DigitalTwinSimulation(graph, n_vehicles=n_vehicles)
        with st.spinner("Simulating fleet telematics..."):
            summary = sim.run(n_ticks=n_ticks)

        cols = st.columns(4)
        cols[0].metric("Deliveries", summary["total_deliveries"])
        cols[1].metric("Total energy (kWh)", summary["total_energy_kwh"])
        cols[2].metric("Total distance (km)", summary["total_distance_km"])
        cols[3].metric("Avg battery SOH", summary["avg_battery_soh"])

        events_df = pd.DataFrame(
            [{"tick": e.tick, "vehicle_id": e.vehicle_id, "event_type": e.event_type, **e.details} for e in sim.events]
        )
        st.write("Event type distribution:")
        st.bar_chart(events_df["event_type"].value_counts())
        st.write("Raw event log (last 200 rows):")
        st.dataframe(events_df.tail(200))
