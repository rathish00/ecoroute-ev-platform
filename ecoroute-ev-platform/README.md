# ⚡ EcoRoute — Eco-Routing & Dynamic EV Charging Logistics Engine

**An enterprise-grade backend for electric-vehicle commercial fleets, fusing Graph Theory, Machine Learning, and Mathematical Optimization to minimize energy consumption, protect long-term battery health, and prevent charging-grid congestion.**

[![CI](https://img.shields.io/github/actions/workflow/status/YOUR_USERNAME/ecoroute-ev-platform/ci.yml?label=CI)](https://github.com/YOUR_USERNAME/ecoroute-ev-platform/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Why This Exists

Standard fleet-routing software treats an EV like a combustion vehicle with a smaller tank: it optimizes for distance or time and hopes the battery holds out.

That assumption breaks down fast in the real world.

A loaded van climbing a hill in freezing weather can consume significantly more energy than the same vehicle travelling on a flat highway. At the same time, a fleet of vehicles all racing to the same "nearest" charger can create queues and grid spikes instead of solving the problem.

**EcoRoute treats energy, battery health, and grid load as first-class optimization variables from the ground up.**

---

# The Three Pillars

## 📍 1. Energy-Aware Spatial Routing Engine

Models the city as a directed graph where every road segment carries a **dynamic energy cost**, not just a distance.

Routing uses Dijkstra's algorithm over dynamically computed kWh weights that incorporate:

* Slope gradient
* Traffic congestion
* Wind
* Precipitation
* Road wetness
* Local micro-climate conditions
* Vehicle and cargo characteristics

See [`ecoroute.routing`](src/ecoroute/routing).

---

## 🔋 2. Non-Linear Battery Depletion Predictor

An **XGBoost regressor** is used to predict the energy required for a specific street segment based on vehicle and environmental conditions.

The model considers:

* Cargo weight
* Temperature
* Slope
* Wind
* Traffic
* Battery State-of-Health (SOH)

If XGBoost is unavailable, the system automatically falls back to scikit-learn's `GradientBoostingRegressor`.

See [`ecoroute.ml`](src/ecoroute/ml).

---

## ⚡ 3. Centralized Smart Grid Dispatcher

A **PuLP-backed optimization engine** assigns vehicles to charging stations while considering:

* Station capacity
* Live queue wait times
* Electricity prices
* Charging requirements
* Fleet demand

The primary implementation uses the CBC solver through PuLP, with a SciPy-based fallback when required.

See [`ecoroute.optimizer`](src/ecoroute/optimizer).

---

# Extended Capabilities

| Feature                                            | Module                                       | What It Solves                                                                                                                                                              |
| -------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **🔋 State-of-Health (SOH) Guard**                 | `ecoroute.ml.battery_model.SOHPenaltyMatrix` | Penalizes or hard-blocks DC fast charging for degraded batteries, routing them toward gentler AC chargers and flatter roads.                                                |
| **🎯 Multi-Agent Grid Demand Matching**            | `ecoroute.grid_game.nash_dispatch`           | Multiple fleets sharing public charging infrastructure converge toward an approximate Nash equilibrium using sequential best-response dynamics and load-responsive pricing. |
| **🌦️ Spatio-Temporal Weather & Elevation Fusion** | `ecoroute.weather.h3_fusion`                 | Tiles the city into H3 hexagonal cells and combines local wind, precipitation, and elevation data to provide localized environmental conditions.                            |
| **🚐 Fleet Telematics Digital Twin**               | `ecoroute.simulation.digital_twin`           | Runs a live simulation sandbox of virtual EV vans delivering, consuming battery energy, and requesting charging dispatch.                                                   |

---

# Architecture

EcoRoute is organized as a modular Python backend where routing, machine learning, optimization, environmental data, and simulation operate as independent components.

See [`docs/architecture.md`](docs/architecture.md) for the full system architecture and module responsibility table.

```text
src/ecoroute/

├── utils/
│   └── physics.py
│       # Closed-form vehicle-dynamics energy model
│
├── weather/
│   └── h3_fusion.py
│       # Hex-indexed micro-climate fusion
│
├── routing/
│   ├── graph_builder.py
│   │   # Synthetic / OSM-style city graph construction
│   │
│   └── router.py
│       # Energy-aware Dijkstra routing + SOH gentle mode
│
├── ml/
│   ├── data_generator.py
│   │   # Physics-grounded synthetic telemetry
│   │
│   ├── battery_model.py
│   │   # XGBoost regressor + SOH Penalty Matrix
│   │
│   └── train.py
│       # CLI training entry point
│
├── optimizer/
│   └── grid_dispatcher.py
│       # PuLP MILP vehicle-to-charger assignment
│
├── grid_game/
│   └── nash_dispatch.py
│       # Multi-fleet congestion game
│
└── simulation/
    └── digital_twin.py
        # End-to-end fleet simulation engine
```

---

# Optional Dependency Strategy

EcoRoute is designed to remain usable even when optional heavy dependencies are unavailable.

Where possible, the project provides fallbacks for:

* `xgboost`
* `pulp`
* `h3`
* `streamlit`

The system can fall back to alternatives such as:

* scikit-learn
* SciPy
* Pure-Python implementations

This allows the core project to run in a minimal environment while providing enhanced functionality when the production dependencies are installed.

---

# Quickstart

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ecoroute-ev-platform.git
cd ecoroute-ev-platform
```

## 2. Create a Virtual Environment

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train the Battery Depletion Model

Generate physics-grounded synthetic telemetry and train the battery depletion model:

```bash
PYTHONPATH=src python -m ecoroute.ml.train --rows 20000 --output models/battery_model.bin
```

---

# Run the Fleet Digital Twin

Run a fleet simulation with 30 virtual vehicles:

```bash
python scripts/run_simulation.py \
    --vehicles 30 \
    --ticks 200 \
    --grid-rows 10 \
    --grid-cols 10
```

---

# Launch the Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

---

# Run the Test Suite

```bash
PYTHONPATH=src pytest --cov=ecoroute --cov-report=term-missing
```

---

# Minimal Usage Example

```python
from ecoroute.routing.graph_builder import from_synthetic_grid
from ecoroute.routing.router import EnergyAwareRouter, RouteContext
from ecoroute.utils.physics import VehicleProfile

# Build a synthetic city graph
graph = from_synthetic_grid(rows=10, cols=10)

# Initialize the energy-aware router
router = EnergyAwareRouter(graph)

# Define the routing context
ctx = RouteContext(
    vehicle=VehicleProfile(),
    cargo_kg=350,
    city_temperature_c=12,
    battery_soh=0.79,
    gentle_mode=True,
)

# Calculate the lowest-energy route
route = router.shortest_energy_path(
    (0, 0),
    (9, 9),
    ctx,
)

print(
    route.total_energy_kwh,
    "kWh over",
    route.total_distance_km,
    "km",
)
```

The example demonstrates the **SOH Guard** in action:

```text
Battery SOH = 79%
Gentle Mode = enabled
Cargo = 350 kg
Temperature = 12°C
```

The router can therefore incorporate battery-health constraints while calculating the energy-efficient path.

---

# System Flow

```text
                    ┌─────────────────────┐
                    │   Fleet Vehicles    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Vehicle Telemetry   │
                    │ Cargo / Battery SOH │
                    └──────────┬──────────┘
                               │
                               ▼
             ┌────────────────────────────────┐
             │ Energy Prediction / ML Model   │
             │                                │
             │ XGBoost / GradientBoosting     │
             └───────────────┬────────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Energy-Aware Router    │
                 │                        │
                 │ Dynamic Dijkstra       │
                 └────────────┬───────────┘
                              │
                              ▼
                  ┌──────────────────────┐
                  │ Charging Requirement  │
                  └──────────┬───────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Smart Grid Dispatcher       │
              │                             │
              │ MILP / Optimization         │
              └──────────────┬──────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Charging Allocation  │
                  └──────────────────────┘
```

---

# Roadmap

* [ ] Replace the synthetic-grid graph builder with a real **OSMnx + DEM** ingestion pipeline
* [ ] Add live weather API integration for `h3_fusion`
* [ ] Add live electricity/grid-pricing API integration
* [ ] Persist digital-twin telemetry in a time-series database such as TimescaleDB
* [ ] Replace the best-response Nash approximation with a full mechanism-design auction for very large fleets
* [ ] Add real-world fleet telemetry ingestion
* [ ] Add historical energy-consumption analytics
* [ ] Add production-grade route visualization

---

# Testing

The project includes automated tests covering core functionality.

Run:

```bash
PYTHONPATH=src pytest
```

For coverage:

```bash
PYTHONPATH=src pytest \
    --cov=ecoroute \
    --cov-report=term-missing
```

Before submitting a pull request, run:

```bash
pytest
ruff check src tests
```

---

# Contributing

Issues and pull requests are welcome.

Before submitting a contribution:

1. Run the test suite.
2. Run Ruff.
3. Keep modules focused and maintainable.
4. Add tests for new functionality.
5. Update documentation when behavior changes.

---

# License

This project is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## Project Vision

EcoRoute is designed around a simple idea:

> **For electric fleets, the shortest route is not always the cheapest route.**

A truly intelligent EV logistics system needs to understand the interaction between:

**Roads + Energy + Weather + Traffic + Battery Health + Charging Infrastructure + Grid Demand**

EcoRoute brings those components together into a single optimization-oriented engine.
