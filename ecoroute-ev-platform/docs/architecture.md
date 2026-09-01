# Architecture

## System overview

```
                 ┌──────────────────────────────────────┐
                 │   SPATIAL DATA ENGINEERING LAYER      │
                 │   ecoroute.routing.graph_builder      │
                 │   ecoroute.weather.h3_fusion          │
                 │   -> city graph + hex micro-climates  │
                 └───────────────────┬────────────────────┘
                                     │ per-edge features
                                     ▼
                 ┌──────────────────────────────────────┐
                 │   ENERGY-AWARE ROUTING ENGINE          │
                 │   ecoroute.routing.router              │
                 │   -> Dijkstra over dynamic kWh weights │
                 └───────────────────┬────────────────────┘
                                     │ cost function
                                     ▼
                 ┌──────────────────────────────────────┐
                 │   NON-LINEAR BATTERY DEPLETION MODEL   │
                 │   ecoroute.ml.battery_model (XGBoost)  │
                 │   ecoroute.ml.data_generator           │
                 │   + SOH Penalty Matrix (SOH-Guard)      │
                 └───────────────────┬────────────────────┘
                                     │ SoC / SOH state
                                     ▼
                 ┌──────────────────────────────────────┐
                 │   CENTRALIZED SMART GRID DISPATCHER    │
                 │   ecoroute.optimizer.grid_dispatcher   │
                 │   (PuLP MILP: vehicle -> station)      │
                 │   ecoroute.grid_game.nash_dispatch     │
                 │   (multi-fleet congestion game)         │
                 └───────────────────┬────────────────────┘
                                     │ assignments / events
                                     ▼
                 ┌──────────────────────────────────────┐
                 │   FLEET DIGITAL TWIN SIMULATION        │
                 │   ecoroute.simulation.digital_twin      │
                 │   -> orchestrates all of the above       │
                 └───────────────────┬────────────────────┘
                                     │
                                     ▼
                 ┌──────────────────────────────────────┐
                 │   STREAMLIT OPERATIONS DASHBOARD        │
                 │   dashboard/app.py                     │
                 └──────────────────────────────────────┘
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `ecoroute.utils.physics` | Closed-form longitudinal vehicle-dynamics model (rolling resistance, aero drag, gradient force, regen braking, cold-weather and SOH penalties). Used both to label synthetic training data and as a fast fallback cost function. |
| `ecoroute.weather.h3_fusion` | Tiles the service area into hex cells (H3 when installed, deterministic fallback otherwise) and fuses per-cell wind, precipitation, road-wetness and elevation signals. |
| `ecoroute.routing.graph_builder` | Builds the city road network as a `networkx.DiGraph`; ships a deterministic synthetic-grid generator so the whole stack runs with zero external data dependencies. |
| `ecoroute.routing.router` | `EnergyAwareRouter` runs Dijkstra with a pluggable, per-trip cost function (`physics_cost_fn` or `ml_cost_fn`), including an SOH-aware "gentle mode" that avoids high-peak-current segments. |
| `ecoroute.ml.data_generator` | Generates physics-grounded synthetic telemetry for supervised training (schema-compatible with real fleet telematics). |
| `ecoroute.ml.battery_model` | `BatteryDepletionModel` (XGBoost, sklearn-GBR fallback) + `SOHPenaltyMatrix` (the SOH-Guard: blocks/penalizes DC fast charging for degraded batteries). |
| `ecoroute.optimizer.grid_dispatcher` | Single-fleet vehicle-to-station assignment as a MILP (PuLP/CBC, scipy-greedy fallback), respecting station capacity and SOH-Guard constraints. |
| `ecoroute.grid_game.nash_dispatch` | Multi-fleet charging-slot congestion game; sequential best-response dynamics converge to an approximate Nash equilibrium with load-responsive pricing. |
| `ecoroute.simulation.digital_twin` | Orchestrates N virtual vehicles driving, consuming battery, and requesting charge dispatch, producing a continuous telemetry event stream for validation and demos. |
| `dashboard/app.py` | Streamlit UI exposing all of the above interactively. |

## Design principles

1. **Dependency-optional by design.** Every optional heavy dependency
   (`xgboost`, `pulp`, `h3`, `streamlit`) has a pure-Python or scikit-learn/
   scipy fallback, detected at import time. The platform is fully
   functional and testable in a minimal environment, and automatically
   upgrades to the production-grade backend when it's installed.
2. **Schema-first synthetic data.** The synthetic telemetry generator uses
   the exact feature schema a real CAN-bus/telematics feed would provide,
   so swapping in production data requires no changes to `battery_model.py`
   or `router.py`.
3. **Composable cost functions.** The router doesn't hard-code physics or
   ML -- it accepts any `CostFunction`, so new energy models (e.g. a
   neural spatio-temporal model) plug in without touching graph or
   shortest-path code.
4. **Explainable optimization over black-box game solving.** The
   multi-fleet dispatcher uses sequential best-response dynamics (a
   provably-converging procedure for congestion/potential games) rather
   than a general-purpose equilibrium solver, keeping dispatch latency
   predictable and behaviour auditable.
