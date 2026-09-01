#!/usr/bin/env python3
"""
run_simulation.py
==================
Command-line entry point for the Fleet Telematics Digital Twin.

Usage:
    python scripts/run_simulation.py --vehicles 30 --ticks 200 --grid 10x10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecoroute.routing.graph_builder import from_synthetic_grid
from ecoroute.simulation.digital_twin import DigitalTwinSimulation


def main():
    parser = argparse.ArgumentParser(description="Run the EV fleet digital-twin simulation")
    parser.add_argument("--vehicles", type=int, default=25)
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--events-out", type=str, default="")
    args = parser.parse_args()

    graph = from_synthetic_grid(rows=args.grid_rows, cols=args.grid_cols)
    sim = DigitalTwinSimulation(graph, n_vehicles=args.vehicles, seed=args.seed)

    print(f"Running digital twin: {args.vehicles} vehicles, {args.ticks} ticks, "
          f"{args.grid_rows}x{args.grid_cols} grid...")
    summary = sim.run(n_ticks=args.ticks)
    print(json.dumps(summary, indent=2))

    if args.events_out:
        out_path = Path(args.events_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for e in sim.events:
                f.write(json.dumps({"tick": e.tick, "vehicle_id": e.vehicle_id, "event_type": e.event_type, **e.details}) + "\n")
        print(f"Wrote {len(sim.events)} events -> {out_path}")


if __name__ == "__main__":
    main()
