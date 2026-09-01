#!/usr/bin/env python3
"""
train.py
========
CLI entry point: generates synthetic telemetry, trains the
BatteryDepletionModel, prints validation metrics, and saves the trained
model + a copy of the training data under `models/` and `data/sample/`.

Usage:
    python -m ecoroute.ml.train --rows 20000 --output models/battery_model.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecoroute.ml.battery_model import BatteryDepletionModel
from ecoroute.ml.data_generator import generate_telemetry


def main():
    parser = argparse.ArgumentParser(description="Train the Non-Linear Battery Depletion Predictor")
    parser.add_argument("--rows", type=int, default=20000, help="Number of synthetic telemetry rows to generate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="models/battery_model.bin")
    parser.add_argument("--save-data", type=str, default="data/sample/telemetry.csv")
    args = parser.parse_args()

    print(f"Generating {args.rows} rows of synthetic physics-grounded telemetry...")
    df = generate_telemetry(n_rows=args.rows, seed=args.seed)

    if args.save_data:
        out_path = Path(args.save_data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved training data -> {out_path}")

    print("Training BatteryDepletionModel...")
    model = BatteryDepletionModel()
    metrics = model.fit(df)
    print("Validation metrics:")
    print(json.dumps(metrics, indent=2))

    model.save(args.output)
    print(f"Saved trained model -> {args.output}")

    print("\nTop feature importances:")
    for feature, score in list(model.feature_importance().items())[:5]:
        print(f"  {feature:>16s}: {score:.4f}")


if __name__ == "__main__":
    main()
