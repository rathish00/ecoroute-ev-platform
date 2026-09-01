import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from ecoroute.ml.battery_model import BatteryDepletionModel, SOHPenaltyMatrix
from ecoroute.ml.data_generator import generate_telemetry


@pytest.fixture(scope="module")
def small_dataset():
    return generate_telemetry(n_rows=2500, seed=99)


def test_model_trains_and_generalizes(small_dataset):
    model = BatteryDepletionModel()
    metrics = model.fit(small_dataset)
    assert metrics["r2"] > 0.9, "Regressor should explain the vast majority of variance on synthetic physics data"
    assert metrics["mae_kwh"] < 0.5


def test_predict_one_matches_batch_predict(small_dataset):
    model = BatteryDepletionModel()
    model.fit(small_dataset)
    row = small_dataset.iloc[0]
    features = {c: row[c] for c in model.feature_columns}
    single_pred = model.predict_one(features)
    batch_pred = model.predict(small_dataset.iloc[[0]])[0]
    assert abs(single_pred - batch_pred) < 1e-6


def test_soh_guard_hard_blocks_severely_degraded_battery_from_fast_charging():
    guard = SOHPenaltyMatrix()
    assert guard.charger_penalty(0.5, "dc_fast_150kw") == float("inf")
    assert guard.charger_penalty(0.5, "ac_level2_11kw") == 1.0


def test_soh_guard_penalty_increases_monotonically_as_soh_degrades():
    guard = SOHPenaltyMatrix()
    p_high = guard.charger_penalty(0.84, "dc_fast_150kw")
    p_mid = guard.charger_penalty(0.75, "dc_fast_150kw")
    assert p_mid > p_high or p_mid == float("inf")


def test_gentle_routing_flag_threshold():
    guard = SOHPenaltyMatrix()
    assert guard.requires_gentle_routing(0.6) is True
    assert guard.requires_gentle_routing(0.99) is False
