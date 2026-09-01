"""
battery_model.py
=================
Non-Linear Battery Depletion Predictor (Pillar 2) + State-of-Health (SOH)
Penalty Matrix.

BatteryDepletionModel wraps an XGBoost regressor trained on the physics-
grounded synthetic telemetry from `data_generator.py` (or real fleet
telemetry with the same schema). If `xgboost` is not installed in the
current environment, it transparently falls back to
`sklearn.ensemble.GradientBoostingRegressor`, which is API-compatible
enough for this use case -- so the rest of the platform never has to know
which backend trained the model.

SOHPenaltyMatrix implements the "SOH-Guard": once a vehicle's battery
State-of-Health drops below a configurable threshold, it forbids/penalizes
fast (DC) charging and flags the vehicle for gentle-mode routing, to slow
further degradation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ecoroute.ml.data_generator import FEATURE_COLUMNS, TARGET_COLUMN

try:
    import xgboost as xgb

    _BACKEND = "xgboost"
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingRegressor

    _BACKEND = "sklearn_gbr"


class BatteryDepletionModel:
    """High-precision regressor predicting kWh cost of a road segment."""

    def __init__(self, **model_kwargs):
        self.backend = _BACKEND
        self.feature_columns: List[str] = list(FEATURE_COLUMNS)
        self._is_fitted = False

        if self.backend == "xgboost":
            defaults = dict(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=42,
            )
            defaults.update(model_kwargs)
            self.model = xgb.XGBRegressor(**defaults)
        else:
            defaults = dict(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.85,
                random_state=42,
            )
            defaults.update(model_kwargs)
            self.model = GradientBoostingRegressor(**defaults)

    def fit(self, df: pd.DataFrame, test_size: float = 0.15, random_state: int = 42) -> Dict[str, float]:
        x = df[self.feature_columns].values
        y = df[TARGET_COLUMN].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        self.model.fit(x_train, y_train)
        self._is_fitted = True

        preds = self.model.predict(x_test)
        metrics = {
            "backend": self.backend,
            "mae_kwh": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
        }
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(df[self.feature_columns].values)

    def predict_one(self, features: Dict[str, float]) -> float:
        self._check_fitted()
        row = np.array([[features[c] for c in self.feature_columns]])
        return float(self.model.predict(row)[0])

    def feature_importance(self) -> Dict[str, float]:
        self._check_fitted()
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return {}
        return {col: float(v) for col, v in sorted(zip(self.feature_columns, importances), key=lambda t: -t[1])}

    def save(self, path: str) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "xgboost":
            self.model.save_model(str(path))
        else:
            import joblib

            joblib.dump(self.model, path)
        meta = {"backend": self.backend, "feature_columns": self.feature_columns}
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str) -> "BatteryDepletionModel":
        path = Path(path)
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        instance = cls.__new__(cls)
        instance.backend = meta["backend"]
        instance.feature_columns = meta["feature_columns"]
        instance._is_fitted = True
        if instance.backend == "xgboost":
            instance.model = xgb.XGBRegressor()
            instance.model.load_model(str(path))
        else:
            import joblib

            instance.model = joblib.load(path)
        return instance

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("BatteryDepletionModel must be fit() or load()ed before use.")


# ---------------------------------------------------------------------------
# SOH-Guard: State-of-Health Penalty Matrix
# ---------------------------------------------------------------------------

CHARGER_TYPES = ("dc_fast_150kw", "dc_fast_50kw", "ac_level2_11kw", "ac_level1_7kw")


@dataclass
class SOHPolicy:
    """Configurable thresholds for the SOH-Guard."""

    fast_charge_block_threshold: float = 0.85  # below this SOH, DC fast charging is penalized
    hard_block_threshold: float = 0.70  # below this SOH, DC fast charging is forbidden outright
    gentle_routing_threshold: float = 0.85  # below this SOH, prefer flat/calm routes


class SOHPenaltyMatrix:
    """
    Translates a vehicle's battery State-of-Health into (a) a charger-type
    cost-penalty matrix consumed by the optimizer, and (b) a routing-mode
    flag consumed by the router.
    """

    def __init__(self, policy: SOHPolicy = None):
        self.policy = policy or SOHPolicy()

    def charger_penalty(self, soh: float, charger_type: str) -> float:
        """
        Returns a multiplicative cost penalty (>= 1.0) the optimizer should
        apply to assigning this vehicle to this charger type. `inf` means
        the assignment is forbidden.
        """
        is_fast = charger_type.startswith("dc_fast")
        if not is_fast:
            return 1.0

        if soh < self.policy.hard_block_threshold:
            return float("inf")
        if soh < self.policy.fast_charge_block_threshold:
            # Linear ramp: right at the threshold the penalty is mild,
            # it grows sharply as SOH approaches the hard-block floor.
            span = self.policy.fast_charge_block_threshold - self.policy.hard_block_threshold
            severity = (self.policy.fast_charge_block_threshold - soh) / span
            return 1.0 + 4.0 * severity
        return 1.0

    def penalty_matrix(self, soh: float) -> Dict[str, float]:
        return {ct: self.charger_penalty(soh, ct) for ct in CHARGER_TYPES}

    def requires_gentle_routing(self, soh: float) -> bool:
        return soh < self.policy.gentle_routing_threshold
