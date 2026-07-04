"""
tests/test_modeling.py — Classical Modeling Step Tests
Ember Energy | IEEE Paper

Covers:
  - src/modeling/metrics.py       : mape, rmse, agg_metrics
  - src/modeling/features.py      : clean_features, prepare_xy
  - src/modeling/forecasters.py   : naive, linear_trend, holt, arima, ml_1step
  - src/modeling/tune.py          : tune_hyperparameters
  - src/modeling/walk_forward.py  : walk_forward_evaluate
  - CLI integration               : 03_modeling.py outputs exist + correct schema
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET    = "Demand"
TRAIN_END = 2016
VAL_END   = 2020
TEST_END  = 2024


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics:

    def test_mape_perfect_forecast(self):
        from src.modeling.metrics import mape
        a = np.array([100.0, 200.0, 300.0])
        assert mape(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_mape_50_percent_error(self):
        from src.modeling.metrics import mape
        a = np.array([100.0, 100.0])
        b = np.array([150.0, 150.0])
        assert mape(a, b) == pytest.approx(50.0, abs=0.01)

    def test_mape_excludes_zero_actuals(self):
        from src.modeling.metrics import mape
        a = np.array([0.0, 100.0])
        b = np.array([999.0, 150.0])
        assert mape(a, b) == pytest.approx(50.0, abs=0.01)

    def test_mape_all_zeros_returns_nan(self):
        from src.modeling.metrics import mape
        import math
        assert math.isnan(mape([0.0, 0.0], [1.0, 2.0]))

    def test_rmse_perfect(self):
        from src.modeling.metrics import rmse
        a = np.array([10.0, 20.0, 30.0])
        assert rmse(a, a) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        from src.modeling.metrics import rmse
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert rmse(a, b) == pytest.approx(np.sqrt((9 + 16) / 2), abs=1e-6)

    def test_agg_metrics_returns_dataframe(self, model_ready_df):
        from src.modeling.metrics import agg_metrics
        res = pd.DataFrame({
            "Country": ["Tunisia"] * 4,
            "Model":   ["Ridge"] * 4,
            "y_actual": [10.0, 11.0, 12.0, 13.0],
            "y_pred":   [10.1, 11.1, 12.1, 13.1],
        })
        df = agg_metrics(res)
        assert "MAE" in df.columns
        assert "RMSE" in df.columns
        assert "MAPE" in df.columns
        assert len(df) == 1

    def test_agg_metrics_multi_model(self):
        from src.modeling.metrics import agg_metrics
        res = pd.DataFrame({
            "Country": ["Tunisia"] * 4,
            "Model":   ["Ridge", "Ridge", "RF", "RF"],
            "y_actual": [10.0, 11.0, 10.0, 11.0],
            "y_pred":   [10.1, 11.2, 9.9,  10.8],
        })
        df = agg_metrics(res)
        assert len(df) == 2
        assert set(df["Model"]) == {"Ridge", "RF"}


# ═══════════════════════════════════════════════════════════════════════════════
# Feature cleaning
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureCleaning:

    def test_clean_features_removes_inf(self):
        from src.modeling.features import clean_features
        X = np.array([[1.0, np.inf], [2.0, 3.0]])
        X_c = clean_features(X.copy())
        assert np.isfinite(X_c).all()

    def test_clean_features_removes_nan(self):
        from src.modeling.features import clean_features
        X = np.array([[1.0, np.nan], [2.0, 4.0]])
        X_c = clean_features(X.copy())
        assert not np.isnan(X_c).any()

    def test_clean_features_uses_column_median(self):
        from src.modeling.features import clean_features
        X = np.array([[1.0], [3.0], [np.nan]])
        X_c = clean_features(X.copy())
        assert X_c[2, 0] == pytest.approx(2.0, abs=0.01)

    def test_prepare_xy_drops_target_nan(self, model_ready_df):
        from src.modeling.features import prepare_xy
        df = model_ready_df.copy()
        df.loc[0, TARGET] = np.nan
        feat_cols = [c for c in df.columns if c not in ["Area", "Year", TARGET]]
        X, y = prepare_xy(df, feat_cols, TARGET)
        assert not np.isnan(y).any()

    def test_prepare_xy_returns_clean_arrays(self, model_ready_df):
        from src.modeling.features import prepare_xy
        feat_cols = [c for c in model_ready_df.columns
                     if c not in ["Area", "Year", TARGET]]
        X, y = prepare_xy(model_ready_df, feat_cols, TARGET)
        assert np.isfinite(X).all()
        assert len(X) == len(y)


# ═══════════════════════════════════════════════════════════════════════════════
# 1-step forecasters
# ═══════════════════════════════════════════════════════════════════════════════

class TestForecasters:

    @pytest.fixture
    def train_series(self):
        np.random.seed(42)
        return pd.Series([10 + 0.5 * i + np.random.normal(0, 0.1)
                          for i in range(20)])

    def test_naive_returns_last_value(self, train_series):
        from src.modeling.forecasters import naive_1step
        assert naive_1step(train_series) == train_series.iloc[-1]

    def test_linear_trend_returns_float(self, train_series):
        from src.modeling.forecasters import linear_trend_1step
        pred = linear_trend_1step(train_series)
        assert isinstance(pred, float)
        assert np.isfinite(pred)

    def test_linear_trend_extrapolates_upward(self, train_series):
        from src.modeling.forecasters import linear_trend_1step
        pred = linear_trend_1step(train_series)
        assert pred > train_series.mean()

    def test_holt_returns_finite(self, train_series):
        from src.modeling.forecasters import holt_1step
        pred = holt_1step(train_series)
        assert np.isfinite(pred)

    def test_arima_returns_finite(self, train_series):
        from src.modeling.forecasters import arima_1step
        pred = arima_1step(train_series)
        assert np.isfinite(pred)

    def test_ml_1step_ridge(self, model_ready_df):
        from src.modeling.forecasters import ml_1step
        feat_cols = [c for c in model_ready_df.columns
                     if c not in ["Area", "Year", TARGET]
                     and "lag" not in c.lower()]
        country   = model_ready_df[model_ready_df["Area"] == "Tunisia"]
        train_df  = country[country["Year"] <= TRAIN_END].reset_index(drop=True)
        test_row  = country[country["Year"] == TRAIN_END + 1].iloc[0]
        pred = ml_1step(
            train_df, test_row, Ridge, feat_cols, TARGET,
            scaler=StandardScaler(), alpha=1.0,
        )
        assert isinstance(pred, float)
        assert np.isfinite(pred)

    def test_ml_1step_rf(self, model_ready_df):
        from src.modeling.forecasters import ml_1step
        feat_cols = [c for c in model_ready_df.columns
                     if c not in ["Area", "Year", TARGET]
                     and "lag" not in c.lower()]
        country   = model_ready_df[model_ready_df["Area"] == "Tunisia"]
        train_df  = country[country["Year"] <= TRAIN_END].reset_index(drop=True)
        test_row  = country[country["Year"] == TRAIN_END + 1].iloc[0]
        pred = ml_1step(
            train_df, test_row, RandomForestRegressor,
            feat_cols, TARGET, n_estimators=10, random_state=42,
        )
        assert isinstance(pred, float)
        assert np.isfinite(pred)


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward evaluation
# ═══════════════════════════════════════════════════════════════════════════════

class TestWalkForward:

    @pytest.fixture
    def simple_best_hp(self):
        return {
            "Ridge":        {"alpha": 1.0},
            "RandomForest": {"n_estimators": 10, "random_state": 42},
            "XGBoost":      {"n_estimators": 10, "learning_rate": 0.1,
                             "max_depth": 3},
        }

    def test_walk_forward_returns_dataframe(self, model_ready_df, simple_best_hp):
        from src.modeling.walk_forward import walk_forward_evaluate
        feat_cols = [c for c in model_ready_df.columns
                     if c not in ["Area", "Year", TARGET]
                     and "lag" not in c.lower()]
        res = walk_forward_evaluate(
            model_ready_df, feat_cols, TARGET,
            [VAL_END + 1], simple_best_hp,
        )
        assert isinstance(res, pd.DataFrame)
        assert not res.empty

    def test_walk_forward_has_required_columns(self, model_ready_df, simple_best_hp):
        from src.modeling.walk_forward import walk_forward_evaluate
        feat_cols = [c for c in model_ready_df.columns
                     if c not in ["Area", "Year", TARGET]
                     and "lag" not in c.lower()]
        res = walk_forward_evaluate(
            model_ready_df, feat_cols, TARGET,
            [VAL_END + 1], simple_best_hp,
        )
        for col in ["Country", "Year", "Model", "y_actual", "y_pred"]:
            assert col in res.columns

    def test_walk_forward_covers_all_countries(self, model_ready_df, simple_best_hp):
        from src.modeling.walk_forward import walk_forward_evaluate
        feat_cols = [c for c in model_ready_df.columns
                     if c not in ["Area", "Year", TARGET]
                     and "lag" not in c.lower()]
        res = walk_forward_evaluate(
            model_ready_df, feat_cols, TARGET,
            [VAL_END + 1], simple_best_hp,
        )
        assert set(res["Country"].unique()) == set(COUNTRIES)

    def test_walk_forward_predictions_are_finite(self, model_ready_df, simple_best_hp):
        from src.modeling.walk_forward import walk_forward_evaluate
        feat_cols = [c for c in model_ready_df.columns
                     if c not in ["Area", "Year", TARGET]
                     and "lag" not in c.lower()]
        res = walk_forward_evaluate(
            model_ready_df, feat_cols, TARGET,
            [VAL_END + 1], simple_best_hp,
        )
        assert res["y_pred"].apply(np.isfinite).all()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelingCLI:

    @pytest.fixture(autouse=True)
    def check_prereq(self):
        if not os.path.exists("outputs/preprocessing/ember_model_ready.csv"):
            pytest.skip("Run make run-preprocessing first")

    def test_outputs_exist(self):
        for fname in ["test_benchmarking.csv", "best_models.csv", "best_hp.json"]:
            path = f"outputs/modeling/{fname}"
            assert os.path.exists(path), f"Missing: {path}"

    def test_test_benchmarking_schema(self):
        df = pd.read_csv("outputs/modeling/test_benchmarking.csv")
        for col in ["Country", "Model", "MAE", "RMSE", "MAPE"]:
            assert col in df.columns

    def test_best_models_has_all_countries(self):
        df = pd.read_csv("outputs/modeling/best_models.csv")
        for c in COUNTRIES:
            assert c in df["Country"].values

    def test_mape_values_reasonable(self):
        df = pd.read_csv("outputs/modeling/test_benchmarking.csv")
        assert df["MAPE"].max() < 100, "MAPE > 100% — something is wrong"
        assert df["MAPE"].min() >= 0

    def test_best_hp_json_valid(self):
        with open("outputs/modeling/best_hp.json") as f:
            hp = json.load(f)
        assert "Ridge" in hp
        assert "alpha" in hp["Ridge"]
