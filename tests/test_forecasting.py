"""
tests/test_forecasting.py — Forecasting Step Tests
Ember Energy | IEEE Paper

Covers:
  - Statistical forecasters (naive, linear trend, Holt, ARIMA)
  - clean_X() and exogenous extrapolation
  - bootstrap_forecast_ci() — length + bounds
  - growth_summary() — CAGR calculation
  - CLI integration: demand_forecast_2025_2030.csv schema + values
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

COUNTRIES      = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET         = "Demand"
FORECAST_YEARS = list(range(2025, 2031))
HORIZON        = len(FORECAST_YEARS)


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical forecasters
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatisticalForecasters:

    @pytest.fixture
    def ts(self):
        np.random.seed(5)
        return np.array([10 + 0.5 * i + np.random.normal(0, 0.1)
                         for i in range(20)])

    def test_naive_forecast_length(self, ts):
        fc = np.full(HORIZON, ts[-1])
        assert len(fc) == HORIZON
        assert fc[0] == ts[-1]

    def test_linear_trend_length(self, ts):
        from src.modeling.forecasters import linear_trend_1step
        import pandas as pd
        pred = linear_trend_1step(pd.Series(ts))
        assert np.isfinite(pred)

    def test_holt_forecast_length(self, ts):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        m  = ExponentialSmoothing(ts, trend="add", damped_trend=True).fit(optimized=True)
        fc = m.forecast(HORIZON)
        assert len(fc) == HORIZON
        assert all(np.isfinite(fc))

    def test_arima_forecast_length(self, ts):
        from statsmodels.tsa.arima.model import ARIMA
        m  = ARIMA(ts, order=(1, 1, 1)).fit()
        pm = m.get_forecast(steps=HORIZON).predicted_mean
        fc = np.array(pm)   # works for both ndarray and Series
        assert len(fc) == HORIZON
        assert all(np.isfinite(fc))

    def test_forecast_statistical_holt(self, ts):
        from src.forecasting.forecasters import forecast_statistical
        fc = forecast_statistical(ts, "Holt", HORIZON)
        assert len(fc) == HORIZON
        assert all(np.isfinite(fc))

    def test_forecast_statistical_linear_trend(self, ts):
        from src.forecasting.forecasters import forecast_statistical
        fc = forecast_statistical(ts, "LinearTrend", HORIZON)
        assert len(fc) == HORIZON

    def test_forecast_statistical_arima(self, ts):
        from src.forecasting.forecasters import forecast_statistical
        fc = forecast_statistical(ts, "ARIMA_1_1_1", HORIZON)
        assert len(fc) == HORIZON


# ═══════════════════════════════════════════════════════════════════════════════
# Feature cleaning
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanX:

    def test_clean_x_removes_inf(self):
        from src.forecasting.forecasters import clean_x
        arr = np.array([[1.0, np.inf], [2.0, 3.0]])
        out = clean_x(arr)
        assert np.isfinite(out).all()

    def test_clean_x_removes_nan(self):
        from src.forecasting.forecasters import clean_x
        arr = np.array([[1.0, np.nan], [2.0, 4.0]])
        out = clean_x(arr)
        assert not np.isnan(out).any()

    def test_clean_x_uses_median(self):
        from src.forecasting.forecasters import clean_x
        arr = np.array([[1.0], [3.0], [np.nan]])
        out = clean_x(arr)
        assert out[2, 0] == pytest.approx(2.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# Exogenous extrapolation
# ═══════════════════════════════════════════════════════════════════════════════

class TestExogExtrapolation:

    def test_extrapolate_returns_correct_length(self, model_ready_df):
        from src.forecasting.forecasters import extrapolate_exog
        sub  = model_ready_df[model_ready_df["Area"] == "Tunisia"]
        cols = ["CO2_intensity", "trend"]
        ext  = extrapolate_exog(sub, cols, HORIZON)
        assert len(ext) == len(cols)
        for col in cols:
            assert len(ext[col]) == HORIZON

    def test_extrapolate_values_are_finite(self, model_ready_df):
        from src.forecasting.forecasters import extrapolate_exog
        sub  = model_ready_df[model_ready_df["Area"] == "Tunisia"]
        cols = ["CO2_intensity"]
        ext  = extrapolate_exog(sub, cols, HORIZON)
        assert all(np.isfinite(ext["CO2_intensity"]))


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════════

class TestBootstrapCI:

    def test_bootstrap_ci_returns_three_arrays(self, model_ready_df):
        from src.forecasting.forecasters import bootstrap_forecast_ci, get_ml_cls_map
        sub     = model_ready_df[model_ready_df["Area"] == "Tunisia"].sort_values("Year")
        feat_c  = [c for c in sub.columns if c not in ["Area", "Year", TARGET]
                   and not c.startswith("Demand_lag")]
        raw_c   = ["CO2_intensity"]
        params  = {"alpha": 1.0}
        ml_map  = get_ml_cls_map()

        fc, lo, hi = bootstrap_forecast_ci(
            sub, "Ridge", feat_c, raw_c, params,
            HORIZON, TARGET, ml_map, n_boot=10, ci=90,
        )
        assert len(fc) == HORIZON
        assert len(lo) == HORIZON
        assert len(hi) == HORIZON

    def test_bootstrap_lower_le_point_le_upper(self, model_ready_df):
        from src.forecasting.forecasters import bootstrap_forecast_ci, get_ml_cls_map
        sub    = model_ready_df[model_ready_df["Area"] == "Tunisia"].sort_values("Year")
        feat_c = [c for c in sub.columns if c not in ["Area", "Year", TARGET]
                  and not c.startswith("Demand_lag")]
        raw_c  = ["CO2_intensity"]
        ml_map = get_ml_cls_map()

        fc, lo, hi = bootstrap_forecast_ci(
            sub, "Ridge", feat_c, raw_c, {"alpha": 1.0},
            HORIZON, TARGET, ml_map, n_boot=10, ci=90,
        )
        assert all(lo[i] <= fc[i] for i in range(HORIZON))
        assert all(fc[i] <= hi[i] for i in range(HORIZON))


# ═══════════════════════════════════════════════════════════════════════════════
# Growth summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestGrowthSummary:

    def test_growth_summary_has_all_countries(self, model_ready_df):
        from src.forecasting.growth import compute_growth_summary
        fc_rows = []
        for c in COUNTRIES:
            sub = model_ready_df[model_ready_df["Area"] == c]
            last_demand = sub["Demand"].values[-1]
            for yr in FORECAST_YEARS:
                fc_rows.append({
                    "Country":  c, "Year": yr,
                    "Forecast": last_demand * (1 + 0.02) ** (yr - 2024),
                    "Lower_90": last_demand * 0.95,
                    "Upper_90": last_demand * 1.05,
                    "Model":    "Ridge",
                })
        fc_df = pd.DataFrame(fc_rows)
        best_df = pd.DataFrame({
            "Country": COUNTRIES,
            "Model":   ["Ridge"] * len(COUNTRIES),
            "MAPE":    [3.0] * len(COUNTRIES),
        })
        growth = compute_growth_summary(model_ready_df, fc_df, best_df, TARGET)
        assert len(growth) == len(COUNTRIES)

    def test_cagr_positive_for_growing_series(self, model_ready_df):
        from src.forecasting.growth import compute_growth_summary
        fc_rows = []
        for c in COUNTRIES:
            sub = model_ready_df[model_ready_df["Area"] == c]
            base = sub["Demand"].values[-1]
            for yr in FORECAST_YEARS:
                fc_rows.append({
                    "Country":  c, "Year": yr,
                    "Forecast": base * (1.03 ** (yr - 2024)),
                    "Lower_90": base * 0.98, "Upper_90": base * 1.05,
                    "Model": "Ridge",
                })
        fc_df = pd.DataFrame(fc_rows)
        best_df = pd.DataFrame({
            "Country": COUNTRIES, "Model": ["Ridge"] * 7,
            "MAPE": [3.0] * 7,
        })
        growth = compute_growth_summary(model_ready_df, fc_df, best_df, TARGET)
        assert (growth["CAGR 24-30 %"] > 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestForecastingCLI:

    @pytest.fixture(autouse=True)
    def check_prereq(self):
        if not os.path.exists("outputs/modeling/best_models.csv"):
            pytest.skip("Run make run-modeling first")
        if not os.path.exists("outputs/forecasting/demand_forecast_2025_2030.csv"):
            pytest.skip("Run make run-forecasting first")

    def test_forecast_csv_exists(self):
        assert os.path.exists("outputs/forecasting/demand_forecast_2025_2030.csv")

    def test_growth_summary_exists(self):
        assert os.path.exists("outputs/forecasting/demand_growth_summary.csv")

    def test_forecast_schema(self):
        df = pd.read_csv("outputs/forecasting/demand_forecast_2025_2030.csv")
        for col in ["Country", "Year", "Forecast", "Lower_90", "Upper_90"]:
            assert col in df.columns

    def test_forecast_years_correct(self):
        df = pd.read_csv("outputs/forecasting/demand_forecast_2025_2030.csv")
        assert set(df["Year"].unique()) == set(FORECAST_YEARS)

    def test_forecast_covers_all_countries(self):
        df = pd.read_csv("outputs/forecasting/demand_forecast_2025_2030.csv")
        for c in COUNTRIES:
            assert c in df["Country"].values

    def test_forecast_values_positive(self):
        df = pd.read_csv("outputs/forecasting/demand_forecast_2025_2030.csv")
        assert (df["Forecast"] > 0).all()

    def test_ci_bounds_valid(self):
        df = pd.read_csv("outputs/forecasting/demand_forecast_2025_2030.csv")
        assert (df["Lower_90"] <= df["Forecast"]).all()
        assert (df["Upper_90"] >= df["Forecast"]).all()
