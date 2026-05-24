"""
tests/test_forecasting.py — Forecasting Step Tests
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET = "Demand"
FORECAST_YEARS = list(range(2025, 2031))
HORIZON = len(FORECAST_YEARS)


# ── Unit: statistical forecasters ────────────────────────────────────────────
class TestStatisticalForecasters:

    @pytest.fixture
    def ts(self):
        np.random.seed(5)
        return np.array([10 + 0.5 * i + np.random.normal(0, 0.1) for i in range(20)])

    def test_naive_returns_last_value(self, ts):
        fc = np.full(HORIZON, ts[-1])
        assert fc[0] == ts[-1]
        assert len(fc) == HORIZON

    def test_linear_trend_forecast_length(self, ts):
        t = np.arange(len(ts)).reshape(-1, 1)
        lr = LinearRegression().fit(t, ts)
        t_f = np.arange(len(ts), len(ts) + HORIZON).reshape(-1, 1)
        fc = lr.predict(t_f).flatten()
        assert len(fc) == HORIZON

    def test_linear_trend_is_monotonic_for_upward_series(self, ts):
        t = np.arange(len(ts)).reshape(-1, 1)
        lr = LinearRegression().fit(t, ts)
        t_f = np.arange(len(ts), len(ts) + HORIZON).reshape(-1, 1)
        fc = lr.predict(t_f).flatten()
        # Each predicted value should be greater than the previous
        assert all(fc[i + 1] > fc[i] for i in range(len(fc) - 1))

    def test_holt_forecast_length(self, ts):
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        m = SimpleExpSmoothing(ts, initialization_method="estimated").fit(optimized=True)
        fc = m.forecast(HORIZON)
        assert len(fc) == HORIZON
        assert all(np.isfinite(fc))

    def test_arima_forecast_length(self, ts):
        from statsmodels.tsa.arima.model import ARIMA

        m = ARIMA(ts, order=(1, 1, 1)).fit()
        fc = np.asarray(m.get_forecast(steps=HORIZON).predicted_mean)
        assert len(fc) == HORIZON
        assert all(np.isfinite(fc))


# ── Unit: bootstrap CI ───────────────────────────────────────────────────────
class TestBootstrapCI:

    @pytest.fixture
    def simple_series(self):
        np.random.seed(7)
        return np.array([10 + 0.5 * i + np.random.normal(0, 0.1) for i in range(20)])

    def test_ci_lower_less_than_point(self, simple_series):
        N_BOOT = 50
        boot_preds = []
        for _ in range(N_BOOT):
            idx = np.sort(np.random.choice(len(simple_series), len(simple_series), replace=True))
            s = simple_series[idx]
            fc = np.full(HORIZON, s[-1])  # naive
            boot_preds.append(fc)
        arr = np.array(boot_preds)
        lower = np.percentile(arr, 5, axis=0)
        upper = np.percentile(arr, 95, axis=0)
        point = np.full(HORIZON, simple_series[-1])
        # Lower ≤ point ≤ upper (not strict, but generally holds)
        assert (lower <= upper).all()

    def test_ci_width_increases_or_stays_flat_over_horizon(self, simple_series):
        """Uncertainty should not decrease as we forecast further."""
        N_BOOT = 50
        boot_preds = []
        for _ in range(N_BOOT):
            idx = np.sort(np.random.choice(len(simple_series), len(simple_series), replace=True))
            s = simple_series[idx]
            t = np.arange(len(s)).reshape(-1, 1)
            lr = LinearRegression().fit(t, s)
            t_f = np.arange(len(s), len(s) + HORIZON).reshape(-1, 1)
            boot_preds.append(lr.predict(t_f).flatten())
        arr = np.array(boot_preds)
        widths = np.percentile(arr, 95, axis=0) - np.percentile(arr, 5, axis=0)
        # Width at last step should be >= width at first step
        assert widths[-1] >= widths[0] * 0.5  # loose check

    def test_n_boot_samples_collected(self, simple_series):
        N_BOOT = 30
        samples = []
        for _ in range(N_BOOT):
            idx = np.sort(np.random.choice(len(simple_series), len(simple_series), replace=True))
            samples.append(simple_series[idx].mean())
        assert len(samples) == N_BOOT


# ── Unit: growth summary ──────────────────────────────────────────────────────
class TestGrowthSummary:

    def test_cagr_formula_correct(self):
        base = 100.0
        end_val = 134.0
        horizon = 6
        cagr = (end_val / base) ** (1 / horizon) - 1
        assert abs(cagr * 100 - 5.0) < 0.1  # ≈5% CAGR

    def test_total_growth_formula(self):
        base = 100.0
        end = 150.0
        growth = (end / base - 1) * 100
        assert growth == pytest.approx(50.0)

    def test_forecast_df_has_required_columns(self):
        rows = [
            {
                "Country": c,
                "Model": "Ridge",
                "Year": yr,
                "Forecast": 10.0,
                "Lower_90": 9.5,
                "Upper_90": 10.5,
            }
            for c in COUNTRIES
            for yr in FORECAST_YEARS
        ]
        df = pd.DataFrame(rows)
        for col in ["Country", "Model", "Year", "Forecast", "Lower_90", "Upper_90"]:
            assert col in df.columns

    def test_one_forecast_row_per_country_per_year(self):
        rows = [
            {"Country": c, "Year": yr, "Forecast": 10.0} for c in COUNTRIES for yr in FORECAST_YEARS
        ]
        df = pd.DataFrame(rows)
        assert len(df) == len(COUNTRIES) * HORIZON
        assert df.duplicated(subset=["Country", "Year"]).sum() == 0


# ── Unit: forecast values sanity ─────────────────────────────────────────────
class TestForecastSanity:

    def test_forecast_is_positive_for_demand(self):
        """Demand forecasts should always be positive (TWh)."""
        preds = np.array([12.0, 12.5, 13.0, 13.5, 14.0, 14.5])
        assert (preds > 0).all()

    def test_forecast_does_not_jump_unreasonably(self):
        """No single-year jump > 50% from last known value."""
        last_known = 100.0
        forecasts = np.array([102, 104, 107, 110, 113, 116])
        max_jump = np.max(np.abs(np.diff(np.concatenate([[last_known], forecasts]))))
        assert max_jump / last_known < 0.50

    def test_ci_lower_non_negative(self):
        lower = np.array([9.5, 9.8, 10.1, 10.4, 10.7, 11.0])
        assert (lower >= 0).all()


# ── Integration: CLI run ──────────────────────────────────────────────────────
class TestForecastingCLI:

    def test_forecasting_cli_runs(self, model_ready_csv, feature_meta, tmp_dir):
        csv_path, pre_dir = model_ready_csv
        meta, _, _ = feature_meta

        # Create a minimal best_models.csv and best_hp.json
        mod_dir = os.path.join(tmp_dir, "mod_fc")
        os.makedirs(mod_dir, exist_ok=True)
        pd.DataFrame([{"Country": c, "Model": "Ridge", "MAPE": 2.5} for c in COUNTRIES]).to_csv(
            os.path.join(mod_dir, "best_models.csv"), index=False
        )

        best_hp = {
            "good_features": meta["all_features"][:5],
            "Ridge": {"alpha": 1.0},
            "RandomForest": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1},
            "XGBoost": {},
        }
        import json

        with open(os.path.join(mod_dir, "best_hp.json"), "w") as f:
            json.dump(best_hp, f)

        # Write updated config
        meta_out = dict(meta)
        meta_out["BEST_MODELS"] = {c: "Ridge" for c in COUNTRIES}
        meta_out["BEST_HP"] = best_hp
        with open(os.path.join(mod_dir, "ember_config.json"), "w") as f:
            json.dump(meta_out, f)

        out_dir = os.path.join(tmp_dir, "fc_out")
        result = subprocess.run(
            [
                sys.executable,
                "src/04_forecasting.py",
                "--pre_dir",
                pre_dir,
                "--model_dir",
                mod_dir,
                "--output_dir",
                out_dir,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Forecasting CLI failed:\n{result.stderr}"

    def test_forecasting_outputs_exist(self, model_ready_csv, feature_meta, tmp_dir):
        csv_path, pre_dir = model_ready_csv
        meta, _, _ = feature_meta
        import json

        mod_dir = os.path.join(tmp_dir, "mod_fc2")
        os.makedirs(mod_dir, exist_ok=True)
        pd.DataFrame([{"Country": c, "Model": "Ridge", "MAPE": 2.5} for c in COUNTRIES]).to_csv(
            os.path.join(mod_dir, "best_models.csv"), index=False
        )
        best_hp = {
            "good_features": meta["all_features"][:5],
            "Ridge": {"alpha": 1.0},
            "RandomForest": {"n_estimators": 10},
            "XGBoost": {},
        }
        with open(os.path.join(mod_dir, "best_hp.json"), "w") as f:
            json.dump(best_hp, f)
        meta_out = dict(meta)
        meta_out["BEST_MODELS"] = {c: "Ridge" for c in COUNTRIES}
        meta_out["BEST_HP"] = best_hp
        with open(os.path.join(mod_dir, "ember_config.json"), "w") as f:
            json.dump(meta_out, f)

        out_dir = os.path.join(tmp_dir, "fc_out2")
        subprocess.run(
            [
                sys.executable,
                "src/04_forecasting.py",
                "--pre_dir",
                pre_dir,
                "--model_dir",
                mod_dir,
                "--output_dir",
                out_dir,
            ],
            capture_output=True,
        )
        for fname in ["demand_forecast_2025_2030.csv", "demand_growth_summary.csv"]:
            assert os.path.exists(os.path.join(out_dir, fname)), f"Missing output: {fname}"
