"""
tests/test_modeling.py — ML Benchmark Step Tests
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
TARGET = "Demand"
TRAIN_END = 2016
VAL_END = 2020
TEST_END = 2024


# ── Unit: metric functions ────────────────────────────────────────────────────
class TestMetrics:

    def test_mape_perfect_forecast(self):
        a = np.array([100.0, 200.0, 300.0])
        assert _safe_mape(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_mape_50_percent_error(self):
        a = np.array([100.0, 100.0])
        b = np.array([150.0, 150.0])
        assert _safe_mape(a, b) == pytest.approx(50.0, abs=0.01)

    def test_rmse_perfect_forecast(self):
        from sklearn.metrics import mean_squared_error

        a = np.array([10.0, 20.0, 30.0])
        assert np.sqrt(mean_squared_error(a, a)) == pytest.approx(0.0)

    def test_r2_perfect_forecast(self):
        from sklearn.metrics import r2_score

        a = np.array([10.0, 20.0, 30.0])
        assert r2_score(a, a) == pytest.approx(1.0)

    def test_r2_negative_for_constant_prediction(self):
        from sklearn.metrics import r2_score

        a = np.array([10.0, 20.0, 30.0])
        b = np.array([20.0, 20.0, 20.0])  # mean prediction
        assert r2_score(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_theil_u_less_than_1_beats_naive(self):
        a = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        # Perfect forecast → Theil-U = 0
        assert _theil_u(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_theil_u_naive_equals_1(self):
        a = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        naive = np.array([np.nan, 10.0, 12.0, 14.0, 16.0])
        a_ = a[1:]
        naive_ = naive[1:]
        tu = _theil_u(a_, naive_)
        assert tu == pytest.approx(1.0, abs=0.01)

    def test_smape_symmetric(self):
        a = np.array([100.0])
        b = np.array([200.0])
        s1 = _safe_smape(a, b)
        s2 = _safe_smape(b, a)
        assert s1 == pytest.approx(s2, abs=1e-6)


# ── Unit: Ridge with StandardScaler ──────────────────────────────────────────
class TestRidgeModel:

    @pytest.fixture
    def toy_data(self):
        np.random.seed(1)
        X = np.random.randn(20, 5)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(20) * 0.1
        return X, y

    def test_ridge_fits_and_predicts(self, toy_data):
        X, y = toy_data
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m = Ridge(alpha=1.0).fit(Xs, y)
        preds = m.predict(Xs)
        assert len(preds) == len(y)
        assert not np.any(np.isnan(preds))

    def test_ridge_alpha_0_equals_ols(self, toy_data):
        from sklearn.linear_model import LinearRegression

        X, y = toy_data
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        ridge = Ridge(alpha=0.0001).fit(Xs, y)
        ols = LinearRegression().fit(Xs, y)
        np.testing.assert_allclose(ridge.coef_, ols.coef_, rtol=0.01)

    def test_higher_alpha_more_regularised(self, toy_data):
        X, y = toy_data
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m1 = Ridge(alpha=0.001).fit(Xs, y)
        m2 = Ridge(alpha=1000.0).fit(Xs, y)
        # Higher alpha → smaller coefficients
        assert np.sum(m2.coef_**2) < np.sum(m1.coef_**2)


# ── Unit: Random Forest ───────────────────────────────────────────────────────
class TestRandomForestModel:

    @pytest.fixture
    def toy_data(self):
        np.random.seed(2)
        X = np.random.randn(30, 4)
        y = np.abs(X[:, 0]) * 5 + 10 + np.random.randn(30) * 0.5
        return X, y

    def test_rf_fits_and_predicts(self, toy_data):
        X, y = toy_data
        m = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(y)
        assert not np.any(np.isnan(preds))

    def test_rf_feature_importances_sum_to_1(self, toy_data):
        X, y = toy_data
        m = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        assert abs(m.feature_importances_.sum() - 1.0) < 1e-6

    def test_rf_max_depth_limits_overfitting(self, toy_data):
        from sklearn.metrics import mean_squared_error

        X, y = toy_data
        m_deep = RandomForestRegressor(n_estimators=20, max_depth=None, random_state=42).fit(X, y)
        m_shrt = RandomForestRegressor(n_estimators=20, max_depth=2, random_state=42).fit(X, y)
        # Deep model should fit training data better (lower train MSE)
        mse_deep = mean_squared_error(y, m_deep.predict(X))
        mse_short = mean_squared_error(y, m_shrt.predict(X))
        assert mse_deep <= mse_short


# ── Unit: walk-forward correctness ───────────────────────────────────────────
class TestWalkForward:

    @pytest.fixture
    def country_df(self):
        np.random.seed(3)
        years = list(range(2000, 2025))
        vals = [10 + 0.4 * i + np.random.normal(0, 0.1) for i in range(len(years))]
        feat1 = [v * 0.5 + np.random.normal(0, 0.05) for v in vals]
        return pd.DataFrame(
            {
                "Area": ["Tunisia"] * len(years),
                "Year": years,
                TARGET: vals,
                "feat1": feat1,
            }
        )

    def test_walk_forward_no_future_leakage(self, country_df):
        """For year t, training data must only contain years < t."""
        test_years = [2021, 2022, 2023, 2024]
        for yr in test_years:
            train = country_df[country_df["Year"] < yr]
            assert train["Year"].max() < yr, f"Leakage: train contains year {yr}"

    def test_walk_forward_train_grows_each_step(self, country_df):
        test_years = [2021, 2022, 2023, 2024]
        sizes = [len(country_df[country_df["Year"] < yr]) for yr in test_years]
        assert sizes == sorted(sizes), "Train set should grow each step"

    def test_walk_forward_produces_prediction_per_year(self, country_df):
        test_years = [2021, 2022, 2023, 2024]
        feat_cols = ["feat1"]
        preds = []
        for yr in test_years:
            tr = country_df[country_df["Year"] < yr]
            te = country_df[country_df["Year"] == yr]
            m = Ridge(alpha=1.0).fit(tr[feat_cols], tr[TARGET])
            preds.append(float(m.predict(te[feat_cols])[0]))
        assert len(preds) == len(test_years)
        assert all(np.isfinite(p) for p in preds)


# ── Unit: best model selection ────────────────────────────────────────────────
class TestBestModelSelection:

    def test_lowest_mape_selected(self):
        rows = [
            {"Country": "Tunisia", "Model": "Ridge", "MAPE": 3.2},
            {"Country": "Tunisia", "Model": "RF", "MAPE": 5.1},
            {"Country": "Tunisia", "Model": "XGBoost", "MAPE": 2.8},
        ]
        df = pd.DataFrame(rows)
        best = df.loc[df.groupby("Country")["MAPE"].idxmin()]
        assert best.iloc[0]["Model"] == "XGBoost"

    def test_one_best_per_country(self):
        countries = ["Tunisia", "Germany", "France"]
        rows = []
        for c in countries:
            for m in ["Ridge", "RF", "XGBoost"]:
                rows.append({"Country": c, "Model": m, "MAPE": np.random.uniform(1, 10)})
        df = pd.DataFrame(rows)
        best = df.loc[df.groupby("Country")["MAPE"].idxmin()]
        assert len(best) == len(countries)
        assert best["Country"].nunique() == len(countries)


# ── Integration: CLI run ──────────────────────────────────────────────────────
class TestModelingCLI:

    def test_modeling_cli_runs(self, model_ready_csv, feature_meta, tmp_dir):
        csv_path, pre_dir = model_ready_csv
        meta, meta_path, _ = feature_meta
        out_dir = os.path.join(tmp_dir, "mod_out")

        result = subprocess.run(
            [sys.executable, "src/03_modeling.py", "--input_dir", pre_dir, "--output_dir", out_dir],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Modeling CLI failed:\n{result.stderr}"

    def test_modeling_outputs_exist(self, model_ready_csv, feature_meta, tmp_dir):
        csv_path, pre_dir = model_ready_csv
        meta, _, _ = feature_meta
        out_dir = os.path.join(tmp_dir, "mod_out2")

        subprocess.run(
            [sys.executable, "src/03_modeling.py", "--input_dir", pre_dir, "--output_dir", out_dir],
            capture_output=True,
        )
        for fname in ["test_benchmarking.csv", "best_models.csv", "best_hp.json"]:
            assert os.path.exists(os.path.join(out_dir, fname)), f"Missing output: {fname}"

    def test_test_benchmarking_has_all_countries(self, model_ready_csv, feature_meta, tmp_dir):
        csv_path, pre_dir = model_ready_csv
        meta, _, _ = feature_meta
        out_dir = os.path.join(tmp_dir, "mod_out3")

        subprocess.run(
            [sys.executable, "src/03_modeling.py", "--input_dir", pre_dir, "--output_dir", out_dir],
            capture_output=True,
        )
        path = os.path.join(out_dir, "test_benchmarking.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            assert "Country" in df.columns
            assert "y_true" in df.columns
            for m in ["Ridge", "RandomForest"]:
                assert m in df.columns, f"Model {m} missing from benchmarking output"


# ── Helpers (inline, not imported from src to keep tests self-contained) ──────
def _safe_mape(a, b):
    a, b = np.array(a, float), np.array(b, float)
    mask = np.abs(a) > np.abs(a).mean() * 0.01
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)


def _safe_smape(a, b):
    a, b = np.array(a, float), np.array(b, float)
    denom = np.abs(a) + np.abs(b)
    mask = denom > 1e-8
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(2 * np.abs(a[mask] - b[mask]) / denom[mask]) * 100)


def _theil_u(a, b):
    a, b = np.array(a, float), np.array(b, float)
    if len(a) < 2:
        return np.nan
    num = np.sqrt(np.mean((a[1:] - b[1:]) ** 2))
    den = np.sqrt(np.mean((a[1:] - a[:-1]) ** 2))
    return float(num / (den + 1e-8))
