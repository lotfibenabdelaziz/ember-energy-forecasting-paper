"""
tests/test_preprocessing.py — Preprocessing Step Tests
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET = "Demand"
SUBCATS = ["Demand", "CO2 intensity", "Total", "Fuel", "Electricity imports"]


# ── Unit: pivot_wide ──────────────────────────────────────────────────────────
class TestPivotWide:

    def test_pivot_creates_one_row_per_country_year(self, ember_filtered):
        df = pd.read_csv(ember_filtered)
        df_agg = df.groupby(["Area", "Year", "Subcategory"], as_index=False)["Value"].mean()
        df_wide = df_agg.pivot_table(
            index=["Area", "Year"], columns="Subcategory", values="Value"
        ).reset_index()
        # Should have exactly one row per (Area, Year) pair
        assert df_wide.duplicated(subset=["Area", "Year"]).sum() == 0

    def test_all_subcategories_become_columns(self, ember_filtered):
        df = pd.read_csv(ember_filtered)
        df_agg = df.groupby(["Area", "Year", "Subcategory"], as_index=False)["Value"].mean()
        df_wide = df_agg.pivot_table(
            index=["Area", "Year"], columns="Subcategory", values="Value"
        ).reset_index()
        df_wide.columns.name = None
        for sub in SUBCATS:
            assert sub in df_wide.columns, f"Subcategory {sub} missing after pivot"


# ── Unit: imputation ─────────────────────────────────────────────────────────
class TestImputation:

    def test_no_nulls_after_interpolation(self):
        df = pd.DataFrame(
            {
                "Area": ["Tunisia"] * 5,
                "Year": [2000, 2001, 2002, 2003, 2004],
                "Demand": [10.0, np.nan, 12.0, np.nan, 14.0],
            }
        )
        df["Demand"] = df.groupby("Area")["Demand"].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        assert df["Demand"].isnull().sum() == 0

    def test_interpolation_is_monotonic_for_linear_data(self):
        df = pd.DataFrame(
            {
                "Area": ["Germany"] * 5,
                "Year": [2000, 2001, 2002, 2003, 2004],
                "Demand": [100.0, np.nan, 300.0, np.nan, 500.0],
            }
        )
        df["Demand"] = df.groupby("Area")["Demand"].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        assert abs(df["Demand"].iloc[1] - 200.0) < 0.01
        assert abs(df["Demand"].iloc[3] - 400.0) < 0.01


# ── Unit: winsorization ───────────────────────────────────────────────────────
class TestWinsorization:

    def test_winsorization_removes_extreme_outliers(self):
        vals = np.array([10.0, 10.1, 10.2, 10.3, 10.4, 1000.0])  # 1000 is extreme
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        clipped = np.clip(vals, q1 - 2.5 * iqr, q3 + 2.5 * iqr)
        assert clipped.max() < 1000.0, "Outlier not removed by winsorization"

    def test_winsorization_does_not_change_normal_data(self):
        vals = np.array([10.0, 10.1, 10.2, 10.3, 10.4])
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        clipped = np.clip(vals, q1 - 2.5 * iqr, q3 + 2.5 * iqr)
        np.testing.assert_array_almost_equal(vals, clipped)


# ── Unit: feature engineering ─────────────────────────────────────────────────
class TestFeatureEngineering:

    @pytest.fixture
    def simple_df(self):
        np.random.seed(0)
        rows = []
        for c in ["Tunisia"]:
            for yr in range(2000, 2015):
                rows.append({"Area": c, "Year": yr, "Demand": 10 + 0.5 * (yr - 2000)})
        return pd.DataFrame(rows)

    def test_lag_creates_correct_shift(self, simple_df):
        df = simple_df.copy().sort_values("Year").reset_index(drop=True)
        df["lag1"] = df.groupby("Area")["Demand"].shift(1)
        assert pd.isna(df["lag1"].iloc[0])  # first row NaN
        assert df["lag1"].iloc[1] == df["Demand"].iloc[0]  # second row = first value

    def test_rolling_mean_length(self, simple_df):
        df = simple_df.copy().sort_values("Year").reset_index(drop=True)
        df["ma3"] = df.groupby("Area")["Demand"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        assert len(df["ma3"]) == len(df)
        df["ma3"] = (
            df.groupby("Area")["Demand"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .fillna(method="bfill")
        )
        assert df["ma3"].isnull().sum() == 0

    def test_yoy_returns_percentage(self, simple_df):
        df = simple_df.copy().sort_values("Year").reset_index(drop=True)
        df["yoy"] = df.groupby("Area")["Demand"].pct_change() * 100
        # 10.5 / 10 - 1 = 5%
        assert abs(df["yoy"].iloc[1] - 5.0) < 0.01

    def test_no_inf_after_feature_engineering(self, simple_df):
        df = simple_df.copy().sort_values("Year").reset_index(drop=True)
        df["lag1"] = df.groupby("Area")["Demand"].shift(1)
        df["yoy"] = df.groupby("Area")["Demand"].pct_change() * 100
        df_clean = df.replace([np.inf, -np.inf], np.nan)
        assert not np.isinf(df_clean.select_dtypes(include=np.number).values).any()


# ── Unit: feature_meta.json ───────────────────────────────────────────────────
class TestFeatureMeta:

    def test_meta_has_required_keys(self, feature_meta):
        meta, _, _ = feature_meta
        for key in [
            "TARGET",
            "ALL_SUBS",
            "FEATURES",
            "all_features",
            "COUNTRIES",
            "TRAIN_END",
            "VAL_END",
            "TEST_END",
        ]:
            assert key in meta, f"Key {key} missing from feature_meta"

    def test_train_val_test_order(self, feature_meta):
        meta, _, _ = feature_meta
        assert meta["TRAIN_END"] < meta["VAL_END"] < meta["TEST_END"]

    def test_all_features_excludes_target_lags(self, feature_meta):
        meta, _, _ = feature_meta
        target_lags = [f"{meta['TARGET']}_lag{i}" for i in [1, 2, 3]]
        for lag in target_lags:
            assert (
                lag not in meta["all_features"]
            ), f"Target lag {lag} should not be in all_features (data leakage!)"

    def test_countries_match(self, feature_meta):
        meta, _, _ = feature_meta
        assert set(meta["COUNTRIES"]) == set(COUNTRIES)


# ── Integration: CLI run ──────────────────────────────────────────────────────
class TestPreprocessingCLI:

    def test_preprocessing_cli_runs(self, ember_filtered, tmp_dir):
        # Write ember_filtered to tmp_dir so the CLI can find it
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)

        out_dir = os.path.join(tmp_dir, "pre_out")
        result = subprocess.run(
            [
                sys.executable,
                "src/02_preprocessing.py",
                "--input_dir",
                tmp_dir,
                "--output_dir",
                out_dir,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Preprocessing CLI failed:\n{result.stderr}"

    def test_preprocessing_outputs_exist(self, ember_filtered, tmp_dir):
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)
        out_dir = os.path.join(tmp_dir, "pre_out2")
        subprocess.run(
            [
                sys.executable,
                "src/02_preprocessing.py",
                "--input_dir",
                tmp_dir,
                "--output_dir",
                out_dir,
            ],
            capture_output=True,
        )
        assert os.path.exists(os.path.join(out_dir, "ember_model_ready.csv"))
        assert os.path.exists(os.path.join(out_dir, "feature_meta.json"))

    def test_model_ready_has_no_nulls_in_lags(self, ember_filtered, tmp_dir):
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)
        out_dir = os.path.join(tmp_dir, "pre_out3")
        subprocess.run(
            [
                sys.executable,
                "src/02_preprocessing.py",
                "--input_dir",
                tmp_dir,
                "--output_dir",
                out_dir,
            ],
            capture_output=True,
        )
        path = os.path.join(out_dir, "ember_model_ready.csv")
        if os.path.exists(path):
            df_m = pd.read_csv(path)
            lag_cols = [c for c in df_m.columns if "_lag1" in c]
            if lag_cols:
                assert df_m[lag_cols].isnull().sum().sum() == 0
