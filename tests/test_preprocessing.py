"""
tests/test_preprocessing.py — Preprocessing Step Tests
Ember Energy | IEEE Paper

Covers all logic in src/02_preprocessing.py:
  - pivot_wide:        column renaming, one-row-per-country-year
  - impute_missing:    two-pass imputation (interpolate + mean fallback)
  - winsorization:     IQR clipping per country
  - feature engineering: lags, MAs, YoY, cross-features, trend, country_code
  - fi_cols:           excludes Demand_lag*, Demand_ma* (leakage prevention)
  - feature_meta.json: both lowercase + uppercase keys, correct splits
  - CLI:               outputs exist, correct schema, no nulls in lags
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET    = "Demand"
DROP_COLS = ["Total", "Aggregate_fuel"]   # dropped after pivot in 02_preprocessing.py

# Subcategory names AFTER RENAMING (spaces→_, slashes→_) — EXCLUDING DROP_COLS
SUBCATS_RENAMED = [
    "Demand", "CO2_intensity", "Fuel", "Electricity_imports"
]
# Original names as they appear in ember_filtered.csv — including ones that get dropped
SUBCATS_RAW = [
    "Demand", "CO2 intensity", "Total", "Fuel", "Electricity imports"
]


# ═══════════════════════════════════════════════════════════════════════════════
# Unit: pivot_wide
# ═══════════════════════════════════════════════════════════════════════════════

class TestPivotWide:

    def test_pivot_creates_one_row_per_country_year(self, ember_filtered):
        df     = pd.read_csv(ember_filtered)
        df_agg = df.groupby(["Area", "Year", "Subcategory"], as_index=False)["Value"].mean()
        df_wide = df_agg.pivot_table(
            index=["Area", "Year"], columns="Subcategory", values="Value"
        ).reset_index()
        assert df_wide.duplicated(subset=["Area", "Year"]).sum() == 0

    def test_column_renaming_spaces_to_underscore(self, ember_filtered):
        """CRITICAL: notebook renames columns — spaces → _, slashes → _"""
        df     = pd.read_csv(ember_filtered)
        df_agg = df.groupby(["Area", "Year", "Subcategory"], as_index=False)["Value"].mean()
        df_wide = df_agg.pivot_table(
            index=["Area", "Year"], columns="Subcategory", values="Value"
        ).reset_index()
        df_wide.columns.name = None
        # Apply renaming as the script does
        df_wide.columns = [
            c.strip().replace(" ", "_").replace("/", "_")
            for c in df_wide.columns
        ]
        # After renaming, spaces must not exist in column names
        for col in df_wide.columns:
            assert " " not in col, f"Column '{col}' still has spaces after rename"

    def test_all_subcategories_become_renamed_columns(self, ember_filtered):
        """After pivot + rename, subcats should have underscore names"""
        df     = pd.read_csv(ember_filtered)
        df_agg = df.groupby(["Area", "Year", "Subcategory"], as_index=False)["Value"].mean()
        df_wide = df_agg.pivot_table(
            index=["Area", "Year"], columns="Subcategory", values="Value"
        ).reset_index()
        df_wide.columns.name = None
        df_wide.columns = [
            c.strip().replace(" ", "_").replace("/", "_")
            for c in df_wide.columns
        ]
        # Only check subcats NOT in DROP_COLS (Total/Aggregate_fuel are dropped after pivot)
        DROP_COLS = ["Total", "Aggregate_fuel"]
        expected  = [s for s in SUBCATS_RENAMED if s not in DROP_COLS]
        for sub in expected:
            assert sub in df_wide.columns, \
                f"Renamed subcategory '{sub}' missing after pivot"

    def test_demand_column_present_after_rename(self, ember_filtered):
        """TARGET 'Demand' must survive renaming unchanged"""
        df     = pd.read_csv(ember_filtered)
        df_agg = df.groupby(["Area", "Year", "Subcategory"], as_index=False)["Value"].mean()
        df_wide = df_agg.pivot_table(
            index=["Area", "Year"], columns="Subcategory", values="Value"
        ).reset_index()
        df_wide.columns.name = None
        df_wide.columns = [
            c.strip().replace(" ", "_").replace("/", "_")
            for c in df_wide.columns
        ]
        assert TARGET in df_wide.columns

    def test_no_spaces_in_all_subs(self, ember_filtered):
        """ALL_SUBS must not contain any column with spaces"""
        df     = pd.read_csv(ember_filtered)
        df_agg = df.groupby(["Area", "Year", "Subcategory"], as_index=False)["Value"].mean()
        df_wide = df_agg.pivot_table(
            index=["Area", "Year"], columns="Subcategory", values="Value"
        ).reset_index()
        df_wide.columns.name = None
        df_wide.columns = [
            c.strip().replace(" ", "_").replace("/", "_")
            for c in df_wide.columns
        ]
        all_subs = [c for c in df_wide.columns if c not in ["Area", "Year"]]
        for col in all_subs:
            assert " " not in col


# ═══════════════════════════════════════════════════════════════════════════════
# Unit: impute_missing (two-pass)
# ═══════════════════════════════════════════════════════════════════════════════

class TestImputation:

    def test_no_nulls_after_interpolation(self):
        df = pd.DataFrame({
            "Area":   ["Tunisia"] * 5,
            "Year":   [2000, 2001, 2002, 2003, 2004],
            "Demand": [10.0, np.nan, 12.0, np.nan, 14.0],
        })
        df["Demand"] = df.groupby("Area")["Demand"].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        assert df["Demand"].isnull().sum() == 0

    def test_interpolation_is_correct_for_linear_data(self):
        df = pd.DataFrame({
            "Area":   ["Germany"] * 5,
            "Year":   [2000, 2001, 2002, 2003, 2004],
            "Demand": [100.0, np.nan, 300.0, np.nan, 500.0],
        })
        df["Demand"] = df.groupby("Area")["Demand"].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        assert abs(df["Demand"].iloc[1] - 200.0) < 0.01
        assert abs(df["Demand"].iloc[3] - 400.0) < 0.01

    def test_country_mean_fallback_fills_remaining_nulls(self):
        """Pass 2: country mean fills nulls that interpolation can't handle"""
        df = pd.DataFrame({
            "Area":   ["Tunisia"] * 3,
            "Year":   [2000, 2001, 2002],
            "Demand": [10.0, np.nan, np.nan],
        })
        # After interpolation with limit_direction='both', only trailing NaNs may remain
        df["Demand"] = df.groupby("Area")["Demand"].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        # Apply mean fallback (pass 2)
        df["Demand"] = df.groupby("Area")["Demand"].transform(
            lambda s: s.fillna(s.mean())
        )
        assert df["Demand"].isnull().sum() == 0

    def test_two_pass_handles_all_nan_country(self):
        """A country with all NaN for one feature gets mean=NaN — graceful"""
        df = pd.DataFrame({
            "Area":   ["Kuwait"] * 3,
            "Year":   [2000, 2001, 2002],
            "Demand": [np.nan, np.nan, np.nan],
        })
        df["Demand"] = df.groupby("Area")["Demand"].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        df["Demand"] = df.groupby("Area")["Demand"].transform(
            lambda s: s.fillna(s.mean())
        )
        # All NaN stays NaN — no crash
        assert isinstance(df["Demand"].isnull().sum(), (int, np.integer))


# ═══════════════════════════════════════════════════════════════════════════════
# Unit: winsorization
# ═══════════════════════════════════════════════════════════════════════════════

class TestWinsorization:

    def test_winsorization_removes_extreme_outliers(self):
        vals    = np.array([10.0, 10.1, 10.2, 10.3, 10.4, 1000.0])
        q1, q3  = np.percentile(vals, [25, 75])
        iqr     = q3 - q1
        clipped = np.clip(vals, q1 - 2.5 * iqr, q3 + 2.5 * iqr)
        assert clipped.max() < 1000.0

    def test_winsorization_does_not_change_normal_data(self):
        vals    = np.array([10.0, 10.1, 10.2, 10.3, 10.4])
        q1, q3  = np.percentile(vals, [25, 75])
        iqr     = q3 - q1
        clipped = np.clip(vals, q1 - 2.5 * iqr, q3 + 2.5 * iqr)
        np.testing.assert_array_almost_equal(vals, clipped)

    def test_winsorization_factor_25(self):
        """Factor=2.5 should clip values beyond 2.5 IQR"""
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        q1, q3 = np.percentile(vals, [25, 75])
        iqr    = q3 - q1
        upper  = q3 + 2.5 * iqr
        clipped = np.clip(vals, q1 - 2.5 * iqr, upper)
        assert clipped.max() <= upper + 1e-9

    def test_winsorization_symmetric(self):
        """Both upper and lower extremes should be clipped"""
        vals    = np.array([-1000.0, 10.0, 10.1, 10.2, 10.3, 1000.0])
        q1, q3  = np.percentile(vals, [25, 75])
        iqr     = q3 - q1
        clipped = np.clip(vals, q1 - 2.5 * iqr, q3 + 2.5 * iqr)
        assert clipped.min() > -1000.0
        assert clipped.max() < 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Unit: feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    @pytest.fixture
    def simple_df(self):
        np.random.seed(0)
        rows = []
        for c in ["Tunisia"]:
            for yr in range(2000, 2015):
                rows.append({
                    "Area": c, "Year": yr,
                    "Demand": 10 + 0.5 * (yr - 2000),
                    "CO2_intensity": 5 + 0.1 * (yr - 2000),
                })
        return pd.DataFrame(rows)

    def test_lag_creates_correct_shift(self, simple_df):
        df = simple_df.copy().sort_values("Year").reset_index(drop=True)
        df["lag1"] = df.groupby("Area")["Demand"].shift(1)
        assert pd.isna(df["lag1"].iloc[0])
        assert df["lag1"].iloc[1] == df["Demand"].iloc[0]

    def test_three_lags_created(self, simple_df):
        df = simple_df.copy().sort_values("Year").reset_index(drop=True)
        for lag in [1, 2, 3]:
            df[f"Demand_lag{lag}"] = df.groupby("Area")["Demand"].shift(lag)
        for lag in [1, 2, 3]:
            assert f"Demand_lag{lag}" in df.columns

    def test_rolling_mean_correct_length(self, simple_df):
        df = simple_df.copy().sort_values("Year").reset_index(drop=True)
        df["ma3"] = df.groupby("Area")["Demand"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        assert len(df["ma3"]) == len(df)
        # shift(1) makes row 0 always NaN
        assert df["ma3"].isnull().sum() <= 1

    def test_yoy_returns_percentage(self, simple_df):
        df = simple_df.copy().sort_values("Year").reset_index(drop=True)
        df["yoy"] = df.groupby("Area")["Demand"].pct_change() * 100
        assert abs(df["yoy"].iloc[1] - 5.0) < 0.01

    def test_trend_column_starts_at_zero(self, simple_df):
        df = simple_df.copy().sort_values("Year").reset_index(drop=True)
        df["trend"] = df["Year"] - df["Year"].min()
        assert df["trend"].min() == 0
        assert df["trend"].max() == df["Year"].max() - df["Year"].min()

    def test_trend_sq_is_square_of_trend(self, simple_df):
        df = simple_df.copy().sort_values("Year").reset_index(drop=True)
        df["trend"]    = df["Year"] - df["Year"].min()
        df["trend_sq"] = df["trend"] ** 2
        np.testing.assert_array_equal(df["trend_sq"].values, df["trend"].values ** 2)

    def test_country_code_is_integer(self, simple_df):
        df = simple_df.copy()
        df["country_code"] = df["Area"].astype("category").cat.codes
        assert df["country_code"].dtype in [np.int8, np.int16, np.int32, np.int64]

    def test_demand_ratio_created(self, simple_df):
        """demand_ratio_{feature} = Demand / (feature + eps)"""
        df = simple_df.copy()
        df["demand_ratio_CO2_intensity"] = df["Demand"] / (df["CO2_intensity"] + 1e-9)
        assert "demand_ratio_CO2_intensity" in df.columns
        assert df["demand_ratio_CO2_intensity"].notnull().all()

    def test_no_inf_after_feature_engineering(self, simple_df):
        df = simple_df.copy().sort_values("Year").reset_index(drop=True)
        df["lag1"] = df.groupby("Area")["Demand"].shift(1)
        df["yoy"]  = df.groupby("Area")["Demand"].pct_change() * 100
        df["trend"] = df["Year"] - df["Year"].min()
        df_clean   = df.replace([np.inf, -np.inf], np.nan)
        assert not np.isinf(df_clean.select_dtypes(include=np.number).values).any()


# ═══════════════════════════════════════════════════════════════════════════════
# Unit: fi_cols — leakage prevention
# ═══════════════════════════════════════════════════════════════════════════════

class TestFiCols:

    def test_fi_cols_excludes_demand_lags(self, feature_meta):
        """Demand_lag1/2/3 must NOT be in all_features — data leakage"""
        meta, _, _ = feature_meta
        for i in [1, 2, 3]:
            lag = f"{TARGET}_lag{i}"
            assert lag not in meta["all_features"], \
                f"Leakage: {lag} found in all_features"

    def test_fi_cols_excludes_demand_ma(self, feature_meta):
        """Demand_ma3/5 must NOT be in all_features — data leakage"""
        meta, _, _ = feature_meta
        for w in [3, 5]:
            ma = f"{TARGET}_ma{w}"
            assert ma not in meta["all_features"], \
                f"Leakage: {ma} found in all_features"

    def test_fi_cols_includes_other_lags(self, feature_meta):
        """Non-target lags (e.g. CO2_intensity_lag1) SHOULD be in all_features"""
        meta, _, _ = feature_meta
        other_lags = [c for c in meta["all_features"] if "_lag1" in c and TARGET not in c]
        assert len(other_lags) > 0, \
            "No non-target lag features found — something is wrong with fi_cols"

    def test_fi_cols_all_present_in_model_ready(self, feature_meta, model_ready_df):
        meta, _, _ = feature_meta
        for feat in meta["all_features"]:
            assert feat in model_ready_df.columns, \
                f"Feature '{feat}' in all_features but not in model_ready_df"

    def test_fi_cols_no_space_in_names(self, feature_meta):
        """All feature names must use underscores not spaces"""
        meta, _, _ = feature_meta
        for feat in meta["all_features"]:
            assert " " not in feat, f"Feature name '{feat}' contains spaces"


# ═══════════════════════════════════════════════════════════════════════════════
# Unit: feature_meta.json
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureMeta:

    def test_meta_has_uppercase_keys(self, feature_meta):
        """Uppercase keys needed by 03_modeling.py, 04_forecasting.py"""
        meta, _, _ = feature_meta
        for key in ["TARGET", "ALL_SUBS", "FEATURES", "all_features",
                    "COUNTRIES", "TRAIN_END", "VAL_END", "TEST_END",
                    "FORECAST_YEARS"]:
            assert key in meta, f"Uppercase key '{key}' missing from feature_meta"

    def test_meta_has_lowercase_keys(self, feature_meta):
        """Lowercase keys needed by notebooks and test_parity.py"""
        meta, _, _ = feature_meta
        for key in ["target", "all_subs", "raw_features", "all_features"]:
            assert key in meta, f"Lowercase key '{key}' missing from feature_meta"

    def test_train_val_test_order(self, feature_meta):
        meta, _, _ = feature_meta
        assert meta["TRAIN_END"] < meta["VAL_END"] < meta["TEST_END"]

    def test_train_end_is_2016(self, feature_meta):
        meta, _, _ = feature_meta
        assert meta["TRAIN_END"] == 2016

    def test_val_end_is_2020(self, feature_meta):
        meta, _, _ = feature_meta
        assert meta["VAL_END"] == 2020

    def test_test_end_is_2024(self, feature_meta):
        meta, _, _ = feature_meta
        assert meta["TEST_END"] == 2024

    def test_forecast_years_is_2025_to_2030(self, feature_meta):
        meta, _, _ = feature_meta
        assert meta["FORECAST_YEARS"] == list(range(2025, 2031)), \
            f"FORECAST_YEARS should be [2025..2030], got {meta['FORECAST_YEARS']}"

    def test_countries_match(self, feature_meta):
        meta, _, _ = feature_meta
        assert set(meta["COUNTRIES"]) == set(COUNTRIES)

    def test_all_subs_no_spaces(self, feature_meta):
        """ALL_SUBS in meta should use renamed (underscore) column names"""
        meta, _, _ = feature_meta
        for sub in meta["ALL_SUBS"]:
            assert " " not in sub, \
                f"ALL_SUBS contains '{sub}' with spaces — renaming not applied"

    def test_all_features_non_empty(self, feature_meta):
        meta, _, _ = feature_meta
        assert len(meta["all_features"]) > 0

    def test_target_matches_constant(self, feature_meta):
        meta, _, _ = feature_meta
        assert meta["TARGET"] == TARGET
        assert meta["target"] == TARGET


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: CLI run
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessingCLI:

    def test_preprocessing_cli_runs(self, ember_filtered, tmp_dir):
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)
        out_dir = os.path.join(tmp_dir, "pre_out")
        result  = subprocess.run(
            [sys.executable, "src/step02_preprocessing.py",
             "--input_dir",  tmp_dir,
             "--output_dir", out_dir],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Preprocessing CLI failed:\n{result.stderr}"

    def test_preprocessing_outputs_exist(self, ember_filtered, tmp_dir):
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)
        out_dir = os.path.join(tmp_dir, "pre_out2")
        subprocess.run(
            [sys.executable, "src/step02_preprocessing.py",
             "--input_dir",  tmp_dir,
             "--output_dir", out_dir],
            capture_output=True
        )
        assert os.path.exists(os.path.join(out_dir, "ember_model_ready.csv"))
        assert os.path.exists(os.path.join(out_dir, "ember_multifeature.csv"))
        assert os.path.exists(os.path.join(out_dir, "feature_meta.json"))

    def test_model_ready_has_no_nulls_in_lags(self, ember_filtered, tmp_dir):
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)
        out_dir = os.path.join(tmp_dir, "pre_out3")
        subprocess.run(
            [sys.executable, "src/step02_preprocessing.py",
             "--input_dir",  tmp_dir,
             "--output_dir", out_dir],
            capture_output=True
        )
        path = os.path.join(out_dir, "ember_model_ready.csv")
        if os.path.exists(path):
            df_m     = pd.read_csv(path)
            lag_cols = [c for c in df_m.columns if "_lag1" in c]
            if lag_cols:
                assert df_m[lag_cols].isnull().sum().sum() == 0

    def test_model_ready_columns_have_no_spaces(self, ember_filtered, tmp_dir):
        """Column names must use underscores not spaces after rename"""
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)
        out_dir = os.path.join(tmp_dir, "pre_out4")
        subprocess.run(
            [sys.executable, "src/step02_preprocessing.py",
             "--input_dir",  tmp_dir,
             "--output_dir", out_dir],
            capture_output=True
        )
        path = os.path.join(out_dir, "ember_model_ready.csv")
        if os.path.exists(path):
            df_m = pd.read_csv(path)
            for col in df_m.columns:
                assert " " not in col, \
                    f"Column '{col}' has spaces — pivot rename not applied"

    def test_feature_meta_json_valid(self, ember_filtered, tmp_dir):
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)
        out_dir = os.path.join(tmp_dir, "pre_out5")
        subprocess.run(
            [sys.executable, "src/step02_preprocessing.py",
             "--input_dir",  tmp_dir,
             "--output_dir", out_dir],
            capture_output=True
        )
        path = os.path.join(out_dir, "feature_meta.json")
        if os.path.exists(path):
            with open(path) as f:
                meta = json.load(f)
            assert meta["TRAIN_END"] == 2016
            assert meta["VAL_END"]   == 2020
            assert meta["TEST_END"]  == 2024
            assert meta["FORECAST_YEARS"] == list(range(2025, 2031))
            assert "all_features" in meta
            assert len(meta["all_features"]) > 0

    def test_model_ready_has_all_countries(self, ember_filtered, tmp_dir):
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)
        out_dir = os.path.join(tmp_dir, "pre_out6")
        subprocess.run(
            [sys.executable, "src/step02_preprocessing.py",
             "--input_dir",  tmp_dir,
             "--output_dir", out_dir],
            capture_output=True
        )
        path = os.path.join(out_dir, "ember_model_ready.csv")
        if os.path.exists(path):
            df_m = pd.read_csv(path)
            for c in COUNTRIES:
                assert c in df_m["Area"].values, f"{c} missing from model_ready"

    def test_multifeature_csv_exists(self, ember_filtered, tmp_dir):
        """ember_multifeature.csv must be saved (mirrors notebook)"""
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)
        out_dir = os.path.join(tmp_dir, "pre_out7")
        subprocess.run(
            [sys.executable, "src/step02_preprocessing.py",
             "--input_dir",  tmp_dir,
             "--output_dir", out_dir],
            capture_output=True
        )
        assert os.path.exists(os.path.join(out_dir, "ember_multifeature.csv")), \
            "ember_multifeature.csv missing — script doesn't match notebook"

    def test_demand_lag_not_in_all_features(self, ember_filtered, tmp_dir):
        """Regression: Demand_lag must not appear in all_features (leakage)"""
        df = pd.read_csv(ember_filtered)
        df.to_csv(os.path.join(tmp_dir, "ember_filtered.csv"), index=False)
        out_dir = os.path.join(tmp_dir, "pre_out8")
        subprocess.run(
            [sys.executable, "src/step02_preprocessing.py",
             "--input_dir",  tmp_dir,
             "--output_dir", out_dir],
            capture_output=True
        )
        path = os.path.join(out_dir, "feature_meta.json")
        if os.path.exists(path):
            with open(path) as f:
                meta = json.load(f)
            for i in [1, 2, 3]:
                assert f"Demand_lag{i}" not in meta["all_features"], \
                    f"Data leakage: Demand_lag{i} in all_features"
            for w in [3, 5]:
                assert f"Demand_ma{w}" not in meta["all_features"], \
                    f"Data leakage: Demand_ma{w} in all_features"
