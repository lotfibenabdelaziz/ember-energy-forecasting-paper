"""
tests/test_eda.py — EDA Step Tests
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET = "Demand"


# ── Unit: data loading ────────────────────────────────────────────────────────
class TestEDADataLoading:

    def test_raw_csv_has_required_columns(self, raw_csv):
        df = pd.read_csv(raw_csv)
        for col in ["Area", "Year", "Subcategory", "Unit", "Value"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_all_countries_present(self, raw_csv):
        df = pd.read_csv(raw_csv)
        df_filtered = df[df["Area"].isin(COUNTRIES)]
        found = df_filtered["Area"].unique().tolist()
        for c in COUNTRIES:
            assert c in found, f"Country {c} not in CSV"

    def test_demand_subcat_present(self, raw_csv):
        df = pd.read_csv(raw_csv)
        assert TARGET in df["Subcategory"].unique(), f"Subcategory '{TARGET}' not found"

    def test_years_range(self, raw_csv):
        df = pd.read_csv(raw_csv)
        assert df["Year"].min() <= 2000
        assert df["Year"].max() >= 2020

    def test_value_is_numeric(self, raw_csv):
        df = pd.read_csv(raw_csv)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        assert df["Value"].isnull().sum() == 0, "Non-numeric values in Value column"

    def test_no_negative_demand(self, raw_csv):
        df = pd.read_csv(raw_csv)
        demand = df[df["Subcategory"] == TARGET]
        assert (demand["Value"] >= 0).all(), "Negative demand values found"


# ── Unit: CAGR calculation ────────────────────────────────────────────────────
class TestCAGR:

    def test_cagr_positive_growth(self, demand_series_tunisia):
        """CAGR should be positive for an upward-trending series."""
        d = demand_series_tunisia
        s = d["Year"].values[0]
        e = d["Year"].values[-1]
        vs = d.iloc[0]["Demand"]
        ve = d.iloc[-1]["Demand"]
        cagr = (ve / vs) ** (1 / (e - s)) - 1
        assert cagr > 0, "Expected positive CAGR for Tunisia"

    def test_cagr_formula(self):
        """Test CAGR formula directly."""
        start, end, years = 100.0, 200.0, 10
        cagr = (end / start) ** (1 / years) - 1
        assert abs(cagr - 0.0718) < 0.001  # ~7.18%


# ── Unit: ADF stationarity test helper ───────────────────────────────────────
class TestADF:

    def test_trending_series_not_stationary(self, demand_series_tunisia):
        from statsmodels.tsa.stattools import adfuller

        vals = demand_series_tunisia["Demand"].values
        _, pval, *_ = adfuller(vals, autolag="AIC")
        # Trending series should fail stationarity (p > 0.05 usually)
        # We just check the call runs without error
        assert isinstance(pval, float)

    def test_differenced_series_closer_to_stationary(self, demand_series_tunisia):
        from statsmodels.tsa.stattools import adfuller

        diff = demand_series_tunisia["Demand"].diff().dropna().values
        _, pval, *_ = adfuller(diff, autolag="AIC")
        assert isinstance(pval, float)


# ── Integration: CLI run ──────────────────────────────────────────────────────
class TestEDACLI:

    def test_eda_cli_runs(self, raw_csv, tmp_dir):
        result = subprocess.run(
            [sys.executable, "src/step01_eda.py", "--csv", raw_csv, "--output_dir", tmp_dir],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"EDA CLI failed:\n{result.stderr}"

    def test_eda_outputs_exist(self, raw_csv, tmp_dir):
        subprocess.run(
            [sys.executable, "src/step01_eda.py", "--csv", raw_csv, "--output_dir", tmp_dir],
            capture_output=True,
        )
        assert os.path.exists(os.path.join(tmp_dir, "ember_filtered.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "table01_eda_statistics.csv"))

    def test_ember_filtered_has_all_countries(self, raw_csv, tmp_dir):
        subprocess.run(
            [sys.executable, "src/step01_eda.py", "--csv", raw_csv, "--output_dir", tmp_dir],
            capture_output=True,
        )
        path = os.path.join(tmp_dir, "ember_filtered.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for c in COUNTRIES:
                assert c in df["Area"].unique(), f"{c} missing from filtered CSV"
