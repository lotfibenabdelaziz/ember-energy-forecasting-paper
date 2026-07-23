"""
tests/conftest.py — Shared pytest fixtures
Ember Energy Forecasting | IEEE Paper
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import logging
logging.getLogger("api.main").setLevel(logging.ERROR)

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET    = "Demand"
# Renamed subcategories (spaces→_, slashes→_) matching 02_preprocessing.py
SUBCATS_RENAMED = [
    "Demand", "CO2_intensity", "Demand_per_capita", "Fuel", "Electricity_imports"
]
# Raw subcategory names as they appear in ember_filtered.csv
SUBCATS_RAW = [
    "Demand", "CO2 intensity", "Demand per capita", "Fuel", "Electricity imports"
]
YEARS     = list(range(2000, 2025))
TRAIN_END = 2016
VAL_END   = 2020
TEST_END  = 2024


# ── Raw Ember CSV (long format) ───────────────────────────────────────────────
@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory):
    """Minimal synthetic Ember yearly CSV in long format."""
    tmp = tmp_path_factory.mktemp("data")
    rows = []
    np.random.seed(42)
    for country in COUNTRIES:
        base = {"Tunisia": 12, "Austria": 60, "Germany": 500,
                "Egypt": 80, "Canada": 550, "France": 460, "Kuwait": 50}[country]
        for year in YEARS:
            for subcat in SUBCATS_RAW:
                noise = np.random.normal(0, base * 0.02)
                value = base * (1 + 0.02 * (year - 2000)) + noise
                rows.append({
                    "Area":        country,
                    "Year":        year,
                    "Subcategory": subcat,
                    "Unit":        "TWh",
                    "Value":       round(value, 3),
                })
    df   = pd.DataFrame(rows)
    path = str(tmp / "ember_yearly.csv")
    df.to_csv(path, index=False)
    return path


# ── Filtered long-format CSV (output of 01_eda.py) ───────────────────────────
@pytest.fixture(scope="session")
def ember_filtered(tmp_path_factory):
    """Minimal ember_filtered.csv using RAW subcategory names."""
    tmp = tmp_path_factory.mktemp("eda")
    rows = []
    np.random.seed(42)
    for country in COUNTRIES:
        base = {"Tunisia": 12, "Austria": 60, "Germany": 500,
                "Egypt": 80, "Canada": 550, "France": 460, "Kuwait": 50}[country]
        for year in YEARS:
            for subcat in SUBCATS_RAW:
                noise = np.random.normal(0, base * 0.02)
                value = base * (1 + 0.02 * (year - 2000)) + noise
                rows.append({
                    "Area":        country,
                    "Year":        year,
                    "Subcategory": subcat,
                    "Unit":        "TWh",
                    "Value":       round(value, 3),
                })
    df   = pd.DataFrame(rows)
    path = str(tmp / "ember_filtered.csv")
    df.to_csv(path, index=False)
    return path


# ── Model-ready wide DataFrame (output of 02_preprocessing.py) ───────────────
@pytest.fixture(scope="session")
def model_ready_df():
    """
    Synthetic model-ready DataFrame with RENAMED columns (spaces→_)
    and all engineered features. Mirrors 02_preprocessing.py output.
    """
    np.random.seed(42)
    rows = []
    for country in COUNTRIES:
        base = {"Tunisia": 12, "Austria": 60, "Germany": 500,
                "Egypt": 80, "Canada": 550, "France": 460, "Kuwait": 50}[country]
        for year in YEARS:
            demand = base * (1 + 0.02 * (year - 2000)) + np.random.normal(0, 0.5)
            rows.append({
                "Area":             country,
                "Year":             year,
                "Demand":           round(demand, 3),
                "CO2_intensity":    round(0.4 + np.random.normal(0, 0.01), 4),
                "Demand_per_capita": round(demand / 8.0 + np.random.normal(0, 0.05), 3),
                "Fuel":             round(demand * 0.3 + np.random.normal(0, 0.1), 3),
                "Electricity_imports": round(demand * 0.05 + np.random.normal(0, 0.05), 3),
                "Demand_lag1":      round(demand * 0.98, 3),
                "Demand_lag2":      round(demand * 0.96, 3),
                "Demand_lag3":      round(demand * 0.94, 3),
                "CO2_intensity_lag1": round(0.4 + np.random.normal(0, 0.01), 4),
                "Demand_ma3":       round(demand * 0.99, 3),
                "Demand_ma5":       round(demand * 0.97, 3),
                "Demand_yoy":       round(np.random.normal(2, 0.5), 3),
                "trend":            year - 2000,
                "trend_sq":         (year - 2000) ** 2,
                "country_code":     COUNTRIES.index(country),
            })
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


# ── Feature meta fixture ──────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def feature_meta(model_ready_df, tmp_path_factory):
    """
    Write feature_meta.json matching new 02_preprocessing.py output
    (both lowercase and uppercase keys, raw_features present).
    """
    tmp  = tmp_path_factory.mktemp("pre")
    path = str(tmp / "feature_meta.json")

    feature_cols = [
        c for c in model_ready_df.columns
        if c not in ["Area", "Year", "Demand",
                     "Demand_lag1", "Demand_lag2", "Demand_lag3",
                     "Demand_ma3", "Demand_ma5"]
    ]
    all_subs     = ["Demand", "CO2_intensity", "Demand_per_capita", "Fuel", "Electricity_imports"]
    raw_features = ["CO2_intensity", "Demand_per_capita", "Fuel", "Electricity_imports"]

    meta = {
        # Lowercase keys (notebook-compatible)
        "target":        "Demand",
        "all_subs":      all_subs,
        "raw_features":  raw_features,
        "all_features":  feature_cols,
        "top_corr_20":   feature_cols[:20],
        # Uppercase keys (script-compatible)
        "TARGET":        "Demand",
        "ALL_SUBS":      all_subs,
        "FEATURES":      raw_features,
        "COUNTRIES":     COUNTRIES,
        "TRAIN_END":     TRAIN_END,
        "VAL_END":       VAL_END,
        "TEST_END":      TEST_END,
        "FORECAST_YEARS": list(range(2025, 2031)),
        "DROP_COLS":     ["Total", "Aggregate_fuel"],
    }

    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta, path, str(tmp)


# ── Demand series for a single country ───────────────────────────────────────
@pytest.fixture(scope="session")
def demand_series_tunisia(model_ready_df):
    return (
        model_ready_df[model_ready_df["Area"] == "Tunisia"]
        [["Year", "Demand"]]
        .sort_values("Year")
        .reset_index(drop=True)
    )


# ── Temp dir ──────────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)
