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

# ── Constants ─────────────────────────────────────────────────────────────────
COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET = "Demand"
SUBCATS = ["Demand", "CO2 intensity", "Total", "Fuel", "Electricity imports"]
YEARS = list(range(2000, 2025))

TRAIN_END = 2016
VAL_END = 2020
TEST_END = 2024


# ── Raw Ember CSV (long format) ───────────────────────────────────────────────
@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory):
    """Minimal synthetic Ember yearly CSV in long format."""
    tmp = tmp_path_factory.mktemp("data")
    rows = []
    np.random.seed(42)
    for country in COUNTRIES:
        base = {
            "Tunisia": 12,
            "Austria": 60,
            "Germany": 500,
            "Egypt": 80,
            "Canada": 550,
            "France": 460,
            "Kuwait": 50,
        }[country]
        for year in YEARS:
            for subcat in SUBCATS:
                noise = np.random.normal(0, base * 0.02)
                value = base * (1 + 0.02 * (year - 2000)) + noise
                rows.append(
                    {
                        "Area": country,
                        "Year": year,
                        "Subcategory": subcat,
                        "Unit": "TWh",
                        "Value": round(value, 3),
                    }
                )
    df = pd.DataFrame(rows)
    path = str(tmp / "ember_yearly.csv")
    df.to_csv(path, index=False)
    return path


# ── Filtered long-format CSV ──────────────────────────────────────────────────
@pytest.fixture(scope="session")
def ember_filtered(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eda")
    rows = []
    np.random.seed(42)
    for country in COUNTRIES:
        base = {
            "Tunisia": 12,
            "Austria": 60,
            "Germany": 500,
            "Egypt": 80,
            "Canada": 550,
            "France": 460,
            "Kuwait": 50,
        }[country]
        for year in YEARS:
            for subcat in SUBCATS:
                value = base * (1 + 0.02 * (year - 2000)) + np.random.normal(0, 0.5)
                rows.append(
                    {
                        "Area": country,
                        "Year": year,
                        "Subcategory": subcat,
                        "Unit": "TWh",
                        "Value": round(value, 3),
                    }
                )
    df = pd.DataFrame(rows)
    path = str(tmp / "ember_filtered.csv")
    df.to_csv(path, index=False)
    return path


# ── ember_model_ready.csv (wide format) ──────────────────────────────────────
@pytest.fixture(scope="session")
def model_ready_csv(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("preprocessing")
    rows = []
    np.random.seed(42)
    for country in COUNTRIES:
        base = {
            "Tunisia": 12,
            "Austria": 60,
            "Germany": 500,
            "Egypt": 80,
            "Canada": 550,
            "France": 460,
            "Kuwait": 50,
        }[country]
        for year in YEARS:
            row = {"Area": country, "Year": year}
            for subcat in SUBCATS:
                val = base * (1 + 0.02 * (year - 2000)) + np.random.normal(0, 0.3)
                row[subcat] = round(val, 3)
            # Add lag features
            for subcat in SUBCATS:
                for lag in [1, 2, 3]:
                    row[f"{subcat}_lag{lag}"] = row[subcat] * (1 - 0.01 * lag)
                for w in [3, 5]:
                    row[f"{subcat}_ma{w}"] = row[subcat] * 0.99
                row[f"{subcat}_yoy"] = 2.0 + np.random.normal(0, 0.5)
            rows.append(row)
    df = pd.DataFrame(rows)
    path = str(tmp / "ember_model_ready.csv")
    df.to_csv(path, index=False)
    return path, tmp


# ── feature_meta.json ────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def feature_meta(model_ready_csv):
    path, tmp = model_ready_csv
    df = pd.read_csv(path)
    target_lags = [f"{TARGET}_lag{i}" for i in [1, 2, 3]]
    excl = ["Area", "Year", TARGET] + target_lags
    feats = [c for c in df.columns if c not in excl]
    meta = {
        "TARGET": TARGET,
        "ALL_SUBS": SUBCATS,
        "FEATURES": [s for s in SUBCATS if s != TARGET],
        "all_features": feats,
        "COUNTRIES": COUNTRIES,
        "TRAIN_END": TRAIN_END,
        "VAL_END": VAL_END,
        "TEST_END": TEST_END,
        "FORECAST_YEARS": list(range(2025, 2031)),
    }
    meta_path = str(tmp / "feature_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta, meta_path, str(tmp)


# ── Minimal demand series (single country, sorted) ────────────────────────────
@pytest.fixture
def demand_series_tunisia():
    np.random.seed(0)
    years = list(range(2000, 2025))
    values = [12 + 0.3 * i + np.random.normal(0, 0.2) for i in range(len(years))]
    return pd.DataFrame({"Year": years, "Demand": values})


# ── Temp directory ────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)
