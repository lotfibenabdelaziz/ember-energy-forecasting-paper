"""
tests/test_staging.py — Tests for src/staging.py (star-schema warehouse)
============================================================================
Ember Energy Forecasting | IEEE Paper

Uses the shared `raw_csv` fixture from conftest.py (synthetic, 7 countries
x 2000-2024 x 5 subcategories) so these tests don't touch the real dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.staging import build_star_schema, load_from_warehouse

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TRAIN_END, VAL_END, TEST_END = 2016, 2020, 2024


@pytest.fixture()
def warehouse_db(raw_csv, tmp_path):
    db_path = tmp_path / "test_ember.db"
    build_star_schema(
        raw_csv,
        db_path=db_path,
        countries=COUNTRIES,
        train_end=TRAIN_END,
        val_end=VAL_END,
        test_end=TEST_END,
        backend="sqlite",
    )
    return db_path


@pytest.mark.unit
def test_build_creates_expected_row_counts(warehouse_db):
    df_all = load_from_warehouse(warehouse_db)
    # 7 countries x 25 years x 5 subcategories (raw_csv fixture's shape)
    assert len(df_all) == 7 * 25 * 5
    assert set(df_all["Area"].unique()) == set(COUNTRIES)
    assert df_all["Year"].min() == 2000
    assert df_all["Year"].max() == 2024


@pytest.mark.unit
def test_train_split_never_exceeds_train_end(warehouse_db):
    """The core guarantee: querying the train view can never surface a year
    beyond train_end, regardless of what the caller does downstream."""
    df_train = load_from_warehouse(warehouse_db, split="train")
    assert df_train["Year"].max() <= TRAIN_END
    assert len(df_train) > 0


@pytest.mark.unit
def test_train_split_is_strict_subset_of_all(warehouse_db):
    df_all = load_from_warehouse(warehouse_db)
    df_train = load_from_warehouse(warehouse_db, split="train")
    assert len(df_train) < len(df_all)
    assert set(df_train["Year"].unique()).issubset(set(df_all["Year"].unique()))


@pytest.mark.unit
def test_country_filter(warehouse_db):
    df = load_from_warehouse(warehouse_db, countries=["Austria"])
    assert set(df["Area"].unique()) == {"Austria"}


@pytest.mark.unit
def test_output_schema_matches_ember_filtered_csv_shape(warehouse_db):
    """load_from_warehouse() must be a drop-in for step02's CSV-sourced
    load_filtered() — same columns, same dtypes-compatible shape."""
    df = load_from_warehouse(warehouse_db)
    assert list(df.columns) == ["Area", "Year", "Subcategory", "Unit", "Value"]
    assert pd.api.types.is_integer_dtype(df["Year"]) or pd.api.types.is_numeric_dtype(df["Year"])
    assert pd.api.types.is_numeric_dtype(df["Value"])


@pytest.mark.unit
def test_rebuild_is_idempotent(raw_csv, tmp_path):
    """Building twice into the same path must not error or duplicate rows
    (build_star_schema drops and recreates the DB file each time)."""
    db_path = tmp_path / "idempotent.db"
    build_star_schema(raw_csv, db_path=db_path, countries=COUNTRIES,
                       train_end=TRAIN_END, val_end=VAL_END, test_end=TEST_END)
    first = load_from_warehouse(db_path)
    build_star_schema(raw_csv, db_path=db_path, countries=COUNTRIES,
                       train_end=TRAIN_END, val_end=VAL_END, test_end=TEST_END)
    second = load_from_warehouse(db_path)
    assert len(first) == len(second)
