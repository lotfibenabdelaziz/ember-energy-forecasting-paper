"""
src/staging.py — Star-schema staging layer (SQLite / DuckDB)
================================================================
Ember Energy | IEEE Paper — Data Modeling

See warehouse/schema.sql for the full rationale. Short version: the raw
Ember release is a long/EAV table; step02_preprocessing.py pivots it into a
wide feature table for modeling, which is the right shape for pandas/
sklearn and isn't being changed. What this module adds is an OPTIONAL
staging step in front of that: load the raw CSV into a small dimensional
model (fact_energy_measurement + dim_country/dim_year/dim_subcategory)
where each year row carries an explicit split_label ('train'/'val'/'test'/
'forecast'). That makes "only train-window data may inform a fit" a
queryable constraint instead of a convention — the exact class of bug that
caused the preprocessing leakage fixed in step02_preprocessing.py.

Backend: SQLite (Python stdlib — always available, zero extra dependency)
by default. DuckDB is used automatically if installed (`pip install duckdb`)
because it's a better fit for analytical queries over this kind of table —
this project doesn't need that performance at ~1,000 fact rows, but the
swap is one import away for anyone who wants it. Both are embedded,
serverless, single-file databases — no server process, nothing to deploy,
appropriate for a dataset this size (this is deliberately NOT a Data Vault
or a server-backed warehouse — see the review's discussion of over-
engineering for why that would be the wrong tool here).

CLI:
    python src/staging.py build  --csv data/raw/yearly_full_release_long_format.csv
    python src/staging.py query  --split train --country Austria
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os as _os
from pathlib import Path
import sqlite3
import sys as _sys

import pandas as pd

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from src.config import cfg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "warehouse" / "schema.sql"
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "warehouse" / "ember.db"

try:
    import duckdb  # type: ignore

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


def _connect(db_path: str | Path, backend: str = "auto"):
    """Return a DB-API-compatible connection. `backend`: 'sqlite' | 'duckdb' | 'auto'."""
    use_duckdb = HAS_DUCKDB and backend in ("duckdb", "auto") and str(db_path).endswith(".duckdb")
    if backend == "duckdb" and not HAS_DUCKDB:
        raise ImportError("duckdb not installed — `pip install duckdb`, or use backend='sqlite'")
    if use_duckdb:
        return duckdb.connect(str(db_path))
    return sqlite3.connect(str(db_path))


def _split_label(year: int, train_end: int, val_end: int, test_end: int) -> str:
    if year <= train_end:
        return "train"
    if year <= val_end:
        return "val"
    if year <= test_end:
        return "test"
    return "forecast"


def build_star_schema(
    csv_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    countries: list[str] | None = None,
    train_end: int | None = None,
    val_end: int | None = None,
    test_end: int | None = None,
    backend: str = "sqlite",
) -> Path:
    """
    Load the raw long-format Ember CSV into the star schema defined in
    warehouse/schema.sql. Splits (train/val/test/forecast) come from
    params.yaml via src.config.cfg unless overridden — same single source
    of truth the rest of the pipeline uses.
    """
    countries = countries or cfg.countries
    train_end = train_end if train_end is not None else cfg.train_end
    val_end = val_end if val_end is not None else cfg.val_end
    test_end = test_end if test_end is not None else cfg.test_end

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    log.info("Reading raw CSV: %s", csv_path)
    raw = pd.read_csv(csv_path)
    raw.columns = raw.columns.str.strip()
    raw = raw[raw["Area"].isin(countries)][["Area", "Year", "Subcategory", "Unit", "Value"]].copy()
    raw["Year"] = raw["Year"].astype(int)
    raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce")
    log.info("Filtered to %d countries: %s rows", len(countries), raw.shape)

    conn = _connect(db_path, backend=backend)
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql) if hasattr(conn, "executescript") else conn.execute(schema_sql)

    # ── Dimensions ────────────────────────────────────────────────────────
    dim_country = pd.DataFrame({"country_key": range(len(countries)), "area_name": countries})
    years = sorted(raw["Year"].unique().tolist())
    dim_year = pd.DataFrame(
        {
            "year_key": years,
            "year": years,
            "split_label": [_split_label(y, train_end, val_end, test_end) for y in years],
        }
    )
    subcats = sorted(raw["Subcategory"].unique().tolist())
    units = raw.drop_duplicates("Subcategory").set_index("Subcategory")["Unit"].to_dict()
    dim_subcategory = pd.DataFrame(
        {
            "subcategory_key": range(len(subcats)),
            "subcategory_name": subcats,
            "unit": [units.get(s) for s in subcats],
        }
    )

    dim_country.to_sql("dim_country", conn, if_exists="append", index=False)
    dim_year.to_sql("dim_year", conn, if_exists="append", index=False)
    dim_subcategory.to_sql("dim_subcategory", conn, if_exists="append", index=False)

    # ── Fact ──────────────────────────────────────────────────────────────
    country_map = dict(zip(dim_country["area_name"], dim_country["country_key"], strict=True))
    subcat_map = dict(
        zip(dim_subcategory["subcategory_name"], dim_subcategory["subcategory_key"], strict=True)
    )

    fact = pd.DataFrame(
        {
            "country_key": raw["Area"].map(country_map),
            "year_key": raw["Year"],
            "subcategory_key": raw["Subcategory"].map(subcat_map),
            "value": raw["Value"],
            "load_date": _dt.date.today().isoformat(),
            "source_file": str(csv_path),
        }
    )
    # Multiple raw rows can share (country, year, subcategory) in the source
    # release (see step02's own groupby(...).mean() pivot step) — average
    # them here so the fact table's primary key holds, same aggregation the
    # existing pivot_wide() already performs downstream.
    fact = fact.groupby(["country_key", "year_key", "subcategory_key"], as_index=False).agg(
        value=("value", "mean"),
        load_date=("load_date", "first"),
        source_file=("source_file", "first"),
    )
    fact.to_sql("fact_energy_measurement", conn, if_exists="append", index=False)

    conn.commit() if hasattr(conn, "commit") else None
    conn.close()

    log.info(
        "Star schema built → %s (%d countries × %d years × %d subcategories, %d fact rows)",
        db_path,
        len(countries),
        len(years),
        len(subcats),
        len(fact),
    )
    return db_path


def load_from_warehouse(
    db_path: str | Path = DEFAULT_DB_PATH,
    countries: list[str] | None = None,
    split: str | None = None,
    backend: str = "sqlite",
) -> pd.DataFrame:
    """
    Query the warehouse back into the same long-format shape step01_eda.py
    writes to outputs/eda/ember_filtered.csv (Area, Year, Subcategory, Unit,
    Value) — a drop-in source for step02_preprocessing.py's load_filtered().

    `split=None` returns all years (train+val+test+forecast) — the normal
    case, since the split-safety guarantee lives in *how downstream code
    queries/filters*, not in withholding rows here. Pass split='train' to
    get the leak-safe slice directly, e.g. for an ad hoc leakage check.
    """
    conn = _connect(db_path, backend=backend)
    view = "v_train_measurements" if split == "train" else "v_all_measurements"
    query = f"""
        SELECT area_name AS Area, year AS Year, subcategory_name AS Subcategory,
               unit AS Unit, value AS Value
        FROM {view}
    """
    if countries:
        placeholders = ", ".join(f"'{c}'" for c in countries)
        query += f" WHERE area_name IN ({placeholders})"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ── CLI ───────────────────────────────────────────────────────────────────


def _cli() -> None:
    p = argparse.ArgumentParser(description="Ember staging warehouse (star schema)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build the star schema from the raw CSV")
    b.add_argument("--csv", default=cfg.csv_path)
    b.add_argument("--db", default=str(DEFAULT_DB_PATH))
    b.add_argument("--backend", choices=["sqlite", "duckdb"], default="sqlite")

    q = sub.add_parser("query", help="Query the warehouse and print row counts")
    q.add_argument("--db", default=str(DEFAULT_DB_PATH))
    q.add_argument("--split", choices=["train", "val", "test", "forecast"], default=None)
    q.add_argument("--country", default=None)
    q.add_argument("--backend", choices=["sqlite", "duckdb"], default="sqlite")

    args = p.parse_args()

    if args.cmd == "build":
        build_star_schema(args.csv, args.db, backend=args.backend)
    elif args.cmd == "query":
        countries = [args.country] if args.country else None
        df = load_from_warehouse(
            args.db, countries=countries, split=args.split, backend=args.backend
        )
        log.info("Query returned %s", df.shape)
        print(df.head(20).to_string())


if __name__ == "__main__":
    _cli()
