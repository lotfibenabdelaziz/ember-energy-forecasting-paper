-- warehouse/schema.sql — Ember Energy staging star schema
-- ============================================================================
-- Compatible with both SQLite (stdlib sqlite3 — default backend, zero extra
-- dependency) and DuckDB (optional backend — see src/staging.py::BACKEND).
--
-- WHY THIS EXISTS
-- ----------------
-- The raw Ember release is a long/EAV table (Area, Year, Subcategory, Value).
-- src/step02_preprocessing.py pivots it into a wide, denormalized feature
-- table for modeling — that part is correct and doesn't need a warehouse.
--
-- What a flat CSV pipeline *doesn't* give you is a structural guarantee that
-- "no stage may read data from beyond its intended time window." That was
-- exactly the bug in the original preprocessing pipeline (imputation,
-- winsorization, and RF feature-selection statistics were computed over the
-- full 2000-2024 series before any split existed). dim_year.split_label
-- below makes the split boundary a first-class, queryable column instead of
-- a convention every downstream script has to remember to respect —
-- `SELECT ... WHERE split_label = 'train'` is leak-safe by construction; a
-- pandas script that forgets to filter by year is not.
--
-- This is a staging/ingestion layer, not a replacement for the modeling
-- pipeline. step02_preprocessing.py can optionally source its raw rows from
-- here (`--source warehouse`) instead of the CSV directly; either path
-- produces the same long-format frame.

DROP TABLE IF EXISTS fact_energy_measurement;
DROP TABLE IF EXISTS dim_country;
DROP TABLE IF EXISTS dim_year;
DROP TABLE IF EXISTS dim_subcategory;

-- ── Dimensions ────────────────────────────────────────────────────────────

CREATE TABLE dim_country (
    country_key   INTEGER PRIMARY KEY,
    area_name     TEXT NOT NULL UNIQUE
);

CREATE TABLE dim_year (
    year_key      INTEGER PRIMARY KEY,   -- the calendar year itself, e.g. 2016
    year          INTEGER NOT NULL,
    split_label   TEXT NOT NULL          -- 'train' | 'val' | 'test' | 'forecast'
        CHECK (split_label IN ('train', 'val', 'test', 'forecast'))
);

CREATE TABLE dim_subcategory (
    subcategory_key    INTEGER PRIMARY KEY,
    subcategory_name   TEXT NOT NULL UNIQUE,
    unit                TEXT
);

-- ── Fact ──────────────────────────────────────────────────────────────────

CREATE TABLE fact_energy_measurement (
    country_key       INTEGER NOT NULL REFERENCES dim_country(country_key),
    year_key          INTEGER NOT NULL REFERENCES dim_year(year_key),
    subcategory_key   INTEGER NOT NULL REFERENCES dim_subcategory(subcategory_key),
    value             REAL,
    load_date         TEXT NOT NULL,   -- ISO date this row was staged (audit trail)
    source_file       TEXT NOT NULL,   -- provenance: which raw CSV this came from
    PRIMARY KEY (country_key, year_key, subcategory_key)
);

CREATE INDEX idx_fact_year   ON fact_energy_measurement(year_key);
CREATE INDEX idx_fact_split  ON fact_energy_measurement(year_key)
    ;  -- split filtering goes through dim_year.split_label via JOIN

-- Convenience view: exactly the leakage-safe "train only" slice any
-- preprocessing/feature-fitting step should query against.
CREATE VIEW v_train_measurements AS
SELECT f.*, c.area_name, y.year, y.split_label, s.subcategory_name, s.unit
FROM fact_energy_measurement f
JOIN dim_country c      ON f.country_key = c.country_key
JOIN dim_year y         ON f.year_key = y.year_key
JOIN dim_subcategory s  ON f.subcategory_key = s.subcategory_key
WHERE y.split_label = 'train';

-- Convenience view: the full long-format frame (all splits), equivalent to
-- what step01_eda.py currently writes as outputs/eda/ember_filtered.csv.
CREATE VIEW v_all_measurements AS
SELECT f.*, c.area_name, y.year, y.split_label, s.subcategory_name, s.unit
FROM fact_energy_measurement f
JOIN dim_country c      ON f.country_key = c.country_key
JOIN dim_year y         ON f.year_key = y.year_key
JOIN dim_subcategory s  ON f.subcategory_key = s.subcategory_key;
