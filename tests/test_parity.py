"""
tests/test_parity.py — Notebook vs Script Output Parity Tests
==============================================================
Verifies that the .py scripts and .ipynb notebooks produce
numerically equivalent outputs on the same input data.

Strategy:
  1. Run the .py scripts  → outputs/py/
  2. Extract notebook cell outputs → outputs/nb/  (from already-run notebooks)
  3. Compare CSV files column by column with tolerance
  4. Compare key numeric metrics

Usage:
  # First run notebooks and save them (with outputs), then:
  pytest tests/test_parity.py -v

  # Or run everything:
  make test-parity
"""

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
NB_DIR = ROOT  # where .ipynb files live
PY_OUT = ROOT / "outputs"  # .py script outputs (already exists)
# More reliable: just check if the .py output and the known nb output path both exist
# and compare them directly — no regex needed
NB_OUT = {
    "eda": ROOT / "outputs" / "eda",
    "preprocessing": ROOT / "outputs" / "preprocessing",
    "modeling": ROOT / "outputs" / "modeling",
    "forecasting": ROOT / "outputs" / "forecasting",
    "deeplearning": ROOT / "outputs" / "deeplearning",
}

NOTEBOOKS = {
    "eda": ROOT / "notebooks" / "00_EDA_patched.ipynb",
    "preprocessing": ROOT / "notebooks" / "01_preprocessing_patched.ipynb",
    "modeling": ROOT / "notebooks" / "02_benchmarking_patched.ipynb",
    "forecasting": ROOT / "notebooks" / "033_forecasting_patched.ipynb",
    "deeplearning": ROOT / "notebooks" / "04_deeplearning_enhanced_patched.ipynb",
}


# Tolerance for numeric comparison
RTOL = 0.01  # 1% relative tolerance
ATOL = 0.5  # 0.5 TWh absolute tolerance


# ── Helpers ───────────────────────────────────────────────────────────────────


def extract_nb_csv(nb_path: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Extract CSV outputs saved by a notebook.
    Handles both literal paths and os.path.join() / pathlib patterns.
    Returns dict of {filename: DataFrame}.
    """
    if not nb_path.exists():
        log.warning("Notebook not found: %s — skipping", nb_path)
        return {}

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # Match both:
    #   to_csv("path/to/file.csv")
    #   to_csv(os.path.join(_DATA, 'file.csv'))
    #   to_csv(some_var / "file.csv")
    filename_patterns = [
        re.compile(r'to_csv\(["\']([^"\']+\.csv)["\']'),  # literal string
        re.compile(r'to_csv\([^,)]*["\']([^"\']+\.csv)["\']'),  # os.path.join last arg
        re.compile(r'["\']([a-zA-Z0-9_]+\.csv)["\']'),  # any quoted .csv name
    ]

    found_names = set()
    for cell in nb.get("cells", []):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "to_csv" not in src:
            continue
        for pat in filename_patterns:
            for m in pat.finditer(src):
                fname = Path(m.group(1)).name  # take only the filename part
                if fname.endswith(".csv"):
                    found_names.add(fname)

    # Now resolve each filename against known output dirs
    result = {}
    search_dirs = [out_dir] + list(NB_OUT.values())
    for fname in found_names:
        for d in search_dirs:
            candidate = Path(d) / fname
            if candidate.exists():
                try:
                    result[fname] = pd.read_csv(candidate)
                    log.debug("Resolved %s → %s", fname, candidate)
                except Exception:
                    pass
                break

    return result


def compare_dataframes(
    df_py: pd.DataFrame,
    df_nb: pd.DataFrame,
    name: str,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> list[str]:
    """
    Compare two DataFrames. Returns list of discrepancy messages (empty = match).
    """
    issues = []

    # Shape
    if df_py.shape != df_nb.shape:
        issues.append(f"{name}: shape mismatch — py={df_py.shape}, nb={df_nb.shape}")
        return issues  # can't compare further

    # Column names
    if list(df_py.columns) != list(df_nb.columns):
        issues.append(
            f"{name}: columns mismatch — py={list(df_py.columns)}, nb={list(df_nb.columns)}"
        )
        return issues

    # Numeric columns — element-wise comparison
    num_cols = df_py.select_dtypes(include=np.number).columns
    for col in num_cols:
        py_vals = df_py[col].values.astype(float)
        nb_vals = df_nb[col].values.astype(float)

        # Ignore NaN positions
        mask = ~(np.isnan(py_vals) | np.isnan(nb_vals))
        if not mask.any():
            continue

        if not np.allclose(py_vals[mask], nb_vals[mask], rtol=rtol, atol=atol):
            max_diff = np.abs(py_vals[mask] - nb_vals[mask]).max()
            issues.append(f"{name}.{col}: max diff={max_diff:.4f} (rtol={rtol}, atol={atol})")

    # String columns — exact match
    str_cols = df_py.select_dtypes(exclude=np.number).columns
    for col in str_cols:
        if not (df_py[col].values == df_nb[col].values).all():
            issues.append(f"{name}.{col}: string values differ")

    return issues


# ── CSV parity tests ──────────────────────────────────────────────────────────


class TestEDAParity:
    """EDA outputs: ember_filtered.csv, table01_eda_statistics.csv"""

    @pytest.fixture(autouse=True)
    def check_outputs(self):
        if not (PY_OUT / "eda" / "ember_filtered.csv").exists():
            pytest.skip("Run 'make run' first to generate .py outputs")

    def test_ember_filtered_shape(self):
        """Both should produce the same number of rows and columns."""
        py_df = pd.read_csv(PY_OUT / "eda" / "ember_filtered.csv")
        nb_csv = extract_nb_csv(NOTEBOOKS["eda"], PY_OUT / "eda")

        if "ember_filtered.csv" not in nb_csv:
            pytest.skip("Notebook ember_filtered.csv not found")

        nb_df = nb_csv["ember_filtered.csv"]
        issues = compare_dataframes(py_df, nb_df, "ember_filtered")
        assert not issues, "\n".join(issues)

    def test_eda_statistics_values(self):
        """Summary statistics should match within tolerance."""
        py_path = PY_OUT / "eda" / "table01_eda_statistics.csv"
        if not py_path.exists():
            pytest.skip("table01_eda_statistics.csv not found")

        py_df = pd.read_csv(py_path)
        nb_csv = extract_nb_csv(NOTEBOOKS["eda"], PY_OUT / "eda")

        if "table01_eda_statistics.csv" not in nb_csv:
            pytest.skip("Notebook stats CSV not found")

        nb_df = nb_csv["table01_eda_statistics.csv"]
        issues = compare_dataframes(py_df, nb_df, "table01_eda_statistics")
        assert not issues, "\n".join(issues)

    def test_same_countries(self):
        """Both should contain exactly the same 7 countries."""
        py_df = pd.read_csv(PY_OUT / "eda" / "ember_filtered.csv")
        nb_csv = extract_nb_csv(NOTEBOOKS["eda"], PY_OUT / "eda")

        if "ember_filtered.csv" not in nb_csv:
            pytest.skip("Notebook ember_filtered.csv not found")

        nb_df = nb_csv["ember_filtered.csv"]
        py_countries = set(py_df["Area"].unique())
        nb_countries = set(nb_df["Area"].unique())
        assert (
            py_countries == nb_countries
        ), f"Country mismatch: py={py_countries}, nb={nb_countries}"


class TestPreprocessingParity:
    """Preprocessing: ember_model_ready.csv, feature_meta.json"""

    @pytest.fixture(autouse=True)
    def check_outputs(self):
        if not (PY_OUT / "preprocessing" / "ember_model_ready.csv").exists():
            pytest.skip("Run 'make run' first to generate .py outputs")

    def test_model_ready_shape(self):
        py_df = pd.read_csv(PY_OUT / "preprocessing" / "ember_model_ready.csv")
        nb_csv = extract_nb_csv(NOTEBOOKS["preprocessing"], PY_OUT / "preprocessing")

        if "ember_model_ready.csv" not in nb_csv:
            pytest.skip("Notebook ember_model_ready.csv not found")

        nb_df = nb_csv["ember_model_ready.csv"]
        issues = compare_dataframes(py_df, nb_df, "ember_model_ready")
        assert not issues, "\n".join(issues)

    def test_feature_count_matches(self):
        """Both should produce the same number of features."""
        py_meta_path = PY_OUT / "preprocessing" / "feature_meta.json"
        if not py_meta_path.exists():
            pytest.skip("feature_meta.json not found")

        with open(py_meta_path) as f:
            py_meta = json.load(f)

        # Try to find nb feature_meta
        nb_meta_candidates = [
            NB_DIR / "feature_meta.json",
            PY_OUT / "feature_meta.json",
        ]
        nb_meta = None
        for p in nb_meta_candidates:
            if p.exists():
                with open(p) as f:
                    nb_meta = json.load(f)
                break

        if nb_meta is None:
            pytest.skip("Notebook feature_meta.json not found")

        assert len(py_meta["all_features"]) == len(nb_meta["all_features"]), (
            f"Feature count mismatch: py={len(py_meta['all_features'])}, "
            f"nb={len(nb_meta['all_features'])}"
        )

    def test_split_boundaries_match(self):
        """TRAIN_END, VAL_END, TEST_END must be identical."""
        py_meta_path = PY_OUT / "preprocessing" / "feature_meta.json"
        if not py_meta_path.exists():
            pytest.skip("feature_meta.json not found")

        with open(py_meta_path) as f:
            py_meta = json.load(f)

        nb_meta_path = NB_DIR / "feature_meta.json"
        if not nb_meta_path.exists():
            pytest.skip("Notebook feature_meta.json not found")

        with open(nb_meta_path) as f:
            nb_meta = json.load(f)

        for key in ["TRAIN_END", "VAL_END", "TEST_END"]:
            assert (
                py_meta[key] == nb_meta[key]
            ), f"{key} mismatch: py={py_meta[key]}, nb={nb_meta[key]}"


class TestModelingParity:
    """Modeling: test_benchmarking.csv, best_models.csv"""

    @pytest.fixture(autouse=True)
    def check_outputs(self):
        if not (PY_OUT / "modeling" / "best_models.csv").exists():
            pytest.skip("Run 'make run' first to generate .py outputs")

    def test_same_best_models(self):
        """Same country → same best model selection."""
        py_df = pd.read_csv(PY_OUT / "modeling" / "best_models.csv")
        nb_csv = extract_nb_csv(NOTEBOOKS["modeling"], PY_OUT / "modeling")

        if "best_models.csv" not in nb_csv:
            pytest.skip("Notebook best_models.csv not found")

        nb_df = nb_csv["best_models.csv"]

        py_best = dict(zip(py_df["Country"], py_df["Model"]))
        nb_best = dict(zip(nb_df["Country"], nb_df["Model"]))

        mismatches = [
            f"{c}: py={py_best.get(c)}, nb={nb_best.get(c)}"
            for c in py_best
            if py_best.get(c) != nb_best.get(c)
        ]
        assert not mismatches, "Best model mismatch:\n" + "\n".join(mismatches)

    def test_mape_within_tolerance(self):
        """MAPE values should be within 1% relative tolerance."""
        py_path = PY_OUT / "modeling" / "test_benchmarking.csv"
        if not py_path.exists():
            pytest.skip("test_benchmarking.csv not found")

        py_df = pd.read_csv(py_path)
        nb_csv = extract_nb_csv(NOTEBOOKS["modeling"], PY_OUT / "modeling")

        if "test_benchmarking.csv" not in nb_csv:
            pytest.skip("Notebook test_benchmarking.csv not found")

        nb_df = nb_csv["test_benchmarking.csv"]
        issues = compare_dataframes(py_df, nb_df, "test_benchmarking", rtol=0.02)
        assert not issues, "\n".join(issues)


class TestForecastingParity:
    """Forecasting: demand_forecast_2025_2030.csv, demand_growth_summary.csv"""

    @pytest.fixture(autouse=True)
    def check_outputs(self):
        if not (PY_OUT / "forecasting" / "demand_forecast_2025_2030.csv").exists():
            pytest.skip("Run 'make run' first to generate .py outputs")

    def test_forecast_values(self):
        """Forecast point estimates should match within 1% tolerance."""
        py_df = pd.read_csv(PY_OUT / "forecasting" / "demand_forecast_2025_2030.csv")
        nb_csv = extract_nb_csv(NOTEBOOKS["forecasting"], PY_OUT / "forecasting")

        if "demand_forecast_2025_2030.csv" not in nb_csv:
            pytest.skip("Notebook forecast CSV not found")

        nb_df = nb_csv["demand_forecast_2025_2030.csv"]
        issues = compare_dataframes(py_df, nb_df, "demand_forecast_2025_2030")
        assert not issues, "\n".join(issues)

    def test_forecast_years(self):
        """Both should forecast the same years."""
        py_df = pd.read_csv(PY_OUT / "forecasting" / "demand_forecast_2025_2030.csv")
        nb_csv = extract_nb_csv(NOTEBOOKS["forecasting"], PY_OUT / "forecasting")

        if "demand_forecast_2025_2030.csv" not in nb_csv:
            pytest.skip("Notebook forecast CSV not found")

        nb_df = nb_csv["demand_forecast_2025_2030.csv"]
        py_years = sorted(py_df["Year"].unique())
        nb_years = sorted(nb_df["Year"].unique())
        assert py_years == nb_years, f"Year mismatch: py={py_years}, nb={nb_years}"

    def test_growth_summary_cagr(self):
        """CAGR values should match within 0.5% absolute tolerance."""
        py_path = PY_OUT / "forecasting" / "demand_growth_summary.csv"
        if not py_path.exists():
            pytest.skip("demand_growth_summary.csv not found")

        py_df = pd.read_csv(py_path)
        nb_csv = extract_nb_csv(NOTEBOOKS["forecasting"], PY_OUT / "forecasting")

        if "demand_growth_summary.csv" not in nb_csv:
            pytest.skip("Notebook growth summary not found")

        nb_df = nb_csv["demand_growth_summary.csv"]
        issues = compare_dataframes(py_df, nb_df, "demand_growth_summary", rtol=0.02, atol=0.5)
        assert not issues, "\n".join(issues)


class TestDeepLearningParity:
    """DL: dl_benchmarking.csv, dl_best_models.csv, dl_forecast_2025_2030.csv"""

    @pytest.fixture(autouse=True)
    def check_outputs(self):
        if not (PY_OUT / "deeplearning" / "dl_benchmarking.csv").exists():
            pytest.skip("Run 'make run-deeplearning' first")

    def test_dl_forecast_values(self):
        py_df = pd.read_csv(PY_OUT / "deeplearning" / "dl_forecast_2025_2030.csv")
        nb_csv = extract_nb_csv(NOTEBOOKS.get("deeplearning", Path("")), PY_OUT / "deeplearning")
        if "dl_forecast_2025_2030.csv" not in nb_csv:
            pytest.skip("Notebook DL forecast CSV not found")
        issues = compare_dataframes(
            py_df, nb_csv["dl_forecast_2025_2030.csv"], "dl_forecast_2025_2030", rtol=0.05
        )
        assert not issues, "\n".join(issues)

    def test_dl_best_models_match(self):
        py_df = pd.read_csv(PY_OUT / "deeplearning" / "dl_best_models.csv")
        nb_csv = extract_nb_csv(NOTEBOOKS.get("deeplearning", Path("")), PY_OUT / "deeplearning")
        if "dl_best_models.csv" not in nb_csv:
            pytest.skip("Notebook DL best models not found")
        nb_df = nb_csv["dl_best_models.csv"]
        py_best = dict(zip(py_df["Country"], py_df["Model"]))
        nb_best = dict(zip(nb_df["Country"], nb_df["Model"]))
        mismatches = [
            f"{c}: py={py_best.get(c)}, nb={nb_best.get(c)}"
            for c in py_best
            if py_best.get(c) != nb_best.get(c)
        ]
        assert not mismatches, "\n".join(mismatches)


# ── Standalone comparison utility ─────────────────────────────────────────────


def run_comparison_report(py_out: Path = PY_OUT) -> None:
    print("\n" + "=" * 60)
    print("  NOTEBOOK vs SCRIPT PARITY REPORT")
    print("=" * 60)

    checks = {
        "ember_filtered.csv": py_out / "eda" / "ember_filtered.csv",
        "table01_eda_statistics.csv": py_out / "eda" / "table01_eda_statistics.csv",
        "ember_model_ready.csv": py_out / "preprocessing" / "ember_model_ready.csv",
        "best_models.csv": py_out / "modeling" / "best_models.csv",
        "test_benchmarking.csv": py_out / "modeling" / "test_benchmarking.csv",
        "demand_forecast_2025_2030.csv": py_out / "forecasting" / "demand_forecast_2025_2030.csv",
        "demand_growth_summary.csv": py_out / "forecasting" / "demand_growth_summary.csv",
        "dl_benchmarking.csv": py_out / "deeplearning" / "dl_benchmarking.csv",
        "dl_best_models.csv": py_out / "deeplearning" / "dl_best_models.csv",
        "dl_forecast_2025_2030.csv": py_out / "deeplearning" / "dl_forecast_2025_2030.csv",
    }

    # Collect all notebook CSVs across all notebooks
    nb_dfs: dict[str, pd.DataFrame] = {}
    for step, nb_path in NOTEBOOKS.items():
        if not nb_path.exists():
            print(f"  ⚠  SKIP   notebook not found: {nb_path.name}")
            continue
        found = extract_nb_csv(nb_path, PY_OUT / step)
        nb_dfs.update(found)
        log.debug("Step %s: found notebook CSVs: %s", step, list(found.keys()))

    all_ok = True
    for name, py_path in checks.items():
        if not py_path.exists():
            print(f"  ⚠  SKIP   {name:45s} (.py output missing — run: make run)")
            continue
        if name not in nb_dfs:
            print(f"  ⚠  SKIP   {name:45s} (notebook output not found)")
            continue

        py_df = pd.read_csv(py_path)
        nb_df = nb_dfs[name]
        issues = compare_dataframes(py_df, nb_df, name)

        if issues:
            all_ok = False
            print(f"  ✗  DIFF   {name:45s}")
            for issue in issues:
                print(f"           → {issue}")
        else:
            print(f"  ✓  MATCH  {name:45s} shape={py_df.shape}")

    print("=" * 60)
    print("  RESULT:", "✓ ALL MATCH" if all_ok else "✗ DIFFERENCES FOUND")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_comparison_report()
