"""
tests/test_robustness.py — Unit tests for src/modeling/robustness.py

Covers:
    compute_robustness_score()   — threshold classification, direction
                                    flips, edge cases (missing/zero anchor,
                                    single-point series, empty input)
    robustness_vs_accuracy()     — merge correctness, rank computation,
                                    axes_agree logic
    flag_best_model_conflicts()  — conflict detection against the exact
                                    failure mode the module exists to catch
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.robustness import (
    compute_robustness_score,
    flag_best_model_conflicts,
    robustness_vs_accuracy,
    WATCH_THRESHOLD,
    UNSTABLE_THRESHOLD,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _forecast_df(country: str, model: str, values: list[float], start_year: int = 2025) -> pd.DataFrame:
    """Build a minimal [Country, Model, Year, Forecast] frame for one series."""
    years = list(range(start_year, start_year + len(values)))
    return pd.DataFrame({
        "Country": country,
        "Model": model,
        "Year": years,
        "Forecast": values,
    })


# ── compute_robustness_score: threshold classification ──────────────────────


def test_stable_trajectory_flagged_stable():
    """A smooth trajectory staying within 20% of the anchor is STABLE."""
    df = _forecast_df("Tunisia", "Ridge", [26.0, 26.5, 27.0, 27.5, 28.0, 28.5])
    last_known = {"Tunisia": 26.0}

    result = compute_robustness_score(df, last_known)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["robustness_flag"] == "STABLE"
    assert row["max_deviation_pct"] < WATCH_THRESHOLD


def test_watch_trajectory_flagged_watch():
    """A trajectory deviating 20-40% from the anchor is WATCH, not STABLE/UNSTABLE."""
    # last_known=100, worst deviation should land at ~30%
    df = _forecast_df("Kuwait", "Theta", [100.0, 105.0, 115.0, 125.0, 130.0, 130.0])
    last_known = {"Kuwait": 100.0}

    result = compute_robustness_score(df, last_known)

    row = result.iloc[0]
    assert WATCH_THRESHOLD < row["max_deviation_pct"] <= UNSTABLE_THRESHOLD
    assert row["robustness_flag"] == "WATCH"


def test_unstable_trajectory_flagged_unstable():
    """
    Reproduces the exact real-world failure pattern observed in this
    project: a model whose recursive forecast collapses toward/through
    zero (e.g. BayesianRidge on Germany, -10.9 TWh from a base of 508.2).
    """
    df = _forecast_df("Germany", "BayesianRidge",
                       [480.0, 300.0, 100.0, -10.9, 50.0, 90.0])
    last_known = {"Germany": 508.2}

    result = compute_robustness_score(df, last_known)

    row = result.iloc[0]
    assert row["max_deviation_pct"] > UNSTABLE_THRESHOLD
    assert row["robustness_flag"] == "UNSTABLE"
    # Sanity: matches the ~102% figure actually observed in production logs
    assert row["max_deviation_pct"] == pytest.approx(102.2, abs=1.0)


def test_boundary_exactly_at_watch_threshold_is_not_watch():
    """max_dev == WATCH_THRESHOLD exactly should be STABLE (strict '>' comparison)."""
    base = 100.0
    boundary_val = base * (1 + WATCH_THRESHOLD / 100)  # exactly 20% above base
    df = _forecast_df("Austria", "RandomForest", [base, boundary_val, boundary_val, boundary_val, boundary_val, boundary_val])
    last_known = {"Austria": base}

    result = compute_robustness_score(df, last_known)
    assert result.iloc[0]["robustness_flag"] == "STABLE"


# ── compute_robustness_score: max_yoy_change_pct ─────────────────────────────


def test_yoy_catches_v_shaped_collapse_even_with_plausible_endpoint():
    """
    A trajectory that collapses mid-horizon then recovers close to the
    anchor should still be caught by max_yoy_change_pct, even though
    max_deviation_pct alone might look acceptable at the final year.
    """
    # Endpoint (90) is close to base (100) -> low max_deviation_pct
    # But the mid-horizon collapse to 10 is a huge internal YoY jump
    df = _forecast_df("France", "ElasticNet", [95.0, 10.0, 15.0, 60.0, 85.0, 90.0])
    last_known = {"France": 100.0}

    result = compute_robustness_score(df, last_known)
    row = result.iloc[0]

    # The 95 -> 10 step is an ~89% YoY drop
    assert row["max_yoy_change_pct"] > 80.0


# ── compute_robustness_score: direction flips ────────────────────────────────


def test_monotonic_trajectory_has_zero_flips():
    df = _forecast_df("Canada", "LinearTrend", [600.0, 610.0, 620.0, 630.0, 640.0, 650.0])
    last_known = {"Canada": 600.0}

    result = compute_robustness_score(df, last_known)
    assert result.iloc[0]["direction_flips"] == 0


def test_oscillating_trajectory_has_multiple_flips():
    """Up-down-up-down pattern should register multiple direction reversals."""
    df = _forecast_df("Egypt", "ARIMA", [240.0, 260.0, 240.0, 260.0, 240.0, 260.0])
    last_known = {"Egypt": 240.0}

    result = compute_robustness_score(df, last_known)
    assert result.iloc[0]["direction_flips"] >= 3


# ── compute_robustness_score: edge cases ─────────────────────────────────────


def test_country_missing_from_last_known_is_skipped():
    """A country with no anchor value should be silently excluded, not raise."""
    df = _forecast_df("Tunisia", "Ridge", [26.0, 27.0, 28.0, 29.0, 30.0, 31.0])
    last_known: dict[str, float] = {}  # Tunisia not present

    result = compute_robustness_score(df, last_known)
    assert result.empty


def test_zero_anchor_is_skipped():
    """A zero last_known value would cause division by zero — must be skipped."""
    df = _forecast_df("Tunisia", "Ridge", [26.0, 27.0, 28.0, 29.0, 30.0, 31.0])
    last_known = {"Tunisia": 0.0}

    result = compute_robustness_score(df, last_known)
    assert result.empty


def test_single_point_series_is_skipped():
    """A forecast with fewer than 2 points can't compute YoY change — skip it."""
    df = _forecast_df("Tunisia", "Ridge", [26.0])
    last_known = {"Tunisia": 26.0}

    result = compute_robustness_score(df, last_known)
    assert result.empty


def test_empty_forecast_df_returns_empty_result_not_error():
    df = pd.DataFrame(columns=["Country", "Model", "Year", "Forecast"])
    result = compute_robustness_score(df, {"Tunisia": 26.0})
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_multiple_countries_and_models_scored_independently():
    """Each (Country, Model) pair gets its own row, computed independently."""
    df = pd.concat([
        _forecast_df("Tunisia", "Ridge", [26.0, 27.0, 28.0, 29.0, 30.0, 31.0]),
        _forecast_df("Tunisia", "BayesianRidge", [26.0, 15.0, 5.0, -5.0, 10.0, 20.0]),
        _forecast_df("Germany", "Holt", [500.0, 495.0, 490.0, 485.0, 480.0, 475.0]),
    ])
    last_known = {"Tunisia": 26.0, "Germany": 508.0}

    result = compute_robustness_score(df, last_known)

    assert len(result) == 3
    assert set(result["Country"]) == {"Tunisia", "Germany"}
    # Ridge (stable) and BayesianRidge (unstable) for Tunisia must differ
    tunisia_rows = result[result["Country"] == "Tunisia"]
    flags = set(tunisia_rows["robustness_flag"])
    assert "STABLE" in flags
    assert "UNSTABLE" in flags


def test_result_sorted_by_country_then_deviation():
    df = pd.concat([
        _forecast_df("Tunisia", "BayesianRidge", [26.0, 15.0, 5.0, -5.0, 10.0, 20.0]),
        _forecast_df("Tunisia", "Ridge", [26.0, 27.0, 28.0, 29.0, 30.0, 31.0]),
    ])
    last_known = {"Tunisia": 26.0}

    result = compute_robustness_score(df, last_known)
    # Within Tunisia, the more stable model (lower max_deviation_pct) comes first
    assert result.iloc[0]["max_deviation_pct"] <= result.iloc[1]["max_deviation_pct"]


# ── robustness_vs_accuracy ────────────────────────────────────────────────────


def test_robustness_vs_accuracy_merges_and_ranks_correctly():
    # 4 models so accuracy_rank/robustness_rank can genuinely diverge by
    # more than 2 — with only 2 models the rank gap can never exceed 1,
    # making axes_agree trivially True regardless of actual disagreement.
    robustness_df = pd.DataFrame({
        "Country":            ["Tunisia"] * 4,
        "Model":              ["Ridge", "BayesianRidge", "Holt", "LinearTrend"],
        "max_deviation_pct":  [5.0, 90.0, 8.0, 6.0],
        "max_yoy_change_pct": [3.0, 60.0, 4.0, 3.5],
        "trajectory_std":     [1.0, 15.0, 1.2, 1.1],
        "direction_flips":    [0, 2, 0, 0],
        "robustness_flag":    ["STABLE", "UNSTABLE", "STABLE", "STABLE"],
    })
    accuracy_df = pd.DataFrame({
        "Country": ["Tunisia"] * 4,
        "Model":   ["Ridge", "BayesianRidge", "Holt", "LinearTrend"],
        "MAPE":    [2.5, 0.5, 3.0, 2.8],  # BayesianRidge wins on accuracy...
    })

    result = robustness_vs_accuracy(robustness_df, accuracy_df)

    br_row = result[result["Model"] == "BayesianRidge"].iloc[0]

    # ...but loses badly on robustness rank (worst of 4) -> this is the
    # accurate-but-unstable conflict case the whole module exists to catch
    assert br_row["accuracy_rank"] == 1     # best MAPE, out of 4
    assert br_row["robustness_rank"] == 4   # worst robustness, out of 4
    assert br_row["axes_agree"] == False    # rank_gap = 3 > 2


def test_robustness_vs_accuracy_axes_agree_when_ranks_close():
    robustness_df = pd.DataFrame({
        "Country":            ["Austria", "Austria"],
        "Model":              ["RandomForest", "XGBoost"],
        "max_deviation_pct":  [5.0, 6.0],
        "max_yoy_change_pct": [3.0, 3.5],
        "trajectory_std":     [1.0, 1.2],
        "direction_flips":    [0, 0],
        "robustness_flag":    ["STABLE", "STABLE"],
    })
    accuracy_df = pd.DataFrame({
        "Country": ["Austria", "Austria"],
        "Model":   ["RandomForest", "XGBoost"],
        "MAPE":    [1.5, 1.6],
    })

    result = robustness_vs_accuracy(robustness_df, accuracy_df)
    assert result["axes_agree"].all()


# ── flag_best_model_conflicts ─────────────────────────────────────────────────


def test_flag_best_model_conflicts_detects_unstable_winner():
    best_models_df = pd.DataFrame({
        "Country": ["Germany"],
        "Model":   ["BayesianRidge"],
        "MAPE":    [0.5],
    })
    robustness_df = pd.DataFrame({
        "Country":           ["Germany"],
        "Model":             ["BayesianRidge"],
        "robustness_flag":   ["UNSTABLE"],
        "max_deviation_pct": [102.2],
    })

    result = flag_best_model_conflicts(best_models_df, robustness_df)

    assert len(result) == 1
    assert result.iloc[0]["conflict"] == True
    assert result.iloc[0]["Best_Model"] == "BayesianRidge"
    assert "Model" not in result.columns  # renamed to Best_Model


def test_flag_best_model_conflicts_no_conflict_when_stable():
    best_models_df = pd.DataFrame({
        "Country": ["Canada"],
        "Model":   ["LinearTrend"],
        "MAPE":    [0.86],
    })
    robustness_df = pd.DataFrame({
        "Country":           ["Canada"],
        "Model":             ["LinearTrend"],
        "robustness_flag":   ["STABLE"],
        "max_deviation_pct": [3.2],
    })

    result = flag_best_model_conflicts(best_models_df, robustness_df)
    assert result.iloc[0]["conflict"] == False


def test_flag_best_model_conflicts_counts_across_countries():
    """Reproduces the real project result: 5/7 countries flagged."""
    best_models_df = pd.DataFrame({
        "Country": ["Germany", "Egypt", "Canada", "France", "Kuwait", "Austria", "Tunisia"],
        "Model":   ["BayesianRidge"] * 5 + ["RandomForest", "Ridge"],
        "MAPE":    [0.5] * 7,
    })
    robustness_df = pd.DataFrame({
        "Country": ["Germany", "Egypt", "Canada", "France", "Kuwait", "Austria", "Tunisia"],
        "Model":   ["BayesianRidge"] * 5 + ["RandomForest", "Ridge"],
        "robustness_flag": ["UNSTABLE"] * 5 + ["STABLE", "STABLE"],
        "max_deviation_pct": [102.2, 73.8, 154.1, 79.0, 55.4, 5.0, 4.0],
    })

    result = flag_best_model_conflicts(best_models_df, robustness_df)
    assert int(result["conflict"].sum()) == 5
    assert len(result) == 7
