"""
tests/test_significance.py — Unit tests for src/modeling/significance.py

Covers:
    diebold_mariano()       — statistical properties, degenerate variance
    wilcoxon_test()         — zero-diff edge case, real difference detection
    paired_ttest()          — basic sanity
    run_significance_tests()— pairwise combination logic, year alignment,
                               skip conditions (too few models, too few
                               overlapping years), better_model selection
    significance_summary()  — per-country aggregation correctness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.significance import (
    diebold_mariano,
    paired_ttest,
    run_significance_tests,
    significance_summary,
    wilcoxon_test,
)


# ── diebold_mariano ────────────────────────────────────────────────────────


def test_dm_identical_errors_gives_no_significant_difference():
    """Two models with identical errors have zero loss differential."""
    errors = np.array([1.0, -2.0, 3.0, -1.5, 2.5])
    dm_stat, p = diebold_mariano(errors, errors.copy())

    assert dm_stat == pytest.approx(0.0, abs=1e-6)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_dm_model_a_clearly_worse_gives_positive_stat():
    """
    Model A has much larger errors than Model B -> loss differential
    (errors_a^2 - errors_b^2) is consistently positive -> positive DM stat.
    """
    errors_a = np.array([10.0, -12.0, 11.0, -9.0, 10.5])   # large errors
    errors_b = np.array([1.0, -1.5, 0.8, -1.2, 1.1])        # small errors

    dm_stat, p = diebold_mariano(errors_a, errors_b)
    assert dm_stat > 0


def test_dm_zero_variance_returns_neutral_result():
    """
    Degenerate case: if the loss differential has ~zero variance,
    the function must not raise (division by near-zero) and should
    return the documented neutral fallback (0.0, 1.0).
    """
    errors_a = np.array([1.0, 1.0, 1.0, 1.0])
    errors_b = np.array([1.0, 1.0, 1.0, 1.0])
    dm_stat, p = diebold_mariano(errors_a, errors_b)
    assert dm_stat == 0.0
    assert p == 1.0


def test_dm_returns_valid_p_value_range():
    rng = np.random.default_rng(42)
    errors_a = rng.normal(0, 2, size=20)
    errors_b = rng.normal(0, 1, size=20)
    _, p = diebold_mariano(errors_a, errors_b)
    assert 0.0 <= p <= 1.0


# ── wilcoxon_test ──────────────────────────────────────────────────────────


def test_wilcoxon_all_zero_diff_returns_neutral():
    """If |errors_a| == |errors_b| everywhere, there's nothing to test."""
    errors_a = np.array([1.0, -2.0, 3.0])
    errors_b = np.array([-1.0, 2.0, -3.0])  # same absolute values
    stat, p = wilcoxon_test(errors_a, errors_b)
    assert stat == 0.0
    assert p == 1.0


def test_wilcoxon_detects_real_difference():
    errors_a = np.array([5.0, 6.0, 5.5, 6.5, 5.2, 6.1, 5.8])
    errors_b = np.array([1.0, 1.2, 0.9, 1.1, 1.0, 1.3, 0.8])
    stat, p = wilcoxon_test(errors_a, errors_b)
    assert 0.0 <= p <= 1.0
    # Model B's errors are consistently smaller -> should be a low p-value
    assert p < 0.10


def test_wilcoxon_handles_tiny_sample_without_raising():
    """scipy.stats.wilcoxon can raise ValueError on very small/degenerate
    input — the function must catch it and return the neutral fallback."""
    errors_a = np.array([1.0])
    errors_b = np.array([2.0])
    stat, p = wilcoxon_test(errors_a, errors_b)
    assert isinstance(stat, float)
    assert isinstance(p, float)


# ── paired_ttest ───────────────────────────────────────────────────────────


def test_paired_ttest_returns_valid_floats():
    errors_a = np.array([2.0, 2.5, 1.8, 2.2, 2.1])
    errors_b = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    stat, p = paired_ttest(errors_a, errors_b)
    assert isinstance(stat, float)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


# ── run_significance_tests ────────────────────────────────────────────────


def _wf_row(country, year, model, y_actual, y_pred):
    return {"Country": country, "Year": year, "Model": model,
            "y_actual": y_actual, "y_pred": y_pred}


def _make_wf_predictions() -> pd.DataFrame:
    """Two models, four years, one country — the minimum realistic shape."""
    rows = []
    years = [2021, 2022, 2023, 2024]
    actuals = [100.0, 105.0, 98.0, 110.0]
    ridge_preds = [101.0, 104.0, 99.0, 109.0]      # close to actual
    naive_preds = [90.0, 95.0, 130.0, 70.0]        # far off

    for y, a, r, n in zip(years, actuals, ridge_preds, naive_preds):
        rows.append(_wf_row("Tunisia", y, "Ridge", a, r))
        rows.append(_wf_row("Tunisia", y, "Naive", a, n))

    return pd.DataFrame(rows)


def test_run_significance_tests_produces_one_row_per_pair():
    df = _make_wf_predictions()
    result = run_significance_tests(df)

    # 1 country x C(2,2)=1 pair -> exactly one row
    assert len(result) == 1
    row = result.iloc[0]
    assert row["Country"] == "Tunisia"
    assert {row["Model_A"], row["Model_B"]} == {"Ridge", "Naive"}


def test_run_significance_tests_identifies_better_model_by_mape():
    df = _make_wf_predictions()
    result = run_significance_tests(df)
    row = result.iloc[0]

    # Ridge's predictions are much closer to actual -> Ridge must win
    assert row["better_model"] == "Ridge"
    assert row["MAPE_A"] != row["MAPE_B"]


def test_run_significance_tests_skips_country_with_single_model():
    rows = [_wf_row("Austria", y, "Ridge", 70.0 + i, 70.0 + i)
            for i, y in enumerate([2021, 2022, 2023, 2024])]
    df = pd.DataFrame(rows)

    result = run_significance_tests(df)
    assert result.empty


def test_run_significance_tests_skips_pairs_with_too_few_overlapping_years():
    """Pairs need >= 3 overlapping years; fewer than that must be skipped."""
    rows = [
        _wf_row("Egypt", 2021, "Ridge", 200.0, 201.0),
        _wf_row("Egypt", 2022, "Ridge", 210.0, 209.0),
        _wf_row("Egypt", 2021, "ARIMA", 200.0, 195.0),
        _wf_row("Egypt", 2022, "ARIMA", 210.0, 215.0),
        # only 2 overlapping years -> below the len(merged) < 3 cutoff
    ]
    df = pd.DataFrame(rows)

    result = run_significance_tests(df)
    assert result.empty


def test_run_significance_tests_respects_alpha_threshold():
    df = _make_wf_predictions()
    # Extremely strict alpha -> nothing should be flagged significant
    result_strict = run_significance_tests(df, alpha=1e-10)
    assert not result_strict["significant_DM"].any()

    # Very loose alpha -> everything with p<1 will be "significant"
    result_loose = run_significance_tests(df, alpha=0.999999)
    assert result_loose["significant_DM"].any() or result_loose["significant_WX"].any()


def test_run_significance_tests_multiple_countries_independent():
    df = pd.concat([
        _make_wf_predictions(),
        _make_wf_predictions().assign(Country="Kuwait"),
    ])
    result = run_significance_tests(df)
    assert set(result["Country"]) == {"Tunisia", "Kuwait"}
    assert len(result) == 2  # one pair per country


# ── significance_summary ──────────────────────────────────────────────────


def test_significance_summary_counts_correctly():
    results = pd.DataFrame({
        "Country":        ["Tunisia", "Tunisia", "Kuwait"],
        "Model_A":        ["Ridge", "Ridge", "ARIMA"],
        "Model_B":        ["Naive", "Holt", "Naive"],
        "significant_DM": [True, False, False],
        "significant_WX": [True, True, False],
    })

    summary = significance_summary(results)

    tunisia_row = summary[summary["Country"] == "Tunisia"].iloc[0]
    assert tunisia_row["Total_pairs"] == 2
    assert tunisia_row["Significant_DM"] == 1
    assert tunisia_row["Significant_WX"] == 2
    assert tunisia_row["Any_significant"] == True

    kuwait_row = summary[summary["Country"] == "Kuwait"].iloc[0]
    assert kuwait_row["Total_pairs"] == 1
    assert kuwait_row["Any_significant"] == False


def test_significance_summary_empty_input_returns_empty_df():
    results = pd.DataFrame(columns=[
        "Country", "Model_A", "Model_B", "significant_DM", "significant_WX",
    ])
    summary = significance_summary(results)
    assert summary.empty
