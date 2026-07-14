"""
src/modeling/significance.py
Statistical significance tests for forecast comparison
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray,
                    h: int = 1) -> tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Parameters
    ----------
    errors_a, errors_b : forecast errors (actual - predicted)
    h                  : forecast horizon (1 for 1-step)

    Returns
    -------
    (dm_stat, p_value)
    """
    d   = errors_a**2 - errors_b**2   # loss differential
    n   = len(d)
    d_bar = np.mean(d)

    # Newey-West variance estimate (accounts for autocorrelation)
    gamma0 = np.var(d, ddof=1)
    if n > 1:
        gamma1 = np.cov(d[:-1], d[1:])[0, 1]
    else:
        gamma1 = 0

    var_d  = (gamma0 + 2 * gamma1) / n
    if var_d <= 0:
        return 0.0, 1.0

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def wilcoxon_test(errors_a: np.ndarray,
                  errors_b: np.ndarray) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired forecast errors.
    Non-parametric — no normality assumption.
    Best for small samples (n < 30).

    Returns (statistic, p_value)
    """
    diff = np.abs(errors_a) - np.abs(errors_b)
    if np.all(diff == 0):
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(diff, zero_method="wilcox",
                                  alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def paired_ttest(errors_a: np.ndarray,
                 errors_b: np.ndarray) -> tuple[float, float]:
    """
    Paired t-test for mean error difference.
    Assumes normality — use only when n > 30.
    """
    diff = np.abs(errors_a) - np.abs(errors_b)
    stat, p = stats.ttest_1samp(diff, 0)
    return float(stat), float(p)


def run_significance_tests(
    wf_predictions: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run all significance tests comparing every model pair per country.

    Parameters
    ----------
    wf_predictions : walk-forward predictions DataFrame
                     columns: [Country, Year, Model, y_actual, y_pred]
    alpha          : significance level (default 0.05)

    Returns
    -------
    DataFrame with columns:
      [Country, Model_A, Model_B, DM_stat, DM_p, WX_stat, WX_p,
       significant_DM, significant_WX, better_model]
    """
    rows = []
    for country, grp in wf_predictions.groupby("Country"):
        models   = grp["Model"].unique()
        if len(models) < 2:
            continue

        for m_a, m_b in combinations(models, 2):
            g_a = grp[grp["Model"] == m_a].sort_values("Year")
            g_b = grp[grp["Model"] == m_b].sort_values("Year")

            # Align on same years
            merged = g_a[["Year","y_actual","y_pred"]].merge(
                g_b[["Year","y_pred"]].rename(columns={"y_pred":"y_pred_b"}),
                on="Year"
            )
            if len(merged) < 3:
                continue

            err_a = (merged["y_actual"] - merged["y_pred"]).values
            err_b = (merged["y_actual"] - merged["y_pred_b"]).values

            dm_stat, dm_p   = diebold_mariano(err_a, err_b)
            wx_stat, wx_p   = wilcoxon_test(err_a, err_b)

            mape_a = np.mean(np.abs(err_a / merged["y_actual"])) * 100
            mape_b = np.mean(np.abs(err_b / merged["y_actual"])) * 100
            better = m_a if mape_a < mape_b else m_b

            rows.append({
                "Country":        country,
                "Model_A":        m_a,
                "Model_B":        m_b,
                "MAPE_A":         round(mape_a, 3),
                "MAPE_B":         round(mape_b, 3),
                "DM_stat":        round(dm_stat, 3),
                "DM_p":           round(dm_p, 4),
                "WX_stat":        round(wx_stat, 3),
                "WX_p":           round(wx_p, 4),
                "significant_DM": dm_p < alpha,
                "significant_WX": wx_p < alpha,
                "better_model":   better,
            })

    return pd.DataFrame(rows)


def significance_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Per-country summary: which model wins and is it significant?
    """
    rows = []
    for country, grp in results.groupby("Country"):
        sig_dm = grp[grp["significant_DM"]]
        sig_wx = grp[grp["significant_WX"]]

        rows.append({
            "Country":           country,
            "Total_pairs":       len(grp),
            "Significant_DM":    len(sig_dm),
            "Significant_WX":    len(sig_wx),
            "Any_significant":   len(sig_dm) > 0 or len(sig_wx) > 0,
        })
    return pd.DataFrame(rows)