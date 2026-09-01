"""
compute_significance_corrected.py — HLN-corrected DM + pooled cross-country test
====================================================================================
Addresses two issues raised in reviewer feedback on the original
significance.py output:

1. DM test lacked the Harvey-Leybourne-Newbold (1997) small-sample
   correction. At T=4 test years, uncorrected DM's asymptotic normal
   approximation is known to be liberal (inflates significance) --
   exactly the pattern observed (DM: 7-29/55 significant pairs per
   country, Wilcoxon: 0/55 in every country). This was NOT a genuine
   disagreement between two valid tests; it was uncorrected DM
   overstating significance at T=4.

2. Per-country T=4 is fragile as a headline claim regardless of
   correction. Adds a POOLED cross-country test (T=4 years x 7
   countries = 28), normalized to percentage-error terms first (same
   convention as MAPE elsewhere in this paper) so no single country's
   absolute TWh scale dominates the pooled sample.

Usage: python compute_significance_corrected.py
Reads:  outputs/modeling/wf_test_predictions.csv
Writes: outputs/modeling/significance_percountry_hln.csv
        outputs/modeling/significance_pooled.csv
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
ALPHA = 0.05


def diebold_mariano_hln(pct_err1: np.ndarray, pct_err2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample
    correction. Operates on percentage-error series (not raw TWh), so it
    is directly comparable across countries of very different scale.

    Returns (hln_statistic, p_value), using a t-distribution with T-1
    degrees of freedom instead of the standard normal DM originally used
    -- essential at T as small as 4, where the asymptotic normal
    approximation is known to be liberal.
    """
    d = pct_err1**2 - pct_err2**2  # squared percentage-error loss differential
    T = len(d)
    if T < 2:
        return np.nan, np.nan

    d_bar = d.mean()
    # Newey-West-style long-run variance (lag truncation per Diebold-Mariano
    # original recommendation: h-1 lags for an h-step loss differential)
    gamma0 = np.var(d, ddof=1)
    var_d = gamma0
    for lag in range(1, h):
        if lag < T:
            cov = np.cov(d[:-lag], d[lag:])[0, 1]
            var_d += 2 * cov
    var_d = max(var_d, 1e-12)  # guard against non-positive variance estimates

    dm_stat = d_bar / np.sqrt(var_d / T)

    # HLN small-sample correction factor
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm_stat * hln_factor

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_hln), df=max(T - 1, 1)))
    return float(dm_hln), float(p_value)


def wilcoxon_signed_rank(pct_err1: np.ndarray, pct_err2: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test on |percentage error| differences."""
    diff = np.abs(pct_err1) - np.abs(pct_err2)
    diff = diff[diff != 0]  # scipy's wilcoxon drops zeros; do so explicitly
    if len(diff) < 2:
        return np.nan, np.nan
    try:
        stat, p = stats.wilcoxon(diff)
        return float(stat), float(p)
    except ValueError:
        return np.nan, np.nan


def pct_error(y_actual: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Signed percentage error, same convention as this project's MAPE."""
    y_actual = np.asarray(y_actual, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_actual != 0
    out = np.full_like(y_actual, np.nan)
    out[mask] = (y_pred[mask] - y_actual[mask]) / y_actual[mask] * 100
    return out


def main():
    res = pd.read_csv("outputs/modeling/wf_test_predictions.csv")
    models = sorted(res["Model"].unique())
    print(f"Models found: {len(models)} -> {models}")
    print(f"Expected pairs per country: C({len(models)},2) = {len(models) * (len(models) - 1) // 2}")

    # ── Per-country, HLN-corrected ──────────────────────────────────────────
    percountry_rows = []
    for country in COUNTRIES:
        sub = res[res["Country"] == country]
        for m1, m2 in itertools.combinations(models, 2):
            s1 = sub[sub["Model"] == m1].sort_values("Year")
            s2 = sub[sub["Model"] == m2].sort_values("Year")
            if s1.empty or s2.empty or len(s1) != len(s2):
                continue
            pe1 = pct_error(s1["y_actual"].values, s1["y_pred"].values)
            pe2 = pct_error(s2["y_actual"].values, s2["y_pred"].values)
            valid = ~(np.isnan(pe1) | np.isnan(pe2))
            pe1, pe2 = pe1[valid], pe2[valid]
            if len(pe1) < 2:
                continue

            dm_stat, dm_p = diebold_mariano_hln(pe1, pe2)
            wx_stat, wx_p = wilcoxon_signed_rank(pe1, pe2)
            percountry_rows.append({
                "Country": country, "Model_A": m1, "Model_B": m2,
                "T": len(pe1), "DM_HLN_stat": dm_stat, "DM_HLN_p": dm_p,
                "DM_HLN_significant": dm_p < ALPHA if not np.isnan(dm_p) else False,
                "WX_stat": wx_stat, "WX_p": wx_p,
                "WX_significant": wx_p < ALPHA if not np.isnan(wx_p) else False,
            })

    percountry_df = pd.DataFrame(percountry_rows)
    summary = percountry_df.groupby("Country").agg(
        DM_HLN_significant=("DM_HLN_significant", "sum"),
        WX_significant=("WX_significant", "sum"),
        Total_pairs=("Country", "count"),
    ).reset_index()
    print("\n=== Per-country (HLN-corrected) ===")
    print(summary.to_string(index=False))
    percountry_df.to_csv("outputs/modeling/significance_percountry_hln.csv", index=False)

    # ── Pooled across all 7 countries (T=4x7=28) ────────────────────────────
    pooled_rows = []
    for m1, m2 in itertools.combinations(models, 2):
        pe1_all, pe2_all = [], []
        for country in COUNTRIES:
            sub = res[res["Country"] == country]
            s1 = sub[sub["Model"] == m1].sort_values("Year")
            s2 = sub[sub["Model"] == m2].sort_values("Year")
            if s1.empty or s2.empty or len(s1) != len(s2):
                continue
            pe1 = pct_error(s1["y_actual"].values, s1["y_pred"].values)
            pe2 = pct_error(s2["y_actual"].values, s2["y_pred"].values)
            valid = ~(np.isnan(pe1) | np.isnan(pe2))
            pe1_all.append(pe1[valid])
            pe2_all.append(pe2[valid])

        if not pe1_all:
            continue
        pe1_pooled = np.concatenate(pe1_all)
        pe2_pooled = np.concatenate(pe2_all)
        if len(pe1_pooled) < 2:
            continue

        dm_stat, dm_p = diebold_mariano_hln(pe1_pooled, pe2_pooled)
        wx_stat, wx_p = wilcoxon_signed_rank(pe1_pooled, pe2_pooled)
        pooled_rows.append({
            "Model_A": m1, "Model_B": m2, "T_pooled": len(pe1_pooled),
            "DM_HLN_stat": dm_stat, "DM_HLN_p": dm_p,
            "DM_HLN_significant": dm_p < ALPHA if not np.isnan(dm_p) else False,
            "WX_stat": wx_stat, "WX_p": wx_p,
            "WX_significant": wx_p < ALPHA if not np.isnan(wx_p) else False,
        })

    pooled_df = pd.DataFrame(pooled_rows)
    n_dm_sig = pooled_df["DM_HLN_significant"].sum()
    n_wx_sig = pooled_df["WX_significant"].sum()
    print(f"\n=== Pooled cross-country (T=28) ===")
    print(f"DM (HLN-corrected) significant: {n_dm_sig} / {len(pooled_df)}")
    print(f"Wilcoxon significant:           {n_wx_sig} / {len(pooled_df)}")
    print(pooled_df.sort_values("DM_HLN_p").to_string(index=False))
    pooled_df.to_csv("outputs/modeling/significance_pooled.csv", index=False)

    print("\nSaved -> outputs/modeling/significance_percountry_hln.csv")
    print("Saved -> outputs/modeling/significance_pooled.csv")


if __name__ == "__main__":
    main()
