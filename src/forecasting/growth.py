"""
src/forecasting/growth.py — CAGR + Growth Summary
====================================================
Ember Energy | IEEE Paper

compute_growth_summary() — shared between 04_forecasting.py and test_forecasting.py
"""

from __future__ import annotations

import pandas as pd

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]


def compute_growth_summary(
    df: pd.DataFrame,
    fc_df: pd.DataFrame,
    best_df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    CAGR + total growth from 2024 (last historical) to 2030 (last forecast).
    Mirrors 04_forecasting.py exactly.

    Raises
    ------
    ValueError : if fc_df doesn't actually contain both 2025 and 2030 —
                 e.g. a misconfigured --forecast_until produced a shorter
                 horizon. Without this check, every country silently gets
                 skipped, rows stays empty, and the final sort_values()
                 fails with a confusing KeyError instead of explaining why.
    """
    available_years = sorted(fc_df["Year"].unique()) if not fc_df.empty else []
    if 2025 not in available_years or 2030 not in available_years:
        raise ValueError(
            f"compute_growth_summary requires forecast data for both 2025 "
            f"and 2030, but fc_df only contains years {available_years}. "
            f"This usually means --forecast_until was set too low (check "
            f"FORECAST_END in .env / docker-compose.yml — it must resolve "
            f"to at least 2030 for this growth summary to be computed)."
        )

    rows = []
    for country in COUNTRIES:
        hist_sub = df[df["Area"] == country].sort_values("Year")
        if hist_sub.empty:
            continue
        hist_last = hist_sub[target].values[-1]

        fc_2025_row = fc_df[(fc_df["Country"] == country) & (fc_df["Year"] == 2025)]
        fc_2030_row = fc_df[(fc_df["Country"] == country) & (fc_df["Year"] == 2030)]
        if fc_2025_row.empty or fc_2030_row.empty:
            continue

        fc_2025 = fc_2025_row["Forecast"].values[0]
        fc_2030 = fc_2030_row["Forecast"].values[0]
        lo_2030 = fc_2030_row["Lower_90"].values[0] if "Lower_90" in fc_2030_row.columns else fc_2030
        hi_2030 = fc_2030_row["Upper_90"].values[0] if "Upper_90" in fc_2030_row.columns else fc_2030

        cagr    = ((fc_2030 / hist_last) ** (1 / 6) - 1) * 100 if hist_last > 0 else float("nan")
        total_g = (fc_2030 - hist_last) / hist_last * 100 if hist_last > 0 else float("nan")

        best_row  = best_df[best_df["Country"] == country]
        test_mape = best_row["MAPE"].values[0] if not best_row.empty else float("nan")

        quality = (
            "Excellent" if test_mape < 3 else
            "Good"      if test_mape < 7 else
            "Moderate"  if test_mape < 15 else
            "Poor"
        )
        model = fc_2030_row["Model"].values[0] if "Model" in fc_2030_row.columns else "unknown"

        rows.append({
            "Country":          country,
            "2024 (TWh)":       round(hist_last, 1),
            "2025 Forecast":    round(fc_2025, 1),
            "2030 Forecast":    round(fc_2030, 1),
            "2030 90% Lo":      round(lo_2030, 1),
            "2030 90% Hi":      round(hi_2030, 1),
            "Total Growth %":   round(total_g, 1),
            "CAGR 24-30 %":     round(cagr, 2),
            "Test MAPE %":      round(test_mape, 2),
            "Forecast Quality": quality,
            "Model":            model,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(
            "compute_growth_summary produced zero rows despite forecast "
            "data existing for 2025/2030 — check that df/best_df actually "
            "contain matching Country values for the expected countries."
        )
    return result.sort_values("CAGR 24-30 %", ascending=False)
