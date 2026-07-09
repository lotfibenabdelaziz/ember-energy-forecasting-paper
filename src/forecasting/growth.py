"""
src/forecasting/growth.py — CAGR + Growth Summary
====================================================
Ember Energy | IEEE Paper

compute_growth_summary() — shared between 04_forecasting.py and test_forecasting.py
"""

from __future__ import annotations

import pandas as pd

from src.config import cfg

COUNTRIES = cfg.countries


def compute_growth_summary(
    df: pd.DataFrame,
    fc_df: pd.DataFrame,
    best_df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    CAGR + total growth from 2024 (last historical) to 2030 (last forecast).
    Mirrors 04_forecasting.py exactly.
    """
    rows = []
    for country in COUNTRIES:
        hist_sub = df[df["Area"] == country].sort_values("Year")
        if hist_sub.empty:
            continue
        hist_last = hist_sub[target].values[-1]

        fc_first_row = fc_df[(fc_df["Country"] == country) & (fc_df["Year"] == fc_df["Year"].min())]
        fc_last_row  = fc_df[(fc_df["Country"] == country) & (fc_df["Year"] == fc_df["Year"].max())]
        if fc_first_row.empty or fc_last_row.empty:
            continue

        fc_first = fc_first_row["Forecast"].values[0]
        first_year = int(fc_first_row["Year"].values[0])
        fc_last  = fc_last_row["Forecast"].values[0]
        last_year  = int(fc_last_row["Year"].values[0])
        lo_last  = fc_last_row["Lower_90"].values[0] if "Lower_90" in fc_last_row.columns else fc_last
        hi_last  = fc_last_row["Upper_90"].values[0] if "Upper_90" in fc_last_row.columns else fc_last

        n_years = last_year - int(fc_df["Year"].min()) + 1 if fc_df["Year"].min() != fc_df["Year"].max() else 1
        cagr    = ((fc_last / hist_last) ** (1 / n_years) - 1) * 100 if hist_last > 0 else float("nan")
        total_g = (fc_last  - hist_last) / hist_last * 100 if hist_last > 0 else float("nan")

        best_row  = best_df[best_df["Country"] == country]
        test_mape = best_row["MAPE"].values[0] if not best_row.empty else float("nan")

        quality = (
            "Excellent" if test_mape < 3 else
            "Good"      if test_mape < 7 else
            "Moderate"  if test_mape < 15 else
            "Poor"
        )
        model    = fc_last_row["Model"].values[0] if "Model" in fc_last_row.columns else "unknown"

        rows.append({
            "Country":          country,
            f"{last_year - (last_year - first_year)} (TWh)": round(hist_last, 1),
            f"{first_year} Forecast": round(fc_first, 1),
            f"{last_year} Forecast":  round(fc_last, 1),
            f"{last_year} 90% Lo":    round(lo_last, 1),
            f"{last_year} 90% Hi":    round(hi_last, 1),
            "Total Growth %":   round(total_g, 1),
            "CAGR 24-30 %":     round(cagr, 2),
            "Test MAPE %":      round(test_mape, 2),
            "Forecast Quality": quality,
            "Model":            model,
        })

    return pd.DataFrame(rows).sort_values("CAGR 24-30 %", ascending=False)
