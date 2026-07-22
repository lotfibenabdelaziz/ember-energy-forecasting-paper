"""
src/modeling/robustness.py — Recursive Extrapolation Robustness Metric
=========================================================================
Ember Energy | IEEE Paper

Directly answers the reviewer comment:
    "Our evaluation protocol (Train/Val/Test) measures accuracy only.
     It does not measure extrapolation robustness — how a model behaves
     once recursive inputs leave the training distribution. This should
     be reported as a second, independent metric ... not inferred after
     the fact from a country that happened to fail."

MAPE (accuracy) and this module (robustness) are deliberately independent
axes. A model can score well on one and poorly on the other — that is
the entire point of computing both.

Input:  outputs/forecasting/all_models_forecast.csv
        (Country, Model, Year, Forecast — every classical model's point
        forecast for every country, unclamped by CI logic so instability
        is visible)
Output: outputs/modeling/robustness_summary.csv
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Thresholds — calibrated against real-world demand shocks. Historically,
# even severe shocks (COVID-19, 1973 oil crisis) rarely exceed 8-10% YoY.
# A model exceeding these bounds is extrapolating into physically
# implausible territory, not modeling a plausible future.
WATCH_THRESHOLD    = 20.0   # % max deviation from last known value
UNSTABLE_THRESHOLD = 40.0   # % max deviation from last known value


def compute_robustness_score(
    forecast_df: pd.DataFrame,
    last_known: dict[str, float],
) -> pd.DataFrame:
    """
    Score recursive forecast stability per (Country, Model) — independent
    of accuracy. Does not require ground truth (there is none for 2025-2030
    yet); it measures internal plausibility of the trajectory itself.

    Parameters
    ----------
    forecast_df : long-format [Country, Model, Year, Forecast]
                  (typically outputs/forecasting/all_models_forecast.csv)
    last_known  : {country: last historical Demand value} — the anchor
                  point every forecast trajectory is judged against

    Returns
    -------
    DataFrame: [Country, Model, max_deviation_pct, max_yoy_change_pct,
                trajectory_std, monotonic_direction_flips, robustness_flag]
    """
    rows = []

    for (country, model), grp in forecast_df.groupby(["Country", "Model"]):
        grp  = grp.sort_values("Year")
        vals = grp["Forecast"].values.astype(float)
        base = last_known.get(country)

        if base is None or base == 0 or len(vals) < 2:
            continue

        # Max single-point deviation from the last known real value —
        # the core "did this go somewhere implausible" signal.
        max_dev = float(np.max(np.abs(vals - base) / abs(base)) * 100)

        # Max year-over-year jump within the forecast itself — catches
        # trajectories that are locally unstable even if the endpoint
        # looks reasonable (e.g. a V-shaped collapse-and-recover).
        yoy     = np.diff(vals) / np.abs(vals[:-1] + 1e-9)
        max_yoy = float(np.max(np.abs(yoy)) * 100) if len(yoy) else 0.0

        # Overall volatility of the 6-year path.
        traj_std = float(np.std(vals))

        # Direction flips — a plausible trend rarely reverses direction
        # more than once; repeated flips indicate an unstable recursive
        # feedback loop rather than a genuine trend change.
        diffs   = np.diff(vals)
        signs   = np.sign(diffs)
        signs   = signs[signs != 0]
        flips   = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0

        if max_dev > UNSTABLE_THRESHOLD:
            flag = "UNSTABLE"
        elif max_dev > WATCH_THRESHOLD:
            flag = "WATCH"
        else:
            flag = "STABLE"

        rows.append({
            "Country":                   country,
            "Model":                     model,
            "max_deviation_pct":         round(max_dev, 1),
            "max_yoy_change_pct":        round(max_yoy, 1),
            "trajectory_std":            round(traj_std, 1),
            "direction_flips":           flips,
            "robustness_flag":           flag,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        log.warning("compute_robustness_score: no rows produced — check inputs")
        return result

    return result.sort_values(["Country", "max_deviation_pct"]).reset_index(drop=True)


def robustness_vs_accuracy(
    robustness_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge robustness scores with MAPE — the single table that directly
    answers the reviewer's comment. Lets a reader see, per model per
    country, whether a low MAPE actually correlates with a stable
    forecast, or whether the two axes disagree (the interesting case).

    Parameters
    ----------
    robustness_df : output of compute_robustness_score()
    accuracy_df   : test_benchmarking.csv — [Country, Model, MAPE, ...]

    Returns
    -------
    DataFrame: [Country, Model, MAPE, max_deviation_pct, robustness_flag,
                accuracy_rank, robustness_rank, axes_agree]
    """
    merged = robustness_df.merge(
        accuracy_df[["Country", "Model", "MAPE"]],
        on=["Country", "Model"],
        how="left",
    )

    merged["accuracy_rank"]    = merged.groupby("Country")["MAPE"].rank(method="min")
    merged["robustness_rank"]  = merged.groupby("Country")["max_deviation_pct"].rank(method="min")

    # "Axes agree" if a model ranks similarly on both — a large gap between
    # ranks is exactly the accurate-but-unstable (or the reverse) case that
    # motivated this whole module.
    merged["rank_gap"]   = (merged["accuracy_rank"] - merged["robustness_rank"]).abs()
    merged["axes_agree"] = merged["rank_gap"] <= 2

    cols = [
        "Country", "Model", "MAPE", "max_deviation_pct", "max_yoy_change_pct",
        "robustness_flag", "accuracy_rank", "robustness_rank", "axes_agree",
    ]
    return merged[cols].sort_values(["Country", "accuracy_rank"]).reset_index(drop=True)


def flag_best_model_conflicts(
    best_models_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each country, check whether the model selected as "best" on MAPE
    alone (best_models.csv) is flagged UNSTABLE or WATCH on the robustness
    axis. This is the exact failure mode the reviewer describes — a model
    chosen purely on accuracy, without knowing it will misbehave once
    recursion pushes it out of distribution.

    Returns
    -------
    DataFrame: [Country, Best_Model, MAPE, robustness_flag,
                max_deviation_pct, conflict]
    """
    merged = best_models_df.merge(
        robustness_df[["Country", "Model", "robustness_flag", "max_deviation_pct"]],
        left_on=["Country", "Model"],
        right_on=["Country", "Model"],
        how="left",
    )
    merged["conflict"] = merged["robustness_flag"].isin(["WATCH", "UNSTABLE"])

    n_conflicts = int(merged["conflict"].sum())
    if n_conflicts:
        log.warning(
            "%d/%d 'best' models (by MAPE) flagged WATCH/UNSTABLE on robustness — "
            "accuracy alone would have hidden this",
            n_conflicts, len(merged),
        )

    return merged[[
        "Country", "Model", "MAPE", "robustness_flag", "max_deviation_pct", "conflict",
    ]].rename(columns={"Model": "Best_Model"})
