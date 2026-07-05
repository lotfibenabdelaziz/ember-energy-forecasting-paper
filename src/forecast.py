"""
src/forecast.py — Recursive Inference, Bootstrap CI, Growth Summary
======================================================================
Ember Energy | IEEE Paper — Deep Learning module

Mirrors notebook 04_deeplearning_enhanced_patched.ipynb Section 10:
    - recursive_forecast()    : multi-year ahead, feature-aware recursion
    - bootstrap_ci()          : residual-bootstrap confidence intervals
    - growth_summary()        : CAGR + total growth 2024->2030
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from src.dataset import clean_arr

log = logging.getLogger(__name__)


# ── Recursive multi-year forecast ─────────────────────────────────────────────


def recursive_forecast(
    model,
    df_country: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    seq_len: int,
    scaler_X,
    scaler_y,
    forecast_years: list[int],
    device: torch.device,
) -> pd.DataFrame:
    """
    Recursively forecast `forecast_years` ahead.

    At each step:
      1. Take last seq_len rows as input window
      2. Predict next year's Demand
      3. Append predicted Demand + carried-forward features as a new row
      4. Repeat

    Non-target features are carried forward using their last observed
    value (naive persistence) since true future exogenous values
    are unknown — mirrors notebook recursive_forecast() exactly.

    Returns
    -------
    DataFrame with columns [Year, Forecast]
    """
    model.eval()
    history = df_country.sort_values("Year").reset_index(drop=True).copy()
    predictions: list[dict] = []

    for year in forecast_years:
        window_raw = clean_arr(history.tail(seq_len)[feature_cols].values)
        window_sc = scaler_X.transform(window_raw)

        with torch.no_grad():
            x = torch.tensor(window_sc, dtype=torch.float32).unsqueeze(0).to(device)
            y_sc = model(x).squeeze().cpu().item()

        y_pred = float(scaler_y.inverse_transform([[y_sc]])[0][0])
        predictions.append({"Year": year, "Forecast": y_pred})

        # Build next row: carry forward last known feature values,
        # update Demand with the new prediction
        new_row = history.iloc[-1].copy()
        new_row["Year"] = year
        new_row[target] = y_pred
        history = pd.concat([history, new_row.to_frame().T], ignore_index=True)

    return pd.DataFrame(predictions)


def recursive_forecast_all_countries(
    model_registry_fitted: dict,
    df: pd.DataFrame,
    countries: list[str],
    feature_cols: list[str],
    target: str,
    seq_len: int,
    forecast_years: list[int],
    device: torch.device,
) -> pd.DataFrame:
    """
    Run recursive_forecast() for each (country, best_model) pair.

    model_registry_fitted: {country: {"model": fitted_model, "name": str,
                                       "scaler_X": ..., "scaler_y": ...}}

    Returns long-format DataFrame: [Country, Model, Year, Forecast]
    """
    all_forecasts: list[pd.DataFrame] = []

    for country in countries:
        if country not in model_registry_fitted:
            log.warning("No fitted model for %s — skipping forecast", country)
            continue

        entry = model_registry_fitted[country]
        sub = df[df["Area"] == country]

        fc = recursive_forecast(
            entry["model"],
            sub,
            feature_cols,
            target,
            seq_len,
            entry["scaler_X"],
            entry["scaler_y"],
            forecast_years,
            device,
        )
        fc["Country"] = country
        fc["Model"] = entry["name"]
        all_forecasts.append(fc)

    if not all_forecasts:
        return pd.DataFrame(columns=["Country", "Model", "Year", "Forecast"])

    return pd.concat(all_forecasts, ignore_index=True)[["Country", "Model", "Year", "Forecast"]]


# ── Bootstrap confidence intervals ────────────────────────────────────────────


def bootstrap_ci(
    model,
    df_country: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    seq_len: int,
    scaler_X,
    scaler_y,
    forecast_years: list[int],
    device: torch.device,
    residuals: np.ndarray,
    n_boot: int = 200,
    ci: float = 0.90,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Residual-bootstrap confidence intervals around the point forecast.

    For each bootstrap iteration:
      1. Resample residuals with replacement
      2. Add resampled residual to the point forecast for each year
      3. Collect all bootstrap forecasts
      4. Take percentiles for lower/upper CI bounds

    Mirrors notebook bootstrap_ci() exactly.

    Returns
    -------
    DataFrame with columns [Year, Forecast, Lower_90, Upper_90]
    """
    rng = np.random.default_rng(seed)

    # Point forecast (no noise)
    point_fc = recursive_forecast(
        model,
        df_country,
        feature_cols,
        target,
        seq_len,
        scaler_X,
        scaler_y,
        forecast_years,
        device,
    )

    if len(residuals) == 0:
        # No residuals available — return point forecast with no CI band
        point_fc["Lower_90"] = point_fc["Forecast"]
        point_fc["Upper_90"] = point_fc["Forecast"]
        return point_fc

    n_years = len(forecast_years)
    boot_matrix = np.zeros((n_boot, n_years))

    for b in range(n_boot):
        sampled_resid = rng.choice(residuals, size=n_years, replace=True)
        # Accumulate residual noise (uncertainty grows with horizon)
        cum_resid = np.cumsum(sampled_resid) / np.arange(1, n_years + 1)
        boot_matrix[b] = point_fc["Forecast"].values + cum_resid

    alpha = 1 - ci
    lower = np.percentile(boot_matrix, alpha / 2 * 100, axis=0)
    upper = np.percentile(boot_matrix, (1 - alpha / 2) * 100, axis=0)

    point_fc["Lower_90"] = lower
    point_fc["Upper_90"] = upper
    # Ensure point forecast stays within bounds
    point_fc["Lower_90"] = np.minimum(point_fc["Lower_90"], point_fc["Forecast"])
    point_fc["Upper_90"] = np.maximum(point_fc["Upper_90"], point_fc["Forecast"])

    return point_fc


def compute_residuals(walk_forward_results: list[dict]) -> np.ndarray:
    """Extract (actual - predicted) residuals from walk-forward test results."""
    if not walk_forward_results:
        return np.array([])
    return np.array([r["y_actual"] - r["y_pred"] for r in walk_forward_results])


# ── Growth summary ─────────────────────────────────────────────────────────────


def growth_summary(
    df_forecast: pd.DataFrame,
    df_hist: pd.DataFrame,
    countries: list[str],
    base_year: int = 2024,
    target_year: int = 2030,
) -> pd.DataFrame:
    """
    CAGR + total growth from base_year (last historical) to target_year (forecast).

    CAGR = (end_value / start_value)^(1 / n_years) - 1

    Mirrors notebook growth_summary() exactly.

    Returns
    -------
    DataFrame: [Country, Model, base_year_TWh, target_year_TWh,
                Total_Growth_%, CAGR_%]
    """
    rows = []
    n_years = target_year - base_year

    for country in countries:
        hist_val = df_hist[(df_hist["Area"] == country) & (df_hist["Year"] == base_year)]["Demand"]
        if hist_val.empty:
            continue
        base_val = float(hist_val.values[0])

        fc_sub = df_forecast[
            (df_forecast["Country"] == country) & (df_forecast["Year"] == target_year)
        ]
        if fc_sub.empty:
            continue
        target_val = float(fc_sub["Forecast"].values[0])
        model_name = fc_sub["Model"].values[0] if "Model" in fc_sub.columns else "unknown"

        total_growth = (target_val - base_val) / base_val * 100
        cagr = (
            ((target_val / base_val) ** (1 / n_years) - 1) * 100
            if base_val > 0 and n_years > 0
            else float("nan")
        )

        rows.append(
            {
                "Country": country,
                "Model": model_name,
                f"{base_year}_TWh": round(base_val, 2),
                f"{target_year}_TWh": round(target_val, 2),
                "Total_Growth_%": round(total_growth, 2),
                "CAGR_%": round(cagr, 2),
            }
        )

    return pd.DataFrame(rows)
