"""
src/04_forecasting.py — Future Demand Forecasting (2025-2030) CLI
=====================================================================
Ember Energy | IEEE Paper

CLI orchestrator only. Imports from:
    src/forecasting/forecasters.py — clean_X, forecast_statistical,
                                       extrapolate_exog, bootstrap_forecast_ci
    src/forecasting/recursive.py   — forecast_ml_recursive, point_forecast,
                                       get_insample_residuals
    src/forecasting/growth.py      — compute_growth_summary
    src/forecasting/plots.py       — all figures

Mirrors exactly: notebooks/033_forecasting_patched.ipynb

Reads:   outputs/preprocessing/ember_model_ready.csv
         outputs/preprocessing/feature_meta.json
         outputs/modeling/best_models.csv
         outputs/modeling/best_hp.json
Writes:  outputs/forecasting/demand_forecast_2025_2030.csv
         outputs/forecasting/demand_growth_summary.csv
         outputs/forecasting/figures/*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings

import numpy as np
import pandas as pd

from src.forecasting.forecasters import (
    bootstrap_forecast_ci, get_ml_cls_map,
)
from src.forecasting.recursive import point_forecast
from src.forecasting.growth import compute_growth_summary
from src.forecasting.plots import (
    plot_forecast_per_country, plot_forecast_overlay, plot_growth_uncertainty,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET    = "Demand"

STAT_MODELS = {"Naive", "Naïve", "LinearTrend", "Holt", "ARIMA(1,1,1)", "ARIMA_1_1_1"}


# ── Generate forecasts for all countries (with fallback chain) ───────────────

def generate_all_forecasts(
    df:             pd.DataFrame,
    best_df:        pd.DataFrame,
    all_features:   list[str],
    raw_features:   list[str],
    forecast_years: list[int],
    target:         str,
    params_map:     dict,
    ml_cls_map:     dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate forecasts for all countries with fallback chain:
    best_model -> Holt -> LinearTrend (last resort).
    """
    horizon = len(forecast_years)
    fc_records: list[dict] = []
    model_used: dict[str, str] = {}

    for country in COUNTRIES:
        hist_df    = df[df["Area"] == country].sort_values("Year").reset_index(drop=True)
        best_row   = best_df[best_df["Country"] == country]
        if best_row.empty:
            log.warning("No best model for %s — skipping", country)
            continue
        best_model = best_row["Model"].values[0]
        params     = params_map.get(best_model, {})

        log.info("  %-10s | %-16s ...", country, best_model)

        fc, lo, hi = None, None, None
        used_model = best_model

        try:
            fc, lo, hi = bootstrap_forecast_ci(
                hist_df, best_model, all_features, raw_features, params,
                horizon, target, ml_cls_map,
            )
        except Exception as e1:
            log.warning("    Best model failed: %s: %s", type(e1).__name__, e1)
            log.info("    Trying Holt fallback...")
            try:
                fc, lo, hi = bootstrap_forecast_ci(
                    hist_df, "Holt", all_features, raw_features, {},
                    horizon, target, ml_cls_map,
                )
                used_model = "Holt (fallback)"
            except Exception as e2:
                log.warning("    Holt failed too: %s", e2)
                from src.forecasting.forecasters import forecast_statistical
                ts  = hist_df[target].values.astype(float)
                fc  = forecast_statistical(ts, "LinearTrend", horizon)
                std = np.std(ts) * 0.10
                lo  = fc - 1.645 * std
                hi  = fc + 1.645 * std
                used_model = "LinearTrend (last resort)"

        model_used[country] = used_model

        for i, yr in enumerate(forecast_years):
            fc_records.append({
                "Country":  country,
                "Year":     yr,
                "Forecast": round(float(fc[i]), 2),
                "Lower_90": round(float(lo[i]), 2),
                "Upper_90": round(float(hi[i]), 2),
                "Model":    used_model,
            })

    fc_df = pd.DataFrame(fc_records)

    for c, m in model_used.items():
        flag = "OK" if "fallback" not in m and "last resort" not in m else "WARN"
        log.info("  [%s] %-10s: %s", flag, c, m)

    return fc_df, model_used


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember forecasting step")
    p.add_argument("--pre_dir",        default="outputs/preprocessing", help="Preprocessing dir")
    p.add_argument("--model_dir",      default="outputs/modeling",      help="Modeling dir")
    p.add_argument("--output_dir",     default="outputs/forecasting",   help="Output dir")
    p.add_argument("--forecast_until", type=int, default=2030,          help="Last forecast year")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir,         exist_ok=True)

    forecast_years = list(range(2025, args.forecast_until + 1))

    # Load
    df      = pd.read_csv(os.path.join(args.pre_dir, "ember_model_ready.csv"))
    best_df = pd.read_csv(os.path.join(args.model_dir, "best_models.csv"))
    with open(os.path.join(args.pre_dir, "feature_meta.json")) as f:
        meta = json.load(f)

    all_features = meta["all_features"]
    # Fallback: raw_features may be missing from stale feature_meta.json
    # Re-run: python pipeline.py --invalidate preprocessing
    raw_features = meta.get("raw_features", meta.get("FEATURES", []))

    log.info("Data: %s", df.shape)
    log.info("Best models:\n%s", best_df[["Country", "Model", "MAPE"]].to_string())

    # Load tuned hyperparameters
    params_map = {
        "Ridge":        {"alpha": 1.0},
        "RandomForest": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "XGBoost":      {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
                         "tree_method": "hist", "verbosity": 0, "random_state": 42},
        "Holt": {}, "ARIMA(1,1,1)": {}, "ARIMA_1_1_1": {},
        "LinearTrend": {}, "Naive": {}, "Naïve": {},
    }
    best_hp_path = os.path.join(args.model_dir, "best_hp.json")
    if os.path.exists(best_hp_path):
        with open(best_hp_path) as f:
            best_hp = json.load(f)
        for k, v in best_hp.items():
            params_map[k] = v
        log.info("Loaded tuned hyperparameters from best_hp.json")
    else:
        log.warning("best_hp.json not found — using default hyperparameters")

    ml_cls_map = get_ml_cls_map()

    # Generate forecasts
    log.info("── Generating forecasts for %d countries x %d years…",
             len(COUNTRIES), len(forecast_years))
    fc_df, model_used = generate_all_forecasts(
        df, best_df, all_features, raw_features, forecast_years,
        TARGET, params_map, ml_cls_map,
    )
    log.info("Forecast table: %s", fc_df.shape)

    # Plots
    plot_forecast_per_country(df, fc_df, TARGET, fig_dir)
    plot_forecast_overlay(df, fc_df, TARGET, fig_dir)

    # Growth summary
    growth_df = compute_growth_summary(df, fc_df, best_df, TARGET)
    log.info("Growth summary:\n%s", growth_df.to_string())

    plot_growth_uncertainty(fc_df, growth_df, fig_dir)

    # Export
    fc_df.to_csv(os.path.join(args.output_dir, "demand_forecast_2025_2030.csv"), index=False)
    growth_df.to_csv(os.path.join(args.output_dir, "demand_growth_summary.csv"), index=False)

    log.info("=== Forecasting Complete ===")
    log.info("  demand_forecast_2025_2030.csv : %s", fc_df.shape)
    log.info("  demand_growth_summary.csv     : %s", growth_df.shape)
    log.info("  Saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
