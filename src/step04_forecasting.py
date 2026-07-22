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


def generate_all_models_forecast(
    df:             pd.DataFrame,
    all_features:   list[str],
    raw_features:   list[str],
    forecast_years: list[int],
    target:         str,
    params_map:     dict,
    ml_cls_map:     dict,
    model_names:    list[str],
) -> pd.DataFrame:
    """
    Generate a POINT forecast (no bootstrap CI) for every model on every
    country — used purely for the "all models on one plot" dashboard
    comparison, to visualise why some models (Ridge in particular) score
    well on 1-step walk-forward MAPE yet diverge badly on the recursive
    6-step 2025-2030 forecast. Skips CI to stay fast (11 models x 7
    countries would be far too slow with 200-iteration bootstrap each).
    """
    from src.forecasting.recursive import point_forecast

    horizon = len(forecast_years)
    rows: list[dict] = []

    for country in COUNTRIES:
        hist_df    = df[df["Area"] == country].sort_values("Year").reset_index(drop=True)
        last_known = float(hist_df[target].values[-1])

        for model_name in model_names:
            params = params_map.get(model_name, {})
            try:
                fc = point_forecast(
                    hist_df, model_name, all_features, raw_features,
                    params, horizon, target, ml_cls_map,
                )
            except Exception as e:
                log.debug("  %s/%s failed: %s", country, model_name, e)
                continue

            # ── Sanity clamp — same philosophy as bootstrap_forecast_ci's
            # 25% fallback, applied here so one recursively-diverging model
            # (e.g. negative demand) can't blow out the shared chart axis
            # that all 11 models are plotted on together.
            fc = np.asarray(fc, dtype=float)
            if last_known != 0:
                for i in range(len(fc)):
                    band = abs(last_known) * (0.40 + 0.10 * i)  # grows with horizon
                    fc[i] = np.clip(fc[i], last_known - band, last_known + band)

            for i, yr in enumerate(forecast_years):
                rows.append({
                    "Country": country,
                    "Model":   model_name,
                    "Year":    yr,
                    "Forecast": round(float(fc[i]), 2),
                })

    return pd.DataFrame(rows)


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
        "ElasticNet":   {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 5000},
        "BayesianRidge": {"max_iter": 500},
        "RandomForest": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "XGBoost":      {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
                         "tree_method": "hist", "verbosity": 0, "random_state": 42},
        "Holt": {}, "DampedHolt": {}, "ARIMA(1,1,1)": {}, "ARIMA_1_1_1": {},
        "SARIMA": {}, "Theta": {},
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

    # ── All-models forecast (point estimate only) — for dashboard comparison ──
    log.info("── Generating point forecast for ALL models (dashboard comparison)…")
    all_model_names = list(params_map.keys())
    all_models_df = generate_all_models_forecast(
        df, all_features, raw_features, forecast_years,
        TARGET, params_map, ml_cls_map, all_model_names,
    )
    all_models_df.to_csv(os.path.join(args.output_dir, "all_models_forecast.csv"), index=False)
    log.info("All-models forecast table: %s", all_models_df.shape)

    # ── Robustness scoring — answers "Missing Evaluation Axis" review ────────
    # MAPE (test_benchmarking.csv) measures accuracy only. This measures
    # recursive extrapolation stability, independently, for every model.
    from src.modeling.robustness import (
        compute_robustness_score, robustness_vs_accuracy, flag_best_model_conflicts,
    )

    last_known = {}
    for country in COUNTRIES:
        sub = df[df["Area"] == country].sort_values("Year")
        if not sub.empty:
            last_known[country] = float(sub[TARGET].values[-1])

    robustness_df = compute_robustness_score(all_models_df, last_known)
    robustness_df.to_csv(os.path.join(args.output_dir, "robustness_summary.csv"), index=False)
    log.info("Robustness summary: %s", robustness_df.shape)

    test_bench_path = os.path.join(args.model_dir, "test_benchmarking.csv")
    if os.path.exists(test_bench_path) and not robustness_df.empty:
        accuracy_df  = pd.read_csv(test_bench_path)
        combined_df  = robustness_vs_accuracy(robustness_df, accuracy_df)
        combined_df.to_csv(os.path.join(args.output_dir, "robustness_vs_accuracy.csv"), index=False)

        conflicts_df = flag_best_model_conflicts(best_df, robustness_df)
        conflicts_df.to_csv(os.path.join(args.output_dir, "best_model_conflicts.csv"), index=False)
        n_conflicts  = int(conflicts_df["conflict"].sum())
        log.info(
            "Robustness: %d/%d best-by-MAPE models flagged WATCH/UNSTABLE",
            n_conflicts, len(conflicts_df),
        )
    else:
        log.warning("test_benchmarking.csv not found or robustness_df empty — "
                     "skipping robustness_vs_accuracy / best_model_conflicts")

    # Export
    fc_df.to_csv(os.path.join(args.output_dir, "demand_forecast_2025_2030.csv"), index=False)
    growth_df.to_csv(os.path.join(args.output_dir, "demand_growth_summary.csv"), index=False)

    log.info("=== Forecasting Complete ===")
    log.info("  demand_forecast_2025_2030.csv : %s", fc_df.shape)
    log.info("  demand_growth_summary.csv     : %s", growth_df.shape)
    log.info("  all_models_forecast.csv       : %s", all_models_df.shape)
    log.info("  robustness_summary.csv        : %s", robustness_df.shape)
    log.info("  Saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
