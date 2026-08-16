"""
src/05_deeplearning.py — Deep Learning CLI Orchestrator
=========================================================
Ember Energy | IEEE Paper — Deep Learning module

CLI entry point. Imports from:
    src/dataset.py   — data loading, sliding-window dataset
    src/model.py     — MLP, TCN, N-BEATS, TFT architectures
    src/train.py     — training loop + MLflow logging
    src/evaluate.py  — metrics + walk-forward evaluation
    src/plot.py      — all figures
    src/forecast.py  — recursive inference + bootstrap CI + growth summary

Writes:
    outputs/deeplearning/dl_benchmarking.csv
    outputs/deeplearning/dl_best_models.csv
    outputs/deeplearning/dl_wf_predictions.csv
    outputs/deeplearning/dl_forecast_2025_2030.csv
    outputs/deeplearning/dl_growth_summary.csv
    outputs/deeplearning/figures/*.pdf

Usage:
    python src/05_deeplearning.py --pre_dir outputs/preprocessing --output_dir outputs/deeplearning
    python src/05_deeplearning.py --quick     # 1 country, reduced epochs — smoke test
"""

from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd
import torch

from src.dataset import (
    BATCH_SIZE,
    COUNTRIES,
    EPOCHS,
    FORECAST_YEARS,
    LR,
    PATIENCE,
    SEED,
    SEQ_LEN,
    TARGET,
    TEST_END,
    TRAIN_END,
    VAL_END,
    get_all_features,
    load_model_ready,
    make_loaders,
)
from src.evaluate import (
    compare_with_classical,
    compute_benchmarking,
    compute_best_models,
    run_all_walk_forward,
)
from src.forecast import (
    bootstrap_ci,
    growth_summary,
)
from src.model import build_model_registry
from src.plot import (
    plot_dl_forecast,
    plot_dl_vs_classical,
    plot_loss_curves,
    plot_mape_heatmap,
    plot_model_timing,
    plot_walk_forward,
)
from src.train import log_all_dl_to_mlflow, train_model

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember deep learning step")
    p.add_argument("--pre_dir",    default="outputs/preprocessing", help="Preprocessing output dir")
    p.add_argument("--output_dir", default="outputs/deeplearning",  help="Output dir")
    p.add_argument("--model_dir",  default="outputs/modeling",      help="Classical model dir (for comparison)")
    p.add_argument("--epochs",     type=int, default=EPOCHS,        help="Max training epochs")
    p.add_argument("--patience",   type=int, default=PATIENCE,      help="Early stopping patience")
    p.add_argument("--seq_len",    type=int, default=SEQ_LEN,       help="Look-back window size")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE,    help="Training batch size")
    p.add_argument("--lr",         type=float, default=LR,          help="Learning rate")
    p.add_argument(
        "--quick", action="store_true",
        help="Smoke test: 1 country, 100 epochs, patience=10",
    )
    p.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow logging",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir,         exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 60)
    log.info("  EMBER DEEP LEARNING PIPELINE")
    log.info("  Device         : %s", device)
    log.info("  seq_len        : %d", args.seq_len)
    log.info("  epochs         : %d", args.epochs)
    log.info("  patience       : %d", args.patience)
    log.info("=" * 60)

    # ── Quick mode overrides ──────────────────────────────────────────────────
    countries = COUNTRIES
    epochs    = args.epochs
    patience  = args.patience
    if args.quick:
        countries = COUNTRIES[:1]
        epochs    = min(args.epochs, 100)
        patience  = min(args.patience, 10)
        log.warning("QUICK MODE: country=%s, epochs=%d, patience=%d",
                    countries, epochs, patience)

    # ── Load data ─────────────────────────────────────────────────────────────
    df, meta = load_model_ready(args.pre_dir)
    feature_cols = get_all_features(meta)
    log.info("Loaded model-ready data: %s | %d features", df.shape, len(feature_cols))

    n_features    = len(feature_cols)
    model_registry = build_model_registry(args.seq_len, n_features)
    model_names    = list(model_registry.keys())
    log.info("Model registry: %s", model_names)

    # ── Walk-forward evaluation (all models x all countries) ─────────────────
    log.info("── Starting walk-forward evaluation…")
    dl_res, loss_curves, model_timing = run_all_walk_forward(
        df, countries, model_registry, feature_cols, TARGET,
        args.seq_len, TRAIN_END, VAL_END, TEST_END,
        epochs, patience, args.lr, args.batch_size, device,
    )
    dl_res.to_csv(os.path.join(args.output_dir, "dl_wf_predictions.csv"), index=False)
    log.info("Walk-forward predictions saved: %s", dl_res.shape)

    # ── Benchmarking ──────────────────────────────────────────────────────────
    dl_metrics = compute_benchmarking(dl_res)
    dl_metrics.to_csv(os.path.join(args.output_dir, "dl_benchmarking.csv"), index=False)
    log.info("DL benchmarking:\n%s", dl_metrics.to_string())

    best_dl = compute_best_models(dl_metrics)
    best_dl.to_csv(os.path.join(args.output_dir, "dl_best_models.csv"), index=False)
    log.info("Best DL models per country:\n%s", best_dl.to_string())

    # ── Plots: loss curves, walk-forward, heatmap ─────────────────────────────
    plot_loss_curves(loss_curves, countries, model_names, fig_dir)
    plot_walk_forward(dl_res, countries, fig_dir)
    plot_mape_heatmap(dl_metrics, fig_dir)
    plot_model_timing(model_timing, fig_dir)

    # ── Compare with classical models (if available) ─────────────────────────
    classical_path = os.path.join(args.model_dir, "test_benchmarking.csv")
    if os.path.exists(classical_path):
        classical_bm = pd.read_csv(classical_path)
        compare_df   = compare_with_classical(best_dl, classical_bm)
        compare_df.to_csv(os.path.join(args.output_dir, "dl_vs_classical.csv"), index=False)
        plot_dl_vs_classical(compare_df, fig_dir)
        log.info("DL vs Classical comparison:\n%s", compare_df.to_string())
    else:
        log.warning("Classical benchmark not found at %s — skipping comparison", classical_path)

    # ── Refit best model per country on FULL data + forecast 2025-2030 ───────
    log.info("── Refitting best models on full data for forecasting…")
    all_forecasts: list[pd.DataFrame] = []

    for country in countries:
        best_row   = best_dl[best_dl["Country"] == country]
        if best_row.empty:
            log.warning("No best model for %s — skipping forecast", country)
            continue
        model_name = best_row.iloc[0]["Model"]
        reg        = model_registry[model_name]

        sub = df[df["Area"] == country]
        loader_tr, loader_val, scaler_X, scaler_y = make_loaders(
            sub, feature_cols, TARGET, args.seq_len,
            train_end=TEST_END, val_end=TEST_END,
            batch_size=args.batch_size,
        )

        model = reg["cls"](**reg["kwargs"]).to(device)
        train_model(
            model, loader_tr, loader_val,
            epochs=epochs, patience=patience, lr=args.lr, device=device,
            desc=f"{country}-{model_name}-refit",
        )

        # Residuals from walk-forward test results (for bootstrap CI)
        country_wf = dl_res[
            (dl_res["Country"] == country) & (dl_res["Model"] == model_name)
        ]
        residuals = country_wf["y_actual"].values - country_wf["y_pred"].values

        fc = bootstrap_ci(
            model, sub, feature_cols, TARGET, args.seq_len,
            scaler_X, scaler_y, FORECAST_YEARS, device,
            residuals=residuals, n_boot=200, ci=0.90, seed=SEED,
        )
        fc["Country"] = country
        fc["Model"]   = model_name
        all_forecasts.append(fc)

        log.info("  %-10s (%s) forecast 2030 = %.2f TWh [%.2f, %.2f]",
                 country, model_name, fc["Forecast"].iloc[-1],
                 fc["Lower_90"].iloc[-1], fc["Upper_90"].iloc[-1])

    df_forecast = (
        pd.concat(all_forecasts, ignore_index=True)
        if all_forecasts
        else pd.DataFrame(columns=["Year", "Forecast", "Lower_90", "Upper_90", "Country", "Model"])
    )
    df_forecast = df_forecast[["Country", "Model", "Year", "Forecast", "Lower_90", "Upper_90"]]
    df_forecast.to_csv(
        os.path.join(args.output_dir, "dl_forecast_2025_2030.csv"), index=False
    )
    log.info("DL forecast saved: %s", df_forecast.shape)

    plot_dl_forecast(df, df_forecast, countries, fig_dir)

    # ── Growth summary ────────────────────────────────────────────────────────
    growth_df = growth_summary(df_forecast, df, countries, base_year=2024, target_year=2030)
    growth_df.to_csv(os.path.join(args.output_dir, "dl_growth_summary.csv"), index=False)
    log.info("Growth summary:\n%s", growth_df.to_string())

    # ── MLflow logging ────────────────────────────────────────────────────────
    if not args.no_mlflow:
        try:
            log_all_dl_to_mlflow(
                dl_metrics,
                params={
                    "seq_len": args.seq_len, "epochs": epochs,
                    "patience": patience, "lr": args.lr,
                    "batch_size": args.batch_size,
                },
            )
            log.info("MLflow logging complete.")
        except Exception as e:
            log.warning("MLflow logging failed (non-fatal): %s", e)

    log.info("=" * 60)
    log.info("  DEEP LEARNING PIPELINE COMPLETE")
    log.info("  dl_benchmarking.csv         : %s", dl_metrics.shape)
    log.info("  dl_best_models.csv          : %s", best_dl.shape)
    log.info("  dl_forecast_2025_2030.csv   : %s", df_forecast.shape)
    log.info("  Saved → %s", args.output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
