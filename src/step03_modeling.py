"""
src/step03_modeling.py — Train/Val/Test Split + Walk-Forward Benchmarking (CLI)
==============================================================================
Ember Energy | IEEE Paper

CLI orchestrator only. Imports from:
    src/modeling/metrics.py       — mape(), rmse(), agg_metrics()
    src/modeling/features.py      — clean_features(), prepare_xy()
    src/modeling/forecasters.py   — naive/linear/holt/arima/ml 1-step
    src/modeling/tune.py          — tune_hyperparameters()
    src/modeling/walk_forward.py  — walk_forward_evaluate()
    src/modeling/plots.py         — all figures

Mirrors exactly: notebooks/02_benchmarking_patched.ipynb

Reads:   outputs/preprocessing/ember_model_ready.csv
         outputs/preprocessing/feature_meta.json
Writes:  outputs/modeling/test_benchmarking.csv
         outputs/modeling/val_benchmarking.csv
         outputs/modeling/best_models.csv
         outputs/modeling/wf_test_predictions.csv
         outputs/modeling/best_hp.json
         outputs/modeling/figures/*.pdf
"""

from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import os
import warnings

import pandas as pd

from src.config import cfg
from src.modeling.metrics import agg_metrics
from src.modeling.plots import (
    plot_mape_heatmaps,
    plot_residuals,
    plot_skill_score,
    plot_split_viz,
    plot_val_vs_test,
    plot_walk_forward_test,
)
from src.modeling.tune import tune_hyperparameters
from src.modeling.walk_forward import walk_forward_evaluate

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGET = cfg.target


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember modeling/benchmarking step")
    p.add_argument("--input_dir",  default="outputs/preprocessing", help="Preprocessing output dir")
    p.add_argument("--output_dir", default="outputs/modeling",      help="Output dir")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir,         exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(args.input_dir, "ember_model_ready.csv"))
    with open(os.path.join(args.input_dir, "feature_meta.json")) as f:
        meta = json.load(f)

    all_features = meta["all_features"]
    train_end    = meta.get("TRAIN_END", cfg.train_end)
    val_end      = meta.get("VAL_END",   cfg.val_end)
    test_end     = meta.get("TEST_END",  cfg.test_end)

    log.info("Dataset shape: %s | Features: %d", df.shape, len(all_features))

    # ── Log split info (guard against inverted range) ─────────────────────────
    val_years  = list(range(train_end + 1, val_end + 1))
    test_years = list(range(val_end + 1,   test_end + 1))

    if val_years:
        log.info("Split: Train≤%d | Val %d-%d | Test %d-%d",
                 train_end, train_end + 1, val_end, val_end + 1, test_end)
    else:
        log.info("Split: Train≤%d | Val=none | Test %d-%d",
                 train_end, val_end + 1, test_end)

    # ── 1. Split visualization ────────────────────────────────────────────────
    plot_split_viz(df, TARGET, train_end, val_end, fig_dir)

    # ── 2. Hyperparameter tuning ──────────────────────────────────────────────
    best_hp, good_features = tune_hyperparameters(
        df, all_features, TARGET, val_end, args.output_dir
    )

    # ── 3. Walk-forward on TEST ───────────────────────────────────────────────
    log.info("── Walk-forward TEST evaluation: years %s", test_years)
    res = walk_forward_evaluate(df, good_features, TARGET, test_years, best_hp)
    log.info("Test walk-forward results: %s", res.shape)

    # ── 4. Walk-forward on VAL (only if val period exists) ────────────────────
    if val_years:
        if val_years:
            log.info("── Walk-forward VAL evaluation: years %s", val_years)
            val_res     = walk_forward_evaluate(df, good_features, TARGET, val_years, best_hp)
            val_metrics = agg_metrics(val_res)
            log.info("Val walk-forward results: %s", val_res.shape)
        else:
            log.warning("Val years empty (train_end=%d >= val_end=%d) — skipping", train_end, val_end)
            val_res     = pd.DataFrame(columns=["Country", "Model", "y_actual", "y_pred"])
            val_metrics = pd.DataFrame(columns=["Country", "Model", "MAE", "RMSE", "MAPE"])
    else:
        log.warning(
            "Val years empty (train_end=%d >= val_end=%d) — skipping val evaluation",
            train_end, val_end,
        )
        val_res     = pd.DataFrame(columns=["Country", "Model", "y_actual", "y_pred"])
        val_metrics = pd.DataFrame(columns=["Country", "Model", "MAE", "RMSE", "MAPE"])

    # ── 5. Aggregate metrics ──────────────────────────────────────────────────
    test_metrics = agg_metrics(res)
    best_test    = (
        test_metrics
        .loc[test_metrics.groupby("Country")["MAPE"].idxmin()]
        .reset_index(drop=True)
    )
    log.info("Best model per country (Test MAPE):\n%s", best_test.to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not val_metrics.empty and not test_metrics.empty:
        plot_mape_heatmaps(test_metrics, val_metrics, fig_dir)
        plot_val_vs_test(val_metrics, test_metrics, fig_dir)
    elif not test_metrics.empty:
        # Only test heatmap — no val data
        import seaborn as sns
        import matplotlib.pyplot as plt
        mape_piv = test_metrics.pivot(index="Country", columns="Model", values="MAPE")
        fig, ax  = plt.subplots(figsize=(13, 4))
        sns.heatmap(mape_piv, cmap="RdYlGn_r", annot=True, fmt=".1f", ax=ax,
                    linewidths=0.4, cbar_kws={"label": "MAPE %"})
        ax.set_title("TEST — MAPE (%) per Country x Model", fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "mdl_fig_mape_heatmap.pdf"), bbox_inches="tight")
        plt.close(fig)

    plot_walk_forward_test(df, res, best_test, TARGET, train_end, val_end, test_end, fig_dir)
    plot_residuals(df, res, best_test, fig_dir)
    plot_skill_score(test_metrics, fig_dir)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    test_metrics.to_csv(os.path.join(args.output_dir, "test_benchmarking.csv"), index=False)
    val_metrics.to_csv(os.path.join(args.output_dir,  "val_benchmarking.csv"),  index=False)
    best_test.to_csv(os.path.join(args.output_dir,    "best_models.csv"),       index=False)
    res.to_csv(os.path.join(args.output_dir,          "wf_test_predictions.csv"), index=False)

    log.info("=== Modeling Complete ===")
    log.info("  test_benchmarking.csv   : %s", test_metrics.shape)
    log.info("  val_benchmarking.csv    : %s", val_metrics.shape)
    log.info("  best_models.csv         : %s", best_test.shape)
    log.info("  wf_test_predictions.csv : %s", res.shape)
    log.info("  Saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
