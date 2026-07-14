"""
src/plot.py — All Figures for Deep Learning Module
===================================================
Ember Energy | IEEE Paper — Deep Learning module

Mirrors notebook 04_deeplearning_enhanced_patched.ipynb figures exactly:
    - Loss curves per model per country
    - Walk-forward predictions vs actual
    - MAPE heatmap (Country x Model)
    - DL vs Classical comparison bar chart
    - Forecast 2025-2030 with confidence intervals
    - Feature importance bar chart
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET = "Demand"

PALETTE = {
    "Tunisia": "#e63946",
    "Austria": "#2196F3",
    "Germany": "#4CAF50",
    "Egypt": "#9C27B0",
    "Canada": "#00BCD4",
    "France": "#797148",
    "Kuwait": "#4C4879",
}

MODEL_COLORS = {
    "MLP": "#3b9eff",
    "TCN": "#00d4a0",
    "N-BEATS": "#ff9800",
    "TFT": "#e91e63",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def savefig(fig: plt.Figure, path: str) -> None:
    """Save figure to path, creating directories as needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        log.info("Saved → %s", path)
    except Exception as e:
        plt.close(fig)
        log.error("Failed to save %s: %s", path, e)


# ── Loss curves ───────────────────────────────────────────────────────────────


def plot_loss_curves(
    loss_curves: dict,
    countries: list[str],
    model_names: list[str],
    fig_dir: str,
) -> None:
    """
    Grid of train/val loss curves: rows=countries, cols=models.
    Mirrors notebook plot_loss_curves() exactly.
    """
    n_c = len(countries)
    n_m = len(model_names)
    fig, axes = plt.subplots(n_c, n_m, figsize=(4 * n_m, 3 * n_c), sharex=False)

    if n_c == 1:
        axes = axes[np.newaxis, :]
    if n_m == 1:
        axes = axes[:, np.newaxis]

    for i, country in enumerate(countries):
        for j, model_name in enumerate(model_names):
            ax = axes[i][j]
            key = (country, model_name)
            if key not in loss_curves:
                ax.set_visible(False)
                continue
            tr = loss_curves[key]["train"]
            val = loss_curves[key]["val"]
            ax.plot(tr, label="Train", color="#3b9eff", lw=1.2)
            ax.plot(val, label="Val", color="#ff9800", lw=1.2, ls="--")
            ax.set_title(f"{country} — {model_name}", fontsize=8, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=7)
            ax.set_ylabel("MSE", fontsize=7)
            ax.tick_params(labelsize=6)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, frameon=False)

    fig.suptitle("Training & Validation Loss Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "dl_loss_curves.pdf"))


# ── Walk-forward predictions ──────────────────────────────────────────────────


def plot_walk_forward(
    dl_res: pd.DataFrame,
    countries: list[str],
    fig_dir: str,
) -> None:
    """
    Actual vs predicted for each country (best DL model).
    Mirrors notebook walk-forward plot exactly.
    """
    model_names = dl_res["Model"].unique().tolist()
    n_cols = min(4, len(countries))
    n_rows = (len(countries) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, country in enumerate(countries):
        ax = axes[i]
        sub_actual = (
            dl_res[dl_res["Country"] == country].drop_duplicates("Year").sort_values("Year")
        )
        ax.plot(
            sub_actual["Year"],
            sub_actual["y_actual"],
            "k-o",
            lw=2,
            ms=5,
            label="Actual",
            zorder=5,
        )

        for model_name in model_names:
            sub = dl_res[
                (dl_res["Country"] == country) & (dl_res["Model"] == model_name)
            ].sort_values("Year")
            if sub.empty:
                continue
            ax.plot(
                sub["Year"],
                sub["y_pred"],
                color=MODEL_COLORS.get(model_name, "grey"),
                lw=1.5,
                ls="--",
                marker="s",
                ms=4,
                label=model_name,
            )

        ax.set_title(country, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Demand (TWh)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))

    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    for j in range(len(countries), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Walk-Forward Evaluation — DL Models vs Actual", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "dl_walk_forward.pdf"))


# ── MAPE heatmap ──────────────────────────────────────────────────────────────


def plot_mape_heatmap(dl_metrics: pd.DataFrame, fig_dir: str) -> None:
    """
    MAPE heatmap: rows=countries, cols=models.
    Mirrors notebook Section 8 heatmap exactly.
    """
    pivot = dl_metrics.pivot_table(index="Country", columns="Model", values="MAPE")
    pivot = pivot.reindex(COUNTRIES).dropna(how="all")

    fig, ax = plt.subplots(figsize=(len(pivot.columns) * 1.6 + 2, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd_r",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "MAPE (%)"},
    )
    ax.set_title("DL Model MAPE (%) — Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "dl_mape_heatmap.pdf"))


# ── DL vs Classical comparison ────────────────────────────────────────────────


def plot_dl_vs_classical(compare_df: pd.DataFrame, fig_dir: str) -> None:
    """
    Side-by-side MAPE: DL best vs Classical best per country.
    Mirrors notebook Section 9 bar chart exactly.
    """
    x = np.arange(len(compare_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(
        x - width / 2,
        compare_df["Classic_MAPE"],
        width,
        label="Best Classical",
        color="#3b9eff",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2, compare_df["DL_MAPE"], width, label="Best DL", color="#00d4a0", alpha=0.85
    )

    ax.set_xticks(x)
    ax.set_xticklabels(compare_df["Country"], rotation=15, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Best Classical vs Best DL Model — Test MAPE (%)", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "dl_vs_classical.pdf"))


# ── Forecast 2025-2030 ────────────────────────────────────────────────────────


def plot_dl_forecast(
    df_hist: pd.DataFrame,
    df_forecast: pd.DataFrame,
    countries: list[str],
    fig_dir: str,
) -> None:
    """
    Historical demand + DL forecast 2025-2030 per country.
    Mirrors notebook Section 10 forecast plot exactly.
    """
    n_cols = min(4, len(countries))
    n_rows = (len(countries) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, country in enumerate(countries):
        ax = axes[i]
        hist = df_hist[df_hist["Area"] == country].sort_values("Year")
        fc = df_forecast[df_forecast["Country"] == country].sort_values("Year")

        ax.plot(hist["Year"], hist[TARGET], "k-o", lw=2, ms=4, label="Historical")
        ax.plot(
            fc["Year"],
            fc["Forecast"],
            color=PALETTE.get(country, "#3b9eff"),
            lw=2.5,
            ls="--",
            marker="^",
            ms=6,
            label="DL Forecast",
        )

        if "Lower_90" in fc.columns and "Upper_90" in fc.columns:
            ax.fill_between(
                fc["Year"],
                fc["Lower_90"],
                fc["Upper_90"],
                alpha=0.15,
                color=PALETTE.get(country, "#3b9eff"),
                label="90% CI",
            )

        ax.axvline(2024.5, color="grey", ls=":", lw=1)
        ax.set_title(country, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Demand (TWh)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))

    axes[0].legend(fontsize=7, frameon=False)
    for j in range(len(countries), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Electricity Demand Forecast 2025–2030 (DL Models)", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "dl_forecast_2025_2030.pdf"))


# ── Feature importance ────────────────────────────────────────────────────────


def plot_feature_importance(
    feature_cols: list[str],
    importances: np.ndarray,
    fig_dir: str,
    top_n: int = 20,
) -> None:
    """
    Horizontal bar chart of RF feature importances.
    Mirrors notebook Check 5 importance plot.
    """
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    fi[::-1].plot.barh(ax=ax, color="#3b9eff", alpha=0.85)
    ax.set_title(f"Top {top_n} Feature Importances (RF)", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "dl_feature_importance.pdf"))


# ── Model timing ──────────────────────────────────────────────────────────────


def plot_model_timing(model_timing: dict, fig_dir: str) -> None:
    """Bar chart of training time per model."""
    names = list(model_timing.keys())
    times = [model_timing[k] / 60 for k in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, times, color=[MODEL_COLORS.get(n, "#999") for n in names], alpha=0.85)
    for bar, t in zip(bars, times, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{t:.1f} min",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Time (minutes)")
    ax.set_title("Model Training Time", fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "dl_model_timing.pdf"))
