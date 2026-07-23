"""
src/forecasting/plots.py — All Figures for Forecasting Step
================================================================
Ember Energy | IEEE Paper

Mirrors notebooks/033_forecasting_patched.ipynb figures exactly.
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
    }
)
sns.set_theme(style="whitegrid", palette="tab10")


def savefig(fig: plt.Figure, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved → %s", path)
    except Exception as e:
        plt.close(fig)
        log.error("Failed to save %s: %s", path, e)


def plot_forecast_per_country(
    df: pd.DataFrame,
    fc_df: pd.DataFrame,
    target: str,
    fig_dir: str,
    train_end: int = 2016,
    val_end: int = 2020,
    test_end: int = 2024,
    dl_fc_df: pd.DataFrame | None = None,
) -> None:
    clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, country in enumerate(COUNTRIES):
        ax = axes[i]
        col = clrs[i % len(clrs)]
        hist = df[df["Area"] == country].sort_values("Year")
        frow = fc_df[fc_df["Country"] == country]
        if frow.empty:
            ax.set_visible(False)
            continue
        model = frow["Model"].values[0]

        tr = hist[hist["Year"] <= train_end]
        va = hist[(hist["Year"] > train_end) & (hist["Year"] <= val_end)]
        te = hist[hist["Year"] > val_end]

        ax.plot(tr["Year"], tr[target], color="#3a86ff", lw=2, label="Train")
        ax.plot(va["Year"], va[target], color="#ff9f1c", lw=2, label="Val")
        ax.plot(te["Year"], te[target], color="#e63946", lw=2, label="Test")

        ax.plot(
            [hist["Year"].values[-1], frow["Year"].values[0]],
            [hist[target].values[-1], frow["Forecast"].values[0]],
            color=col,
            lw=1.5,
            ls="--",
            alpha=0.5,
        )
        ax.plot(
            frow["Year"],
            frow["Forecast"],
            color=col,
            lw=2.5,
            ls="--",
            marker="D",
            ms=5,
            label=f"Forecast ({model})",
        )
        ax.fill_between(
            frow["Year"], frow["Lower_90"], frow["Upper_90"], alpha=0.18, color=col, label="90% CI"
        )

        # Optional DL forecast overlay
        if dl_fc_df is not None and not dl_fc_df.empty:
            dl_row = dl_fc_df[dl_fc_df["Country"] == country]
            if not dl_row.empty:
                ax.plot(
                    dl_row["Year"],
                    dl_row["Forecast"],
                    color="#00e5aa",
                    lw=1.5,
                    ls=":",
                    marker="s",
                    ms=3,
                    alpha=0.8,
                    label=f"DL ({dl_row['Model'].values[0]})",
                )

        ax.axvspan(train_end + 0.5, val_end + 0.5, alpha=0.05, color="orange")
        ax.axvspan(val_end + 0.5, test_end + 0.5, alpha=0.05, color="red")
        ax.axvline(test_end, color="grey", ls=":", lw=1)

        ax.set_title(country, fontweight="bold", fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel("TWh")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
        if i == 0:
            ax.legend(fontsize=7, ncol=2)

    axes[-1].set_visible(False)
    fig.suptitle(
        "Electricity Demand Forecast 2025-2030\n(Train/Val/Test + Recursive Forecast, 90% CI)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "forecast_per_country.png"))


def plot_forecast_overlay(
    df: pd.DataFrame,
    fc_df: pd.DataFrame,
    target: str,
    fig_dir: str,
    test_end: int = 2024,
    dl_fc_df: pd.DataFrame | None = None,
) -> None:
    clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, country in enumerate(COUNTRIES):
        col = clrs[i % len(clrs)]
        hist = df[df["Area"] == country].sort_values("Year")
        frow = fc_df[fc_df["Country"] == country]
        if frow.empty:
            continue
        ax.plot(hist["Year"], hist[target], color=col, lw=1.5, alpha=0.6)
        ax.plot(frow["Year"], frow["Forecast"], color=col, lw=2.5, ls="--", label=country)
        ax.fill_between(frow["Year"], frow["Lower_90"], frow["Upper_90"], alpha=0.09, color=col)
        if dl_fc_df is not None and not dl_fc_df.empty:
            dl_row = dl_fc_df[dl_fc_df["Country"] == country]
            if not dl_row.empty:
                ax.plot(dl_row["Year"], dl_row["Forecast"], color=col, lw=1.2, ls=":", alpha=0.5)
    ax.axvline(test_end, color="black", ls=":", lw=1.2)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("TWh", fontsize=12)
    ax.set_title(
        "All Countries — Demand History & Forecast 2025-2030", fontsize=13, fontweight="bold"
    )
    ax.legend(ncol=2, fontsize=9, bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "forecast_overlay.png"))


def plot_growth_uncertainty(fc_df: pd.DataFrame, growth_df: pd.DataFrame, fig_dir: str) -> None:
    if growth_df.empty:
        log.warning("growth_df empty — skipping plot")
        return
    clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.bar(
        growth_df["Country"],
        growth_df["CAGR 24-30 %"],
        color=[clrs[i % len(clrs)] for i in range(len(growth_df))],
        edgecolor="white",
    )
    for bar, val in zip(bars, growth_df["CAGR 24-30 %"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Electricity Demand CAGR 2024-2030", fontsize=12, fontweight="bold")
    ax.set_ylabel("CAGR (%)")
    ax.tick_params(axis="x", rotation=30)

    fc_df = fc_df.copy()
    fc_df["CI_Width"] = fc_df["Upper_90"] - fc_df["Lower_90"]
    ci_piv = fc_df.pivot(index="Country", columns="Year", values="CI_Width")
    sns.heatmap(
        ci_piv,
        cmap="OrRd",
        annot=True,
        fmt=".1f",
        ax=axes[1],
        linewidths=0.3,
        cbar_kws={"label": "CI Width (TWh)"},
    )
    axes[1].set_title(
        "Forecast Uncertainty — 90% CI Width per Year", fontsize=11, fontweight="bold"
    )

    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "growth_uncertainty.png"))
