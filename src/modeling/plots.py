"""
src/modeling/plots.py — All Figures for Classical Modeling Step
====================================================================
Ember Energy | IEEE Paper — Classical modeling module

All matplotlib/seaborn figures for 03_modeling.py:
    split viz, MAPE heatmaps, val-vs-test scatter, walk-forward plot,
    residuals, skill score.
"""

from __future__ import annotations

import logging

from src.config import cfg
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

COUNTRIES = cfg.countries


def savefig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


def plot_split_viz(
    df: pd.DataFrame, target: str, train_end: int, val_end: int, fig_dir: str
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=False)
    axes = axes.flatten()
    zone_c = {"Train": "#3a86ff", "Val": "#ff9f1c", "Test": "#e63946"}
    for i, country in enumerate(COUNTRIES):
        ax  = axes[i]
        sub = df[df["Area"] == country].sort_values("Year")
        tr  = sub[sub["Year"] <= train_end]
        va  = sub[(sub["Year"] > train_end) & (sub["Year"] <= val_end)]
        te  = sub[sub["Year"] > val_end]
        ax.plot(tr["Year"], tr[target], color=zone_c["Train"], lw=2, label="Train")
        ax.plot(va["Year"], va[target], color=zone_c["Val"],   lw=2, label="Val")
        ax.plot(te["Year"], te[target], color=zone_c["Test"],  lw=2, label="Test")
        ax.axvline(train_end + 0.5, color="grey", ls="--", lw=1)
        ax.axvline(val_end   + 0.5, color="grey", ls=":",  lw=1)
        ax.set_title(country, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("TWh")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        if i == 0:
            ax.legend(fontsize=8)
    axes[-1].set_visible(False)
    fig.suptitle("Train / Validation / Test Split by Country", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "mdl_fig_split.pdf"))


def plot_mape_heatmaps(
    test_metrics: pd.DataFrame, val_metrics: pd.DataFrame, fig_dir: str
) -> None:
    mape_piv     = test_metrics.pivot(index="Country", columns="Model", values="MAPE")
    val_mape_piv = val_metrics.pivot(index="Country",  columns="Model", values="MAPE")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.heatmap(mape_piv, cmap="RdYlGn_r", annot=True, fmt=".1f", ax=axes[0],
                linewidths=0.4, cbar_kws={"label": "MAPE %"})
    axes[0].set_title("TEST — MAPE (%) per Country × Model", fontweight="bold")
    sns.heatmap(val_mape_piv, cmap="RdYlGn_r", annot=True, fmt=".1f", ax=axes[1],
                linewidths=0.4, cbar_kws={"label": "MAPE %"})
    axes[1].set_title("VALIDATION — MAPE (%) per Country × Model", fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "mdl_fig_mape_heatmap.pdf"))


def plot_val_vs_test(
    val_metrics: pd.DataFrame, test_metrics: pd.DataFrame, fig_dir: str
) -> None:
    merged = (
        val_metrics[["Country", "Model", "MAPE"]].rename(columns={"MAPE": "Val_MAPE"})
        .merge(
            test_metrics[["Country", "Model", "MAPE"]].rename(columns={"MAPE": "Test_MAPE"}),
            on=["Country", "Model"],
        )
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in merged["Model"].unique():
        sub = merged[merged["Model"] == m]
        ax.scatter(sub["Val_MAPE"], sub["Test_MAPE"], label=m, s=60)
    max_v = merged[["Val_MAPE", "Test_MAPE"]].max().max()
    ax.plot([0, max_v], [0, max_v], "k--", lw=0.8, label="Val=Test line")
    ax.set_xlabel("Val MAPE (%)")
    ax.set_ylabel("Test MAPE (%)")
    ax.set_title("Val vs Test MAPE — Overfitting Check", fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "mdl_fig_val_vs_test.pdf"))


def plot_walk_forward_test(
    df: pd.DataFrame,
    res: pd.DataFrame,
    best_test: pd.DataFrame,
    target: str,
    train_end: int,
    val_end: int,
    test_end: int,
    fig_dir: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    axes = axes.flatten()
    clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, country in enumerate(COUNTRIES):
        ax = axes[i]
        full   = df[df["Area"] == country].sort_values("Year")
        best_m = best_test[best_test["Country"] == country]["Model"].values[0]
        ax.plot(full["Year"], full[target], "k-", lw=2, label="Actual", zorder=5)
        c_res = res[res["Country"] == country]
        for j, model in enumerate(c_res["Model"].unique()):
            m_res = c_res[c_res["Model"] == model]
            lw = 2.5 if model == best_m else 1
            ls = "-"  if model == best_m else "--"
            ax.plot(m_res["Year"], m_res["y_pred"], ls=ls, lw=lw,
                    color=clrs[j % len(clrs)], label=model, alpha=0.85)
        ax.axvspan(val_end + 0.5, test_end + 0.5, alpha=0.06, color="red", label="Test zone")
        ax.axvspan(train_end + 0.5, val_end + 0.5, alpha=0.06, color="orange", label="Val zone")
        ax.set_title(f"{country}\n[Best: {best_m}]", fontweight="bold", fontsize=9)
        ax.set_xlabel("Year"); ax.set_ylabel("TWh")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        if i == 0:
            ax.legend(fontsize=6, ncol=2)
    axes[-1].set_visible(False)
    fig.suptitle("Walk-Forward Test Predictions vs Actual (all models)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "mdl_fig_walk_forward.pdf"))


def plot_residuals(
    df: pd.DataFrame, res: pd.DataFrame, best_test: pd.DataFrame, fig_dir: str
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    axes = axes.flatten()
    for i, country in enumerate(COUNTRIES):
        best_m = best_test[best_test["Country"] == country]["Model"].values[0]
        r = res[(res["Country"] == country) & (res["Model"] == best_m)]
        clrs_bar = ["tomato" if e < 0 else "steelblue" for e in r["error"]]
        axes[i].bar(r["Year"], r["error"], color=clrs_bar)
        axes[i].axhline(0, color="black", lw=0.8)
        axes[i].set_title(f"{country} — {best_m}", fontweight="bold", fontsize=9)
        axes[i].set_xlabel("Year"); axes[i].set_ylabel("Residual (TWh)")
    axes[-1].set_visible(False)
    fig.suptitle("Test Residuals — Best Model per Country", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "mdl_fig_residuals.pdf"))


def plot_skill_score(test_metrics: pd.DataFrame, fig_dir: str) -> None:
    naive_mape = (
        test_metrics[test_metrics["Model"] == "Naive"][["Country", "MAPE"]]
        .rename(columns={"MAPE": "naive_mape"})
    )
    skill_df = test_metrics.merge(naive_mape, on="Country", how="inner")
    skill_df["skill"] = (1 - skill_df["MAPE"] / skill_df["naive_mape"]) * 100
    skill_df = skill_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["skill"])
    skill_piv = skill_df.pivot(index="Country", columns="Model", values="skill")

    if skill_piv.empty:
        log.warning("Skill score pivot empty — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(skill_piv, cmap="RdYlGn", annot=True, fmt=".1f", ax=ax,
                center=0, linewidths=0.4, cbar_kws={"label": "Skill vs Naïve (%)"})
    ax.set_title("Skill Score vs Naïve Baseline (Test Set) — Higher is Better",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "mdl_fig_skill_score.pdf"))
