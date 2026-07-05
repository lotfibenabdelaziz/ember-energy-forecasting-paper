"""
src/01_eda.py — Exploratory Data Analysis
Ember Energy | IEEE Paper

Reads:   data/raw/yearly_full_release_long_format.csv
Writes:  outputs/eda/ember_filtered.csv
         outputs/eda/ember_multifeature.csv
         outputs/eda/figures/*.pdf
         outputs/eda/table01_eda_statistics.csv
"""

import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET = "Demand"
PALETTE = {
    "Tunisia": "#1f77b4",
    "Austria": "#d62728",
    "Germany": "#2ca02c",
    "Egypt": "#9467bd",
    "Canada": "#e6a817",
    "France": "#8c564b",
    "Kuwait": "#17becf",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def savefig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading: %s", csv_path)
    df_all = pd.read_csv(csv_path)
    df_all.columns = df_all.columns.str.strip()
    df_long = df_all[df_all["Area"].isin(COUNTRIES)][
        ["Area", "Year", "Subcategory", "Unit", "Value"]
    ].copy()
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")

    df_demand = df_long[df_long["Subcategory"] == TARGET].copy()
    log.info(
        "Shape: %s | Subcategories: %d | Countries: %s",
        df_long.shape,
        df_long["Subcategory"].nunique(),
        COUNTRIES,
    )
    return df_long, df_demand


# ── Section 1: Demand trends ──────────────────────────────────────────────────
def plot_demand_overlay(df_demand: pd.DataFrame, fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in COUNTRIES:
        d = df_demand[df_demand["Area"] == c].sort_values("Year")
        ax.plot(d["Year"], d["Value"], color=PALETTE[c], lw=1.8, marker="o", ms=3, label=c)
    ax.set_title(f"{TARGET} — All Countries (TWh)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("TWh")
    ax.legend(ncol=4, frameon=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "eda_fig1_demand_overlay.pdf"))


def plot_demand_multiples(df_demand: pd.DataFrame, fig_dir: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    axes = axes.flatten()
    for i, c in enumerate(COUNTRIES):
        ax = axes[i]
        d = df_demand[df_demand["Area"] == c].sort_values("Year")
        ax.plot(d["Year"], d["Value"], color=PALETTE[c], lw=1.8)
        ax.fill_between(d["Year"], d["Value"], alpha=0.1, color=PALETTE[c])
        ax.set_title(c, fontweight="bold")
        ax.set_ylabel("TWh")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    axes[-1].set_visible(False)
    fig.suptitle(f"{TARGET} — Individual Panels", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "eda_fig2_demand_multiples.pdf"))


# ── Section 2: Growth rates & CAGR ───────────────────────────────────────────
def plot_yoy(df_demand: pd.DataFrame, fig_dir: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    axes = axes.flatten()
    for i, c in enumerate(COUNTRIES):
        ax = axes[i]
        d = df_demand[df_demand["Area"] == c].sort_values("Year")
        yoy = d["Value"].pct_change() * 100
        ax.bar(
            d["Year"].values[1:],
            yoy.values[1:],
            color=[PALETTE[c] if v >= 0 else "#e74c3c" for v in yoy.values[1:]],
            width=0.8,
        )
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(c, fontweight="bold")
        ax.set_ylabel("YoY (%)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    axes[-1].set_visible(False)
    fig.suptitle("Year-over-Year Growth Rate", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "eda_fig3_yoy.pdf"))


def plot_cagr_heatmap(df_demand: pd.DataFrame, fig_dir: str) -> None:
    periods = [(2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020), (2020, 2024)]
    rows = []
    for c in COUNTRIES:
        d = df_demand[df_demand["Area"] == c].set_index("Year")["Value"].sort_index()
        row = {"Country": c}
        for s, e in periods:
            try:
                v_s = d.loc[s]
                v_e = d.loc[e]
                row[f"{s}–{e}"] = round((v_e / v_s) ** (1 / (e - s)) - 1, 4) * 100
            except KeyError:
                row[f"{s}–{e}"] = np.nan
        rows.append(row)
    df_cagr = pd.DataFrame(rows).set_index("Country")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        df_cagr,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.4,
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Rolling 5-Year CAGR (%) per Country", fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "eda_fig4_cagr_heatmap.pdf"))


# ── Section 3: Distribution ───────────────────────────────────────────────────
def plot_distribution(df_demand: pd.DataFrame, fig_dir: str) -> None:
    # Boxplot
    fig, ax = plt.subplots(figsize=(11, 5))
    data = [df_demand[df_demand["Area"] == c]["Value"].dropna().values for c in COUNTRIES]
    colors = [PALETTE[c] for c in COUNTRIES]
    bp = ax.boxplot(
        data, patch_artist=True, labels=COUNTRIES, medianprops={"color": "black", "lw": 1.8}
    )
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(f"{TARGET} Distribution per Country (TWh)", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "eda_fig5_boxplot.pdf"))


# ── Section 4: Stationarity (ADF) ────────────────────────────────────────────
def run_adf(df_demand: pd.DataFrame, output_dir: str) -> None:
    rows = []
    for c in COUNTRIES:
        s = df_demand[df_demand["Area"] == c]["Value"].dropna().values
        if len(s) < 5:
            continue
        adf_stat, p_val, _, _, crit, _ = adfuller(s, autolag="AIC")
        rows.append(
            {
                "Country": c,
                "ADF_stat": round(adf_stat, 4),
                "p_value": round(p_val, 4),
                "Stationary": "Yes" if p_val < 0.05 else "No",
                "Crit_5pct": round(crit["5%"], 4),
            }
        )
    df_adf = pd.DataFrame(rows)
    log.info("=== ADF Stationarity Tests ===\n%s", df_adf.to_string(index=False))
    df_adf.to_csv(os.path.join(output_dir, "eda_adf_tests.csv"), index=False)


# ── Section 5: Normalised demand & summary ───────────────────────────────────
def plot_normalised(df_demand: pd.DataFrame, fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    base_yr = 2000
    for c in COUNTRIES:
        d = df_demand[df_demand["Area"] == c].sort_values("Year")
        base = d[d["Year"] == base_yr]["Value"].values
        if len(base) == 0:
            continue
        idx = d["Value"] / base[0] * 100
        ax.plot(d["Year"], idx, color=PALETTE[c], lw=1.8, label=c)
    ax.axhline(100, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(f"Normalised {TARGET} (Base {base_yr} = 100)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Index")
    ax.legend(ncol=4, frameon=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "eda_fig6_normalised.pdf"))


def build_summary_table(df_demand: pd.DataFrame, output_dir: str) -> None:
    rows = []
    for c in COUNTRIES:
        d = df_demand[df_demand["Area"] == c].sort_values("Year")
        s = d["Value"]
        try:
            v_2000 = float(d[d["Year"] == 2000]["Value"].values[0])
        except IndexError:
            v_2000 = np.nan
        try:
            v_2024 = float(d[d["Year"] == 2024]["Value"].values[0])
        except IndexError:
            v_2024 = float(d["Value"].iloc[-1])
        slope = np.polyfit(d["Year"], s, 1)[0]
        rows.append(
            {
                "Country": c,
                "2000 (TWh)": round(v_2000, 1),
                "2024 (TWh)": round(v_2024, 1),
                "Total Growth": f"{round((v_2024 / v_2000 - 1) * 100 if v_2000 else np.nan, 1)}%",
                "Slope TWh/yr": round(slope, 2),
                "Mean TWh": round(float(s.mean()), 1),
                "Std TWh": round(float(s.std()), 1),
            }
        )
    df_stats = pd.DataFrame(rows)
    log.info("=== Summary Statistics ===\n%s", df_stats.to_string(index=False))
    df_stats.to_csv(os.path.join(output_dir, "table01_eda_statistics.csv"), index=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember EDA step")
    p.add_argument("--csv", required=True, help="Path to Ember yearly CSV")
    p.add_argument("--output_dir", default="outputs/eda", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df_long, df_demand = load_data(args.csv)

    # Save filtered data
    df_long.to_csv(os.path.join(args.output_dir, "ember_filtered.csv"), index=False)
    log.info("Saved → ember_filtered.csv")

    # Plots
    plot_demand_overlay(df_demand, fig_dir)
    plot_demand_multiples(df_demand, fig_dir)
    plot_yoy(df_demand, fig_dir)
    plot_cagr_heatmap(df_demand, fig_dir)
    plot_distribution(df_demand, fig_dir)
    run_adf(df_demand, args.output_dir)
    plot_normalised(df_demand, fig_dir)
    build_summary_table(df_demand, args.output_dir)

    log.info("EDA complete → %s", args.output_dir)


if __name__ == "__main__":
    main()
