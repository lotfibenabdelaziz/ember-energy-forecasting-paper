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

# ── Windows home/cache-dir fix — see src/config.py for full explanation ──────
if os.name == "nt":
    import tempfile

    _fallback_dir = tempfile.gettempdir()
    os.environ.setdefault("USERPROFILE", _fallback_dir)
    os.environ.setdefault("LOCALAPPDATA", _fallback_dir)
    os.environ.setdefault("HOMEDRIVE", os.path.splitdrive(_fallback_dir)[0] or "C:")
    os.environ.setdefault("HOMEPATH", os.path.splitdrive(_fallback_dir)[1] or "\\Temp")
    os.environ.setdefault("HOME", _fallback_dir)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(_fallback_dir, "mplcache"))

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


def _boxplot_compat(ax, data, *, tick_labels=None, **kwargs):
    """
    matplotlib.Axes.boxplot()'s tick-label kwarg was renamed twice:
    `labels` (< 3.9) -> `tick_labels` (>= 3.9, `labels` deprecated) ->
    `labels` removed entirely (>= 3.11). pyproject.toml only pins
    matplotlib>=3.6.0 with no upper bound, so depending on what a fresh
    install resolves, either kwarg name can be the only one that works.
    Try the current name first, fall back to the old one on TypeError.
    """
    try:
        return ax.boxplot(data, tick_labels=tick_labels, **kwargs)
    except TypeError:
        return ax.boxplot(data, labels=tick_labels, **kwargs)


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw Ember CSV, filter to the 7 target countries, and keep only
    the original Subcategory column (CO2_intensity, Demand, Demand_per_capita,
    Electricity_imports, Fuel — the ~5-variable set proven stable across
    all 7 countries, with no missing-generation-category NaN cascades).
    """
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


def plot_all_subcategories_grid(df_long: pd.DataFrame, fig_dir: str) -> None:
    """
    Unified 5-panel grid — Demand plus the four other retained
    subcategories (CO2 intensity, Demand per capita, Electricity
    imports, Fuel) — each panel overlaying all 7 countries, same visual
    style/palette as the original standalone Demand overlay. Replaces
    the previous split between a standalone Demand figure and a
    separate 4-panel grid for the others, per paper figure consolidation.
    """
    # NOTE: df_long is RAW long-format data (straight from ember_filtered.csv),
    # where Subcategory values use SPACES ("CO2 intensity"), not the
    # underscored names ("CO2_intensity") used only after step02's pivot to
    # wide format. Filtering with underscores here silently matches nothing
    # for any multi-word subcategory, leaving those panels empty.
    subcats = ["Demand", "CO2 intensity", "Demand per capita", "Electricity imports", "Fuel"]
    titles = ["Demand", "CO2 Intensity", "Demand per Capita", "Electricity Imports", "Fuel"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()

    for i, (subcat, title) in enumerate(zip(subcats, titles, strict=True)):
        ax = axes[i]
        sub_df = df_long[df_long["Subcategory"] == subcat]
        if sub_df.empty:
            ax.set_visible(False)
            continue
        unit = sub_df["Unit"].iloc[0] if "Unit" in sub_df.columns else ""

        for c in COUNTRIES:
            d = sub_df[sub_df["Area"] == c].sort_values("Year")
            if d.empty:
                continue
            # Fuel carries multiple undifferentiated component rows per
            # (Area, Year) in this dataset (no fuel-type column to split
            # them) — summed as total generation capacity, matching the
            # same logic already established for the dashboard's "Total
            # Generation Capacity" series. The other subcategories
            # (including Demand) are genuinely single-valued per year;
            # mean is a safe no-op in those cases.
            agg_fn = "sum" if subcat == "Fuel" else "mean"
            d = d.groupby("Year", as_index=False)["Value"].agg(agg_fn)
            ax.plot(d["Year"], d["Value"], color=PALETTE[c], lw=1.5, marker="o", ms=2.5, label=c)

        ax.set_title(f"{title} — All Countries ({unit})", fontweight="bold", fontsize=10.5)
        ax.set_xlabel("Year")
        ax.set_ylabel(unit)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))

    axes[-1].set_visible(False)  # 6th grid slot unused (5 subcategories)
    axes[0].legend(ncol=4, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.3))
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "eda_fig1_all_subcategories_grid.pdf"))


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
    bp = _boxplot_compat(
        ax,
        data,
        patch_artist=True,
        tick_labels=COUNTRIES,
        medianprops={"color": "black", "lw": 1.8},
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
    plot_all_subcategories_grid(df_long, fig_dir)
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
