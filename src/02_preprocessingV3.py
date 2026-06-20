"""
src/02_preprocessing.py — Data Preprocessing & Feature Engineering
===================================================================
Ember Energy | IEEE Paper

Mirrors EXACTLY: notebooks/01_preprocessing_patched.ipynb
Cell-by-cell conversion — no deviations.

Reads:   outputs/eda/ember_filtered.csv
Writes:  outputs/preprocessing/ember_model_ready.csv
         outputs/preprocessing/ember_multifeature.csv
         outputs/preprocessing/feature_meta.json
         outputs/preprocessing/figures/*.pdf

Notebook → Script mapping:
    Cell 0-1  : constants, libraries
    Cell 2    : load_filtered()
    Cell 3    : pivot_wide()        [rename + DROP_COLS]
    Cell 4    : quality_report()
    Cell 5-6  : plot_check1()
    Cell 7    : impute_missing()    [2-pass: interpolate + mean]
    Cell 8-9  : quality_report + plot_check2()
    Cell 10   : remove_duplicates()
    Cell 11   : plot_check3()
    Cell 12   : winsorise_group() + apply_winsorization()
    Cell 13-15: quality_report + plot_check4()
    Cell 16   : lags + MA + yoy + fix NaN/inf → fillna(0)
    Cell 17   : cross-features + trend + country_code
    Cell 18   : quality_report FINAL
    Cell 19-21: plot_check5() [top25 corr + heatmap + pairplot]
    Cell 22   : RF importance → fi_cols
    Cell 23   : save df_feat + df_model + feature_meta.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Cell 1: Constants ─────────────────────────────────────────────────────────
COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET    = "Demand"
DROP_COLS = ["Total", "Aggregate_fuel"]   # Cell 3 — dropped after pivot

PALETTE = {
    "Tunisia": "#e63946", "Austria": "#2196F3", "Germany": "#4CAF50",
    "Egypt":   "#9C27B0", "Canada":  "#00BCD4", "France":  "#797148",
    "Kuwait":  "#4C4879",
}

pd.set_option("display.float_format", "{:.3f}".format)
sns.set_theme(style="whitegrid", palette="tab10")

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "axes.titlesize":    11,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── Utilities ─────────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


# ── Cell 4: quality_report — mirrors notebook exactly ────────────────────────

def quality_report(df: pd.DataFrame, label: str = "") -> None:
    log.info("═" * 60)
    log.info("  Quality Report: %s", label)
    log.info("═" * 60)
    log.info("  Shape      : %s", df.shape)
    log.info("  Duplicates : %d", df.duplicated().sum())
    miss = df.isnull().sum()
    miss_nonzero = miss[miss > 0]
    if len(miss_nonzero):
        log.info("  Missing:\n%s", miss_nonzero.to_string())
    else:
        log.info("  Missing    : None")
    num = df.select_dtypes(include=np.number).columns
    if len(num):
        log.info("  Numeric Summary:\n%s", df[num].describe().round(2).to_string())


# ── Cell 2: Load filtered CSV ─────────────────────────────────────────────────

def load_filtered(input_dir: str) -> pd.DataFrame:
    """Mirrors notebook Cell 2 exactly."""
    path = os.path.join(input_dir, "ember_filtered.csv")
    df_all = pd.read_csv(path)
    df_all.columns = df_all.columns.str.strip()
    df = (
        df_all[df_all["Area"].isin(COUNTRIES)]
        [["Area", "Year", "Subcategory", "Unit", "Value"]]
        .copy()
    )
    df["Year"]  = df["Year"].astype(int)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    log.info("Shape        : %s", df.shape)
    log.info("Years        : %d – %d", df.Year.min(), df.Year.max())
    log.info("Countries    : %s", sorted(df.Area.unique()))
    log.info("Subcategories: %d → %s", df.Subcategory.nunique(),
             sorted(df.Subcategory.unique()))
    return df


# ── Cell 3: Pivot wide ────────────────────────────────────────────────────────

def pivot_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Mirrors notebook Cell 3 exactly."""
    df_agg = (
        df.groupby(["Area", "Year", "Subcategory"], as_index=False)["Value"]
        .mean()
    )
    df_wide = df_agg.pivot_table(
        index=["Area", "Year"],
        columns="Subcategory",
        values="Value",
    ).reset_index()
    df_wide.columns.name = None
    df_wide.columns = [
        c.strip().replace(" ", "_").replace("/", "_")
        for c in df_wide.columns
    ]
    df_wide = df_wide.drop(columns=DROP_COLS, errors="ignore")

    ALL_SUBS = [c for c in df_wide.columns if c not in ["Area", "Year"]]
    FEATURES = [c for c in ALL_SUBS if c not in ([TARGET] + DROP_COLS)]

    log.info("Wide shape   : %s", df_wide.shape)
    log.info("Subcategories: %s", ALL_SUBS)
    log.info("Target       : %s", TARGET)
    log.info("Features     : %s", FEATURES)
    return df_wide, ALL_SUBS, FEATURES


# ── Cells 5-6: Check 1 plots ──────────────────────────────────────────────────

def plot_check1_missing(
    df_wide: pd.DataFrame, all_subs: list[str], fig_dir: str
) -> None:
    """Mirrors notebook Cell 5."""
    miss_pct = (
        df_wide.groupby("Area")[all_subs]
        .apply(lambda g: g.isnull().mean() * 100)
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(miss_pct, cmap="Reds", annot=True, fmt=".1f", ax=ax,
                linewidths=0.4, cbar_kws={"label": "Missing %"})
    ax.set_title("Check 1 — Missing % per Country × Feature (Raw)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "check1_missing.pdf"))


def plot_check1_corr(corr_raw: pd.Series, fig_dir: str) -> None:
    """Mirrors notebook Cell 6."""
    fig, ax = plt.subplots(figsize=(8, 4))
    clrs    = ["steelblue" if v >= 0 else "tomato" for v in corr_raw]
    ax.barh(corr_raw.index, corr_raw.values, color=clrs)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Check 1 — Correlation with Demand (Raw)", fontweight="bold")
    for i, v in enumerate(corr_raw):
        ax.text(v + 0.01 * np.sign(v), i, f"{v:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "check1_corr_raw.pdf"))


# ── Cell 7: Impute missing ────────────────────────────────────────────────────

def impute_missing(df_wide: pd.DataFrame, all_subs: list[str]) -> pd.DataFrame:
    """Mirrors notebook Cell 7 exactly."""
    all_years = list(range(df_wide["Year"].min(), df_wide["Year"].max() + 1))
    idx_full  = pd.MultiIndex.from_product([COUNTRIES, all_years], names=["Area", "Year"])
    df_wide   = df_wide.set_index(["Area", "Year"]).reindex(idx_full).reset_index()
    log.info("After reindex: %s | Gaps: %d",
             df_wide.shape, df_wide[all_subs].isnull().sum().sum())

    # Pass 1: linear interpolation
    for col in all_subs:
        df_wide[col] = df_wide.groupby("Area")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )

    # Pass 2: country mean fallback
    for col in all_subs:
        df_wide[col] = df_wide.groupby("Area")[col].transform(
            lambda s: s.fillna(s.mean())
        )

    log.info("Remaining nulls: %d", df_wide[all_subs].isnull().sum().sum())
    return df_wide


# ── Cell 9: Check 2 plot ──────────────────────────────────────────────────────

def plot_check2_timeseries(
    df_wide: pd.DataFrame, all_subs: list[str], fig_dir: str
) -> None:
    """Mirrors notebook Cell 9."""
    n_subs = len(all_subs)
    fig, axes = plt.subplots(
        len(COUNTRIES), n_subs,
        figsize=(n_subs * 3.2, len(COUNTRIES) * 2.2),
        sharey=False,
    )
    fig.suptitle(
        "Check 2 — All Features After Imputation (each Country × Subcategory)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    for i, country in enumerate(COUNTRIES):
        sub = df_wide[df_wide["Area"] == country].sort_values("Year")
        for j, col in enumerate(all_subs):
            ax  = axes[i][j]
            clr = "tomato" if col == TARGET else "#3a86ff"
            ax.plot(sub["Year"], sub[col], color=clr, lw=1.4)
            ax.set_title(f"{col}\n{country}", fontsize=6.5,
                         fontweight="bold" if col == TARGET else "normal")
            ax.tick_params(labelsize=5)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, integer=True))
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "check2_timeseries.pdf"))


# ── Cell 10: Deduplicate ──────────────────────────────────────────────────────

def remove_duplicates(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Mirrors notebook Cell 10 exactly."""
    before   = len(df_wide)
    df_wide  = df_wide.drop_duplicates(subset=["Area", "Year"]).reset_index(drop=True)
    log.info("Duplicates removed: %d", before - len(df_wide))
    return df_wide


# ── Cell 11: Check 3 plot ─────────────────────────────────────────────────────

def plot_check3_coverage(df_wide: pd.DataFrame, fig_dir: str) -> None:
    """Mirrors notebook Cell 11."""
    pivot_cov = df_wide.pivot_table(
        index="Area", columns="Year", values=TARGET, aggfunc="count"
    )
    fig, ax = plt.subplots(figsize=(16, 3))
    sns.heatmap(pivot_cov, cmap="YlGn", ax=ax, linewidths=0.3,
                annot=True, fmt=".0f", annot_kws={"size": 7},
                cbar_kws={"label": "Count"})
    ax.set_title("Check 3 — Coverage Matrix (Country × Year)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "check3_coverage.pdf"))


# ── Cell 12: Winsorization ────────────────────────────────────────────────────

def winsorise_group(
    group: pd.DataFrame, cols: list[str], factor: float = 2.5
) -> pd.DataFrame:
    """Mirrors notebook Cell 12 winsorise_group() exactly."""
    g = group.copy()
    for col in cols:
        q1, q3 = g[col].quantile([0.25, 0.75])
        iqr    = q3 - q1
        g[col] = g[col].clip(q1 - factor * iqr, q3 + factor * iqr)
    return g


def apply_winsorization(
    df_wide: pd.DataFrame, all_subs: list[str]
) -> pd.DataFrame:
    """Mirrors notebook Cell 12 exactly."""
    df_clean = (
        df_wide.groupby("Area", group_keys=False)
        .apply(lambda g: winsorise_group(g, all_subs))
        .reset_index(drop=True)
    )
    diff = (df_wide[all_subs] - df_clean[all_subs]).abs()
    log.info("Cells clipped per feature:\n%s", (diff > 0).sum().to_string())
    return df_clean


# ── Cells 14-15: Check 4 plots ────────────────────────────────────────────────

def plot_check4_boxplots(
    df_wide: pd.DataFrame, df_clean: pd.DataFrame,
    all_subs: list[str], fig_dir: str,
) -> None:
    """Mirrors notebook Cell 14 exactly."""
    fig, axes = plt.subplots(2, len(all_subs), figsize=(len(all_subs) * 3, 8))
    for j, col in enumerate(all_subs):
        axes[0][j].boxplot(df_wide[col].dropna(), patch_artist=True,
                           boxprops={"facecolor": "#ffcccb"})
        axes[0][j].set_title(f"{col}\n(Before)", fontsize=8,
                             fontweight="bold" if col == TARGET else "normal")
        axes[1][j].boxplot(df_clean[col].dropna(), patch_artist=True,
                           boxprops={"facecolor": "#cce5ff"})
        axes[1][j].set_title(f"{col}\n(After)", fontsize=8,
                             fontweight="bold" if col == TARGET else "normal")
        for r in [0, 1]:
            axes[r][j].tick_params(labelsize=7, bottom=False, labelbottom=False)
    axes[0][0].set_ylabel("Before", fontsize=9)
    axes[1][0].set_ylabel("After",  fontsize=9)
    fig.suptitle("Check 4 — Outlier Treatment: Before vs After",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "check4_boxplots.pdf"))


def plot_check4_corr(corr_clean: pd.Series, fig_dir: str) -> None:
    """Mirrors notebook Cell 15 exactly."""
    fig, ax = plt.subplots(figsize=(8, 4))
    clrs    = ["steelblue" if v >= 0 else "tomato" for v in corr_clean]
    ax.barh(corr_clean.index, corr_clean.values, color=clrs)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Check 4 — Correlation with Demand (Post-Cleaning)",
                 fontweight="bold")
    for i, v in enumerate(corr_clean):
        ax.text(v + 0.01 * np.sign(v), i, f"{v:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "check4_corr_clean.pdf"))


# ── Cells 16-17: Feature engineering ─────────────────────────────────────────

def engineer_features(
    df_clean: pd.DataFrame,
    all_subs: list[str],
    corr_clean: pd.Series,
) -> pd.DataFrame:
    """
    Mirrors notebook Cells 16-17 exactly.
    Cell 16: lags + MA + YoY + fix NaN/inf → fillna(0)
    Cell 17: cross-features + trend + country_code
    """
    df_feat = df_clean.sort_values(["Area", "Year"]).reset_index(drop=True)

    # Cell 16 — Lags 1-3
    for col in all_subs:
        for lag in [1, 2, 3]:
            df_feat[f"{col}_lag{lag}"] = df_feat.groupby("Area")[col].shift(lag)

    # Cell 16 — Rolling MA-3, MA-5 (shift-1)
    for col in all_subs:
        for w in [3, 5]:
            df_feat[f"{col}_ma{w}"] = df_feat.groupby("Area")[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )

    # Cell 16 — YoY % change
    for col in all_subs:
        df_feat[f"{col}_yoy"] = df_feat.groupby("Area")[col].pct_change() * 100

    # Cell 16 — Fix NaN/inf in yoy + ratio cols → fillna(0)
    yoy_cols   = [c for c in df_feat.columns if c.endswith("_yoy")]
    ratio_cols = [c for c in df_feat.columns if c.startswith("demand_ratio_")]
    df_feat[yoy_cols + ratio_cols] = (
        df_feat[yoy_cols + ratio_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    log.info("NaN in yoy cols after fix: %d",
             df_feat[yoy_cols].isnull().sum().sum())
    log.info("After lags + MA + YoY: %s", df_feat.shape)

    # Cell 17 — Cross-feature interactions
    subs_lower = {c: c.lower() for c in all_subs}
    gen_cols   = [c for c in all_subs if any(
        k in subs_lower[c]
        for k in ["generation", "wind", "solar", "hydro",
                  "coal", "gas", "nuclear", "other"]
    )]

    if len(gen_cols) >= 2:
        df_feat["total_generation"] = df_feat[gen_cols].sum(axis=1)
        for c in gen_cols:
            df_feat[f"share_{c}"] = df_feat[c] / (df_feat["total_generation"] + 1e-9)
        log.info("Generation share features: %s", gen_cols)

    # Demand intensity vs top 2 correlated features
    top2 = corr_clean.abs().nlargest(min(2, len(corr_clean))).index.tolist()
    for feat in top2:
        df_feat[f"demand_ratio_{feat}"] = df_feat[TARGET] / (df_feat[feat] + 1e-9)
        log.info("Created: demand_ratio_%s", feat)

    # Cell 17 — Trend terms + country code
    df_feat["trend"]        = df_feat["Year"] - df_feat["Year"].min()
    df_feat["trend_sq"]     = df_feat["trend"] ** 2
    df_feat["country_code"] = df_feat["Area"].astype("category").cat.codes

    log.info("Final feature matrix shape: %s", df_feat.shape)
    return df_feat


# ── Cells 19-21: Check 5 plots ────────────────────────────────────────────────

def plot_check5_top_corr(corr_full: pd.Series, fig_dir: str) -> None:
    """Mirrors notebook Cell 19 exactly."""
    top25 = corr_full.head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    clrs    = ["steelblue" if v >= 0 else "tomato" for v in top25.values]
    ax.barh(top25.index[::-1], top25.values[::-1], color=clrs[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(
        "Check 5 — Top 25 Features Correlated with Demand\n(all engineered features)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Pearson Correlation")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "check5_top_corr.pdf"))


def plot_check5_heatmap(
    df_feat: pd.DataFrame, all_subs: list[str], fig_dir: str
) -> None:
    """Mirrors notebook Cell 20 exactly."""
    corr = df_feat[all_subs].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, mask=mask, linewidths=0.4, vmin=-1, vmax=1)
    ax.set_title("Check 5 — Subcategory Correlation Matrix",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "check5_corr_heatmap.pdf"))


def plot_check5_pairplot(
    df_feat: pd.DataFrame, all_subs: list[str], fig_dir: str
) -> None:
    """Mirrors notebook Cell 21 exactly."""
    g = sns.pairplot(
        df_feat[all_subs + ["Area"]].dropna(),
        hue="Area", vars=all_subs,
        diag_kind="kde", plot_kws={"alpha": 0.45, "s": 18}, corner=True,
    )
    g.figure.suptitle(
        "Check 5 — Subcategory Pairplot (coloured by Country)",
        y=1.01, fontsize=12, fontweight="bold",
    )
    savefig(g.figure, os.path.join(fig_dir, "check5_pairplot.pdf"))


# ── Cell 22: RF importance → fi_cols ─────────────────────────────────────────

def compute_fi_cols(
    df_feat: pd.DataFrame,
) -> tuple[list[str], pd.Series, RandomForestRegressor]:
    """
    Mirrors notebook Cell 22 exactly.
    Returns (fi_cols, corr_full, rf_model)
    """
    num_cols  = df_feat.select_dtypes(include=np.number).columns.tolist()
    num_cols  = [c for c in num_cols if c not in ["Year", "country_code"]]
    corr_full = df_feat[num_cols].corr()[TARGET].drop(TARGET, errors="ignore").dropna()
    corr_full = corr_full.sort_values(key=abs, ascending=False)

    fi_cols = [
        c for c in num_cols
        if c != TARGET
        and not c.startswith(f"{TARGET}_lag")
        and not c.startswith(f"{TARGET}_ma")
    ]

    # Explicitly replace inf/-inf with NaN BEFORE dropna
    df_fi = (
        df_feat[fi_cols + [TARGET]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # Confirm no infs remain
    assert not np.isinf(df_fi.values).any(), "Still has inf values!"
    log.info("RF training shape: %s", df_fi.shape)

    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    rf.fit(df_fi[fi_cols], df_fi[TARGET])

    fi = pd.Series(rf.feature_importances_, index=fi_cols).sort_values(ascending=False)
    log.info("Top 5 RF features:\n%s", fi.head(5).to_string())

    return fi_cols, corr_full, rf


# ── Cell 23: Save ─────────────────────────────────────────────────────────────

def save_outputs(
    df_feat:  pd.DataFrame,
    df_model: pd.DataFrame,
    fi_cols:  list[str],
    corr_full: pd.Series,
    all_subs: list[str],
    features: list[str],
    output_dir: str,
    train_until: int,
) -> None:
    """Mirrors notebook Cell 23 exactly."""
    df_feat.to_csv(
        os.path.join(output_dir, "ember_multifeature.csv"), index=False)
    df_model.to_csv(
        os.path.join(output_dir, "ember_model_ready.csv"),  index=False)

    feature_meta = {
        # Notebook lowercase keys — mirrors Cell 23 exactly
        "target":        TARGET,
        "all_subs":      all_subs,
        "raw_features":  features,
        "all_features":  fi_cols,
        "top_corr_20":   corr_full.head(20).index.tolist(),
        # Uppercase keys for downstream .py scripts
        "TARGET":        TARGET,
        "ALL_SUBS":      all_subs,
        "FEATURES":      features,
        "COUNTRIES":     COUNTRIES,
        "DROP_COLS":     DROP_COLS,
        "TRAIN_END":     train_until,
        "VAL_END":       2020,
        "TEST_END":      2024,
        "FORECAST_YEARS": list(range(2025, 2031)),
    }

    with open(os.path.join(output_dir, "feature_meta.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)

    log.info("✓ ember_multifeature.csv → %s", df_feat.shape)
    log.info("✓ ember_model_ready.csv  → %s", df_model.shape)
    log.info("✓ feature_meta.json")
    log.info("✅ Preprocessing complete — proceed to 03_modeling.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember preprocessing step")
    p.add_argument("--input_dir",   default="outputs/eda",          help="EDA output dir")
    p.add_argument("--output_dir",  default="outputs/preprocessing", help="Output dir")
    p.add_argument("--train_until", type=int, default=2016,          help="Last training year")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir,         exist_ok=True)

    # Cell 2 — Load
    df = load_filtered(args.input_dir)

    # Cell 3 — Pivot + rename + DROP_COLS
    df_wide, ALL_SUBS, FEATURES = pivot_wide(df)

    # Cell 4 — Quality: RAW
    quality_report(df_wide, "RAW Wide (post-pivot)")

    # Cells 5-6 — Check 1 plots
    corr_raw = df_wide[ALL_SUBS].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    plot_check1_missing(df_wide, ALL_SUBS, fig_dir)
    plot_check1_corr(corr_raw, fig_dir)

    # Cell 7 — Impute
    df_wide = impute_missing(df_wide, ALL_SUBS)

    # Cell 8 — Quality: After Imputation
    quality_report(df_wide, "After Imputation")

    # Cell 9 — Check 2 plot
    plot_check2_timeseries(df_wide, ALL_SUBS, fig_dir)

    # Cell 10 — Deduplicate
    df_wide = remove_duplicates(df_wide)

    # Cell 11 — Quality + Check 3 plot
    quality_report(df_wide, "After Dedup")
    plot_check3_coverage(df_wide, fig_dir)

    # Cell 12 — Winsorize
    df_wide_before = df_wide.copy()
    df_clean       = apply_winsorization(df_wide, ALL_SUBS)

    # Cell 13 — Quality: After Winsorization
    quality_report(df_clean, "After Winsorization")

    # Cells 14-15 — Check 4 plots
    corr_clean = (
        df_clean[ALL_SUBS].corr()[TARGET]
        .drop(TARGET)
        .sort_values(ascending=False)
    )
    plot_check4_boxplots(df_wide_before, df_clean, ALL_SUBS, fig_dir)
    plot_check4_corr(corr_clean, fig_dir)

    # Cells 16-17 — Feature engineering
    df_feat = engineer_features(df_clean, ALL_SUBS, corr_clean)

    # Cell 18 — Quality: FINAL
    quality_report(df_feat, "FINAL Feature Matrix")

    # Cell 19 — corr_full (needed for save)
    num_cols  = df_feat.select_dtypes(include=np.number).columns.tolist()
    num_cols  = [c for c in num_cols if c not in ["Year", "country_code"]]
    corr_full = df_feat[num_cols].corr()[TARGET].drop(TARGET, errors="ignore").dropna()
    corr_full = corr_full.sort_values(key=abs, ascending=False)

    # Cells 19-21 — Check 5 plots
    plot_check5_top_corr(corr_full, fig_dir)
    plot_check5_heatmap(df_feat, ALL_SUBS, fig_dir)
    plot_check5_pairplot(df_feat, ALL_SUBS, fig_dir)

    # Cell 22 — RF importance → fi_cols
    fi_cols, corr_full, _ = compute_fi_cols(df_feat)

    # Cell 23 — Drop rows with lag NaN + save
    lag_cols = [c for c in df_feat.columns if "_lag" in c]
    df_model = df_feat.dropna(subset=lag_cols).reset_index(drop=True)

    # Plot split
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=False)
    axes      = axes.flatten()
    zone_c    = {"Train": "#2196F3", "Val": "#FF9800", "Test": "#F44336"}
    for i, c in enumerate(COUNTRIES):
        ax = axes[i]
        d  = df_model[df_model["Area"] == c].sort_values("Year")
        tr = d[d["Year"] <= args.train_until]
        va = d[(d["Year"] > args.train_until) & (d["Year"] <= 2020)]
        te = d[d["Year"] > 2020]
        ax.plot(tr["Year"], tr[TARGET], color=zone_c["Train"], lw=2, label="Train")
        ax.plot(va["Year"], va[TARGET], color=zone_c["Val"],   lw=2, label="Val")
        ax.plot(te["Year"], te[TARGET], color=zone_c["Test"],  lw=2, label="Test")
        ax.axvline(args.train_until + 0.5, color="grey", ls="--", lw=1)
        ax.axvline(2020 + 0.5,             color="grey", ls=":",  lw=1)
        ax.set_title(c, fontweight="bold")
        ax.set_ylabel("TWh")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_visible(False)
    fig.suptitle(f"Train / Val / Test Split — {TARGET}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "split.pdf"))

    save_outputs(df_feat, df_model, fi_cols, corr_full,
                 ALL_SUBS, FEATURES, args.output_dir, args.train_until)


if __name__ == "__main__":
    main()
