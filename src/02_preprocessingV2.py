"""
src/02_preprocessing.py — Data Preprocessing & Feature Engineering
Ember Energy | IEEE Paper

Mirrors exactly: notebooks/01_preprocessing_patched.ipynb

Reads:   outputs/eda/ember_filtered.csv
Writes:  outputs/preprocessing/ember_model_ready.csv
         outputs/preprocessing/ember_multifeature.csv
         outputs/preprocessing/feature_meta.json
         outputs/preprocessing/figures/*.pdf

Key design decisions (from notebook):
  - Subcategory columns renamed: spaces → _, slashes → _
  - All 7 subcategories pivoted wide as features
  - Demand is target; other subcategories are exogenous features
  - Lags 1-3, MA-3, MA-5, YoY for all subcategories
  - Cross-feature interactions: generation shares + demand ratios
  - Trend + trend² + country_code
  - fi_cols excludes Demand_lag* and Demand_ma* (prevent circularity)
  - feature_meta has both lowercase (notebook) and uppercase (script) keys
"""

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

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
TARGET    = "Demand"

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


def savefig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


def quality_report(df: pd.DataFrame, label: str) -> None:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    null_pct = df[num_cols].isnull().mean() * 100
    inf_cnt  = np.isinf(df[num_cols].values).sum()
    log.info("=== Quality Report: %s ===", label)
    log.info("  Shape      : %s", df.shape)
    log.info("  NaN cols   : %d", (null_pct > 0).sum())
    log.info("  Inf values : %d total", inf_cnt)
    log.info("  Target NaN : %d", df[TARGET].isnull().sum() if TARGET in df.columns else -1)


# Step 0 — Load filtered CSV
def load_filtered(input_dir: str) -> pd.DataFrame:
    path = os.path.join(input_dir, "ember_filtered.csv")
    df   = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Year"]  = df["Year"].astype(int)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df[df["Area"].isin(COUNTRIES)][
        ["Area", "Year", "Subcategory", "Unit", "Value"]
    ].copy()
    log.info("Loaded: %s | Years %d–%d | Countries: %d | Subcats: %d",
             df.shape, df["Year"].min(), df["Year"].max(),
             df["Area"].nunique(), df["Subcategory"].nunique())
    return df


# Step 1 — Pivot wide  (notebook cell 1)
def pivot_wide(df_long: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    df_agg = (
        df_long
        .groupby(["Area", "Year", "Subcategory"], as_index=False)["Value"]
        .mean()
    )
    df_wide = df_agg.pivot_table(
        index=["Area", "Year"],
        columns="Subcategory",
        values="Value",
    ).reset_index()
    df_wide.columns.name = None

    # Rename columns: spaces → _, slashes → _  (CRITICAL — matches notebook)
    df_wide.columns = [
        c.strip().replace(" ", "_").replace("/", "_")
        for c in df_wide.columns
    ]

    ALL_SUBS = [c for c in df_wide.columns if c not in ["Area", "Year"]]
    FEATURES = [c for c in ALL_SUBS if c != TARGET]
    log.info("Wide: %s | ALL_SUBS: %s", df_wide.shape, ALL_SUBS)
    return df_wide, ALL_SUBS, FEATURES


# Step 2 — Impute  (notebook cell 2)
def impute_missing(df_wide: pd.DataFrame, all_subs: list) -> pd.DataFrame:
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


# Step 3 — Deduplicate  (notebook cell 3)
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["Area", "Year"]).reset_index(drop=True)
    log.info("Duplicates removed: %d → %d rows", before, len(df))
    return df


# Step 4 — IQR Winsorization  (notebook cell 4)
def winsorise_group(group: pd.DataFrame, cols: list, factor: float = 2.5) -> pd.DataFrame:
    g = group.copy()
    for col in cols:
        q1, q3 = g[col].quantile(0.25), g[col].quantile(0.75)
        iqr    = q3 - q1
        g[col] = g[col].clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)
    return g


def apply_winsorization(df: pd.DataFrame, all_subs: list) -> pd.DataFrame:
    df_clean = (
        df.groupby("Area", group_keys=False)
        .apply(lambda g: winsorise_group(g, all_subs, factor=2.5))
        .reset_index(drop=True)
    )
    diff = (df[all_subs] - df_clean[all_subs]).abs()
    log.info("Winsorization applied. Cells clipped: %d", (diff > 0).sum().sum())
    return df_clean


# Step 5 — Feature engineering  (notebook cell 5)
def build_features(
    df_clean: pd.DataFrame,
    all_subs: list,
    features: list,
) -> tuple[pd.DataFrame, pd.DataFrame, list, pd.Series]:

    df_feat = df_clean.sort_values(["Area", "Year"]).reset_index(drop=True)

    # Lags 1–3 (all subcategories)
    for col in all_subs:
        for lag in [1, 2, 3]:
            df_feat[f"{col}_lag{lag}"] = df_feat.groupby("Area")[col].shift(lag)

    # Rolling MA-3, MA-5 (shift-1 to avoid leakage)
    for col in all_subs:
        for w in [3, 5]:
            df_feat[f"{col}_ma{w}"] = df_feat.groupby("Area")[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )

    # YoY % change
    for col in all_subs:
        df_feat[f"{col}_yoy"] = df_feat.groupby("Area")[col].pct_change() * 100

    log.info("After lags + MA + YoY: %s", df_feat.shape)

    # Cross-feature interactions (domain-aware — mirrors notebook exactly)
    subs_lower = {c: c.lower() for c in all_subs}
    gen_cols   = [c for c in all_subs if any(
        k in subs_lower[c]
        for k in ["generation", "wind", "solar", "hydro", "coal", "gas", "nuclear", "other"]
    )]

    if len(gen_cols) >= 2:
        df_feat["total_generation"] = df_feat[gen_cols].sum(axis=1)
        for c in gen_cols:
            df_feat[f"share_{c}"] = df_feat[c] / (df_feat["total_generation"] + 1e-9)
        log.info("Generation share features: %s", gen_cols)

    # Demand intensity vs top 2 correlated raw features
    raw_num  = [c for c in all_subs if c != TARGET]
    corr_raw = df_feat[raw_num + [TARGET]].corr()[TARGET].drop(TARGET).dropna()
    top2     = corr_raw.abs().nlargest(min(2, len(corr_raw))).index.tolist()
    for feat in top2:
        df_feat[f"demand_ratio_{feat}"] = df_feat[TARGET] / (df_feat[feat] + 1e-9)
        log.info("Created: demand_ratio_%s", feat)

    # Trend terms + country code
    df_feat["trend"]        = df_feat["Year"] - df_feat["Year"].min()
    df_feat["trend_sq"]     = df_feat["trend"] ** 2
    df_feat["country_code"] = df_feat["Area"].astype("category").cat.codes

    log.info("Final feature matrix: %s", df_feat.shape)

    # Drop rows where any lag is NaN (matches notebook save cell)
    lag_cols = [c for c in df_feat.columns if "_lag" in c]
    df_model = df_feat.dropna(subset=lag_cols).reset_index(drop=True)
    log.info("Model-ready (no lag NaN): %s", df_model.shape)

    # corr_full — mirrors notebook Check 5
    num_cols_all = df_feat.select_dtypes(include=np.number).columns.tolist()
    num_cols_all = [c for c in num_cols_all if c not in ["Year", "country_code"]]
    corr_full    = df_feat[num_cols_all].corr()[TARGET].drop(TARGET, errors="ignore").dropna()
    corr_full    = corr_full.sort_values(key=abs, ascending=False)

    # fi_cols — mirrors notebook: excludes Demand_lag* and Demand_ma*
    num_model = df_model.select_dtypes(include=np.number).columns.tolist()
    fi_cols   = [
        c for c in num_model
        if c != TARGET
        and not c.startswith(f"{TARGET}_lag")
        and not c.startswith(f"{TARGET}_ma")
        and c not in ["Year", "country_code"]
    ]

    # Remove features with >80% missing
    null_frac    = df_model[fi_cols].isnull().mean()
    good_fi_cols = [c for c in fi_cols if null_frac[c] < 0.80]
    dropped      = set(fi_cols) - set(good_fi_cols)
    if dropped:
        log.warning("Dropped %d features (>80%% NaN): %s", len(dropped), list(dropped)[:5])

    log.info("fi_cols: %d features", len(good_fi_cols))
    return df_feat, df_model, good_fi_cols, corr_full


# Plots
def plot_feature_corr(df_feat: pd.DataFrame, all_subs: list, fig_dir: str) -> None:
    sub_cols = [c for c in all_subs if c in df_feat.columns]
    corr     = df_feat[sub_cols].corr()
    fig, ax  = plt.subplots(figsize=(9, 7))
    mask     = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.4, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 7})
    ax.set_title("Check 5 — Subcategory Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "pre_fig_feature_corr.pdf"))


def plot_top_corr(corr_full: pd.Series, fig_dir: str) -> None:
    top25 = corr_full.head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    clrs    = ["steelblue" if v >= 0 else "tomato" for v in top25.values]
    ax.barh(top25.index[::-1], top25.values[::-1], color=clrs[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Check 5 — Top 25 Features Correlated with Demand",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson Correlation")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "pre_fig_top_corr.pdf"))


def plot_missing_heatmap(df_wide: pd.DataFrame, all_subs: list, fig_dir: str) -> None:
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
    savefig(fig, os.path.join(fig_dir, "pre_fig_missing.pdf"))


def plot_split(
    df_model: pd.DataFrame, train_end: int, val_end: int, fig_dir: str
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=False)
    axes      = axes.flatten()
    zone_c    = {"Train": "#2196F3", "Val": "#FF9800", "Test": "#F44336"}
    for i, c in enumerate(COUNTRIES):
        ax = axes[i]
        d  = df_model[df_model["Area"] == c].sort_values("Year")
        tr = d[d["Year"] <= train_end]
        va = d[(d["Year"] > train_end) & (d["Year"] <= val_end)]
        te = d[d["Year"] > val_end]
        ax.plot(tr["Year"], tr[TARGET], color=zone_c["Train"], lw=2, label="Train")
        ax.plot(va["Year"], va[TARGET], color=zone_c["Val"],   lw=2, label="Val")
        ax.plot(te["Year"], te[TARGET], color=zone_c["Test"],  lw=2, label="Test")
        ax.axvline(train_end + 0.5, color="grey", ls="--", lw=1)
        ax.axvline(val_end   + 0.5, color="grey", ls=":",  lw=1)
        ax.set_title(c, fontweight="bold")
        ax.set_ylabel("TWh")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_visible(False)
    fig.suptitle(f"Train / Val / Test Split — {TARGET}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "pre_fig_split.pdf"))


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember preprocessing step")
    p.add_argument("--input_dir",   default="outputs/eda",          help="EDA output dir")
    p.add_argument("--output_dir",  default="outputs/preprocessing", help="Output dir")
    p.add_argument("--train_until", type=int, default=2016,          help="Last training year")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir,         exist_ok=True)

    # 0. Load
    df_long = load_filtered(args.input_dir)

    # 1. Pivot
    df_wide, ALL_SUBS, FEATURES = pivot_wide(df_long)
    quality_report(df_wide, "After Pivot")
    plot_missing_heatmap(df_wide, ALL_SUBS, fig_dir)

    # 2. Impute
    df_wide = impute_missing(df_wide, ALL_SUBS)
    quality_report(df_wide, "After Imputation")

    # 3. Deduplicate
    df_wide = remove_duplicates(df_wide)

    # 4. Winsorize
    df_clean = apply_winsorization(df_wide, ALL_SUBS)
    quality_report(df_clean, "After Winsorization")

    # 5. Feature engineering
    df_feat, df_model, fi_cols, corr_full = build_features(
        df_clean, ALL_SUBS, FEATURES
    )
    quality_report(df_feat,  "Full Feature Matrix")
    quality_report(df_model, "Model-Ready Matrix")

    # Plots
    plot_feature_corr(df_feat, ALL_SUBS, fig_dir)
    plot_top_corr(corr_full, fig_dir)
    plot_split(df_model, args.train_until, 2020, fig_dir)

    # 6. Save — mirrors notebook section 6 exactly
    df_feat.to_csv(
        os.path.join(args.output_dir, "ember_multifeature.csv"), index=False)
    df_model.to_csv(
        os.path.join(args.output_dir, "ember_model_ready.csv"),  index=False)

    top_corr_20  = corr_full.head(20).index.tolist()
    feature_meta = {
        # Notebook-compatible lowercase keys
        "target":        TARGET,
        "all_subs":      ALL_SUBS,
        "raw_features":  FEATURES,
        "all_features":  fi_cols,
        "top_corr_20":   top_corr_20,
        # Script-compatible uppercase keys (used by 03_modeling.py etc.)
        "TARGET":        TARGET,
        "ALL_SUBS":      ALL_SUBS,
        "FEATURES":      FEATURES,
        "COUNTRIES":     COUNTRIES,
        "TRAIN_END":     args.train_until,
        "VAL_END":       2020,
        "TEST_END":      2024,
        "FORECAST_YEARS": list(range(2025, 2031)),
    }

    with open(os.path.join(args.output_dir, "feature_meta.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)

    log.info("=== Preprocessing Complete ===")
    log.info("  ember_multifeature.csv : %s", df_feat.shape)
    log.info("  ember_model_ready.csv  : %s", df_model.shape)
    log.info("  fi_cols count          : %d", len(fi_cols))
    log.info("  top_corr_20            : %s", top_corr_20[:5])
    log.info("  Saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
