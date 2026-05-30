"""
src/02_preprocessing.py — Data Preprocessing & Feature Engineering
Ember Energy | IEEE Paper

Reads:   ../outputs/eda/ember_filtered.csv
Writes:  ../outputs/preprocessing/ember_model_ready.csv
         ../outputs/preprocessing/feature_meta.json
         ../outputs/preprocessing/figures/*.pdf
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
TARGET = "Demand"

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


def savefig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


def quality_report(df: pd.DataFrame, label: str, target: str) -> None:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    null_pct = df[num_cols].isnull().mean() * 100
    inf_cnt = np.isinf(df[num_cols].values).sum(axis=0)
    log.info("=== Quality Report: %s ===", label)
    log.info("  Shape      : %s", df.shape)
    log.info("  NaN cols   : %d", (null_pct > 0).sum())
    log.info("  Inf values : %d total", inf_cnt.sum())
    log.info("  Target NaN : %d", df[target].isnull().sum())


# ── Step 1: Pivot to wide format ──────────────────────────────────────────────
def pivot_wide(df_long: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    df_agg = df_long.groupby(["Area", "Year", "Subcategory"], as_index=False)["Value"].mean()
    df_wide = df_agg.pivot_table(
        index=["Area", "Year"], columns="Subcategory", values="Value"
    ).reset_index()
    df_wide.columns.name = None
    all_subs = [c for c in df_wide.columns if c not in ["Area", "Year"]]
    log.info("Wide format: %s | Subcategories: %d", df_wide.shape, len(all_subs))
    return df_wide, all_subs


# ── Step 2: Complete Year×Country grid + interpolation ───────────────────────
def impute_missing(df_wide: pd.DataFrame, all_subs: list) -> pd.DataFrame:
    all_years = list(range(df_wide["Year"].min(), df_wide["Year"].max() + 1))
    idx_full = pd.MultiIndex.from_product([COUNTRIES, all_years], names=["Area", "Year"])
    df_wide = df_wide.set_index(["Area", "Year"]).reindex(idx_full).reset_index()
    # Linear interpolation per country, then forward/back fill
    for col in all_subs:
        df_wide[col] = df_wide.groupby("Area")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
    remaining = df_wide[all_subs].isnull().sum().sum()
    log.info("Missing after interpolation: %d", remaining)
    return df_wide


# ── Step 3: Duplicate removal ─────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["Area", "Year"]).reset_index(drop=True)
    log.info("Duplicates removed: %d → %d", before, len(df))
    return df


# ── Step 4: IQR Winsorization per country ────────────────────────────────────
def winsorise_group(group: pd.DataFrame, cols: list, factor: float = 2.5) -> pd.DataFrame:
    g = group.copy()
    for col in cols:
        q1, q3 = g[col].quantile(0.25), g[col].quantile(0.75)
        iqr = q3 - q1
        g[col] = g[col].clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)
    return g


def apply_winsorization(df: pd.DataFrame, all_subs: list) -> pd.DataFrame:
    df_clean = (
        df.groupby("Area", group_keys=False)
        .apply(lambda g: winsorise_group(g, all_subs, factor=2.5))
        .reset_index(drop=True)
    )
    log.info("Winsorization applied (IQR factor=2.5).")
    return df_clean


# ── Step 5: Feature engineering ──────────────────────────────────────────────
def build_features(df_clean: pd.DataFrame, all_subs: list) -> tuple[pd.DataFrame, dict]:
    df_feat = df_clean.sort_values(["Area", "Year"]).reset_index(drop=True)
    features = [c for c in all_subs if c != TARGET]

    # Lags 1–3 for all subcategories
    for col in all_subs:
        for lag in [1, 2, 3]:
            df_feat[f"{col}_lag{lag}"] = df_feat.groupby("Area")[col].shift(lag)

    # Rolling 3-year moving average (shifted to avoid leakage)
    for col in all_subs:
        for w in [3, 5]:
            df_feat[f"{col}_ma{w}"] = df_feat.groupby("Area")[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )

    # Year-over-year % change
    for col in all_subs:
        df_feat[f"{col}_yoy"] = df_feat.groupby("Area")[col].pct_change() * 100

    # Cross-feature interactions (domain-aware)
    subs_lower = {c: c.lower() for c in all_subs}
    gen_cols = [c for c in all_subs if "generation" in subs_lower[c] or "fuel" in subs_lower[c]]
    co2_cols = [c for c in all_subs if "co2" in subs_lower[c] or "emission" in subs_lower[c]]

    if gen_cols and TARGET in all_subs:
        df_feat["gen_demand_ratio"] = df_feat[gen_cols[0]] / (df_feat[TARGET] + 1e-8)
    if co2_cols and TARGET in all_subs:
        df_feat["co2_intensity_demand"] = df_feat[co2_cols[0]] / (df_feat[TARGET] + 1e-8)

    log.info("Features after engineering: %s", df_feat.shape)

    # Drop rows with NaN in any lag column
    lag_cols = [c for c in df_feat.columns if "_lag" in c]
    df_model = df_feat.dropna(subset=lag_cols).reset_index(drop=True)

    # Final feature list — exclude target lags to prevent circularity
    num_cols = df_model.select_dtypes(include=np.number).columns.tolist()
    target_lag = [f"{TARGET}_lag{i}" for i in [1, 2, 3]]
    all_features = [c for c in num_cols if c != TARGET and c not in target_lag and "Year" not in c]

    # Remove features with > 80% missing (safety check)
    null_frac = df_model[all_features].isnull().mean()
    good_features = [c for c in all_features if null_frac[c] < 0.80]
    dropped = set(all_features) - set(good_features)
    if dropped:
        log.warning("Dropped %d features (>80%% missing): %s", len(dropped), list(dropped)[:5])

    meta = {
        "TARGET": TARGET,
        "ALL_SUBS": all_subs,
        "FEATURES": features,
        "all_features": good_features,
        "COUNTRIES": COUNTRIES,
        "TRAIN_END": 2016,
        "VAL_END": 2020,
        "TEST_END": 2024,
        "FORECAST_YEARS": list(range(2025, 2030)),
    }
    log.info("Final feature count: %d", len(good_features))
    return df_model, meta


# ── Quality plots ────────────────────────────────────────────────────────────
def plot_feature_corr(df_model: pd.DataFrame, all_subs: list, target: str, fig_dir: str) -> None:
    sub_cols = [c for c in all_subs if c in df_model.columns]
    corr = df_model[sub_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.4,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Subcategory Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "pre_fig_feature_corr.pdf"))


def plot_split(
    df_model: pd.DataFrame, target: str, train_end: int, val_end: int, fig_dir: str
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=False)
    axes = axes.flatten()
    zone_c = {"Train": "#2196F3", "Val": "#FF9800", "Test": "#F44336"}
    for i, c in enumerate(COUNTRIES):
        ax = axes[i]
        d = df_model[df_model["Area"] == c].sort_values("Year")
        tr = d[d["Year"] <= train_end]
        va = d[(d["Year"] > train_end) & (d["Year"] <= val_end)]
        te = d[d["Year"] > val_end]
        ax.plot(tr["Year"], tr[target], color=zone_c["Train"], lw=2, label="Train")
        ax.plot(va["Year"], va[target], color=zone_c["Val"], lw=2, label="Val")
        ax.plot(te["Year"], te[target], color=zone_c["Test"], lw=2, label="Test")
        ax.axvline(train_end + 0.5, color="grey", ls="--", lw=1)
        ax.axvline(val_end + 0.5, color="grey", ls=":", lw=1)
        ax.set_title(c, fontweight="bold")
        ax.set_ylabel("TWh")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_visible(False)
    fig.suptitle(f"Train / Val / Test Split — {target}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, os.path.join(fig_dir, "pre_fig_split.pdf"))


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ember preprocessing step")
    p.add_argument("--input_dir", default="outputs/eda", help="EDA output dir")
    p.add_argument("--output_dir", default="outputs/preprocessing", help="Output dir")
    p.add_argument("--train_until", type=int, default=2016, help="Last year of training window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df_long = pd.read_csv(os.path.join(args.input_dir, "ember_filtered.csv"))
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")

    # 1. Pivot
    df_wide, all_subs = pivot_wide(df_long)
    quality_report(df_wide, "After Pivot", TARGET)

    # 2. Impute
    df_wide = impute_missing(df_wide, all_subs)
    quality_report(df_wide, "After Imputation", TARGET)

    # 3. Deduplicate
    df_wide = remove_duplicates(df_wide)

    # 4. Winsorize
    df_clean = apply_winsorization(df_wide, all_subs)
    quality_report(df_clean, "After Winsorization", TARGET)

    # 5. Feature engineering
    df_model, meta = build_features(df_clean, all_subs)
    meta["TRAIN_END"] = args.train_until  # honour --train_until flag
    quality_report(df_model, "Final Feature Matrix", TARGET)

    # Plots
    plot_feature_corr(df_model, all_subs, TARGET, fig_dir)
    plot_split(df_model, TARGET, meta["TRAIN_END"], meta["VAL_END"], fig_dir)

    # Save
    df_model.to_csv(os.path.join(args.output_dir, "ember_model_ready.csv"), index=False)
    with open(os.path.join(args.output_dir, "feature_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info("=== Preprocessing Complete ===")
    log.info("  ember_model_ready.csv : %s", df_model.shape)
    log.info("  Features              : %d", len(meta["all_features"]))
    log.info("  Saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
