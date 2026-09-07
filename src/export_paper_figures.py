"""
src/export_paper_figures.py
Exports all pipeline figures in IEEE paper format:
  - 300 DPI minimum
  - Times New Roman font
  - Column width: 3.5 inches (single) or 7.16 inches (double)
  - PDF vector format
                              LaTeX usage
        begin{figure}[t]
        entering
        includegraphics[width=columnwidth]{fig1_demand_history}
        caption{Annual electricity demand 2000--2024 for seven countries.
                Vertical dashed lines mark COVID-19 (2020) and the Ukraine
                energy crisis (2022).}
        label{fig:history}
        end{figure}
"""

import os
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "lines.linewidth": 1.5,
    }
)

IEEE_SINGLE_COL = 3.5  # inches — single column
IEEE_DOUBLE_COL = 7.16  # inches — double column
OUT_DIR = "outputs/paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

COUNTRIES = ["Tunisia", "Austria", "Germany", "Egypt", "Canada", "France", "Kuwait"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def fig1_demand_history() -> None:
    """Figure 1: Historical demand 2000-2024 per country (double column)."""
    df = pd.read_csv("outputs/preprocessing/ember_model_ready.csv")
    fig, axes = plt.subplots(2, 4, figsize=(IEEE_DOUBLE_COL, 4.5), sharey=False)
    axes = axes.flatten()
    for i, country in enumerate(COUNTRIES):
        ax = axes[i]
        sub = df[df["Area"] == country].sort_values("Year")
        ax.plot(sub["Year"], sub["Demand"], color=COLORS[i], lw=1.5)
        ax.fill_between(sub["Year"], sub["Demand"], alpha=0.08, color=COLORS[i])
        ax.axvline(2020, color="red", ls=":", lw=0.8, alpha=0.6)
        ax.axvline(2022, color="orange", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(country, fontsize=9, fontweight="bold")
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("TWh", fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    axes[-1].set_visible(False)
    fig.suptitle("Annual Electricity Demand 2000–2024", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig1_demand_history.pdf")
    plt.close(fig)
    print("✓ fig1_demand_history.pdf")


def fig2_mape_heatmap() -> None:
    """Figure 2: MAPE heatmap all models X all countries (single column)."""
    df = pd.read_csv("outputs/modeling/test_benchmarking.csv")
    pivot = df.pivot(index="Country", columns="Model", values="MAPE").round(2)
    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL, 2.8))
    import seaborn as sns

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        ax=ax,
        linewidths=0.4,
        cbar_kws={"label": "MAPE (%)"},
    )
    ax.set_title("Test MAPE (%) — All Models × All Countries (2021–2024)", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig2_mape_heatmap.pdf")
    plt.close(fig)
    print("✓ fig2_mape_heatmap.pdf")


def fig3_forecast_comparison() -> None:
    """Figure 3: Classical vs DL forecast 2025-2030 (double column)."""
    fc_cl = pd.read_csv("outputs/forecasting/demand_forecast_2025_2030.csv")
    fc_dl = pd.read_csv("outputs/deeplearning/dl_forecast_2025_2030.csv")
    df = pd.read_csv("outputs/preprocessing/ember_model_ready.csv")

    fig, axes = plt.subplots(2, 4, figsize=(IEEE_DOUBLE_COL, 5.5))
    axes = axes.flatten()

    for i, country in enumerate(COUNTRIES):
        ax = axes[i]
        hist = df[df["Area"] == country].sort_values("Year")
        cl = fc_cl[fc_cl["Country"] == country].sort_values("Year")
        dl = fc_dl[fc_dl["Country"] == country].sort_values("Year")

        ax.plot(hist["Year"], hist["Demand"], color="#555", lw=1.2, label="Historical")
        ax.plot(
            cl["Year"],
            cl["Forecast"],
            color="#1f77b4",
            lw=1.5,
            ls="--",
            marker="o",
            ms=3,
            label="Classical",
        )
        if "Forecast" in dl.columns:
            ax.plot(
                dl["Year"],
                dl["Forecast"],
                color="#2ca02c",
                lw=1.5,
                ls="--",
                marker="s",
                ms=3,
                label="DL",
            )
        if "Lower_90" in cl.columns:
            ax.fill_between(cl["Year"], cl["Lower_90"], cl["Upper_90"], alpha=0.12, color="#1f77b4")
        ax.axvline(2024, color="gray", ls=":", lw=0.8)
        ax.set_title(country, fontsize=9, fontweight="bold")
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("TWh", fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
        if i == 0:
            ax.legend(fontsize=6, loc="upper left")

    axes[-1].set_visible(False)
    fig.suptitle(
        "Electricity Demand Forecast 2025–2030: Classical vs Deep Learning",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_forecast_comparison.pdf")
    plt.close(fig)
    print("✓ fig3_forecast_comparison.pdf")


def fig4_cagr_bar() -> None:
    """Figure 4: CAGR 2024-2030 bar chart (single column)."""
    df = pd.read_csv("outputs/forecasting/demand_growth_summary.csv")
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.8))
    bars = ax.bar(
        df["Country"], df["CAGR 24-30 %"], color=COLORS[: len(df)], edgecolor="white", linewidth=0.5
    )
    for bar, val in zip(bars, df["CAGR 24-30 %"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Demand CAGR 2024–2030 by Country", fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.axhline(0, color="black", lw=0.5)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_cagr_bar.pdf")
    plt.close(fig)
    print("✓ fig4_cagr_bar.pdf")


def fig5_classical_vs_dl_scatter() -> None:
    """Figure 5: Classical vs DL MAPE scatter (single column)."""
    cl = pd.read_csv("outputs/modeling/test_benchmarking.csv")
    dl = pd.read_csv("outputs/deeplearning/dl_benchmarking.csv")

    cl_best = cl.loc[cl.groupby("Country")["MAPE"].idxmin()][["Country", "MAPE"]].rename(
        columns={"MAPE": "Classic_MAPE"}
    )
    dl_best = dl.loc[dl.groupby("Country")["MAPE"].idxmin()][["Country", "MAPE"]].rename(
        columns={"MAPE": "DL_MAPE"}
    )
    merged = cl_best.merge(dl_best, on="Country")

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.0))
    for i, row in merged.iterrows():
        ax.scatter(
            row["Classic_MAPE"],
            row["DL_MAPE"],
            color=COLORS[cast(int, i) % len(COLORS)],
            s=60,
            zorder=3,
        )
        ax.annotate(
            row["Country"],
            (row["Classic_MAPE"], row["DL_MAPE"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=6,
        )

    lim = max(merged["Classic_MAPE"].max(), merged["DL_MAPE"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4, label="Equal performance")
    ax.set_xlabel("Classical MAPE (%)")
    ax.set_ylabel("DL MAPE (%)")
    ax.set_title("Classical vs DL Test MAPE", fontweight="bold")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig5_classical_vs_dl.pdf")
    plt.close(fig)
    print("✓ fig5_classical_vs_dl.pdf")


if __name__ == "__main__":
    print("Generating IEEE paper figures...")
    fig1_demand_history()
    fig2_mape_heatmap()
    fig3_forecast_comparison()
    fig4_cagr_bar()
    fig5_classical_vs_dl_scatter()
    print(f"\n✓ All figures saved to {OUT_DIR}/")
    print("  Ready for LaTeX: \\includegraphics[width=\\columnwidth]{fig1_demand_history}")
