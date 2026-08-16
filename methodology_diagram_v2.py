"""
methodology_diagram_v2.py — Pipeline architecture figure, icon edition
==========================================================================
Restyled version: modern flat-icon badges per block, softer color palette,
no longer matching the reference figure's style. Icons combine confirmed-
safe Unicode symbols (tested: gear/balance/refresh/check render cleanly in
this environment's font) with simple hand-drawn vector icons for concepts
without a reliable safe symbol (database, bar-chart, funnel, branch,
neural-net, trend-line) — avoiding emoji, which are NOT supported here.

Usage: python methodology_diagram_v2.py
Output: methodology_diagram_v2.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Polygon
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})

fig, ax = plt.subplots(figsize=(20, 11))
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.axis("off")

# ── Palette ──────────────────────────────────────────────────────────────
INK       = "#1f2430"
BG_SOURCE = "#eef4ea"
BG_STEP   = "#eef1f7"
BG_MODEL  = "#fdf3e2"
BG_ACC    = "#e3edfa"
BG_ROB    = "#fbe9df"
BG_REFIT  = "#f0e9f7"
BG_FC     = "#eef1f7"
ED_SOURCE = "#4c7a3f"
ED_STEP   = "#4a5578"
ED_MODEL  = "#a9772e"
ED_ACC    = "#2d5ea6"
ED_ROB    = "#c1541a"
ED_REFIT  = "#6a4a99"
ED_FC     = "#4a5578"


def box(x, y, w, h, text, fc, ec, fs=9.3, weight="normal", lw=1.8, zorder=3, text_dy=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder))
    ax.text(x + w / 2, y + h / 2 + text_dy, text, ha="center", va="center",
             fontsize=fs, fontweight=weight, color=INK, zorder=zorder + 1, linespacing=1.4)


def panel(x, y, w, h, fc, ec, lw=2.0, zorder=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.14",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder))


def arrow(x1, y1, x2, y2, lw=3.0, color="black"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=18, linewidth=lw, color=color, zorder=1))


def badge(cx, cy, r, fc, ec):
    """Circular icon background badge."""
    ax.add_patch(Circle((cx, cy), r, facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=4))


def icon_symbol(cx, cy, char, fc="white", ec="#333", size=0.32, fontsize=15, color="black"):
    badge(cx, cy, size, fc, ec)
    ax.text(cx, cy, char, ha="center", va="center", fontsize=fontsize, zorder=5, color=color,
            fontfamily="DejaVu Sans")


def icon_database(cx, cy, r=0.3, ec="#333"):
    badge(cx, cy, r, "white", ec)
    w, h = r * 1.1, r * 0.9
    ax.add_patch(Ellipse((cx, cy + h / 2), w, h * 0.4, facecolor="white", edgecolor=ec, lw=1.1, zorder=5))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="square,pad=0",
                 facecolor="white", edgecolor="none", zorder=4.5))
    ax.plot([cx - w / 2, cx - w / 2], [cy - h / 2, cy + h / 2], color=ec, lw=1.1, zorder=5)
    ax.plot([cx + w / 2, cx + w / 2], [cy - h / 2, cy + h / 2], color=ec, lw=1.1, zorder=5)
    ax.add_patch(Ellipse((cx, cy - h / 2), w, h * 0.4, facecolor="none", edgecolor=ec, lw=1.1, zorder=5))


def icon_barchart(cx, cy, r=0.3, ec="#333"):
    badge(cx, cy, r, "white", ec)
    heights = [0.10, 0.18, 0.14]
    xs = [cx - 0.15, cx, cx + 0.15]
    for x, h in zip(xs, heights):
        ax.add_patch(FancyBboxPatch((x - 0.045, cy - 0.18), 0.09, h, boxstyle="square,pad=0",
                     facecolor=ED_STEP, edgecolor="none", zorder=5))


def icon_funnel(cx, cy, r=0.3, ec="#333"):
    badge(cx, cy, r, "white", ec)
    pts = [(cx - 0.16, cy + 0.14), (cx + 0.16, cy + 0.14), (cx + 0.04, cy - 0.02), (cx + 0.04, cy - 0.16),
           (cx - 0.04, cy - 0.16), (cx - 0.04, cy - 0.02)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=ED_STEP, edgecolor="none", zorder=5))


def icon_branch(cx, cy, r=0.3, ec="#333"):
    badge(cx, cy, r, "white", ec)
    ax.plot([cx - 0.16, cx], [cy, cy], color=ED_STEP, lw=2, zorder=5)
    for dy in (-0.13, 0, 0.13):
        ax.plot([cx, cx + 0.16], [cy, cy + dy], color=ED_STEP, lw=2, zorder=5)


def icon_neuralnet(cx, cy, r=0.3, ec="#333"):
    badge(cx, cy, r, "white", ec)
    layer1 = [(cx - 0.14, cy + 0.12), (cx - 0.14, cy), (cx - 0.14, cy - 0.12)]
    layer2 = [(cx + 0.02, cy + 0.16), (cx + 0.02, cy), (cx + 0.02, cy - 0.16)]
    layer3 = [(cx + 0.16, cy + 0.08), (cx + 0.16, cy - 0.08)]
    for a in layer1:
        for b in layer2:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#bbb", lw=0.5, zorder=4.7)
    for a in layer2:
        for b in layer3:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#bbb", lw=0.5, zorder=4.7)
    for layer, c in [(layer1, ED_MODEL), (layer2, ED_MODEL), (layer3, ED_MODEL)]:
        for p in layer:
            ax.add_patch(Circle(p, 0.03, facecolor=c, edgecolor="none", zorder=5))


def icon_trendup(cx, cy, r=0.3, ec="#333"):
    badge(cx, cy, r, "white", ec)
    xs = [cx - 0.17, cx - 0.06, cx + 0.03, cx + 0.17]
    ys = [cy - 0.12, cy - 0.02, cy - 0.06, cy + 0.14]
    ax.plot(xs, ys, color=ED_FC, lw=2, zorder=5)
    ax.add_patch(Polygon([(xs[-1] - 0.05, ys[-1] - 0.01), (xs[-1] + 0.02, ys[-1] + 0.05),
                           (xs[-1] - 0.01, ys[-1] - 0.07)], closed=True, facecolor=ED_FC, zorder=5))


# ══════════════════════════════════════════════════════════════════════════
# DATA SOURCE PANEL
# ══════════════════════════════════════════════════════════════════════════
panel(5.6, 7.3, 11.1, 3.3, BG_SOURCE, ED_SOURCE)
icon_database(6.35, 10.25, 0.32, ED_SOURCE)
ax.text(11.15, 10.35, "Data Source & Extraction", ha="center", fontsize=13.5, fontweight="bold")

box(5.9, 8.75, 3.6, 1.4, "Ember Global Electricity\nDataset (2000\u20132024)\n7 countries", "white", "#888", fs=10)
box(10.1, 8.75, 6.3, 1.4,
    "5 retained subcategories:\nDemand \u00b7 CO2 intensity \u00b7 Demand per capita\nElectricity imports \u00b7 Fuel",
    "white", "#888", fs=9.5)
ax.text(13.25, 8.6, "(Total / Aggregate fuel excluded \u2014 collinearity)",
        ha="center", fontsize=7.8, style="italic", color="#555")
box(6.0, 7.55, 10.4, 0.95,
    "Long-format extraction (Area, Year, Subcategory, Unit, Value)  \u2192  filtered to 7 target countries",
    "white", "#888", fs=9)

arrow(5.9, 9.45, 4.5, 9.45)

# ══════════════════════════════════════════════════════════════════════════
# LEFT COLUMN: EDA -> Preprocessing -> Train/Val/Test
# ══════════════════════════════════════════════════════════════════════════
box(0.3, 8.75, 4.0, 1.4, "Data Visualization (EDA)\nTrend \u00b7 YoY growth \u00b7 rolling CAGR\nADF stationarity test",
    BG_STEP, ED_STEP, fs=9.3, text_dy=-0.08)
icon_barchart(1.1, 9.85, 0.3, ED_STEP)
arrow(2.3, 8.75, 2.3, 7.55, color=ED_STEP)

box(0.3, 6.15, 4.0, 1.4, "Data Preprocessing\nLag (1\u20133yr) \u00b7 moving-avg (3/5yr) \u00b7\nYoY features; drop incomplete rows",
    BG_STEP, ED_STEP, fs=9.3, text_dy=-0.08)
icon_funnel(1.1, 7.25, 0.3, ED_STEP)
arrow(2.3, 6.15, 2.3, 4.95, color=ED_STEP)

box(0.3, 3.55, 4.0, 1.4, "Data Preparation\nTrain (\u22642016) \u00b7 Val (2017\u201320) \u00b7\nTest (2021\u201324), walk-forward split",
    BG_STEP, ED_STEP, fs=9.3, text_dy=-0.08)
icon_branch(1.1, 4.65, 0.3, ED_STEP)
arrow(2.3, 3.55, 2.3, 2.4, color=ED_STEP)

# ══════════════════════════════════════════════════════════════════════════
# MODEL DEVELOPMENT
# ══════════════════════════════════════════════════════════════════════════
box(0.3, 0.35, 4.0, 2.0, "", BG_MODEL, ED_MODEL)
icon_symbol(1.05, 2.13, "\u2699", "white", ED_MODEL, size=0.26, fontsize=15)
ax.text(2.55, 2.13, "Model Development", ha="center", fontsize=9.5, fontweight="bold")
ax.text(2.3, 1.9, "Walk-forward training \u00b7 3-fold\nTimeSeriesSplit tuning", ha="center", fontsize=7.5, linespacing=1.3)

box(0.45, 1.15, 3.7, 0.42, "Naive \u00b7 LinearTrend \u00b7 Holt \u00b7 DampedHolt \u00b7 Theta", "white", ED_MODEL, fs=7.2)
box(0.45, 0.68, 3.7, 0.42, "ARIMA \u00b7 SARIMA \u00b7 Ridge \u00b7 ElasticNet \u00b7 BayesianRidge", "white", ED_MODEL, fs=7.2)
box(0.45, 0.42, 3.7, 0.21, "RandomForest \u00b7 XGBoost", "white", ED_MODEL, fs=7.5)

arrow(4.3, 1.35, 5.15, 1.35)

# ══════════════════════════════════════════════════════════════════════════
# DEEP LEARNING
# ══════════════════════════════════════════════════════════════════════════
box(5.3, 0.35, 2.5, 2.0, "", BG_MODEL, ED_MODEL)
icon_neuralnet(6.55, 1.95, 0.28, ED_MODEL)
ax.text(6.55, 1.4, "Deep Learning", ha="center", fontsize=9, fontweight="bold")
ax.text(6.55, 1.0, "MLP \u00b7 TCN\nN-BEATS \u00b7 TFT\n\n(single-step,\nsliding window)", ha="center", fontsize=7.8, linespacing=1.3)

arrow(7.9, 1.35, 8.75, 1.35)

# ══════════════════════════════════════════════════════════════════════════
# DUAL-AXIS EVALUATION
# ══════════════════════════════════════════════════════════════════════════
panel(8.9, 0.2, 5.4, 2.3, "#f7f7fa", "#666")
ax.text(11.6, 2.32, "Dual-Axis Model Evaluation", ha="center", fontsize=10.5, fontweight="bold")

box(9.05, 1.15, 2.55, 1.05, "", BG_ACC, ED_ACC)
icon_symbol(9.5, 1.98, "\u2713", "white", ED_ACC, size=0.22, fontsize=13, color=ED_ACC)
ax.text(10.4, 1.98, "ACCURACY", ha="center", fontsize=8.3, fontweight="bold")
ax.text(10.3, 1.5, "MAPE (Test)\nDM + Wilcoxon\nsignificance", ha="center", fontsize=7.4, linespacing=1.3)

box(11.75, 1.15, 2.4, 1.05, "", BG_ROB, ED_ROB)
icon_symbol(12.15, 1.98, "\u2696", "white", ED_ROB, size=0.22, fontsize=13, color=ED_ROB)
ax.text(13.0, 1.98, "ROBUSTNESS", ha="center", fontsize=7.8, fontweight="bold")
ax.text(12.95, 1.48, "Max deviation \u00b7 Max YoY\ndirection flips \u2192\nSTABLE/WATCH/UNSTABLE", ha="center", fontsize=6.8, linespacing=1.3)

ax.text(11.6, 0.55, "Ground-truth-free \u2014 computed independently of accuracy",
        ha="center", fontsize=7.8, style="italic", color="#444")

arrow(14.3, 1.35, 15.05, 1.35)

# ══════════════════════════════════════════════════════════════════════════
# MODEL REFITTING
# ══════════════════════════════════════════════════════════════════════════
box(15.15, 0.35, 2.1, 2.0, "", BG_REFIT, ED_REFIT)
icon_symbol(16.2, 2.05, "\u21bb", "white", ED_REFIT, size=0.22, fontsize=13, color=ED_REFIT)
ax.text(16.2, 1.55, "Final Model\nRefitting", ha="center", fontsize=8.6, fontweight="bold", linespacing=1.3)
ax.text(16.2, 0.9, "Winning model\nrefit on FULL\n2000\u20132024\nhistory", ha="center", fontsize=7.4, linespacing=1.3)
ax.text(16.2, 2.55, "(evaluation used\nTrain-only fit)", ha="center", fontsize=6.8, style="italic", color="#555", linespacing=1.2)

arrow(17.25, 1.35, 18.0, 1.35)

# ══════════════════════════════════════════════════════════════════════════
# FORECAST
# ══════════════════════════════════════════════════════════════════════════
box(18.1, 0.35, 1.65, 2.0, "", BG_FC, ED_FC)
icon_trendup(18.93, 2.0, 0.28, ED_FC)
ax.text(18.93, 1.35, "Model\nForecast", ha="center", fontsize=8.2, fontweight="bold", linespacing=1.3)
ax.text(18.93, 0.75, "2025\u20132030\nrecursive +\nbootstrap CI\n+ IEA compare", ha="center", fontsize=6.9, linespacing=1.3)

plt.tight_layout()
fig.savefig("methodology_diagram_v2.png", dpi=300, bbox_inches="tight")
print("Saved -> methodology_diagram_v2.png")
