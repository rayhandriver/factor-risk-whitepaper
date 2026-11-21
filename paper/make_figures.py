"""Build the whitepaper figures from the study's saved result files.

The result files (daily portfolio returns, bootstrap draws, experiment metrics) are not
part of this repository. Point RESULTS_DIR at them to regenerate the figures:

    RESULTS_DIR=/path/to/results python paper/make_figures.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

R = Path(os.environ["RESULTS_DIR"])
OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# --- style --------------------------------------------------------------------
FACTOR, PCA, BLEND = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1a1a1a", "#5f5e5a", "#e4e3df"

for f in Path(os.environ.get("FONT_DIR", "")).glob("XCharter-*.otf") if os.environ.get("FONT_DIR") else []:
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": ["XCharter", "DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "text.color": INK, "legend.frameon": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "pdf.fonttype": 42,
})
W = 6.3  # inches, text width of the paper


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220, facecolor="white")
    plt.close(fig)


def ci_row(ax, y, point, lo, hi, color):
    ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
    ax.plot(point, y, "o", color=color, ms=6, mec="white", mew=1.2, zorder=3)


# --- 1. rolling realized volatility of the minimum-variance portfolio -----------
def fig_rolling_vol():
    ts = pd.read_parquet(R / "race_timeseries.parquet")
    vol = ts[["gmv_f", "gmv_p"]].rolling(63).std().dropna() * np.sqrt(252) * 100
    gap = vol["gmv_f"] - vol["gmv_p"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 3.9), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.16})
    a.plot(vol.index, vol["gmv_f"], color=FACTOR, lw=1.4, label="Factor model")
    a.plot(vol.index, vol["gmv_p"], color=PCA, lw=1.4, label="PCA model")
    a.set_ylabel("Realized volatility (%, annualized)")
    a.legend(loc="upper right", ncol=2, handlelength=1.4)
    a.grid(axis="x", visible=False)
    b.fill_between(gap.index, gap, 0, where=gap >= 0, color=PCA, alpha=0.55, lw=0)
    b.fill_between(gap.index, gap, 0, where=gap < 0, color=FACTOR, alpha=0.55, lw=0)
    b.axhline(0, color=MUTED, lw=0.6)
    b.set_ylabel("Factor minus PCA\n(pct. points)")
    b.grid(axis="x", visible=False)
    b.text(0.01, 0.92, "Above zero: PCA portfolio is less volatile",
           transform=b.transAxes, va="top", color=MUTED, fontsize=8)
    save(fig, "rolling_volatility")


# --- 2. how concentrated each model's risk is -----------------------------------
def fig_spectrum():
    sp = pd.read_parquet(R / "race_spectra.parquet").head(20)
    fig, ax = plt.subplots(figsize=(W * 0.78, 3.0))
    ax.plot(sp["rank"], sp["pca_cumvar"] * 100, "-o", color=PCA, lw=1.6, ms=3.5, label="PCA model")
    ax.plot(sp["rank"], sp["factor_cumvar"] * 100, "-o", color=FACTOR, lw=1.6, ms=3.5, label="Factor model")
    p10, f10 = sp["pca_cumvar"].iloc[9] * 100, sp["factor_cumvar"].iloc[9] * 100
    ax.annotate(f"{p10:.1f}%", (10, p10), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5)
    ax.annotate(f"{f10:.1f}%", (10, f10), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8.5)
    ax.axvline(10, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("Number of largest risk drivers included")
    ax.set_ylabel("Share of total variance explained (%)")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(38, 80)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    save(fig, "variance_explained")


# --- 3. where the winner flips: universe size ------------------------------------
def fig_crossover():
    m = json.load(open(R / "crossover" / "metrics.json"))["per_topk"]
    ks = ["15", "50", "100", "250", "500"]
    fig, ax = plt.subplots(figsize=(W * 0.78, 3.0))
    for i, k in enumerate(ks):
        base = m[k]["metrics"]["pca"]["track_realized_te_pct"]
        pt = m[k]["te_diff_pct"] / base * 100
        lo, hi = (v / base * 100 for v in m[k]["te_ci95_pct"])
        color = FACTOR if hi < 0 else PCA if lo > 0 else MUTED
        ax.plot([i, i], [lo, hi], color=color, lw=2, solid_capstyle="round")
        ax.plot(i, pt, "o", color=color, ms=6.5, mec="white", mew=1.2, zorder=3)
        ax.annotate("0.0%" if abs(pt) < 0.05 else f"{pt:+.1f}%", (i, pt), xytext=(9, 5 if abs(pt) < 0.05 else -3), textcoords="offset points", fontsize=8.5)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xticks(range(len(ks)), [f"Top {k}" for k in ks])
    ax.set_xlim(-0.5, len(ks) - 0.3)
    ax.set_xlabel("Stocks in the universe (largest S&P 500 names by weight)")
    ax.set_ylabel("Tracking-error gap, factor vs. PCA\n(% of PCA's tracking error)")
    ax.grid(axis="x", visible=False)
    ax.text(0.98, 0.06, "Below zero: factor model tracks tighter", transform=ax.transAxes,
            ha="right", color=FACTOR, fontsize=8.5)
    ax.text(0.03, 0.93, "Above zero: PCA model tracks tighter", transform=ax.transAxes,
            ha="left", color=PCA, fontsize=8.5)
    save(fig, "universe_crossover")


# --- 4. is the blend really better? ----------------------------------------------
def fig_blend():
    s = json.load(open(R / "hybrid_deep" / "metrics.json"))["significance"]
    rows = [("25% factor blend vs. PCA", "a025_minus_pca"),
            ("50% factor blend vs. PCA", "a050_minus_pca"),
            ("25% factor blend vs. factor", "a025_minus_factor"),
            ("50% factor blend vs. factor", "a050_minus_factor")]
    fig, ax = plt.subplots(figsize=(W * 0.82, 2.3))
    for i, (label, key) in enumerate(rows):
        lo, hi = s[key]["track_te_ci95_pct"]
        sig = hi < 0 or lo > 0
        ci_row(ax, -i, s[key]["track_te_diff_pct"], lo, hi, BLEND if sig else MUTED)
        ax.text(0.03, -i, "significant" if sig else "not significant",
                va="center", fontsize=8, color=MUTED)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks([-i for i in range(len(rows))], [r[0] for r in rows])
    ax.tick_params(axis="y", colors=INK)
    ax.set_ylim(-len(rows) + 0.4, 0.6)
    ax.set_xlim(-0.115, 0.075)
    ax.set_xticks([-0.10, -0.05, 0, 0.05])
    ax.set_xlabel("Difference in tracking error (percentage points)")
    ax.grid(axis="y", visible=False)
    save(fig, "blend_significance")


# --- 5. tax-loss harvesting: how often is the replacement in the same sector? ----
def fig_tlh():
    m = json.load(open(R / "tlh_swap" / "metrics.json"))["metrics"]
    rows = [("PCA model", "pca", PCA), ("Blend,\n25% factor", "blend_a025", BLEND),
            ("Blend,\n50% factor", "blend_a050", BLEND), ("Factor model", "factor", FACTOR)]
    fig, ax = plt.subplots(figsize=(W * 0.66, 2.8))
    for i, (label, key, color) in enumerate(rows):
        v = m[key]["sector_match_rate"] * 100
        ax.bar(i, v, width=0.5, color=color)
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9.5)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="x", colors=INK)
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("Replacements in the same sector (%)")
    ax.grid(axis="x", visible=False)
    save(fig, "tlh_sector_match")


if __name__ == "__main__":
    for f in (fig_rolling_vol, fig_spectrum, fig_crossover, fig_blend, fig_tlh):
        f()
    print("wrote figures to", OUT)
