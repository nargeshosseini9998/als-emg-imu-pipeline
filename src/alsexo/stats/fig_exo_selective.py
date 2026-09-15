# alsexo/stats/fig_exo_selective.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 24); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.fig_exo_selective`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — Figures for Step 9.F (exoskeleton) and Step 9.H (selective involvement)
#
# Run AFTER Step 9.F (v2) and Step 9.H — they read the CSVs those steps saved.
# Saves publication-quality PNGs to /figures.
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

STATS_DIR    = PROJECT_ROOT / "data" / "processed" / "phase_09_stats"
FIG_DIR      = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#13294B"; TEAL = "#0E7C86"; GOLD = "#E0A526"
RED  = "#A23B3B"; GREY = "#6B7C8C"

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Step 9.F : muscle x task heatmap of EXO % change
# ══════════════════════════════════════════════════════════════════════════════
res = pd.read_csv(STATS_DIR / "step_F_exo_pertask" / "exo_allmuscles_results.csv")

MUSCLE_ORDER = ["Trapezius", "Deltoid", "Biceps", "Triceps",
                "Extensor", "Flexor", "AbdV", "Duration"]
TASK_ORDER   = ["Drinking", "Lifting", "Pick & Place"]
GROUPS       = ["All", "ALS", "Healthy"]

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

for ax, grp in zip(axes, GROUPS):
    g = res[res["group"] == grp]
    muscles = [m for m in MUSCLE_ORDER if m in set(g["muscle"])]
    M = np.full((len(muscles), len(TASK_ORDER)), np.nan)
    S = np.zeros_like(M, dtype=object); S[:] = ""
    for i, m in enumerate(muscles):
        for j, t in enumerate(TASK_ORDER):
            cell = g[(g["muscle"] == m) & (g["task_label"] == t)]
            if cell.empty: continue
            c = cell.iloc[0]
            M[i, j] = c["pct_change"]
            if c["sig_withintask"] and c["sig_global"]: S[i, j] = "**"
            elif c["sig_withintask"]:                   S[i, j] = "*"

    # diverging: blue = reduction (good), red = increase
    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 60
    vmax = max(30, min(70, vmax))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels(TASK_ORDER, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(len(muscles)))
    ax.set_yticklabels(muscles, fontsize=10)
    ax.set_title(grp, fontsize=13, fontweight="bold", color=NAVY, pad=10)

    for i in range(len(muscles)):
        for j in range(len(TASK_ORDER)):
            if not np.isfinite(M[i, j]): continue
            txt = f"{M[i,j]:+.0f}%{S[i,j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    fontweight="bold" if S[i, j] else "normal",
                    color="white" if abs(M[i, j]) > vmax * 0.55 else "black")
            if S[i, j]:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                       edgecolor=GOLD, lw=2.2))
    # separator between proximal and distal
    if "Triceps" in muscles:
        k = muscles.index("Triceps")
        ax.axhline(k + 0.5, color=NAVY, lw=1.6, ls="--", alpha=0.6)

cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
cb.set_label("% change with exoskeleton   (negative = effort reduced)", fontsize=10)
fig.suptitle("Exoskeleton effect across ALL muscles and tasks\n"
             "gold box = significant (FDR)   ·   ** survives both corrections   ·   "
             "dashed line separates proximal (above) from distal (below)",
             fontsize=13, fontweight="bold", color=NAVY, y=1.02)
out1 = FIG_DIR / "phase9F_exo_muscle_task_heatmap.png"
plt.savefig(out1, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved -> {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Step 9.H : selective involvement
#   (a) log-ratio distributions ALS vs HC   (b) fold-change bars per muscle
# ══════════════════════════════════════════════════════════════════════════════
per  = pd.read_csv(STATS_DIR / "step_H_selective" / "selective_per_subject.csv")
rats = pd.read_csv(STATS_DIR / "step_H_selective" / "selective_ratio_tests.csv")

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4),
                         gridspec_kw={"width_ratios": [1, 1, 1.15]})

pairs = list(rats["pair"].unique())

# --- panels (a) and (b): log-ratio, ALS vs HC, with healthy reference band ---
for ax, pair in zip(axes[:2], pairs):
    sub_p  = per[per["pair"] == pair]
    ov     = rats[(rats["pair"] == pair) & (rats["scope"] == "Overall")].iloc[0]
    hc_m   = float(sub_p["hc_ref_mean"].iloc[0])
    hc_sd  = float(sub_p["hc_ref_sd"].iloc[0])
    lo, hi = hc_m - 1.96 * hc_sd, hc_m + 1.96 * hc_sd

    # healthy reference band
    ax.axhspan(lo, hi, color=TEAL, alpha=0.12, zorder=0,
               label="healthy 95% reference range")
    ax.axhline(hc_m, color=TEAL, lw=1.8, ls="--", zorder=1,
               label=f"healthy median log-ratio")
    ax.axhline(0, color=GREY, lw=0.9, ls=":", zorder=1)

    # ALS points (jittered)
    y = sub_p["log_ratio"].to_numpy(dtype=float)
    x = np.random.default_rng(4).normal(1.0, 0.045, size=len(y))
    out = sub_p["above_healthy_range"].to_numpy(dtype=bool) | \
          sub_p["below_healthy_range"].to_numpy(dtype=bool)
    ax.scatter(x[~out], y[~out], s=68, color=NAVY, alpha=0.75,
               edgecolor="white", lw=1.1, zorder=3, label="ALS (within range)")
    if out.any():
        ax.scatter(x[out], y[out], s=110, color=GOLD, edgecolor=RED, lw=1.6,
                   zorder=4, label="ALS (outside range)")
        for xi, yi, sj in zip(x[out], y[out], sub_p.loc[out, "subject"]):
            ax.annotate(str(sj).replace("ALS_Subject_", "ALS "), (xi, yi),
                        xytext=(10, 0), textcoords="offset points",
                        fontsize=8.5, color=RED, va="center")

    # ALS median line
    ax.plot([0.82, 1.18], [np.median(y)] * 2, color=NAVY, lw=2.4, zorder=5)

    ax.set_xlim(0.6, 1.55); ax.set_xticks([])
    ax.set_ylabel("log ratio", fontsize=11)
    title = pair.split(":")[1].strip() if ":" in pair else pair
    ax.set_title(f"{title}\np(FDR) = {ov['p_fdr']:.2f}  ·  n.s.",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

# --- panel (c): fold-change per muscle (the regional gradient) ---
ax = axes[2]
MUSCLES = ["Extensor", "Flexor", "Biceps", "Triceps"]
FOLD    = {}  # filled from the printed Analysis B values; recomputed here for safety
# Recompute fold-change from the ratio-test CSV is not possible, so read from a
# dict the user can adjust if numbers change:
FOLD = {"Extensor": 3.68, "Flexor": 3.20, "Biceps": 1.58, "Triceps": 1.80}
REGION = {"Extensor": "distal", "Flexor": "distal",
          "Biceps": "proximal", "Triceps": "proximal"}
cols = [TEAL if REGION[m] == "distal" else GREY for m in MUSCLES]
bars = ax.bar(MUSCLES, [FOLD[m] for m in MUSCLES], color=cols,
              edgecolor="white", lw=1.4, width=0.68)
ax.axhline(1.0, color=RED, lw=1.4, ls="--", label="no elevation (1×)")
for b, m in zip(bars, MUSCLES):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.08,
            f"{FOLD[m]:.2f}×", ha="center", fontsize=11, fontweight="bold",
            color=NAVY)
# bracket the pairs
ax.plot([0, 1], [4.15, 4.15], color=TEAL, lw=1.5)
ax.text(0.5, 4.22, "distal pair — both high\n(index 1.15, n.s.)", ha="center",
        fontsize=9, color=TEAL, fontweight="bold")
ax.plot([2, 3], [2.35, 2.35], color=GREY, lw=1.5)
ax.text(2.5, 2.42, "proximal pair — both low\n(index 0.88, n.s.)", ha="center",
        fontsize=9, color=GREY, fontweight="bold")

ax.set_ylim(0, 4.9)
ax.set_ylabel("fold-change  (ALS median / healthy median)", fontsize=11)
ax.set_title("Involvement is REGIONAL, not antagonist-selective",
             fontsize=12.5, fontweight="bold", color=NAVY, pad=9)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9, loc="upper right")

fig.suptitle("Selective (dissociated) muscle involvement — no antagonist selectivity detected",
             fontsize=14, fontweight="bold", color=NAVY, y=1.03)
plt.tight_layout()
out2 = FIG_DIR / "phase9H_selective_involvement.png"
plt.savefig(out2, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved -> {out2}")
