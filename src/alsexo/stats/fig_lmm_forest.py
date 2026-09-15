# alsexo/stats/fig_lmm_forest.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 22); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.fig_lmm_forest`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — LMM coefficient (forest) plot
#
# Visualises the fixed effects of the two mixed models (Extensor, Biceps)
# as coefficient + 95% CI on the log scale, with a zero reference line.
# A small companion panel shows the ICC for each model.
#
# All numbers are the REAL Step 9.G LMM output (coef, SE).
# 95% CI = coef ± 1.96*SE.
#
# =========================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = FIG_ROOT
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALS_RED = "#B5485D"
HC_BLUE = "#3D6E9E"
INK     = "#1F2D4D"
GRID    = "#D7DEE8"
MUTE    = "#9AA7B8"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": MUTE,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.dpi": 110,
})

# ── REAL fixed effects: (label, coef, SE, p) ─────────────────────────────────
# order is top-to-bottom as we want them displayed (so reverse for plotting)
extensor = [
    ("ALS vs Healthy",            1.249, 0.371, 0.0008),
    ("Task: lifting",            -1.165, 0.145, 0.0000),
    ("Task: pick & place",       -0.527, 0.145, 0.0003),
    ("EXO vs no-EXO",            -0.078, 0.119, 0.5116),
    ("ALS × lifting",             0.198, 0.186, 0.2869),
    ("ALS × pick & place",        0.365, 0.186, 0.0497),
    ("ALS × EXO",                 0.078, 0.153, 0.6118),
]
biceps = [
    ("ALS vs Healthy",            0.702, 0.312, 0.0243),
    ("Task: lifting",            -0.230, 0.155, 0.1366),
    ("Task: pick & place",        0.008, 0.155, 0.9562),
    ("EXO vs no-EXO",            -0.542, 0.126, 0.0000),
    ("ALS × lifting",            -0.288, 0.198, 0.1467),
    ("ALS × pick & place",       -0.183, 0.198, 0.3545),
    ("ALS × EXO",                 0.280, 0.163, 0.0861),
]
ICC = {"RMS Extensor": 0.775, "RMS Biceps": 0.653}

def panel(ax, effects, title):
    labels = [e[0] for e in effects]
    coefs  = np.array([e[1] for e in effects])
    ses    = np.array([e[2] for e in effects])
    ps     = np.array([e[3] for e in effects])
    ci     = 1.96 * ses
    y = np.arange(len(effects))[::-1]   # first effect at top

    ax.axvline(0, color=MUTE, linestyle="--", linewidth=1.2, zorder=1)
    for yi, c, e, p in zip(y, coefs, ci, ps):
        sig = p < 0.05
        col = ALS_RED if sig else MUTE
        ax.plot([c-e, c+e], [yi, yi], color=col, linewidth=2.2, zorder=2,
                solid_capstyle="round")
        ax.scatter(c, yi, s=70, color=col, edgecolor="white",
                   linewidth=1.3, zorder=3)
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        ax.annotate(f"{c:+.2f}{star}", (c, yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5,
                    fontweight="bold" if sig else "normal", color=col)

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=22,
                 color=INK, loc="left")
    ax.set_xlabel("coefficient (log EMG units)", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_xlim(-1.9, 1.9)
    ax.margins(y=0.12)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
panel(axes[0], extensor, "RMS Extensor  (distal)")
panel(axes[1], biceps,   "RMS Biceps  (proximal)")

# subtitle note about significance colour
fig.text(0.5, 0.965, "Fixed effects from the linear mixed model  ·  point = coefficient, line = 95% CI",
         ha="center", fontsize=11, color=INK)
fig.text(0.5, 0.93, "red = significant (CI clears 0)   grey = not significant (CI crosses 0)   ref: Healthy · drinking · no-EXO",
         ha="center", fontsize=9, color=MUTE)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(FIG_DIR/"fig_ch10_lmm_forest.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ── companion: ICC bar ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.4, 3.2))
names = list(ICC.keys()); vals = [ICC[n] for n in names]
bars = ax.barh(names[::-1], vals[::-1], color=[HC_BLUE, ALS_RED][:len(names)][::-1],
               height=0.5, edgecolor="white")
ax.axvline(0.5, color=MUTE, linestyle=":", linewidth=1.2)
ax.text(0.5, -0.72, "0.50 = high-clustering threshold", color=MUTE, fontsize=8.5, ha="center")
for b, v in zip(bars, vals[::-1]):
    ax.text(v+0.015, b.get_y()+b.get_height()/2, f"{v:.2f}",
            va="center", fontweight="bold", fontsize=11, color=INK)
ax.set_xlim(0, 1); ax.set_xlabel("ICC (between-subject variance share)", fontsize=10, labelpad=18)
ax.set_title("78% of variance is between-subject → LMM justified", fontsize=11,
             fontweight="bold", color=INK, loc="left", pad=14)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig_ch10_icc.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("saved:")
for f in sorted(FIG_DIR.glob("fig_ch10*.png")): print("  ", f.name)
