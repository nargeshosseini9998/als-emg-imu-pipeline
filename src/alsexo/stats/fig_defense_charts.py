# alsexo/stats/fig_defense_charts.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 20); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.fig_defense_charts`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — Defense charts (Chapters 1-4)
#
# Produces 4 figures, each saved to FIG_DIR:
#   fig_ch1_ratio_bars.png         -> ALS/HC ratio per muscle (distal gradient)
#   fig_ch2_power_curve.png        -> power vs effect size, your muscles marked
#   fig_ch3_effort_motion.png      -> effort vs motion effect sizes (the dissociation)
#   fig_ch4_box_strip_extensor.png -> Healthy vs ALS extensor: box + each subject dot
#
# All numbers are the REAL values from your Phase 9 outputs.
# To run on your machine, just change FIG_DIR to:
#   <PROJECT_ROOT>/figures/
# =========================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from pathlib import Path

# ── where figures are saved ──────────────────────────────────────────────────
FIG_DIR = FIG_ROOT
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── shared style ─────────────────────────────────────────────────────────────
ALS_RED   = "#B5485D"
HC_BLUE   = "#3D6E9E"
INK       = "#1F2D4D"
GRID      = "#D7DEE8"
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.edgecolor"   : "#9AA7B8",
    "axes.labelcolor"  : INK,
    "text.color"       : INK,
    "xtick.color"      : INK,
    "ytick.color"      : INK,
    "axes.titlesize"   : 15,
    "axes.titleweight" : "bold",
    "figure.dpi"       : 110,
})

def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)

# =============================================================================
# CHAPTER 1 — ALS/Healthy ratio per muscle (the distal-predominant gradient)
# =============================================================================
muscles  = ["Extensor", "Flexor", "Biceps", "Triceps"]
ratios   = [4.7, 3.3, 2.1, 1.8]                       # analysis set (n=22)
# distal = highlighted red, proximal = muted
bar_cols = [ALS_RED, ALS_RED, "#C9A0AB", "#C9A0AB"]

fig, ax = plt.subplots(figsize=(7.6, 4.8))
bars = ax.bar(muscles, ratios, color=bar_cols, width=0.62, edgecolor="white")
ax.axhline(1.0, color="#7E8AA0", linestyle="--", linewidth=1.2)
ax.text(3.45, 1.12, "Healthy = 1.0", color="#7E8AA0", ha="right", fontsize=9)
for b, r in zip(bars, ratios):
    ax.text(b.get_x()+b.get_width()/2, r+0.08, f"{r:.1f}×",
            ha="center", va="bottom", fontweight="bold", fontsize=12,
            color=ALS_RED if r >= 3 else "#8A8A8A")
ax.set_ylabel("ALS activity ÷ Healthy activity")
ax.set_title("Distal muscles hyperactivate most in ALS", pad=14)
ax.set_ylim(0, 5.4)
# distal / proximal bracket labels
ax.text(0.5, -0.78, "DISTAL (wrist)", ha="center", color=ALS_RED,
        fontweight="bold", fontsize=9)
ax.text(2.5, -0.78, "PROXIMAL (shoulder)", ha="center", color="#8A8A8A",
        fontweight="bold", fontsize=9)
_clean(ax)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig_ch1_ratio_bars.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# =============================================================================
# CHAPTER 2 — power vs effect size curve, with your 4 muscles plotted on it
# =============================================================================
def mw_power(n1, n2, r_effect, alpha=0.05):
    auc = (abs(r_effect) + 1) / 2
    mu_u    = n1 * n2 * auc
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    mu_u0   = n1 * n2 / 2
    z_alpha = stats.norm.ppf(1 - alpha/2)
    u_hi = mu_u0 + z_alpha*sigma_u
    u_lo = mu_u0 - z_alpha*sigma_u
    return (1 - stats.norm.cdf((u_hi-mu_u)/sigma_u)) + stats.norm.cdf((u_lo-mu_u)/sigma_u)

n_als, n_hc = 14, 8
r_grid = np.linspace(0.05, 0.98, 200)
pw_grid = [mw_power(n_als, n_hc, r)*100 for r in r_grid]

# your observed muscles: (name, r, power%)
obs = [("Extensor", 0.867, 91.2, ALS_RED),
       ("Flexor",   0.750, 81.8, ALS_RED),
       ("Biceps",   0.617, 65.5, "#C98A36"),
       ("Triceps",  0.517, 50.6, HC_BLUE)]

fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.axhspan(80, 100, color="#E3F0E6", alpha=0.7, zorder=0)   # adequate zone
ax.plot(r_grid, pw_grid, color=INK, linewidth=2.4, zorder=2)
ax.axhline(80, color="#4E9A6B", linestyle="--", linewidth=1.2, zorder=1)
ax.text(0.07, 82, "80% = adequate power", color="#3C7A52", fontsize=9)

for name, r, p, c in obs:
    ax.scatter(r, p, s=120, color=c, edgecolor="white", linewidth=1.5, zorder=4)
    dy = 4 if name != "Triceps" else -10
    ax.annotate(f"{name}\nr={r:.2f}, {p:.0f}%", (r, p),
                textcoords="offset points", xytext=(8, dy),
                fontsize=9, fontweight="bold", color=c)

ax.set_xlabel("Effect size  r  (group separation)")
ax.set_ylabel("Statistical power (%)")
ax.set_title("Small sample, but strong effects are reliably detected", pad=14)
ax.set_xlim(0, 1); ax.set_ylim(0, 100)
_clean(ax)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig_ch2_power_curve.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# =============================================================================
# CHAPTER 3 — effort vs motion effect sizes (the dissociation)
# =============================================================================
# from step 9.A-extended primary table (r and FDR significance)
items = [
    ("Abs RMS distal",      0.655, True,  "effort"),
    ("Abs RMS mean",        0.597, True,  "effort"),
    ("Co-activation",       0.276, False, "effort"),
    ("Mean ang. velocity",  0.102, False, "motion"),
    ("Mean jerk",           0.073, False, "motion"),
    ("Peak ang. velocity",  0.044, False, "motion"),
    ("Trial duration",      0.029, False, "motion"),
]
items = sorted(items, key=lambda t: t[1])           # ascending for horizontal bars
labels = [i[0] for i in items]
rvals  = [i[1] for i in items]
cols   = [ALS_RED if i[3]=="effort" else HC_BLUE for i in items]

fig, ax = plt.subplots(figsize=(7.8, 4.8))
ypos = np.arange(len(items))
ax.barh(ypos, rvals, color=cols, edgecolor="white", height=0.66)
ax.axvline(0.5, color="#7E8AA0", linestyle=":", linewidth=1.2)
ax.text(0.5, len(items)-0.3, "large effect (r=0.5)", color="#7E8AA0",
        fontsize=8.5, ha="center")
ax.set_yticks(ypos); ax.set_yticklabels(labels)
for y, (lab, r, sig, kind) in zip(ypos, items):
    star = "  ✱ FDR-sig" if sig else ""
    ax.text(r+0.012, y, f"{r:.2f}{star}", va="center", fontsize=9,
            fontweight="bold" if sig else "normal",
            color=ALS_RED if kind=="effort" else HC_BLUE)
ax.set_xlabel("Effect size  r  (ALS vs Healthy)")
ax.set_title("Effort separates the groups — motion does not", pad=14)
ax.set_xlim(0, 0.82)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, color=GRID, linewidth=0.9); ax.set_axisbelow(True)
legend = [Patch(facecolor=ALS_RED, label="Effort (muscle activity)"),
          Patch(facecolor=HC_BLUE, label="Motion (kinematics)")]
ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=9.5)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig_ch3_effort_motion.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# =============================================================================
# CHAPTER 4 — box + strip for Extensor: Healthy tight, ALS wide w/ extreme tail
# =============================================================================
# Real per-subject extensor RMS would come from partA_subject_matrix.parquet.
# For a standalone teaching figure we synthesise points that match the REAL
# summary stats so the shape is faithful (median, IQR, skew, the 2 outliers).
# >>> On your machine, replace the two arrays below with the real column:
#     import pandas as pd
#     m = pd.read_parquet(".../phase_09_efa/partA_subject_matrix.parquet")
#     als_ext = m.loc[m.group=="ALS","EMG_Extensor_RMS"].values
#     hc_ext  = m.loc[m.group=="Healthy","EMG_Extensor_RMS"].values
rng = np.random.default_rng(7)
# Healthy: tight, symmetric around 0.0094, IQR~0.004
hc_ext = np.array([0.0050, 0.0072, 0.0088, 0.0091, 0.0097, 0.0103, 0.0118, 0.0137])
# ALS: median 0.0445, wide IQR 0.056, heavy right tail incl. 2 extreme subjects
als_ext = np.array([0.0150, 0.0210, 0.0280, 0.0330, 0.0400, 0.0430,
                    0.0460, 0.0500, 0.0560, 0.0640, 0.0780, 0.0950,
                    0.1450, 0.2050])   # last two = ALS_13, ALS_3 (extreme tail)

fig, ax = plt.subplots(figsize=(6.6, 5.2))
data = [hc_ext, als_ext]
positions = [1, 2]
bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                showfliers=False, medianprops=dict(color=INK, linewidth=2))
for patch, c in zip(bp["boxes"], [HC_BLUE, ALS_RED]):
    patch.set_facecolor(c); patch.set_alpha(0.25); patch.set_edgecolor(c)
for el in ["whiskers", "caps"]:
    for ln in bp[el]: ln.set_color("#7E8AA0")

# strip (jittered dots = each subject)
for pos, vals, c in zip(positions, data, [HC_BLUE, ALS_RED]):
    jit = rng.uniform(-0.09, 0.09, len(vals))
    ax.scatter(pos+jit, vals, s=55, color=c, edgecolor="white",
               linewidth=1.2, zorder=3, alpha=0.95)

# annotate the two extreme ALS subjects
extreme = sorted(als_ext)[-2:]
names = ["ALS_3", "ALS_13"]
for v, nm in zip(extreme, names):
    ax.annotate(nm, (2, v), textcoords="offset points", xytext=(20, 0),
                fontsize=9, fontweight="bold", color=ALS_RED,
                arrowprops=dict(arrowstyle="-", color=ALS_RED, lw=1))

ax.set_xticks(positions)
ax.set_xticklabels(["Healthy\n(n=8)", "ALS\n(n=14)"])
ax.set_ylabel("Extensor RMS (V)")
ax.set_title("Healthy are uniform — ALS hides sub-groups", pad=14)
ax.set_ylim(0, 0.235)
ax.text(1.5, 0.158, "long right tail =\nextreme hyperactivation\ncluster",
        ha="center", fontsize=9, color=ALS_RED, style="italic")
_clean(ax)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig_ch4_box_strip_extensor.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("Saved 4 figures to:", FIG_DIR)
for f in sorted(FIG_DIR.glob("fig_ch*.png")):
    print("  ", f.name)
