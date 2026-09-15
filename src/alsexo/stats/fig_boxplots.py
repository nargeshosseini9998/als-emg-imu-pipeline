# alsexo/stats/fig_boxplots.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 17); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.fig_boxplots`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

"""
Phase 9.B — Box/violin plots of the 4 absolute RMS muscles, ALS vs HC.
THE headline figure: shows the distal-predominant gradient AND the dispersion
difference (ALS wide, HC tight) with individual subject points overlaid.

Source: trial_kpis has kpi_abs_rms_distal & kpi_abs_rms_mean per condition, but
NOT per-muscle absolute RMS. Per-muscle absolute RMS must come from Phase 6
features. This script reads the Phase 6 per-subject feature files and computes
the median absolute RMS per muscle per subject.

If your Phase 9 already saved a subject x muscle absolute-RMS table, point
SUBJECT_RMS_PATH at it instead and skip the Phase-6 aggregation block.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT     = PROCESSED_ROOT
FIG_DIR  = FIG_ROOT / "phase09_boxplots"
FIG_DIR.mkdir(parents=True, exist_ok=True)
EXCLUDE  = {"ALS_Subject_15", "ALS_Subject_6"}

# canonical muscle -> the RMS column name in your Phase 6 features
# (adjust the right-hand strings if your feature columns differ)
MUSCLES = {
    "Extensor": "EMG_ExtCarpRad_RMS",
    "Flexor":   "EMG_FlexCarpRad_RMS",
    "Biceps":   "EMG_BicBrachii_RMS",
    "Triceps":  "EMG_TricBrachii_RMS",
}
DISTAL = {"Extensor", "Flexor"}

# ---- aggregate per-muscle absolute RMS from Phase 6 ----
rows = []
for f in sorted((ROOT / "phase_06_features").glob("*__features.parquet")):
    subj = f.name.split("__")[0]
    if subj in EXCLUDE:
        continue
    df = pd.read_parquet(f)
    rec = {"subject": subj,
           "group": "ALS" if subj.startswith("ALS") else "Healthy"}
    for m, col in MUSCLES.items():
        # find the column (robust to slight name variants)
        cands = [c for c in df.columns if c == col] or \
                [c for c in df.columns if m.lower() in c.lower() and c.lower().endswith("rms")]
        rec[m] = df[cands[0]].median() if cands else np.nan
    rows.append(rec)

data = pd.DataFrame(rows)
print("subjects:", len(data), "| ALS:", (data.group=="ALS").sum(),
      "| HC:", (data.group=="Healthy").sum())
print(data.round(4).to_string(index=False))

# ---- plot: 4 panels, ALS vs HC box + jittered points ----
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=False)
C = {"ALS": "#d1495b", "Healthy": "#3a7ca5"}
for ax, m in zip(axes, MUSCLES):
    groups = ["Healthy", "ALS"]
    vals = [data.loc[data.group==g, m].dropna().values for g in groups]
    bp = ax.boxplot(vals, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=2),
                    showfliers=False)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(C[g]); patch.set_alpha(0.45)
    # jittered individual points
    for i, g in enumerate(groups):
        v = data.loc[data.group==g, m].dropna().values
        x = np.random.normal(i+1, 0.06, len(v))
        ax.scatter(x, v, color=C[g], edgecolor="white", s=42, zorder=3, linewidth=0.6)
    ax.set_xticks([1,2]); ax.set_xticklabels(groups, fontsize=11)
    title = m + ("  (distal)" if m in DISTAL else "  (proximal)")
    ax.set_title(title, fontsize=13,
                 fontweight="bold" if m in DISTAL else "normal",
                 color="#21295C" if m in DISTAL else "#5B6B7A")
    ax.set_ylabel("absolute RMS (V)", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Absolute EMG RMS by muscle — ALS vs Healthy (each dot = one subject)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
out = FIG_DIR / "phase9_rms_boxplots.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved ->", out)
plt.show()
