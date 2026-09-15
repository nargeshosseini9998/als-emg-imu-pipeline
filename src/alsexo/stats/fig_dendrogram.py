# alsexo/stats/fig_dendrogram.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 18); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.fig_dendrogram`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

"""
Phase 9.D — Ward hierarchical clustering dendrogram.
Shows HOW subjects merge into clusters, with leaf labels coloured by TRUE
diagnosis (ALS red / Healthy blue). Makes the 'three clusters' concrete and
visually shows mild-ALS subjects sitting among the healthy controls.

Recomputed from the Phase 8 matrix so it works regardless of what the Phase 9
notebook saved. Uses the 4 absolute RMS features if present, else all features.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

ROOT    = PROCESSED_ROOT
P8      = ROOT / "phase_08_matrix"
FIG_DIR = FIG_ROOT / "phase09_dendrogram"
FIG_DIR.mkdir(parents=True, exist_ok=True)
EXCLUDE = {"ALS_Subject_15"}

X = pd.read_parquet(P8 / "X_matrix.parquet")
y = pd.read_parquet(P8 / "y_labels.parquet")

# group per subject (subject is X's index)
Xr = X.reset_index().rename(columns={"index": "subject"})
if "subject" not in Xr.columns:
    Xr = Xr.rename(columns={Xr.columns[0]: "subject"})
Xr = Xr.merge(y[["subject", "group_raw"]], on="subject", how="left")
Xr = Xr[~Xr.subject.isin(EXCLUDE)].reset_index(drop=True)

# prefer the 4 absolute RMS muscle features if they exist; else use abs_rms cols
rms_cols = [c for c in X.columns if "abs_rms" in c.lower()]
feat_cols = rms_cols if rms_cols else list(X.columns)
print("clustering on:", feat_cols)

Z_data = StandardScaler().fit_transform(Xr[feat_cols].values)
link = linkage(Z_data, method="ward")

# colour leaf labels by true group
labels = (Xr["subject"].str.replace("_Subject", "").str.replace("Healthy","HC")).tolist()
groups = Xr["group_raw"].tolist()

fig, ax = plt.subplots(figsize=(13, 5.5))
dendrogram(link, labels=labels, ax=ax, color_threshold=0.7*max(link[:,2]),
           leaf_font_size=10)
# recolour the x tick labels by diagnosis
for lbl in ax.get_xmajorticklabels():
    subj_is_als = "ALS" in lbl.get_text()
    lbl.set_color("#d1495b" if subj_is_als else "#3a7ca5")
    lbl.set_fontweight("bold")
ax.set_title("Ward hierarchical clustering — leaf colour = true diagnosis "
             "(red ALS, blue Healthy)", fontsize=13, fontweight="bold")
ax.set_ylabel("Ward linkage distance", fontsize=11)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
out = FIG_DIR / "phase9_dendrogram.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved ->", out)
print("\nRead it: ALS subjects whose red labels sit inside an otherwise-blue "
      "branch are the mild phenotype indistinguishable by EMG amplitude.")
plt.show()
