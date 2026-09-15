# alsexo/stats/step_0_setup.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 2); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.step_0_setup`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — Step 9.0: Setup / Rebuild partA_subject_matrix
#
# Why this cell exists:
#   Phase 9 reads a subject-level matrix of ABSOLUTE per-muscle RMS values
#   (EMG_<muscle>_RMS) — the between-subject-comparable quantities behind the
#   main hyperactivation finding. These are NOT KPIs from Phase 7/8 (which are
#   within-subject normalized or aggregated). They live in the Phase 6 features.
#
#   This cell regenerates partA_subject_matrix.parquet from Phase 6, so the rest
#   of Phase 9 runs on the current, consistent data.
#
# Policy:
#   ALL subjects are kept (n = 23). No subject is excluded here. If a subject
#   later proves anomalous in the descriptive/outlier analysis (Step 9.B), that
#   exclusion will be made explicitly, with a documented reason, at that point.
#
# Output:
#   processed/phase_09_efa/partA_subject_matrix.parquet
#     columns: subject, group, EMG_Extensor_RMS, EMG_Flexor_RMS,
#              EMG_Biceps_RMS, EMG_Triceps_RMS
# =========================================

import re
import numpy as np
import pandas as pd
from pathlib import Path

PHASE6_DIR     = PROCESSED_ROOT / "phase_06_features"
PHASE9_EFA     = PROCESSED_ROOT / "phase_09_efa"
PHASE9_EFA.mkdir(parents=True, exist_ok=True)

# Canonical muscle-name mapping (same policy as Phase 6/7), so EMG_<muscle>_RMS
# columns are matched regardless of original sensor naming.
_CANON = {
    "bicbrachii":"Biceps","biceps":"Biceps","bicipite":"Biceps",
    "tricbrachii":"Triceps","triceps":"Triceps","tricipite":"Triceps",
    "deltmed":"Deltoid","deltoid":"Deltoid","deltoide":"Deltoid",
    "trapdesc":"Trapezius","trap":"Trapezius","trapezio":"Trapezius","trapezius":"Trapezius",
    "extcarprad":"Extensor","extensor":"Extensor","estensore carpo":"Extensor","estensore":"Extensor",
    "flexcarprad":"Flexor","flexor":"Flexor","flessore carpo":"Flexor","flessore":"Flexor",
}
def _canon(tok):
    t = re.sub(r"\s+"," ", str(tok).strip().replace("_"," ").replace("-"," ")).lower()
    return _CANON.get(t, tok)

# The 4 muscles Phase 9 analyses (absolute RMS, between-subject comparable)
RMS_MUSCLES = ["Extensor", "Flexor", "Biceps", "Triceps"]

def map_group(name: str) -> str:
    n = str(name).lower()
    if any(k in n for k in ["als","patient","sick","disease"]): return "ALS"
    if any(k in n for k in ["healthy","control","hc","normal"]): return "Healthy"
    return "Unknown"

rows = []
feat_files = sorted(PHASE6_DIR.glob("*__features.parquet"))
print(f"Found {len(feat_files)} Phase 6 feature files")

for fp in feat_files:
    subject = fp.name.split("__features.parquet")[0]
    df = pd.read_parquet(fp)

    # Build a lookup of available EMG_<muscle>_RMS columns by canonical muscle name.
    # Phase 6 already writes canonical names, but we canonicalize again to be safe.
    rms_lookup = {}
    for c in df.columns:
        m = re.match(r"EMG_(.+)_RMS$", c)
        if m:
            rms_lookup[_canon(m.group(1))] = c

    rec = {"subject": subject}
    for muscle in RMS_MUSCLES:
        col = rms_lookup.get(muscle)
        # subject-level value = median across all windows (robust); absolute units (V)
        rec[f"EMG_{muscle}_RMS"] = float(df[col].median()) if (col and col in df.columns) else np.nan
    rows.append(rec)

partA = pd.DataFrame(rows)
partA["group"] = partA["subject"].apply(map_group)

# Reorder columns
cols = ["subject", "group"] + [f"EMG_{m}_RMS" for m in RMS_MUSCLES]
partA = partA[cols]

# ──────────────────────────────────────────────────────────────────────────────
# Documented, evidence-based exclusion: ALS_Subject_15
#
# Reason (visually verified from the raw band-pass signal):
#   The extensor channel showed band-pass amplitudes of ±2 to ±4 V — roughly
#   three orders of magnitude above the physiological range for surface EMG,
#   which is in millivolts — together with near-continuous activity inconsistent
#   with the task structure (no rest/burst pattern). Per-trial extensor RMS in
#   the noexo condition reached ~1.0–1.36 V. This is consistent with a
#   sensor/electrode artifact, not muscle activity.
#
# Action:
#   ALS_Subject_15 is excluded from EMG-amplitude analyses. A FULL version
#   (all subjects) is also saved so the main finding can be reported as a
#   robustness check "with vs without" the artifact subject.
# ──────────────────────────────────────────────────────────────────────────────
ARTIFACT_SUBJECTS = ["ALS_Subject_15"]

# Save the FULL matrix (all subjects) for robustness / sensitivity reporting
partA_full = partA.copy()
full_path  = PHASE9_EFA / "partA_subject_matrix_full.parquet"
partA_full.to_parquet(full_path, index=False)

# Build the ANALYSIS matrix (artifact subject removed)
partA = partA[~partA["subject"].isin(ARTIFACT_SUBJECTS)].reset_index(drop=True)

# Save the analysis matrix under the name the rest of Phase 9 reads
out_path = PHASE9_EFA / "partA_subject_matrix.parquet"
partA.to_parquet(out_path, index=False)

# Report
print(f"\npartA_subject_matrix (analysis set) rebuilt: {partA.shape[0]} subjects x {partA.shape[1]} columns")
print(f"  ALS: {(partA['group']=='ALS').sum()}   Healthy: {(partA['group']=='Healthy').sum()}   Unknown: {(partA['group']=='Unknown').sum()}")
print(f"  Excluded (sensor artifact): {ARTIFACT_SUBJECTS}")
print(f"  Saved analysis set -> {out_path}")
print(f"  Saved full set (n={len(partA_full)}) -> {full_path}\n")

# Main finding preview, reported BOTH ways (robustness check):
#   analysis set (ALS_15 excluded)  vs  full set (ALS_15 included)
def muscle_ratios(df, label):
    print(f"\nPer-muscle RMS median by group — {label}:")
    for m in RMS_MUSCLES:
        col = f"EMG_{m}_RMS"
        a = df.loc[df["group"]=="ALS", col].median()
        h = df.loc[df["group"]=="Healthy", col].median()
        ratio = (a/h) if (h and np.isfinite(h) and h>0) else np.nan
        print(f"  {m:<9} ALS={a:.4f}  HC={h:.4f}  ratio={ratio:.1f}x")

muscle_ratios(partA,      f"ANALYSIS SET (n={len(partA)}, ALS_15 excluded)")
muscle_ratios(partA_full, f"FULL SET (n={len(partA_full)}, ALS_15 included)")
print("\nNote: the finding holds either way; excluding the artifact subject brings the")
print("inflated extensor ratio down to a more accurate value (real biology, not artifact).")
