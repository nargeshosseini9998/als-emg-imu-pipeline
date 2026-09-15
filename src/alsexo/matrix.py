# alsexo/matrix.py
# Phase 8 (v2) — subject-level feature matrix: coverage/variance screen, clinical 3-tier
# selection (20 features), median imputation, StandardScaler, labels, and the documented
# ALS_Subject_15 artifact flag. Extracted verbatim from 08_v2_matrix.ipynb (cells 1-2).
# NOTE: this module runs on import (script-style, as in the notebook); execute via the CLI.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 8 — Feature Matrix Construction
# Input:  processed/phase_07_kpis/subject_kpis.parquet
# Output: phase_08_matrix/{X_matrix, X_scaled, y_labels, meta}.parquet
#         phase_08_matrix/{feature_report, selected_features_info}.csv
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# --- Paths ---
SUBJ_KPI_PATH  = PROCESSED_ROOT / "phase_07_kpis" / "subject_kpis.parquet"
PHASE8_DIR     = PROCESSED_ROOT / "phase_08_matrix"
PHASE8_DIR.mkdir(parents=True, exist_ok=True)

# --- Config ---
MIN_COVERAGE    = 0.80   # >=19/23 subjects must have a value
MIN_STD         = 1e-6   # near-zero-variance drop
MAX_FEATURES_FA = 20     # >=5 subjects per variable rule of thumb

print("="*60); print("PHASE 8 — Feature Matrix Construction"); print("="*60)
df_subj = pd.read_parquet(SUBJ_KPI_PATH)
print(f"\nLoaded: {df_subj.shape[0]} subjects x {df_subj.shape[1]} columns")

# --- Step 2: metadata vs KPI ---
META_COLS_CANDIDATES = [
    "subject","group","Group","label","class","exo","EXO","condition","Condition",
    "session","Session","task","task_type","Task",
    "n_trials","n_trials_valid_interval","n_trials_emg_valid","n_trials_has_bictri_pair",
    "rate_valid_interval","rate_emg_valid","rate_has_bictri_pair",
]
meta_cols = [c for c in META_COLS_CANDIDATES if c in df_subj.columns]
print(f"\nMetadata columns ({len(meta_cols)}): {meta_cols}")
df_meta = df_subj[meta_cols].copy() if meta_cols else pd.DataFrame(index=df_subj.index)

# --- Step 3: KPI candidates (overall __median only) ---
kpi_cols_all    = [c for c in df_subj.columns if c.startswith("kpi_")]
kpi_median_cols = [c for c in kpi_cols_all if c.endswith("__median")]
kpi_overall_cols= [c for c in kpi_median_cols if "__taskmed__" not in c]
print(f"\nKPI discovery: total={len(kpi_cols_all)}  overall__median={len(kpi_overall_cols)}")

# --- Step 4: coverage filter ---
df_candidates      = df_subj[kpi_overall_cols].copy()
n_subjects         = len(df_candidates)
coverage           = df_candidates.notna().mean()
high_coverage_cols = coverage[coverage >= MIN_COVERAGE].index.tolist()
print(f"\nCoverage >= {MIN_COVERAGE*100:.0f}%: {len(kpi_overall_cols)} -> {len(high_coverage_cols)}")

# --- Step 5: variance filter ---
df_high_cov      = df_subj[high_coverage_cols].copy()
feature_std      = df_high_cov.std()
nonzero_var_cols = feature_std[feature_std > MIN_STD].index.tolist()
print(f"Variance > {MIN_STD}: {len(high_coverage_cols)} -> {len(nonzero_var_cols)}")

# --- Step 6: feature report ---
rows = []
for col in nonzero_var_cols:
    s = df_subj[col].dropna()
    rows.append({"feature":col,"coverage_pct":round(coverage[col]*100,1),
                 "n_valid":int(coverage[col]*n_subjects),
                 "median":round(s.median(),4) if len(s) else np.nan,
                 "std":round(s.std(),4) if len(s) else np.nan,
                 "min":round(s.min(),4) if len(s) else np.nan,
                 "max":round(s.max(),4) if len(s) else np.nan,
                 "skewness":round(s.skew(),3) if len(s)>2 else np.nan})
df_report = pd.DataFrame(rows).sort_values("coverage_pct", ascending=False)
df_report.to_csv(PHASE8_DIR / "feature_report.csv", index=False)
print(f"\nFeature report saved: {len(df_report)} candidates")

# --- Step 7: clinically-informed priority selection ---
# UPDATED: abs_rms effort KPIs added as Priority 1; efficiency/cost composites removed;
# NMD intentionally excluded as a model feature.
FEATURE_PRIORITY_PATTERNS = [
    ("kpi_abs_rms_distal__median",          "absolute distal EMG RMS (extensor/flexor) - primary effort discriminator", 1),
    ("kpi_abs_rms_mean__median",            "absolute mean EMG RMS across muscles",        1),
    ("kpi_iemg_envn_global_sum__median",    "total EMG activity all muscles",              1),
    ("kpi_coact_overlap_bic_tric__median",  "co-activation biceps-triceps",                1),
    ("kpi_peak_gyro_mag__median",           "peak angular velocity",                       1),
    ("kpi_mean_jerk__median",               "movement smoothness (jerk)",                  1),
    ("kpi_peak_envn_Biceps__median",        "peak activation biceps",                      1),
    ("kpi_peak_envn_Triceps__median",       "peak activation triceps",                     1),
    ("kpi_peak_envn_Deltoid__median",       "peak activation deltoid",                     1),
    ("kpi_iemg_envn_Biceps__median",        "iEMG biceps",                                 2),
    ("kpi_iemg_envn_Triceps__median",       "iEMG triceps",                                2),
    ("kpi_iemg_envn_Deltoid__median",       "iEMG deltoid",                                2),
    ("kpi_duty_Biceps__median",             "duty cycle biceps",                           2),
    ("kpi_duty_Triceps__median",            "duty cycle triceps",                          2),
    ("kpi_mean_gyro_mag__median",           "mean angular velocity",                       2),
    ("kpi_duration_s__median",              "trial duration",                              2),
    ("kpi_peak_envn_Trapezius__median",     "peak activation trapezius",                   2),
    ("kpi_peak_envn_Extensor__median",      "peak activation extensor",                    2),
    ("kpi_peak_envn_Flexor__median",        "peak activation flexor",                      2),
    ("kpi_duty_Deltoid__median",            "duty cycle deltoid",                          3),
]

selected_features = []
for col_name, description, priority in FEATURE_PRIORITY_PATTERNS:
    if col_name in nonzero_var_cols:
        selected_features.append({"feature":col_name,"clinical_meaning":description,"priority":priority})

seen, selected_unique = set(), []
for item in selected_features:
    if item["feature"] not in seen:
        seen.add(item["feature"]); selected_unique.append(item)
selected_unique.sort(key=lambda x: x["priority"])
selected_col_names = [i["feature"] for i in selected_unique]

if len(selected_col_names) > MAX_FEATURES_FA:
    print(f"\nWARN: {len(selected_col_names)} features > FA limit {MAX_FEATURES_FA}; keeping priority 1-2.")
    selected_col_names = [i["feature"] for i in selected_unique if i["priority"] <= 2][:MAX_FEATURES_FA]

print(f"\nFinal features selected: {len(selected_col_names)}")
print(f"\n{'Feature':<48}{'Cov':>6} {'P':>2}  meaning")
print("-"*100)
for item in selected_unique:
    if item["feature"] in selected_col_names:
        cov = coverage.get(item["feature"], np.nan)
        print(f"  {item['feature']:<46}{cov*100:>5.0f}% {item['priority']}   {item['clinical_meaning']}")

# warn if the key discriminator is missing
if "kpi_abs_rms_distal__median" not in selected_col_names:
    print("\n[CHECK] kpi_abs_rms_distal__median NOT selected - is it present in subject_kpis.parquet?")

# --- Step 8: impute (median) ---
X = df_subj[selected_col_names].copy()
n_imp = X.isna().sum().sum()
for col in X.columns:
    X[col] = X[col].fillna(X[col].median())
assert X.isna().sum().sum() == 0
print(f"\nImputation: {n_imp} NaN filled with column median")

# --- Step 9: labels from subject name ---
def map_group_from_name(name: str) -> float:
    n = name.lower().strip()
    if any(k in n for k in ["als","patient","sick","disease"]): return 1.0
    if any(k in n for k in ["healthy","control","hc","normal"]): return 0.0
    return np.nan
subjects = df_subj["subject"].astype(str)
y      = subjects.map(map_group_from_name)
groups = y.map({1.0:"ALS", 0.0:"Healthy"})
if y.isna().sum() > 0:
    print(f"\nERROR: unlabeled subjects: {subjects[y.isna()].tolist()}")
else:
    print(f"\nLabels: {(y==1).sum()} ALS, {(y==0).sum()} Healthy")

# --- Step 10: scale ---
scaler   = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
print(f"\nScaled: mean~{X_scaled.mean().mean():.4f}  std~{X_scaled.std().mean():.4f}")

# --- Step 11: save ---
X.index = subjects.values; X_scaled.index = subjects.values
X.to_parquet(PHASE8_DIR / "X_matrix.parquet", index=True)
X_scaled.to_parquet(PHASE8_DIR / "X_scaled.parquet", index=True)
pd.DataFrame({"subject":subjects.values,"y":y.values,"group_raw":groups.values}).to_parquet(PHASE8_DIR/"y_labels.parquet", index=False)
if "subject" not in df_meta.columns:
    df_meta.insert(0, "subject", subjects.values)
df_meta.to_parquet(PHASE8_DIR / "meta.parquet", index=False)
feat_info = pd.DataFrame(selected_unique)
feat_info = feat_info[feat_info["feature"].isin(selected_col_names)].merge(
    df_report[["feature","coverage_pct","std","skewness"]], on="feature", how="left")
feat_info.to_csv(PHASE8_DIR / "selected_features_info.csv", index=False)

print(f"\n{'='*60}\nPhase 8 complete -> {PHASE8_DIR}\n{'='*60}")
print(f"   X: {X.shape[0]} subjects x {X.shape[1]} features")
print(f"   ALS={int((y==1).sum())}  Healthy={int((y==0).sum())}  NaN remaining={X.isna().sum().sum()>0}")
print("\n   Top 5 by variance:")
for fn, fs in X.std().sort_values(ascending=False).head(5).items():
    print(f"      {fn:<48} std={fs:.4f}")

# ===== ALS_15 artifact flag (explicit, logged, non-destructive) =====
# ALS_Subject_15 has a confirmed EMG device artifact (extensor RMS ~0.957 V).
# We do NOT delete it: we flag it so every downstream analysis can include or
# exclude it, and we can report the finding "with vs without" the artifact subject.
import pandas as pd
from pathlib import Path

PHASE8_DIR = PHASE8_DIR
ARTIFACT_SUBJECTS = ["ALS_Subject_15"]   # reason: EMG device artifact

meta = pd.read_parquet(PHASE8_DIR / "meta.parquet")
if "subject" not in meta.columns:
    raise KeyError("meta.parquet has no 'subject' column - check Cell 1 save step.")
meta["excluded"] = meta["subject"].isin(ARTIFACT_SUBJECTS)
meta["exclude_reason"] = meta["subject"].map(
    {s: "EMG device artifact (extensor RMS ~0.957 V)" for s in ARTIFACT_SUBJECTS}).fillna("")
meta.to_parquet(PHASE8_DIR / "meta.parquet", index=False)

print(f"Flagged excluded: {meta.loc[meta['excluded'],'subject'].tolist()}")
print(f"Analysis set (excluded=False): {(~meta['excluded']).sum()} of {len(meta)} subjects")
print("\nDownstream usage:")
print("  full set     : all subjects")
print("  analysis set : meta.loc[~meta['excluded'], 'subject']")
