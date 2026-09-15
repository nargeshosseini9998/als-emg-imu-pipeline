# alsexo/stats/step_a_extended_all_kpis.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 19); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.step_a_extended_all_kpis`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — Step 9.A-extended: Group comparison on ALL between-subject-comparable KPIs
#
# Same methodology as Step 9.A/9.C:
#   Shapiro-Wilk (normality) -> justify non-parametric
#   Mann-Whitney U (ALS vs Healthy), effect size r = Z/sqrt(N)
#   Benjamini-Hochberg FDR correction across the tested KPIs
#
# IMPORTANT — only between-subject-COMPARABLE KPIs are tested:
#   Included (absolute / bounded, comparable):
#     - absolute RMS per muscle (the primary effort measure)
#     - IMU motion: peak gyro, mean gyro, mean jerk
#     - co-activation (ONE representative; the 4 co-act KPIs are redundant, rho~0.99)
#     - trial duration
#   Secondary/exploratory (bounded but threshold-dependent): duty cycles
#   EXCLUDED (within-subject normalized -> NOT comparable between subjects):
#     - peak envelope, IEMG per muscle and global  (firewall)
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

SUBJ_KPI_PATH  = PROCESSED_ROOT / "phase_07_kpis" / "subject_kpis.parquet"
PHASE9_DIR     = PROCESSED_ROOT / "phase_09_stats" / "step_A_extended"
PHASE9_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_SUBJECTS = ["ALS_Subject_15"]   # documented sensor artifact (consistent with rest of Phase 9)

# ---- comparable KPI sets (subject-level medians) ----
PRIMARY_COMPARABLE = {
    "kpi_abs_rms_distal__median"        : "Absolute RMS distal (Ext/Flex)",
    "kpi_abs_rms_mean__median"          : "Absolute RMS mean (all muscles)",
    "kpi_peak_gyro_mag__median"         : "Peak angular velocity",
    "kpi_mean_gyro_mag__median"         : "Mean angular velocity",
    "kpi_mean_jerk__median"             : "Mean jerk (smoothness)",
    "kpi_coact_overlap_bic_tric__median": "Co-activation (Bic-Tri overlap)",
    "kpi_duration_s__median"            : "Trial duration",
}
SECONDARY_COMPARABLE = {  # bounded but threshold-dependent -> exploratory
    "kpi_duty_Biceps__median"   : "Duty cycle Biceps",
    "kpi_duty_Triceps__median"  : "Duty cycle Triceps",
    "kpi_duty_Deltoid__median"  : "Duty cycle Deltoid",
    "kpi_duty_Extensor__median" : "Duty cycle Extensor",
    "kpi_duty_Flexor__median"   : "Duty cycle Flexor",
    "kpi_duty_Trapezius__median": "Duty cycle Trapezius",
}

def mannwhitney_r(als_vals, hc_vals):
    """Mann-Whitney U + rank-biserial-style effect size r = Z / sqrt(N)."""
    als_vals = np.asarray(als_vals, float); als_vals = als_vals[np.isfinite(als_vals)]
    hc_vals  = np.asarray(hc_vals,  float); hc_vals  = hc_vals[np.isfinite(hc_vals)]
    n1, n2 = len(als_vals), len(hc_vals)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan, n1, n2
    U, p = stats.mannwhitneyu(als_vals, hc_vals, alternative="two-sided")
    # normal approximation for Z (with continuity), then r = Z/sqrt(N)
    N = n1 + n2
    mu_U = n1*n2/2.0
    sigma_U = np.sqrt(n1*n2*(N+1)/12.0)
    Z = (U - mu_U)/sigma_U if sigma_U > 0 else 0.0
    r = abs(Z)/np.sqrt(N)
    return U, p, r, n1, n2

def fdr_bh(pvals):
    p = np.asarray(pvals, float); m = np.isfinite(p); out = np.full_like(p, np.nan)
    idx = np.where(m)[0]; pv = p[idx]; n = len(pv)
    order = np.argsort(pv); ranked = pv[order]
    adj = ranked * n / (np.arange(n)+1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n); res[order] = np.clip(adj, 0, 1)
    out[idx] = res; return out

# ---- load & label ----
df = pd.read_parquet(SUBJ_KPI_PATH)
df = df[~df["subject"].isin(ARTIFACT_SUBJECTS)].copy()
def grp(n):
    n=str(n).lower()
    return "ALS" if "als" in n else ("Healthy" if "healthy" in n else "?")
df["group"] = df["subject"].apply(grp)
n_als = (df["group"]=="ALS").sum(); n_hc = (df["group"]=="Healthy").sum()
print(f"Subjects: {len(df)} (ALS={n_als}, Healthy={n_hc}); excluded {ARTIFACT_SUBJECTS}\n")

def run_block(kpi_map, title, do_fdr=True):
    print("="*78); print(title); print("="*78)
    rows=[]
    for col,label in kpi_map.items():
        if col not in df.columns:
            print(f"  [skip] {label}: column not found"); continue
        als = pd.to_numeric(df.loc[df.group=="ALS",  col], errors="coerce")
        hc  = pd.to_numeric(df.loc[df.group=="Healthy",col], errors="coerce")
        # normality (within each group) — for the methods note
        sw_als = stats.shapiro(als.dropna())[1] if als.notna().sum()>=3 else np.nan
        sw_hc  = stats.shapiro(hc.dropna())[1]  if hc.notna().sum()>=3 else np.nan
        U,p,r,n1,n2 = mannwhitney_r(als, hc)
        rows.append({"kpi":col,"label":label,"als_med":als.median(),"hc_med":hc.median(),
                     "U":U,"p_raw":p,"r":r,"sw_als":sw_als,"sw_hc":sw_hc})
    res = pd.DataFrame(rows)
    if do_fdr and len(res):
        res["p_fdr"] = fdr_bh(res["p_raw"].values)
    else:
        res["p_fdr"] = res.get("p_raw", np.nan)
    # print
    print(f"\n{'KPI':<34}{'ALS med':>10}{'HC med':>10}{'p_raw':>9}{'p_FDR':>9}{'r':>7}  dir sig")
    print("-"*92)
    for _,x in res.sort_values('p_fdr').iterrows():
        direction = "ALS>HC" if x['als_med']>x['hc_med'] else "ALS<HC"
        eff = "large" if x['r']>=0.5 else ("medium" if x['r']>=0.3 else "small")
        sig = "*" if (np.isfinite(x['p_fdr']) and x['p_fdr']<0.05) else " "
        print(f"  {x['label']:<32}{x['als_med']:>10.4f}{x['hc_med']:>10.4f}"
              f"{x['p_raw']:>9.4f}{x['p_fdr']:>9.4f}{x['r']:>7.3f}  {direction} {eff} {sig}")
    return res

res_primary   = run_block(PRIMARY_COMPARABLE,   "PRIMARY comparable KPIs (absolute / bounded)")
print()
res_secondary = run_block(SECONDARY_COMPARABLE, "SECONDARY (duty cycles — bounded but threshold-dependent; exploratory)")

# save
res_primary.to_csv(PHASE9_DIR/"comparable_primary_results.csv", index=False)
res_secondary.to_csv(PHASE9_DIR/"comparable_secondary_results.csv", index=False)
print(f"\nSaved -> {PHASE9_DIR}")
print("  comparable_primary_results.csv")
print("  comparable_secondary_results.csv")

print(f"""
INTERPRETATION GUIDE:
  - Compare these effect sizes (r) to the absolute-RMS finding (extensor r~0.87).
  - Expect: effort KPIs (abs RMS) separate strongly; motion KPIs (gyro, jerk)
    should NOT separate (that is the effort-motion dissociation, restated).
  - Co-activation: if higher in ALS, supports the guarded/stiffening interpretation.
  - Duty cycles are exploratory only (threshold-dependent on a normalized envelope).
""")
