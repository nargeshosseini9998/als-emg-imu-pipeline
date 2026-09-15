# alsexo/stats/step_b_descriptive.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 6); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.step_b_descriptive`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — Step 9.B: Complete Descriptive Statistics
#
# Purpose:
#   Full characterization of each KPI distribution per group:
#   - Central tendency: mean, median
#   - Spread: SD, IQR, range (min-max)
#   - Shape: skewness, kurtosis
#   - Outliers: Tukey fence (1.5*IQR rule)
#   - 95% CI for median (bootstrap)
#
# This produces the paper-ready "Table 1" equivalent.
#
# Output:
#   phase_09_stats/step_B_descriptive/descriptive_stats.csv
#   phase_09_stats/step_B_descriptive/outliers.csv
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

np.random.seed(42)


PHASE9_STATS = PROCESSED_ROOT / "phase_09_stats" / "step_B_descriptive"
PHASE9_STATS.mkdir(parents=True, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
df = pd.read_parquet(
    PROCESSED_ROOT / "phase_09_efa" / "partA_subject_matrix.parquet"
)

def map_group(name):
    n = str(name).lower()
    if "als" in n:     return "ALS"
    if "healthy" in n: return "Healthy"
    return None

if "group" not in df.columns:
    df["group"] = df["subject"].map(map_group)

# Add duration
try:
    df7 = pd.read_parquet(PROCESSED_ROOT / "phase_07_kpis" / "trial_kpis.parquet")
    df7 = df7.copy()
    df7["group"] = df7["subject"].apply(map_group)
    df7["tid"]   = df7["trial_id"].astype(str).str.lower()
    noexo7 = df7[df7["tid"].str.contains("noexo")].copy()
    noexo7["kpi_duration_s"] = pd.to_numeric(noexo7["kpi_duration_s"], errors="coerce")
    dur = noexo7.groupby("subject")["kpi_duration_s"].median().reset_index()
    df = df.merge(dur, on="subject", how="left")
except Exception as e:
    print(f"Duration merge skipped: {e}")

KPI_COLS = {
    "EMG_Extensor_RMS" : "RMS Extensor (V)",
    "EMG_Flexor_RMS"   : "RMS Flexor (V)",
    "EMG_Biceps_RMS"   : "RMS Biceps (V)",
    "EMG_Triceps_RMS"  : "RMS Triceps (V)",
    "kpi_duration_s"   : "Trial Duration (s)",
}
KPI_COLS = {k: v for k, v in KPI_COLS.items() if k in df.columns}

GROUPS = ["ALS", "Healthy", "All"]
n_als     = (df["group"] == "ALS").sum()
n_healthy = (df["group"] == "Healthy").sum()

print("=" * 70)
print("PHASE 9 — Step 9.B: Complete Descriptive Statistics")
print("=" * 70)
print(f"\nALS n={n_als}, Healthy n={n_healthy}, Total n={len(df)}")

# ─── Bootstrap CI for median ──────────────────────────────────────────────────
def bootstrap_median_ci(x, n_boot=2000, ci=0.95):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    boot_medians = [np.median(np.random.choice(x, size=len(x), replace=True))
                    for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_medians, alpha*100)), \
           float(np.percentile(boot_medians, (1-alpha)*100))

# ─── Full descriptive stats function ─────────────────────────────────────────
def describe_full(x, label):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {}

    q1, q3   = np.percentile(x, [25, 75])
    iqr_val  = q3 - q1
    ci_lo, ci_hi = bootstrap_median_ci(x)

    # Tukey fence outliers (1.5 × IQR rule)
    fence_lo = q1 - 1.5 * iqr_val
    fence_hi = q3 + 1.5 * iqr_val
    n_outliers = int(np.sum((x < fence_lo) | (x > fence_hi)))

    return {
        "n"           : n,
        "mean"        : round(float(np.mean(x)), 6),
        "SD"          : round(float(np.std(x, ddof=1)), 6),
        "median"      : round(float(np.median(x)), 6),
        "Q1"          : round(float(q1), 6),
        "Q3"          : round(float(q3), 6),
        "IQR"         : round(float(iqr_val), 6),
        "min"         : round(float(np.min(x)), 6),
        "max"         : round(float(np.max(x)), 6),
        "CI95_lo"     : round(ci_lo, 6),
        "CI95_hi"     : round(ci_hi, 6),
        "skewness"    : round(float(stats.skew(x)), 3),
        "kurtosis"    : round(float(stats.kurtosis(x)), 3),
        "n_outliers"  : n_outliers,
        "fence_lo"    : round(float(fence_lo), 6),
        "fence_hi"    : round(float(fence_hi), 6),
    }

# ─── Run descriptive stats ────────────────────────────────────────────────────
all_rows  = []
out_rows  = []

print(f"\n{'─'*70}")
print(f"{'KPI':<25} {'Group':<10} {'n':>4} {'Mean':>10} {'SD':>10} "
      f"{'Median':>10} {'IQR':>10} {'95% CI Median':>20} {'Skew':>6}")
print(f"{'─'*70}")

for kpi, label in KPI_COLS.items():
    for group in GROUPS:
        if group == "All":
            vals = df[kpi].dropna().values
        else:
            vals = df[df["group"] == group][kpi].dropna().values

        desc = describe_full(vals, label)
        if not desc:
            continue

        row = {"kpi": kpi, "label": label, "group": group, **desc}
        all_rows.append(row)

        ci_str = f"[{desc['CI95_lo']:.4f}, {desc['CI95_hi']:.4f}]"
        print(f"  {label:<23} {group:<10} {desc['n']:>4} "
              f"{desc['mean']:>10.4f} {desc['SD']:>10.4f} "
              f"{desc['median']:>10.4f} {desc['IQR']:>10.4f} "
              f"{ci_str:>20} {desc['skewness']:>6.2f}")

        # Track outliers
        if desc["n_outliers"] > 0:
            if group == "All":
                sub_df = df
            else:
                sub_df = df[df["group"] == group]
            outlier_subjects = sub_df[
                (sub_df[kpi] < desc["fence_lo"]) |
                (sub_df[kpi] > desc["fence_hi"])
            ]["subject"].tolist()
            out_rows.append({
                "kpi"       : kpi,
                "label"     : label,
                "group"     : group,
                "n_outliers": desc["n_outliers"],
                "fence_lo"  : desc["fence_lo"],
                "fence_hi"  : desc["fence_hi"],
                "outlier_subjects": "; ".join(outlier_subjects),
            })

    print()

# ─── Outlier summary ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("OUTLIER SUMMARY (Tukey 1.5×IQR fence)")
print(f"{'='*70}")

if out_rows:
    df_out = pd.DataFrame(out_rows)
    for _, row in df_out.iterrows():
        print(f"\n  {row['label']} — {row['group']}:")
        print(f"    n_outliers : {row['n_outliers']}")
        print(f"    fence      : [{row['fence_lo']:.4f}, {row['fence_hi']:.4f}]")
        print(f"    subjects   : {row['outlier_subjects']}")
else:
    print("  No outliers detected by Tukey fence.")
    df_out = pd.DataFrame()

# ─── Table 1 — Paper-ready format ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("TABLE 1 — Paper-ready: Median [IQR] and Mean ± SD")
print(f"{'='*70}")
print(f"\n{'KPI':<28} {'ALS (n=15)':>22} {'Healthy (n=8)':>22}")
print(f"{'':28} {'Median [IQR]':>22} {'Median [IQR]':>22}")
print(f"{'':28} {'Mean ± SD':>22} {'Mean ± SD':>22}")
print("-" * 74)

df_desc = pd.DataFrame(all_rows)

for kpi, label in KPI_COLS.items():
    als_row = df_desc[(df_desc["kpi"]==kpi) & (df_desc["group"]=="ALS")]
    hc_row  = df_desc[(df_desc["kpi"]==kpi) & (df_desc["group"]=="Healthy")]

    if als_row.empty or hc_row.empty:
        continue

    a = als_row.iloc[0]
    h = hc_row.iloc[0]

    # Scientific notation for very small values
    def fmt(v):
        if abs(v) < 0.001:
            return f"{v:.2e}"
        return f"{v:.4f}"

    als_med_iqr = f"{fmt(a['median'])} [{fmt(a['IQR'])}]"
    hc_med_iqr  = f"{fmt(h['median'])} [{fmt(h['IQR'])}]"
    als_mean_sd = f"{fmt(a['mean'])} ± {fmt(a['SD'])}"
    hc_mean_sd  = f"{fmt(h['mean'])} ± {fmt(h['SD'])}"

    print(f"  {label:<26} {als_med_iqr:>22}  {hc_med_iqr:>22}")
    print(f"  {'':26} {als_mean_sd:>22}  {hc_mean_sd:>22}")
    print()

# ─── Distribution shape summary ───────────────────────────────────────────────
print(f"{'='*70}")
print("DISTRIBUTION SHAPE SUMMARY")
print(f"{'='*70}")
print(f"\n{'KPI':<28} {'Group':<10} {'Skewness':>10} {'Kurtosis':>10} "
      f"{'Shape'}")
print("-" * 72)

for _, row in df_desc[df_desc["group"] != "All"].iterrows():
    skew = row["skewness"]
    kurt = row["kurtosis"]
    if abs(skew) < 0.5:   shape = "Approximately symmetric"
    elif abs(skew) < 1.0: shape = "Mildly skewed"
    elif abs(skew) < 2.0: shape = "Moderately skewed"
    else:                  shape = "Heavily skewed"
    if kurt > 3:           shape += " + heavy tails"

    print(f"  {row['label']:<26} {row['group']:<10} "
          f"{skew:>10.3f} {kurt:>10.3f}  {shape}")

# ─── Save ─────────────────────────────────────────────────────────────────────
df_desc.to_csv(PHASE9_STATS / "descriptive_stats.csv", index=False)
if not df_out.empty:
    df_out.to_csv(PHASE9_STATS / "outliers.csv", index=False)

print(f"\nOutputs saved:")
print(f"  descriptive_stats.csv -> full stats per KPI per group")
print(f"  outliers.csv          -> Tukey fence outlier details")
