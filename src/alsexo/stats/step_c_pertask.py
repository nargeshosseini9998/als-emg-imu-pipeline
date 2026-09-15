# alsexo/stats/step_c_pertask.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 8); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.step_c_pertask`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — Step 9.C: Per-task Statistical Analysis
#
# Purpose:
#   Test whether ALS vs Healthy differences vary across tasks.
#   Tasks: drinking, lifting, pick&place (high + low combined)
#
#   For each task × KPI combination:
#   - Mann-Whitney U test
#   - Effect size (rank-biserial r)
#   - FDR correction within each task
#
#   Additional: Friedman test within ALS group
#   → Do ALS patients show task-dependent variation in EMG?
#
# Output:
#   phase_09_stats/step_C_pertask/pertask_stats.csv
#   phase_09_stats/step_C_pertask/task_comparison_summary.csv
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


PHASE9_STATS = PROCESSED_ROOT / "phase_09_stats" / "step_C_pertask"
PHASE9_STATS.mkdir(parents=True, exist_ok=True)

PHASE6_DIR = PROCESSED_ROOT / "phase_06_features"
PHASE7_DIR = PROCESSED_ROOT / "phase_07_kpis"

print("=" * 65)
print("PHASE 9 — Step 9.C: Per-task Statistical Analysis")
print("=" * 65)

# ─── Load Phase 6 data ────────────────────────────────────────────────────────
def map_group(name):
    n = str(name).lower()
    if "als" in n:     return "ALS"
    if "healthy" in n: return "Healthy"
    return None

def extract_task(tid):
    t = str(tid).lower()
    if "drinking"   in t: return "drinking"
    if "lifting"    in t: return "lifting"
    if "pick"       in t: return "pick_place"
    return "other"

feat_files = sorted(PHASE6_DIR.glob("*__features.parquet"))
dfs = []
for fpath in feat_files:
    subject = fpath.name.split("__features.parquet")[0]
    df_s    = pd.read_parquet(fpath)
    df_s["subject"] = subject
    df_s["group"]   = map_group(subject)
    dfs.append(df_s)

df_all = pd.concat(dfs, ignore_index=True)

# Documented exclusion — consistent with Step 9.0 (sensor artifact, extensor ±2–4 V)
ARTIFACT_SUBJECTS = ["ALS_Subject_15"]
df_all = df_all[~df_all["subject"].isin(ARTIFACT_SUBJECTS)].copy()
print(f"Subjects after exclusion: {df_all['subject'].nunique()}  (excluded {ARTIFACT_SUBJECTS})")

# Filter no-EXO
tid    = df_all["trial_id"].astype(str).str.lower()
noexo  = df_all[tid.str.contains("noexo")].copy()
noexo["task"] = noexo["trial_id"].apply(extract_task)

EMG_FEATS = {
    "EMG_Extensor_RMS": "RMS Extensor (V)",
    "EMG_Flexor_RMS"  : "RMS Flexor (V)",
    "EMG_Biceps_RMS"  : "RMS Biceps (V)",
    "EMG_Triceps_RMS" : "RMS Triceps (V)",
}

for f in EMG_FEATS:
    if f in noexo.columns:
        noexo[f] = pd.to_numeric(noexo[f], errors="coerce")

# Add duration from Phase 7
try:
    df7     = pd.read_parquet(PHASE7_DIR / "trial_kpis.parquet").copy()
    df7     = df7[~df7["subject"].isin(ARTIFACT_SUBJECTS)].copy()
    df7["group"] = df7["subject"].apply(map_group)
    df7["tid"]   = df7["trial_id"].astype(str).str.lower()
    df7["task"]  = df7["trial_id"].apply(extract_task)
    df7_noexo    = df7[df7["tid"].str.contains("noexo")].copy()
    df7_noexo["kpi_duration_s"] = pd.to_numeric(df7_noexo["kpi_duration_s"], errors="coerce")
    EMG_FEATS["kpi_duration_s"] = "Trial Duration (s)"
    has_duration = True
except Exception as e:
    print(f"Duration skipped: {e}")
    has_duration = False

TASKS = ["drinking", "lifting", "pick_place"]
TASK_LABELS = {
    "drinking"  : "Drinking",
    "lifting"   : "Lifting",
    "pick_place": "Pick & Place",
}

# ─── Helper functions ──────────────────────────────────────────────────────────
def rank_biserial_r(U, n1, n2):
    return float(1 - (2 * U) / (n1 * n2))

def effect_label(r):
    a = abs(r)
    if a >= 0.50: return "large"
    if a >= 0.30: return "medium"
    return "small"

def bh_correction(p_vals, alpha=0.05):
    n     = len(p_vals)
    if n == 0: return np.array([]), np.array([])
    order = np.argsort(p_vals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    p_adj = np.minimum(1.0, p_vals * n / ranks)
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    return p_adj, p_adj <= alpha

# ─── Per-task subject-level aggregation ───────────────────────────────────────
def get_task_subj(task, feat):
    """Get subject-level median for a specific task and feature."""
    if feat == "kpi_duration_s" and has_duration:
        src = df7_noexo[df7_noexo["task"] == task]
        agg = src.groupby(["subject","group"])[feat].median().reset_index()
    else:
        src = noexo[noexo["task"] == task]
        agg = src.groupby(["subject","group"])[feat].median().reset_index()
    return agg

# ─── Main analysis ────────────────────────────────────────────────────────────
print(f"\nAnalysis: Mann-Whitney U per task per KPI")
print(f"No-EXO condition only | FDR correction within each task\n")

all_results = []

for task in TASKS:
    task_label = TASK_LABELS[task]
    task_rows  = []

    for feat, feat_label in EMG_FEATS.items():
        agg = get_task_subj(task, feat)
        if agg.empty or feat not in agg.columns:
            continue

        als_vals = agg[agg["group"]=="ALS"][feat].dropna().values
        hc_vals  = agg[agg["group"]=="Healthy"][feat].dropna().values

        if len(als_vals) < 3 or len(hc_vals) < 3:
            continue

        U, p = stats.mannwhitneyu(als_vals, hc_vals, alternative="two-sided")
        r    = rank_biserial_r(U, len(als_vals), len(hc_vals))

        task_rows.append({
            "task"            : task,
            "task_label"      : task_label,
            "kpi"             : feat,
            "label"           : feat_label,
            "n_als"           : len(als_vals),
            "n_healthy"       : len(hc_vals),
            "als_median"      : round(float(np.median(als_vals)), 6),
            "als_iqr"         : round(float(np.percentile(als_vals,75)-np.percentile(als_vals,25)), 6),
            "hc_median"       : round(float(np.median(hc_vals)), 6),
            "hc_iqr"          : round(float(np.percentile(hc_vals,75)-np.percentile(hc_vals,25)), 6),
            "U_stat"          : U,
            "p_raw"           : round(p, 6),
            "r"               : round(r, 3),
            "effect_label"    : effect_label(r),
            "direction"       : "ALS>HC" if np.median(als_vals) > np.median(hc_vals) else "ALS<HC",
        })

    if task_rows:
        p_vals = np.array([row["p_raw"] for row in task_rows])
        p_adj, sig = bh_correction(p_vals)
        for i, row in enumerate(task_rows):
            row["p_fdr"]      = round(float(p_adj[i]), 6)
            row["significant"] = bool(sig[i])
        all_results.extend(task_rows)

df_results = pd.DataFrame(all_results)

# ─── Print results by task ────────────────────────────────────────────────────
for task in TASKS:
    task_label = TASK_LABELS[task]
    sub = df_results[df_results["task"] == task].sort_values("p_raw")

    n_als = sub["n_als"].iloc[0] if not sub.empty else "?"
    n_hc  = sub["n_healthy"].iloc[0] if not sub.empty else "?"

    print(f"\n{'='*65}")
    print(f"Task: {task_label}  (ALS n={n_als}, HC n={n_hc})")
    print(f"{'='*65}")
    print(f"\n{'KPI':<24} {'ALS med':>10} {'HC med':>10} {'p_raw':>7} "
          f"{'p_FDR':>7} {'r':>6} {'Effect':>7} {'Sig':>4} {'Dir':>6}")
    print("-" * 82)

    for _, row in sub.iterrows():
        sig_s = "✓" if row["significant"] else ""
        print(f"  {row['label']:<22} {row['als_median']:>10.4f} {row['hc_median']:>10.4f} "
              f"{row['p_raw']:>7.4f} {row['p_fdr']:>7.4f} {row['r']:>6.3f} "
              f"{row['effect_label']:>7} {sig_s:>4} {row['direction']:>6}")

# ─── Cross-task comparison of effect sizes ────────────────────────────────────
print(f"\n{'='*65}")
print("CROSS-TASK COMPARISON — Effect size (r) per KPI per task")
print(f"{'='*65}")
print(f"\nNegative r = ALS > HC (compensatory hyperactivation)")
print(f"* = significant after FDR\n")

print(f"{'KPI':<24} {'Drinking':>12} {'Lifting':>12} {'Pick&Place':>12}  {'Best task'}")
print("-" * 70)

summary_rows = []
for feat, feat_label in EMG_FEATS.items():
    row_data = {"kpi": feat, "label": feat_label}
    r_vals   = {}
    for task in TASKS:
        sub = df_results[(df_results["task"]==task) & (df_results["kpi"]==feat)]
        if sub.empty:
            r_vals[task] = np.nan
            row_data[f"r_{task}"]   = np.nan
            row_data[f"p_{task}"]   = np.nan
            row_data[f"sig_{task}"] = False
        else:
            r    = sub.iloc[0]["r"]
            p    = sub.iloc[0]["p_fdr"]
            sig  = sub.iloc[0]["significant"]
            r_vals[task]            = r
            row_data[f"r_{task}"]   = r
            row_data[f"p_{task}"]   = p
            row_data[f"sig_{task}"] = sig

    valid_r = {t: r for t, r in r_vals.items() if np.isfinite(r)}
    best    = max(valid_r, key=lambda t: abs(valid_r[t])) if valid_r else "?"
    row_data["best_task"] = best

    def fmt_r(task):
        r   = r_vals.get(task, np.nan)
        sig = df_results[(df_results["task"]==task) & (df_results["kpi"]==feat)]
        s   = "✓" if (not sig.empty and sig.iloc[0]["significant"]) else " "
        return f"{r:+.3f}{s}" if np.isfinite(r) else "  n/a"

    print(f"  {feat_label:<22} {fmt_r('drinking'):>12} "
          f"{fmt_r('lifting'):>12} {fmt_r('pick_place'):>12}  "
          f"{TASK_LABELS.get(best, best)}")
    summary_rows.append(row_data)

# ─── Friedman test: task effect within ALS ────────────────────────────────────
#
# Friedman test is the non-parametric equivalent of repeated-measures ANOVA.
# Question: Do ALS patients show significantly different EMG across tasks?
# This is important because it tells us whether task choice matters for
# exoskeleton control — if ALS EMG varies by task, the controller needs
# task-specific parameters.

print(f"\n{'='*65}")
print("FRIEDMAN TEST: Task effect within ALS group")
print(f"{'='*65}")
print(f"H0: No difference in EMG across tasks within ALS patients")
print(f"(non-parametric repeated measures — subjects are the blocking factor)\n")

friedman_rows = []
for feat, feat_label in EMG_FEATS.items():
    if feat == "kpi_duration_s" and not has_duration:
        continue

    # Build subject × task matrix for ALS
    task_data = {}
    for task in TASKS:
        agg = get_task_subj(task, feat)
        als_agg = agg[agg["group"]=="ALS"][["subject", feat]].dropna()
        task_data[task] = als_agg.set_index("subject")[feat]

    # Only subjects with all 3 tasks
    df_wide = pd.DataFrame(task_data).dropna()
    if len(df_wide) < 4:
        print(f"  {feat_label:<28} insufficient data (n={len(df_wide)})")
        continue

    groups_friedman = [df_wide[t].values for t in TASKS]
    stat, p = stats.friedmanchisquare(*groups_friedman)

    sig = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"  {feat_label:<28} chi2={stat:.3f}  p={p:.4f}  [{sig}]  n_subjects={len(df_wide)}")

    # If significant, post-hoc Wilcoxon pairwise
    if p < 0.05:
        print(f"    Post-hoc Wilcoxon (pairwise):")
        pairs = [("drinking","lifting"), ("drinking","pick_place"), ("lifting","pick_place")]
        ph_p  = []
        for t1, t2 in pairs:
            d1 = df_wide[t1].values
            d2 = df_wide[t2].values
            try:
                _, pp = stats.wilcoxon(d1, d2, alternative="two-sided")
            except:
                pp = np.nan
            ph_p.append(pp)

        # Bonferroni correction for 3 pairs
        ph_p_adj = [min(p_*3, 1.0) for p_ in ph_p]
        for (t1, t2), pp, pp_adj in zip(pairs, ph_p, ph_p_adj):
            s = "✓" if pp_adj < 0.05 else ""
            print(f"    {TASK_LABELS[t1]} vs {TASK_LABELS[t2]}: "
                  f"p={pp:.4f}  p_Bonf={pp_adj:.4f} {s}")

    friedman_rows.append({
        "kpi"          : feat,
        "label"        : feat_label,
        "n_subjects"   : len(df_wide),
        "friedman_chi2": round(stat, 3),
        "p_friedman"   : round(p, 4),
        "significant"  : p < 0.05,
    })

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SUMMARY — Best discriminative task per KPI")
print(f"{'='*65}")

sig_any = df_results[df_results["significant"]]
print(f"\nSignificant findings across all tasks (p_FDR < 0.05):")
if len(sig_any):
    for _, row in sig_any.sort_values("p_fdr").iterrows():
        print(f"  {row['task_label']:<15} {row['label']:<25} "
              f"p_FDR={row['p_fdr']:.4f}  r={row['r']:.3f} [{row['effect_label']}]")
else:
    print("  None reached FDR significance within individual tasks")
    print("  (FDR correction within each task is strict with small n)")

print(f"\nLargest effect sizes per task (top 2):")
for task in TASKS:
    sub = df_results[df_results["task"]==task].sort_values("r", key=abs, ascending=False)
    print(f"\n  {TASK_LABELS[task]}:")
    for _, row in sub.head(2).iterrows():
        sig_s = "✓" if row["significant"] else "(trending)"
        print(f"    {row['label']:<25} r={row['r']:+.3f} [{row['effect_label']}] {sig_s}")

# ─── Save ─────────────────────────────────────────────────────────────────────
df_results.to_csv(PHASE9_STATS / "pertask_stats.csv", index=False)
pd.DataFrame(summary_rows).to_csv(PHASE9_STATS / "task_comparison_summary.csv", index=False)
if friedman_rows:
    pd.DataFrame(friedman_rows).to_csv(PHASE9_STATS / "friedman_results.csv", index=False)

print(f"\nOutputs saved:")
print(f"  pertask_stats.csv           -> per task × KPI results")
print(f"  task_comparison_summary.csv -> effect size table")
print(f"  friedman_results.csv        -> within-ALS task variation")
