# alsexo/stats/step_f_exo_effect.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 14); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.step_f_exo_effect`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — Step 9.F (v2): EXO Effect — ALL MUSCLES, ALL TASKS
#
# WHY THIS WAS REWRITTEN
#   The previous version tested only 4 muscles (Extensor, Flexor, Biceps, Triceps)
#   and concluded the exoskeleton "only offloads the biceps". That conclusion was
#   drawn from an INCOMPLETE muscle set: Deltoid, Trapezius and AbdV were recorded
#   but never tested. An upper-limb exoskeleton supports the shoulder/elbow chain,
#   so the shoulder muscles (Deltoid, Trapezius) are precisely the ones most likely
#   to respond. A muscle cannot be declared unaffected if it was never analysed.
#
# WHAT THIS VERSION DOES
#   - Tests ALL 7 recorded muscles + trial duration.
#   - For every task (drinking, lifting, pick&place) and every group (All/ALS/Healthy).
#   - Paired Wilcoxon signed-rank (same subject, EXO vs no-EXO).
#   - TWO FDR corrections reported side by side, for robustness:
#       (a) WITHIN-TASK  : corrected across the muscles inside each task  (primary)
#       (b) GLOBAL       : corrected across every test in the group        (conservative)
#   - A muscle x task summary grid so the anatomical pattern is visible at a glance.
#
# OUTPUT
#   phase_09_stats/step_F_exo_pertask/exo_allmuscles_results.csv
#   phase_09_stats/step_F_exo_pertask/exo_allmuscles_grid.csv
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PHASE6_DIR     = PROCESSED_ROOT / "phase_06_features"
PHASE7_DIR     = PROCESSED_ROOT / "phase_07_kpis"
PHASE9_STATS   = PROCESSED_ROOT / "phase_09_stats" / "step_F_exo_pertask"
PHASE9_STATS.mkdir(parents=True, exist_ok=True)

# Documented exclusion, consistent with the rest of Phase 9
ARTIFACT_SUBJECTS = ["ALS_Subject_15"]   # sensor artifact (extensor +/-2-4 V)

print("=" * 72)
print("PHASE 9 — Step 9.F (v2): EXO effect — ALL muscles, ALL tasks")
print("=" * 72)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def map_group(name):
    n = str(name).lower()
    if "als" in n:     return "ALS"
    if "healthy" in n: return "Healthy"
    return None

def extract_task(tid):
    t = str(tid).lower()
    if "drinking" in t: return "drinking"
    if "lifting"  in t: return "lifting"
    if "pick"     in t: return "pick_place"
    return "other"

def extract_condition(tid):
    t = str(tid).lower()
    if "noexo" in t: return "noexo"
    if "exo"   in t: return "exo"
    return "unknown"

def wilcoxon_effect_r(w_stat, n):
    return float(1 - (4 * w_stat) / (n * (n + 1)))

def effect_label(r):
    a = abs(r)
    if a >= 0.50: return "large"
    if a >= 0.30: return "medium"
    return "small"

def bh_correction(p_vals, alpha=0.05):
    p_vals = np.asarray(p_vals, dtype=float)
    n = len(p_vals)
    if n == 0: return np.array([]), np.array([])
    order = np.argsort(p_vals)
    ranks = np.empty_like(order); ranks[order] = np.arange(1, n + 1)
    p_adj = np.minimum(1.0, p_vals * n / ranks)
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    return p_adj, p_adj <= alpha

# ─── Load Phase 6 features ────────────────────────────────────────────────────
dfs = []
for fpath in sorted(PHASE6_DIR.glob("*__features.parquet")):
    subject = fpath.name.split("__features.parquet")[0]
    d = pd.read_parquet(fpath)
    d["subject"]   = subject
    d["group"]     = map_group(subject)
    d["task"]      = d["trial_id"].apply(extract_task)
    d["condition"] = d["trial_id"].apply(extract_condition)
    dfs.append(d)
df_all = pd.concat(dfs, ignore_index=True)

# Apply the documented exclusion
n_before = df_all["subject"].nunique()
df_all = df_all[~df_all["subject"].isin(ARTIFACT_SUBJECTS)].copy()
print(f"\nSubjects: {df_all['subject'].nunique()} (excluded {ARTIFACT_SUBJECTS}, was {n_before})")

df_exo = df_all[df_all["condition"].isin(["exo", "noexo"])].copy()

# ─── ALL muscles present in the data (not just 4) ─────────────────────────────
MUSCLE_ORDER = ["Trapezius", "Deltoid", "Biceps", "Triceps", "Extensor", "Flexor", "AbdV"]
REGION = {
    "Trapezius": "shoulder/neck", "Deltoid": "shoulder",
    "Biceps": "upper arm", "Triceps": "upper arm",
    "Extensor": "forearm (distal)", "Flexor": "forearm (distal)",
    "AbdV": "hand (distal)",
}
EMG_FEATS = {}
for m in MUSCLE_ORDER:
    col = f"EMG_{m}_RMS"
    if col in df_exo.columns:
        df_exo[col] = pd.to_numeric(df_exo[col], errors="coerce")
        EMG_FEATS[col] = f"RMS {m}"
print(f"Muscles tested ({len(EMG_FEATS)}): {[v.replace('RMS ','') for v in EMG_FEATS.values()]}")

# Trial duration from Phase 7
has_dur = False
try:
    df7 = pd.read_parquet(PHASE7_DIR / "trial_kpis.parquet").copy()
    df7 = df7[~df7["subject"].isin(ARTIFACT_SUBJECTS)].copy()
    df7["group"]     = df7["subject"].apply(map_group)
    df7["task"]      = df7["trial_id"].apply(extract_task)
    df7["condition"] = df7["trial_id"].apply(extract_condition)
    df7["kpi_duration_s"] = pd.to_numeric(df7["kpi_duration_s"], errors="coerce")
    EMG_FEATS["kpi_duration_s"] = "Trial Duration"
    has_dur = True
except Exception as e:
    print(f"  (duration unavailable: {e})")

TASKS  = ["drinking", "lifting", "pick_place"]
GROUPS = ["All", "ALS", "Healthy"]
TASK_LABELS = {"drinking": "Drinking", "lifting": "Lifting", "pick_place": "Pick & Place"}

def subject_values(task, condition, feat):
    """Subject-level median of `feat` for a given task+condition."""
    src = df7 if (feat == "kpi_duration_s" and has_dur) else df_exo
    sub = src[(src["task"] == task) & (src["condition"] == condition)]
    if sub.empty or feat not in sub.columns:
        return pd.DataFrame(columns=["subject", "group", feat])
    return sub.groupby(["subject", "group"])[feat].median().reset_index()

# ─── Run all paired tests ─────────────────────────────────────────────────────
rows = []
for group in GROUPS:
    for task in TASKS:
        for feat, label in EMG_FEATS.items():
            a = subject_values(task, "noexo", feat)
            b = subject_values(task, "exo",   feat)
            if a.empty or b.empty:
                continue
            m = a.merge(b, on=["subject", "group"], suffixes=("_noexo", "_exo"))
            if group != "All":
                m = m[m["group"] == group]
            m = m.dropna(subset=[f"{feat}_noexo", f"{feat}_exo"])
            n = len(m)
            if n < 3:
                continue
            x = m[f"{feat}_noexo"].to_numpy(dtype=float)
            y = m[f"{feat}_exo"].to_numpy(dtype=float)
            diff = y - x
            if np.all(diff == 0):
                continue
            try:
                w, p = stats.wilcoxon(diff, alternative="two-sided")
                r = wilcoxon_effect_r(w, n)
            except Exception:
                continue
            med_no, med_ex = float(np.median(x)), float(np.median(y))
            pct = ((med_ex - med_no) / med_no * 100.0) if med_no != 0 else np.nan
            muscle = label.replace("RMS ", "") if label.startswith("RMS") else "Duration"
            rows.append({
                "group": group, "task": task, "task_label": TASK_LABELS[task],
                "feature": feat, "label": label, "muscle": muscle,
                "region": REGION.get(muscle, "-"),
                "n": n, "median_noexo": med_no, "median_exo": med_ex,
                "pct_change": pct, "p_raw": float(p),
                "r": r, "effect": effect_label(r),
            })

df_res = pd.DataFrame(rows)
if df_res.empty:
    raise RuntimeError("No paired EXO/noEXO tests could be run — check condition labels.")

# ─── TWO FDR corrections (robustness) ─────────────────────────────────────────
# (a) WITHIN-TASK: correct across muscles inside each group x task
df_res["p_fdr_withintask"] = np.nan
for (g, t), idx in df_res.groupby(["group", "task"]).groups.items():
    idx = list(idx)
    padj, _ = bh_correction(df_res.loc[idx, "p_raw"].to_numpy())
    df_res.loc[idx, "p_fdr_withintask"] = padj

# (b) GLOBAL: correct across every test within a group
df_res["p_fdr_global"] = np.nan
for g, idx in df_res.groupby("group").groups.items():
    idx = list(idx)
    padj, _ = bh_correction(df_res.loc[idx, "p_raw"].to_numpy())
    df_res.loc[idx, "p_fdr_global"] = padj

df_res["sig_withintask"] = df_res["p_fdr_withintask"] < 0.05
df_res["sig_global"]     = df_res["p_fdr_global"] < 0.05

# ─── Report ───────────────────────────────────────────────────────────────────
for group in GROUPS:
    g = df_res[df_res["group"] == group]
    if g.empty: continue
    print(f"\n{'='*72}\nGROUP: {group}\n{'='*72}")
    n_tests = len(g)
    print(f"({n_tests} tests: {g['feature'].nunique()} measures x {g['task'].nunique()} tasks)\n")
    print(f"{'Measure':<16}{'Region':<18}{'Task':<14}{'noEXO':>9}{'EXO':>9}{'%chg':>8}"
          f"{'p_raw':>8}{'FDR_task':>10}{'FDR_glob':>10}{'r':>7}  Sig")
    print("-" * 118)
    for task in TASKS:
        sub = g[g["task"] == task].sort_values("p_raw")
        for _, r0 in sub.iterrows():
            star = ""
            if r0["sig_withintask"] and r0["sig_global"]: star = "**"     # survives both
            elif r0["sig_withintask"]:                     star = "*"      # within-task only
            print(f"  {r0['muscle']:<14}{r0['region']:<18}{r0['task_label']:<14}"
                  f"{r0['median_noexo']:>9.4f}{r0['median_exo']:>9.4f}{r0['pct_change']:>7.1f}%"
                  f"{r0['p_raw']:>8.4f}{r0['p_fdr_withintask']:>10.4f}{r0['p_fdr_global']:>10.4f}"
                  f"{r0['r']:>7.3f}  {star}")
        print()

# ─── Muscle x task grid (the anatomical picture) ──────────────────────────────
print(f"\n{'='*72}")
print("MUSCLE x TASK GRID — % change with EXO  (negative = EXO reduces effort)")
print("** = significant under BOTH corrections | * = within-task FDR only")
print(f"{'='*72}")

for group in GROUPS:
    g = df_res[(df_res["group"] == group)]
    if g.empty: continue
    print(f"\n  --- {group} ---")
    print(f"  {'Muscle':<14}{'Region':<18}" + "".join(f"{TASK_LABELS[t]:>18}" for t in TASKS))
    print("  " + "-" * (32 + 18 * len(TASKS)))
    order = [m for m in MUSCLE_ORDER if m in set(g["muscle"])] + \
            (["Duration"] if "Duration" in set(g["muscle"]) else [])
    for muscle in order:
        line = f"  {muscle:<14}{REGION.get(muscle,'-'):<18}"
        for t in TASKS:
            cell = g[(g["muscle"] == muscle) & (g["task"] == t)]
            if cell.empty:
                line += f"{'—':>18}"
            else:
                c = cell.iloc[0]
                mark = "**" if (c["sig_withintask"] and c["sig_global"]) else ("*" if c["sig_withintask"] else "")
                line += f"{c['pct_change']:>15.1f}%{mark:<3}"
        print(line)

# ─── Significant findings summary ─────────────────────────────────────────────
print(f"\n{'='*72}")
print("SIGNIFICANT EXO EFFECTS")
print(f"{'='*72}")
for group in GROUPS:
    g = df_res[(df_res["group"] == group) & (df_res["sig_withintask"])]
    print(f"\n  {group}: {len(g)} significant (within-task FDR)")
    for _, r0 in g.sort_values("p_fdr_withintask").iterrows():
        both = "  [also survives global FDR]" if r0["sig_global"] else ""
        direction = "reduced" if r0["pct_change"] < 0 else "increased"
        print(f"    {r0['muscle']:<12} {r0['task_label']:<14} {direction} {abs(r0['pct_change']):.1f}%"
              f"  p_FDR={r0['p_fdr_withintask']:.4f}  r={r0['r']:.3f}{both}")

# ─── Save ─────────────────────────────────────────────────────────────────────
df_res.to_csv(PHASE9_STATS / "exo_allmuscles_results.csv", index=False)
grid = df_res.pivot_table(index=["group", "muscle", "region"], columns="task_label",
                          values="pct_change", aggfunc="first").reset_index()
grid.to_csv(PHASE9_STATS / "exo_allmuscles_grid.csv", index=False)
print(f"\nSaved:")
print(f"  exo_allmuscles_results.csv -> every test, both FDR corrections")
print(f"  exo_allmuscles_grid.csv    -> muscle x task % change grid")
