# alsexo/stats/step_h_selective_involvement.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 23); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.step_h_selective_involvement`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — Step 9.H: Selective (Dissociated) Muscle Involvement in ALS
#
# RESEARCH QUESTION
#   In ALS, dissociated patterns of muscle involvement have been described —
#   preferential impairment of BICEPS relative to TRICEPS, and of distal
#   EXTENSOR relative to FLEXOR muscles. These patterns are not universal and
#   may depend on clinical phenotype and disease stage.
#   Do OUR participants show a similar selective pattern?
#
# METHODOLOGICAL DESIGN (and why)
#
#   1) MEASURE: absolute RMS (volts).
#      Within-subject normalized envelopes CANNOT be used here: normalising each
#      muscle to its own peak sets every muscle's maximum to 1 by construction,
#      erasing exactly the between-muscle imbalance we are trying to measure.
#
#   2) THE CONFOUND, AND THE FIX:
#      Absolute EMG amplitude depends on anatomy (electrode siting, subcutaneous
#      tissue, muscle size), so a biceps/triceps ratio is NOT expected to be 1.0
#      even in a healthy person. The raw ratio is therefore uninterpretable alone.
#      FIX: the HEALTHY GROUP IS THE REFERENCE. We do not ask "is the ALS ratio
#      above 1?" — we ask "is the ALS ratio DIFFERENT FROM the healthy ratio?"
#      The anatomical confound affects both groups equally and cancels out.
#
#   3) LOG-RATIO, not raw ratio:
#      Raw ratios are skewed and unstable (small denominators explode). The
#      log-ratio log(A/B) is symmetric — a 2x shift up and a 2x shift down are
#      equidistant from zero — and is the standard treatment for ratio data.
#      log(A/B) > 0  =>  A relatively higher than B.
#
#   4) THREE COMPLEMENTARY ANALYSES:
#      A) RATIO TEST      — is the ALS log-ratio shifted vs healthy? (the direct test)
#      B) EFFECT-SIZE X-CHECK — is one muscle of the pair more affected than its partner?
#      C) PER-SUBJECT PREVALENCE — how MANY individual patients show the pattern?
#         (directly addresses "not observed in all patients")
#
# OUTPUT
#   phase_09_stats/step_H_selective/selective_ratio_tests.csv
#   phase_09_stats/step_H_selective/selective_per_subject.csv
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PHASE6_DIR     = PROCESSED_ROOT / "phase_06_features"
OUT_DIR        = PROCESSED_ROOT / "phase_09_stats" / "step_H_selective"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_SUBJECTS = ["ALS_Subject_15"]   # documented sensor artifact

# The two antagonist pairs from the literature.
# Convention: log(NUMERATOR / DENOMINATOR).
#   Literature: biceps preferentially impaired vs triceps
#               extensor preferentially impaired vs flexor
PAIRS = [
    ("Biceps",   "Triceps", "Elbow: Biceps / Triceps"),
    ("Extensor", "Flexor",  "Wrist: Extensor / Flexor"),
]

TASKS = ["drinking", "lifting", "pick_place"]
TASK_LABELS = {"drinking": "Drinking", "lifting": "Lifting", "pick_place": "Pick & Place"}

print("=" * 76)
print("PHASE 9 — Step 9.H: Selective (dissociated) muscle involvement")
print("=" * 76)

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

def rank_biserial(u, n1, n2):
    return float(1 - (2 * u) / (n1 * n2))

def effect_label(r):
    a = abs(r)
    if a >= 0.50: return "large"
    if a >= 0.30: return "medium"
    return "small"

def bh_correction(p_vals, alpha=0.05):
    p_vals = np.asarray(p_vals, dtype=float)
    n = len(p_vals)
    if n == 0: return np.array([])
    order = np.argsort(p_vals)
    ranks = np.empty_like(order); ranks[order] = np.arange(1, n + 1)
    p_adj = np.minimum(1.0, p_vals * n / ranks)
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    return p_adj

# ─── Load Phase 6 ─────────────────────────────────────────────────────────────
dfs = []
for fp in sorted(PHASE6_DIR.glob("*__features.parquet")):
    subject = fp.name.split("__features.parquet")[0]
    d = pd.read_parquet(fp)
    d["subject"]   = subject
    d["group"]     = map_group(subject)
    d["task"]      = d["trial_id"].apply(extract_task)
    d["condition"] = d["trial_id"].apply(extract_condition)
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
df = df[~df["subject"].isin(ARTIFACT_SUBJECTS)].copy()

# Analyse the NO-EXO condition: the unassisted, native neuromuscular pattern.
df = df[df["condition"] == "noexo"].copy()
print(f"\nSubjects: {df['subject'].nunique()}  (no-EXO trials; {ARTIFACT_SUBJECTS} excluded)")
print(f"  ALS: {df[df['group']=='ALS']['subject'].nunique()}   "
      f"Healthy: {df[df['group']=='Healthy']['subject'].nunique()}")
print("\nMeasure: ABSOLUTE RMS (volts). Log-ratio log(A/B). "
      "Healthy group = reference for the anatomical baseline.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS A — RATIO TEST:  is the ALS log-ratio shifted relative to healthy?
# ═══════════════════════════════════════════════════════════════════════════════
def subject_logratio(sub_df, num, den):
    """Subject-level log-ratio from median absolute RMS of each muscle."""
    cn, cd = f"EMG_{num}_RMS", f"EMG_{den}_RMS"
    if cn not in sub_df.columns or cd not in sub_df.columns:
        return pd.DataFrame()
    agg = sub_df.groupby(["subject", "group"])[[cn, cd]].median().reset_index()
    agg = agg[(agg[cn] > 0) & (agg[cd] > 0)]           # log needs positive values
    agg["ratio"]     = agg[cn] / agg[cd]
    agg["log_ratio"] = np.log(agg["ratio"])
    return agg

rows = []
per_subject_all = []

print("=" * 76)
print("ANALYSIS A — Is the ALS antagonist balance shifted vs healthy?")
print("=" * 76)

for num, den, pair_label in PAIRS:
    print(f"\n{'-'*76}\n{pair_label}\n{'-'*76}")
    print(f"{'Scope':<16}{'ALS ratio':>11}{'HC ratio':>11}{'ALS log':>10}{'HC log':>9}"
          f"{'p_raw':>9}{'r':>8}  Effect")
    print("-" * 76)

    scopes = [("Overall", df)] + [(TASK_LABELS[t], df[df["task"] == t]) for t in TASKS]

    for scope_name, scope_df in scopes:
        agg = subject_logratio(scope_df, num, den)
        if agg.empty:
            continue
        als = agg[agg["group"] == "ALS"]["log_ratio"].dropna()
        hc  = agg[agg["group"] == "Healthy"]["log_ratio"].dropna()
        if len(als) < 3 or len(hc) < 3:
            continue
        u, p = stats.mannwhitneyu(als, hc, alternative="two-sided")
        r = rank_biserial(u, len(als), len(hc))
        als_ratio = float(np.exp(als.median()))
        hc_ratio  = float(np.exp(hc.median()))
        print(f"  {scope_name:<14}{als_ratio:>11.3f}{hc_ratio:>11.3f}"
              f"{als.median():>10.3f}{hc.median():>9.3f}{p:>9.4f}{r:>8.3f}  {effect_label(r)}")
        rows.append({
            "pair": pair_label, "numerator": num, "denominator": den,
            "scope": scope_name,
            "n_als": len(als), "n_hc": len(hc),
            "als_ratio_median": als_ratio, "hc_ratio_median": hc_ratio,
            "als_logratio_median": float(als.median()),
            "hc_logratio_median": float(hc.median()),
            "shift": float(als.median() - hc.median()),
            "p_raw": float(p), "r": r, "effect": effect_label(r),
        })
        if scope_name == "Overall":
            a = agg.copy(); a["pair"] = pair_label
            per_subject_all.append(a)

df_ratio = pd.DataFrame(rows)

# FDR correction within each pair (across the 4 scopes)
df_ratio["p_fdr"] = np.nan
for pair, idx in df_ratio.groupby("pair").groups.items():
    idx = list(idx)
    df_ratio.loc[idx, "p_fdr"] = bh_correction(df_ratio.loc[idx, "p_raw"].to_numpy())
df_ratio["significant"] = df_ratio["p_fdr"] < 0.05

print(f"\n{'='*76}")
print("ANALYSIS A — after FDR correction (within each pair, across scopes)")
print(f"{'='*76}")
print(f"\n{'Pair':<28}{'Scope':<15}{'p_raw':>9}{'p_FDR':>9}{'r':>8}  Sig")
print("-" * 76)
for _, r0 in df_ratio.iterrows():
    star = "***" if r0["p_fdr"] < 0.001 else ("**" if r0["p_fdr"] < 0.01 else ("*" if r0["p_fdr"] < 0.05 else ""))
    print(f"  {r0['pair']:<26}{r0['scope']:<15}{r0['p_raw']:>9.4f}{r0['p_fdr']:>9.4f}"
          f"{r0['r']:>8.3f}  {star}")

# Interpretation of direction
print(f"\n{'='*76}")
print("INTERPRETATION — direction of any shift")
print(f"{'='*76}")
for pair, sub in df_ratio.groupby("pair"):
    ov = sub[sub["scope"] == "Overall"]
    if ov.empty: continue
    o = ov.iloc[0]
    num, den = o["numerator"], o["denominator"]
    shift = o["shift"]
    sig = o["significant"]
    print(f"\n  {pair}")
    print(f"    ALS ratio {o['als_ratio_median']:.3f}  vs  Healthy ratio {o['hc_ratio_median']:.3f}")
    if not sig:
        print(f"    -> NO significant shift (p_FDR={o['p_fdr']:.3f}). The antagonist balance in ALS")
        print(f"       is not distinguishable from healthy: no evidence of SELECTIVE involvement")
        print(f"       of {num} relative to {den} in this cohort, by EMG amplitude.")
    else:
        if shift > 0:
            print(f"    -> ALS shows a RELATIVELY HIGHER {num} vs {den} than healthy")
            print(f"       (p_FDR={o['p_fdr']:.4f}, r={o['r']:.3f}) -> selective imbalance detected.")
        else:
            print(f"    -> ALS shows a RELATIVELY LOWER {num} vs {den} than healthy")
            print(f"       (p_FDR={o['p_fdr']:.4f}, r={o['r']:.3f}) -> selective imbalance detected.")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS B — EFFECT-SIZE CROSS-CHECK
#   Is one muscle of the pair more affected (ALS vs HC) than its partner?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*76}")
print("ANALYSIS B — Effect-size cross-check: which muscle is more affected?")
print(f"{'='*76}")
print("(Each muscle tested ALS vs HC separately; compare the effect sizes.)\n")
print(f"{'Muscle':<14}{'ALS med':>10}{'HC med':>10}{'ratio':>8}{'p_raw':>9}{'r':>8}  Effect")
print("-" * 70)

muscles = sorted({m for p in PAIRS for m in p[:2]})
eff = {}
for m in ["Biceps", "Triceps", "Extensor", "Flexor"]:
    col = f"EMG_{m}_RMS"
    if col not in df.columns: continue
    agg = df.groupby(["subject", "group"])[col].median().reset_index()
    a = agg[agg["group"] == "ALS"][col].dropna()
    h = agg[agg["group"] == "Healthy"][col].dropna()
    if len(a) < 3 or len(h) < 3: continue
    u, p = stats.mannwhitneyu(a, h, alternative="two-sided")
    r = rank_biserial(u, len(a), len(h))
    ratio = float(a.median() / h.median()) if h.median() > 0 else np.nan
    eff[m] = {"r": r, "p": float(p), "ratio": ratio,
              "als": float(a.median()), "hc": float(h.median())}
    print(f"  {m:<12}{a.median():>10.4f}{h.median():>10.4f}{ratio:>8.2f}x{p:>8.4f}{r:>8.3f}  {effect_label(r)}")

print("\nPairwise comparison of effect magnitude:")
print("NOTE: the rank-biserial r SATURATES at 1.0 when the groups do not overlap,")
print("      so it cannot rank magnitudes. The FOLD-CHANGE (ALS median / HC median)")
print("      is used as the primary magnitude measure; r is reported for context.\n")
for num, den, pair_label in PAIRS:
    if num in eff and den in eff:
        fn, fd = eff[num]["ratio"], eff[den]["ratio"]
        rn, rd = abs(eff[num]["r"]), abs(eff[den]["r"])
        print(f"  {pair_label}")
        print(f"    {num:<10} {fn:.2f}x elevated   (|r|={rn:.3f}, p={eff[num]['p']:.4f})")
        print(f"    {den:<10} {fd:.2f}x elevated   (|r|={rd:.3f}, p={eff[den]['p']:.4f})")
        if not (np.isfinite(fn) and np.isfinite(fd) and fd > 0):
            print("    -> fold-change not computable.\n")
            continue
        # relative selectivity: how much more elevated is the numerator muscle?
        sel = fn / fd
        more = num if fn > fd else den
        print(f"    -> {more} is more elevated. Selectivity index = {sel:.2f}"
              f"  (fold-change {num} / fold-change {den})")
        if 0.80 <= sel <= 1.25:
            print(f"       Both muscles are elevated to a COMPARABLE degree (index near 1)")
            print(f"       -> this does NOT support selective involvement of one over the other.")
        else:
            direction = num if sel > 1 else den
            print(f"       {direction} is disproportionately elevated -> consistent with")
            print(f"       SELECTIVE involvement, in the direction reported in the literature"
                  f"{' (as expected)' if direction == num else ' (OPPOSITE to expectation)'}.")
        print()

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS C — PER-SUBJECT PREVALENCE
#   "These patterns are not observed in all patients" — so count how many show it.
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*76}")
print("ANALYSIS C — Per-subject prevalence: how many patients show the pattern?")
print(f"{'='*76}")
print("A patient is counted as showing the dissociation if their log-ratio exceeds")
print("the healthy reference range (median + 1.96 SD of the healthy log-ratios).\n")

per_rows = []
for num, den, pair_label in PAIRS:
    agg = subject_logratio(df, num, den)
    if agg.empty: continue
    hc = agg[agg["group"] == "Healthy"]["log_ratio"].dropna()
    if len(hc) < 3: continue
    hc_mean, hc_sd = float(hc.mean()), float(hc.std(ddof=1))
    upper = hc_mean + 1.96 * hc_sd
    lower = hc_mean - 1.96 * hc_sd

    als = agg[agg["group"] == "ALS"].copy()
    als["above_ref"] = als["log_ratio"] > upper
    als["below_ref"] = als["log_ratio"] < lower
    n_above = int(als["above_ref"].sum())
    n_below = int(als["below_ref"].sum())
    n_tot   = len(als)

    print(f"\n  {pair_label}")
    print(f"    Healthy reference log-ratio: {hc_mean:.3f} +/- {hc_sd:.3f}  "
          f"(95% range: {lower:.3f} to {upper:.3f})")
    print(f"    ALS patients ABOVE the healthy range: {n_above}/{n_tot} "
          f"({100*n_above/n_tot:.0f}%)  -> relatively higher {num}")
    print(f"    ALS patients BELOW the healthy range: {n_below}/{n_tot} "
          f"({100*n_below/n_tot:.0f}%)  -> relatively higher {den}")
    print(f"    ALS patients WITHIN the healthy range: {n_tot-n_above-n_below}/{n_tot} "
          f"({100*(n_tot-n_above-n_below)/n_tot:.0f}%)  -> no dissociation")

    print(f"\n    Per-patient detail (ratio {num}/{den}):")
    for _, r0 in als.sort_values("log_ratio", ascending=False).iterrows():
        flag = "^ ABOVE" if r0["above_ref"] else ("v BELOW" if r0["below_ref"] else "  within")
        print(f"      {r0['subject']:<22} ratio={r0['ratio']:>7.3f}  log={r0['log_ratio']:>7.3f}   {flag}")
        per_rows.append({
            "pair": pair_label, "subject": r0["subject"], "group": "ALS",
            "ratio": float(r0["ratio"]), "log_ratio": float(r0["log_ratio"]),
            "hc_ref_mean": hc_mean, "hc_ref_sd": hc_sd,
            "above_healthy_range": bool(r0["above_ref"]),
            "below_healthy_range": bool(r0["below_ref"]),
        })

# ─── Save ─────────────────────────────────────────────────────────────────────
df_ratio.to_csv(OUT_DIR / "selective_ratio_tests.csv", index=False)
pd.DataFrame(per_rows).to_csv(OUT_DIR / "selective_per_subject.csv", index=False)
print(f"\n\nSaved:")
print(f"  selective_ratio_tests.csv  -> group-level ratio tests (all scopes, FDR)")
print(f"  selective_per_subject.csv  -> per-patient dissociation status")
