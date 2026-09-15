# alsexo/stats/step_g2_gap_closure.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 21); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.step_g2_gap_closure`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — Step 9.G: Gap-Closure Analysis
#   "Can the exoskeleton make ALS patients function like healthy subjects?"
#
# Reframes the professor's question as GAP CLOSURE, not within-group change:
#   reference   = Healthy, no-exo            (what "normal" looks like)
#   baseline    = ALS, no-exo                (the gap to close)
#   treated     = ALS, with-exo              (did it move toward normal?)
#   target_exo  = Healthy, with-exo          (does the target itself move?)
#
# Two definitions of "closed the gap", shown together:
#   (1) BAND TEST   : does the ALS+exo median fall inside the Healthy no-exo IQR?
#   (2) % GAP CLOSED: how much of the ALS-vs-Healthy gap did the exo remove?
#         gap_closed% = (|ALS_noexo - HC_noexo| - |ALS_exo - HC_noexo|)
#                       / |ALS_noexo - HC_noexo| * 100
#
# KPIs: ALL between-subject comparable measures
#   effort   : absolute RMS per muscle (+ distal/mean if present)
#   motion   : peak/mean angular velocity, jerk (smoothness)
#   function : trial duration
#   guarding : co-activation (one representative)
#
# Firewall note: the BETWEEN-group leg (ALS vs HC) requires comparable
# absolute KPIs — within-subject normalized KPIs are NOT used here.
#
# Output:
#   phase_09_stats/step_G_gap_closure/gap_closure_results.csv
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

SUBJ_KPI_PATH  = PROCESSED_ROOT / "phase_07_kpis" / "subject_kpis.parquet"   # fallback below
TRIAL_KPI_PATH = PROCESSED_ROOT / "phase_07_kpis" / "trial_kpis.parquet"
PHASE6_DIR     = PROCESSED_ROOT / "phase_06_features"
OUT_DIR        = PROCESSED_ROOT / "phase_09_stats" / "step_G_gap_closure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_SUBJECTS = ["ALS_Subject_15"]

def map_group(name):
    n = str(name).lower()
    if "als" in n:     return "ALS"
    if "healthy" in n: return "Healthy"
    return None

def extract_condition(tid):
    t = str(tid).lower()
    if "noexo" in t: return "noexo"
    if "exo"   in t: return "exo"
    return "unknown"

# ──────────────────────────────────────────────────────────────────────────────
# Build a long subject×condition table of comparable KPIs.
#   - absolute RMS per muscle : from Phase 6 features (the comparable effort vars)
#   - motion / duration / coact : from Phase 7 trial_kpis if available
# Each value = subject-level median across that subject's trials in that condition.
# ──────────────────────────────────────────────────────────────────────────────

# ---- 1) absolute RMS per muscle, per condition, from Phase 6 ----
RMS_MUSCLES = ["Extensor", "Flexor", "Biceps", "Triceps"]
rms_rows = []
for fp in sorted(PHASE6_DIR.glob("*__features.parquet")):
    subject = fp.name.split("__features.parquet")[0]
    if subject in ARTIFACT_SUBJECTS:
        continue
    d = pd.read_parquet(fp)
    d["condition"] = d["trial_id"].apply(extract_condition)
    for cond in ["noexo", "exo"]:
        dc = d[d["condition"] == cond]
        if dc.empty:
            continue
        rec = {"subject": subject, "group": map_group(subject), "condition": cond}
        for m in RMS_MUSCLES:
            col = f"EMG_{m}_RMS"
            rec[f"rms_{m}"] = float(pd.to_numeric(dc[col], errors="coerce").median()) if col in dc.columns else np.nan
        rms_rows.append(rec)
rms_long = pd.DataFrame(rms_rows)

# ---- 2) motion / duration / coact, per condition, from Phase 7 trial_kpis ----
# We auto-detect which comparable columns exist (names vary across pipelines).
CANDIDATE_KPIS = {
    "kpi_peak_gyro_mag"          : ("Peak angular velocity", "motion"),
    "kpi_mean_gyro_mag"          : ("Mean angular velocity", "motion"),
    "kpi_mean_jerk"              : ("Mean jerk (smoothness)", "motion"),
    "kpi_duration_s"            : ("Trial duration",         "function"),
    "kpi_coact_overlap_bic_tric" : ("Co-activation",          "guarding"),
}
extra_long = pd.DataFrame()
if TRIAL_KPI_PATH.exists():
    t = pd.read_parquet(TRIAL_KPI_PATH).copy()
    t["group"]     = t["subject"].apply(map_group)
    t["condition"] = t["trial_id"].apply(extract_condition)
    t = t[~t["subject"].isin(ARTIFACT_SUBJECTS)]
    present = [c for c in CANDIDATE_KPIS if c in t.columns]
    for c in present:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    if present:
        extra_long = (t[t["condition"].isin(["noexo","exo"])]
                      .groupby(["subject","group","condition"])[present]
                      .median().reset_index())

# ---- merge effort + extras into one wide-per-condition table ----
long = rms_long.merge(extra_long, on=["subject","group","condition"], how="outer") \
       if not extra_long.empty else rms_long

# Define the KPI list actually available
KPI_DEFS = []
for m in RMS_MUSCLES:
    KPI_DEFS.append((f"rms_{m}", f"RMS {m}", "effort", "lower"))
for c,(lab,dom) in CANDIDATE_KPIS.items():
    if c in long.columns:
        # duration & coact: "better" direction is toward healthy, handled via gap math
        KPI_DEFS.append((c, lab, dom, "lower"))

KPI_DEFS = [(col,lab,dom,dirn) for (col,lab,dom,dirn) in KPI_DEFS if col in long.columns]

# ──────────────────────────────────────────────────────────────────────────────
# Gap-closure computation
# ──────────────────────────────────────────────────────────────────────────────
def grp_cond(col, group, cond):
    s = long[(long["group"]==group) & (long["condition"]==cond)][col].dropna()
    return s.values

rows = []
for col, label, domain, _dirn in KPI_DEFS:
    hc_noexo  = grp_cond(col, "Healthy", "noexo")
    als_noexo = grp_cond(col, "ALS",     "noexo")
    als_exo   = grp_cond(col, "ALS",     "exo")
    hc_exo    = grp_cond(col, "Healthy", "exo")
    if min(len(hc_noexo), len(als_noexo), len(als_exo)) < 3:
        continue

    hc_med   = float(np.median(hc_noexo))
    hc_q1, hc_q3 = np.percentile(hc_noexo, [25, 75])
    als_b    = float(np.median(als_noexo))   # ALS baseline
    als_t    = float(np.median(als_exo))     # ALS treated (exo)

    gap_before = als_b - hc_med
    gap_after  = als_t - hc_med
    # % of the gap removed (positive = moved toward healthy)
    pct_closed = (abs(gap_before) - abs(gap_after)) / abs(gap_before) * 100 if gap_before != 0 else np.nan

    # BAND test: does ALS+exo median fall within Healthy no-exo IQR?
    in_band_before = (hc_q1 <= als_b <= hc_q3)
    in_band_after  = (hc_q1 <= als_t <= hc_q3)
    entered_band   = (not in_band_before) and in_band_after

    # Is the ALS-vs-HC gap even significant at baseline? (worth closing only if so)
    try:
        _, p_base = stats.mannwhitneyu(als_noexo, hc_noexo, alternative="two-sided")
    except Exception:
        p_base = np.nan
    # Residual gap after exo: still different from healthy?
    try:
        _, p_resid = stats.mannwhitneyu(als_exo, hc_noexo, alternative="two-sided")
    except Exception:
        p_resid = np.nan

    rows.append({
        "kpi": col, "label": label, "domain": domain,
        "HC_noexo_med": round(hc_med,4),
        "HC_IQR": f"[{hc_q1:.4f}, {hc_q3:.4f}]",
        "ALS_noexo_med": round(als_b,4),
        "ALS_exo_med": round(als_t,4),
        "gap_before": round(gap_before,4),
        "gap_after": round(gap_after,4),
        "pct_gap_closed": round(pct_closed,1) if np.isfinite(pct_closed) else np.nan,
        "ALS_in_HC_band_before": in_band_before,
        "ALS_in_HC_band_after": in_band_after,
        "entered_band_with_exo": entered_band,
        "p_gap_baseline": round(p_base,4) if np.isfinite(p_base) else np.nan,
        "p_gap_residual": round(p_resid,4) if np.isfinite(p_resid) else np.nan,
    })

res = pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────
print("="*78)
print("PHASE 9 — Step 9.G: Gap-Closure  ('does exo make ALS look healthy?')")
print("="*78)
print(f"\nComparable KPIs analysed: {len(res)}  (firewall: between-group leg uses absolute/comparable only)")
print("Positive % gap closed = ALS+exo moved toward the Healthy reference.\n")

order = {"effort":0, "motion":1, "function":2, "guarding":3}
res = res.sort_values(by=["domain","kpi"], key=lambda s: s.map(order) if s.name=="domain" else s)

print(f"{'KPI':<26}{'dom':<9}{'HC':>9}{'ALS':>9}{'ALS+exo':>9}{'%closed':>9}  band  baseline→residual")
print("-"*92)
for _,x in res.iterrows():
    band = "ENTERS" if x["entered_band_with_exo"] else ("in" if x["ALS_in_HC_band_after"] else "out")
    sig_base  = "*" if (np.isfinite(x['p_gap_baseline']) and x['p_gap_baseline']<0.05) else " "
    sig_resid = "*" if (np.isfinite(x['p_gap_residual']) and x['p_gap_residual']<0.05) else " "
    pc = f"{x['pct_gap_closed']:+.0f}%" if np.isfinite(x['pct_gap_closed']) else "  n/a"
    print(f"  {x['label']:<24}{x['domain']:<9}{x['HC_noexo_med']:>9.4f}{x['ALS_noexo_med']:>9.4f}"
          f"{x['ALS_exo_med']:>9.4f}{pc:>9}  {band:<6} p={x['p_gap_baseline']}{sig_base} → p={x['p_gap_residual']}{sig_resid}")

print("\nReading guide:")
print("  band=ENTERS : exo moved ALS INTO the healthy IQR for this KPI (strong 'like healthy')")
print("  baseline p* : ALS truly differed from healthy WITHOUT exo (a real gap existed)")
print("  residual p  : if no longer * , the ALS+exo group is statistically indistinguishable from healthy")
print("  %closed     : >0 toward healthy, <0 exo widened the gap (often: exo helped healthy more)")

res.to_csv(OUT_DIR / "gap_closure_results.csv", index=False)
print(f"\nSaved -> {OUT_DIR/'gap_closure_results.csv'}")
