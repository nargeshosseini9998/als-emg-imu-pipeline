# alsexo/kpis.py
# Phase 7 (v4) — trial- and subject-level KPIs + NMD index + KPI redundancy check.
# Extracted verbatim from 07_v4_KPIs.ipynb (cells 1-5 and 9); paths replaced by alsexo.paths.
# Entry points: run_phase7_all_subjects_v4(), run_redundancy_check()

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# ===== Cell 1: Imports, paths, config =====
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PHASE3_NORM_DIR = PROCESSED_ROOT / "phase_03_normalization" / "normalized_env_per_subject"
PHASE6_DIR      = PROCESSED_ROOT / "phase_06_features"
PHASE7_DIR      = PROCESSED_ROOT / "phase_07_kpis"
PHASE7_DIR.mkdir(parents=True, exist_ok=True)

TRIAL_OUT     = PHASE7_DIR / "trial_kpis.parquet"
SUBJ_OUT      = PHASE7_DIR / "subject_kpis.parquet"
PHASE7_QC_CSV = PHASE7_DIR / "phase7_qc_report.csv"

# Canonical sensor mapping (same policy as Phase 6)
_P7_RAW = {
    "BicBrachii": "Biceps", "Biceps": "Biceps", "bicipite": "Biceps",
    "TricBrachii": "Triceps", "Triceps": "Triceps", "tricipite": "Triceps",
    "DeltMed": "Deltoid", "Deltoid": "Deltoid", "Deltoide": "Deltoid", "deltoide": "Deltoid",
    "Trap": "Trapezius", "TrapDesc": "Trapezius", "Trap_Desc": "Trapezius",
    "Trap Desc": "Trapezius", "Trap-Desc": "Trapezius",
    "Trapezio": "Trapezius", "Trapezius": "Trapezius", "trapezio": "Trapezius",
    "ExtCarpRad": "Extensor", "Extensor": "Extensor", "Estensore carpo": "Extensor", "estensore": "Extensor", "EXT": "Extensor",
    "FlexCarpRad": "Flexor", "Flexor": "Flexor", "Flessore carpo": "Flexor", "flessore": "Flexor", "Flex": "Flexor",
    "Abduttore V": "AbdV", "AbdV": "AbdV", "abdV": "AbdV", "AbdVfing": "AbdV", "AbdVFing": "AbdV",
    "Abd": "AbdV", "Abd5": "AbdV", "abd5": "AbdV", "Abd 5 dito": "AbdV", "abd 5 dito": "AbdV",
    "IntDors": "IntDors", "interdors": "IntDors", "inter dors": "IntDors", "Primo Inter Dors": "IntDors", "InterV": "IntDors",
}
def _p7_norm(s):
    s = str(s).strip().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).lower()
P7_MAP = {_p7_norm(k): v for k, v in _P7_RAW.items()}
def canon_sensor_phase7(x):
    raw = re.sub(r"\s+", " ", str(x).strip().replace("_", " ").replace("-", " "))
    return P7_MAP.get(_p7_norm(raw), raw)

CO_ACT_MUSCLES = ("Biceps", "Triceps")
DUTY_TH = 0.05
MIN_SAMPLES_PER_MUSCLE = 200
MIN_SAMPLES_ANY_EMG = 200
DISTAL_MUSCLES = ("Extensor", "Flexor")   # effort side of NMD = the discriminating muscles
META_COLS = ["group","Group","label","class","exo","EXO","condition","Condition",
             "session","Session","test_session","Test","task","task_type","Task","movement",
             "weight","Weight","level","pick_level","side","arm","trial_type"]
print("Paths & config loaded.")

# ===== Cell 2: Math helpers (robust) =====
def pick_one(series):
    if series is None: return np.nan
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if len(s) else np.nan

def pick_mode(series):
    if series is None: return None
    s = series.dropna().astype(str)
    if len(s) == 0: return None
    vc = s.value_counts()
    return str(vc.index[0]) if len(vc) else None

def env_pos(x):
    return np.maximum(np.asarray(x, dtype=float), 0.0)

def trapz_auc(t, x):
    t = np.asarray(t, dtype=float); x = np.asarray(x, dtype=float)
    m = np.isfinite(t) & np.isfinite(x); t, x = t[m], x[m]
    if t.size < 2: return np.nan
    idx = np.argsort(t); t, x = t[idx], x[idx]
    trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")   # numpy>=2: trapezoid
    return float(trapz_fn(x, t))

def peak(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    return float(np.max(x)) if x.size else np.nan

def duty_cycle(t, epos, th=DUTY_TH):
    t = np.asarray(t, dtype=float); epos = np.asarray(epos, dtype=float)
    m = np.isfinite(t) & np.isfinite(epos); t, epos = t[m], epos[m]
    if t.size < 2: return np.nan
    idx = np.argsort(t); t, epos = t[idx], epos[idx]
    dt = np.diff(t)
    if dt.size == 0 or np.sum(dt) <= 0: return np.nan
    active = (epos[:-1] > th).astype(float)   # time-weighted, not sample-count
    return float(np.sum(dt * active) / np.sum(dt))

def coact_minmax_overlap(t, e1, e2):
    num = trapz_auc(t, np.minimum(e1, e2)); den = trapz_auc(t, np.maximum(e1, e2))
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0: return np.nan
    return float(num / den)

def cci_overlap(t, e1, e2):
    num = trapz_auc(t, np.minimum(e1, e2)); den = trapz_auc(t, (e1 + e2))
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0: return np.nan
    return float(2.0 * num / den)

def weighted_mean(values, weights):
    v = np.asarray(values, dtype=float); w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0); v, w = v[m], w[m]
    if v.size == 0: return np.nan
    return float(np.sum(v * w) / np.sum(w))

def robust_overlap_interp(tb, eb, tt, et):
    """Interpolate triceps onto biceps grid only inside the time-overlap (no extrapolation bias)."""
    tb=np.asarray(tb,float); eb=np.asarray(eb,float); tt=np.asarray(tt,float); et=np.asarray(et,float)
    mb=np.isfinite(tb)&np.isfinite(eb); mt=np.isfinite(tt)&np.isfinite(et)
    tb,eb=tb[mb],eb[mb]; tt,et=tt[mt],et[mt]
    if tb.size<2 or tt.size<2: return None,None,None
    ib=np.argsort(tb); tb,eb=tb[ib],eb[ib]; it=np.argsort(tt); tt,et=tt[it],et[it]
    lo=max(tb.min(),tt.min()); hi=min(tb.max(),tt.max())
    if not (hi>lo): return None,None,None
    keep=(tb>=lo)&(tb<=hi); tb2,eb2=tb[keep],eb[keep]
    if tb2.size<2: return None,None,None
    return tb2, eb2, np.interp(tb2, tt, et)
print("Helpers loaded.")

# ===== Cell 3: Trial-level extractors =====
def abs_rms_from_phase6(df_trial_feat):
    """Absolute RMS effort proxy (between-subject comparable) from Phase 6 EMG_<muscle>_RMS."""
    out = {"kpi_abs_rms_mean": np.nan, "kpi_abs_rms_distal": np.nan}
    rms_cols = [c for c in df_trial_feat.columns if c.startswith("EMG_") and c.endswith("_RMS")]
    if not rms_cols: return out
    per_win_mean = df_trial_feat[rms_cols].astype(float).mean(axis=1)
    out["kpi_abs_rms_mean"] = float(np.nanmean(per_win_mean)) if per_win_mean.notna().any() else np.nan
    distal_cols = [c for c in rms_cols if any(f"_{m}_" in c for m in DISTAL_MUSCLES)]
    if distal_cols:
        per_win_distal = df_trial_feat[distal_cols].astype(float).mean(axis=1)
        out["kpi_abs_rms_distal"] = float(np.nanmean(per_win_distal)) if per_win_distal.notna().any() else np.nan
    return out

def extract_trial_kpis_from_phase6(df_trial_feat):
    out = {}
    if df_trial_feat is None or df_trial_feat.empty: return out
    out["subject"]    = str(df_trial_feat["subject"].iloc[0]) if "subject" in df_trial_feat.columns else None
    out["trial_id"]   = str(df_trial_feat["trial_id"].iloc[0]) if "trial_id" in df_trial_feat.columns else None
    out["episode_id"] = df_trial_feat["episode_id"].iloc[0] if "episode_id" in df_trial_feat.columns else None
    for c in META_COLS:
        if c in df_trial_feat.columns: out[c] = pick_mode(df_trial_feat[c])
    if out.get("task_type") is None and out.get("task") is not None: out["task_type"] = out.get("task")
    out["t_on"]  = pick_one(df_trial_feat["t_on"])  if "t_on"  in df_trial_feat.columns else np.nan
    out["t_off"] = pick_one(df_trial_feat["t_off"]) if "t_off" in df_trial_feat.columns else np.nan
    t_on, t_off = out["t_on"], out["t_off"]
    if np.isfinite(t_on) and np.isfinite(t_off) and (t_off > t_on):
        out["kpi_duration_s"] = float(t_off - t_on)
    elif "t_start" in df_trial_feat.columns and "t_end" in df_trial_feat.columns:
        out["kpi_duration_s"] = float(df_trial_feat["t_end"].max() - df_trial_feat["t_start"].min())
    else:
        out["kpi_duration_s"] = np.nan
    if "t_start" in df_trial_feat.columns and "t_end" in df_trial_feat.columns:
        w = (df_trial_feat["t_end"].astype(float) - df_trial_feat["t_start"].astype(float)).to_numpy()
    elif "win_len_s" in df_trial_feat.columns:
        w = df_trial_feat["win_len_s"].astype(float).to_numpy()
    else:
        w = np.ones(len(df_trial_feat), dtype=float)
    has_imu = df_trial_feat["has_imu"].astype(bool) if "has_imu" in df_trial_feat.columns else pd.Series([True]*len(df_trial_feat), index=df_trial_feat.index)
    imu_ok  = (df_trial_feat["imu_ok"]==True) if "imu_ok" in df_trial_feat.columns else pd.Series([True]*len(df_trial_feat), index=df_trial_feat.index)
    imu_target = (has_imu & imu_ok)
    df_imu = df_trial_feat.loc[imu_target].copy() if imu_target.any() else df_trial_feat.copy()
    w_imu = w[imu_target.to_numpy()] if imu_target.any() else w
    out["kpi_peak_gyro_mag"] = peak(df_imu["IMU_all_gyro_mag_peak"].to_numpy()) if "IMU_all_gyro_mag_peak" in df_imu.columns else np.nan
    out["kpi_mean_gyro_mag"] = weighted_mean(df_imu["IMU_all_gyro_mag_mean"].to_numpy(), w_imu) if "IMU_all_gyro_mag_mean" in df_imu.columns else np.nan
    jerk_cols = [c for c in df_imu.columns if c.endswith(("jerk_norm_mean","jerk_abs_mean_1d"))]
    out["kpi_mean_jerk"] = weighted_mean(df_imu[jerk_cols].astype(float).mean(axis=1).to_numpy(), w_imu) if jerk_cols else np.nan
    if "EMG_CoAct_Bic_Tri" in df_trial_feat.columns:
        co = pd.to_numeric(df_trial_feat["EMG_CoAct_Bic_Tri"], errors="coerce").to_numpy(dtype=float)
        out["kpi_coact_win_mean"]   = weighted_mean(co, w)
        out["kpi_coact_win_median"] = float(np.nanmedian(co)) if np.isfinite(co).any() else np.nan
    else:
        out["kpi_coact_win_mean"] = np.nan; out["kpi_coact_win_median"] = np.nan
    out.update(abs_rms_from_phase6(df_trial_feat))   # effort proxy for NMD
    return out

def extract_emg_trial_kpis_from_phase3(df_emg, trial_id, t_on, t_off,
                                       co_act_muscles=CO_ACT_MUSCLES, duty_th=DUTY_TH,
                                       min_samples_per_muscle=MIN_SAMPLES_PER_MUSCLE, min_any=MIN_SAMPLES_ANY_EMG):
    out = {"kpi_emg_valid": False, "kpi_has_emg_any": False, "kpi_has_bictri_pair": False}
    if not (np.isfinite(t_on) and np.isfinite(t_off) and (t_off > t_on)): return out
    d = df_emg[df_emg["trial_id"] == str(trial_id)]
    if d.empty: return out
    d = d[(d["t_emg"] >= t_on) & (d["t_emg"] <= t_off)]
    if d.shape[0] < min_any: return out
    out["kpi_emg_valid"] = True
    global_iemg = 0.0; got_any = False
    for muscle, dm in d.groupby("sensor"):
        if dm.shape[0] < min_samples_per_muscle:
            out[f"kpi_peak_envn_{muscle}"]=np.nan; out[f"kpi_iemg_envn_{muscle}"]=np.nan; out[f"kpi_duty_{muscle}"]=np.nan
            continue
        t = dm["t_emg"].to_numpy(dtype=float); e = env_pos(dm["env_norm"].to_numpy(dtype=float))
        out[f"kpi_peak_envn_{muscle}"] = peak(e)
        auc = trapz_auc(t, e); out[f"kpi_iemg_envn_{muscle}"] = auc
        out[f"kpi_duty_{muscle}"] = duty_cycle(t, e, th=duty_th)
        if np.isfinite(auc): global_iemg += auc; got_any = True
    out["kpi_iemg_envn_global_sum"] = float(global_iemg) if got_any else np.nan
    out["kpi_has_emg_any"] = bool(got_any)
    peak_cols = [c for c in out.keys() if c.startswith("kpi_peak_envn_")]
    if peak_cols:
        vals = [out[c] for c in peak_cols]
        out["kpi_peak_envn_global_max"] = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
    else:
        out["kpi_peak_envn_global_max"] = np.nan
    m1, m2 = co_act_muscles
    db = d[d["sensor"]==m1][["t_emg","env_norm"]].copy(); dt = d[d["sensor"]==m2][["t_emg","env_norm"]].copy()
    if (db.shape[0] >= min_samples_per_muscle) and (dt.shape[0] >= min_samples_per_muscle):
        tb=db["t_emg"].to_numpy(float); eb=env_pos(db["env_norm"].to_numpy(float))
        tt=dt["t_emg"].to_numpy(float); et=env_pos(dt["env_norm"].to_numpy(float))
        t_ov, eb2, et2 = robust_overlap_interp(tb, eb, tt, et)
        if t_ov is not None:
            out["kpi_coact_overlap_bic_tric"] = coact_minmax_overlap(t_ov, eb2, et2)
            out["kpi_cci_bic_tric"]           = cci_overlap(t_ov, eb2, et2)
        else:
            out["kpi_coact_overlap_bic_tric"] = np.nan; out["kpi_cci_bic_tric"] = np.nan
    else:
        out["kpi_coact_overlap_bic_tric"] = np.nan; out["kpi_cci_bic_tric"] = np.nan
    out["kpi_has_bictri_pair"] = bool(np.isfinite(out.get(f"kpi_iemg_envn_{m1}",np.nan)) and np.isfinite(out.get(f"kpi_iemg_envn_{m2}",np.nan)))
    return out
print("Extractors loaded.")

# ===== Cell 4: NMD index (subject-level, z-scored across subjects) =====
def add_nmd_index(df_subject_kpis,
                  effort_col="kpi_abs_rms_distal__median",
                  motion_col="kpi_peak_gyro_mag__median"):
    """
    Neuromuscular Dissociation:  NMD = z(effort) - z(motion)
      effort = absolute distal RMS (Extensor/Flexor)   [between-subject comparable]
      motion = peak gyro magnitude
    z-scored ACROSS subjects (needs the whole group), so this is a FINAL pass.
    High positive NMD = high effort relative to motion = ALS compensatory pattern.
    """
    df = df_subject_kpis.copy()
    def zscore(col):
        if col not in df.columns: return None
        v = pd.to_numeric(df[col], errors="coerce"); mu, sd = v.mean(), v.std(ddof=0)
        if not np.isfinite(sd) or sd == 0: return pd.Series(np.nan, index=df.index)
        return (v - mu) / sd
    z_eff = zscore(effort_col); z_mot = zscore(motion_col)
    if z_eff is None or z_mot is None:
        df["kpi_NMD"] = np.nan
        print(f"[WARN] NMD not computed; missing: {effort_col if z_eff is None else ''} {motion_col if z_mot is None else ''}")
        return df
    df["kpi_NMD_z_effort"] = z_eff
    df["kpi_NMD_z_motion"] = z_mot
    df["kpi_NMD"]          = z_eff - z_mot
    return df
print("NMD loaded.")

# ===== Cell 5: Main runner (trial -> subject -> NMD) =====
def run_phase7_all_subjects_v4(limit_subjects=None, overwrite_outputs=True):
    trial_rows = []; qc_rows = []
    feat_files = sorted(PHASE6_DIR.glob("*__features.parquet"))
    if limit_subjects is not None: feat_files = feat_files[:limit_subjects]
    if overwrite_outputs:
        for p in [TRIAL_OUT, SUBJ_OUT, PHASE7_QC_CSV]:
            if p.exists(): p.unlink()
    for feat_path in tqdm(feat_files, desc="Phase7v4 (subjects)"):
        subject = feat_path.name.split("__features.parquet")[0]
        emg_path = PHASE3_NORM_DIR / f"{subject}__emg_env_norm.parquet"
        if emg_path.exists():
            df_emg = pd.read_parquet(emg_path)
            df_emg["trial_id"] = df_emg["trial_id"].astype(str)
            df_emg["sensor"]   = df_emg["sensor"].astype(str).map(canon_sensor_phase7)
        else:
            df_emg = pd.DataFrame(columns=["trial_id","sensor","t_emg","env_norm"])
        df_feat = pd.read_parquet(feat_path); df_feat["trial_id"] = df_feat["trial_id"].astype(str)
        df_feat["subject"] = df_feat["subject"].astype(str) if "subject" in df_feat.columns else subject
        n_trials=n_valid=n_emg=n_bt=0
        for trial_id, df_trial in df_feat.groupby("trial_id"):
            n_trials += 1
            base = extract_trial_kpis_from_phase6(df_trial) or {"subject":subject, "trial_id":str(trial_id)}
            t_on, t_off = base.get("t_on", np.nan), base.get("t_off", np.nan)
            if np.isfinite(t_on) and np.isfinite(t_off) and (t_off > t_on): n_valid += 1
            emg_kpis = extract_emg_trial_kpis_from_phase3(df_emg, str(trial_id), t_on, t_off)
            row = {**base, **emg_kpis}
            if row.get("kpi_emg_valid", False): n_emg += 1
            if row.get("kpi_has_bictri_pair", False): n_bt += 1
            # NOTE: weak composites (efficiency, cost) intentionally NOT computed in v4.
            trial_rows.append(row)
        qc_rows.append({"subject":subject,"n_trials":n_trials,"n_trials_valid_interval":n_valid,
                        "n_trials_emg_valid":n_emg,"n_trials_has_bictri_pair":n_bt,
                        "rate_valid_interval":(n_valid/n_trials) if n_trials else np.nan,
                        "rate_emg_valid":(n_emg/n_trials) if n_trials else np.nan,
                        "rate_has_bictri_pair":(n_bt/n_trials) if n_trials else np.nan})
    df_trial_kpis = pd.DataFrame(trial_rows); df_trial_kpis.to_parquet(TRIAL_OUT, index=False)
    print(f"[OK] trial KPIs: {df_trial_kpis.shape} -> {TRIAL_OUT}")
    df_qc = pd.DataFrame(qc_rows); df_qc.to_csv(PHASE7_QC_CSV, index=False)
    print(f"[OK] QC CSV -> {PHASE7_QC_CSV}")
    DROP=["t_on","t_off"]
    num_cols = [c for c in df_trial_kpis.select_dtypes(include=[np.number]).columns if c not in DROP]
    subj_overall = df_trial_kpis.groupby("subject")[num_cols].agg(["median","mean","std"])
    subj_overall.columns = [f"{c}__{stat}" for c, stat in subj_overall.columns]; subj_overall = subj_overall.reset_index()
    if "task_type" in df_trial_kpis.columns:
        subj_task = df_trial_kpis.groupby(["subject","task_type"])[num_cols].median().reset_index()
        stw = subj_task.pivot(index="subject", columns="task_type")
        stw.columns = [f"{c}__taskmed__{t}" for c, t in stw.columns]; stw = stw.reset_index()
        df_subject_kpis = subj_overall.merge(stw, on="subject", how="left")
    else:
        df_subject_kpis = subj_overall
    df_subject_kpis = df_subject_kpis.merge(df_qc, on="subject", how="left")
    for mk in ["group","exo","session"]:
        if mk in df_trial_kpis.columns:
            ms = df_trial_kpis.groupby("subject")[mk].apply(lambda s: pick_mode(s)).reset_index()
            df_subject_kpis = df_subject_kpis.merge(ms, on="subject", how="left")
    df_subject_kpis = add_nmd_index(df_subject_kpis)   # FINAL PASS
    df_subject_kpis.to_parquet(SUBJ_OUT, index=False)
    print(f"[OK] subject KPIs: {df_subject_kpis.shape} -> {SUBJ_OUT}")
    return df_trial_kpis, df_subject_kpis, df_qc
print("Runner loaded.")

# =========================================
# Phase 7 — KPI Redundancy Check (non-redundancy criterion)
#
# Purpose: make the "non-redundant" KPI criterion explicit and defensible.
# This cell computes the pairwise Spearman correlation among the subject-level
# KPIs and reports any pair above a threshold, so redundant KPIs are visible
# rather than only argued. Spearman (rank) is used because the KPIs are skewed
# and non-normal (confirmed in the normality analysis), so a rank correlation
# is more appropriate than Pearson.
#
# This is a DIAGNOSTIC cell — it does not delete anything. It documents which
# KPIs carry overlapping information; removal decisions remain explicit and
# reasoned (e.g. the two composite KPIs removed during KPI design).
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path


def run_redundancy_check():
    """Pairwise Spearman correlation among subject-level overall-median KPIs (diagnostic)."""

    SUBJ_KPI_PATH  = PROCESSED_ROOT / "phase_07_kpis" / "subject_kpis.parquet"
    PHASE7_DIR     = PROCESSED_ROOT / "phase_07_kpis"

    REDUNDANCY_THRESHOLD = 0.90   # |rho| above this = flagged as redundant

    # --- Load subject KPIs and keep the interpretable overall-median KPIs ---
    df = pd.read_parquet(SUBJ_KPI_PATH)
    kpi_cols = [c for c in df.columns
                if c.startswith("kpi_") and c.endswith("__median") and "__taskmed__" not in c]

    # numeric, drop all-NaN / zero-variance columns so correlation is defined
    X = df[kpi_cols].apply(pd.to_numeric, errors="coerce")
    keep = [c for c in kpi_cols if X[c].notna().sum() >= 3 and X[c].std(skipna=True) > 1e-9]
    X = X[keep]
    print(f"KPIs entering redundancy check: {len(keep)}")

    # --- Spearman correlation matrix ---
    corr = X.corr(method="spearman")

    # --- Find redundant pairs (upper triangle only, no self-pairs) ---
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = corr.iloc[i, j]
            if np.isfinite(rho) and abs(rho) >= REDUNDANCY_THRESHOLD:
                pairs.append((cols[i], cols[j], float(rho)))

    pairs.sort(key=lambda t: abs(t[2]), reverse=True)

    print(f"\nRedundant KPI pairs (|Spearman rho| >= {REDUNDANCY_THRESHOLD}): {len(pairs)}")
    print("-" * 78)
    if pairs:
        for a, b, rho in pairs:
            print(f"  rho={rho:+.3f}   {a}")
            print(f"               {b}")
            print()
    else:
        print("  None — no KPI pair exceeds the redundancy threshold.")

    # --- Per-KPI maximum correlation with any other KPI (quick redundancy profile) ---
    print("\nPer-KPI highest correlation with any other KPI (top 10):")
    max_corr = {}
    for c in cols:
        others = corr[c].drop(labels=[c])
        max_corr[c] = others.abs().max()
    prof = pd.Series(max_corr).sort_values(ascending=False)
    for c, v in prof.head(10).items():
        flag = "  <-- redundant" if v >= REDUNDANCY_THRESHOLD else ""
        print(f"  {c:<48} max|rho|={v:.3f}{flag}")

    # --- Save the matrix + flagged pairs for the appendix / defense ---
    corr.to_csv(PHASE7_DIR / "kpi_correlation_matrix.csv")
    pd.DataFrame(pairs, columns=["kpi_1", "kpi_2", "spearman_rho"]).to_csv(
        PHASE7_DIR / "kpi_redundant_pairs.csv", index=False)
    print(f"\nSaved:")
    print(f"  kpi_correlation_matrix.csv  -> full Spearman matrix")
    print(f"  kpi_redundant_pairs.csv     -> flagged pairs (|rho| >= {REDUNDANCY_THRESHOLD})")

    # --- Defense note printed inline ---
    print(f"""
    NOTE FOR DEFENSE:
      This check operationalises the 'non-redundant' KPI criterion. Pairs above
      |rho|={REDUNDANCY_THRESHOLD} share most of their rank information; among such a pair only
      one needs to be retained. The two composite KPIs removed during KPI design
      (efficiency = peak gyro / IEMG; cost = IEMG x jerk) were removed for exactly
      this reason — the efficiency index correlated near-perfectly with peak gyro —
      in addition to mixing incompatible (absolute vs normalized) units.
    """)
