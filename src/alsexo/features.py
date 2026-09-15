# alsexo/features.py
# Phase 6 — window-level feature extraction (EMG time/frequency/AR features, IMU kinematics,
# biceps-triceps co-activation). Extracted verbatim from 06_featurs.ipynb (cell 0);
# only the path definitions were replaced by alsexo.paths.
# Public entry points: run_phase6_v3(subject), phase6_qc_report_v3(subject, df_feat)

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 6 v3 - Feature Extraction (EMG + IMU)
# Updates:
# - CoAct is time-aligned (pivot + interp + trapz) => robust to unequal samples/jitter
# - ZC uses adaptive threshold: th = ZC_TH_REL * std(bp)
# - WAMP uses adaptive threshold: th = max(WAMP_TH_ABS, WAMP_TH_REL * std(bp))
# - fs_emg fallback from t_emg if fs_emg_est missing
# - sanitize sensor tokens for safe column names
# - jerk prefers acc_dyn_xyz, else acc_xyz, else acc_dyn scalar
# Output: processed/phase_06_features/{subject}__features.parquet
# =========================================

import re
import numpy as np
import pandas as pd
from scipy import signal
from scipy.linalg import toeplitz
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────

PHASE2B_DIR     = PROCESSED_ROOT / "phase_02b_preprocess_imu"
PHASE3_NORM_DIR = PROCESSED_ROOT / "phase_03_normalization" / "normalized_env_per_subject"
PHASE5_DIR      = PROCESSED_ROOT / "phase_05_windows"
PHASE6_DIR      = PROCESSED_ROOT / "phase_06_features"
PHASE6_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# Canonicalization (English)
# ─────────────────────────────────────────
_EN_CANON_RAW = {
    # Biceps
    "BicBrachii": "Biceps", "Biceps": "Biceps", "bicipite": "Biceps",
    # Triceps
    "TricBrachii": "Triceps", "Triceps": "Triceps", "tricipite": "Triceps",
    # Deltoid
    "DeltMed": "Deltoid", "Deltoid": "Deltoid", "Deltoide": "Deltoid", "deltoide": "Deltoid",
    # Trapezius variants
    "Trap": "Trapezius", "TrapDesc": "Trapezius", "Trap_Desc": "Trapezius",
    "Trap Desc": "Trapezius", "Trap-Desc": "Trapezius",
    "Trapezio": "Trapezius", "trapezio": "Trapezius", "Trapezius": "Trapezius",
    # Wrist extensor
    "ExtCarpRad": "Extensor", "Extensor": "Extensor", "Estensore carpo": "Extensor",
    "estensore": "Extensor", "EXT": "Extensor",
    # Wrist flexor
    "FlexCarpRad": "Flexor", "Flexor": "Flexor", "Flessore carpo": "Flexor",
    "flessore": "Flexor", "Flex": "Flexor",
    # Hand: Abd V
    "Abduttore V": "AbdV", "AbdV": "AbdV", "abdV": "AbdV", "AbdVfing": "AbdV",
    "AbdVFing": "AbdV", "Abd": "AbdV", "Abd5": "AbdV", "abd5": "AbdV",
    "Abd 5 dito": "AbdV", "abd 5 dito": "AbdV",
    # Dorsal interossei
    "IntDors": "IntDors", "interdors": "IntDors", "inter dors": "IntDors",
    "Primo Inter Dors": "IntDors", "InterV": "IntDors",
}

def _norm_key(s: str) -> str:
    s = str(s).strip()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

EN_CANON = {_norm_key(k): v for k, v in _EN_CANON_RAW.items()}

def canon_sensor_name(x: str) -> str:
    raw = str(x).strip()
    raw = re.sub(r"\s+", " ", raw)
    return EN_CANON.get(_norm_key(raw), raw)

def safe_token(s: str) -> str:
    """Make sensor names safe for column tokens."""
    s = re.sub(r"\s+", "_", str(s).strip())
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    return s

# Co-activation pair MUST be in canonical English
CO_ACT_MUSCLES = ("Biceps", "Triceps")

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
AR_ORDER = 4
MIN_EMG_SAMPLES = 32
MIN_IMU_SAMPLES = 8

# Adaptive thresholds
ZC_TH_REL = 0.01          # ZC threshold = 0.01 * std(bp)
WAMP_TH_REL = 0.01        # WAMP threshold = 0.01 * std(bp)
WAMP_TH_ABS = 50e-6       # fallback absolute threshold (Volts) if std is tiny

# ─────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────
def mav(x):
    x = np.asarray(x)
    return float(np.mean(np.abs(x))) if x.size > 0 else np.nan

def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2))) if x.size > 0 else np.nan

def wl(x):
    x = np.asarray(x)
    return float(np.sum(np.abs(np.diff(x)))) if x.size > 1 else np.nan

def wamp(x, th):
    x = np.asarray(x)
    return int(np.sum(np.abs(np.diff(x)) > th)) if x.size > 1 else 0

def zc(x, th=0.0):
    """
    Zero crossing count with threshold (to avoid noise).
    Counts sign changes where abs(diff) > th.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0
    s = np.sign(x)
    s[s == 0] = 1
    crosses = (s[:-1] * s[1:] < 0)
    if th > 0:
        crosses = crosses & (np.abs(x[1:] - x[:-1]) > th)
    return int(np.sum(crosses))

def mnf(x, fs):
    x = np.asarray(x, dtype=float)
    if x.size < 32 or (not np.isfinite(fs)) or fs <= 0:
        return np.nan
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(256, x.size))
    s = np.sum(Pxx)
    if s <= 0:
        return np.nan
    return float(np.average(f, weights=Pxx))

def mdf(x, fs):
    x = np.asarray(x, dtype=float)
    if x.size < 32 or (not np.isfinite(fs)) or fs <= 0:
        return np.nan
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(256, x.size))
    c = np.cumsum(Pxx)
    if c.size == 0 or c[-1] <= 0:
        return np.nan
    idx = np.searchsorted(c, 0.5 * c[-1])
    idx = min(max(idx, 0), len(f)-1)
    return float(f[idx])

def ar_coeffs_yw(x, order=AR_ORDER):
    x = np.asarray(x, dtype=float)
    if x.size < order + 10:
        return [np.nan] * order
    x = x - np.mean(x)
    r = np.correlate(x, x, mode='full')[x.size-1:x.size+order]
    if not np.all(np.isfinite(r)) or r[0] == 0:
        return [np.nan] * order
    R = toeplitz(r[:-1])
    rhs = r[1:]
    try:
        a = np.linalg.solve(R, rhs)
        return [float(v) for v in a]
    except np.linalg.LinAlgError:
        return [np.nan] * order

def robust_dt_from_time(t):
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 3:
        return np.nan
    t = np.unique(t)
    if t.size < 3:
        return np.nan
    dt = np.diff(np.sort(t))
    dt = dt[(dt > 0) & np.isfinite(dt)]
    return float(np.median(dt)) if dt.size else np.nan

def mean_jerk_norm(acc_xyz_df, dt):
    """3D jerk norm mean from xyz acceleration columns."""
    if acc_xyz_df is None or len(acc_xyz_df) < 3 or (not np.isfinite(dt)) or dt <= 0:
        return np.nan
    acc = acc_xyz_df.to_numpy(dtype=float)
    if acc.shape[0] < 3:
        return np.nan
    jerk = np.diff(acc, axis=0) / dt
    jn = np.sqrt(np.sum(jerk**2, axis=1))
    return float(np.nanmean(jn)) if jn.size else np.nan

def mean_jerk_1d(x, dt):
    """1D jerk mean abs (for scalar acc_dyn magnitude)."""
    if x is None or len(x) < 3 or (not np.isfinite(dt)) or dt <= 0:
        return np.nan
    arr = np.asarray(x, dtype=float)
    if arr.size < 3:
        return np.nan
    j = np.diff(arr) / dt
    return float(np.nanmean(np.abs(j))) if j.size else np.nan

def coact_overlap_auc_time(emg_win, m1="Biceps", m2="Triceps"):
    """
    Time-aligned overlap co-activation in [0,1]:
      AUC(min(env1,env2)) / AUC(max(env1,env2))
    Robust to unequal sample counts/jitter via time pivot + interpolation.
    """
    if emg_win is None or emg_win.empty:
        return np.nan

    sub = emg_win[emg_win["sensor"].isin([m1, m2])][["t_emg", "sensor", "env_norm"]]
    if sub.empty:
        return np.nan

    wide = (
        sub.pivot_table(index="t_emg", columns="sensor", values="env_norm", aggfunc="mean")
           .sort_index()
    )
    if (m1 not in wide.columns) or (m2 not in wide.columns):
        return np.nan

    t = wide.index.to_numpy(dtype=float)
    a = wide[m1].to_numpy(dtype=float)
    b = wide[m2].to_numpy(dtype=float)

    dt = robust_dt_from_time(t)
    if (not np.isfinite(dt)) or dt <= 0:
        return np.nan

    # regular grid (more stable integration)
    tg = np.arange(t[0], t[-1] + 1e-12, dt)
    if tg.size < 3:
        return np.nan

    a_ok = np.isfinite(a)
    b_ok = np.isfinite(b)
    if (not a_ok.any()) or (not b_ok.any()):
        return np.nan

    a_i = np.interp(tg, t[a_ok], a[a_ok])
    b_i = np.interp(tg, t[b_ok], b[b_ok])

    a_i = np.maximum(a_i, 0.0)
    b_i = np.maximum(b_i, 0.0)

    mn = np.minimum(a_i, b_i)
    mx = np.maximum(a_i, b_i)

    denom = np.trapezoid(mx, x=tg)
    if (not np.isfinite(denom)) or denom <= 0:
        return np.nan

    return float(np.trapezoid(mn, x=tg) / denom)

# ─────────────────────────────────────────
# Core: extract one window
# ─────────────────────────────────────────
def extract_window_features(row, emg_trial, imu_trial):
    feat = row.to_dict()
    ts, te = float(row["t_start"]), float(row["t_end"])

    # true presence flags (not QC flags)
    feat["has_emg"] = False
    feat["has_imu"] = False

    # ------------ EMG ------------
    if emg_trial is not None and (not emg_trial.empty):
        emg_win = emg_trial[(emg_trial["t_emg"] >= ts) & (emg_trial["t_emg"] < te)]
        feat["has_emg"] = (not emg_win.empty)

        if not emg_win.empty:
            fs_emg = float(row.get("fs_emg_est", np.nan))

            # compute per-sensor features
            for sensor, g in emg_win.groupby("sensor"):
                g = g.sort_values("t_emg")
                bp = g["emg_bp"].to_numpy(dtype=float)

                if bp.size < MIN_EMG_SAMPLES:
                    continue

                # fallback fs from time if needed
                fs_used = fs_emg
                if (not np.isfinite(fs_used)) or fs_used <= 0:
                    dt_emg = robust_dt_from_time(g["t_emg"].to_numpy())
                    fs_used = (1.0 / dt_emg) if np.isfinite(dt_emg) and dt_emg > 0 else np.nan

                # adaptive thresholds
                sd = float(np.nanstd(bp))
                th_zc = ZC_TH_REL * sd if np.isfinite(sd) and sd > 0 else 0.0
                th_wamp = WAMP_TH_REL * sd if np.isfinite(sd) and sd > 0 else WAMP_TH_ABS
                th_wamp = max(WAMP_TH_ABS, th_wamp)

                sensor_tok = safe_token(sensor)
                p = f"EMG_{sensor_tok}_"

                feat[p + "MAV"]  = mav(bp)
                feat[p + "RMS"]  = rms(bp)
                feat[p + "WL"]   = wl(bp)
                feat[p + "WAMP"] = wamp(bp, th_wamp)
                feat[p + "ZC"]   = zc(bp, th=th_zc)
                feat[p + "MNF"]  = mnf(bp, fs_used)
                feat[p + "MDF"]  = mdf(bp, fs_used)

                ar = ar_coeffs_yw(bp, AR_ORDER)
                for i, c in enumerate(ar, 1):
                    feat[p + f"AR{i}"] = c

                # keep debug fs if you want (optional)
                feat[p + "fs_used"] = float(fs_used) if np.isfinite(fs_used) else np.nan

            # CoAct (time-aligned) on env_norm
            feat["EMG_CoAct_Bic_Tri"] = coact_overlap_auc_time(emg_win, CO_ACT_MUSCLES[0], CO_ACT_MUSCLES[1])

            # global effort summary (sum of MAV across muscles present in this window)
            mav_keys = [k for k in feat.keys() if k.startswith("EMG_") and k.endswith("_MAV")]
            if mav_keys:
                feat["EMG_global_effort_MAV_sum"] = float(np.nansum([feat[k] for k in mav_keys]))

    # ------------ IMU ------------
    if imu_trial is not None and (not imu_trial.empty):
        imu_win = imu_trial[(imu_trial["t_imu"] >= ts) & (imu_trial["t_imu"] < te)]
        feat["has_imu"] = (not imu_win.empty)

        if not imu_win.empty:
            for sensor, g in imu_win.groupby("sensor"):
                g = g.sort_values("t_imu")
                dt = robust_dt_from_time(g["t_imu"].to_numpy())
                sensor_tok = safe_token(sensor)
                p = f"IMU_{sensor_tok}_"

                if "gyro_mag" in g.columns:
                    feat[p + "gyro_mag_mean"] = float(g["gyro_mag"].mean())
                    feat[p + "gyro_mag_std"]  = float(g["gyro_mag"].std())
                    feat[p + "gyro_mag_peak"] = float(g["gyro_mag"].max())

                if "acc_mag" in g.columns:
                    feat[p + "acc_mag_mean"] = float(g["acc_mag"].mean())
                    feat[p + "acc_mag_std"]  = float(g["acc_mag"].std())
                    feat[p + "acc_mag_peak"] = float(g["acc_mag"].max())

                if "acc_dyn" in g.columns:
                    feat[p + "acc_dyn_mean"] = float(g["acc_dyn"].mean())
                    feat[p + "acc_dyn_std"]  = float(g["acc_dyn"].std())

                # jerk: prefer dynamic xyz if available, else raw xyz, else scalar acc_dyn
                dyn_xyz = ["acc_dyn_x", "acc_dyn_y", "acc_dyn_z"]
                raw_xyz = ["acc_x_g", "acc_y_g", "acc_z_g"]

                if all(c in g.columns for c in dyn_xyz) and len(g) >= MIN_IMU_SAMPLES:
                    feat[p + "jerk_norm_mean"] = mean_jerk_norm(g[dyn_xyz], dt)
                elif all(c in g.columns for c in raw_xyz) and len(g) >= MIN_IMU_SAMPLES:
                    feat[p + "jerk_norm_mean"] = mean_jerk_norm(g[raw_xyz], dt)
                elif "acc_dyn" in g.columns and len(g) >= MIN_IMU_SAMPLES:
                    feat[p + "jerk_abs_mean_1d"] = mean_jerk_1d(g["acc_dyn"].to_numpy(), dt)

                feat[p + "dt_median_s"] = float(dt) if np.isfinite(dt) else np.nan
                if "fs_imu_used" in g.columns:
                    feat[p + "fs_imu_used_med"] = float(pd.Series(g["fs_imu_used"]).median())

            # global summaries across all sensors inside window
            if "gyro_mag" in imu_win.columns:
                feat["IMU_all_gyro_mag_mean"] = float(imu_win["gyro_mag"].mean())
                feat["IMU_all_gyro_mag_peak"] = float(imu_win["gyro_mag"].max())

    return feat

# ─────────────────────────────────────────
# Runner
# ─────────────────────────────────────────
def run_phase6_v3(subject_name: str):
    win_path = PHASE5_DIR / f"{subject_name}__windows.parquet"
    emg_path = PHASE3_NORM_DIR / f"{subject_name}__emg_env_norm.parquet"
    imu_path = PHASE2B_DIR / f"{subject_name}__imu_gate.parquet"

    if not win_path.exists():
        print(f"[FAIL] Missing windows file: {win_path}")
        return None

    windows = pd.read_parquet(win_path)
    windows["trial_id"] = windows["trial_id"].astype(str)

    df_emg = pd.read_parquet(emg_path) if emg_path.exists() else pd.DataFrame()
    df_imu = pd.read_parquet(imu_path) if imu_path.exists() else pd.DataFrame()

    # Canonicalize sensors (EMG+IMU)
    if not df_emg.empty:
        df_emg["trial_id"] = df_emg["trial_id"].astype(str)
        df_emg["sensor"] = df_emg["sensor"].astype(str).map(canon_sensor_name)

    if not df_imu.empty:
        df_imu["trial_id"] = df_imu["trial_id"].astype(str)
        df_imu["sensor"] = df_imu["sensor"].astype(str).map(canon_sensor_name)

    # Keep windows with at least one modality OK (Phase5 QC)
    if "emg_ok" in windows.columns:
        emg_ok = (windows["emg_ok"] == True)
    else:
        emg_ok = pd.Series([True]*len(windows), index=windows.index)

    if "imu_ok" in windows.columns:
        imu_ok = (windows["imu_ok"] == True)
    else:
        imu_ok = pd.Series([True]*len(windows), index=windows.index)

    valid = windows[emg_ok | imu_ok].copy()
    if valid.empty:
        print(f"[WARN] No valid windows for {subject_name}")
        return None

    features = []
    for tid, w_trial in tqdm(valid.groupby("trial_id"), desc=f"Phase6v3 {subject_name}"):
        emg_trial = df_emg[df_emg["trial_id"] == tid].copy() if not df_emg.empty else pd.DataFrame()
        imu_trial = df_imu[df_imu["trial_id"] == tid].copy() if not df_imu.empty else pd.DataFrame()

        if not emg_trial.empty:
            emg_trial = emg_trial.sort_values("t_emg")
        if not imu_trial.empty:
            imu_trial = imu_trial.sort_values("t_imu")

        for _, row in w_trial.iterrows():
            features.append(extract_window_features(row, emg_trial, imu_trial))

    df_feat = pd.DataFrame(features)

    # Ensure CoAct column exists (even if all NaN)
    if "EMG_CoAct_Bic_Tri" not in df_feat.columns:
        df_feat["EMG_CoAct_Bic_Tri"] = np.nan

    out_path = PHASE6_DIR / f"{subject_name}__features.parquet"
    df_feat.to_parquet(out_path, index=False)

    co_cov = float(df_feat["EMG_CoAct_Bic_Tri"].notna().mean())
    print(f"✅ Saved: {out_path} | rows={len(df_feat)} | CoAct non-null ratio={co_cov:.3f}")

    return df_feat

# ─────────────────────────────────────────
# Phase 6 QC Report v3 (core-feature aware)
# Output: processed/phase_06_features/phase6_qc_report.csv
# ─────────────────────────────────────────
PHASE6_QC_CSV = PHASE6_DIR / "phase6_qc_report.csv"

def _status(val, ok_rule, warn_rule=None):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "FAIL"
    if ok_rule(val):
        return "OK"
    if warn_rule is not None and warn_rule(val):
        return "WARN"
    return "FAIL"

def phase6_qc_report_v3(subject_name: str, df_feat: pd.DataFrame, save_csv: bool = True, verbose: bool = True):
    n = len(df_feat) if df_feat is not None else 0
    if df_feat is None or n == 0:
        print(f"=== Phase 6 QC Report | {subject_name} ===")
        print("[FAIL] df_feat is empty or None")
        return None

    required_keys = ["subject","trial_id","window_id","t_start","t_end"]
    optional_keys = ["episode_id"]

    missing_required = [c for c in required_keys if c not in df_feat.columns]
    missing_optional = [c for c in optional_keys if c not in df_feat.columns]

    has_emg = df_feat["has_emg"].astype(bool) if "has_emg" in df_feat.columns else pd.Series([False]*n, index=df_feat.index)
    has_imu = df_feat["has_imu"].astype(bool) if "has_imu" in df_feat.columns else pd.Series([False]*n, index=df_feat.index)

    emg_ok = (df_feat["emg_ok"] == True) if "emg_ok" in df_feat.columns else pd.Series([True]*n, index=df_feat.index)
    imu_ok = (df_feat["imu_ok"] == True) if "imu_ok" in df_feat.columns else pd.Series([True]*n, index=df_feat.index)

    win_len = (df_feat["t_end"] - df_feat["t_start"]).astype(float)
    win_len_med = float(np.nanmedian(win_len))
    win_len_p01 = float(np.nanpercentile(win_len, 1))
    win_len_p99 = float(np.nanpercentile(win_len, 99))

    # Core feature columns only (avoid counting CoAct/global summaries as coverage)
    emg_core_cols = [c for c in df_feat.columns if c.startswith("EMG_") and c.endswith(("_MAV","_RMS","_WL","_MNF","_MDF"))]
    imu_core_cols = [c for c in df_feat.columns if c.startswith("IMU_") and (
        ("gyro_mag_" in c) or ("acc_mag_" in c) or c.endswith("jerk_norm_mean") or c.endswith("jerk_abs_mean_1d")
    )]

    emg_any_feat = df_feat[emg_core_cols].notna().any(axis=1) if emg_core_cols else pd.Series([False]*n, index=df_feat.index)
    imu_any_feat = df_feat[imu_core_cols].notna().any(axis=1) if imu_core_cols else pd.Series([False]*n, index=df_feat.index)

    emg_target = (has_emg & emg_ok)
    imu_target = (has_imu & imu_ok)

    emg_feat_cov = float((emg_any_feat[emg_target].mean()) if emg_target.any() else np.nan)
    imu_feat_cov = float((imu_any_feat[imu_target].mean()) if imu_target.any() else np.nan)

    emg_nan_ratio = float(df_feat[emg_core_cols].isna().mean().mean()) if emg_core_cols else np.nan
    imu_nan_ratio = float(df_feat[imu_core_cols].isna().mean().mean()) if imu_core_cols else np.nan

    # sign sanity
    emg_mav_cols = [c for c in df_feat.columns if c.startswith("EMG_") and c.endswith("_MAV")]
    emg_wl_cols  = [c for c in df_feat.columns if c.startswith("EMG_") and c.endswith("_WL")]
    emg_mav_neg_rate = float((df_feat[emg_mav_cols] < 0).mean().mean()) if emg_mav_cols else np.nan
    emg_wl_neg_rate  = float((df_feat[emg_wl_cols]  < 0).mean().mean()) if emg_wl_cols else np.nan

    # CoAct required + sanity
    coact_col = "EMG_CoAct_Bic_Tri"
    coact_exists = (coact_col in df_feat.columns)

    bic_mav_cols = [c for c in df_feat.columns if c.startswith("EMG_Biceps_") and c.endswith("_MAV")]
    tri_mav_cols = [c for c in df_feat.columns if c.startswith("EMG_Triceps_") and c.endswith("_MAV")]
    both_present = pd.Series([False]*n, index=df_feat.index)
    if bic_mav_cols and tri_mav_cols:
        both_present = df_feat[bic_mav_cols].notna().any(axis=1) & df_feat[tri_mav_cols].notna().any(axis=1)

    coact_cov = np.nan
    coact_bad_rate = np.nan
    if coact_exists:
        co = df_feat[coact_col].astype(float)
        coact_cov = float(co[both_present].notna().mean()) if both_present.any() else np.nan
        finite = np.isfinite(co)
        coact_bad_rate = float(((co[finite] < 0) | (co[finite] > 1)).mean()) if finite.any() else np.nan

    # IMU dt sanity
    dt_cols = [c for c in df_feat.columns if c.startswith("IMU_") and c.endswith("dt_median_s")]
    dt_bad_rate = float(((df_feat[dt_cols].astype(float) <= 0) | (~np.isfinite(df_feat[dt_cols].astype(float)))).mean().mean()) if dt_cols else np.nan

    gyro_cols = [c for c in df_feat.columns if c.startswith("IMU_") and "gyro_mag_" in c]
    acc_cols  = [c for c in df_feat.columns if c.startswith("IMU_") and "acc_mag_" in c]
    jerk_cols = [c for c in df_feat.columns if c.startswith("IMU_") and (c.endswith("jerk_norm_mean") or c.endswith("jerk_abs_mean_1d"))]

    gyro_bad_rate = float((df_feat[gyro_cols] < 0).mean().mean()) if gyro_cols else np.nan
    acc_bad_rate  = float((df_feat[acc_cols]  < 0).mean().mean()) if acc_cols  else np.nan
    jerk_bad_rate = float((df_feat[jerk_cols] < 0).mean().mean()) if jerk_cols else np.nan

    checks = []
    checks.append(("df_feat non-empty", "OK" if n > 0 else "FAIL", f"n={n}"))
    checks.append(("required key columns present", "OK" if len(missing_required)==0 else "FAIL", f"missing_required={missing_required}"))
    checks.append(("optional key columns present", "OK" if len(missing_optional)==0 else "WARN", f"missing_optional={missing_optional}"))

    checks.append(("window_len median ~ 0.2s",
                   _status(win_len_med, ok_rule=lambda v: abs(v-0.2) <= 1e-3, warn_rule=lambda v: abs(v-0.2) <= 5e-3),
                   f"median={win_len_med:.6f}, p01={win_len_p01:.6f}, p99={win_len_p99:.6f}"))

    checks.append(("EMG core feature coverage among (has_emg & emg_ok)",
                   _status(emg_feat_cov, ok_rule=lambda v: v >= 0.98, warn_rule=lambda v: v >= 0.90),
                   f"coverage={emg_feat_cov:.3f}"))

    checks.append(("IMU core feature coverage among (has_imu & imu_ok)",
                   _status(imu_feat_cov, ok_rule=lambda v: v >= 0.98, warn_rule=lambda v: v >= 0.90),
                   f"coverage={imu_feat_cov:.3f}"))

    checks.append(("EMG core NaN ratio",
                   _status(emg_nan_ratio, ok_rule=lambda v: v <= 0.10, warn_rule=lambda v: v <= 0.25),
                   f"nan_ratio={emg_nan_ratio:.3f}"))

    checks.append(("IMU core NaN ratio",
                   _status(imu_nan_ratio, ok_rule=lambda v: v <= 0.10, warn_rule=lambda v: v <= 0.25),
                   f"nan_ratio={imu_nan_ratio:.3f}"))

    checks.append(("EMG MAV negative rate",
                   _status(emg_mav_neg_rate, ok_rule=lambda v: v == 0.0, warn_rule=lambda v: v <= 0.001),
                   f"neg_rate={emg_mav_neg_rate:.6f}"))

    checks.append(("EMG WL negative rate",
                   _status(emg_wl_neg_rate, ok_rule=lambda v: v == 0.0, warn_rule=lambda v: v <= 0.001),
                   f"neg_rate={emg_wl_neg_rate:.6f}"))

    checks.append(("CoAct column exists", "OK" if coact_exists else "FAIL", f"{coact_col}"))
    if coact_exists:
        checks.append(("CoAct coverage among windows with both Biceps & Triceps",
                       _status(coact_cov, ok_rule=lambda v: v >= 0.95, warn_rule=lambda v: v >= 0.80),
                       f"coverage={coact_cov if np.isfinite(coact_cov) else np.nan:.3f}, both_present={int(both_present.sum())}"))
        if np.isfinite(coact_bad_rate):
            checks.append(("CoAct in [0,1] bad rate",
                           _status(coact_bad_rate, ok_rule=lambda v: v == 0.0, warn_rule=lambda v: v <= 0.01),
                           f"bad_rate={coact_bad_rate:.4f}"))

    checks.append(("IMU dt_median invalid rate",
                   _status(dt_bad_rate, ok_rule=lambda v: v == 0.0, warn_rule=lambda v: v <= 0.01),
                   f"bad_rate={dt_bad_rate:.4f}"))

    checks.append(("IMU gyro_mag negative rate",
                   _status(gyro_bad_rate, ok_rule=lambda v: v == 0.0, warn_rule=lambda v: v <= 0.001),
                   f"neg_rate={gyro_bad_rate:.6f}"))

    checks.append(("IMU acc_mag negative rate",
                   _status(acc_bad_rate, ok_rule=lambda v: v == 0.0, warn_rule=lambda v: v <= 0.001),
                   f"neg_rate={acc_bad_rate:.6f}"))

    checks.append(("IMU jerk negative rate",
                   _status(jerk_bad_rate, ok_rule=lambda v: v == 0.0, warn_rule=lambda v: v <= 0.001),
                   f"neg_rate={jerk_bad_rate:.6f}"))

    critical_names = {
        "df_feat non-empty",
        "required key columns present",
        "window_len median ~ 0.2s",
        "EMG core feature coverage among (has_emg & emg_ok)",
        "IMU core feature coverage among (has_imu & imu_ok)",
        "CoAct column exists",
    }
    any_fail_critical = any((name in critical_names) and (st == "FAIL") for name, st, _ in checks)
    overall = "FAIL" if any_fail_critical else ("WARN" if any(st == "WARN" for _, st, _ in checks) else "OK")

    if verbose:
        print(f"\n=== Phase 6 QC Report v3 | {subject_name} ===")
        for name, st, detail in checks:
            print(f"[{st}] {name}: {detail}")
        print(f"\nOverall QC: {overall}")
        print("\nQuick counts:")
        print("  #EMG core cols:", len(emg_core_cols), "| #IMU core cols:", len(imu_core_cols))
        print("  EMG target windows:", int(emg_target.sum()), "| IMU target windows:", int(imu_target.sum()))
        if coact_exists:
            print("  CoAct both-present windows:", int(both_present.sum()), "| CoAct coverage:", coact_cov)

    row_out = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subject": subject_name,
        "n_windows_features": int(n),
        "missing_required_key_cols": ";".join(missing_required) if missing_required else "",
        "missing_optional_key_cols": ";".join(missing_optional) if missing_optional else "",
        "overall_qc": overall,

        "win_len_med": win_len_med,
        "win_len_p01": win_len_p01,
        "win_len_p99": win_len_p99,

        "emg_feat_cov": emg_feat_cov,
        "imu_feat_cov": imu_feat_cov,
        "emg_nan_ratio": emg_nan_ratio,
        "imu_nan_ratio": imu_nan_ratio,

        "emg_mav_neg_rate": emg_mav_neg_rate,
        "emg_wl_neg_rate": emg_wl_neg_rate,

        "coact_exists": bool(coact_exists),
        "coact_cov": coact_cov,
        "coact_bad_rate": coact_bad_rate,
        "coact_both_present_windows": int(both_present.sum()),

        "dt_bad_rate": dt_bad_rate,
        "gyro_bad_rate": gyro_bad_rate,
        "acc_bad_rate": acc_bad_rate,
        "jerk_bad_rate": jerk_bad_rate,

        "n_emg_core_cols": int(len(emg_core_cols)),
        "n_imu_core_cols": int(len(imu_core_cols)),
        "n_emg_target": int(emg_target.sum()),
        "n_imu_target": int(imu_target.sum()),
        "emg_missing_ratio": float((~has_emg).mean()) if "has_emg" in df_feat.columns else np.nan,
        "imu_missing_ratio": float((~has_imu).mean()) if "has_imu" in df_feat.columns else np.nan,
    }

    qc_df = pd.DataFrame([row_out])

    if save_csv:
        if PHASE6_QC_CSV.exists():
            old = pd.read_csv(PHASE6_QC_CSV)
            new = pd.concat([old, qc_df], ignore_index=True)
        else:
            new = qc_df
        new.to_csv(PHASE6_QC_CSV, index=False)
        if verbose:
            print(f"\n✅ Saved/updated global QC CSV: {PHASE6_QC_CSV}")

    return qc_df
