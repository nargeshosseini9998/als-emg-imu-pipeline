# alsexo/preprocess_emg.py
# Phase 2 — EMG preprocessing (ported from 01_preprocessing.ipynb).
#
#   raw EMG -> NaN->0 -> 20–450 Hz Butterworth (order 4, zero-phase)
#           -> conditional 50 Hz IIR notch (Q=30) if Welch PSD shows a 45–55 Hz
#              peak >= 5x the 60–80 Hz reference band
#           -> full-wave rectification -> 6 Hz Butterworth (order 4, zero-phase)
#              = linear envelope
#   Saturation repair (NaN masking, ±0.5 s) is applied ONLY to trials listed in
#   configs/pipeline.yaml -> phase2.repair_trials. Those trials are admitted to
#   the pipeline even though they FAILED the Phase-1 QC rule.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch

from .config import pipeline_cfg
from .paths import PHASE2_DIR, subject_csv_files
from .qc import load_qc_subject, compute_trial_status
from .trigno_io import read_trigno_csv


def butter_bandpass(low_hz, high_hz, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass")


def butter_lowpass(cutoff_hz, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, cutoff_hz / nyq, btype="lowpass")


def notch_50hz(fs, q=30.0, f0=50.0):
    return iirnotch(w0=f0 / (fs / 2.0), Q=q)


def has_50hz_peak(x, fs, band=(45, 55), ref_band=(60, 80), prominence_ratio=5.0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < int(fs * 2):
        return False
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), int(fs * 2)))
    band_mask = (f >= band[0]) & (f <= band[1])
    ref_mask = (f >= ref_band[0]) & (f <= ref_band[1])
    if not np.any(band_mask) or not np.any(ref_mask):
        return False
    band_pow = np.mean(Pxx[band_mask])
    ref_pow = np.mean(Pxx[ref_mask]) + 1e-12
    return (band_pow / ref_pow) >= prominence_ratio


def preprocess_emg_signal(x, fs_emg, bp_low=20, bp_high=450, bp_order=4,
                          apply_notch_if_peak=True, notch_q=30.0, notch_ratio=5.0,
                          env_lp=6.0, env_order=4):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    b_bp, a_bp = butter_bandpass(bp_low, bp_high, fs_emg, order=bp_order)
    emg_bp = filtfilt(b_bp, a_bp, x)

    notch_applied = False
    if apply_notch_if_peak and has_50hz_peak(emg_bp, fs_emg, prominence_ratio=notch_ratio):
        b_n, a_n = notch_50hz(fs_emg, q=notch_q)
        emg_bp = filtfilt(b_n, a_n, emg_bp)
        notch_applied = True

    rect = np.abs(emg_bp)
    b_lp, a_lp = butter_lowpass(env_lp, fs_emg, order=env_order)
    env = filtfilt(b_lp, a_lp, rect)
    return emg_bp, env, notch_applied


# -------------------------
# Saturation repair
# -------------------------
def saturation_mask(x, run_len_samples=500, tol=0.0):
    x = np.asarray(x, dtype=float)
    mask = np.zeros(len(x), dtype=bool)
    finite = np.isfinite(x)
    if np.sum(finite) < run_len_samples + 1:
        return mask
    x2 = x.copy()
    x2[~finite] = np.nan
    dx = np.diff(x2)
    same = (dx == 0) if tol == 0.0 else (np.abs(dx) <= tol)
    start, count = None, 0
    for i, s in enumerate(same):
        if s and np.isfinite(dx[i]):
            if start is None:
                start, count = i, 2
            else:
                count += 1
        else:
            if start is not None and count >= run_len_samples:
                mask[start:start + count] = True
            start, count = None, 0
    if start is not None and count >= run_len_samples:
        mask[start:start + count] = True
    return mask


def expand_mask_in_time(mask, fs, pad_seconds=0.5):
    if pad_seconds <= 0:
        return mask
    pad = int(round(pad_seconds * fs))
    if pad <= 0:
        return mask
    out = mask.copy()
    for i in np.where(mask)[0]:
        out[max(0, i - pad):min(len(out), i + pad + 1)] = True
    return out


# -------------------------
# Trial / subject drivers
# -------------------------
def preprocess_trial_to_long_df(subject_name: str, csv_path: Path, qc_subj: pd.DataFrame, cfg: dict):
    sensors, per_sensor, meta = read_trigno_csv(csv_path)
    trial_id = csv_path.stem
    repair_trials = set(cfg["repair_trials"].get(subject_name, []))
    rows, notch_log, skip_log = [], [], []

    for sensor in sensors:
        emg = per_sensor[sensor].get("emg")
        if emg is None or len(emg) == 0:
            skip_log.append({"subject": subject_name, "trial_id": trial_id, "sensor": sensor,
                             "reason": "no_emg_in_reader_layout", "layout": meta.get("layout", ""),
                             "block_kinds": meta.get("block_kinds", "")})
            continue
        t = emg["t_emg"].values
        x = emg["emg_mv"].values

        qrow = qc_subj[(qc_subj["trial_id"] == trial_id) & (qc_subj["sensor"] == sensor)]
        if len(qrow) == 1 and np.isfinite(qrow["emg_fs_est"].values[0]):
            fs_emg = float(qrow["emg_fs_est"].values[0])
        else:
            fs_emg = float(cfg["fs_fallback"])

        emg_bp, env, notch_applied = preprocess_emg_signal(
            x, fs_emg, bp_low=cfg["bp_low_hz"], bp_high=cfg["bp_high_hz"], bp_order=cfg["bp_order"],
            apply_notch_if_peak=cfg["notch_conditional"], notch_q=cfg["notch_q"],
            notch_ratio=cfg["notch_prominence_ratio"], env_lp=cfg["env_lp_hz"], env_order=cfg["env_order"],
        )

        if trial_id in repair_trials:
            m = saturation_mask(x, run_len_samples=cfg["sat_run_len"], tol=cfg["sat_tol"])
            m = expand_mask_in_time(m, fs=fs_emg, pad_seconds=cfg["repair_pad_seconds"])
            if np.any(m):
                emg_bp = emg_bp.copy(); env = env.copy()
                emg_bp[m] = np.nan; env[m] = np.nan

        notch_log.append({"subject": subject_name, "trial_id": trial_id, "sensor": sensor,
                          "fs_emg_used": fs_emg, "notch_applied": notch_applied})
        rows.append(pd.DataFrame({"subject": subject_name, "trial_id": trial_id, "sensor": sensor,
                                  "t_emg": t, "emg_bp": emg_bp, "env": env}))

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out, pd.DataFrame(notch_log), pd.DataFrame(skip_log)


def run_phase2_for_subject(subject_name: str) -> pd.DataFrame:
    cfg = pipeline_cfg()["phase2"]
    qc_cfg = pipeline_cfg()["qc"]
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)

    files = subject_csv_files(subject_name)
    qc_subj = load_qc_subject(subject_name)
    trial_status = compute_trial_status(qc_subj, gap_thr=qc_cfg["gap_thr"], sat_thr=qc_cfg["sat_thr"])
    ok_trials = set(trial_status.loc[trial_status["status"] == "OK", "trial_id"].astype(str))
    repair_trials = set(cfg["repair_trials"].get(subject_name, []))
    allowed = ok_trials | repair_trials
    files_allowed = [fp for fp in files if fp.stem in allowed]
    n_fail = int((trial_status["status"] == "FAIL").sum())
    print(f"[P2] {subject_name}: files={len(files)} OK={len(ok_trials)} FAIL={n_fail} repair={sorted(repair_trials)} -> processing {len(files_allowed)}")

    all_long, notch_logs = [], []
    for fp in files_allowed:
        try:
            long_df, notch_df, _ = preprocess_trial_to_long_df(subject_name, fp, qc_subj, cfg)
        except Exception as e:  # keep going, log the file
            print(f"[SKIP FILE] {fp.name} due to {type(e).__name__}: {e}")
            continue
        if not long_df.empty:
            all_long.append(long_df)
        if not notch_df.empty:
            notch_logs.append(notch_df)

    subj_long = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    if subj_long.empty:
        print("WARNING: No EMG rows produced. Nothing saved.")
        return subj_long
    out = PHASE2_DIR / f"{subject_name}__emg_bp_env.parquet"
    subj_long.to_parquet(out, index=False)
    if notch_logs:
        pd.concat(notch_logs, ignore_index=True).to_csv(PHASE2_DIR / f"{subject_name}__notch_log.csv", index=False)
    print("Saved:", out)
    return subj_long
