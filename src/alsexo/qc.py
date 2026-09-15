# alsexo/qc.py
# Phase 0/1 — parsing metadata + per-signal quality control.
#
# Ported from notebooks 00_qc_check.ipynb / 00_qc_healthy.ipynb (identical logic).
# QC criteria (see configs/pipeline.yaml -> qc):
#   - effective sampling rate and time gaps (gap = dt > gap_factor * median dt)
#   - fraction of non-finite samples
#   - near-zero variance (dead channel)
#   - saturation ratio: fraction of samples inside runs of identical values
#     longer than `sat_run` samples (EMG: 500 samples ~ 0.4 s @ 1259 Hz;
#     IMU magnitudes: 60 samples ~ 0.4 s @ 148 Hz)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import pipeline_cfg
from .paths import QC_ALL_PATH, IO_ALL_PATH, QC_DIR, subject_csv_files
from .trigno_io import read_trigno_csv


# -------------------------
# Signal-level QC helpers
# -------------------------
def estimate_fs_and_gaps(t, gap_factor=2.0):
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return np.nan, 0, np.nan, np.nan
    fs = 1.0 / np.mean(dt)
    med_dt = np.median(dt)
    gap_count = int(np.sum(dt > gap_factor * med_dt))
    return fs, gap_count, float(np.min(dt)), float(np.max(dt))


def nan_ratio(x):
    x = np.asarray(x)
    return float(np.mean(~np.isfinite(x)))


def near_zero_variance(x, eps=1e-10):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return True, np.nan
    v = float(np.var(x))
    return (v < eps), v


def saturation_ratio(x, run_len_samples=500, tol=0.0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    same = (np.diff(x) == 0) if tol == 0.0 else (np.abs(np.diff(x)) <= tol)
    total_marked = 0
    current = 1
    for s in same:
        if s:
            current += 1
        else:
            if current >= run_len_samples:
                total_marked += current
            current = 1
    if current >= run_len_samples:
        total_marked += current
    return float(total_marked / max(len(x), 1))


def qc_one_signal(t, x, name_prefix, gap_factor=2.0, sat_run=500, sat_tol=0.0, var_eps=1e-10):
    fs, gap_count, dt_min, dt_max = estimate_fs_and_gaps(t, gap_factor=gap_factor)
    nratio = nan_ratio(x)
    is_lowvar, var = near_zero_variance(x, eps=var_eps)
    sratio = saturation_ratio(x, run_len_samples=sat_run, tol=sat_tol)

    t = np.asarray(t, dtype=float)
    t0 = float(np.nanmin(t)) if np.any(np.isfinite(t)) else np.nan
    t1 = float(np.nanmax(t)) if np.any(np.isfinite(t)) else np.nan
    dur = float(t1 - t0) if (np.isfinite(t0) and np.isfinite(t1)) else np.nan

    return {
        f"{name_prefix}_fs_est": fs,
        f"{name_prefix}_gaps": gap_count,
        f"{name_prefix}_dt_min": dt_min,
        f"{name_prefix}_dt_max": dt_max,
        f"{name_prefix}_nan_ratio": nratio,
        f"{name_prefix}_var": var,
        f"{name_prefix}_lowvar_flag": bool(is_lowvar),
        f"{name_prefix}_saturation_ratio": sratio,
        f"{name_prefix}_duration_s": dur,
        f"{name_prefix}_n_samples": int(np.sum(np.isfinite(t))),
    }


def _nan_qc(prefix: str):
    return {
        f"{prefix}_fs_est": np.nan,
        f"{prefix}_gaps": np.nan,
        f"{prefix}_dt_min": np.nan,
        f"{prefix}_dt_max": np.nan,
        f"{prefix}_nan_ratio": np.nan,
        f"{prefix}_var": np.nan,
        f"{prefix}_lowvar_flag": False,
        f"{prefix}_saturation_ratio": np.nan,
        f"{prefix}_duration_s": np.nan,
        f"{prefix}_n_samples": 0,
    }


def qc_trigno_parsed(per_sensor: dict, qc: dict | None = None) -> pd.DataFrame:
    qc = qc or pipeline_cfg()["qc"]
    rows = []
    for sensor, d in per_sensor.items():
        emg = d.get("emg")
        imu = d.get("imu")
        r = {"sensor": sensor}

        if emg is None or len(emg) == 0:
            r.update(_nan_qc("emg"))
        else:
            r.update(qc_one_signal(
                t=emg["t_emg"].values, x=emg["emg_mv"].values, name_prefix="emg",
                gap_factor=qc["gap_factor"], sat_run=qc["emg_sat_run_samples"],
                sat_tol=qc["sat_tol"], var_eps=qc["var_eps"],
            ))

        if imu is None or len(imu) == 0:
            r.update(_nan_qc("imu_gyro_mag"))
            r.update(_nan_qc("imu_acc_mag"))
        else:
            acc_mag = np.sqrt(imu["acc_x_g"].values**2 + imu["acc_y_g"].values**2 + imu["acc_z_g"].values**2)
            gyro_mag = np.sqrt(imu["gyro_x_dps"].values**2 + imu["gyro_y_dps"].values**2 + imu["gyro_z_dps"].values**2)
            for name, x in [("imu_gyro_mag", gyro_mag), ("imu_acc_mag", acc_mag)]:
                r.update(qc_one_signal(
                    t=imu["t_imu"].values, x=x, name_prefix=name,
                    gap_factor=qc["gap_factor"], sat_run=qc["imu_sat_run_samples"],
                    sat_tol=qc["sat_tol"], var_eps=qc["var_eps"],
                ))
        rows.append(r)
    return pd.DataFrame(rows)


# -------------------------
# Subject-level driver
# -------------------------
def qc_and_meta_for_subject(subject_name: str, files: list[Path]):
    """Parse + QC all trial CSVs of one subject.

    Returns (qc_subject_df [trial x sensor rows], io_subject_df [trial rows]).
    """
    qc_rows, meta_rows = [], []
    for fp in files:
        sensors, per_sensor, meta = read_trigno_csv(fp)
        qc_df = qc_trigno_parsed(per_sensor)
        qc_df.insert(0, "trial_id", fp.stem)
        qc_df.insert(0, "file", fp.name)
        qc_df.insert(0, "subject", subject_name)
        qc_df["layout"] = meta.get("layout", "")
        qc_df["mapping_ok"] = meta.get("mapping_ok", True)
        qc_df["nsensors_file"] = meta.get("nsensors", np.nan)
        qc_df["block_kinds"] = meta.get("block_kinds", "")
        # informational flags (NOT failures)
        qc_df["imu_only_sensor"] = (qc_df["emg_n_samples"].fillna(0) == 0) & (qc_df["imu_gyro_mag_n_samples"].fillna(0) > 0)
        qc_df["emg_only_sensor"] = (qc_df["emg_n_samples"].fillna(0) > 0) & (qc_df["imu_gyro_mag_n_samples"].fillna(0) == 0)
        qc_rows.append(qc_df)

        meta_rows.append({
            "subject": subject_name, "trial_id": fp.stem, "file": fp.name,
            "layout": meta.get("layout", ""), "nsensors": meta.get("nsensors", np.nan),
            "block_kinds": meta.get("block_kinds", ""),
            "ncols_available_in_header": meta.get("ncols_available_in_header", np.nan),
            "ncols_requested_usecols": meta.get("ncols_requested_usecols", np.nan),
            "ncols_read_after_drop_empty": meta.get("ncols_read_after_drop_empty", np.nan),
            "ncols_used": meta.get("ncols_used", np.nan),
            "nrows_loaded": meta.get("nrows_loaded", np.nan),
            "sensor_name_source": meta.get("sensor_name_source", ""),
            "mapping_ok": meta.get("mapping_ok", True),
            "mapping_issues": meta.get("mapping_issues", ""),
            "final_sensor_names": "|".join(meta.get("final_sensor_names", []) or []),
        })
    qc_subject_df = pd.concat(qc_rows, ignore_index=True) if qc_rows else pd.DataFrame()
    return qc_subject_df, pd.DataFrame(meta_rows)


def compute_trial_status(qc_subj: pd.DataFrame, gap_thr: int = 0, sat_thr: float = 0.01) -> pd.DataFrame:
    """Trial-level FAIL rule (conservative). Identical to Phases 1/2/2b.

    FAIL if any sensor row has: mapping issue, any time gap (EMG or IMU),
    or — when EMG exists — a dead channel or saturation ratio > sat_thr.
    """
    qc = qc_subj.copy()
    qc["qc_fail"] = (
        (qc["mapping_ok"].fillna(True) == False) |
        (qc["emg_gaps"].fillna(0) > gap_thr) |
        (qc["imu_gyro_mag_gaps"].fillna(0) > gap_thr) |
        (qc["imu_acc_mag_gaps"].fillna(0) > gap_thr) |
        ((qc["emg_n_samples"].fillna(0) > 0) & (
            (qc["emg_lowvar_flag"].fillna(False) == True) |
            (qc["emg_saturation_ratio"].fillna(0) > sat_thr)))
    )
    trial_status = qc.groupby("trial_id")["qc_fail"].any().reset_index()
    trial_status["status"] = np.where(trial_status["qc_fail"], "FAIL", "OK")
    return trial_status


def load_qc_subject(subject_name: str, qc_all_path: Path = QC_ALL_PATH) -> pd.DataFrame:
    qc_all = pd.read_csv(qc_all_path)
    qc_subj = qc_all[qc_all["subject"].astype(str) == str(subject_name)].copy()
    if qc_subj.empty:
        raise FileNotFoundError(f"No QC rows found for subject '{subject_name}' in {qc_all_path.name}")
    return qc_subj


def run_phase0_all(subjects: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run parsing + QC for every subject and (re)write the two master CSVs."""
    QC_DIR.mkdir(parents=True, exist_ok=True)
    qc_all, io_all = [], []
    cfg = pipeline_cfg()["qc"]
    for s in subjects:
        files = subject_csv_files(s)
        qc_df, io_df = qc_and_meta_for_subject(s, files)
        status = compute_trial_status(qc_df, gap_thr=cfg["gap_thr"], sat_thr=cfg["sat_thr"])
        n_fail = int((status["status"] == "FAIL").sum())
        print(f"[QC] {s:20s} trials={len(files):2d} FAIL={n_fail}")
        qc_all.append(qc_df)
        io_all.append(io_df)
    qc_master = pd.concat(qc_all, ignore_index=True)
    io_master = pd.concat(io_all, ignore_index=True)
    qc_master.to_csv(QC_ALL_PATH, index=False)
    io_master.to_csv(IO_ALL_PATH, index=False)
    print("Saved:", QC_ALL_PATH)
    print("Saved:", IO_ALL_PATH)
    return qc_master, io_master
