# alsexo/preprocess_imu.py
# Phase 2b — IMU preprocessing (ported from 02_preprocessing_imu.ipynb).
#
#   gyro_mag  = ||omega||                  (deg/s)
#   gyro_gate = LP(gyro_mag, 10 Hz, order 4, zero-phase)
#   acc_mag   = ||a||                      (g)
#   acc_dyn   = acc_mag - LP(acc_mag, 0.5 Hz, order 2)   # gravity/baseline removal
#                                                        # applied to the MAGNITUDE
#   acc_gate  = LP(|acc_dyn|, 10 Hz, order 4)
#
# The same QC admission rule as Phase 2 is used (OK trials + repair trials);
# no saturation repair is performed on IMU signals.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .config import pipeline_cfg
from .paths import PHASE2B_DIR, subject_csv_files
from .qc import load_qc_subject, compute_trial_status
from .trigno_io import read_trigno_csv


def lowpass_filt(x, fs, cutoff_hz=10.0, order=4):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    b, a = butter(order, cutoff_hz / (0.5 * fs), btype="lowpass")
    return filtfilt(b, a, x)


def preprocess_imu_trial(subject_name: str, csv_path: Path, qc_subj: pd.DataFrame, cfg: dict):
    sensors, per_sensor, meta = read_trigno_csv(csv_path)
    trial_id = csv_path.stem
    rows = []
    for sensor in sensors:
        imu = per_sensor[sensor].get("imu")
        if imu is None or len(imu) == 0:
            continue
        qrow = qc_subj[(qc_subj["trial_id"] == trial_id) & (qc_subj["sensor"] == sensor)]
        if len(qrow) == 1 and np.isfinite(qrow["imu_gyro_mag_fs_est"].values[0]):
            fs_imu = float(qrow["imu_gyro_mag_fs_est"].values[0])
        else:
            fs_imu = float(cfg["fs_fallback"])

        t = imu["t_imu"].to_numpy()
        gx, gy, gz = (imu[c].to_numpy() for c in ("gyro_x_dps", "gyro_y_dps", "gyro_z_dps"))
        gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
        gyro_gate = lowpass_filt(gyro_mag, fs=fs_imu, cutoff_hz=cfg["gate_lp_hz"], order=cfg["gate_lp_order"])

        out = pd.DataFrame({
            "subject": subject_name, "trial_id": trial_id, "sensor": sensor, "t_imu": t,
            "gyro_x_dps": gx, "gyro_y_dps": gy, "gyro_z_dps": gz,
            "gyro_mag": gyro_mag, "gyro_gate": gyro_gate, "fs_imu_used": fs_imu,
        })

        ax, ay, az = (imu[c].to_numpy() for c in ("acc_x_g", "acc_y_g", "acc_z_g"))
        acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
        acc_baseline = lowpass_filt(acc_mag, fs=fs_imu, cutoff_hz=cfg["gravity_lp_hz"], order=cfg["gravity_lp_order"])
        acc_dyn = acc_mag - acc_baseline
        acc_gate = lowpass_filt(np.abs(acc_dyn), fs=fs_imu, cutoff_hz=cfg["gate_lp_hz"], order=cfg["gate_lp_order"])
        out["acc_x_g"], out["acc_y_g"], out["acc_z_g"] = ax, ay, az
        out["acc_mag"], out["acc_dyn"], out["acc_gate"] = acc_mag, acc_dyn, acc_gate
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_phase2b_for_subject(subject_name: str) -> pd.DataFrame:
    cfg = pipeline_cfg()["phase2b"]
    p2 = pipeline_cfg()["phase2"]
    qc_cfg = pipeline_cfg()["qc"]
    PHASE2B_DIR.mkdir(parents=True, exist_ok=True)

    files = subject_csv_files(subject_name)
    qc_subj = load_qc_subject(subject_name)
    trial_status = compute_trial_status(qc_subj, gap_thr=qc_cfg["gap_thr"], sat_thr=qc_cfg["sat_thr"])
    ok_trials = set(trial_status.loc[trial_status["status"] == "OK", "trial_id"].astype(str))
    allowed = ok_trials | set(p2["repair_trials"].get(subject_name, []))
    files_allowed = [fp for fp in files if fp.stem in allowed]
    print(f"[P2b] {subject_name}: processing {len(files_allowed)} of {len(files)} files")

    dfs = [preprocess_imu_trial(subject_name, fp, qc_subj, cfg) for fp in files_allowed]
    dfs = [d for d in dfs if not d.empty]
    subj_imu = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if subj_imu.empty:
        print("WARNING: No IMU rows produced. Nothing saved.")
        return subj_imu
    out = PHASE2B_DIR / f"{subject_name}__imu_gate.parquet"
    subj_imu.to_parquet(out, index=False)
    print("Saved:", out)
    return subj_imu
