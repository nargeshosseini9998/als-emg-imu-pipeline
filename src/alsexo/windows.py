# alsexo/windows.py
# Phase 5 — temporal windowing inside detected episodes (ported from 05_segmentation.ipynb).
#
#   200 ms windows, 50 % overlap (100 ms step), strictly inside [t_on, t_off].
#   Per-modality window QC: coverage >= 60 % of the expected sample count and
#   max time gap inside the window <= 50 ms -> emg_ok / imu_ok flags.
#   Only window boundaries + flags are stored (metadata table); signals are not copied.

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import pipeline_cfg
from .paths import PHASE2B_DIR, PHASE3_NORM_DIR, PHASE4_DIR, PHASE5_DIR


def estimate_fs(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    t = np.sort(np.unique(t[np.isfinite(t)]))
    if len(t) < 2:
        return np.nan
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if len(dt) == 0:
        return np.nan
    return float(1.0 / np.median(dt))


def make_windows(t_on, t_off, win_len_s, step_s):
    if (t_off - t_on) < win_len_s:
        return []
    starts = np.arange(t_on, t_off - win_len_s + 1e-12, step_s)
    return [(float(s), float(s + win_len_s)) for s in starts]


def slice_by_time(df, tcol, t_start, t_end):
    return df[(df[tcol] >= t_start) & (df[tcol] < t_end)]


def window_timegap_ok(t, max_gap_s):
    gaps = np.diff(t)
    gaps = gaps[np.isfinite(gaps)]
    if len(gaps) == 0:
        return False, np.nan
    mg = float(np.max(gaps))
    return (mg <= max_gap_s), mg


def expected_samples(fs, win_len_s):
    return float(fs * win_len_s) if np.isfinite(fs) and fs > 0 else np.nan


def qc_coverage(n_samples, n_expected, min_ratio):
    if not np.isfinite(n_expected) or n_expected <= 0:
        return False
    return (n_samples / n_expected) >= min_ratio


def run_phase5_for_subject(subject_name: str) -> pd.DataFrame:
    cfg = pipeline_cfg()["phase5"]
    win_len, step = cfg["win_len_s"], cfg["win_len_s"] * (1.0 - cfg["overlap"])
    PHASE5_DIR.mkdir(parents=True, exist_ok=True)

    ep_path = PHASE4_DIR / f"{subject_name}__episodes.parquet"
    if not ep_path.exists():
        raise FileNotFoundError(f"Episodes file not found: {ep_path}")
    episodes = pd.read_parquet(ep_path)
    if episodes.empty:
        print(f"[P5] {subject_name}: no episodes -> empty window table")
        pd.DataFrame().to_parquet(PHASE5_DIR / f"{subject_name}__windows.parquet", index=False)
        return pd.DataFrame()
    episodes["trial_id"] = episodes["trial_id"].astype(str)
    emg_path = PHASE3_NORM_DIR / f"{subject_name}__emg_env_norm.parquet"
    imu_path = PHASE2B_DIR / f"{subject_name}__imu_gate.parquet"
    df_emg = pd.read_parquet(emg_path) if emg_path.exists() else pd.DataFrame()
    df_imu = pd.read_parquet(imu_path) if imu_path.exists() else pd.DataFrame()
    for d in (df_emg, df_imu):
        if not d.empty:
            d["trial_id"] = d["trial_id"].astype(str)

    fs_emg_by_trial = {tid: estimate_fs(g["t_emg"].to_numpy()) for tid, g in df_emg.groupby("trial_id")} if not df_emg.empty else {}
    fs_imu_by_trial = {tid: estimate_fs(g["t_imu"].to_numpy()) for tid, g in df_imu.groupby("trial_id")} if not df_imu.empty else {}

    rows = []
    for (trial_id, episode_id), ep_grp in episodes.groupby(["trial_id", "episode_id"]):
        ep0 = ep_grp.iloc[0]
        t_on, t_off = float(ep0["t_on"]), float(ep0["t_off"])
        if cfg["include_context"]:
            t_on_eff, t_off_eff = t_on - cfg["context_pad_s"], t_off + cfg["context_pad_s"]
        else:
            t_on_eff, t_off_eff = t_on, t_off
        win_list = make_windows(t_on_eff, t_off_eff, win_len, step)
        if not win_list:
            continue
        emg_trial = df_emg[df_emg["trial_id"] == trial_id].sort_values("t_emg") if not df_emg.empty else pd.DataFrame()
        imu_trial = df_imu[df_imu["trial_id"] == trial_id].sort_values("t_imu") if not df_imu.empty else pd.DataFrame()
        fs_emg, fs_imu = fs_emg_by_trial.get(trial_id, np.nan), fs_imu_by_trial.get(trial_id, np.nan)
        exp_emg, exp_imu = expected_samples(fs_emg, win_len), expected_samples(fs_imu, win_len)

        for w_id, (ts, te) in enumerate(win_list, start=1):
            emg_win = slice_by_time(emg_trial, "t_emg", ts, te) if not emg_trial.empty else pd.DataFrame()
            imu_win = slice_by_time(imu_trial, "t_imu", ts, te) if not imu_trial.empty else pd.DataFrame()
            emg_n = int(emg_win["t_emg"].nunique()) if len(emg_win) else 0
            imu_n = int(imu_win["t_imu"].nunique()) if len(imu_win) else 0
            emg_cov_ok = qc_coverage(emg_n, exp_emg, cfg["min_coverage_ratio"]) if emg_n > 0 else False
            imu_cov_ok = qc_coverage(imu_n, exp_imu, cfg["min_coverage_ratio"]) if imu_n > 0 else False
            emg_gap_ok, emg_max_gap = (False, np.nan)
            imu_gap_ok, imu_max_gap = (False, np.nan)
            if emg_n > 1:
                emg_gap_ok, emg_max_gap = window_timegap_ok(np.sort(emg_win["t_emg"].unique()), cfg["max_time_gap_s"])
            if imu_n > 1:
                imu_gap_ok, imu_max_gap = window_timegap_ok(np.sort(imu_win["t_imu"].unique()), cfg["max_time_gap_s"])
            rows.append({
                "subject": subject_name, "trial_id": trial_id, "episode_id": int(episode_id), "window_id": int(w_id),
                "t_start": float(ts), "t_end": float(te), "win_len_s": float(win_len), "step_s": float(step),
                "task_type": ep0.get("task_type", "unknown"), "data_source": ep0.get("data_source", "unknown"),
                "gate_source": ep0.get("gate_source", "unknown"), "t_on": t_on, "t_off": t_off,
                "context_used": bool(cfg["include_context"]),
                "fs_emg_est": fs_emg, "fs_imu_est": fs_imu, "emg_expected_n": exp_emg, "imu_expected_n": exp_imu,
                "emg_n": emg_n, "imu_n": imu_n, "emg_n_rows": int(len(emg_win)), "imu_n_rows": int(len(imu_win)),
                "emg_cov_ok": bool(emg_cov_ok), "imu_cov_ok": bool(imu_cov_ok),
                "emg_gap_ok": bool(emg_gap_ok), "imu_gap_ok": bool(imu_gap_ok),
                "emg_max_gap_s": emg_max_gap, "imu_max_gap_s": imu_max_gap,
                "emg_ok": bool(emg_cov_ok and emg_gap_ok), "imu_ok": bool(imu_cov_ok and imu_gap_ok),
            })
    windows_df = pd.DataFrame(rows)
    out = PHASE5_DIR / f"{subject_name}__windows.parquet"
    windows_df.to_parquet(out, index=False)
    if not windows_df.empty:
        ok = windows_df[["emg_ok", "imu_ok"]].mean()
        print(f"[P5] {subject_name}: {len(windows_df)} windows | emg_ok={ok['emg_ok']:.3f} imu_ok={ok['imu_ok']:.3f}")
    return windows_df
