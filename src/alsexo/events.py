# alsexo/events.py
# Phase 4 — movement-episode detection.
#
# This module consolidates the FIVE detector implementations that were spread
# over 04_events.ipynb, 04_events_als3.ipynb and 04_exceptions.ipynb into one
# function family selected by a `variant` name:
#
#   variant        original location                    subjects
#   -------------  -----------------------------------  ---------------------------
#   standard       04_events.ipynb / 04_exceptions c8   all Healthy; ALS 1,2,5,14,15,16
#   standard_v1    04_events_als3.ipynb                 ALS 3
#   standard_als   04_exceptions.ipynb cell 0           ALS 8, 9, 10
#   tremor         04_exceptions.ipynb cells 5–7        ALS 13, 12, 11
#   weak           04_exceptions.ipynb cells 9–10       ALS 7, 4
#
# IMPORTANT — provenance statement (see docs/METHODS.md §4):
# The variant and every per-subject / per-trial parameter override were chosen
# by VISUAL INSPECTION of the gate signal and the detected episodes, not by an
# automatic rule. All of those choices are recorded in configs/phase4_episodes.yaml
# and configs/manual_curation.yaml; nothing is hard-coded here. The code below
# reproduces the original behaviour exactly, including a few quirks that are
# flagged with "# QUIRK" comments so they are visible rather than silently fixed.

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt

from .config import phase4_cfg, curation_cfg
from .paths import PHASE2B_DIR, PHASE3_NORM_DIR, PHASE4_DIR

EMPTY_COLS = ["t_on", "t_off", "duration_s", "thr_on", "thr_off", "peak_gate"]


# ============================================================================
# Gate construction
# ============================================================================
def build_gate_imu(df_imu_trial: pd.DataFrame, use_acc_gate=True, alpha=1.0,
                   time_bin_res=0.001, smooth_kernel=1):
    """Composite motion gate from IMU: g(t) = max(gyro_gate, alpha * acc_gate).

    The long-format table (one row per sensor sample) is collapsed to a single
    time series by taking the MEDIAN over sensors within 1 ms bins.
    """
    df = df_imu_trial.copy()
    df["t_bin"] = (df["t_imu"] / time_bin_res).round().astype(int)
    agg = {"t_imu": "mean", "gyro_gate": "median"}
    if use_acc_gate and "acc_gate" in df.columns:
        agg["acc_gate"] = "median"
    df_agg = df.groupby("t_bin", as_index=False).agg(agg).sort_values("t_imu")
    t = df_agg["t_imu"].to_numpy(dtype=float)
    gyro = df_agg["gyro_gate"].to_numpy(dtype=float)
    if use_acc_gate and "acc_gate" in df_agg.columns:
        acc = df_agg["acc_gate"].to_numpy(dtype=float)
        gate = np.maximum(gyro, alpha * acc)
        source = f"max(gyro_gate, {alpha:.2f} x acc_gate)"
    else:
        gate, source = gyro, "gyro_gate"
    if smooth_kernel > 1:
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        gate = medfilt(gate, kernel_size=smooth_kernel)
    log = {"n_sensors_used": int(df["sensor"].nunique()) if "sensor" in df else 1,
           "gate_median": float(np.median(gate)) if len(gate) else np.nan,
           "gate_95p": float(np.percentile(gate, 95)) if len(gate) else np.nan,
           "smooth_kernel": int(smooth_kernel)}
    return t, gate, source, log


def build_gate_emg(df_emg_trial: pd.DataFrame, time_bin_res=0.001, smooth_kernel=1,
                   apply_tkeo=False, lowvar_mode="x5"):
    """Fallback gate when a trial has no IMU: normalised envelope, median over
    sensors within 1 ms bins.

    lowvar_mode: what to do when var(gate) < 1e-5
        "x5"    -> gate *= 5          (standard / standard_v1 / standard_als)
        "p95"   -> gate *= p95/25     (weak)
    """
    df = df_emg_trial.copy()
    env_cols = [c for c in df.columns if c.startswith("env_norm_")]
    if not env_cols:
        env_cols = [c for c in df.columns if "env_norm" in c.lower()]
    if not env_cols:
        raise ValueError("No env_norm columns found.")
    df["emg_gate"] = df[env_cols].max(axis=1)
    df["t_bin"] = (df["t_emg"] / time_bin_res).round().astype(int)
    df_agg = df.groupby("t_bin", as_index=False).agg({"t_emg": "mean", "emg_gate": "median"}).sort_values("t_emg")
    t = df_agg["t_emg"].to_numpy(dtype=float)
    gate = df_agg["emg_gate"].to_numpy(dtype=float)
    if smooth_kernel > 1:
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        gate = medfilt(gate, kernel_size=smooth_kernel)
    if apply_tkeo and len(gate) > 2:
        tk = gate[1:-1] ** 2 - gate[:-2] * gate[2:]
        gate = np.concatenate(([gate[0]], tk, [gate[-1]]))
    gate_var = float(np.var(gate)) if len(gate) else np.nan
    if np.isfinite(gate_var) and gate_var < 1e-5:
        if lowvar_mode == "x5":
            gate = gate * 5.0
        else:
            p95 = np.percentile(gate, 95) if len(gate) else 0.0
            gate = gate * ((p95 / 25.0) if p95 > 0 else 4.0)
    log = {"n_muscles_used": int(len(env_cols)),
           "gate_median": float(np.median(gate)) if len(gate) else np.nan,
           "gate_95p": float(np.percentile(gate, 95)) if len(gate) else np.nan,
           "gate_var": gate_var}
    return t, gate, f"max_of_{len(env_cols)}_env_norm_muscles", log


def _lowpass_simple(data, cutoff, fs, order=2):
    if len(data) < 10 or fs <= 0:
        return data
    wn = cutoff / (0.5 * fs)
    if wn >= 1.0:
        return data
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, data)


def build_gate_tremor(df_trial: pd.DataFrame, use_imu: bool, lp_imu=7.0, lp_emg=3.0):
    """Gate used by the `tremor` variant.

    # QUIRK: this variant does NOT bin/median over sensors. It takes the raw
    # long-format columns as-is (rows are blocked per sensor), estimates fs from
    # the mean sample interval of that concatenated time vector, and low-pass
    # filters the concatenated series. Episodes are therefore detected per
    # sensor block in sequence; episodes with negative duration (crossing a
    # sensor boundary) are removed by the duration filter. Reproduced verbatim.
    """
    if use_imu:
        t = df_trial["t_imu"].values
        acc = df_trial["acc_gate"].values if "acc_gate" in df_trial.columns else 0
        gate_raw = np.maximum(df_trial["gyro_gate"].values, acc)
        fs = 1 / np.mean(np.diff(t)) if len(t) > 1 else 148.0
        gate = _lowpass_simple(gate_raw, lp_imu, fs)
    else:
        t = df_trial["t_emg"].values
        env_cols = [c for c in df_trial.columns if c.startswith("env_norm_")] or \
                   [c for c in df_trial.columns if "env_norm" in c.lower()]
        gate_raw = df_trial[env_cols].max(axis=1).values
        fs = 1 / np.mean(np.diff(t)) if len(t) > 1 else 1259.0
        gate = _lowpass_simple(gate_raw, lp_emg, fs)
    log = {"gate_median": float(np.median(gate)), "gate_95p": float(np.percentile(gate, 95))}
    return np.asarray(t, float), np.asarray(gate, float), "max(gyro_gate, acc_gate)_lp", log


# ============================================================================
# Detectors
# ============================================================================
def _hysteresis_intervals(t, gate, thr_on, thr_off, min_raw_s=0.0):
    episodes, in_ep, t_start, last_open = [], False, None, False
    for i in range(len(t)):
        if not in_ep:
            if gate[i] >= thr_on:
                in_ep, t_start = True, float(t[i])
        else:
            if gate[i] <= thr_off:
                if float(t[i]) - t_start >= min_raw_s:
                    episodes.append([t_start, float(t[i])])
                in_ep, t_start = False, None
    if in_ep and (float(t[-1]) - t_start) >= min_raw_s:
        episodes.append([t_start, float(t[-1])])
        last_open = True
    return episodes, last_open


def _merge(intervals, gap):
    merged = [intervals[0]]
    for cur in intervals[1:]:
        if cur[0] - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], cur[1])
        else:
            merged.append(cur)
    return merged


def detect_episodes_mad(t, gate, k_on=5.0, k_off=3.5, min_duration_s=1.8, merge_gap_s=0.45,
                        pad_s=0.15, peak_multiplier=2.5, secondary_merge_gap=0.8,
                        adjust_rule="standard", third_merge_gap=1.2):
    """Double-threshold detector with Hampel (median + k*MAD) thresholds.

    adjust_rule (automatic onset-threshold correction):
        "standard"     if thr_on > P95:          k_on = max(1.0, 0.7*k_on)
        "standard_als" if thr_on > 0.85*P95 or P95 < 60:
                                                 k_on = max(1.5, 0.55*k_on); k_off = max(0.8, 0.6*k_off)
        "weak"         if thr_on > 1.1*P95:      k_on = max(1.2, 0.65*k_on)
    third_merge_gap: fixed third merge stage (1.2 s) used by the standard family;
                     None disables it (weak variant).
    """
    t = np.asarray(t, float); gate = np.asarray(gate, float)
    m = np.isfinite(t) & np.isfinite(gate)
    t, gate = t[m], gate[m]
    log = {"raw_samples": len(t),
           "gate_median": float(np.median(gate)) if len(gate) else np.nan,
           "gate_95p": float(np.percentile(gate, 95)) if len(gate) else np.nan}
    if len(t) < 20:
        log["warning"] = "too_few_samples"
        return pd.DataFrame(columns=EMPTY_COLS), log

    med = float(np.median(gate))
    mad = float(np.median(np.abs(gate - med))) + 1e-12
    thr_on, thr_off = med + k_on * mad, med + k_off * mad
    log["thr_on"], log["thr_off"] = float(thr_on), float(thr_off)
    p95 = log["gate_95p"]

    if adjust_rule == "standard" and thr_on > p95:
        k_on = max(1.0, k_on * 0.7); thr_on = med + k_on * mad
        log["thr_on_adjusted"] = float(thr_on); log["warning"] = "thr_adjusted_down"
    elif adjust_rule == "standard_als" and (thr_on > p95 * 0.85 or p95 < 60.0):
        k_on = max(1.5, k_on * 0.55); thr_on = med + k_on * mad
        k_off = max(0.8, k_off * 0.6); thr_off = med + k_off * mad
        log["thr_on_adjusted"] = float(thr_on); log["thr_off_adjusted"] = float(thr_off)
        log["warning"] = "thr_adjusted_for_weak_ALS_signal"
    elif adjust_rule == "weak" and thr_on > p95 * 1.1:
        k_on = max(1.2, k_on * 0.65); thr_on = med + k_on * mad; thr_off = med + k_off * mad
        log["thr_on_adjusted"] = float(thr_on); log["warning"] = "thr_adjusted_down"

    episodes, last_open = _hysteresis_intervals(t, gate, thr_on, thr_off)
    if last_open:
        log["last_episode_open"] = True
    if not episodes:
        log["warning"] = "no_episodes_detected"
        return pd.DataFrame(columns=EMPTY_COLS), log

    t_min, t_max = float(t[0]), float(t[-1])
    padded = sorted([[max(t_min, a - pad_s), min(t_max, b + pad_s)] for a, b in episodes], key=lambda x: x[0])
    merged = _merge(padded, merge_gap_s)                 # stage 1
    merged = _merge(merged, secondary_merge_gap)         # stage 2
    if third_merge_gap is not None:
        merged = _merge(merged, third_merge_gap)         # stage 3 (fixed 1.2 s)

    adaptive_peak_thr = med + peak_multiplier * mad
    log["adaptive_peak_thr"] = float(adaptive_peak_thr)
    final = []
    for a, b in merged:
        dur = b - a
        if dur < min_duration_s:
            continue
        seg = gate[(t >= a) & (t <= b)]
        if len(seg) == 0:
            continue
        peak = float(np.max(seg))
        if peak >= adaptive_peak_thr:
            final.append([a, b, dur, thr_on, thr_off, peak])
    df_ep = pd.DataFrame(final, columns=EMPTY_COLS)
    log["n_episodes_after_postproc"] = int(len(df_ep))
    return df_ep, log


def detect_episodes_tremor(t, gate, k_on=4.8, k_off=6.2, min_duration_s=1.8, merge_gap_s=0.4,
                           pad_s=0.15, min_peak_gate=28.0, baseline_percentile=15,
                           thr_off_multiplier=1.3, min_raw_s=0.3):
    """Tremor / irregular-burst detector (inverted hysteresis, k_off > k_on).

    Baseline statistics come from the lowest `baseline_percentile` % of the gate.
    thr_off is additionally scaled by `thr_off_multiplier`; thr_on is capped at
    the gate median. Single merge stage; peak filter uses an absolute value.
    # QUIRK: the original code had a task-dependent multiplier (1.1 for
    # drinking/lifting, 1.3 otherwise) but was always called without the task
    # argument, so 1.3 was applied to every trial. Reproduced as 1.3.
    """
    t = np.asarray(t, float); gate = np.asarray(gate, float)
    m = np.isfinite(t) & np.isfinite(gate)
    t, gate = t[m], gate[m]
    log = {"raw_samples": len(t), "gate_median": float(np.median(gate)), "gate_95p": float(np.percentile(gate, 95))}
    if len(t) < 20:
        log["warning"] = "too_few_samples"
        return pd.DataFrame(), log

    base = gate[gate <= np.percentile(gate, baseline_percentile)]
    if len(base) < 10:
        base = gate
    med = np.median(base)
    mad = np.median(np.abs(base - med)) + 1e-12
    thr_on = med + k_on * mad
    thr_off = (med + k_off * mad) * thr_off_multiplier
    thr_on = min(thr_on, np.percentile(gate, 50))
    log.update({"thr_on": float(thr_on), "thr_off": float(thr_off), "baseline_perc_used": baseline_percentile,
                "thr_off_multiplier": thr_off_multiplier, "k_on": k_on, "k_off": k_off})

    episodes, _ = _hysteresis_intervals(t, gate, thr_on, thr_off, min_raw_s=min_raw_s)
    if not episodes:
        log["warning"] = "no_episodes_raw"
        return pd.DataFrame(), log
    padded = sorted([[max(t[0], a - pad_s), min(t[-1], b + pad_s)] for a, b in episodes], key=lambda x: x[0])
    merged = _merge(padded, merge_gap_s)
    final = []
    for a, b in merged:
        dur = b - a
        if dur < min_duration_s:
            continue
        seg = gate[(t >= a) & (t <= b)]
        if len(seg) == 0:
            continue
        peak = np.max(seg)
        if peak >= min_peak_gate:
            final.append([a, b, dur, peak])
    df_ep = pd.DataFrame(final, columns=["t_on", "t_off", "duration_s", "peak_gate"])
    log["n_episodes_final"] = len(df_ep)
    return df_ep, log


# ============================================================================
# Parameter resolution from config
# ============================================================================
def _task_key(trial_id: str, task_settings: dict) -> str:
    return next((k for k in task_settings if k != "default" and k in trial_id.lower()), "default")


def resolve_trial_params(subject: str, trial_id: str, use_imu: bool, cfg: dict) -> dict[str, Any]:
    """Return the fully resolved parameter set for one trial.

    Order of precedence (lowest -> highest):
      variant defaults -> task settings -> variant task rules -> subject overrides
      -> per-trial overrides in configs/phase4_episodes.yaml
    """
    subj_cfg = cfg["subjects"][subject]
    variant = subj_cfg["variant"]
    v = copy.deepcopy(cfg["variants"][variant])
    task_key = _task_key(trial_id, v["task_settings"])
    ts = v["task_settings"][task_key]
    p = {"variant": variant, "task_key": task_key, "use_imu": use_imu,
         "min_valid_duration_s": v["min_valid_duration_s"],
         "min_valid_peak_gate": v["min_valid_peak_gate"]}

    if variant in ("standard", "standard_v1", "standard_als", "weak"):
        p.update({"k_on": v["k_on"], "k_off": v["k_off"], "pad_s": v["pad_s"],
                  "peak_multiplier": v["peak_multiplier"], "secondary_merge_gap": v["secondary_merge_gap"],
                  "min_duration_s": ts["min_duration_s"], "merge_gap_s": ts["merge_gap_s"],
                  "alpha_acc": v["alpha_acc"], "smooth_kernel": v.get("smooth_kernel", 1),
                  "apply_tkeo": False, "adjust_rule": v["adjust_rule"], "third_merge_gap": v["third_merge_gap"],
                  "lowvar_boost_imu": v.get("lowvar_boost_imu"), "emg_lowvar_mode": v.get("emg_lowvar_mode", "x5")})
        # --- task / condition rules of the standard family ---
        if v.get("min_merge_gap_all_tasks") is not None:
            p["merge_gap_s"] = max(p["merge_gap_s"], v["min_merge_gap_all_tasks"])
            p["secondary_merge_gap"] = max(p["secondary_merge_gap"], v["min_secondary_gap_all_tasks"])
        if v.get("noexo_rule") and "noexo" in trial_id.lower():
            r = v["noexo_rule"]
            p["merge_gap_s"] *= r["merge_gap_factor"]
            p["secondary_merge_gap"] = max(p["secondary_merge_gap"], r["min_secondary_gap"])
            p["min_duration_s"] *= r["min_duration_factor"]
        if not use_imu and v.get("emg_fallback"):
            f = v["emg_fallback"]
            # QUIRK (standard/standard_v1/standard_als): the min-duration factor is
            # applied twice in the original code (once before, once inside the
            # branch) -> effective factor = factor**2.
            p["min_duration_s"] *= f["min_duration_factor"] ** (2 if f.get("double_min_duration_factor") else 1)
            p["merge_gap_s"] *= f["merge_gap_factor"]
            p["secondary_merge_gap"] = f["secondary_merge_gap"]
            p["k_on"], p["k_off"], p["peak_multiplier"] = f["k_on"], f["k_off"], f["peak_multiplier"]
        if variant == "weak":
            w = v["weak_settings"]
            p.update({"k_on": w["k_on"], "k_off": w["k_off"], "peak_multiplier": w["peak_multiplier"],
                      "secondary_merge_gap": w["secondary_merge_gap"], "smooth_kernel": w["smooth_kernel"],
                      "min_valid_duration_s": w["min_valid_duration_s"], "min_valid_peak_gate": w["min_valid_peak_gate"],
                      "apply_tkeo": (not use_imu), "alpha_acc": w["alpha_acc_boost"] if use_imu else p["alpha_acc"]})
            p["min_duration_s"] *= w["min_duration_factor"]
            p["merge_gap_s"] *= w["merge_gap_factor"]
    elif variant == "tremor":
        p.update({"k_on": v["k_on"], "k_off": v["k_off"], "pad_s": v["pad_s"],
                  "min_duration_s": ts.get("min_duration_s", v["default_min_duration_s"]),
                  "merge_gap_s": ts["merge_gap_s"], "min_peak_gate": v["min_valid_peak_gate"],
                  "baseline_percentile": v["baseline_percentile"], "thr_off_multiplier": v["thr_off_multiplier"],
                  "lp_imu": v["gate_lp_imu_hz"], "lp_emg": v["gate_lp_emg_hz"], "min_raw_s": v["min_raw_episode_s"]})
    else:
        raise ValueError(f"unknown variant {variant}")

    # --- subject-level overrides and per-trial overrides (visual-inspection choices) ---
    for k, val in (subj_cfg.get("overrides") or {}).items():
        p[k] = val
    for trial_pat, ov in (subj_cfg.get("trial_overrides") or {}).items():
        if trial_pat == trial_id:
            for k, val in ov.items():
                p[k] = val
    return p


# ============================================================================
# Subject driver
# ============================================================================
def run_phase4_for_subject(subject_name: str, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or phase4_cfg()
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    imu_path = PHASE2B_DIR / f"{subject_name}__imu_gate.parquet"
    emg_path = PHASE3_NORM_DIR / f"{subject_name}__emg_env_norm.parquet"
    df_imu = pd.read_parquet(imu_path) if imu_path.exists() else pd.DataFrame()
    df_emg = pd.read_parquet(emg_path) if emg_path.exists() else pd.DataFrame()
    for d in (df_imu, df_emg):
        if not d.empty:
            d["trial_id"] = d["trial_id"].astype(str)
    if df_imu.empty and df_emg.empty:
        raise FileNotFoundError(f"No data found for {subject_name}")
    trials_imu = set(df_imu["trial_id"].unique()) if not df_imu.empty else set()
    trials_emg = set(df_emg["trial_id"].unique()) if not df_emg.empty else set()
    all_trials = sorted(trials_imu | trials_emg)
    variant = cfg["subjects"][subject_name]["variant"]
    print(f"[P4] {subject_name}: variant={variant} trials={len(all_trials)} (IMU {len(trials_imu)}, EMG-only {len(trials_emg - trials_imu)})")

    events, logs = [], []
    for trial_id in all_trials:
        use_imu = trial_id in trials_imu
        df_trial = (df_imu if use_imu else df_emg)
        df_trial = df_trial[df_trial["trial_id"] == trial_id].copy()
        if df_trial.empty:
            continue
        p = resolve_trial_params(subject_name, trial_id, use_imu, cfg)
        source_type = "IMU" if use_imu else "EMG_fallback"

        # ---- gate ----
        if variant == "tremor":
            t, gate, gate_source, gate_log = build_gate_tremor(df_trial, use_imu, p["lp_imu"], p["lp_emg"])
        elif use_imu:
            t, gate, gate_source, gate_log = build_gate_imu(df_trial, True, p["alpha_acc"], smooth_kernel=p["smooth_kernel"])
        else:
            t, gate, gate_source, gate_log = build_gate_emg(df_trial, smooth_kernel=p["smooth_kernel"],
                                                          apply_tkeo=p["apply_tkeo"], lowvar_mode=p["emg_lowvar_mode"])
        gate_var = float(np.var(gate)) if len(gate) else np.nan
        gate_log["gate_var"] = gate_var
        if variant != "tremor" and p.get("lowvar_boost_imu") and np.isfinite(gate_var) and gate_var < p["lowvar_boost_imu"]["var_threshold"]:
            gate = gate * p["lowvar_boost_imu"]["factor"]
            gate_log["sanity_flag"] = "low_variance_boosted"
        elif np.isfinite(gate_var) and gate_var < 1e-5:
            gate_log["sanity_flag"] = "very_low_variance"

        # ---- detection ----
        if variant == "tremor":
            ep_df, ep_log = detect_episodes_tremor(
                t, gate, k_on=p["k_on"], k_off=p["k_off"], min_duration_s=p["min_duration_s"],
                merge_gap_s=p["merge_gap_s"], pad_s=p["pad_s"], min_peak_gate=p["min_peak_gate"],
                baseline_percentile=p["baseline_percentile"], thr_off_multiplier=p["thr_off_multiplier"],
                min_raw_s=p["min_raw_s"])
        else:
            ep_df, ep_log = detect_episodes_mad(
                t, gate, k_on=p["k_on"], k_off=p["k_off"], min_duration_s=p["min_duration_s"],
                merge_gap_s=p["merge_gap_s"], pad_s=p["pad_s"], peak_multiplier=p["peak_multiplier"],
                secondary_merge_gap=p["secondary_merge_gap"], adjust_rule=p["adjust_rule"],
                third_merge_gap=p["third_merge_gap"])

        # ---- final validity filter ----
        if not ep_df.empty:
            before = len(ep_df)
            ep_df = ep_df[ep_df["duration_s"] >= p["min_valid_duration_s"]].copy()
            if p["min_valid_peak_gate"] is not None:
                ep_df = ep_df[ep_df["peak_gate"] >= p["min_valid_peak_gate"]].copy()
            if before - len(ep_df) > 0:
                ep_log["discarded_episodes"] = int(before - len(ep_df))
        # ---- subject-specific post-processing rules (recorded in phase4_episodes.yaml) ----
        if not ep_df.empty and p.get("prune_rule") and p["task_key"] == p["prune_rule"].get("task"):
            r = p["prune_rule"]; before = len(ep_df)
            ep_df = ep_df[ep_df["duration_s"] >= r["min_duration_s"]].copy()
            if len(ep_df) > r["max_episodes"]:
                ep_df = ep_df.sort_values("peak_gate").iloc[1:].reset_index(drop=True)   # drop weakest
            if len(ep_df) >= r["max_episodes"]:
                second = ep_df.iloc[1]
                if second["duration_s"] < r["second_min_duration_s"] or second["peak_gate"] < r["second_min_peak"]:
                    ep_df = ep_df.drop(ep_df.index[1]).reset_index(drop=True)
            if len(ep_df) < before:
                ep_log["prune_rule_removed"] = int(before - len(ep_df))
        if p.get("max_episodes_per_trial") is not None:
            while len(ep_df) > p["max_episodes_per_trial"]:               # keep the N longest
                ep_df = ep_df.sort_values("duration_s").iloc[1:].reset_index(drop=True)
                ep_log["max_episodes_rule"] = f"trimmed to {p['max_episodes_per_trial']}"
        if p.get("drop_all_episodes"):
            ep_df = pd.DataFrame()
            ep_log["manually_dropped"] = "all"

        log = {"subject": subject_name, "trial_id": trial_id, "task_type": p["task_key"], "variant": variant,
               "gate_source": gate_source, "data_source": source_type,
               **{f"param_{k}": v for k, v in p.items() if not isinstance(v, dict)},
               **gate_log, **ep_log}
        logs.append(log)
        if ep_df.empty:
            continue

        # recompute peak_gate on the final gate (as in the original notebooks)
        ep_df["peak_gate"] = [float(np.nanmax(gate[(t >= a) & (t <= b)])) if np.any((t >= a) & (t <= b)) else np.nan
                              for a, b in zip(ep_df["t_on"], ep_df["t_off"])]
        ep_df = ep_df.sort_values("t_on").reset_index(drop=True)
        ep_df["episode_id"] = range(1, len(ep_df) + 1)
        ep_df["subject"], ep_df["trial_id"], ep_df["task_type"] = subject_name, trial_id, p["task_key"]
        ep_df["gate_source"], ep_df["data_source"], ep_df["variant"] = gate_source, source_type, variant
        events.append(ep_df)

    events_df = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    logs_df = pd.DataFrame(logs)
    events_df = apply_manual_curation(subject_name, events_df)
    events_df.to_parquet(PHASE4_DIR / f"{subject_name}__episodes.parquet", index=False)
    logs_df.to_csv(PHASE4_DIR / f"{subject_name}__episodes_log.csv", index=False)
    print(f"[P4] {subject_name}: {len(events_df)} valid episodes saved")
    return events_df, logs_df


# ============================================================================
# Manual curation (post-detection episode removals, from configs/manual_curation.yaml)
# ============================================================================
def apply_manual_curation(subject_name: str, events_df: pd.DataFrame) -> pd.DataFrame:
    cur = curation_cfg().get("episode_removals", {}).get(subject_name, [])
    if not cur or events_df.empty:
        return events_df
    df = events_df.copy()
    for rule in cur:
        mask = df["trial_id"].astype(str).str.contains(rule["trial"], case=False, regex=False)
        sub = df[mask].sort_values("t_on")
        if sub.empty:
            print(f"[P4-curation] {subject_name}: no episodes match trial '{rule['trial']}' (nothing to drop)")
            continue
        action = rule["action"]
        if action == "drop_all":
            idx = list(sub.index)
        elif action == "drop_longest":
            idx = [sub["duration_s"].idxmax()]
        elif action == "drop_last":
            idx = [sub.index[-1]]
        elif action == "drop_kth":
            k = int(rule["k"])
            idx = [sub.index[k - 1]] if 1 <= k <= len(sub) else []
        else:
            raise ValueError(f"unknown curation action {action}")
        df = df.drop(index=idx)
        print(f"[P4-curation] {subject_name}/{rule['trial']}: {action} -> removed {len(idx)} episode(s) ({rule.get('reason','')})")
        tid = sub["trial_id"].iloc[0]
        m2 = df["trial_id"] == tid
        df.loc[m2, "episode_id"] = df[m2].sort_values("t_on").groupby("trial_id").cumcount() + 1
    return df.reset_index(drop=True)
