# alsexo/normalize.py
# Phase 3 — within-subject envelope normalisation (ported from 03_normalization.ipynb).
#
# For every (subject, sensor):
#   env_scale = P95 of the envelope pooled over all admitted trials, estimated on
#               a random 2 % sample drawn within each sensor (min 5 000 /
#               max 200 000 samples, random_state 0); tiny negative envelope
#               values (filtfilt ringing) are clipped to 0 before the percentile.
#   env_norm  = env / env_scale                     (dimensionless, ~1 = near-max)
#   Scales < 1e-8 (or NaN) are flagged and the sensor is left un-normalised (NaN).
#
# NOTE: env_norm is used only for WITHIN-subject purposes (Phase-4 EMG-fallback
# gate, Phase-6 co-activation, Phase-7 peak/IEMG/duty). Between-group amplitude
# comparisons use the absolute band-passed EMG (Phase 6 RMS).

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import pipeline_cfg
from .paths import PHASE2_DIR, PHASE3_DIR, PHASE3_NORM_DIR, PHASE3_BAD_DIR, PHASE3_MASTER_PATH


def load_phase2_subject(subject_name: str, columns=None) -> pd.DataFrame:
    path = PHASE2_DIR / f"{subject_name}__emg_bp_env.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Phase-2 file not found: {path}")
    df = pd.read_parquet(path, columns=columns)
    df["subject"] = df["subject"].astype(str)
    df["sensor"] = df["sensor"].astype(str)
    return df


def sample_for_percentile(df_subj: pd.DataFrame, mode="random_per_sensor", frac=0.02,
                          min_per_sensor=5_000, max_per_sensor=200_000, random_state=0) -> pd.DataFrame:
    df = df_subj.copy()
    env = df["env"].to_numpy(dtype=float)
    df = df.loc[np.isfinite(env)].copy()
    df["env_clip"] = np.clip(df["env"].to_numpy(dtype=float), 0.0, None)

    if mode == "none":
        return df
    if mode == "random_global":
        n = int(round(len(df) * frac))
        n = max(1, min(n, len(df)))
        return df.sample(n=n, random_state=random_state)
    if mode == "random_per_sensor":
        parts = []
        for _, g in df.groupby(["subject", "sensor"], sort=False):
            n = int(round(len(g) * frac))
            n = max(min_per_sensor, n)
            n = min(n, len(g), max_per_sensor)
            parts.append(g.sample(n=n, random_state=random_state) if n < len(g) else g)
        return pd.concat(parts, ignore_index=True)
    raise ValueError("mode must be one of: none, random_per_sensor, random_global")


def compute_percentile_scales(df_sampled: pd.DataFrame, p: int = 95, too_small_thr: float = 1e-8) -> pd.DataFrame:
    q = p / 100.0
    scales = (df_sampled.groupby(["subject", "sensor"], as_index=False)
              .agg(env_scale=("env_clip", lambda x: float(np.quantile(x.to_numpy(), q)) if len(x) else np.nan),
                   n_used_for_scale=("env_clip", "size")))
    scales["p"] = int(p)
    scales["env_scale"] = scales["env_scale"].replace(0.0, np.nan)
    scales["scale_too_small_flag"] = scales["env_scale"].isna() | (scales["env_scale"] < too_small_thr)
    scales.loc[scales["scale_too_small_flag"], "env_scale"] = np.nan
    return scales[["subject", "sensor", "p", "env_scale", "scale_too_small_flag", "n_used_for_scale"]]


def compute_scales_for_subject(subject_name: str) -> pd.DataFrame:
    cfg = pipeline_cfg()["phase3"]
    df_subj = load_phase2_subject(subject_name, columns=["subject", "trial_id", "sensor", "env"])
    df_s = sample_for_percentile(df_subj, mode=cfg["sampling_mode"], frac=cfg["frac"],
                                 min_per_sensor=cfg["min_per_sensor"], max_per_sensor=cfg["max_per_sensor"],
                                 random_state=cfg["random_state"])
    scales = compute_percentile_scales(df_s, p=cfg["percentile"], too_small_thr=cfg["too_small_thr"])
    scales["sampling_mode"] = cfg["sampling_mode"]
    scales["frac"] = float(cfg["frac"])
    scales["min_per_sensor"] = int(cfg["min_per_sensor"])
    scales["max_per_sensor"] = int(cfg["max_per_sensor"])
    scales["random_state"] = int(cfg["random_state"])
    scales["phase2_rows"] = int(len(df_subj))
    scales["phase2_trials"] = int(df_subj["trial_id"].nunique())
    scales["computed_utc"] = datetime.now(timezone.utc).isoformat()

    bad = scales[scales["scale_too_small_flag"]]
    if not bad.empty:
        PHASE3_BAD_DIR.mkdir(parents=True, exist_ok=True)
        bad.to_csv(PHASE3_BAD_DIR / f"{subject_name}__p{cfg['percentile']}__bad_scales.csv", index=False)
        print(f"[P3] {subject_name}: {len(bad)} bad scale(s) flagged")
    return scales


def apply_normalization(subject_name: str, scales_master: pd.DataFrame, p: int = 95) -> pd.DataFrame:
    scales_sub = scales_master[(scales_master["subject"].astype(str) == str(subject_name)) &
                               (scales_master["p"].astype(int) == int(p)) &
                               (~scales_master["scale_too_small_flag"].astype(bool)) &
                               (scales_master["env_scale"].notna())].copy()
    if scales_sub.empty:
        raise ValueError(f"No valid scale for {subject_name} (p={p}).")
    df = load_phase2_subject(subject_name)
    df_merged = df.merge(scales_sub[["sensor", "env_scale"]], on="sensor", how="left")
    df_merged["env_norm"] = np.where(df_merged["env_scale"].notna(), df_merged["env"] / df_merged["env_scale"], np.nan)
    out = PHASE3_NORM_DIR / f"{subject_name}__emg_env_norm.parquet"
    df_merged.to_parquet(out, index=False)
    print(f"[P3] {subject_name}: normalised -> {out.name} (sensor coverage {df_merged['env_scale'].notna().mean():.3f})")
    return df_merged


def run_phase3_all(subjects: list[str]) -> pd.DataFrame:
    """Compute the master scale table for all subjects, then normalise each subject."""
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3_NORM_DIR.mkdir(parents=True, exist_ok=True)
    p = pipeline_cfg()["phase3"]["percentile"]
    master = pd.concat([compute_scales_for_subject(s) for s in subjects], ignore_index=True)
    master.to_csv(PHASE3_MASTER_PATH, index=False)
    print("Saved master scales:", PHASE3_MASTER_PATH)
    for s in subjects:
        apply_normalization(s, master, p=p)
    return master
