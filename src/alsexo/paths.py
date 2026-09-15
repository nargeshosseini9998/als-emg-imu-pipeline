# alsexo/paths.py
# Single source of truth for all filesystem locations.
#
# The project root is resolved in this order:
#   1. the ALSEXO_PROJECT_ROOT environment variable, if set
#   2. the repository root (two levels above this file's package directory)
#
# Raw data are NOT part of the repository (see README / .gitignore).
# Place them under <PROJECT_ROOT>/data/raw/<Subject>/EMG&IMUTest/*.csv

from __future__ import annotations

import os
from pathlib import Path


def _resolve_root() -> Path:
    env = os.environ.get("ALSEXO_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # .../als-emg-exo-pipeline/src/alsexo/paths.py -> repo root
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = _resolve_root()

DATA_ROOT      = PROJECT_ROOT / "data"
RAW_ROOT       = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
FIG_ROOT       = PROJECT_ROOT / "figures"
CONFIG_ROOT    = PROJECT_ROOT / "configs"

# Per-phase output folders (names kept identical to the original pipeline)
QC_DIR          = PROCESSED_ROOT / "qc"
PHASE2_DIR      = PROCESSED_ROOT / "phase_02_preprocess_emg"
PHASE2B_DIR     = PROCESSED_ROOT / "phase_02b_preprocess_imu"
PHASE3_DIR      = PROCESSED_ROOT / "phase_03_normalization"
PHASE3_NORM_DIR = PHASE3_DIR / "normalized_env_per_subject"
PHASE3_BAD_DIR  = PHASE3_DIR / "bad_scales_reports"
PHASE4_DIR      = PROCESSED_ROOT / "phase_04_events"
PHASE5_DIR      = PROCESSED_ROOT / "phase_05_windows"
PHASE6_DIR      = PROCESSED_ROOT / "phase_06_features"
PHASE7_DIR      = PROCESSED_ROOT / "phase_07_kpis"
PHASE8_DIR      = PROCESSED_ROOT / "phase_08_matrix"
PHASE9_EFA_DIR  = PROCESSED_ROOT / "phase_09_efa"
PHASE9_STATS    = PROCESSED_ROOT / "phase_09_stats"
PHASE10_DIR     = PROCESSED_ROOT / "phase_10_supervised"

QC_ALL_PATH = QC_DIR / "00_QC_ALL_SUBJECTS.csv"
IO_ALL_PATH = QC_DIR / "00_IO_ALL.csv"
PHASE3_MASTER_PATH = PHASE3_DIR / "03_ENV_SCALE_ALL_SUBJECTS.csv"


def ensure_dirs() -> None:
    """Create every processed/figure folder (idempotent)."""
    for d in [
        QC_DIR, PHASE2_DIR, PHASE2B_DIR, PHASE3_DIR, PHASE3_NORM_DIR, PHASE3_BAD_DIR,
        PHASE4_DIR, PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR,
        PHASE9_STATS, PHASE10_DIR, FIG_ROOT,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def subject_raw_dir(subject: str) -> Path:
    """Folder containing the functional-task CSV exports of one subject."""
    d = RAW_ROOT / subject / "EMG&IMUTest"
    if not d.exists():
        # one subject folder uses a lower-case 't'
        alt = RAW_ROOT / subject / "EMG&IMUtest"
        if alt.exists():
            return alt
    return d


def subject_csv_files(subject: str) -> list[Path]:
    d = subject_raw_dir(subject)
    if not d.exists():
        raise FileNotFoundError(f"Subject dir not found: {d}")
    return sorted(d.glob("*.csv"))
