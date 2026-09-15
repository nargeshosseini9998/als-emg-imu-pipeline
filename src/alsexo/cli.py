# alsexo/cli.py
# Command-line runner:  python -m alsexo.cli <phase> [--subjects S1 S2 ...]
#
#   python -m alsexo.cli all              # run Phases 0 -> 10 in order
#   python -m alsexo.cli 0                # parsing + QC (writes qc/00_QC_ALL_SUBJECTS.csv)
#   python -m alsexo.cli 2 2b 3 4 5 6 7   # any subset, in the given order
#   python -m alsexo.cli 9                # all Phase-9 statistical steps
#   python -m alsexo.cli 9:step_g_lmm     # one Phase-9 step
#
# Phase-level modules that were script-style notebook cells (Phases 7-10) are executed
# with runpy so their top-level code runs exactly as in the notebooks.

from __future__ import annotations

import argparse
import runpy
import sys
import time

from . import paths
from .config import all_subjects, analysis_subjects

PHASE9_ORDER = [
    "step_0_setup",
    "step_a_normality_power",
    "step_b_descriptive",
    "step_c_pertask",
    "step_d_bootstrap_clusters",
    "step_e_fisher",
    "step_f_exo_effect",
    "step_g_lmm",
    "step_a_extended_all_kpis",
    "step_g2_gap_closure",
    "step_h_selective_involvement",
    "step_f2_bayes_deltoid",
    # figures
    "fig_boxplots",
    "fig_dendrogram",
    "fig_defense_charts",
    "fig_lmm_forest",
    "fig_exo_selective",
]


def _run_module(modname: str) -> None:
    print(f"\n{'=' * 70}\n>>> running {modname}\n{'=' * 70}")
    runpy.run_module(modname, run_name="__main__")


def run_phase(phase: str, subjects: list[str]) -> None:
    t0 = time.time()
    if phase == "0":
        from .qc import run_phase0_all
        run_phase0_all(subjects)
    elif phase == "2":
        from .preprocess_emg import run_phase2_for_subject
        for s in subjects:
            run_phase2_for_subject(s)
    elif phase == "2b":
        from .preprocess_imu import run_phase2b_for_subject
        for s in subjects:
            run_phase2b_for_subject(s)
    elif phase == "3":
        from .normalize import run_phase3_all
        run_phase3_all(subjects)
    elif phase == "4":
        from .events import run_phase4_for_subject
        for s in subjects:
            run_phase4_for_subject(s)
    elif phase == "5":
        from .windows import run_phase5_for_subject
        for s in subjects:
            run_phase5_for_subject(s)
    elif phase == "6":
        from .features import run_phase6_v3, phase6_qc_report_v3, PHASE6_QC_CSV
        if PHASE6_QC_CSV.exists():
            PHASE6_QC_CSV.unlink()  # rebuild the QC log from scratch
        for s in subjects:
            df = run_phase6_v3(s)
            if df is not None:
                phase6_qc_report_v3(s, df, save_csv=True, verbose=False)
    elif phase == "7":
        from .kpis import run_phase7_all_subjects_v4, run_redundancy_check
        run_phase7_all_subjects_v4()
        run_redundancy_check()
    elif phase == "8":
        _run_module("alsexo.matrix")
    elif phase == "9":
        for step in PHASE9_ORDER:
            _run_module(f"alsexo.stats.{step}")
    elif phase.startswith("9:"):
        _run_module(f"alsexo.stats.{phase.split(':', 1)[1]}")
    elif phase == "10":
        _run_module("alsexo.supervised")
    else:
        raise SystemExit(f"Unknown phase '{phase}'")
    print(f"<<< phase {phase} done in {time.time() - t0:.1f} s")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="ALS EMG/IMU exoskeleton pipeline")
    ap.add_argument("phases", nargs="+", help="0 2 2b 3 4 5 6 7 8 9 9:<step> 10 | all")
    ap.add_argument("--subjects", nargs="*", default=None, help="subset of subjects (Phases 0-6)")
    args = ap.parse_args(argv)

    paths.ensure_dirs()
    subjects = args.subjects or all_subjects()
    phases = ["0", "2", "2b", "3", "4", "5", "6", "7", "8", "9", "10"] if args.phases == ["all"] else args.phases
    print(f"PROJECT_ROOT = {paths.PROJECT_ROOT}")
    print(f"subjects ({len(subjects)}): {subjects}")
    print(f"analysis set ({len(analysis_subjects())}): excludes {sorted(set(subjects) - set(analysis_subjects()))}")
    for ph in phases:
        run_phase(ph, subjects)


if __name__ == "__main__":
    main(sys.argv[1:])
