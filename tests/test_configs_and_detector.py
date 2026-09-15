"""Smoke tests that need no data: config integrity + detector behaviour on a synthetic gate."""
try:
    import numpy as np  # type: ignore[import]
except Exception as e:  # pragma: no cover - environment missing numpy
    raise ImportError("numpy is required to run these tests") from e
try:
    import pandas as pd  # type: ignore[import]
except Exception as e:  # pragma: no cover - environment missing pandas
    raise ImportError("pandas is required to run these tests") from e

from alsexo.config import phase4_cfg, pipeline_cfg, subjects_cfg, curation_cfg, all_subjects, analysis_subjects
from alsexo.events import detect_episodes_mad, detect_episodes_tremor, resolve_trial_params


def test_every_subject_has_a_phase4_variant():
    cfg = phase4_cfg()
    for s in all_subjects():
        assert s in cfg["subjects"], s
        assert cfg["subjects"][s]["variant"] in cfg["variants"]


def test_analysis_set_excludes_documented_subject():
    assert "ALS_Subject_15" in all_subjects()
    assert "ALS_Subject_15" not in analysis_subjects()
    assert len(analysis_subjects()) == 22


def test_curation_subjects_exist():
    for s in curation_cfg()["episode_removals"]:
        assert s in all_subjects(), s


def test_resolve_params_tremor_overrides():
    cfg = phase4_cfg()
    p = resolve_trial_params("ALS_Subject_12", "pick&place_high_exo", True, cfg)
    assert p["variant"] == "tremor" and p["k_on"] == 5.2 and p["k_off"] == 9.5 and p["merge_gap_s"] == 0.15


def test_resolve_params_standard_merge_floor():
    cfg = phase4_cfg()
    p = resolve_trial_params("Healthy_Subject_1", "drinking_exo", True, cfg)
    assert p["merge_gap_s"] == 2.0 and p["secondary_merge_gap"] == 4.0 and p.get("max_episodes_per_trial") is None
    p6 = resolve_trial_params("Healthy_Subject_6", "drinking_exo", True, cfg)
    assert p6["max_episodes_per_trial"] == 4


def test_detector_finds_two_bursts():
    fs = 148.0
    t = np.arange(0, 40, 1 / fs)
    gate = 5 + np.random.default_rng(0).normal(0, 0.5, t.size)
    gate[(t > 5) & (t < 12)] += 150
    gate[(t > 20) & (t < 28)] += 150
    ep, log = detect_episodes_mad(t, gate, merge_gap_s=2.0, secondary_merge_gap=4.0)
    assert len(ep) == 2 and abs(ep["t_on"].iloc[0] - 5) < 0.3 and abs(ep["t_off"].iloc[1] - 28) < 0.3
    ep2, _ = detect_episodes_tremor(t, gate)
    assert len(ep2) == 2


def test_pipeline_yaml_core_values():
    c = pipeline_cfg()
    assert c["phase2"]["bp_low_hz"] == 20 and c["phase2"]["bp_high_hz"] == 450
    assert c["phase5"]["win_len_s"] == 0.2 and c["phase5"]["overlap"] == 0.5
    assert subjects_cfg()["excluded_from_analysis"]
