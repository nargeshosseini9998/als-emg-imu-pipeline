# alsexo/config.py
# YAML config loading. All numeric parameters of the pipeline live in
# configs/*.yaml so that the code contains no hidden constants.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_ROOT


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml (cached)."""
    path = CONFIG_ROOT / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pipeline_cfg() -> dict[str, Any]:
    return load_yaml("pipeline")


def subjects_cfg() -> dict[str, Any]:
    return load_yaml("subjects")


def phase4_cfg() -> dict[str, Any]:
    return load_yaml("phase4_episodes")


def curation_cfg() -> dict[str, Any]:
    return load_yaml("manual_curation")


def all_subjects() -> list[str]:
    """Subjects that enter the pipeline (Phase 0 onwards)."""
    return list(subjects_cfg()["subjects"])


def analysis_subjects() -> list[str]:
    """Subjects retained for group-level statistics (Phase 8 onwards)."""
    excl = set(subjects_cfg().get("excluded_from_analysis", {}).keys())
    return [s for s in all_subjects() if s not in excl]
