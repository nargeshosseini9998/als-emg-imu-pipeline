# ALS EMG/IMU exoskeleton pipeline

Reproducible analysis pipeline behind the MSc thesis
**"Data-Driven Functional Classification of ALS and Adaptive Assistive Control Strategies"**
(Università degli Studi di Napoli Federico II — PRISMA Lab).

The pipeline takes surface-EMG + IMU recordings (Delsys Trigno Discover CSV exports) of
ALS patients and healthy controls performing functional upper-limb tasks with and without a
passive exoskeleton, and produces: quality-controlled signals, movement episodes, window-level
features, clinical KPIs, group statistics, an exoskeleton-effect analysis, a linear mixed model,
clustering, and a leave-one-subject-out classifier with SHAP attribution.

> **Data are not included.** The recordings are clinical data and stay outside the repository
> (see `data/README.md` and `.gitignore`). Everything else — code, parameters, subject-level
> decisions — is here.

---

## 1. Installation

```bash
git clone <this repo>
cd als-emg-exo-pipeline
python -m venv .venv && source .venv/bin/activate      # optional
pip install -e .                                       # installs the `alsexo` package + deps
pytest tests/                                          # no data needed
```

Python ≥ 3.10. The thesis results were produced with numpy 2.4.4 / pandas 3.0.2 / scipy 1.17.1
on macOS (see `requirements.txt`).

## 2. Data layout

Put the raw exports under `data/raw/`:

```
data/raw/<Subject>/EMG&IMUTest/<task>_<exo|noexo>.csv
```

`<Subject>` = `ALS_Subject_N` or `Healthy_Subject_N` (list in `configs/subjects.yaml`);
`<task>` = `drinking`, `lifting`, `pick&place_high`, `pick&place_low`.
By default the project root is the repository root; set `ALSEXO_PROJECT_ROOT=/path` to use
another location.

## 3. Running

```bash
python -m alsexo.cli all              # Phases 0 -> 10, all subjects
python -m alsexo.cli 0                # parsing + QC only
python -m alsexo.cli 4 5 6            # a subset of phases
python -m alsexo.cli 4 --subjects ALS_Subject_12 ALS_Subject_13
python -m alsexo.cli 9:step_g_lmm     # one Phase-9 step
```

| Phase | Module | Output folder (`data/processed/`) |
|---|---|---|
| 0 / 1 | `alsexo.qc` | `qc/00_QC_ALL_SUBJECTS.csv`, `qc/00_IO_ALL.csv` |
| 2 | `alsexo.preprocess_emg` | `phase_02_preprocess_emg/` |
| 2b | `alsexo.preprocess_imu` | `phase_02b_preprocess_imu/` |
| 3 | `alsexo.normalize` | `phase_03_normalization/` |
| 4 | `alsexo.events` | `phase_04_events/` |
| 5 | `alsexo.windows` | `phase_05_windows/` |
| 6 | `alsexo.features` | `phase_06_features/` |
| 7 | `alsexo.kpis` | `phase_07_kpis/` |
| 8 | `alsexo.matrix` | `phase_08_matrix/` |
| 9 | `alsexo.stats.*` | `phase_09_efa/`, `phase_09_stats/step_*/`, `figures/` |
| 10 | `alsexo.supervised` | `phase_10_supervised/` |

## 4. Where the decisions live

| File | Contents |
|---|---|
| `configs/pipeline.yaml` | every numeric parameter of Phases 0–3, 5–8, 10 |
| `configs/phase4_episodes.yaml` | episode-detector **variant per subject** and **per-trial parameter overrides** |
| `configs/manual_curation.yaml` | episodes removed by hand after visual inspection, with reasons |
| `configs/subjects.yaml` | subject list, one subject never ingested (ALS_6), one excluded from analysis (ALS_15) |
| `docs/METHODS.md` | the full methods description, written from the code |

**A note on Phase 4.** Episode detection was not a single automatic algorithm. Three detector
families were used, and for eleven ALS subjects the family and parameters were chosen by
visual inspection of the gate signal; five trials had episodes removed by hand. All of this is
disclosed and recorded in the two YAML files above so that the exact thesis episode set can be
regenerated. See `docs/METHODS.md` §4 and §12.

## 5. Repository layout

```
configs/            YAML parameters and per-subject decisions
docs/METHODS.md     methods, rewritten from the code (single source of truth)
notebooks/legacy/   original notebooks (outputs stripped, comments translated) — provenance only
scripts/            gen_phase4_config.py (rebuilds phase4_episodes.yaml from the original rules)
src/alsexo/         the package (one module per phase; Phase 9 = one module per step)
tests/              data-free smoke tests
data/               NOT versioned; raw and processed data live here at run time
```

## 6. Reproducing the thesis numbers

Run `python -m alsexo.cli all` on the original data. The reference values are the CSV files in
`data/processed/phase_09_stats/` and `phase_10_supervised/` produced by the notebooks in
`notebooks/legacy/` (the package reproduces their logic exactly, quirks included — grep for
`# QUIRK` in `src/alsexo/events.py`). Note that a few figures quoted in earlier drafts of the
thesis came from an older run with n = 21; the current outputs (n = 22) are the ones the code
regenerates (see `docs/METHODS.md` §9.4).

## 7. Citation

If you use this code, please cite the thesis (see `CITATION.cff` once the final version is deposited).

## License

MIT for the code. The data are not covered by this license and are not distributed.
