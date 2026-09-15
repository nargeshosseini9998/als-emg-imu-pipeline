# notebooks/

`legacy/` contains the original analysis notebooks exactly as used to produce the thesis
results, with three modifications only:

1. all cell outputs were stripped (they contained ~35 MB of tables/figures);
2. Persian comments and messages were translated to English;
3. the absolute project path was replaced by `<PROJECT_ROOT>` (edit it, or run the
   `alsexo` package instead — see the main README).

They are kept for provenance: every rule in `configs/phase4_episodes.yaml` and
`configs/manual_curation.yaml` cites the notebook and cell it came from. The
maintained, runnable implementation is the `alsexo` package (`src/alsexo/`).

Mapping notebook -> package module

| notebook | module |
|---|---|
| 00_qc_check / 00_qc_healthy | `alsexo.qc` |
| 01_preprocessing | `alsexo.preprocess_emg` |
| 02_preprocessing_imu | `alsexo.preprocess_imu` |
| 03_normalization | `alsexo.normalize` |
| 04_events, 04_events_als3, 04_exceptions | `alsexo.events` (+ `configs/phase4_episodes.yaml`) |
| 05_segmentation | `alsexo.windows` |
| 06_featurs | `alsexo.features` |
| 07_v4_KPIs | `alsexo.kpis` |
| 08_v2_matrix | `alsexo.matrix` |
| 09_complete | `alsexo.stats.*` (one module per cell) |
| 10_supervised | `alsexo.supervised` |
