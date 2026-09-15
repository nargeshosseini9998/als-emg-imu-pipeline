# notebooks/

`legacy/` contains the original analysis notebooks used to produce the thesis
results (outputs stripped; absolute paths replaced by `<PROJECT_ROOT>`).

The maintained, runnable implementation is the `alsexo` package (`src/alsexo/`),
which reproduces the same analysis in a configurable, testable form. The notebooks
are kept for reference and traceability of the published results.

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
