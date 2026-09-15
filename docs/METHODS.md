# Methods — as implemented

**Data-Driven Functional Classification of ALS and Adaptive Assistive Control Strategies**
MSc thesis, Università degli Studi di Napoli Federico II / PRISMA Lab.

This document describes the pipeline **exactly as it is implemented in `src/alsexo/`**. It
replaces the earlier "Methods Justification Map" (Parts 1–5), which had been written against
a previous revision of the code. Wherever the two differed, the code was taken as the
reference. Every parameter quoted below is read from `configs/*.yaml` by the code; nothing is
hard-coded. Section 12 lists, explicitly, every place where a decision was made by hand.

Tag legend (as in the original map): 🟦 standard method (cite) · 🟨 standard method adapted
to this dataset (cite + data-driven reason) · 🟩 own design (defended by logic and internal
validation, no borrowed citation).

---

## 0. Dataset and cohort

* **Participants.** 16 ALS patients and 8 healthy controls were recorded. One ALS participant
  (`ALS_Subject_6`) has no CSV export of the functional tasks (only vendor `.shpf` session
  files) and never entered the pipeline. **23 subjects** (15 ALS, 8 healthy) were processed
  through Phases 0–7. One ALS subject (`ALS_Subject_15`) was then removed from all group-level
  analyses because of a confirmed electrode/device artifact on the wrist-extensor channel
  (±2–4 V swings, extensor RMS ≈ 0.96 V where every other channel is in the mV range;
  verified visually). **The analysis set is therefore n = 22 (14 ALS, 8 healthy).** A parallel
  "full set" (n = 23) file is kept for robustness checks (`phase_09_efa/partA_subject_matrix_full.parquet`).
* **Tasks.** Drinking, lifting, pick-and-place (high shelf) and pick-and-place (low shelf), each
  performed without (`noexo`) and with (`exo`) a passive upper-limb exoskeleton. Not every
  subject completed every task; the per-subject inventory is produced by Phase 0
  (`qc/00_IO_ALL.csv`).
* **Instrumentation.** Delsys Trigno sensors: surface EMG (nominal 1259.26 Hz) and a
  triaxial accelerometer + gyroscope per sensor (nominal 148.15 Hz). Up to eight muscles:
  descending trapezius, medial deltoid, biceps brachii, triceps brachii, extensor carpi
  radialis, flexor carpi radialis, first dorsal interosseous, abductor digiti minimi.
* **Muscle vocabulary.** Sensor labels varied across sessions (English, Italian, abbreviated).
  Two normalisation tables exist: `trigno_io.ITALIAN_TO_CANONICAL` (Phase 0, eight canonical
  short names) and `features._EN_CANON_RAW` (Phase 6 onwards: `Trapezius, Deltoid, Biceps,
  Triceps, Extensor, Flexor, IntDors, AbdV`). Phase 6 is where the second vocabulary is applied;
  all downstream feature names use it.

---

## 1. Phase 0 — parsing, and Phase 1 — signal quality control

### 1.1 Parser (🟩)
Trigno Discover CSV exports do not have a constant column layout: a sensor block may contain
EMG + IMU (14 columns), IMU only (12) or EMG only (2), and the metadata header length varies.
`trigno_io.read_trigno_csv` locates the header row by its `… Time Series` pattern, recovers
sensor names from the preceding `Name (serial)` line, and walks the columns left-to-right with
a header-driven state machine that recognises each block kind. If names are missing or
inconsistent and exactly eight blocks are found, the canonical eight-muscle montage is assumed;
otherwise placeholder names are used and the file is flagged. Layout, block kinds and every
anomaly are written to `qc/00_IO_ALL.csv`.

### 1.2 QC metrics (🟦), computed per trial × sensor (`qc.qc_one_signal`)
| Metric | Definition | Parameter |
|---|---|---|
| Effective sampling rate | 1 / mean(Δt) | — |
| Time gaps | count of Δt > `gap_factor` × median(Δt) | 2.0 |
| Missing data | fraction of non-finite samples | — |
| Dead channel | variance < `var_eps` | 1e-10 |
| Saturation ratio | fraction of samples inside runs of identical values ≥ `sat_run` samples | EMG 500 samples (≈0.40 s); IMU magnitudes 60 samples (≈0.40 s) |

IMU QC is run on the gyroscope and accelerometer **magnitudes**.

### 1.3 Trial admission rule (🟨) — `qc.compute_trial_status`
A trial **fails** if any of its sensors shows: a parser mapping problem, any time gap in EMG or
IMU, or — when EMG is present — a dead channel or a saturation ratio > 1 %.
**Result on this dataset: one trial failed** (`Healthy_Subject_5 / lifting_exo`, EMG
saturation). It was not discarded but *repaired* (Phase 2, §2.4) and re-admitted; this is the
only trial for which the repair path was used. Structural anomalies (7-sensor sessions, mixed
layouts, EMG-only sensors) are flagged, not rejected.

---

## 2. Phase 2 — EMG preprocessing (`preprocess_emg`)

Applied to every admitted trial and EMG channel, using the sampling rate estimated in Phase 1
(fallback 1259.259 Hz if missing). Non-finite samples are set to 0 before filtering.

| Step | Implementation | Tag |
|---|---|---|
| 2.1 Band-pass | 20–450 Hz Butterworth, order 4, zero-phase (`filtfilt`) | 🟦 De Luca 2010; SENIAM |
| 2.2 Power-line removal | **Conditional**: Welch PSD (2-s segments); if mean power in 45–55 Hz ≥ 5 × mean power in 60–80 Hz, apply an IIR notch at 50 Hz, Q = 30 (zero-phase). Otherwise no notch. Per-channel decisions are logged. | 🟩 (rationale: De Luca 2002 cautions against routine notching) |
| 2.3 Linear envelope | full-wave rectification → 6 Hz Butterworth, order 4, zero-phase | 🟦 Winter |
| 2.4 Saturation repair | for trials listed in `phase2.repair_trials` only: runs of identical raw samples ≥ 500 samples are masked, the mask is dilated by ±0.5 s, and both `emg_bp` and `env` are set to NaN inside the mask | 🟩 |

Output: one long-format parquet per subject with `t_emg, emg_bp (mV), env`.

---

## 2b. Phase 2b — IMU preprocessing (`preprocess_imu`)

Sampling rate from Phase 1 (fallback 148.148 Hz).

| Quantity | Definition | Tag |
|---|---|---|
| `gyro_mag` | ‖ω‖ (deg/s) | 🟦 |
| `gyro_gate` | 10 Hz Butterworth (order 4, zero-phase) of `gyro_mag` | 🟨 |
| `acc_mag` | ‖a‖ (g) | 🟦 |
| `acc_dyn` | `acc_mag` − LP(`acc_mag`, 0.5 Hz, order 2). **The baseline (gravity) is removed from the magnitude, not per axis.** | 🟨 (van Hees 2013 for the frequency separation; applied to the magnitude here) |
| `acc_gate` | 10 Hz Butterworth (order 4) of |`acc_dyn`| | 🟨 |

No orientation estimation (quaternion / Kalman) is performed: the analysis needs movement
intensity and timing, not absolute limb orientation (🟩). The raw axes are kept for the jerk
computation in Phase 6.

---

## 3. Phase 3 — within-subject envelope normalisation (`normalize`)

For each (subject, sensor): the envelope of **all admitted trials** (both conditions, all tasks)
is pooled; tiny negative values caused by zero-phase filtering are clipped to 0; a random
sample of 2 % of the samples is drawn **within each sensor** (minimum 5 000, maximum
200 000, `random_state = 0`); the **95th percentile** of that sample is the scale; and
`env_norm = env / scale`. Scales < 1e-8 or undefined are flagged and the sensor left as NaN
(none occurred: `bad_scales_reports/` is empty).

* Peak-dynamic normalisation instead of %MVC (🟦 Halaki & Ginn 2012; Burden 2010) — MVC was not
  reliably obtainable in patients; P95 instead of the maximum for robustness to single spikes (🟨).
* **Scope.** `env_norm` is used only for within-subject purposes: the EMG-fallback gate in
  Phase 4, the co-activation index in Phase 6 and the peak/IEMG/duty KPIs in Phase 7. All
  between-group amplitude comparisons use the absolute band-passed EMG (Phase 6 RMS), because
  within-subject normalisation removes exactly the between-subject amplitude information such a
  comparison needs (🟩, and the key methodological point of the thesis).

---

## 4. Phase 4 — movement-episode detection (`events`, `configs/phase4_episodes.yaml`)

### 4.1 What is shared by all variants
* **Gate signal.** For trials with IMU data: the long-format table is collapsed to one series by
  the median over sensors within 1 ms bins, and g(t) = max(gyro_gate, α · acc_gate), α = 1
  (α = 1.8 in the weak variant) (🟩 multimodal gate). For the few trials without IMU
  (`ALS_Subject_3`: `lifting_noexo`, `pick&place_high_noexo`) the gate is the median over
  sensors of `env_norm` in 1 ms bins ("EMG fallback").
* **Double-threshold hysteresis** with robust thresholds θ = median + k·MAD (Hampel identifier,
  🟦 Bonato 1998; Leys 2013): an episode opens when g ≥ θ_on and closes when g ≤ θ_off.
* **Post-processing**: ±pad (0.15 s), merging of near-by intervals, minimum duration, peak
  criterion, then a final validity filter on duration and peak (🟦 Merlo 2003 for the
  merge/reject logic).
* **Task-specific settings** (minimum duration / merge gap / minimum peak): drinking 1.2 s /
  0.30 s / 60; lifting 1.8 s / 0.35 s / 100; pick&place 2.2 s / 1.20 s / 75 (🟨). Note that in
  the standard variant the merge gap is immediately raised to `max(merge_gap, 2.0 s)` and the
  secondary gap to `max(·, 4.0 s)` for every task, so the effective first-stage merge gap is
  **2.0 s for all tasks**.

### 4.2 The five detector variants actually used
| Variant | Subjects | Distinctive rules |
|---|---|---|
| `standard` | Healthy 1–8; ALS 1, 2, 5, 14, 15, 16 | k_on 5.0, k_off 3.5; if θ_on > P95 of the gate, k_on ← max(1, 0.7·k_on); three merge stages (task gap → secondary 4.0 s → fixed 1.2 s); peak ≥ median + 2.5·MAD; validity: duration ≥ **3.5 s** and peak ≥ **50**; from `Healthy_Subject_6` onward a rule "keep at most the 4 longest episodes per trial" was added (Healthy 6–8, ALS 1, 2, 5, 15, 16). |
| `standard_v1` | ALS 3 | as `standard` but without the 2 s / 4 s merge floor and without the peak validity filter (earlier revision, separate notebook) |
| `standard_als` | ALS 8, 9, 10 | task merge gaps 0.50/1.00/1.50 s raised to ≥ 1.2 s (secondary ≥ 3.5 s); `noexo` trials: merge gap × 1.6, secondary ≥ 4.0 s, min duration × 0.75; threshold reduction when θ_on > 0.85·P95 or P95 < 60 (k_on ← max(1.5, 0.55·k_on), k_off ← max(0.8, 0.6·k_off)); gate × 1.5 if its variance < 1e-4; validity 2.0 s / 40 |
| `tremor` | ALS 13, 12, 11 | baseline statistics from the lowest 15 % of the gate; **inverted hysteresis** k_off (6.2) > k_on (4.8) so bursts end quickly at dips; θ_off × 1.3; θ_on capped at the gate median; raw intervals < 0.3 s ignored; single merge stage; absolute peak filter 28; gate additionally low-passed at 7 Hz; per-trial k_on / k_off / merge gap set individually (§12) |
| `weak` | ALS 7, 4 | k_on 2.8, k_off 1.5; gate median-filtered (kernel 7) and accelerometer boost α = 1.8; min duration × 0.65, merge gap × 2.2, secondary gap 5.5 s; peak ≥ median + 1.4·MAD; threshold reduction when θ_on > 1.1·P95 (k_on ← max(1.2, 0.65·k_on)); two merge stages; validity 2.0 s / 35; TKEO conditioning only on EMG-fallback gates (none occurred for these two subjects) |

EMG-fallback trials (standard family) use k_on 2.0, k_off 1.0, peak multiplier 1.2, merge gap ×
2.0, secondary gap 3.5 s, and minimum duration × 0.36 (the 0.6 factor is applied twice in the
original code; reproduced as such).

### 4.3 How the variant was chosen — honest statement (🟩)
**The variant and all per-subject / per-trial overrides were chosen by visual inspection** of
the gate signal with the detected episodes overlaid, not by an automatic signal-quality rule.
The `weak` variant does contain a P95-of-gate threshold (50), but its trigger condition is
`is_ALS or P95 < 50`, so it fired for every trial of the two subjects it was applied to
regardless of P95. The complete decision record — variant per subject, parameter per trial, and
the reason — is `configs/phase4_episodes.yaml`; the five hand-removals of individual episodes
are in `configs/manual_curation.yaml` (§12). Visual verification of automatic onset detection
is common practice in EMG burst analysis; what this pipeline adds is that every such decision
is machine-readable and reproducible.

### 4.4 Outputs
`phase_04_events/<subject>__episodes.parquet` (`t_on, t_off, duration_s, peak_gate, task_type,
data_source, variant, episode_id`) and a per-trial log with all resolved parameters and
threshold values.

---

## 5. Phase 5 — windowing (`windows`)
Inside each episode: 200 ms windows, 50 % overlap (100 ms step), strictly within
[t_on, t_off] (🟦 Smith 2011; Englehart & Hudgins 2003). A window is `emg_ok` (resp. `imu_ok`)
if it contains ≥ 60 % of the expected number of samples for that modality and the largest gap
between consecutive samples is ≤ 50 ms (🟩 window-level QC). Only window boundaries and flags
are stored.

---

## 6. Phase 6 — feature extraction (`features`)
Windows with `emg_ok OR imu_ok` enter; a modality is computed only if it has ≥ 32 EMG / ≥ 8 IMU
samples in the window.

**EMG, per muscle, on the absolute band-passed signal `emg_bp` (mV)** — 🟦 Hudgins 1993;
Phinyomark 2012:
MAV, RMS, WL, WAMP, ZC, MNF, MDF, AR(1–4).
* WAMP threshold = max(50 µV, 0.01 · std(emg_bp)); ZC threshold = 0.01 · std(emg_bp) (🟨 adaptive
  thresholds so the two threshold-based features are comparable across muscles/subjects).
* MNF / MDF from a Welch PSD with `nperseg = min(256, n)`; with ≈ 252 samples per window this is
  effectively a single-segment periodogram.
* AR(4) by Yule–Walker on the mean-removed signal.

**IMU, per sensor and pooled ("IMU_all_")**: mean / SD / peak of `gyro_mag` and `acc_mag`,
and the mean norm of the 3-D jerk (finite difference of the raw acceleration axes divided by
the median Δt) (🟦 Flash & Hogan 1985; Hogan & Sternad 2009).

**Co-activation** `EMG_CoAct_Bic_Tri` = ∫min(ẽ_B, ẽ_T) / ∫max(ẽ_B, ẽ_T) on the **normalised**
envelopes, time-aligned by interpolation onto a regular grid (🟦 Falconer & Winter 1985).

A per-subject QC row (coverage ≥ 98 % OK / ≥ 90 % WARN; NaN ≤ 10 % OK / ≤ 25 % WARN; sign checks;
CoAct ∈ [0, 1]) is appended to `phase6_qc_report.csv`.

---

## 7. Phase 7 (v4) — KPIs (`kpis`)

**Trial level** (one row per trial; `trial_kpis.parquet`):
* From the normalised envelope over the trial's episodes: peak and IEMG (trapezoidal integral)
  per muscle, global sums/maxima, duty cycle (time-weighted fraction of samples with
  `env_norm` > 0.05), co-activation `∫min/∫max` and CCI `∫min/∫(e1+e2)` for biceps–triceps
  (🟦 Falconer & Winter; Rudolph 2000).
* From Phase-6 window features: duration-weighted mean and peak angular velocity, mean jerk,
  window-mean/median CoAct, **and the absolute-RMS effort proxies** `kpi_abs_rms_mean` (all
  muscles) and `kpi_abs_rms_distal` (extensor + flexor).
* The two composite indices of the earlier version (efficiency = peak gyro / IEMG; cost =
  IEMG × jerk) were **removed** in v4: efficiency correlated ρ ≈ 0.97 with peak gyro, both mixed
  absolute and normalised units, and a product of two quantities is not interpretable. They
  are mentioned here so their absence is not mistaken for an omission.

**Subject level** (`subject_kpis.parquet`): median / mean / SD over trials of every KPI, and
per-task medians (`__taskmed__`).

**Neuromuscular dissociation index** (🟩): NMD = z(effort) − z(motion), effort =
`kpi_abs_rms_distal__median`, motion = `kpi_peak_gyro_mag__median`, z-scored across subjects in a
final pass. High positive NMD = high effort relative to produced motion. It is a descriptive
index and is *not* used as a model feature.

**Redundancy check**: pairwise Spearman correlation of all overall-median KPIs; pairs with
|ρ| ≥ 0.90 are listed (`kpi_redundant_pairs.csv`).

---

## 8. Phase 8 (v2) — subject × feature matrix (`matrix`)
1. Candidates: overall-median KPIs (267 columns) → coverage ≥ 80 % of subjects and SD > 1e-6 →
   36 candidates.
2. Clinically prioritised selection to 20 features in three tiers (tier 1: the two absolute-RMS
   effort KPIs, global IEMG, biceps–triceps co-activation, peak angular velocity, jerk, peak
   activation of biceps/triceps/deltoid; tier 2: per-muscle IEMG and duty cycles, mean angular
   velocity, trial duration, peak activation of trapezius/extensor/flexor; tier 3: deltoid duty
   cycle). This selection does not use the group labels (🟨 theory-guided selection at n ≈ 22).
3. Median imputation; `StandardScaler`; labels ALS = 1 / HC = 0.
4. `ALS_Subject_15` is flagged and excluded (§0); the subjects-to-features ratio (22/20) is
   recorded as inadequate for any multivariate model — which is why Phase 9 works with four
   absolute-RMS features and Phase 10 with the same four.

---

## 9. Phase 9 — statistics (`stats/`)
All steps use the analysis set (n = 22) unless stated, α = 0.05, Benjamini–Hochberg FDR where
several tests are run, seeds fixed (`np.random.seed(42)`).

* **9.0 Subject matrix.** Per subject and muscle: **median over all windows** (both
  conditions) of the absolute RMS → `EMG_Extensor_RMS, EMG_Flexor_RMS, EMG_Biceps_RMS,
  EMG_Triceps_RMS` (`partA_subject_matrix.parquet`; full-set copy with n = 23).
* **9.A Normality and power.** Shapiro–Wilk per group and feature (🟦). Post-hoc power of the
  Mann–Whitney U test from a normal approximation using the observed rank-biserial effect size.
* **9.B Descriptives.** Median, IQR, bootstrap 95 % CI of the median (2 000 resamples), outliers.
* **9.C Group comparison per task.** Mann–Whitney U (ALS vs HC) per muscle and task,
  rank-biserial r, BH-FDR within task (🟦 Mann & Whitney 1947; Benjamini & Hochberg 1995);
  Friedman test across tasks within ALS, with post-hoc pairwise Wilcoxon (🟦 Friedman 1937).
* **9.A-extended.** The same test battery on *all* between-subject-comparable KPIs (absolute RMS
  per muscle, peak/mean angular velocity, jerk, one co-activation KPI, trial duration; duty
  cycles as exploratory). Within-subject normalised KPIs are explicitly excluded from any
  between-group test.
* **9.D Clustering and stability.** Ward hierarchical clustering on the four z-scored RMS
  features (🟦 Ward 1963), cut at k = 2 and k = 3; 1 000 bootstrap resamples; per-cluster
  Jaccard stability (🟦 Hennig 2007) and a subject × subject co-occurrence matrix.
  Current outputs (n = 22): k = 3 sizes 16 / 5 / 1; Jaccard 0.559 / 0.471 / 0.640
  (moderate / unstable / moderate); within-cluster co-occurrence 0.80 and 0.75 versus ≈ 0.10
  between. **No silhouette or cophenetic statistics are computed in the pipeline.**
* **9.E External validation.** k = 2: Fisher's exact test cluster × group, **p = 0.051**;
  k = 3: permutation test (10 000 permutations) **p = 0.113**, Cramér's V = 0.46. The
  clusters are therefore reported as exploratory, hypothesis-generating structure.
* **9.F Exoskeleton effect (v2).** For every task, group and **all seven recorded muscles plus
  trial duration**: paired Wilcoxon signed-rank (exo vs no-exo, same subject), rank-biserial r,
  and two FDR schemes reported side by side — within task (primary) and global (conservative)
  (🟦 Wilcoxon 1945). An earlier version that tested only four muscles is superseded.
* **9.F-Bayes.** For the ALS deltoid in the drinking task (frequentist p = 0.054, r = 0.67): a
  Bayesian estimate on paired log-differences d_i = log(EMG_exo) − log(EMG_noexo),
  d_i ~ Normal(μ, σ), sceptical priors μ ~ Normal(0, 0.5), σ ~ Half-Normal(0.5), posterior by
  grid/Monte-Carlo; reports P(μ < 0) and a 95 % credible interval for the multiplicative change.
* **9.G Linear mixed-effects model.** `log(EMG + 1e-8) ~ C(group) + C(task) + C(condition)` with a
  random intercept per subject, REML (statsmodels `MixedLM`), 124 trial-level observations from
  22 subjects; ICC = σ²_subject / (σ²_subject + σ²_resid) (extensor 0.775); likelihood-ratio test
  of the full versus a reduced model refitted with ML (🟦 Laird & Ware 1982).
* **9.G2 Gap closure.** Reference = healthy no-exo; for each comparable KPI, does the ALS+exo
  median fall inside the healthy no-exo IQR (band test), and what percentage of the ALS-vs-HC
  gap does the exoskeleton remove?
* **9.H Selective muscle involvement.** For the biceps/triceps and extensor/flexor pairs:
  log-ratio in ALS versus healthy (Mann–Whitney), cross-check of per-muscle effect sizes
  (selectivity index = ratio of fold changes: extensor/flexor 1.15, biceps/triceps 0.88), and
  per-subject prevalence of the pattern.
* **Figures**: box plots, dendrogram, chapter charts, LMM forest plot, exoskeleton / selectivity
  panels (`fig_*` modules).

---

## 10. Phase 10 — supervised classification (`supervised`)
* Input: from Phase-6 features, **no-exo trials only**, median per subject of the four absolute
  RMS features; `ALS_Subject_15` excluded; n = 22 (14 ALS / 8 HC).
* Models (🟦): logistic regression (L2, C = 1, balanced class weights); SVM-RBF (C = 1,
  γ = scale, balanced); random forest (500 trees, max depth 2, min leaf 3, balanced) —
  deliberately low-capacity for n = 22 (🟨).
* Validation: leave-one-subject-out (22 folds); `StandardScaler` **fitted inside each fold** on
  the training subjects only; metrics: balanced accuracy (primary, 🟦 Brodersen 2010), AUC,
  sensitivity, specificity, F1. Results: balanced accuracy 0.76 / 0.76 / 0.77, AUC 0.82 / 0.67
  / 0.83 (LogReg / SVM / RF).
* Cross-model consistency (🟩): 15 subjects classified correctly by all three models, 4 wrong by
  all three (`ALS_1, ALS_5, ALS_16, Healthy_7`), 3 split.
* Interpretation: SHAP `TreeExplainer` on a random forest **refitted on all 22 subjects** (🟦
  Lundberg & Lee 2017), mean |SHAP| ranking extensor > flexor > triceps > biceps, all pushing
  higher RMS → ALS; Gini importances 0.44 / 0.30 / 0.14 / 0.12 agree.

---

## 11. References (standard methods only)
De Luca CJ, Gilmore LD, Kuznetsov M, Roy SH (2010) *J Biomech* 43:1573–1579 · De Luca CJ
(2002) *Surface electromyography: detection and recording*, DelSys · Hermens HJ et al. (2000)
*J Electromyogr Kinesiol* 10:361–374 (SENIAM) · Winter DA, *Biomechanics and Motor Control of
Human Movement*, Wiley · van Hees VT et al. (2013) *PLoS ONE* 8:e61691 · Halaki M, Ginn K (2012)
in *Computational Intelligence in Electromyography Analysis*, InTech · Burden A (2010)
*J Electromyogr Kinesiol* 20:1023–1035 · Bonato P, D'Alessio T, Knaflitz M (1998) *IEEE TBME*
45:287–299 · Merlo A, Farina D, Merletti R (2003) *IEEE TBME* 50:316–323 · Leys C et al. (2013)
*J Exp Soc Psychol* 49:764–766 · Solnik S et al. (2010) *Eur J Appl Physiol* 110:489–498 (TKEO)
· Smith LH et al. (2011) *IEEE TNSRE* 19:186–192 · Englehart K, Hudgins B (2003) *IEEE TBME*
50:848–854 · Hudgins B, Parker P, Scott RN (1993) *IEEE TBME* 40:82–94 · Phinyomark A et al.
(2012) *Expert Syst Appl* 39:7420–7431 · Welch PD (1967) *IEEE Trans Audio Electroacoust*
15:70–73 · Falconer K, Winter DA (1985) *Electromyogr Clin Neurophysiol* 25:135–149 · Rudolph
KS et al. (2000) *Knee Surg Sports Traumatol Arthrosc* 8:262–269 · Flash T, Hogan N (1985)
*J Neurosci* 5:1688–1703 · Hogan N, Sternad D (2009) *J Mot Behav* 41:529–534 · Shapiro SS,
Wilk MB (1965) *Biometrika* 52:591–611 · Mann HB, Whitney DR (1947) *Ann Math Stat* 18:50–60 ·
Benjamini Y, Hochberg Y (1995) *J R Stat Soc B* 57:289–300 · Efron B (1979) *Ann Stat* 7:1–26
· Ward JH (1963) *J Am Stat Assoc* 58:236–244 · Hennig C (2007) *Comput Stat Data Anal*
52:258–271 · Wilcoxon F (1945) *Biometrics Bull* 1:80–83 · Friedman M (1937) *J Am Stat Assoc*
32:675–701 · Laird NM, Ware JH (1982) *Biometrics* 38:963–974 · Breiman L (2001) *Mach Learn*
45:5–32 · Cortes C, Vapnik V (1995) *Mach Learn* 20:273–297 · Brodersen KH et al. (2010) *ICPR*
3121–3124 · Lundberg SM, Lee S-I (2017) *NeurIPS* 30.

Removed from the earlier reference list because the corresponding analyses are not in the
pipeline: Kaiser 1960/1974 (KMO), Bartlett 1950, Horn 1965, Velicer 1976, Zwick & Velicer 1986,
Sokal & Rohlf 1962 (cophenetic), Rousseeuw 1987 (silhouette).

---

## 12. Disclosures — every hand-made decision, and deviations from earlier drafts

1. **Cohort.** `ALS_Subject_6` never ingested (no export); `ALS_Subject_15` excluded from
   Phase 8 onward (artifact). n = 23 processed, n = 22 analysed.
2. **QC.** One trial failed QC (`Healthy_Subject_5 / lifting_exo`) and was repaired rather than
   discarded. Earlier drafts stated "0 trials failed"; that is incorrect.
3. **Phase 4 variant selection** was manual (visual inspection), not driven by a P95 rule
   (§4.3). Eleven ALS subjects use a non-default variant; three of them have per-trial
   parameters (`ALS_11, 12, 13`), and `ALS_14` has per-trial parameters inside the standard
   variant. Because the exo/no-exo test was implemented by substring matching
   (`"exo" in trial_id`), several rules written for exo trials also applied to the
   corresponding no-exo trials (`ALS_11, 12, 14`); the recorded parameters reflect what actually
   ran.
4. **Manual episode removals** (`configs/manual_curation.yaml`): `Healthy_8 / pick&place_low_exo`
   (longest episode), `ALS_2 / lifting_exo` (3rd episode), `ALS_5 / pick&place_high_exo` (all),
   `ALS_10 / pick&place_high_exo` (all) and `lifting_exo` (last), `ALS_11 / lifting_exo` (all),
   `ALS_14 / pick&place_high_exo` and `pick&place_high_noexo` (all). In addition `ALS_12`
   lifting trials passed through a positional pruning rule (§4.2, config `prune_rule`).
5. **Final validity filter** (duration ≥ 3.5 s, peak ≥ 50, relaxed for `standard_als` and
   `weak`) is the strongest single filter of Phase 4 and was absent from earlier drafts.
6. **Phase 7.** Efficiency and cost composites removed; absolute-RMS KPIs, NMD and the
   redundancy check added (v4).
7. **Phase 9.** No exploratory factor analysis (KMO / Bartlett / parallel analysis / MAP), no
   silhouette and no cophenetic correlation are part of the pipeline; earlier drafts describing
   those results cannot be regenerated and were removed. Cluster-stability and Fisher numbers
   in earlier drafts (n = 21: Jaccard 0.552/0.457/0.621, co-occurrence 0.90/0.68, Fisher
   p = 0.022, Cramér's V 0.569) are superseded by the n = 22 outputs in §9 (Fisher p = 0.051,
   permutation p = 0.113, V = 0.46).
8. **Phase 10.** The "44 / 30 / 14 / 12 %" figures are Gini importances; SHAP shares are
   ≈ 45 / 30 / 13 / 12 %. SHAP is computed on a model refitted on all subjects, outside the
   LOSO loop.
9. **Reproduction status.** The package was written to reproduce the notebooks line by line
   (quirks marked `# QUIRK`), but it has to be validated against the archived
   `data/processed/` outputs by re-running it on the original data; until then the notebooks in
   `notebooks/legacy/` remain the authoritative record of what was run.
