# alsexo/stats/step_g_lmm.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 16); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.step_g_lmm`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — Step 9.G: Linear Mixed Effects Model
#
# Purpose:
#   Test ALS vs Healthy EMG differences while accounting for:
#   - Within-subject repeated measurements (multiple trials per subject)
#   - Task effects (drinking, lifting, pick&place)
#   - EXO condition effects
#   - Interaction effects (Group×Task, Group×Condition)
#
# Why LMM over Mann-Whitney on medians?
#   - Uses ALL trial-level data (not collapsed to subject median)
#   - Properly models within-subject correlation
#   - Simultaneously estimates multiple effects + interactions
#   - More statistical power with small n
#   - Handles unbalanced data (not all subjects have all tasks)
#
# Model: EMG ~ Group + Task + Condition + Group×Task + Group×Condition
#                  + (1|Subject)
#
# Implemented using statsmodels MixedLM.
# EMG values log-transformed (common for right-skewed EMG amplitude data).
#
# Output:
#   phase_09_stats/step_G_lmm/lmm_results.csv
#   phase_09_stats/step_G_lmm/lmm_summary.txt
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

import statsmodels.formula.api as smf  # listed in requirements.txt
from statsmodels.stats.multitest import multipletests
HAS_STATSMODELS = True


PHASE9_STATS = PROCESSED_ROOT / "phase_09_stats" / "step_G_lmm"
PHASE9_STATS.mkdir(parents=True, exist_ok=True)

PHASE6_DIR = PROCESSED_ROOT / "phase_06_features"

print("=" * 65)
print("PHASE 9 — Step 9.G: Linear Mixed Effects Model")
print("=" * 65)

# ─── Load Phase 6 window-level data ───────────────────────────────────────────
def map_group(name):
    n = str(name).lower()
    if "als" in n:     return "ALS"
    if "healthy" in n: return "Healthy"
    return None

def extract_task(tid):
    t = str(tid).lower()
    if "drinking"   in t: return "drinking"
    if "lifting"    in t: return "lifting"
    if "pick"       in t: return "pick_place"
    return "other"

def extract_condition(tid):
    t = str(tid).lower()
    if "noexo" in t: return "noexo"
    if "exo"   in t: return "exo"
    return "unknown"

print("\nLoading Phase 6 window-level data...")
feat_files = sorted(PHASE6_DIR.glob("*__features.parquet"))
dfs = []
for fpath in feat_files:
    subject = fpath.name.split("__features.parquet")[0]
    df_s    = pd.read_parquet(fpath)
    df_s["subject"]   = subject
    df_s["group"]     = map_group(subject)
    df_s["task"]      = df_s["trial_id"].apply(extract_task)
    df_s["condition"] = df_s["trial_id"].apply(extract_condition)
    dfs.append(df_s)

df_all = pd.concat(dfs, ignore_index=True)

# Filter: valid conditions and tasks only
df_model = df_all[
    (df_all["condition"].isin(["exo","noexo"])) &
    (df_all["task"].isin(["drinking","lifting","pick_place"])) &
    (df_all["group"].notna())
].copy()

# Exclude confirmed artifact subject (this cell reads Phase 6 directly, not partA,
# so the exclusion must be applied here too for consistency with the rest of Phase 9).
# Reason: ALS_Subject_15 extensor band-pass +/-2-4 V (physiologically impossible for
# surface EMG), near-continuous activity -> confirmed sensor artifact, visually verified.
df_model = df_model[df_model["subject"] != "ALS_Subject_15"].copy()

print(f"Windows for modeling: {len(df_model)}")
print(f"Subjects: {df_model['subject'].nunique()}")

# ─── Aggregate to trial level ─────────────────────────────────────────────────
#
# LMM at window level would have autocorrelation between windows of same trial.
# We aggregate to trial level (median per trial) to get independent observations.
# This gives us one observation per subject × task × condition combination.

EMG_TARGETS = {
    "EMG_Extensor_RMS": "RMS Extensor",
    "EMG_Biceps_RMS"  : "RMS Biceps",
}

for feat in EMG_TARGETS:
    if feat in df_model.columns:
        df_model[feat] = pd.to_numeric(df_model[feat], errors="coerce")

df_trial = (
    df_model
    .groupby(["subject","group","task","condition"])[list(EMG_TARGETS.keys())]
    .median()
    .reset_index()
)

print(f"\nTrial-level observations: {len(df_trial)}")
print(f"  ALS trials    : {(df_trial['group']=='ALS').sum()}")
print(f"  Healthy trials: {(df_trial['group']=='Healthy').sum()}")
print(f"\nObservations per subject (should be ~6: 3 tasks × 2 conditions):")
obs_per_subj = df_trial.groupby("subject").size()
print(f"  Min: {obs_per_subj.min()}, Max: {obs_per_subj.max()}, "
      f"Mean: {obs_per_subj.mean():.1f}")

# ─── Set reference levels ─────────────────────────────────────────────────────
#
# Reference levels for interpretation:
#   Group:     Healthy (so ALS coefficient = ALS - Healthy)
#   Task:      drinking (most familiar task)
#   Condition: noexo (baseline without assistance)

df_trial["group"]     = pd.Categorical(df_trial["group"],
                         categories=["Healthy","ALS"], ordered=False)
df_trial["task"]      = pd.Categorical(df_trial["task"],
                         categories=["drinking","lifting","pick_place"], ordered=False)
df_trial["condition"] = pd.Categorical(df_trial["condition"],
                         categories=["noexo","exo"], ordered=False)

# ─── Run LMM for each target EMG feature ──────────────────────────────────────
all_lmm_results = []

for feat, feat_label in EMG_TARGETS.items():
    if feat not in df_trial.columns:
        continue

    df_feat = df_trial[["subject","group","task","condition",feat]].dropna().copy()

    # Log-transform (common for EMG amplitude — right-skewed distribution)
    # Add small constant to handle near-zero values
    df_feat["log_emg"] = np.log(df_feat[feat] + 1e-8)

    print(f"\n{'='*65}")
    print(f"LMM for: {feat_label} (log-transformed)")
    print(f"{'='*65}")
    print(f"Observations: {len(df_feat)}  |  Subjects: {df_feat['subject'].nunique()}")

    # ── Model 1: Main effects + interactions ──────────────────────────────────
    #
    # Formula: log_emg ~ C(group) + C(task) + C(condition)
    #                   + C(group):C(task) + C(group):C(condition)
    # Random: (1|subject) = random intercept per subject

    formula_full = (
        "log_emg ~ C(group, Treatment('Healthy')) "
        "+ C(task, Treatment('drinking')) "
        "+ C(condition, Treatment('noexo')) "
        "+ C(group, Treatment('Healthy')):C(task, Treatment('drinking')) "
        "+ C(group, Treatment('Healthy')):C(condition, Treatment('noexo'))"
    )

    try:
        model_full = smf.mixedlm(
            formula_full,
            data=df_feat,
            groups=df_feat["subject"]
        )
        result_full = model_full.fit(method="lbfgs", reml=True)

        print(f"\nModel converged: {result_full.converged}")
        print(f"Log-likelihood  : {result_full.llf:.4f}")
        print(f"AIC             : {result_full.aic:.4f}")
        print(f"\nFixed effects summary:")
        print(f"{'Parameter':<55} {'Coef':>8} {'SE':>8} {'z':>6} {'p':>8} {'Sig':>4}")
        print("-" * 95)

        params   = result_full.params
        bse      = result_full.bse
        pvalues  = result_full.pvalues
        zscores  = result_full.tvalues

        lmm_rows = []
        for param_name in params.index:
            if param_name == "Group Var":
                continue
            coef = params[param_name]
            se   = bse[param_name]
            z    = zscores[param_name]
            p    = pvalues[param_name]

            sig = "***" if p < 0.001 else ("**" if p < 0.01 else
                  ("*" if p < 0.05 else ("." if p < 0.10 else "")))

            # Clean parameter name for display
            display_name = (param_name
                .replace("C(group, Treatment('Healthy'))[T.ALS]", "ALS vs Healthy")
                .replace("C(task, Treatment('drinking'))[T.", "Task: ")
                .replace("C(condition, Treatment('noexo'))[T.exo]", "EXO vs no-EXO")
                .replace("]", "")
                .replace("C(group, Treatment('Healthy')):C(task, Treatment('drinking'))", "ALS × Task")
                .replace("C(group, Treatment('Healthy')):C(condition, Treatment('noexo'))", "ALS × EXO")
                .replace("[T.ALS]", "[ALS]")
            )

            print(f"  {display_name:<53} {coef:>+8.3f} {se:>8.3f} {z:>6.2f} {p:>8.4f} {sig:>4}")

            lmm_rows.append({
                "feature"   : feat,
                "label"     : feat_label,
                "parameter" : param_name,
                "display"   : display_name,
                "coef"      : round(coef, 4),
                "se"        : round(se, 4),
                "z"         : round(z, 3),
                "p_value"   : round(p, 6),
                "significant": p < 0.05,
            })

        # Random effects variance
        re_var = result_full.cov_re.values[0][0] if hasattr(result_full, 'cov_re') else np.nan
        resid_var = result_full.scale
        icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else np.nan
        print(f"\nRandom effects:")
        print(f"  Subject variance (RE): {re_var:.4f}")
        print(f"  Residual variance    : {resid_var:.4f}")
        print(f"  ICC (intraclass corr): {icc:.4f}")
        print(f"  ICC interpretation   : {'high' if icc > 0.50 else 'moderate' if icc > 0.25 else 'low'} "
              f"within-subject clustering")

        all_lmm_results.extend(lmm_rows)

        # ── Model comparison: full vs reduced ─────────────────────────────────
        # Test whether interactions significantly improve fit
        formula_reduced = (
            "log_emg ~ C(group, Treatment('Healthy')) "
            "+ C(task, Treatment('drinking')) "
            "+ C(condition, Treatment('noexo'))"
        )
        model_red = smf.mixedlm(formula_reduced, data=df_feat,
                                 groups=df_feat["subject"])
        result_red = model_red.fit(method="lbfgs", reml=False)

        # Refit full with REML=False for LRT
        result_full_ml = smf.mixedlm(
            formula_full, data=df_feat, groups=df_feat["subject"]
        ).fit(method="lbfgs", reml=False)

        lrt_stat = 2 * (result_full_ml.llf - result_red.llf)
        lrt_df   = len(result_full_ml.params) - len(result_red.params)
        lrt_p    = stats.chi2.sf(lrt_stat, lrt_df)

        print(f"\nLikelihood Ratio Test (full vs no-interactions):")
        print(f"  LRT statistic: {lrt_stat:.3f}  df={lrt_df}  p={lrt_p:.4f}")
        if lrt_p < 0.05:
            print(f"  -> Interactions significantly improve model fit")
        else:
            print(f"  -> Interactions do not significantly improve fit")

    except Exception as e:
        print(f"  Model fitting failed: {e}")

# ─── Interpretation ───────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("CLINICAL INTERPRETATION OF LMM")
print(f"{'='*65}")

print("""
Key parameters to look for in results:

1. ALS vs Healthy (main effect):
   Positive coef = ALS has HIGHER log(EMG) than Healthy
   This is the compensatory hyperactivation effect.
   Expected: positive and significant.

2. Task effects (lifting, pick_place vs drinking):
   Shows how EMG changes across tasks (controlling for group).

3. EXO vs no-EXO (main effect):
   Negative coef = EXO reduces EMG overall.
   Expected: negative.

4. ALS × EXO interaction:
   KEY: If significant and negative, EXO reduces EMG MORE in ALS.
   If positive, EXO reduces EMG LESS in ALS (what we found earlier).

5. ICC (Intraclass Correlation Coefficient):
   High ICC = large between-subject variability.
   This justifies the use of LMM over simple ANOVA.
   ICC > 0.50 means subjects differ substantially in baseline EMG.
""")

print(f"\nMethods statement for paper:")
print("""
  Linear mixed-effects models (LMM) were fitted to trial-level EMG
  amplitude data to simultaneously estimate the effects of diagnostic
  group (ALS vs Healthy), task (drinking, lifting, pick&place),
  exoskeleton condition (EXO vs no-EXO), and their interactions,
  while accounting for repeated measurements within subjects.

  Subject was included as a random intercept. EMG amplitudes were
  log-transformed prior to modeling to address right-skewed
  distributions identified in the normality analysis (Step 9.A).
  Models were fitted using restricted maximum likelihood (REML)
  via the statsmodels MixedLM implementation.

  Reference levels: Healthy group, drinking task, no-EXO condition.
  Model fit was assessed via AIC and likelihood ratio tests comparing
  models with and without interaction terms.
""")

# ─── Save ─────────────────────────────────────────────────────────────────────
if all_lmm_results:
    df_lmm = pd.DataFrame(all_lmm_results)
    df_lmm.to_csv(PHASE9_STATS / "lmm_results.csv", index=False)
    print(f"\nOutputs saved:")
    print(f"  lmm_results.csv -> all fixed effect estimates")
else:
    print("\nNo results to save.")
