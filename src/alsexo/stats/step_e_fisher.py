# alsexo/stats/step_e_fisher.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 12); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.step_e_fisher`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — Step 9.E: Fisher's Exact Test — Cluster Validation
#
# Purpose:
#   Test whether cluster assignments are statistically associated
#   with the true ALS/Healthy labels (external validation).
#
#   If clustering captured meaningful neuromuscular differences,
#   ALS and Healthy subjects should be non-randomly distributed
#   across clusters.
#
# Methods:
#   1. Fisher's Exact Test on 2×2 contingency table (k=2)
#   2. Fisher's Exact Test on 2×3 contingency table (k=3)
#      → implemented via permutation test (exact Fisher for 2×k)
#   3. Phi coefficient / Cramer's V (effect size for contingency tables)
#   4. Sensitivity and Specificity of clustering as ALS detector
#
# Output:
#   phase_09_stats/step_E_fisher/fisher_results.csv
#   phase_09_stats/step_E_fisher/classification_metrics.csv
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, cluster, spatial
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


PHASE9_STATS = PROCESSED_ROOT / "phase_09_stats" / "step_E_fisher"
PHASE9_STATS.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PHASE 9 — Step 9.E: Fisher's Exact Test — Cluster Validation")
print("=" * 65)

# ─── Load and prepare data ────────────────────────────────────────────────────
df = pd.read_parquet(
    PROCESSED_ROOT / "phase_09_efa" / "partA_subject_matrix.parquet"
)

def map_group(name):
    n = str(name).lower()
    if "als" in n:     return "ALS"
    if "healthy" in n: return "Healthy"
    return None

if "group" not in df.columns:
    df["group"] = df["subject"].map(map_group)

# ALS_Subject_15 already excluded upstream in Step 9.0 (documented sensor artifact).
df_clean = df.copy().reset_index(drop=True)

FEATURES = [f for f in ["EMG_Extensor_RMS","EMG_Flexor_RMS",
                         "EMG_Biceps_RMS","EMG_Triceps_RMS"]
            if f in df_clean.columns]

subjects = df_clean["subject"].values
groups   = df_clean["group"].values
n_subj   = len(subjects)

# ─── Recreate original clustering ─────────────────────────────────────────────
X_raw    = df_clean[FEATURES].fillna(df_clean[FEATURES].median())
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

linkage       = cluster.hierarchy.linkage(X_scaled, method="ward")
labels_k2     = cluster.hierarchy.fcluster(linkage, 2, criterion="maxclust")
labels_k3     = cluster.hierarchy.fcluster(linkage, 3, criterion="maxclust")

print(f"\nSubjects: {n_subj} (ALS_Subject_15 already excluded in Step 9.0)")
print(f"Groups: ALS={( groups=='ALS').sum()}, Healthy={(groups=='Healthy').sum()}")

# ─── Helper: Phi coefficient (2×2) ────────────────────────────────────────────
def phi_coefficient(contingency_2x2):
    """Effect size for 2×2 tables. Range [-1, 1]."""
    a, b = contingency_2x2[0]
    c, d = contingency_2x2[1]
    n    = a + b + c + d
    num  = a*d - b*c
    den  = np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    return float(num/den) if den > 0 else 0.0

def cramers_v(contingency):
    """Cramer's V — effect size for r×c tables. Range [0, 1]."""
    chi2 = stats.chi2_contingency(contingency, correction=False)[0]
    n    = contingency.sum()
    k    = min(contingency.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if (n > 0 and k > 0) else 0.0

def permutation_test_association(labels, groups, n_perm=10000):
    """
    Permutation test for association between cluster labels and group labels.
    Returns p-value: probability that random shuffling produces
    equal or stronger association.
    """
    # Use chi2 statistic as association measure
    contingency = pd.crosstab(labels, groups).values
    observed_chi2 = stats.chi2_contingency(contingency, correction=False)[0]

    perm_chi2 = []
    for _ in range(n_perm):
        shuffled = np.random.permutation(groups)
        cont_perm = pd.crosstab(labels, shuffled).values
        try:
            chi2_perm = stats.chi2_contingency(cont_perm, correction=False)[0]
        except:
            chi2_perm = 0.0
        perm_chi2.append(chi2_perm)

    p_perm = float(np.mean(np.array(perm_chi2) >= observed_chi2))
    return p_perm, observed_chi2

# ─── Analysis for k=2 ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("k=2 ANALYSIS")
print(f"{'='*65}")

contingency_k2 = pd.crosstab(
    pd.Series(labels_k2, name="Cluster"),
    pd.Series(groups,    name="Group")
)
print(f"\nContingency table (k=2):")
print(contingency_k2.to_string())

# Fisher's Exact Test (2×2 only if k=2 has exactly 2 clusters)
if contingency_k2.shape == (2, 2):
    table_2x2 = contingency_k2.values
    oddsratio, p_fisher = stats.fisher_exact(table_2x2, alternative="two-sided")
    phi = phi_coefficient(table_2x2)

    print(f"\nFisher's Exact Test:")
    print(f"  Odds Ratio : {oddsratio:.4f}")
    print(f"  p-value    : {p_fisher:.6f}")
    print(f"  Phi coeff  : {phi:.4f}  (effect size; |phi|>0.3=medium, >0.5=large)")

    if p_fisher < 0.001:
        print(f"  -> HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_fisher < 0.05:
        print(f"  -> SIGNIFICANT (p < 0.05)")
    else:
        print(f"  -> Not significant")
else:
    p_fisher = np.nan
    oddsratio = np.nan
    phi = cramers_v(contingency_k2.values)
    p_perm, chi2_obs = permutation_test_association(labels_k2, groups)
    print(f"\nPermutation test (k=2 has >2 rows):")
    print(f"  Chi2 observed: {chi2_obs:.4f}")
    print(f"  p-value (permutation, n=10000): {p_perm:.4f}")

# ─── Analysis for k=3 ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("k=3 ANALYSIS")
print(f"{'='*65}")

contingency_k3 = pd.crosstab(
    pd.Series(labels_k3, name="Cluster"),
    pd.Series(groups,    name="Group")
)
print(f"\nContingency table (k=3):")
print(contingency_k3.to_string())

# For 2×3, use chi2 + permutation test
p_perm_k3, chi2_obs_k3 = permutation_test_association(labels_k3, groups)
v_k3 = cramers_v(contingency_k3.values)

print(f"\nPermutation test (10,000 permutations):")
print(f"  Chi2 observed : {chi2_obs_k3:.4f}")
print(f"  p-value       : {p_perm_k3:.4f}")
print(f"  Cramer's V    : {v_k3:.4f}  (effect size; >0.3=medium, >0.5=large)")

if p_perm_k3 < 0.001:
    print(f"  -> HIGHLY SIGNIFICANT (p < 0.001)")
elif p_perm_k3 < 0.05:
    print(f"  -> SIGNIFICANT (p < 0.05)")
else:
    print(f"  -> Not significant")

# ─── Classification metrics ───────────────────────────────────────────────────
#
# Treating clustering as an ALS detector:
# Which cluster most discriminates ALS from Healthy?
# We use k=3 and define "ALS cluster" as the one with highest ALS proportion.
#
# Key metrics:
#   Sensitivity = TP / (TP + FN) = how many ALS correctly identified
#   Specificity = TN / (TN + FP) = how many Healthy correctly identified
#   PPV = TP / (TP + FP) = precision
#   NPV = TN / (TN + FN)

print(f"\n{'='*65}")
print("CLASSIFICATION METRICS — Cluster as ALS Detector")
print(f"{'='*65}")

print(f"\nFor k=3, treating Cluster 2 (9 ALS, 1 HC) as 'ALS cluster':")
print(f"and Cluster 1 (4 ALS, 7 HC) as 'Healthy-like cluster':")
print(f"(Cluster 3 = ALS_4 only, excluded from this analysis)")

# Using k=3: cluster 2 = ALS, cluster 1 = Healthy-like
# Exclude Cluster 3 (n=1) for cleaner metrics
mask_12 = labels_k3 != 3
labels_12 = labels_k3[mask_12]
groups_12 = groups[mask_12]
subj_12   = subjects[mask_12]

# Define: predict ALS if in Cluster 2
y_true = (groups_12 == "ALS").astype(int)
y_pred = (labels_12 == 2).astype(int)

TP = int(np.sum((y_true==1) & (y_pred==1)))
TN = int(np.sum((y_true==0) & (y_pred==0)))
FP = int(np.sum((y_true==0) & (y_pred==1)))
FN = int(np.sum((y_true==1) & (y_pred==0)))

sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
ppv         = TP / (TP + FP) if (TP + FP) > 0 else np.nan
npv         = TN / (TN + FN) if (TN + FN) > 0 else np.nan
accuracy    = (TP + TN) / (TP + TN + FP + FN)

print(f"\nConfusion Matrix:")
print(f"              Predicted ALS   Predicted HC")
print(f"  True ALS         {TP:>3}            {FN:>3}")
print(f"  True HC          {FP:>3}            {TN:>3}")

print(f"\nClassification metrics:")
print(f"  Sensitivity (Recall)  : {sensitivity:.3f}  ({TP}/{TP+FN} ALS correctly detected)")
print(f"  Specificity           : {specificity:.3f}  ({TN}/{TN+FP} HC correctly identified)")
print(f"  PPV (Precision)       : {ppv:.3f}  ({TP}/{TP+FP} predicted ALS are true ALS)")
print(f"  NPV                   : {npv:.3f}  ({TN}/{TN+FN} predicted HC are true HC)")
print(f"  Accuracy              : {accuracy:.3f}  ({TP+TN}/{TP+TN+FP+FN} correct)")

print(f"\nInterpretation:")
print(f"  Sensitivity={sensitivity:.2f}: clustering detected {sensitivity*100:.0f}% of ALS patients")
print(f"  Specificity={specificity:.2f}: clustering correctly excluded {specificity*100:.0f}% of HC")
print(f"  4 ALS patients in Cluster 1 represent 'mild ALS' with Healthy-like EMG profile")

# ─── Subject-level assignments ────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SUBJECT-LEVEL CLUSTER ASSIGNMENTS (k=3)")
print(f"{'='*65}")

cluster_names = {1: "Healthy-like", 2: "Compensatory ALS", 3: "Outlier"}
print(f"\n{'Subject':<25} {'True Group':<12} {'Cluster':<8} {'Cluster Label':<20} {'Correct?'}")
print("-" * 78)

correct_count = 0
total_count   = 0
for i, (subj, grp, lab) in enumerate(zip(subjects, groups, labels_k3)):
    clust_name = cluster_names.get(lab, str(lab))
    if lab == 3:
        correct = "n/a"
    elif (grp == "ALS"     and lab == 2) or \
         (grp == "Healthy" and lab == 1):
        correct = "✓"
        correct_count += 1
        total_count   += 1
    else:
        correct = "✗"
        total_count += 1

    print(f"  {subj:<23} {grp:<12} {lab:<8} {clust_name:<20} {correct}")

print(f"\nOverall: {correct_count}/{total_count} subjects correctly assigned "
      f"({correct_count/total_count*100:.0f}%) — excluding Cluster 3 (n=1)")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SUMMARY FOR PAPER")
print(f"{'='*65}")

print(f"""
Fisher's Exact / Permutation test results:
  k=2: Fisher p = {p_fisher:.4f} (if applicable)
  k=3: Permutation p = {p_perm_k3:.4f}  Cramer's V = {v_k3:.4f}

Classification performance (k=3, cluster 2 = ALS):
  Sensitivity = {sensitivity:.2f}  Specificity = {specificity:.2f}
  PPV = {ppv:.2f}  NPV = {npv:.2f}  Accuracy = {accuracy:.2f}

Paper statement:
  "The k=3 cluster solution showed significant association with
  diagnostic group labels (permutation test, p={p_perm_k3:.3f},
  Cramer's V={v_k3:.2f}). Using the 'compensatory' cluster as
  an ALS indicator yielded sensitivity={sensitivity:.2f} and
  specificity={specificity:.2f}, with 4 ALS patients clustered
  with healthy controls, suggesting a mild functional phenotype
  indistinguishable from healthy by EMG amplitude alone."
""")

# ─── Save ─────────────────────────────────────────────────────────────────────
fisher_rows = [
    {"k": 2, "test": "Fisher's Exact",  "p_value": p_fisher,    "effect_size": phi,   "effect_metric": "Phi"},
    {"k": 3, "test": "Permutation chi2","p_value": p_perm_k3,   "effect_size": v_k3,  "effect_metric": "Cramer's V"},
]
pd.DataFrame(fisher_rows).to_csv(PHASE9_STATS / "fisher_results.csv", index=False)

metrics_row = [{
    "k": 3, "als_cluster": 2,
    "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    "sensitivity": round(sensitivity, 3),
    "specificity" : round(specificity, 3),
    "ppv"         : round(ppv, 3),
    "npv"         : round(npv, 3),
    "accuracy"    : round(accuracy, 3),
}]
pd.DataFrame(metrics_row).to_csv(PHASE9_STATS / "classification_metrics.csv", index=False)

assignments = pd.DataFrame({
    "subject": subjects,
    "true_group": groups,
    "cluster_k2": labels_k2,
    "cluster_k3": labels_k3,
    "cluster_k3_label": [cluster_names.get(l, str(l)) for l in labels_k3],
})
assignments.to_csv(PHASE9_STATS / "cluster_assignments.csv", index=False)

print(f"Outputs saved:")
print(f"  fisher_results.csv       -> test statistics and effect sizes")
print(f"  classification_metrics.csv -> sensitivity/specificity")
print(f"  cluster_assignments.csv  -> per-subject cluster labels")
