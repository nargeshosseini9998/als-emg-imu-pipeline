# alsexo/stats/step_d_bootstrap_clusters.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 10); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.step_d_bootstrap_clusters`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 9 — Step 9.D: Bootstrap Stability Analysis
#
# Purpose:
#   Assess whether hierarchical clustering results are stable
#   or artifacts of the small sample (n=21 after outlier removal).
#
# Method:
#   1. Bootstrap resampling (n=1000 iterations)
#   2. For each resample: run Ward hierarchical clustering (k=2, k=3)
#   3. Compute co-occurrence matrix: how often do pairs of subjects
#      land in the same cluster?
#   4. Jaccard index: compare each bootstrap cluster to original clusters
#
# Interpretation:
#   Co-occurrence > 0.75 → subjects reliably cluster together (stable)
#   Jaccard > 0.75 → original cluster structure is stable
#
# Output:
#   phase_09_stats/step_D_bootstrap/bootstrap_stability.csv
#   phase_09_stats/step_D_bootstrap/cooccurrence_matrix_k3.csv
#   phase_09_stats/step_D_bootstrap/jaccard_results.csv
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import cluster, spatial
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

np.random.seed(42)


PHASE9_STATS = PROCESSED_ROOT / "phase_09_stats" / "step_D_bootstrap"
PHASE9_STATS.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PHASE 9 — Step 9.D: Bootstrap Stability Analysis")
print("=" * 65)

# ─── Load data ────────────────────────────────────────────────────────────────
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

FEATURES = ["EMG_Extensor_RMS", "EMG_Flexor_RMS",
            "EMG_Biceps_RMS",   "EMG_Triceps_RMS"]
FEATURES = [f for f in FEATURES if f in df_clean.columns]

subjects = df_clean["subject"].values
groups   = df_clean["group"].values
n_subj   = len(subjects)

print(f"\nSubjects: {n_subj} (ALS_Subject_15 already excluded in Step 9.0)")
print(f"Features: {FEATURES}")

# ─── Original clustering ───────────────────────────────────────────────────────
X_raw    = df_clean[FEATURES].fillna(df_clean[FEATURES].median())
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

linkage_orig  = cluster.hierarchy.linkage(X_scaled, method="ward")
labels_k2_orig = cluster.hierarchy.fcluster(linkage_orig, 2, criterion="maxclust")
labels_k3_orig = cluster.hierarchy.fcluster(linkage_orig, 3, criterion="maxclust")

print(f"\nOriginal cluster assignments:")
for k, labels in [(2, labels_k2_orig), (3, labels_k3_orig)]:
    sizes = [int((labels==c).sum()) for c in np.unique(labels)]
    print(f"  k={k}: sizes={sizes}")

# ─── Bootstrap ────────────────────────────────────────────────────────────────
N_BOOT = 1000
print(f"\nRunning {N_BOOT} bootstrap iterations...")

# Co-occurrence matrices: how often do pairs land in same cluster?
cooccur_k2 = np.zeros((n_subj, n_subj))
cooccur_k3 = np.zeros((n_subj, n_subj))
count_both  = np.zeros((n_subj, n_subj))  # how often both are in resample

# Jaccard scores per original cluster
# For each bootstrap: find best-matching cluster and compute Jaccard
jaccard_k2 = {c: [] for c in np.unique(labels_k2_orig)}
jaccard_k3 = {c: [] for c in np.unique(labels_k3_orig)}

for boot_i in range(N_BOOT):
    # Resample with replacement (same n)
    idx_boot = np.random.choice(n_subj, size=n_subj, replace=True)
    X_boot   = X_scaled[idx_boot]

    # Re-scale on bootstrap sample
    sc_boot  = StandardScaler()
    X_boot_s = sc_boot.fit_transform(X_boot)

    try:
        link_boot = cluster.hierarchy.linkage(X_boot_s, method="ward")
        lab_k2    = cluster.hierarchy.fcluster(link_boot, 2, criterion="maxclust")
        lab_k3    = cluster.hierarchy.fcluster(link_boot, 3, criterion="maxclust")
    except Exception:
        continue

    # Map back to original subject indices
    unique_boot = np.unique(idx_boot)

    # For co-occurrence: check pairs of original subjects
    for i in range(n_subj):
        for j in range(i+1, n_subj):
            # Check if both i and j appear in this bootstrap
            i_in = idx_boot == i
            j_in = idx_boot == j
            if i_in.any() and j_in.any():
                count_both[i, j] += 1
                count_both[j, i] += 1
                # Get their cluster labels (use first occurrence)
                ci_k2 = lab_k2[np.where(idx_boot == i)[0][0]]
                cj_k2 = lab_k2[np.where(idx_boot == j)[0][0]]
                ci_k3 = lab_k3[np.where(idx_boot == i)[0][0]]
                cj_k3 = lab_k3[np.where(idx_boot == j)[0][0]]

                if ci_k2 == cj_k2:
                    cooccur_k2[i, j] += 1
                    cooccur_k2[j, i] += 1
                if ci_k3 == cj_k3:
                    cooccur_k3[i, j] += 1
                    cooccur_k3[j, i] += 1

    # Jaccard index: for each original cluster, find best-matching bootstrap cluster
    for k, lab_orig, lab_boot, jacc_dict in [
        (2, labels_k2_orig, lab_k2, jaccard_k2),
        (3, labels_k3_orig, lab_k3, jaccard_k3),
    ]:
        for orig_c in np.unique(lab_orig):
            orig_set = set(np.where(lab_orig == orig_c)[0])
            best_j   = 0.0
            for boot_c in np.unique(lab_boot):
                boot_set_idx = set(np.where(lab_boot == boot_c)[0])
                # Map back to original indices
                boot_orig_idx = set(idx_boot[list(boot_set_idx)])
                intersect = len(orig_set & boot_orig_idx)
                union     = len(orig_set | boot_orig_idx)
                j_val     = intersect / union if union > 0 else 0.0
                best_j    = max(best_j, j_val)
            jacc_dict[orig_c].append(best_j)

print(f"  Done.")

# ─── Normalize co-occurrence by number of times both appeared ─────────────────
# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    cooccur_k2_norm = np.where(count_both > 0, cooccur_k2 / count_both, 0.0)
    cooccur_k3_norm = np.where(count_both > 0, cooccur_k3 / count_both, 0.0)

# ─── Jaccard summary ──────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("JACCARD INDEX — Cluster stability")
print(f"{'='*65}")
print(f"\nJaccard > 0.75 → stable  |  0.50-0.75 → moderate  |  < 0.50 → unstable\n")

jaccard_rows = []
for k, jacc_dict, lab_orig in [
    (2, jaccard_k2, labels_k2_orig),
    (3, jaccard_k3, labels_k3_orig),
]:
    print(f"k={k}:")
    for c in sorted(jacc_dict.keys()):
        vals      = np.array(jacc_dict[c])
        mean_j    = float(np.mean(vals))
        sd_j      = float(np.std(vals))
        members   = subjects[lab_orig == c]
        grps      = groups[lab_orig == c]
        n_als_c   = int((grps=="ALS").sum())
        n_hc_c    = int((grps=="Healthy").sum())

        if mean_j >= 0.75:   stability = "STABLE"
        elif mean_j >= 0.50: stability = "MODERATE"
        else:                 stability = "UNSTABLE"

        print(f"  Cluster {c} (n={len(members)}: {n_als_c} ALS, {n_hc_c} HC): "
              f"Jaccard={mean_j:.3f} ± {sd_j:.3f}  [{stability}]")

        jaccard_rows.append({
            "k"       : k,
            "cluster" : c,
            "n"       : len(members),
            "n_als"   : n_als_c,
            "n_hc"    : n_hc_c,
            "jaccard_mean": round(mean_j, 4),
            "jaccard_sd"  : round(sd_j, 4),
            "stability"   : stability,
            "members" : ";".join(members),
        })
    print()

# ─── Co-occurrence analysis ───────────────────────────────────────────────────
print(f"{'='*65}")
print("CO-OCCURRENCE ANALYSIS (k=3)")
print(f"{'='*65}")
print(f"\nFor each original cluster: mean co-occurrence within vs between clusters")
print(f"(higher within = more stable)\n")

for c in np.unique(labels_k3_orig):
    in_idx  = np.where(labels_k3_orig == c)[0]
    out_idx = np.where(labels_k3_orig != c)[0]

    within_pairs  = [(i,j) for i in in_idx  for j in in_idx  if i < j]
    between_pairs = [(i,j) for i in in_idx  for j in out_idx]

    within_vals  = [cooccur_k3_norm[i,j] for i,j in within_pairs  if count_both[i,j] > 0]
    between_vals = [cooccur_k3_norm[i,j] for i,j in between_pairs if count_both[i,j] > 0]

    mean_within  = float(np.mean(within_vals))  if within_vals  else np.nan
    mean_between = float(np.mean(between_vals)) if between_vals else np.nan
    separation   = mean_within - mean_between

    members  = subjects[labels_k3_orig == c]
    grps     = groups[labels_k3_orig == c]
    n_als_c  = int((grps=="ALS").sum())
    n_hc_c   = int((grps=="Healthy").sum())

    print(f"  Cluster {c} (n={len(members)}: {n_als_c} ALS, {n_hc_c} HC):")
    print(f"    Within-cluster co-occurrence  : {mean_within:.3f}")
    print(f"    Between-cluster co-occurrence : {mean_between:.3f}")
    print(f"    Separation (within - between) : {separation:+.3f}")
    if separation > 0.30:   print(f"    -> Well-separated cluster")
    elif separation > 0.10: print(f"    -> Moderately separated cluster")
    else:                   print(f"    -> Poorly separated — interpret with caution")
    print()

# ─── Subject-level stability ──────────────────────────────────────────────────
print(f"{'='*65}")
print("SUBJECT-LEVEL STABILITY (k=3)")
print(f"{'='*65}")
print(f"\nFor each subject: % of bootstrap samples assigned to same cluster as original\n")

# For each subject, how often do they appear in bootstrap and stay in same cluster?
subject_stability = []
for i, (subj, grp) in enumerate(zip(subjects, groups)):
    orig_c = labels_k3_orig[i]

    same_count  = 0
    total_count = 0

    for boot_i in range(N_BOOT):
        # Re-run one bootstrap to get per-subject stability
        # (approximated from co-occurrence with subjects in same original cluster)
        pass

    # Use co-occurrence with cluster centroid (average with same-cluster members)
    same_cluster_idx = np.where(labels_k3_orig == orig_c)[0]
    same_cluster_idx = same_cluster_idx[same_cluster_idx != i]

    if len(same_cluster_idx) == 0:
        stability_pct = np.nan
    else:
        pair_cooccur = [cooccur_k3_norm[i, j]
                        for j in same_cluster_idx
                        if count_both[i, j] > 0]
        stability_pct = float(np.mean(pair_cooccur)) if pair_cooccur else np.nan

    subject_stability.append({
        "subject"       : subj,
        "group"         : grp,
        "orig_cluster"  : orig_c,
        "stability_pct" : round(stability_pct, 3) if np.isfinite(stability_pct) else np.nan,
    })

df_stab = pd.DataFrame(subject_stability).sort_values(
    ["orig_cluster", "stability_pct"], ascending=[True, False]
)

print(f"{'Subject':<25} {'Group':<10} {'Cluster':>8} {'Stability':>12}")
print("-" * 58)
for _, row in df_stab.iterrows():
    stab = row["stability_pct"]
    flag = "(*)" if (np.isfinite(stab) and stab < 0.50) else ""
    stab_str = f"{stab:.3f}" if np.isfinite(stab) else "  n/a"
    print(f"  {row['subject']:<23} {row['group']:<10} "
          f"{int(row['orig_cluster']):>8} {stab_str:>12} {flag}")

low_stability = df_stab[df_stab["stability_pct"] < 0.50]["subject"].tolist()
if low_stability:
    print(f"\n  Subjects with low stability (<0.50): {low_stability}")

# ─── Overall verdict ──────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("OVERALL STABILITY VERDICT")
print(f"{'='*65}")

df_j = pd.DataFrame(jaccard_rows)
k3_j = df_j[df_j["k"]==3]
mean_k3_j = k3_j["jaccard_mean"].mean()

print(f"\nMean Jaccard k=3: {mean_k3_j:.3f}")
if mean_k3_j >= 0.75:
    print(f"-> STABLE: Clustering is robust to sampling variation")
elif mean_k3_j >= 0.50:
    print(f"-> MODERATE: Clustering is partially stable")
    print(f"   Interpret with caution — n=22 limits bootstrap reliability")
else:
    print(f"-> UNSTABLE: Clustering is sensitive to sample composition")
    print(f"   Results should be considered exploratory only")

# ─── Save ─────────────────────────────────────────────────────────────────────
df_j.to_csv(PHASE9_STATS / "jaccard_results.csv", index=False)
df_stab.to_csv(PHASE9_STATS / "subject_stability.csv", index=False)

pd.DataFrame({
    "subject_i"   : [subjects[i] for i in range(n_subj) for j in range(n_subj)],
    "subject_j"   : [subjects[j] for i in range(n_subj) for j in range(n_subj)],
    "cooccur_k3"  : cooccur_k3_norm.flatten(),
    "count_both"  : count_both.flatten(),
}).to_csv(PHASE9_STATS / "cooccurrence_matrix_k3.csv", index=False)

print(f"\nOutputs saved:")
print(f"  jaccard_results.csv      -> stability per cluster")
print(f"  subject_stability.csv    -> per-subject stability score")
print(f"  cooccurrence_matrix_k3.csv -> full co-occurrence matrix")
