# alsexo/supervised.py
# Phase 10 — supervised ALS-vs-Healthy classification (LogReg / SVM-RBF / RF), leave-one-subject-out CV,
# in-fold scaling, balanced accuracy, cross-model consistency, SHAP. Extracted verbatim from 10_supervised.ipynb.
# NOTE: runs on import (script-style); execute via the CLI.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Phase 10 — Supervised Learning: ALS vs Healthy Classification
# Fixed: SHAP array shape (22, 4, 2) → use [:, :, 1] for ALS class
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")

import shap  # listed in requirements.txt

PHASE10_DIR    = PROCESSED_ROOT / "phase_10_supervised"
PHASE10_DIR.mkdir(parents=True, exist_ok=True)
PHASE6_DIR     = PROCESSED_ROOT / "phase_06_features"

np.random.seed(42)

print("=" * 65)
print("PHASE 10 — Supervised Learning: ALS vs Healthy")
print("=" * 65)

# ─── Build subject-level feature matrix ───────────────────────────────────────
def map_group(name):
    n = str(name).lower()
    if "als" in n:     return "ALS"
    if "healthy" in n: return "Healthy"
    return None

feat_files = sorted(PHASE6_DIR.glob("*__features.parquet"))
dfs = []
for fpath in feat_files:
    subject = fpath.name.split("__features.parquet")[0]
    df_s    = pd.read_parquet(fpath)
    df_s["subject"]   = subject
    df_s["group"]     = map_group(subject)
    df_s["condition"] = df_s["trial_id"].astype(str).str.lower().apply(
        lambda t: "noexo" if "noexo" in t else ("exo" if "exo" in t else "unknown")
    )
    dfs.append(df_s)

df_all   = pd.concat(dfs, ignore_index=True)
df_noexo = df_all[
    (df_all["condition"] == "noexo") &
    (df_all["subject"]   != "ALS_Subject_15")
].copy()

FEATURES = {
    "EMG_Extensor_RMS": "RMS Extensor",
    "EMG_Flexor_RMS"  : "RMS Flexor",
    "EMG_Biceps_RMS"  : "RMS Biceps",
    "EMG_Triceps_RMS" : "RMS Triceps",
}
FEAT_COLS  = [f for f in FEATURES if f in df_noexo.columns]
FEAT_NAMES = [FEATURES[f] for f in FEAT_COLS]

for f in FEAT_COLS:
    df_noexo[f] = pd.to_numeric(df_noexo[f], errors="coerce")

df_subj = (
    df_noexo
    .groupby(["subject","group"])[FEAT_COLS]
    .median()
    .reset_index()
)

subjects  = df_subj["subject"].values
groups    = df_subj["group"].values
y         = (groups == "ALS").astype(int)
X_raw     = df_subj[FEAT_COLS].fillna(df_subj[FEAT_COLS].median()).values
n_subj    = len(subjects)
n_als     = int(y.sum())
n_healthy = int((y == 0).sum())

print(f"\nSubjects: {n_subj}  (ALS={n_als}, Healthy={n_healthy})")
print(f"Features: {FEAT_NAMES}")

# ─── Models ───────────────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000,
        class_weight="balanced", random_state=42
    ),
    "SVM-RBF": SVC(
        C=1.0, kernel="rbf", gamma="scale",
        probability=True, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=500, max_depth=2, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

# ─── LOSO Cross-Validation ────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"Leave-One-Subject-Out CV ({n_subj} folds)")
print(f"{'='*65}\n")

all_fold_results = []

for model_name, model_template in MODELS.items():
    y_true_loso, y_pred_loso, y_prob_loso = [], [], []

    for test_idx in range(n_subj):
        train_mask = np.ones(n_subj, dtype=bool)
        train_mask[test_idx] = False

        X_train = X_raw[train_mask]
        X_test  = X_raw[[test_idx]]
        y_train = y[train_mask]
        y_test  = y[[test_idx]]

        # Scale on TRAIN only — no leakage
        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(np.nan_to_num(X_train))
        X_test_s  = scaler.transform(np.nan_to_num(X_test))

        model = clone(model_template)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        y_true_loso.append(int(y_test[0]))
        y_pred_loso.append(int(y_pred[0]))
        y_prob_loso.append(float(y_prob[0]))

        all_fold_results.append({
            "model"     : model_name,
            "subject"   : subjects[test_idx],
            "true_group": groups[test_idx],
            "y_true"    : int(y_test[0]),
            "y_pred"    : int(y_pred[0]),
            "y_prob"    : round(float(y_prob[0]), 4),
            "correct"   : int(y_test[0]) == int(y_pred[0]),
        })

    y_t = np.array(y_true_loso)
    y_p = np.array(y_pred_loso)
    y_r = np.array(y_prob_loso)

    bal  = balanced_accuracy_score(y_t, y_p)
    f1   = f1_score(y_t, y_p, average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_t, y_r)
    except:
        auc = np.nan

    print(f"  {model_name:<22} BalAcc={bal:.3f}  AUC={auc:.3f}  "
          f"Sens={sens:.3f}  Spec={spec:.3f}  F1={f1:.3f}")

# ─── Aggregate results ────────────────────────────────────────────────────────
df_folds = pd.DataFrame(all_fold_results)

print(f"\n{'='*65}")
print("MODEL COMPARISON")
print(f"{'='*65}")
print(f"\n{'Model':<22} {'BalAcc':>8} {'AUC':>6} {'Sens':>6} {'Spec':>6} "
      f"{'PPV':>6} {'NPV':>6} {'F1':>6} {'Acc':>6}")
print("-" * 78)

model_summary = []
for model_name in MODELS:
    sub = df_folds[df_folds["model"] == model_name]
    y_t = sub["y_true"].values
    y_p = sub["y_pred"].values
    y_r = sub["y_prob"].values

    bal  = balanced_accuracy_score(y_t, y_p)
    f1   = f1_score(y_t, y_p, average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc  = (tp + tn) / n_subj
    try:
        auc = roc_auc_score(y_t, y_r)
    except:
        auc = np.nan

    print(f"  {model_name:<20} {bal:>8.3f} {auc:>6.3f} "
          f"{sens:>6.3f} {spec:>6.3f} {ppv:>6.3f} {npv:>6.3f} "
          f"{f1:>6.3f} {acc:>6.3f}")

    model_summary.append({
        "model"      : model_name,
        "bal_acc"    : round(bal,  4),
        "auc"        : round(auc,  4) if np.isfinite(auc) else np.nan,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "ppv"        : round(ppv,  4),
        "npv"        : round(npv,  4),
        "f1_macro"   : round(f1,   4),
        "accuracy"   : round(acc,  4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    })

# ─── Per-subject predictions ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print("PER-SUBJECT PREDICTIONS")
print(f"{'='*65}")
print(f"\n{'Subject':<25} {'True':>8}  LR          SVM         RF")
print("-" * 72)

model_keys = list(MODELS.keys())
for subj in subjects:
    rows = {m: df_folds[(df_folds["model"]==m) & (df_folds["subject"]==subj)].iloc[0]
            for m in model_keys}
    y_t     = int(rows[model_keys[0]]["y_true"])
    group   = rows[model_keys[0]]["true_group"]
    n_right = sum(1 for m in model_keys if rows[m]["correct"])

    line = f"  {subj:<23} {group:>8}  "
    for m in model_keys:
        r     = rows[m]
        label = "ALS" if r["y_pred"]==1 else "HC"
        flag  = "✓" if r["correct"] else "✗"
        line += f"{label}{flag}({r['y_prob']:.2f})  "
    line += f"[{n_right}/{len(model_keys)}]"
    print(line)

# ─── Misclassified subjects ────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("MISCLASSIFIED SUBJECTS")
print(f"{'='*65}")

for model_name in model_keys:
    sub   = df_folds[df_folds["model"] == model_name]
    wrong = sub[~sub["correct"]]
    print(f"\n  {model_name} ({len(wrong)} errors):")
    for _, row in wrong.iterrows():
        pred = "ALS" if row["y_pred"]==1 else "HC"
        print(f"    {row['subject']:<25} True={row['true_group']:<8} "
              f"Pred={pred}  prob={row['y_prob']:.3f}")

# ─── Consistency analysis ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("CROSS-MODEL CONSISTENCY")
print(f"{'='*65}")

consistency = {}
for subj in subjects:
    preds = [
        int(df_folds[(df_folds["model"]==m) & (df_folds["subject"]==subj)].iloc[0]["y_pred"])
        for m in model_keys
    ]
    true_y = int(df_folds[df_folds["subject"]==subj].iloc[0]["y_true"])
    consistency[subj] = {
        "n_agree"     : sum(1 for p in preds if p == true_y),
        "all_correct" : all(p == true_y for p in preds),
        "all_wrong"   : all(p != true_y for p in preds),
        "split"       : not (all(p==preds[0] for p in preds)),
    }

all_correct = [s for s, d in consistency.items() if d["all_correct"]]
all_wrong   = [s for s, d in consistency.items() if d["all_wrong"]]
split_vote  = [s for s, d in consistency.items() if d["split"]]

print(f"\n  All 3 models correct   : {len(all_correct)} subjects")
for s in all_correct:
    g = df_folds[df_folds["subject"]==s].iloc[0]["true_group"]
    print(f"    {s} ({g})")

print(f"\n  All 3 models wrong     : {len(all_wrong)} subjects")
for s in all_wrong:
    g = df_folds[df_folds["subject"]==s].iloc[0]["true_group"]
    print(f"    {s} ({g})")

print(f"\n  Split vote (models disagree): {len(split_vote)} subjects")
for s in split_vote:
    g = df_folds[df_folds["subject"]==s].iloc[0]["true_group"]
    print(f"    {s} ({g})")

# ─── SHAP Feature Importance ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SHAP Feature Importance (Random Forest, full dataset)")
print(f"{'='*65}")

scaler_final = StandardScaler()
X_final      = scaler_final.fit_transform(np.nan_to_num(X_raw))

rf_final = RandomForestClassifier(
    n_estimators=500, max_depth=2, min_samples_leaf=3,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf_final.fit(X_final, y)

explainer   = shap.TreeExplainer(rf_final)
shap_values = explainer.shap_values(X_final)

# Fix: handle (22, 4, 2) shape — use [:, :, 1] for ALS class
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    sv_als = shap_values[:, :, 1]
elif isinstance(shap_values, list):
    sv_als = np.array(shap_values[1])
else:
    sv_als = np.array(shap_values)

sv_als = np.array(sv_als, dtype=float)
print(f"SHAP values shape: {sv_als.shape}")

mean_abs_shap = np.abs(sv_als).mean(axis=0)
shap_rank     = [int(i) for i in np.argsort(mean_abs_shap)[::-1]]

print(f"\n{'Rank':<6} {'Feature':<22} {'Mean |SHAP|':>12}  Direction")
print("-" * 58)

shap_summary = []
for rank, fi in enumerate(shap_rank):
    feat_name    = FEAT_NAMES[fi]
    mean_shap    = float(mean_abs_shap[fi])
    mean_raw     = float(sv_als[:, fi].mean())
    direction    = "ALS > HC" if mean_raw > 0 else "ALS < HC"
    print(f"  #{rank+1:<4} {feat_name:<20} {mean_shap:>12.4f}  {direction}")
    shap_summary.append({
        "rank"          : rank + 1,
        "feature"       : FEAT_COLS[fi],
        "feature_label" : feat_name,
        "mean_abs_shap" : round(mean_shap, 6),
        "mean_raw_shap" : round(mean_raw,  6),
        "direction"     : direction,
    })

print(f"\nRandom Forest Gini importances:")
for fi in np.argsort(rf_final.feature_importances_)[::-1]:
    print(f"  {FEAT_NAMES[fi]:<20} {rf_final.feature_importances_[fi]:.4f}")

# ─── Final summary ────────────────────────────────────────────────────────────
best = max(model_summary, key=lambda x: x["bal_acc"])

print(f"\n{'='*65}")
print("FINAL SUMMARY FOR PAPER")
print(f"{'='*65}")
print(f"""
Best model: {best['model']}
  Balanced Accuracy : {best['bal_acc']:.3f}
  AUC-ROC           : {best['auc']:.3f}
  Sensitivity       : {best['sensitivity']:.3f}  ({int(best['sensitivity']*n_als)}/{n_als} ALS)
  Specificity       : {best['specificity']:.3f}  ({int(best['specificity']*n_healthy)}/{n_healthy} HC)
  PPV               : {best['ppv']:.3f}
  NPV               : {best['npv']:.3f}

Top SHAP feature: {shap_summary[0]['feature_label']}
  Mean |SHAP| = {shap_summary[0]['mean_abs_shap']:.4f}
  Direction   : {shap_summary[0]['direction']} (compensatory hyperactivation)

Consistent across all 3 models: {len(all_correct)} subjects correctly classified
Hard cases (all 3 wrong):       {len(all_wrong)} subjects
""")

# ─── Save ─────────────────────────────────────────────────────────────────────
df_folds.to_csv(PHASE10_DIR / "loso_results.csv", index=False)
pd.DataFrame(model_summary).to_csv(PHASE10_DIR / "model_comparison.csv", index=False)
pd.DataFrame(shap_summary).to_csv(PHASE10_DIR / "shap_importance.csv", index=False)

shap_df = pd.DataFrame(sv_als, columns=FEAT_NAMES)
shap_df.insert(0, "subject", subjects)
shap_df.insert(1, "group",   groups)
shap_df.to_csv(PHASE10_DIR / "shap_values.csv", index=False)

# Subject predictions wide format
rows_wide = []
for subj in subjects:
    sub_rows = df_folds[df_folds["subject"] == subj]
    row = {"subject": subj,
           "true_group": sub_rows.iloc[0]["true_group"],
           "y_true": sub_rows.iloc[0]["y_true"]}
    for m in model_keys:
        mr = sub_rows[sub_rows["model"] == m].iloc[0]
        short = m.split()[0]
        row[f"pred_{short}"]    = mr["y_pred"]
        row[f"prob_{short}"]    = mr["y_prob"]
        row[f"correct_{short}"] = mr["correct"]
    row["n_models_correct"] = sum(row.get(f"correct_{m.split()[0]}", 0)
                                  for m in model_keys)
    rows_wide.append(row)

pd.DataFrame(rows_wide).to_csv(PHASE10_DIR / "subject_predictions.csv", index=False)

print(f"Outputs saved to: {PHASE10_DIR}")
print(f"  loso_results.csv        -> per-fold predictions")
print(f"  model_comparison.csv    -> aggregate metrics")
print(f"  shap_importance.csv     -> SHAP feature ranking")
print(f"  shap_values.csv         -> raw SHAP per subject")
print(f"  subject_predictions.csv -> per-subject all-model predictions")
