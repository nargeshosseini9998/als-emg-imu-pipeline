# alsexo/stats/step_f2_bayes_deltoid.py
# Phase 9 — extracted verbatim from 09_complete.ipynb (cell 25); paths replaced by alsexo.paths.
# NOTE: runs on import (script-style); execute via the CLI or `python -m alsexo.stats.step_f2_bayes_deltoid`.

from alsexo.paths import (PROJECT_ROOT, PROCESSED_ROOT, FIG_ROOT, PHASE2B_DIR, PHASE3_NORM_DIR,
    PHASE5_DIR, PHASE6_DIR, PHASE7_DIR, PHASE8_DIR, PHASE9_EFA_DIR, PHASE9_STATS, PHASE10_DIR)

# =========================================
# Step 9.F — Bayesian complement: exoskeleton effect on the ALS deltoid
#
# WHY THIS ANALYSIS
#   The frequentist test of the exoskeleton's effect on the deltoid in ALS gave
#   a LARGE effect (deltoid reduced ~36-45%, r=0.67) but p=0.054 — just short of
#   the 0.05 threshold. A frequentist framework can only call this "not
#   significant", which is easily mis-read as "no effect". That is exactly the
#   situation Bayesian estimation handles better: instead of a binary verdict, it
#   returns the PROBABILITY that the exoskeleton reduces effort, and a credible
#   interval for how large the reduction is.
#
# THE MODEL (deliberately simple and transparent — defensible line by line)
#   Data: within-subject PAIRED differences d_i = log(EMG_exo) - log(EMG_noexo)
#         for each ALS patient, for the deltoid (drinking task, the strongest).
#         Working in log space makes the effect multiplicative and symmetric, and
#         makes a Normal likelihood appropriate.
#
#   Likelihood:  d_i ~ Normal(mu, sigma)
#       mu    = the true mean log-change with the exoskeleton (what we want)
#       sigma = between-patient variability
#
#   PRIORS (weakly-informative, justified below — NOT tuned to get a result):
#       mu    ~ Normal(0, 0.5)
#             Centered at 0 = "no effect" (a SKEPTICAL prior — it does not assume
#             the exo works; if anything it biases AGAINST finding an effect).
#             SD 0.5 in log units allows changes up to ~e^1 ≈ ±170% comfortably,
#             so it is only weakly informative — the data dominates.
#       sigma ~ Half-Normal(0.5)
#             Positive-only, weakly-informative scale for the spread.
#
#   We do NOT use a strong prior favouring the exoskeleton. Using a skeptical
#   prior centered at "no effect" is the conservative, defensible choice: any
#   posterior probability of benefit is obtained DESPITE a prior that starts at
#   zero, not because the prior assumed it.
#
# INFERENCE
#   With a Normal likelihood and these priors, we sample the posterior by a
#   simple, exact-enough grid + Monte Carlo scheme (no MCMC black box needed for
#   a one-parameter-of-interest model), so every number is reproducible and
#   inspectable.
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

PROCESSED    = PROJECT_ROOT / "data" / "processed"
PHASE6_DIR   = PROCESSED / "phase_06_features"
FIG_DIR      = PROJECT_ROOT / "figures"
OUT_DIR      = PROCESSED / "phase_09_stats" / "step_F_bayes"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_SUBJECTS = ["ALS_Subject_15"]
rng = np.random.default_rng(20260101)

NAVY="#13294B"; TEAL="#0E7C86"; GOLD="#E0A526"; RED="#A23B3B"; GREY="#6B7C8C"

# ── Config: which muscle / task / group to analyse ────────────────────────────
MUSCLE = "Deltoid"      # the muscle with the borderline ALS result
TASK   = "drinking"     # strongest ALS deltoid effect in the frequentist analysis
GROUP  = "ALS"

def map_group(name):
    n=str(name).lower()
    return "ALS" if "als" in n else ("Healthy" if "healthy" in n else None)
def get_task(tid):
    t=str(tid).lower()
    return ("drinking" if "drinking" in t else "lifting" if "lifting" in t
            else "pick_place" if "pick" in t else "other")
def get_cond(tid):
    t=str(tid).lower()
    return "noexo" if "noexo" in t else ("exo" if "exo" in t else "unknown")

# ── Build paired log-differences for the chosen muscle/task/group ─────────────
col = f"EMG_{MUSCLE}_RMS"
rows=[]
for fp in sorted(PHASE6_DIR.glob("*__features.parquet")):
    subj = fp.name.split("__features.parquet")[0]
    if subj in ARTIFACT_SUBJECTS: continue
    if map_group(subj) != GROUP: continue
    d = pd.read_parquet(fp)
    if col not in d.columns: continue
    d["task"]=d["trial_id"].apply(get_task); d["cond"]=d["trial_id"].apply(get_cond)
    sub = d[d["task"]==TASK]
    exo   = sub[sub["cond"]=="exo"][col].median()
    noexo = sub[sub["cond"]=="noexo"][col].median()
    if np.isfinite(exo) and np.isfinite(noexo) and exo>0 and noexo>0:
        rows.append({"subject":subj, "exo":exo, "noexo":noexo,
                     "d_log": np.log(exo) - np.log(noexo)})
paired = pd.DataFrame(rows)
d = paired["d_log"].to_numpy()
n = len(d)
print("="*70)
print(f"Bayesian exo effect — {GROUP} · {MUSCLE} · {TASK}")
print("="*70)
print(f"Paired patients: n = {n}")
print(f"Observed mean log-change: {d.mean():+.3f}  (= {100*(np.exp(d.mean())-1):+.1f}% median effect)")
print(f"Observed SD: {d.std(ddof=1):.3f}\n")

# ── Bayesian inference (grid over mu, sigma; weakly-informative priors) ───────
# Priors
def log_prior_mu(mu):     return -0.5*(mu/0.5)**2                 # Normal(0,0.5)
def log_prior_sigma(sg):  return np.where(sg>0, -0.5*(sg/0.5)**2, -np.inf)  # HalfNormal(0.5)

mu_grid    = np.linspace(-1.5, 1.5, 601)
sigma_grid = np.linspace(0.01, 2.0, 400)
MU, SG = np.meshgrid(mu_grid, sigma_grid, indexing="ij")

# log-likelihood of the paired data under Normal(mu, sigma)
ll = np.zeros_like(MU)
for di in d:
    ll += -np.log(SG) - 0.5*((di-MU)/SG)**2
log_post = ll + log_prior_mu(MU) + log_prior_sigma(SG)
log_post -= log_post.max()
post = np.exp(log_post)
post /= post.sum()

# marginal posterior for mu
post_mu = post.sum(axis=1); post_mu /= post_mu.sum()

# posterior summaries via sampling from the grid
flat = post.ravel()
idx = rng.choice(flat.size, size=200000, p=flat)
mu_s = MU.ravel()[idx]; sg_s = SG.ravel()[idx]
# jitter within grid cells for smoothness
mu_s = mu_s + rng.uniform(-1,1,mu_s.size)*(mu_grid[1]-mu_grid[0])/2

mu_mean = mu_s.mean()
ci = np.percentile(mu_s, [2.5, 97.5])
p_reduce = float((mu_s < 0).mean())          # P(exo reduces effort)
p_reduce10 = float((mu_s < np.log(0.90)).mean())  # P(reduction > 10%)
p_reduce20 = float((mu_s < np.log(0.80)).mean())  # P(reduction > 20%)

def pct(x): return 100*(np.exp(x)-1)
print("POSTERIOR (what the data + skeptical prior imply):")
print(f"  Mean effect: {mu_mean:+.3f} log  =  {pct(mu_mean):+.1f}% change")
print(f"  95% credible interval: {pct(ci[0]):+.1f}% to {pct(ci[1]):+.1f}%")
print(f"\n  P(exoskeleton REDUCES deltoid effort)         = {p_reduce*100:.1f}%")
print(f"  P(reduction is at least 10%)                  = {p_reduce10*100:.1f}%")
print(f"  P(reduction is at least 20%)                  = {p_reduce20*100:.1f}%")
print(f"\nContrast with frequentist: p=0.054 -> 'not significant'.")
print(f"Bayesian: ~{p_reduce*100:.0f}% probability the exoskeleton reduces effort.")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                               gridspec_kw={"width_ratios":[1,1.1]})

# (a) paired lines
for _, r in paired.iterrows():
    ax1.plot([0,1], [r["noexo"], r["exo"]], color=GREY, alpha=0.55, lw=1.4,
             marker="o", markersize=5, markerfacecolor="white")
ax1.plot([0,1], [paired["noexo"].median(), paired["exo"].median()],
         color=NAVY, lw=3, marker="o", markersize=9, label="group median", zorder=5)
ax1.set_xlim(-0.3,1.3); ax1.set_xticks([0,1]); ax1.set_xticklabels(["no-EXO","EXO"], fontsize=12)
ax1.set_ylabel(f"{MUSCLE} RMS (V)", fontsize=11)
ax1.set_title(f"Each ALS patient, {TASK}\n(deltoid effort with vs without exo)",
              fontsize=12, fontweight="bold", color=NAVY)
ax1.spines[["top","right"]].set_visible(False)
ax1.legend(fontsize=10)

# (b) posterior of the effect (in %)
mu_pct = pct(mu_grid)
ax2.fill_between(mu_pct, post_mu, color=TEAL, alpha=0.25)
ax2.plot(mu_pct, post_mu, color=TEAL, lw=2)
ax2.axvline(0, color=RED, lw=1.8, ls="--", label="no effect")
ax2.axvline(pct(mu_mean), color=NAVY, lw=2, label=f"mean {pct(mu_mean):+.0f}%")
# shade the reduction region
mask = mu_pct < 0
ax2.fill_between(mu_pct[mask], post_mu[mask], color=TEAL, alpha=0.45)
ax2.set_xlabel("Exoskeleton effect on deltoid effort (%)", fontsize=11)
ax2.set_ylabel("posterior density", fontsize=11)
ax2.set_title(f"Posterior: P(reduces effort) = {p_reduce*100:.0f}%\n"
              f"95% CrI {pct(ci[0]):+.0f}% to {pct(ci[1]):+.0f}%",
              fontsize=12, fontweight="bold", color=NAVY)
ax2.set_yticks([])
ax2.spines[["top","right","left"]].set_visible(False)
ax2.legend(fontsize=10, loc="upper right")
ax2.annotate("reduction\n(exo helps)", xy=(pct(ci[0])*0.6, post_mu.max()*0.35),
             fontsize=9.5, color=TEAL, ha="center", fontweight="bold")

fig.suptitle("Bayesian complement — exoskeleton effect on the ALS deltoid (drinking)",
             fontsize=13.5, fontweight="bold", color=NAVY, y=1.03)
plt.tight_layout()
out = FIG_DIR / "phase9F_bayes_deltoid_ALS.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.show()
print(f"\nSaved figure -> {out}")

# save summary
pd.DataFrame([{
    "muscle":MUSCLE,"task":TASK,"group":GROUP,"n":n,
    "obs_mean_log":d.mean(),"obs_pct":pct(d.mean()),
    "post_mean_pct":pct(mu_mean),"crI_low_pct":pct(ci[0]),"crI_high_pct":pct(ci[1]),
    "P_reduce":p_reduce,"P_reduce_gt10":p_reduce10,"P_reduce_gt20":p_reduce20,
}]).to_csv(OUT_DIR / "bayes_deltoid_ALS_summary.csv", index=False)
print(f"Saved summary -> {OUT_DIR/'bayes_deltoid_ALS_summary.csv'}")
