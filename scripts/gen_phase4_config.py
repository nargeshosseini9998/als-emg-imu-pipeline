"""Generate configs/phase4_episodes.yaml by re-executing the per-trial override
logic of 04_exceptions.ipynb cells 5, 6, 7 (tremor variant) so that no value is
hand-copied."""
import yaml

TRIALS6 = ["drinking_exo","drinking_noexo","lifting_exo","lifting_noexo","pick&place_high_exo","pick&place_high_noexo"]
TASK_TREMOR = {"drinking":{"merge_gap_s":0.35},"lifting":{"merge_gap_s":0.45},"pick&place":{"merge_gap_s":0.50},"default":{"merge_gap_s":0.45}}
def task_key(tid): return next((k for k in TASK_TREMOR if k in tid.lower()), "default")

def als13(tid):
    tk=task_key(tid); k_off=6.2; mg=TASK_TREMOR[tk]["merge_gap_s"]; k_on=4.8
    if tk in ["drinking","lifting"]: mg=1.2; k_off=5.0
    return dict(k_on=k_on,k_off=k_off,merge_gap_s=mg)

def als12(tid):
    tk=task_key(tid); k_off=6.2; mg=TASK_TREMOR[tk]["merge_gap_s"]; k_on=4.8; t=tid.lower()
    if "noexo" in t: k_off=7.0; mg=0.3
    elif "exo" in t: k_off=5.0; mg=0.8
    if tk=="pick&place" and "high_exo" in t: k_off=9.5; mg=0.15; k_on=5.2
    if tk=="pick&place" and "high" in t and "noexo" in t: k_off=8.5; mg=0.25; k_on=5.2
    if tk=="lifting" and "noexo" in t: mg=1.0; k_off=9.5
    if tk=="lifting": mg=max(mg,0.55)
    if tk=="drinking" and "noexo" in t: k_off=8.5
    if tk=="lifting" and "exo" in t: k_on=6.2; k_off=5.8; mg=0.6
    if tk=="lifting" and "noexo" in t: k_off=4.5; mg=1.5
    return dict(k_on=k_on,k_off=k_off,merge_gap_s=mg)

def als11(tid):
    tk=task_key(tid); k_off=6.2; mg=TASK_TREMOR[tk]["merge_gap_s"]; k_on=4.8; t=tid.lower()
    if "noexo" in t: k_off=7.0; mg=0.3
    elif "exo" in t: k_off=5.0; mg=0.8
    if tk=="pick&place" and "high_exo" in t: k_off=6.5; mg=0.5
    if tk=="lifting" and "noexo" in t: mg=1.0; k_off=9.5
    if tk=="drinking" and "exo" in t: k_off=9.5; mg=0.20
    if tk=="drinking" and "noexo" in t: k_off=10; mg=0.15
    if tk=="lifting" and "exo" in t: k_off=6.5; mg=0.5
    return dict(k_on=k_on,k_off=k_off,merge_gap_s=mg)

def make(fn, note):
    return {"variant":"tremor","selection_basis":note,"trial_overrides":{tid:fn(tid) for tid in TRIALS6}}

variants = {
 "standard": {
   "description":"Double-threshold MAD detector, k_on 5.0 / k_off 3.5 (04_events.ipynb).",
   "k_on":5.0,"k_off":3.5,"pad_s":0.15,"peak_multiplier":2.5,"secondary_merge_gap":0.8,
   "alpha_acc":1.0,"smooth_kernel":1,"adjust_rule":"standard","third_merge_gap":1.2,
   "min_merge_gap_all_tasks":2.0,"min_secondary_gap_all_tasks":4.0,
   "min_valid_duration_s":3.5,"min_valid_peak_gate":50.0,
   "task_settings":{"lifting":{"min_duration_s":1.8,"merge_gap_s":0.35,"min_peak_gate":100.0},
                    "drinking":{"min_duration_s":1.2,"merge_gap_s":0.30,"min_peak_gate":60.0},
                    "pick&place":{"min_duration_s":2.2,"merge_gap_s":1.20,"min_peak_gate":75.0},
                    "default":{"min_duration_s":1.5,"merge_gap_s":0.40,"min_peak_gate":70.0}},
   "emg_fallback":{"k_on":2.0,"k_off":1.0,"peak_multiplier":1.2,"min_duration_factor":0.6,
                   "double_min_duration_factor":True,"merge_gap_factor":2.0,"secondary_merge_gap":3.5},
   "emg_lowvar_mode":"x5"},
 "standard_v1": {
   "description":"Earlier revision of `standard` (04_events_als3.ipynb): no 2 s / 4 s merge floor and no peak-gate validity filter.",
   "k_on":5.0,"k_off":3.5,"pad_s":0.15,"peak_multiplier":2.5,"secondary_merge_gap":0.8,
   "alpha_acc":1.0,"smooth_kernel":1,"adjust_rule":"standard","third_merge_gap":1.2,
   "min_merge_gap_all_tasks":None,"min_secondary_gap_all_tasks":None,
   "min_valid_duration_s":3.5,"min_valid_peak_gate":None,
   "task_settings":{"lifting":{"min_duration_s":1.8,"merge_gap_s":0.35,"min_peak_gate":100.0},
                    "drinking":{"min_duration_s":1.2,"merge_gap_s":0.30,"min_peak_gate":60.0},
                    "pick&place":{"min_duration_s":2.2,"merge_gap_s":1.20,"min_peak_gate":75.0},
                    "default":{"min_duration_s":1.5,"merge_gap_s":0.40,"min_peak_gate":70.0}},
   "emg_fallback":{"k_on":2.0,"k_off":1.0,"peak_multiplier":1.2,"min_duration_factor":0.6,
                   "double_min_duration_factor":True,"merge_gap_factor":2.0,"secondary_merge_gap":3.5},
   "emg_lowvar_mode":"x5"},
 "standard_als": {
   "description":"`standard` re-tuned for fragmented ALS signals (04_exceptions.ipynb cell 0): larger merge gaps, extra relaxation for no-exo trials, weaker validity filter, low-variance gate boost, stronger automatic threshold reduction.",
   "k_on":5.0,"k_off":3.5,"pad_s":0.15,"peak_multiplier":2.5,"secondary_merge_gap":0.8,
   "alpha_acc":1.0,"smooth_kernel":1,"adjust_rule":"standard_als","third_merge_gap":1.2,
   "min_merge_gap_all_tasks":1.2,"min_secondary_gap_all_tasks":3.5,
   "noexo_rule":{"merge_gap_factor":1.6,"min_secondary_gap":4.0,"min_duration_factor":0.75},
   "lowvar_boost_imu":{"var_threshold":1.0e-4,"factor":1.5},
   "min_valid_duration_s":2.0,"min_valid_peak_gate":40.0,
   "task_settings":{"lifting":{"min_duration_s":1.8,"merge_gap_s":0.50,"min_peak_gate":100.0},
                    "drinking":{"min_duration_s":1.2,"merge_gap_s":1.00,"min_peak_gate":60.0},
                    "pick&place":{"min_duration_s":2.2,"merge_gap_s":1.50,"min_peak_gate":75.0},
                    "default":{"min_duration_s":1.5,"merge_gap_s":0.60,"min_peak_gate":70.0}},
   "emg_fallback":{"k_on":1.8,"k_off":0.9,"peak_multiplier":1.1,"min_duration_factor":0.6,
                   "double_min_duration_factor":False,"merge_gap_factor":2.5,"secondary_merge_gap":4.0},
   "emg_lowvar_mode":"x5"},
 "weak": {
   "description":"Weak-signal adaptation (04_exceptions.ipynb cells 9-10): lower thresholds, median-filtered gate (kernel 7), accelerometer boost, TKEO on EMG-only trials, relaxed validity filter. Applied to every trial of the subject.",
   "k_on":5.0,"k_off":3.5,"pad_s":0.15,"peak_multiplier":2.5,"secondary_merge_gap":0.8,
   "alpha_acc":1.0,"smooth_kernel":5,"adjust_rule":"weak","third_merge_gap":None,
   "min_merge_gap_all_tasks":None,"min_secondary_gap_all_tasks":None,
   "min_valid_duration_s":3.5,"min_valid_peak_gate":50.0,
   "task_settings":{"lifting":{"min_duration_s":1.8,"merge_gap_s":0.35,"min_peak_gate":100.0},
                    "drinking":{"min_duration_s":1.2,"merge_gap_s":0.30,"min_peak_gate":60.0},
                    "pick&place":{"min_duration_s":2.2,"merge_gap_s":1.20,"min_peak_gate":75.0},
                    "default":{"min_duration_s":1.5,"merge_gap_s":0.40,"min_peak_gate":70.0}},
   "weak_settings":{"gate_95p_threshold":50.0,"k_on":2.8,"k_off":1.5,"peak_multiplier":1.4,
                    "min_duration_factor":0.65,"merge_gap_factor":2.2,"secondary_merge_gap":5.5,
                    "alpha_acc_boost":1.8,"smooth_kernel":7,"min_valid_duration_s":2.0,"min_valid_peak_gate":35.0},
   "emg_lowvar_mode":"p95"},
 "tremor": {
   "description":"Tremor / irregular-burst detector (04_exceptions.ipynb cells 5-7): baseline from lowest 15 % of the gate, inverted hysteresis (k_off > k_on), thr_on capped at the median, single merge stage, absolute peak filter. Per-trial parameters were set by visual inspection and are listed under each subject.",
   "k_on":4.8,"k_off":6.2,"pad_s":0.15,"default_min_duration_s":1.8,
   "baseline_percentile":15,"thr_off_multiplier":1.3,"min_raw_episode_s":0.3,
   "gate_lp_imu_hz":7.0,"gate_lp_emg_hz":3.0,
   "min_valid_duration_s":3.5,"min_valid_peak_gate":28.0,
   "task_settings":{"drinking":{"merge_gap_s":0.35,"min_peak_gate":28.0},
                    "lifting":{"merge_gap_s":0.45,"min_peak_gate":35.0},
                    "pick&place":{"merge_gap_s":0.50,"min_peak_gate":32.0},
                    "default":{"merge_gap_s":0.45,"min_peak_gate":30.0}}},
}

STD = "Default detector; episodes judged acceptable on visual inspection of the gate plots."
subjects = {}
for h in range(1,9): subjects[f"Healthy_Subject_{h}"]={"variant":"standard","selection_basis":STD}
for a in [1,2,5,15,16]: subjects[f"ALS_Subject_{a}"]={"variant":"standard","selection_basis":STD}
# The 'keep at most the 4 longest episodes per trial' rule was added to the standard detector
# midway through 04_events.ipynb (cell 6) and therefore applied only to subjects run after it.
for s_ in [f"Healthy_Subject_{h}" for h in (6,7,8)] + [f"ALS_Subject_{a}" for a in (1,2,5,15,16)]:
    subjects[s_]["overrides"]={"max_episodes_per_trial":4}
subjects["ALS_Subject_14"]={"variant":"standard",
  "selection_basis":"Standard detector; drinking and lifting trials re-tuned on visual inspection (04_exceptions.ipynb cell 8). Both pick&place trials judged corrupted and dropped (see manual_curation.yaml). Note: the exo-condition test used substring matching, so the drinking/lifting overrides applied to BOTH exo and noexo trials.",
  "trial_overrides":{
    "drinking_exo":   {"k_on":3.0,"k_off":6.5,"merge_gap_s":4.5,"secondary_merge_gap":6.5},
    "drinking_noexo": {"k_on":3.0,"k_off":6.5,"merge_gap_s":4.5,"secondary_merge_gap":6.5},
    "lifting_exo":    {"k_on":6.5,"k_off":1.5,"merge_gap_s":0.4,"secondary_merge_gap":0.8},
    "lifting_noexo":  {"k_on":6.5,"k_off":1.5,"merge_gap_s":0.4,"secondary_merge_gap":0.8}}}
subjects["ALS_Subject_3"]={"variant":"standard_v1","selection_basis":"Processed in a separate notebook with the earlier revision of the standard detector (two trials had no IMU and used the EMG-fallback gate)."}
for a in [8,9,10]: subjects[f"ALS_Subject_{a}"]={"variant":"standard_als","selection_basis":"Standard detector over-segmented the fragmented ALS gate; re-tuned merge/validity parameters chosen on visual inspection."}
subjects["ALS_Subject_13"]=make(als13,"Tremor-like bursts: standard hysteresis chattered; inverted-hysteresis detector with task-level merge settings chosen on visual inspection.")
subjects["ALS_Subject_12"]=make(als12,"Tremor-like bursts; per-trial k_on/k_off/merge_gap tuned on visual inspection to fix over-/under-merging in specific trials.")
subjects["ALS_Subject_12"]["overrides"]={"prune_rule":{"task":"lifting","min_duration_s":2.0,"max_episodes":3,
    "second_min_duration_s":2.5,"second_min_peak":120.0,
    "note":"Post-filter for lifting trials (exo AND noexo, substring match): drop episodes < 2 s; if > 3 remain drop the weakest; if >= 3 remain and the 2nd is short/weak drop it (04_exceptions.ipynb cell 6)."}}
subjects["ALS_Subject_11"]=make(als11,"Tremor-like bursts; per-trial parameters tuned on visual inspection. lifting_exo episodes were discarded entirely (see manual_curation.yaml).")
for a in [7,4]: subjects[f"ALS_Subject_{a}"]={"variant":"weak","selection_basis":"Low-amplitude gate (P95 of the gate 29-110 deg/s); weak-signal adaptation applied to all trials."}

order = [f"Healthy_Subject_{i}" for i in range(1,9)] + [f"ALS_Subject_{i}" for i in [1,2,3,4,5,7,8,9,10,11,12,13,14,15,16]]
subjects = {k:subjects[k] for k in order}

doc = {
 "_about": ("Phase 4 (movement-episode detection) configuration. The detector `variant` and every "
            "override below were chosen per subject by visual inspection of the gate signal and the "
            "resulting episodes, NOT by an automatic signal-quality rule. Values are exact copies of the "
            "notebooks used to produce the thesis results (04_events.ipynb, 04_events_als3.ipynb, "
            "04_exceptions.ipynb). Trial overrides for the tremor subjects were derived by re-executing the "
            "original if/elif override chains (see scripts/gen_phase4_config.py)."),
 "variants": variants, "subjects": subjects}
import pathlib; out = pathlib.Path(__file__).resolve().parents[1] / "configs" / "phase4_episodes.yaml"
with open(out,"w") as f: yaml.safe_dump(doc,f,sort_keys=False,allow_unicode=True,width=100)
print(open(out).read()[:1500])
for s in ["ALS_Subject_12","ALS_Subject_11"]:
    print(s, subjects[s]["trial_overrides"])
