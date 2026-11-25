"""Bayesian (TPE, optuna) refit of CdA, Crr, drivetrain eff, extra load and REGEN efficiency on Days 1-3,
held-out Days 5-8. Pre-WSC only. Hand check first: at steady cruise (flat, |a|<0.05) the measured power is only
~10-30 % above car_config, so the energy gap lives in the dynamic part (accel / grade / regen), not in cruise drag.
Bounds are physical: aux <= 100 W, eff <= 0.95.  Objective = stage energy log-ratio^2 + 5-min window RMSE (normalised).
Writes refit_bayes.csv (best per seed), refit_bayes_trials.csv (all trials), fig_refit_bayes.png
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, optuna
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnose as dg
FIT = {(1, "stage1"), (2, "stage1"), (2, "stage2"), (3, "stage1"), (3, "stage2")}
WIN = 300.0
BND = dict(cda=(0.15, 0.35), crr=(0.004, 0.02), eff=(0.85, 0.95), aux=(0.0, 100.0), regen=(0.0, 0.8))

def p_el(s, p):
    pa, pr, pg, pc = dg.terms(s, dict(cda=p["cda"], crr=p["crr"]))
    pm = pa + pr + pg + pc
    return np.where(pm >= 0, pm / p["eff"], pm * p["regen"]) + p["aux"]

def prep(S):
    for s in S.values():
        w = (np.cumsum(s["dt"]) // WIN).astype(int); s["w5"] = w
    return S

def stage_E(s, p): return float((p_el(s, p) * s["dt"]).sum() / 3600)
def meas_E(s): return float((s["p_meas"] * s["dt"]).sum() / 3600)

def win_rmse(S, keys, p):
    r, wt = [], []
    for k in keys:
        s = S[k]; pe = p_el(s, p)
        for u in np.unique(s["w5"]):
            m = s["w5"] == u; d = s["dt"][m]
            if d.sum() < 120: continue
            r.append(np.average(pe[m], weights=d) - np.average(s["p_meas"][m], weights=d)); wt.append(d.sum())
    r, wt = np.array(r), np.array(wt)
    return float(np.sqrt(np.average(r ** 2, weights=wt)))

def score(S, keys, p):
    le = np.mean([np.log(stage_E(S[k], p) / meas_E(S[k])) ** 2 for k in keys])
    return le + (win_rmse(S, keys, p) / 600.0) ** 2

def ratio(S, keys, p): return sum(stage_E(S[k], p) for k in keys) / sum(meas_E(S[k]) for k in keys)

if __name__ == "__main__":
    S = prep(dg.build_samples()); kf = [k for k in S if k in FIT]; kt = [k for k in S if k not in FIT]
    base = dict(cda=dg.DEFAULT["cda"], crr=dg.DEFAULT["crr"], eff=dg.DEFAULT["eff"], aux=dg.DEFAULT["aux"], regen=dg.CAR.regen_eff)
    rows = [dict(run="car_config", **base, score_fit=score(S, kf, base), rmse5_test=win_rmse(S, kt, base),
                 ratio_fit=ratio(S, kf, base), ratio_test=ratio(S, kt, base))]
    trials = []
    for seed in (0, 1, 2):
        st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=60), direction="minimize")
        def obj(t):
            p = {k: t.suggest_float(k, *BND[k]) for k in BND}
            return score(S, kf, p)
        st.optimize(obj, n_trials=500)
        p = st.best_params
        rows.append(dict(run=f"TPE seed {seed}", **p, score_fit=st.best_value, rmse5_test=win_rmse(S, kt, p),
                         ratio_fit=ratio(S, kf, p), ratio_test=ratio(S, kt, p)))
        for t in st.trials: trials.append(dict(seed=seed, value=t.value, **t.params))
    R = pd.DataFrame(rows); R.round(4).to_csv(HERE/"refit_bayes.csv", index=False)
    Tr = pd.DataFrame(trials); Tr.round(5).to_csv(HERE/"refit_bayes_trials.csv", index=False)
    print(R.round(4).to_string(index=False))
    top = Tr.nsmallest(int(len(Tr) * .05), "value")
    print("\ntop 5% trials (mean +- std):"); print(top[list(BND)].agg(["mean", "std"]).round(4).to_string())
    fig, ax = plt.subplots(1, len(BND), figsize=(13, 3))
    for a, k in zip(ax, BND):
        a.scatter(Tr[k], Tr.value, s=4, alpha=.4); a.set_xlabel(k); a.set_ylim(Tr.value.min(), np.percentile(Tr.value, 70))
    ax[0].set_ylabel("objective"); plt.tight_layout(); plt.savefig(HERE/"fig_refit_bayes.png", dpi=170)
