"""Global (differential-evolution, several seeds) refit of CdA, Crr, motor eff, extra load on Days 1-3,
physically bounded, tested on held-out Days 5-8. Pre-WSC only; Model/ is not touched.
Writes refit_d123.csv, refit_d123_landscape.csv, fig_refit_d123.png"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnose as dg

FIT = {(1, "stage1"), (2, "stage1"), (2, "stage2"), (3, "stage1"), (3, "stage2")}
NAMES = ["cda", "crr", "eff", "aux"]
B = [(0.15, 0.35), (0.004, 0.012), (0.85, 0.97), (0.0, 250.0)]   # physically plausible, not the wide diagnose bounds

def P(x):
    p = dict(dg.DEFAULT); p.update(dict(zip(NAMES, x))); return p

def ratio(S, keys, p):
    pred = sum(dg.stage_energy(S[k], p) for k in keys)
    meas = sum(float((S[k]["p_meas"]*S[k]["dt"]).sum()/3600) for k in keys)
    return pred/meas

if __name__ == "__main__":
    S = dg.build_samples()
    kf = [k for k in S if k in FIT]; kt = [k for k in S if k not in FIT]
    f = lambda x: dg._rms(S, kf, P(x))
    rows = []
    p0 = dict(dg.DEFAULT)
    rows.append(dict(run="car_config", **{n: p0[n] for n in NAMES}, rmse_fit=dg._rms(S, kf, p0), rmse_test=dg._rms(S, kt, p0),
                     ratio_fit=ratio(S, kf, p0), ratio_test=ratio(S, kt, p0)))
    for seed in (0, 1, 2, 3):
        r = differential_evolution(f, B, seed=seed, tol=1e-6, maxiter=60, popsize=12, polish=True)
        p = P(r.x)
        rows.append(dict(run=f"DE seed {seed}", **{n: p[n] for n in NAMES}, rmse_fit=r.fun, rmse_test=dg._rms(S, kt, p),
                         ratio_fit=ratio(S, kf, p), ratio_test=ratio(S, kt, p)))
    R = pd.DataFrame(rows); R.round(4).to_csv(HERE/"refit_d123.csv", index=False)
    print(R.round(4).to_string(index=False))
    # landscape: best RMSE over (cda, crr) with eff, aux free-fit analytically skipped -> eff fixed at 0.95, aux optimised on a grid
    cdas = np.linspace(.15, .35, 11); crrs = np.linspace(.004, .012, 9); L = []
    for c in cdas:
        for rr in crrs:
            best = min((dg._rms(S, kf, P([c, rr, .95, a])), a) for a in (0, 50, 100, 150, 200, 250))
            L.append(dict(cda=c, crr=rr, rmse=best[0], aux=best[1]))
    L = pd.DataFrame(L); L.round(4).to_csv(HERE/"refit_d123_landscape.csv", index=False)
    g = L.pivot(index="crr", columns="cda", values="rmse")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.imshow(g.values, origin="lower", aspect="auto", extent=[cdas[0], cdas[-1], crrs[0], crrs[-1]], cmap="viridis_r")
    ax.set_xlabel("CdA (m2)"); ax.set_ylabel("Crr"); plt.colorbar(im, label="fit RMSE (W), eff 0.95, aux optimised")
    ax.set_title("Days 1-3 fit landscape: the valley, not a point", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE/"fig_refit_d123.png", dpi=200)
    print("landscape best:", L.loc[L.rmse.idxmin()].round(4).to_dict(), " worst/best RMSE %.0f/%.0f" % (L.rmse.max(), L.rmse.min()))
