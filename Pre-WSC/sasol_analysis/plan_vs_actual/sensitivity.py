"""
Parameter sensitivity of the strategy (proposal Week 15), with the unchanged model.

Baseline = the effective planner parameters identified from the Sasol telemetry (separate.py; solar 0.17,
solar_check.py). Each parameter is moved one at a time by -10 % and +10 % (regen 0.5 / 0.9) and the
planner's single-day solve (resolve-intraday) is re-run for a fixed day, start SOC and loop count.
Outputs: end-of-day SOC (and the battery energy it represents) and finish time.
The GA inside the solver is seeded (numpy default_rng(None) -> fixed seed) so differences come from the
parameter, not from optimiser randomness. Model code is not changed: CarState instances get the values
after construction, as in run_refined.py.

Second, optimiser-free check: the change in drive energy on the actual Day 1-3 driving (energy-balance windows,
separate.py, real air) for +-10 % of each parameter, and the change in the day's net battery energy when
array output is +-10 % (measured solar vs measured motor energy, actuals_by_day.csv).

usage: python sensitivity.py   (writes sensitivity.csv, sensitivity_energy.csv, fig_sensitivity.png)
"""
import glob
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[2] / "Model"
DAY, SOC, TIME = 4, 90, "08:00"
BASE = dict(motor_eff=0.773, crr=0.0103, cda_m2=0.126, p_idle_w=9.2, array_efficiency=0.17,
            mass_kg=336.5, regen_eff=0.70)
CASES = {"motor_eff": (0.9, 1.1), "crr": (0.9, 1.1), "cda_m2": (0.9, 1.1), "array_efficiency": (0.9, 1.1),
         "mass_kg": (0.9, 1.1), "regen_eff": (0.5 / 0.7, 0.9 / 0.7)}
LABEL = {"motor_eff": "drivetrain efficiency", "crr": "Crr", "cda_m2": "CdA", "array_efficiency": "array efficiency",
         "mass_kg": "mass", "regen_eff": "regen efficiency (0.5 / 0.9)"}


def child(params_json):
    """run one intraday solve with the given car values; print the result JSON path."""
    params = json.loads(params_json)
    os.chdir(MODEL)
    sys.path.insert(0, str(MODEL))
    import numpy as np
    orig_rng = np.random.default_rng
    np.random.default_rng = lambda seed=None: orig_rng(12345 if seed is None else seed)
    import configs.car_config as cc
    orig_init = cc.CarState.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        for key, val in params.items():
            object.__setattr__(self, key, val)

    cc.CarState.__init__ = init
    before = set(glob.glob("output/strategy_*intraday*.json"))
    sys.argv = ["optimizers.trust_region", "resolve-intraday", "--day", str(DAY), "--soc", str(SOC),
                "--time", TIME, "--dist-done", "0", "--variant", "aryaman"]
    os.environ["AGNIRATH_TIER2"] = "serial"
    runpy.run_module("optimizers.trust_region", run_name="__main__", alter_sys=True)
    new = sorted(set(glob.glob("output/strategy_*intraday*.json")) - before)
    print("RESULT_FILE", os.path.abspath(new[-1]))


def run_case(params):
    sys.path.insert(0, str(HERE))
    import patched_run as pr
    r = pr.run_intraday(params, DAY, SOC, TIME, "aryaman")
    if "error" in r:
        raise RuntimeError(r["error"])
    return r


def main():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(MODEL))
    from configs.car_config import default_car
    wh_per_pct = default_car().battery_nominal_wh / 100.0
    rows = []
    b = run_case(BASE)
    rows.append(dict(parameter="baseline", value=None, end_soc=b["end_soc_pct"], eta=b["eta"], km=b["remaining_km"],
                     feasible=b["feasible"]))
    print("baseline", rows[-1])
    for k, (lo, hi) in CASES.items():
        for f in (lo, hi):
            p = dict(BASE); p[k] = BASE[k] * f
            r = run_case(p)
            rows.append(dict(parameter=k, value=p[k], factor=f, end_soc=r["end_soc_pct"], eta=r["eta"],
                             km=r["remaining_km"], feasible=r["feasible"]))
            print(rows[-1])
    R = pd.DataFrame(rows)
    R["d_soc_pct"] = R.end_soc - b["end_soc_pct"]
    R["d_energy_wh"] = R.d_soc_pct * wh_per_pct
    to_min = lambda e: int(e[:2]) * 60 + int(e[3:5])
    R["d_finish_min"] = R.eta.map(to_min) - to_min(b["eta"])
    R.round(3).to_csv(HERE / "sensitivity.csv", index=False)
    E = energy_elasticity()
    E.round(2).to_csv(HERE / "sensitivity_energy.csv", index=False)
    print(E.round(2).to_string(index=False))
    print(R.round(2).to_string(index=False))
    T = R[R.parameter != "baseline"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    order = T.groupby("parameter").d_finish_min.apply(lambda s_: s_.abs().max()).sort_values().index
    for i, k in enumerate(order):
        for f, v in T[T.parameter == k][["factor", "d_finish_min"]].to_numpy():
            ax[0].barh(i, v, color="#c2542d" if v > 0 else "#3b6ea0", alpha=.85)
            ax[0].text(v + (1 if v >= 0 else -1), i, "-10%" if f < 1 else "+10%", va="center",
                       ha="left" if v >= 0 else "right", fontsize=7)
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels([LABEL[k] for k in order], fontsize=8)
    ax[0].axvline(0, color="k", lw=.8)
    ax[0].set_xlabel(f"change in planned finish time (min), Day {DAY}, start {SOC} %, end SOC held at floor")
    ax[0].set_title("Planner (seeded solve; CdA -10 % run is an optimiser miss, see csv)", fontsize=9)
    E2 = E.sort_values("energy_change_pct_plus10", key=abs)
    y = np.arange(len(E2))
    ax[1].barh(y, E2.energy_change_pct_plus10, color="#c2542d", label="+10 %")
    ax[1].barh(y, E2.energy_change_pct_minus10, color="#3b6ea0", label="-10 %")
    ax[1].set_yticks(y); ax[1].set_yticklabels(E2.parameter, fontsize=8); ax[1].axvline(0, color="k", lw=.8)
    ax[1].set_xlabel("change in daily battery drain (% of drive energy), Days 1-3 as driven")
    ax[1].legend(fontsize=7); ax[1].set_title("Optimiser-free: energy on the actual driving", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE / "fig_sensitivity.png", dpi=180)


def energy_elasticity():
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(HERE))
    import identify as idf
    import wind_check as wc
    import separate as sp
    S = wc.attach(idf.build(aligned=True))
    W = sp.windows(S, days=[1, 2, 3])
    f = sp.fit(W)
    M, G = idf.M, idf.G

    def energy(crr=f["crr"], cda=f["cda"], eta=f["eta"], m=M):
        return float(((crr * m * G * W.D + cda * W.A + m * G * W.dh + 0.5 * m / M * 2 * W.dKE) / eta + sp.P0 * W.dur).sum())

    e0 = energy()
    rows = []
    for lab, kw in (("drivetrain efficiency", "eta"), ("Crr", "crr"), ("CdA", "cda"), ("mass", "m")):
        base = dict(crr=f["crr"], cda=f["cda"], eta=f["eta"], m=M)
        out = {}
        for tag, fac in (("plus10", 1.1), ("minus10", 0.9)):
            p = dict(base); p[kw] = base[kw] * fac
            out[tag] = 100 * (energy(**p) / e0 - 1)
        rows.append(dict(parameter=lab, energy_change_pct_plus10=out["plus10"], energy_change_pct_minus10=out["minus10"]))
    A = pd.read_csv(HERE / "actuals_by_day.csv")
    A = A[A.day.isin([1, 2, 3])]
    sol = 0.1 * A.solar_wh.sum() / A.motor_wh.sum() * 100     # +10 % solar offsets this % of the drive energy
    rows.append(dict(parameter="array efficiency (solar offset)", energy_change_pct_plus10=-sol,
                     energy_change_pct_minus10=sol))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
