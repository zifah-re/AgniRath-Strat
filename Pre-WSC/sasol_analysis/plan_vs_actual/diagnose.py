"""
Diagnosis: where does the missing ~1/3 of motor energy come from?

Same inputs as validate_motor.py (actual speed, route gradient). The model's electrical power is
    P_mech = (0.5*rho*CdA*v|v| + m*g*Crr*(1-grad^2/2) + m*g*grad + m*a) * v
    P_el   = P_mech/eff  (P_mech >= 0)   |   P_mech*regen_eff (P_mech < 0)   + P_aux
Measured Bus_Voltage*Bus_Current is averaged over 60 s windows (removes the noise from differentiating
speed). CdA, Crr, eff and P_aux cannot be fitted together (cruising speeds vary too little: they ran to
their bounds with huge standard errors), so each candidate explanation is fitted ON ITS OWN, using the
Day 2 + Day 3 stages only, and then scored on the stages it never saw. The explanation that still works
on held-out stages is the one that carries over.

Writes diagnosis_hypotheses.csv, diagnosis_by_stage.csv, fig_diagnosis.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import actuals          # noqa: E402
import validate_motor as vm  # noqa: E402
from configs.car_config import CarState  # noqa: E402

G = 9.81
RHO = 1.2
WIN_S = 60.0
CAR = CarState()
DEFAULT = dict(cda=CAR.cda_m2, crr=CAR.crr, eff=CAR.motor_eff, aux=CAR.p_idle_w)
FIT_A = {(2, "stage1"), (2, "stage2"), (3, "stage1"), (3, "stage2")}


def build_samples():
    out, days = {}, {}
    for (day, phase), pat in vm.STAGES.items():
        if day not in days:
            days[day] = actuals.load_day(day)
        dist_km, grad = vm.load_route(pat)
        g = vm.stage_frame(days[day], phase)
        if (day, phase) in vm.FULL:
            g["x_km"] = g.x_km * (float(dist_km[-1]) / float(g.x_km.iloc[-1]))
        v = g.v_ms.to_numpy()
        dt = g.dt_c.to_numpy()
        nxt = np.r_[v[1:], v[-1]]
        a = np.where(dt > 0, (nxt - v) / np.maximum(dt, 1e-6), 0.0)
        slope = np.interp(g.x_km.to_numpy(), dist_km, grad) / 100.0
        ts = (g.t - g.t.iloc[0]).dt.total_seconds().to_numpy()
        ok = (dt > 0) & (v > vm.MOVING_MS)
        out[(day, phase)] = dict(v=v[ok], dt=dt[ok], a=a[ok], grad=slope[ok], win=(ts[ok] // WIN_S).astype(int),
                                 p_meas=(g.vbus * g.ibus).to_numpy()[ok])
    return out


def terms(s, p):
    v, grad = s["v"], s["grad"]
    f_aero = 0.5 * RHO * p["cda"] * v * v
    f_roll = CAR.mass_kg * G * p["crr"] * (1 - grad ** 2 / 2)
    f_grav = CAR.mass_kg * G * grad
    f_acc = CAR.mass_kg * s["a"]
    return f_aero * v, f_roll * v, f_grav * v, f_acc * v


def p_el(s, p):
    pa, pr, pg, pc = terms(s, p)
    pm = pa + pr + pg + pc
    return np.where(pm >= 0, pm / p["eff"], pm * CAR.regen_eff) + p["aux"]


def window_means(s, p_pred):
    """Time-weighted mean power per 60 s window: (pred, meas, weight)."""
    w, dt = s["win"], s["dt"]
    uw = np.unique(w)
    pred = np.array([np.average(p_pred[w == k], weights=dt[w == k]) for k in uw])
    meas = np.array([np.average(s["p_meas"][w == k], weights=dt[w == k]) for k in uw])
    wt = np.array([dt[w == k].sum() for k in uw])
    return pred, meas, wt


BOUNDS = dict(cda=(0.05, 0.6), crr=(0.002, 0.03), eff=(0.6, 1.0), aux=(0.0, 400.0))


def _rms(samples, keys, p):
    r, w = [], []
    for k in keys:
        pred, meas, wt = window_means(samples[k], p_el(samples[k], p))
        r.append(pred - meas)
        w.append(wt)
    r, w = np.concatenate(r), np.concatenate(w)
    return float(np.sqrt(np.average(r ** 2, weights=w)))


def fit(samples, keys, free):
    """Least squares on 60 s window means; only the names in `free` move, the rest stay at car_config."""
    def res(x):
        p = dict(DEFAULT)
        p.update(dict(zip(free, x)))
        r = []
        for k in keys:
            pred, meas, wt = window_means(samples[k], p_el(samples[k], p))
            r.append((pred - meas) * np.sqrt(wt / WIN_S))
        return np.concatenate(r)
    lo = [BOUNDS[n][0] for n in free]
    hi = [BOUNDS[n][1] for n in free]
    sol = least_squares(res, [DEFAULT[n] for n in free], bounds=(lo, hi))
    p = dict(DEFAULT)
    p.update(dict(zip(free, sol.x)))
    return p


def stage_energy(s, p):
    return float((p_el(s, p) * s["dt"]).sum() / 3600.0)


HYPOTHESES = {
    "rolling resistance (Crr)": ["crr"],
    "aerodynamic drag / headwind (CdA)": ["cda"],
    "drivetrain efficiency": ["eff"],
    "constant extra load while driving": ["aux"],
    "Crr + extra load": ["crr", "aux"],
}


def main():
    S = build_samples()
    keysA = [k for k in S if k in FIT_A]
    keysT = [k for k in S if k not in FIT_A]
    meas_t = sum(float((S[k]["p_meas"] * S[k]["dt"]).sum() / 3600.0) for k in keysT)
    rows, fitted = [], {}
    p0 = dict(DEFAULT)
    rows.append(dict(hypothesis="car_config as is", fitted="-",
                     rmse_fit_W=_rms(S, keysA, p0), rmse_test_W=_rms(S, keysT, p0),
                     ratio_fit=sum(stage_energy(S[k], p0) for k in keysA) / sum(float((S[k]["p_meas"] * S[k]["dt"]).sum() / 3600) for k in keysA),
                     ratio_test=sum(stage_energy(S[k], p0) for k in keysT) / meas_t))
    fitted["car_config as is"] = p0
    meas_a = sum(float((S[k]["p_meas"] * S[k]["dt"]).sum() / 3600.0) for k in keysA)
    for name, free in HYPOTHESES.items():
        p = fit(S, keysA, free)
        fitted[name] = p
        rows.append(dict(hypothesis=name, fitted=", ".join(f"{n}={p[n]:.4g}" for n in free),
                         rmse_fit_W=_rms(S, keysA, p), rmse_test_W=_rms(S, keysT, p),
                         ratio_fit=sum(stage_energy(S[k], p) for k in keysA) / meas_a,
                         ratio_test=sum(stage_energy(S[k], p) for k in keysT) / meas_t))
    H = pd.DataFrame(rows)
    H.to_csv(HERE / "diagnosis_hypotheses.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 60)
    print("Fitted on Day 2+3 stages, tested on the other stages (60 s window RMSE of battery power, W):")
    print(H.round(3).to_string(index=False))

    single = H.iloc[1:][~H.iloc[1:].hypothesis.str.contains("+", regex=False)]  # two-parameter fit hit its bound
    best = single.sort_values("rmse_test_W").iloc[0].hypothesis
    pb = fitted[best]
    print("\nbest on held-out stages:", best)
    out = []
    for k, s in S.items():
        meas = float((s["p_meas"] * s["dt"]).sum() / 3600.0)
        pa_, pr_, pg_, pc_ = [float((x * s["dt"]).sum() / 3600.0) for x in terms(s, DEFAULT)]
        tot = pa_ + pr_ + pg_ + pc_
        out.append(dict(day=k[0], phase=k[1], measured_wh=meas,
                        default_ratio=stage_energy(s, DEFAULT) / meas,
                        best_ratio=stage_energy(s, pb) / meas,
                        extra_load_ratio=stage_energy(s, fitted["constant extra load while driving"]) / meas,
                        crr_ratio=stage_energy(s, fitted["rolling resistance (Crr)"]) / meas,
                        in_fit=k in FIT_A,
                        aero_share=pa_ / tot, roll_share=pr_ / tot, grav_acc_share=(pg_ + pc_) / tot))
    D = pd.DataFrame(out)
    D.to_csv(HERE / "diagnosis_by_stage.csv", index=False)
    print(D.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9.5, 4))
    lab = [f"D{d} S{p[-1]}" for d, p in zip(D.day, D.phase)]
    x = np.arange(len(D))
    ax.bar(x - 0.27, D.default_ratio, 0.27, color="#8a8a8a", label="car_config values")
    ax.bar(x, D.crr_ratio, 0.27, color="#3d6f9e", label="Crr refitted")
    ax.bar(x + 0.27, D.extra_load_ratio, 0.27, color="#4f7d5a", label="extra driving load refitted")
    ax.axhline(1, color="black", lw=0.8)
    ax.axvspan(-0.5, 4.5 if False else 0.5, alpha=0)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("predicted / measured energy")
    ax.set_ylim(0, 1.45)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    ax.set_title("Refits use Day 2 and Day 3 stages only; the other stages are held out")
    fig.tight_layout()
    fig.savefig(HERE / "fig_diagnosis.png")


if __name__ == "__main__":
    main()
