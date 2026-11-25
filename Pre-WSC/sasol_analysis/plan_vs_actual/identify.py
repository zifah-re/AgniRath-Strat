"""
Physics-based parameter identification from the Sasol telemetry (no black-box optimiser).

Position on the route = speed integrated over time (validate_motor.stage_frame), then re-aligned to the
route by matching the grade pattern in the bus power (align.py; the logger has no GPS/altitude/accel data).
Gradient is read from the route file's stored (smoothed) 'Gradient' at the aligned position. That profile keeps
the net climb but under-states peak grades by 10-27 % (checked against the route altitude); grade from the raw
altitude is too noisy to use. Consequence: eta is biased slightly low and eta_regen high.
Driving modes come from the logged Throttle_Perc / Brake_Status:
  * cruise   throttle = 1, brake off   (cruise control holds speed; the controller motors or regens)
  * coast    throttle = 0, brake off   (bus power is 0 W: motor freewheels, no regen)
  * brake    brake on                  (bus power is 0 W: mechanical brakes, no regen)

Identification, fitted on Days 1-3 only (20 s block means, cruise blocks):
  1. P_bus = (m g sin(th) v + m a v) / eta + P_flat(v-bin)
     The grade + inertia power is exact physics (mass known), so its coefficient gives 1/eta, and the
     per-speed-bin intercepts give the measured flat-road power curve P_flat(v).
  2. eta * (P_flat(v) - P0) = m g Crr v + 1/2 rho CdA v^3  on bins >= 55 km/h (cruising speeds),
     P0 = measured stationary bus draw. Gives Crr and CdA.
  3. regen blocks (cruise, downhill hold, bus power < 0): P_bus = eta_regen * P_mech -> eta_regen.
Spread = bootstrap over blocks. Days 5-8 are held out: stage energy with car_config (regen credited on all
negative power, as the model does) vs identified parameters with the logged modes (no regen when coasting
or braking).

Writes identify_params.csv, identify_flat_power.csv, identify_stage_check.csv, align_report.csv, fig_identify.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import actuals                 # noqa: E402
import validate_motor as vm    # noqa: E402
import align as al              # noqa: E402
from configs.car_config import CarState  # noqa: E402

CAR = CarState()
M, G, RHO = CAR.mass_kg, 9.81, 1.2
BLOCK_S = 20.0
FIT_DAYS = {1, 2, 3}
PROC, ASG = vm.PROC, vm.ASG
# validate_motor stages plus the extra driven stretches that have a route file:
# (pattern, mode) mode: "full" = scale to route length, "partial" = from the start, unscaled, "loop" = x mod length
EXTRA = {
    (1, "loop"): (PROC / "*Day 1 _Rustenburg Loop*.save", "full"),
    (1, "stage2"): (PROC / "*Day 1 _10 Sept Stage 2*.save", "partial"),
    (3, "loop"): (ASG / "Day 3 as-given_Day 3 Loop.kml.save", "loop"),
    (7, "loop"): (PROC / "*Day 7_Van Rhynsdorp Loop*.save", "full"),
    (7, "stage2"): (PROC / "*Day 7_16 Sept Stage 2*.save", "partial"),
}


def stages():
    out = {k: (p, "full" if k in vm.FULL else "partial") for k, p in vm.STAGES.items()}
    out.update(EXTRA)
    return out


def build(aligned=True, report=None):
    rows, days = [], {}
    for (day, phase), (pat, mode) in stages().items():
        if day not in days:
            days[day] = actuals.load_day(day)
        dist_km, grad = vm.load_route(pat)
        g = vm.stage_frame(days[day], phase)
        L = float(dist_km[-1])
        if mode == "full":
            g["x_km"] = g.x_km * (L / float(g.x_km.iloc[-1]))
        elif mode == "loop":
            g["x_km"] = g.x_km % L
        ts = (g.t - g.t.iloc[0]).dt.total_seconds().to_numpy()
        v = g.v_ms.to_numpy()
        p_bus = (g.vbus * g.ibus).to_numpy()
        cruise = (g.thr.to_numpy() >= 0.99) & (g.brk.to_numpy() == 0) & (v > 20 / 3.6) & (g.dt_c.to_numpy() > 0)
        x0 = g.x_km.to_numpy()
        if aligned:
            gap = np.r_[False, np.diff(ts) > vm.MAX_FILL_S]
            g["x_km"] = al.align(x0, v, p_bus, cruise, gap, dist_km, grad, loop_len=(L if mode == "loop" else None))
        if report is not None:
            report.append(dict(day=day, phase=phase, corr_before=al.corr_with_grade(x0, v, p_bus, cruise, dist_km, grad),
                               corr_after=al.corr_with_grade(g.x_km.to_numpy(), v, p_bus, cruise, dist_km, grad)))
        # 5-packet centred smoothing, then a = dv/dt inside continuous runs only
        vs = pd.Series(v).rolling(5, center=True, min_periods=1).mean().to_numpy()
        a = np.gradient(vs, ts)
        cont = g.dt_c.to_numpy() > 0
        cont = cont & np.r_[cont[1:], False]
        th = np.arctan(np.interp(g.x_km.to_numpy(), dist_km, grad) / 100.0)
        mode_ = np.where(g.brk.to_numpy() > 0, "brake", np.where(g.thr.to_numpy() >= 0.99, "cruise",
                         np.where(g.thr.to_numpy() <= 0.01, "coast", "partial")))
        rows.append(pd.DataFrame(dict(day=day, phase=phase, ts=ts, dt=g.dt_c.to_numpy(), v=vs, a=a, th=th,
                                      p=p_bus, mode=mode_, ok=cont & (v > 20 / 3.6))))
    return pd.concat(rows, ignore_index=True)


def blocks(S, sel):
    """20 s block means of the selected packets (blocks with < 60 % coverage dropped)."""
    x = S[sel].copy()
    x["blk"] = x.day.astype(str) + x.phase + "_" + (x.ts // BLOCK_S).astype(int).astype(str)
    x["sn"] = np.sin(x.th); x["cs"] = np.cos(x.th); x["v2"] = x.v ** 2
    x["gi"] = M * G * x.sn * x.v + M * x.a * x.v          # grade + inertia mechanical power, W
    agg = x.groupby("blk").agg(day=("day", "first"), phase=("phase", "first"), v=("v", "mean"), v2=("v2", "mean"),
                               a=("a", "mean"), s=("sn", "mean"), c=("cs", "mean"), gi=("gi", "mean"),
                               p=("p", "mean"), dt=("dt", "sum"))
    return agg[agg.dt >= 0.6 * BLOCK_S]


EDGES = np.arange(40, 85, 5) / 3.6     # speed bins for the flat-road curve
V_FIT = 55 / 3.6                        # road-load fit uses bins at cruising speed


def flat_curve(Bm):
    bi = np.digitize(Bm.v, EDGES)
    ub = np.unique(bi)
    D = (bi[:, None] == ub[None, :]).astype(float)
    X = np.column_stack([Bm.gi, D])
    k = np.linalg.lstsq(X, Bm.p.to_numpy(), rcond=None)[0]
    vb = np.array([Bm.v[bi == u].mean() for u in ub])
    nb = np.array([(bi == u).sum() for u in ub])
    return 1.0 / k[0], vb, k[1:], nb


def road_load(eta, vb, pb, nb, p0):
    m = (vb >= V_FIT) & (nb >= 30)
    w = np.sqrt(nb[m])
    X = np.column_stack([M * G * vb[m], 0.5 * RHO * vb[m] ** 3]) * w[:, None]
    y = eta * (pb[m] - p0) * w
    crr, cda = np.linalg.lstsq(X, y, rcond=None)[0]
    return crr, cda


def mech_power(B, crr, cda):
    return B.gi + M * G * crr * B.c * B.v + 0.5 * RHO * cda * B.v2 * B.v


def identify(S, days, p0, rng=None):
    sel = S.ok & S.day.isin(days) & (S["mode"] == "cruise")
    B = blocks(S, sel) if not isinstance(S, tuple) else None
    if rng is not None:
        B = B.sample(len(B), replace=True, random_state=rng)
    Bm = B[B.p > 50]
    eta, vb, pb, nb = flat_curve(Bm)
    crr, cda = road_load(eta, vb, pb, nb, p0)
    pm = mech_power(B, crr, cda)
    Br = pm < -150          # selected on predicted power only (selecting on measured < 0 biases the slope up)
    eta_r = float(np.sum(pm[Br] * B.p[Br]) / np.sum(pm[Br] ** 2))
    return dict(eta=eta, crr=crr, cda=cda, p0=p0, eta_regen=eta_r, n_motor=len(Bm), n_regen=int(Br.sum()),
                vb=vb, pb=pb, nb=nb)


def predict_power(S, p, mode_aware=True):
    """packet-level predicted bus power."""
    F = M * S.a + M * G * np.sin(S.th) + M * G * p["crr"] * np.cos(S.th) + 0.5 * RHO * p["cda"] * S.v ** 2
    pm = F * S.v
    out = np.where(pm >= 0, pm / p["eta"] + p["p0"], pm * p["eta_regen"])
    if mode_aware:
        out = np.where(S["mode"].isin(["coast", "brake"]), 0.0, out)
    return out


def main():
    rep = []
    S = build(aligned=True, report=rep)
    pd.DataFrame(rep).round(3).to_csv(HERE / "align_report.csv", index=False)
    st = S[(S.v < 0.3) & (S.dt > 0) & S.day.isin(FIT_DAYS)]
    p0 = float(np.average(st.p, weights=st.dt))
    base = dict(crr=CAR.crr, cda=CAR.cda_m2, eta=CAR.motor_eff, p0=CAR.p_idle_w, eta_regen=CAR.regen_eff)
    fit = identify(S, FIT_DAYS, p0)
    boot = pd.DataFrame([{k: v for k, v in identify(S, FIT_DAYS, p0, rng=np.random.RandomState(i)).items()
                          if k in ("eta", "crr", "cda", "eta_regen")} for i in range(300)])
    keys = ["eta", "crr", "cda", "p0", "eta_regen"]
    P = pd.DataFrame(dict(car_config=[base[k] for k in keys], identified=[fit[k] for k in keys],
                          boot_std=[boot[k].std() if k in boot else np.nan for k in keys],
                          boot_p05=[boot[k].quantile(.05) if k in boot else np.nan for k in keys],
                          boot_p95=[boot[k].quantile(.95) if k in boot else np.nan for k in keys]), index=keys)
    P.round(5).to_csv(HERE / "identify_params.csv")
    print(P.round(4).to_string())
    print(f"blocks (Days {sorted(FIT_DAYS)}): motoring {fit['n_motor']}, regenerating {fit['n_regen']};  P0 = measured stationary bus draw")
    cfg_curve = (M * G * CAR.crr * fit["vb"] + 0.5 * RHO * CAR.cda_m2 * fit["vb"] ** 3) / CAR.motor_eff + CAR.p_idle_w
    id_curve = (M * G * fit["crr"] * fit["vb"] + 0.5 * RHO * fit["cda"] * fit["vb"] ** 3) / fit["eta"] + p0
    F = pd.DataFrame(dict(kmh=fit["vb"] * 3.6, n_blocks=fit["nb"], measured_flat_w=fit["pb"], identified_w=id_curve,
                          car_config_w=cfg_curve, used_in_fit=(fit["vb"] >= V_FIT) & (fit["nb"] >= 30)))
    F.round(1).to_csv(HERE / "identify_flat_power.csv", index=False)
    print(F.round(0).to_string(index=False))

    ok = S.ok.to_numpy()
    S["p_cfg"] = predict_power(S, base, mode_aware=False)
    S["p_id"] = predict_power(S, fit, mode_aware=True)
    rows = []
    for (d, ph), g in S[ok].groupby(["day", "phase"]):
        em = (g.p * g.dt).sum() / 3600
        rows.append(dict(day=d, phase=ph, set="fit" if d in FIT_DAYS else "held-out", km=(g.v * g.dt).sum() / 1000,
                         measured_wh=em, cfg_ratio=(g.p_cfg * g.dt).sum() / 3600 / em,
                         identified_ratio=(g.p_id * g.dt).sum() / 3600 / em))
    C = pd.DataFrame(rows)
    C.round(3).to_csv(HERE / "identify_stage_check.csv", index=False)
    print(C.round(3).to_string(index=False))
    for s_ in ("fit", "held-out"):
        c = C[C.set == s_]
        print(f"{s_:8s}: energy ratio car_config {c.cfg_ratio.mul(c.measured_wh).sum() / c.measured_wh.sum():.3f}"
              f"  identified {c.identified_ratio.mul(c.measured_wh).sum() / c.measured_wh.sum():.3f}"
              f"   mean |err| car_config {100 * (c.cfg_ratio - 1).abs().mean():.1f}%  identified {100 * (c.identified_ratio - 1).abs().mean():.1f}%")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    sel = S.ok & S.day.isin(FIT_DAYS) & (S["mode"] == "cruise")
    B = blocks(S, sel)
    se = boot.std()
    ax[0].errorbar(F.kmh, F.measured_flat_w, fmt="o", color="#3b6ea0", label="measured (grade + inertia removed)",
                   yerr=[np.nan] * len(F) if False else None)
    vv = np.linspace(30, 80, 60) / 3.6
    ax[0].plot(vv * 3.6, (M * G * fit["crr"] * vv + 0.5 * RHO * fit["cda"] * vv ** 3) / fit["eta"] + p0, "k-",
               label=f"identified: eta {fit['eta']:.2f}, Crr {fit['crr']:.4f}, CdA {fit['cda']:.3f}")
    ax[0].plot(vv * 3.6, (M * G * CAR.crr * vv + 0.5 * RHO * CAR.cda_m2 * vv ** 3) / CAR.motor_eff + CAR.p_idle_w, "k--", label="car_config")
    ax[0].axvspan(30, 55, color="0.9", zorder=0)
    ax[0].set_xlabel("speed (km/h)"); ax[0].set_ylabel("flat-road bus power (W)"); ax[0].legend(fontsize=7)
    ax[0].set_title("Flat-road power vs speed, Days 1-3 (grey: not used for road-load fit)", fontsize=9)
    m = B.p > 50
    resid = B.p[m] - (mech_power(B[m], fit["crr"], fit["cda"]) - B.gi[m]) / fit["eta"] - p0
    ax[1].scatter(B.gi[m], resid, s=5, alpha=.4)
    xx = np.linspace(B.gi[m].min(), B.gi[m].max(), 20)
    ax[1].plot(xx, xx / fit["eta"], "k-", label=f"slope 1/eta, eta = {fit['eta']:.3f} +- {se['eta']:.3f}")
    ax[1].plot(xx, xx / CAR.motor_eff, "k--", label="car_config eta 0.95")
    ax[1].set_xlabel("grade + inertia mechanical power (W)"); ax[1].set_ylabel("bus power minus flat-road part (W)")
    ax[1].legend(fontsize=7); ax[1].set_title("Drivetrain efficiency from the grade term", fontsize=9)
    lab = [f"D{d} {dict(stage1='s1', stage2='s2', loop='loop')[p]}" for d, p in zip(C.day, C.phase)]
    x = np.arange(len(C))
    ax[2].bar(x - .2, C.cfg_ratio, .4, color="#888", label="car_config, regen on all negative power")
    ax[2].bar(x + .2, C.identified_ratio, .4, color="#3b6ea0", label="identified, logged modes")
    ax[2].axhline(1, color="k", lw=.8)
    nf = (C.set == "fit").sum()
    ax[2].axvline(nf - .5, color="k", ls=":", lw=.8)
    ax[2].set_xticks(x); ax[2].set_xticklabels(lab, rotation=60, fontsize=7)
    ax[2].set_ylabel("predicted / measured energy"); ax[2].legend(fontsize=7)
    ax[2].set_title("Stage energy: fit days | held-out days", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE / "fig_identify.png", dpi=180)


if __name__ == "__main__":
    main()
