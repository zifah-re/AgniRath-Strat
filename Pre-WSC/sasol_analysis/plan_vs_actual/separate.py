"""
Separating Crr, CdA and drivetrain efficiency from the race telemetry (energy-balance method).

Why windows: the route's stored gradient is smoothed (peaks under-stated by 10-27 %) and the logged
position is integrated speed, so packet-level grade is noisy. Over a window of continuous driving the
climb energy is m g dh with dh from the route altitude at the window ends; smoothing keeps the net climb,
so dh is right even where the local grade is not.

Window = 2 km of continuous cruise-control motoring (bus power > 0 all the way, no logger gap), aligned
route position (align.py), real air density (route altitude + Solcast air temperature), Solcast wind
along the heading scaled to car height (x0.60, log profile).  For each window
    E_bus - P0 T = [ Crr m g D + CdA * sum(1/2 rho v_air |v_air| v dt) + m g dh + dKE ] / eta
which is linear in (Crr/eta, CdA/eta, 1/eta): Huber-robust least squares.
Fit on the days the car was mechanically sound (1, 2, 3, 6, 7; mechanical C broke on Days 5 and 8).

Checks: leave-one-day-out; cluster bootstrap over stages; window length 1/2/3 km; wind x0 / x0.6 / x1;
the efficiency each grade class needs with the track-test Crr/CdA (climbs separate tyres from drivetrain:
rolling resistance cannot make a climb cost more, a drivetrain loss scales the climb energy too);
the effective efficiency the model needs at its fixed air density 1.2.

Writes separate_fits.csv, separate_by_grade.csv, separate_by_day.csv, fig_separate.png
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
import identify as idf          # noqa: E402
import wind_check as wc         # noqa: E402

M, G, CAR = idf.M, idf.G, idf.CAR
P0 = 9.2                        # measured stationary bus draw (identify.py)
CLEAN = [1, 2, 3, 6, 7]
TRACK = dict(crr=CAR.crr, cda=CAR.cda_m2)


def huber(X, y, c=1.345, it=100):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(it):
        r = y - X @ b
        s = np.median(np.abs(r - np.median(r))) / 0.6745 + 1e-9
        u = np.abs(r) / (c * s)
        w = np.where(u <= 1, 1.0, 1.0 / u)
        bn = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        if np.allclose(bn, b, rtol=1e-9):
            break
        b = bn
    return b


def windows(S, kw=wc.HEIGHT_FACTOR, w_km=2.0, days=None, rho_fixed=None):
    rows = []
    for (d, ph), g in S.groupby(["day", "phase"]):
        if days is not None and d not in days:
            continue
        g = g.sort_values("ts")
        good = (g.ok & (g["mode"] == "cruise") & (g.p > 0) & (g.dt > 0)).to_numpy()
        v, dt, p = g.v.to_numpy(), g.dt.to_numpy(), g.p.to_numpy()
        rho = np.full(len(g), rho_fixed) if rho_fixed else g.rho.to_numpy()
        va = v - kw * g.w_par.to_numpy()
        alt = g.alt_m.to_numpy()
        i, n = 0, len(g)
        while i < n:
            if not good[i]:
                i += 1
                continue
            j, dist = i, 0.0
            while j < n and good[j] and dist < w_km * 1000:
                dist += v[j] * dt[j]
                j += 1
            if dist >= 0.98 * w_km * 1000:
                sl = slice(i, j)
                rows.append(dict(day=d, phase=ph, dur=dt[sl].sum(), D=dist, E=(p[sl] * dt[sl]).sum(),
                                 A=(0.5 * rho[sl] * va[sl] * np.abs(va[sl]) * v[sl] * dt[sl]).sum(),
                                 dh=alt[j - 1] - alt[i], dKE=0.5 * M * (v[j - 1] ** 2 - v[i] ** 2)))
            i = j if j > i else i + 1
    W = pd.DataFrame(rows)
    W["grade_pct"] = 100 * W.dh / W.D
    W["kmh"] = 3.6 * W.D / W.dur
    return W


def fit(W):
    y = ((W.E - P0 * W.dur) / W.dur).to_numpy()
    X = np.column_stack([M * G * W.D / W.dur, W.A / W.dur, (M * G * W.dh + W.dKE) / W.dur])
    k = huber(X, y)
    eta = 1.0 / k[2]
    return dict(eta=eta, crr=k[0] * eta, cda=k[1] * eta, n=len(W))


def eta_given(W, crr, cda):
    """efficiency needed (ratio of sums) if the road load is fixed."""
    mech = M * G * crr * W.D + cda * W.A + M * G * W.dh + W.dKE
    return float(mech.sum() / (W.E - P0 * W.dur).sum())


def main():
    S = wc.attach(idf.build(aligned=True))
    W = windows(S, days=CLEAN)
    rows = [dict(fit="baseline: 2 km, wind x0.60, real rho, Days 1,2,3,6,7", **fit(W))]
    for d in CLEAN:
        rows.append(dict(fit=f"leave out Day {d}", **fit(W[W.day != d])))
    for wk in (1.0, 3.0):
        rows.append(dict(fit=f"{wk:.0f} km windows", **fit(windows(S, w_km=wk, days=CLEAN))))
    for kw in (0.0, 1.0):
        rows.append(dict(fit=f"wind x{kw:.1f}", **fit(windows(S, kw=kw, days=CLEAN))))
    rng = np.random.RandomState(0)
    cl = W.groupby(["day", "phase"]).groups
    keys = list(cl)
    bs = pd.DataFrame([fit(W.loc[np.concatenate([cl[keys[j]] for j in rng.randint(0, len(keys), len(keys))])])
                       for _ in range(500)])
    for q in (5, 95):
        rows.append(dict(fit=f"cluster bootstrap {q}th percentile", eta=np.percentile(bs.eta, q),
                         crr=np.percentile(bs.crr, q), cda=np.percentile(bs.cda, q), n=len(W)))
    W12 = windows(S, days=CLEAN, rho_fixed=1.2, kw=1.0)
    rows.append(dict(fit="model as built (rho 1.2, 10 m wind): full fit = effective parameters for the planner", **fit(W12)))
    rows.append(dict(fit="model as built (rho 1.2, 10 m wind), track Crr/CdA: efficiency needed",
                     eta=eta_given(W12, **TRACK), crr=TRACK["crr"], cda=TRACK["cda"], n=len(W12)))
    rows.append(dict(fit="real air, track Crr/CdA: efficiency needed", eta=eta_given(W, **TRACK),
                     crr=TRACK["crr"], cda=TRACK["cda"], n=len(W)))
    R = pd.DataFrame(rows)
    R.round(4).to_csv(HERE / "separate_fits.csv", index=False)
    print(R.round(4).to_string(index=False))
    print(f"bootstrap correlations: Crr-CdA {np.corrcoef(bs.crr, bs.cda)[0, 1]:.2f}, eta-Crr {np.corrcoef(bs.eta, bs.crr)[0, 1]:.2f}")

    base = R.iloc[0]
    classes = [("descent < -0.5 %", W.grade_pct < -0.5), ("flat |g| < 0.5 %", W.grade_pct.abs() <= 0.5),
               ("climb 0.5-1.5 %", (W.grade_pct > 0.5) & (W.grade_pct <= 1.5)), ("climb > 1.5 %", W.grade_pct > 1.5)]
    Gc = pd.DataFrame([dict(grade_class=lab, n=int(m.sum()), kmh=W.D[m].sum() / W.dur[m].sum() * 3.6,
                            bus_w=((W.E - P0 * W.dur)[m].sum() / W.dur[m].sum()),
                            eta_needed_track_roadload=eta_given(W[m], **TRACK),
                            eta_needed_fitted_roadload=eta_given(W[m], base.crr, base.cda)) for lab, m in classes])
    Gc.round(3).to_csv(HERE / "separate_by_grade.csv", index=False)
    print(Gc.round(3).to_string(index=False))

    Wall = windows(S)
    Dd = pd.DataFrame([dict(day=d, n=len(g), kmh=g.D.sum() / g.dur.sum() * 3.6,
                            eta_needed_fitted_roadload=eta_given(g, base.crr, base.cda),
                            pred_over_meas_fitted=float(((M * G * base.crr * g.D + base.cda * g.A + M * G * g.dh + g.dKE) / base.eta
                                                         + P0 * g.dur).sum() / g.E.sum()),
                            pred_over_meas_track=float(((M * G * CAR.crr * g.D + CAR.cda_m2 * g.A + M * G * g.dh + g.dKE) / CAR.motor_eff
                                                        + CAR.p_idle_w * g.dur).sum() / g.E.sum()))
                       for d, g in Wall.groupby("day")])
    Dd["set"] = np.where(Dd.day.isin(CLEAN), "fit", "held-out (mechanical C broke)")
    Dd.round(3).to_csv(HERE / "separate_by_day.csv", index=False)
    print(Dd.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].scatter(bs.crr, bs.cda, s=6, alpha=.4, label="bootstrap fits")
    ax[0].scatter([CAR.crr], [CAR.cda_m2], marker="x", c="k", s=70, label="track testing")
    ax[0].scatter([base.crr], [base.cda], marker="o", c="#c2542d", s=40, label="fit")
    ax[0].set_xlabel("Crr"); ax[0].set_ylabel("CdA (m2)"); ax[0].legend(fontsize=7)
    ax[0].set_title("Road load: race data vs track testing", fontsize=9)
    ax[1].hist(bs.eta, bins=30, color="#3b6ea0")
    ax[1].axvline(CAR.motor_eff, color="k", ls="--", label="track 0.95")
    ax[1].axvline(base.eta, color="#c2542d", label=f"fit {base.eta:.2f}")
    ax[1].set_xlabel("drivetrain efficiency (bootstrap)"); ax[1].legend(fontsize=7)
    ax[1].set_title("Drivetrain efficiency", fontsize=9)
    x = np.arange(len(Gc))
    ax[2].bar(x, Gc.eta_needed_track_roadload, color="#888")
    ax[2].axhline(CAR.motor_eff, color="k", ls="--")
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"{g}\n(n={n})" for g, n in zip(Gc.grade_class, Gc.n)], fontsize=7)
    ax[2].set_ylabel("efficiency needed with track Crr/CdA")
    ax[2].set_title("Climbs need the same low efficiency: a drivetrain loss, not tyres", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE / "fig_separate.png", dpi=180)


if __name__ == "__main__":
    main()
