"""
Steady flat cruise: is the measured power really above the 880 W @ 60 km/h track figure, once real air
density and real wind are included?

Selection (aligned route position, identify.build): cruise control on, brake off, |a| < 0.05 m/s2,
|grade| < 0.5 %, 20 s block means, 50-75 km/h.
Air: density from route altitude (barometric formula) and Solcast air_temp; wind from Solcast estimated
actuals (wind_speed_10m, wind_direction_10m), position-matched along the stage like the irradiance in
solar_check.py, resolved along the car heading from the route file.
Wind at car height: the model uses the 10 m wind as-is. A log profile (open country, z0 = 0.03 m) gives
~0.60 x at 1 m height; both are shown.
Predictions use car_config road load (Crr 0.007, CdA 0.16, eta 0.95, 5 W):
  P = (m g Crr v + 0.5 rho CdA (v - w) |v - w| v) / eta + p_idle
Writes wind_check_blocks.csv, wind_check_by_speed.csv, wind_check_by_day.csv, fig_wind_check.png
"""
import json
import glob
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
import solar_check as sc        # noqa: E402

M, G, CAR = idf.M, idf.G, idf.CAR
HEIGHT_FACTOR = np.log(1.0 / 0.03) / np.log(10.0 / 0.03)   # ~0.60


def route_profile(pat):
    p = json.load(open(glob.glob(str(pat))[0], encoding="utf-8"))["profile"]
    d = np.array(p["Distance"], float)
    hd = np.r_[np.array(p["Headings"], float), p["Headings"][-1]][:len(d)]
    alt = np.array(p["Altitude"], float)[:len(d)]
    return d, hd, alt


def weather_nodes(path):
    nodes = json.load(open(path))
    t = pd.to_datetime([r["period_end"] for r in nodes[0]["data"]]) - pd.Timedelta("5min")
    def mat(k):
        return np.array([[float(r.get(k, np.nan)) for r in n["data"]] for n in nodes])
    ws, wd, ta = mat("wind_speed_10m"), mat("wind_direction_10m"), mat("air_temp")
    # direction FROM -> components of the vector the wind blows TOWARD (east, north)
    u = ws * np.sin(np.radians(wd + 180)); v = ws * np.cos(np.radians(wd + 180))
    return t, u, v, ta


def interp2(Mx, pos, tt):
    i0 = np.clip(np.floor(pos).astype(int), 0, max(Mx.shape[0] - 2, 0)); fi = pos - i0
    j0 = np.clip(np.floor(tt).astype(int), 0, Mx.shape[1] - 2); fj = tt - j0
    if Mx.shape[0] == 1:
        return Mx[0, j0] * (1 - fj) + Mx[0, j0 + 1] * fj
    a = Mx[i0, j0] * (1 - fj) + Mx[i0, j0 + 1] * fj
    b = Mx[i0 + 1, j0] * (1 - fj) + Mx[i0 + 1, j0 + 1] * fj
    return a * (1 - fi) + b * fi


def attach(S):
    st = idf.stages()
    parts = []
    for (day, phase), g in S.groupby(["day", "phase"]):
        files = sc.FILES[day].get(phase, [])
        if not files:
            continue
        pat, mode = st[(day, phase)]
        d, hd, alt = route_profile(pat)
        x = g.x_km.to_numpy()
        hr = np.radians(hd)
        head = np.degrees(np.arctan2(np.interp(x, d, np.sin(hr)), np.interp(x, d, np.cos(hr)))) % 360
        h_m = np.interp(x, d, alt)
        t, U, V, TA = weather_nodes(files[0])
        pos = x / d[-1] * (U.shape[0] - 1)
        tu = pd.to_datetime(g.t).dt.tz_convert("UTC").dt.tz_localize(None)
        t0 = t[0].tz_localize(None) if t[0].tzinfo else t[0]
        tt = np.clip((tu - t0).dt.total_seconds().to_numpy() / 300.0, 0, len(t) - 1)
        u, v, ta = interp2(U, pos, tt), interp2(V, pos, tt), interp2(TA, pos, tt)
        # along-track wind (+ = tailwind), model convention v_air = v - w_par
        w_par = u * np.sin(np.radians(head)) + v * np.cos(np.radians(head))
        p_pa = 101325.0 * (1 - 2.25577e-5 * h_m) ** 5.25588
        rho = p_pa / (287.05 * (ta + 273.15))
        parts.append(g.assign(heading=head, alt_m=h_m, w_par=w_par, wind_speed=np.hypot(u, v), air_temp=ta, rho=rho))
    return pd.concat(parts)


def p_model(v, rho, w, crr=CAR.crr, cda=CAR.cda_m2, eta=CAR.motor_eff, p0=CAR.p_idle_w):
    va = v - w
    return (M * G * crr * v + 0.5 * rho * cda * va * np.abs(va) * v) / eta + p0


def main():
    S = idf.build(aligned=True)
    S = attach(S)
    sel = S.ok & (S["mode"] == "cruise") & (S.a.abs() < 0.05) & (np.abs(np.tan(S.th)) < 0.005)
    x = S[sel].copy()
    x["blk"] = x.day.astype(str) + x.phase + "_" + (x.ts // 20).astype(int).astype(str)
    B = x.groupby("blk").agg(day=("day", "first"), phase=("phase", "first"), v=("v", "mean"), p=("p", "mean"),
                             rho=("rho", "mean"), w_par=("w_par", "mean"), wind=("wind_speed", "mean"),
                             alt=("alt_m", "mean"), temp=("air_temp", "mean"), dt=("dt", "sum"))
    B = B[(B.dt >= 12) & (B.v * 3.6 >= 50) & (B.v * 3.6 <= 75)].copy()
    B["kmh"] = B.v * 3.6
    B["p_track"] = p_model(B.v, 1.2, 0.0)                         # the 880 W standard
    B["p_rho"] = p_model(B.v, B.rho, 0.0)                         # + real air density
    B["p_wind10"] = p_model(B.v, B.rho, B.w_par)                  # + 10 m wind (as the model does)
    B["p_wind1"] = p_model(B.v, B.rho, HEIGHT_FACTOR * B.w_par)   # + wind scaled to car height
    B.round(3).to_csv(HERE / "wind_check_blocks.csv")
    cols = ["p", "p_track", "p_rho", "p_wind10", "p_wind1"]
    B["bin"] = pd.cut(B.kmh, [50, 55, 60, 65, 70, 75])
    T = B.groupby("bin", observed=True).agg(n=("p", "size"), measured=("p", "mean"),
                                            measured_se=("p", lambda s: s.std() / np.sqrt(len(s))),
                                            track_rho12_nowind=("p_track", "mean"), real_rho=("p_rho", "mean"),
                                            real_rho_wind10m=("p_wind10", "mean"), real_rho_wind1m=("p_wind1", "mean"),
                                            headwind_ms=("w_par", lambda s: -s.mean()), rho=("rho", "mean"))
    T.round(2).to_csv(HERE / "wind_check_by_speed.csv")
    print(f"steady flat cruise blocks: {len(B)} (height factor {HEIGHT_FACTOR:.2f})")
    print(T.round(2).to_string())
    D = B.groupby("day").agg(n=("p", "size"), kmh=("kmh", "mean"), measured=("p", "mean"), track=("p_track", "mean"),
                             wind10=("p_wind10", "mean"), wind1=("p_wind1", "mean"), headwind_ms=("w_par", lambda s: -s.mean()),
                             wind_ms=("wind", "mean"), rho=("rho", "mean"), alt_m=("alt", "mean"), temp=("temp", "mean"))
    D["meas_over_wind10"] = D.measured / D.wind10
    D["meas_over_wind1"] = D.measured / D.wind1
    D.round(3).to_csv(HERE / "wind_check_by_day.csv")
    print(D.round(2).to_string())
    m60 = B[(B.kmh >= 57) & (B.kmh <= 63)]
    print(f"\n57-63 km/h: n={len(m60)} measured {m60.p.mean():.0f} +- {m60.p.std() / np.sqrt(len(m60)):.0f} W (median {m60.p.median():.0f})"
          f" | track standard {m60.p_track.mean():.0f} | real rho {m60.p_rho.mean():.0f} | + wind 10 m {m60.p_wind10.mean():.0f}"
          f" | + wind 1 m {m60.p_wind1.mean():.0f}")
    for lab, col in (("10 m wind", "p_wind10"), ("car-height wind", "p_wind1")):
        mech = (B[col] - CAR.p_idle_w) * CAR.motor_eff
        eta = np.sum(mech * (B.p - CAR.p_idle_w)) / np.sum((B.p - CAR.p_idle_w) ** 2)
        print(f"effective drivetrain efficiency that matches measured power with {lab}: {eta:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    xb = np.arange(len(T))
    ax[0].errorbar(xb, T.measured, 1.96 * T.measured_se, fmt="o", color="k", capsize=3, label="measured (95 % CI)")
    ax[0].plot(xb, T.track_rho12_nowind, "s--", color="#888", label="car_config, rho 1.2, no wind (track standard)")
    ax[0].plot(xb, T.real_rho_wind10m, "d-", color="#3b6ea0", label="car_config + real rho + 10 m wind")
    ax[0].plot(xb, T.real_rho_wind1m, "^-", color="#7a9cc6", label="car_config + real rho + car-height wind")
    ax[0].set_xticks(xb); ax[0].set_xticklabels([str(b) for b in T.index]); ax[0].set_xlabel("speed bin (km/h)")
    ax[0].set_ylabel("bus power (W)"); ax[0].legend(fontsize=7)
    ax[0].set_title("Steady flat cruise (|a|<0.05, |grade|<0.5 %), all days", fontsize=9)
    ax[1].scatter(-B.w_par, B.p / B.p_rho, s=8, alpha=.5)
    ax[1].axhline(1, color="k", lw=.8); ax[1].axvline(0, color="k", lw=.8)
    ax[1].set_xlabel("headwind component, 10 m (m/s)"); ax[1].set_ylabel("measured / predicted (real rho, no wind)")
    ax[1].set_title("Does the excess follow the wind?", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE / "fig_wind_check.png", dpi=180)


if __name__ == "__main__":
    main()
