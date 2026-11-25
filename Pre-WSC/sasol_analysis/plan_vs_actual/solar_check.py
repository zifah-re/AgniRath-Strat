"""Solar side: measured MPPT array power vs Solcast actual GHI (Model/data/Solar_real, Solar_Solcast untouched).
Per day: 5-min bins of measured array power (sum of 4 MPPT V*I) vs mean-over-nodes Solcast GHI;
efficiency estimate eta = P / (area * GHI). Tilt peaks show up as P/(area*GHI) well above the flat value.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import actuals as ac
REAL = HERE.parents[2]/"Model"/"data"/"Solar_real"
AREA = 5.78

def ghi_series(day):
    rows = {}
    for f in sorted(glob.glob(str(REAL/f"*Day {day}[ _]*.jsonl"))) + sorted(glob.glob(str(REAL/f"*Day {day} *.jsonl"))):
        for node in json.load(open(f)):
            for r in node["data"]:
                rows.setdefault(r["period_end"], []).append((r["ghi"], r["dni"]))
    s = pd.DataFrame({"t": pd.to_datetime(list(rows)), "ghi": [np.mean([a for a, _ in v]) for v in rows.values()],
                      "dni": [np.mean([b for _, b in v]) for v in rows.values()]}).drop_duplicates("t").set_index("t").sort_index()
    return s

if __name__ == "__main__":
    out = []
    for day in range(1, 9):
        try:
            g = ghi_series(day)
        except Exception as e:
            print(day, "no ghi", e); continue
        df = ac.load_day(day)
        df = df[df.phase.isin(ac.DRIVE_PHASES + ["charging", "other", "pre-start"])]
        t = df.t.dt.tz_convert("UTC")
        b = pd.DataFrame({"p": df.p_sol_w.to_numpy(), "kmh": df.kmh.to_numpy(), "ph": df.phase.to_numpy()}, index=t)
        b5 = b.resample("5min").agg({"p": "mean", "kmh": "mean", "ph": "first"}).dropna(subset=["p"])
        # Solcast period_end labels the interval ending there: shift to bin start
        gg = g.copy(); gg.index = gg.index - pd.Timedelta("5min")
        m = b5.join(gg, how="inner")
        m = m[m.ghi > 300]
        m["eta"] = m.p / (AREA * m.ghi)
        m["day"] = day
        out.append(m)
        mv = m[m.kmh > 5]
        print(f"D{day}: n={len(m):3d}  median eta(moving)={mv.eta.median():.3f}  p90={mv.eta.quantile(.9):.3f}  "
              f"stationary median={m[m.kmh<1].eta.median():.3f}  peak P={m.p.max():.0f} W")
    pd.concat(out).to_csv(HERE/"solar_check_5min.csv")

    # ---- summary table + figure (appended) ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import pvlib
    A = pd.concat(out)
    rows = []
    for day, g in A.groupby("day"):
        mv = g[g.kmh > 5]
        rows.append(dict(day=day, n_bins=len(g), eta_flat_moving_median=mv.eta.median() if len(mv) else np.nan,
                         eta_flat_moving_p90=mv.eta.quantile(.9) if len(mv) else np.nan, peak_array_w=g.p.max()))
    S = pd.DataFrame(rows).round(3); S.to_csv(HERE/"solar_check_by_day.csv", index=False)
    # tilted-to-sun efficiency from stationary bins with P > 1100 W (array tilted at the sun)
    LOC = {1: (-25.7, 27.2), 8: (-33.4, 19.3)}
    tl = []
    for day, (lat, lon) in LOC.items():
        x = A[(A.day == day) & (A.p > 1100) & (A.kmh < 1)]
        if x.empty: continue
        sp = pvlib.solarposition.get_solarposition(x.index, lat, lon)
        cz = np.cos(np.radians(sp.apparent_zenith.to_numpy()))
        poa = x.dni.to_numpy() + np.clip(x.ghi.to_numpy() - x.dni.to_numpy() * cz, 0, None)
        tl.append(pd.DataFrame({"day": day, "p_w": x.p.to_numpy(), "poa_sun_wm2": poa, "eta_tilted": x.p.to_numpy() / (AREA * poa)}, index=x.index))
    T = pd.concat(tl); T.round(3).to_csv(HERE/"solar_tilted_peaks.csv")
    print("\ntilted-to-sun efficiency: mean %.3f  median %.3f  (n=%d)" % (T.eta_tilted.mean(), T.eta_tilted.median(), len(T)))
    print(S.to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mvA, stA = A[A.kmh > 5], A[A.kmh <= 5]
    ax.scatter(AREA*mvA.ghi, mvA.p, s=10, c="#3b6ea0", label="driving (array flat)")
    ax.scatter(AREA*stA.ghi, stA.p, s=10, c="#c2542d", label="stationary")
    xx = np.array([0, 6500])
    ax.plot(xx, .245*xx, "k--", lw=1, label="24.5 % (model)"); ax.plot(xx, .17*xx, "k:", lw=1, label="17 % (measured, driving)")
    ax.set_xlabel("area x Solcast GHI (W)"); ax.set_ylabel("measured array power (W)"); ax.legend(fontsize=8)
    ax.set_title("Array power vs irradiance, 5-min bins, GHI > 300 W/m2", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE/"fig_solar_check.png", dpi=200)
