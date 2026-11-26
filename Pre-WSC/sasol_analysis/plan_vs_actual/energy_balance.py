"""
Battery energy balance per stage (proposal Week 12 "SoC" comparison, done as energy because SOC_Ah is stuck
and Pack_Current is logged as 0).

Measured  net battery energy = sum(MPPT array power - motor-controller bus power) dt over the stage
Model     net battery energy = sum(array model - bus model) dt over the SAME packets, as driven (logged speed,
          aligned route position, Solcast estimated-actual irradiance at that position and time):
  original : track Crr 0.007 / CdA 0.16 / eta 0.95 / regen 0.70 / 5 W, solar 0.189 (deployed plan)
  refined  : eta 0.773 / Crr 0.0103 / CdA 0.126 / regen 0.70 / 9.2 W (separate.py, model air), solar 0.17
Both models credit regen on all negative mechanical power, as the planner does. Array model = efficiency x
5.78 m2 x POA of a flat array (= GHI). Energies in Wh; the pack had 3-6 modules over the race
(588 Wh each), so Wh is used rather than SOC %.
Writes energy_balance_by_stage.csv, fig_energy_balance.png
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
import actuals as ac            # noqa: E402
import identify as idf          # noqa: E402
import solar_check as sc        # noqa: E402

AREA = 5.78
ORIG = dict(crr=0.007, cda=0.16, eta=0.95, p0=5.0, eta_regen=0.70, solar=0.189)
REF = dict(crr=0.0103, cda=0.126, eta=0.773, p0=9.2, eta_regen=0.70, solar=0.17)


def solar_frame(day):
    df = ac.load_day(day)
    df.attrs["day"] = day
    df = df[df.phase.isin(["stage1", "stage2", "loop"])]
    df = sc.attach_weather(df).dropna(subset=["ghi"])
    return df[["t", "phase", "p_sol_w", "ghi"]]


def main():
    S = idf.build(aligned=True)
    rows = []
    for day in sorted(S.day.unique()):
        sol = solar_frame(day)
        for ph in S[S.day == day].phase.unique():
            g = S[(S.day == day) & (S.phase == ph) & (S.dt > 0)].copy()
            s = sol[sol.phase == ph]
            if g.empty or s.empty:
                continue
            g["t"] = pd.to_datetime(g.t)
            m = g.merge(s[["t", "p_sol_w", "ghi"]], on="t", how="inner")
            if len(m) < 0.8 * len(g):
                continue
            h = m.dt / 3600.0
            meas_bus = float((m.p * h).sum())
            meas_sol = float((m.p_sol_w * h).sum())
            row = dict(day=day, phase=ph, km=float((m.v * m.dt).sum() / 1000), hours=float(h.sum()),
                       measured_solar_wh=meas_sol, measured_bus_wh=meas_bus, measured_net_wh=meas_sol - meas_bus)
            for tag, p in (("original", ORIG), ("refined", REF)):
                bus = idf.predict_power(m, p, mode_aware=False)
                solw = p["solar"] * AREA * m.ghi.clip(lower=0)
                row[f"{tag}_solar_wh"] = float((solw * h).sum())
                row[f"{tag}_bus_wh"] = float((bus * h).sum())
                row[f"{tag}_net_wh"] = row[f"{tag}_solar_wh"] - row[f"{tag}_bus_wh"]
            rows.append(row)
    R = pd.DataFrame(rows)
    for tag in ("original", "refined"):
        R[f"{tag}_net_err_wh"] = R[f"{tag}_net_wh"] - R.measured_net_wh
    R.round(1).to_csv(HERE / "energy_balance_by_stage.csv", index=False)
    print(R[["day", "phase", "km", "measured_net_wh", "original_net_wh", "refined_net_wh",
             "original_net_err_wh", "refined_net_err_wh"]].round(0).to_string(index=False))
    for s_, days in (("Days 1-3, 6, 7 (car healthy)", [1, 2, 3, 6, 7]), ("Days 5, 8 (mechanical C broke)", [5, 8])):
        x = R[R.day.isin(days)]
        print(f"{s_}: battery drain measured {-x.measured_net_wh.sum():.0f} Wh | original model {-x.original_net_wh.sum():.0f} Wh"
              f" | refined {-x.refined_net_wh.sum():.0f} Wh ; mean |error| per stage original {x.original_net_err_wh.abs().mean():.0f} Wh,"
              f" refined {x.refined_net_err_wh.abs().mean():.0f} Wh")

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(R))
    ax.bar(x - .27, -R.measured_net_wh, .27, color="#4f7f5a", label="measured")
    ax.bar(x, -R.original_net_wh, .27, color="#888", label="original model (track values, solar 0.189)")
    ax.bar(x + .27, -R.refined_net_wh, .27, color="#3b6ea0", label="refined model (identified, solar 0.17)")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d} {dict(stage1='s1', stage2='s2', loop='loop')[p]}" for d, p in zip(R.day, R.phase)], fontsize=8)
    ax.set_ylabel("battery drain over the stage (Wh)\n(+ = discharged, - = charged)")
    ax.legend(fontsize=8)
    ax.set_title("Battery energy balance per stage, as driven: solar in minus motor out", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE / "fig_energy_balance.png", dpi=180)


if __name__ == "__main__":
    main()
