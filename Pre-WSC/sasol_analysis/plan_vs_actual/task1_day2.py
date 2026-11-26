"""
Task 1 (proposal Week 14): re-run Day 2 (routes as actually issued: Stage 1, Zeerust loop 34.9 km, Stage 2)
with the original and the refined parameters, and compare with what was driven.

For each parameter set the unchanged single-day solver (resolve-intraday, seeded) is run from 08:00 at 90 %
SOC for 0..7 Zeerust loops; the plan is the largest loop count that is still feasible (SOC floor, finish
cut-off). Parameter sets:
  original : track-test Crr 0.007 / CdA 0.16 / eta 0.95 / 5 W, solar 0.189 (= 0.21 x 0.90, what the
             deployed plan was solved with)
  refined  : effective planner values identified from the telemetry (separate.py) eta 0.773, Crr 0.0103,
             CdA 0.126, 9.2 W; solar 0.17 (whole array while driving, solar_check.py)
  refined, solar 0.245 : same, unshaded-cell solar (bracket: shading fixed)
Actual Day 2: 302 km (both stages, no loops; ran on 4 modules to the control stop, 16 min late).
Writes task1_day2_sweep.csv, task1_day2_summary.csv, fig_task1_day2.png
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
import patched_run as pr     # noqa: E402

REFINED = dict(motor_eff=0.773, crr=0.0103, cda_m2=0.126, p_idle_w=9.2)
SETS = {"original (track values, solar 0.189)": ({}, 0.189),
        "refined (identified, solar 0.17)": (REFINED, 0.17),
        "refined, solar 0.245 (no shading)": (REFINED, 0.245)}
ACTUAL_KM = 302.0
LOOP_KM = 34.9


def main():
    rows = []
    for name, (params, solar) in SETS.items():
        for n in range(0, 8):
            r = pr.run_intraday(params, 2, 90, "08:00", "as_given", loops_left=n, solar_eff=solar)
            st = {s["stage"]: s for s in r.get("stages", [])}
            rows.append(dict(param_set=name, loops=n, feasible=bool(r.get("feasible")), km=r.get("remaining_km"),
                             eta=r.get("eta"), end_soc=r.get("end_soc_pct"),
                             avg_kmh_stage1=st.get("stage1", {}).get("speed_avg_kmh"),
                             avg_kmh_stage2=st.get("stage2", {}).get("speed_avg_kmh"),
                             solar_wh=sum(s.get("solar_wh", 0) for s in st.values())))
            print(rows[-1])
            if not r.get("feasible") and n > 0:
                break
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "task1_day2_sweep.csv", index=False)
    best = (R[R.feasible].sort_values("loops").groupby("param_set").tail(1)
            .set_index("param_set").reindex(list(SETS)))
    best["vs_actual_km"] = best.km - ACTUAL_KM
    best.to_csv(HERE / "task1_day2_summary.csv")
    print(best.to_string())

    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels = list(SETS) + ["actual (driven)"]
    vals = list(best.km.fillna(0)) + [ACTUAL_KM]
    cols = ["#888", "#3b6ea0", "#9bb7d4", "#4f7f5a"]
    ax.barh(range(len(vals))[::-1], vals, color=cols)
    for i, (v, lab) in enumerate(zip(vals, labels)):
        extra = "" if lab.startswith("actual") else f"  ({int(best.loops.iloc[i]) if not np.isnan(best.loops.iloc[i]) else 0} loops, finish {best.eta.iloc[i]})"
        ax.text(v + 4, len(vals) - 1 - i, f"{v:.0f} km{extra}", va="center", fontsize=8)
    ax.set_yticks(range(len(vals))[::-1]); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, max(vals) * 1.35); ax.set_xlabel("Day 2 distance (km)")
    ax.set_title("Day 2 (routes as issued), start 90 % SOC: plan with original vs refined parameters", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE / "fig_task1_day2.png", dpi=180)


if __name__ == "__main__":
    main()
