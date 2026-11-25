"""Post-hoc calibration of the plan (Model is NOT modified).
Factors come from Day 2/3 only; Days 1,5,6,7,8 are held-out checks.
  1. energy: plan Wh/km divided by (predicted/measured) energy ratio on Day 2/3 stages
  2. stops : unplanned stop allowance = median (measured stopped+silent - planned) min on Day 2/3
"""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).parent
d = pd.read_csv(HERE/"plan_vs_actual_by_day.csv").set_index("day")
st = pd.read_csv(HERE/"diagnosis_by_stage.csv")
FIT, HELD = [2, 3], [1, 5, 6, 7, 8]
f = st[st.day.isin(FIT)]
k = (f.default_ratio*f.measured_wh).sum()/f.measured_wh.sum()
d["wh_before"] = d.plan_wh_per_km
d["wh_after"] = d.plan_wh_per_km/k
d["err_before_pct"] = 100*(d.wh_before/d.act_wh_per_km-1)
d["err_after_pct"] = 100*(d.wh_after/d.act_wh_per_km-1)
d["set"] = ["fit" if i in FIT else "held-out" for i in d.index]
out = d[["set","wh_before","wh_after","act_wh_per_km","err_before_pct","err_after_pct",
         "act_stopped_min","plan_stop_min"]].round(1)
out.to_csv(HERE/"refinement_before_after.csv")
print(f"energy factor k={k:.3f} (plan Wh/km x {1/k:.2f}), from Day 2/3 stages\n")
print(out.to_string())
h = out.loc[HELD].dropna(subset=["err_before_pct"])
print("\nheld-out mean |energy err|: before %.1f%%  after %.1f%%" % (h.err_before_pct.abs().mean(), h.err_after_pct.abs().mean()))
o = out.dropna(subset=["act_wh_per_km"]); x = np.arange(len(o))
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.bar(x-.27, o.wh_before, .27, label="plan (as built)", color="#888")
ax.bar(x, o.wh_after, .27, label="plan, calibrated", color="#3b6ea0")
ax.bar(x+.27, o.act_wh_per_km, .27, label="measured", color="#4f7f5a")
ax.set_xticks(x); ax.set_xticklabels([f"D{i}" for i in o.index]); ax.set_ylabel("Wh/km"); ax.legend(fontsize=8)
ax.set_title("Energy per km: calibrated on Day 2/3, D1/5/6/7/8 held out", fontsize=9)
plt.tight_layout(); plt.savefig(HERE/"fig_refinement.png", dpi=200)
