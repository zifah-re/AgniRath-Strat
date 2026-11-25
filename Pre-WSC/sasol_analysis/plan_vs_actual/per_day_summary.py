"""Per-day plan vs actual table (speed, finish, stops, cause) -> per_day_summary.csv / .md"""
from pathlib import Path
import pandas as pd
import importlib.util
HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("compare", HERE/"compare.py"); 
src = (HERE/"compare.py").read_text()
CAUSE = eval(src[src.index("CAUSE = {")+8:src.index("}\nCAUSE_COLOR")+1])
d = pd.read_csv(HERE/"plan_vs_actual_by_day.csv").set_index("day")
rows = []
for i, r in d.iterrows():
    f = lambda v, n=0: "-" if pd.isna(v) else f"{v:.{n}f}"
    rows.append({
        "Day": i,
        "Plan km (blind)": f(r.plan_km), "As-given plan km": f(r.asg_km), "Driven km": f(r.actual_km),
        "Plan km/h": f(r.plan_speed, 1), "Driven km/h (moving)": "-" if r.actual_km == 0 else f(r.act_avg_moving, 1),
        "Plan finish": r.plan_eta, "As-given finish": r.asg_eta if isinstance(r.asg_eta, str) else "-",
        "Actual last packet": r.act_end if isinstance(r.act_end, str) else "-",
        "Plan stop min": f(r.plan_stop_min), "Logged stop min": f(r.act_stopped_min), "Logger gap min": f(r.act_silent_min),
        "Cause": CAUSE[i][0], "Debrief note": CAUSE[i][1]})
t = pd.DataFrame(rows)
t.to_csv(HERE/"per_day_summary.csv", index=False)
(HERE/"per_day_summary.md").write_text(t.to_markdown(index=False))
print(t.drop(columns="Debrief note").to_string(index=False))
