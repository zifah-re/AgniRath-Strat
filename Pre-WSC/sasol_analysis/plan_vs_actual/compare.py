"""
Plan vs actual for Sasol 2026.

  plan        Model/output/strategy_aryaman_idk.json   (the plan that was deployed)
  as-given    Model/output/strategy_as_given_resolve_day2_*.json (optional; latest is used)
              = same model, same car, Day 2 onward re-solved on the routes actually issued
  actual      debrief distances + telemetry (run actuals.py first)

Writes fig_*.png and plan_vs_actual_by_day.csv next to this file.
    python actuals.py && python compare.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_OUT = HERE.parents[2] / "Model" / "output"

PLAN_C, ASG_C, ACT_C = "#8a8a8a", "#3d6f9e", "#c2542d"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 150})

# Distance driven per day (km), from the debrief. Day 6 is "~57" there; 58.1 makes the eight
# days add up to the 1192.6 km the car covered in total.
ACTUAL_KM = {1: 214.0, 2: 302.0, 3: 256.0, 4: 0.0, 5: 182.0, 6: 58.1, 7: 100.5, 8: 80.0}
assert abs(sum(ACTUAL_KM.values()) - 1192.6) < 0.01

# Primary cause of each day's shortfall, from the debrief.
CAUSE = {
    1: ("Controller / drivetrain", "MC overheated in city loop and on hills; upright bolt sheared; 14 min late"),
    2: ("Battery / pack", "battery lug, no pack voltage, ran on 4 modules; no loops; lower-arm bearing"),
    3: ("DCU / wiring", "DCU errors, loose lug, solar fuse blown (1 h lost); 2 loops"),
    4: ("DCU / wiring", "DCU CAN transceiver, no drive mode; trailered"),
    5: ("Weather / trailer", "bad weather, stage 1 trailered; 2 short loops; rain"),
    6: ("Battery / pack", "module bypassed (3 modules), loose lug, spot weld came out"),
    7: ("Weather / battery", "fog; 3.5 modules; high power draw, DF bolt out; trailered"),
    8: ("Mechanical / wiring", "stage 1 trailered; support part 'C' loss +40% power; safe-state wire"),
}
CAUSE_COLOR = {"Controller / drivetrain": "#b5651d", "Battery / pack": "#8c2d2d", "DCU / wiring": "#5b4b8a",
               "Weather / trailer": "#4a7ba6", "Weather / battery": "#7fa8c9",
               "Mechanical / wiring": "#4f7d5a", "Blind-route assumption": "#d6a21e"}
STATE_COL = {"driving": "#4f7d5a", "stopped": "#c2542d", "no data": "#e4e4e4"}


def load_plans():
    plan = json.load(open(MODEL_OUT / "strategy_aryaman_idk.json"))
    files = sorted(glob.glob(str(MODEL_OUT / "strategy_as_given_resolve_day2_*.json")))
    asg = json.load(open(files[-1])) if files else None
    return plan, asg


def stage_km(day):
    out = {"stage1": 0.0, "loop": 0.0, "stage2": 0.0}
    for s in day.get("stages", []):
        out[s["stage"]] = out.get(s["stage"], 0.0) + float(s["distance_km"])
    return out


def build_table(plan, asg):
    act_d = pd.read_csv(HERE / "actuals_by_day.csv", index_col="day")
    act_p = pd.read_csv(HERE / "actuals_by_phase.csv")
    rows = []
    for d in range(1, 9):
        p = plan["days"][str(d)]
        sk = stage_km(p)
        r = dict(day=d, plan_km=p["distance_km"], plan_stage1=sk["stage1"], plan_loop=sk["loop"],
                 plan_stage2=sk["stage2"], plan_loops=p["n_loops"], plan_eta=p["eta"],
                 plan_speed=p["speed_avg_kmh"], plan_stop_min=p["stop_time_s"] / 60.0,
                 plan_wh_per_km=p["motor_energy_wh"] / p["distance_km"],
                 plan_soc_start=p["soc_start_pct"], plan_soc_end=p["soc_end_pct"])
        if asg and "eta" in asg["days"].get(str(d), {}):
            a = asg["days"][str(d)]
            ak = stage_km(a)
            r.update(asg_km=a["distance_km"], asg_stage1=ak["stage1"], asg_loop=ak["loop"], asg_stage2=ak["stage2"],
                     asg_loops=a["n_loops"], asg_eta=a["eta"], asg_speed=a["speed_avg_kmh"],
                     asg_soc_start=a["soc_start_pct"], asg_soc_end=a["soc_end_pct"])
        r["actual_km"] = ACTUAL_KM[d]
        ad = act_d.loc[d]
        r.update(tel_km=ad.km_telemetry, act_avg_moving=ad.avg_moving_kmh, act_p95=ad.p95_kmh,
                 act_stopped_min=ad.stopped_min_logged, act_silent_min=ad.silent_min_in_span,
                 act_wh_per_km=ad.motor_wh_per_km, act_solar_w=ad.avg_solar_w,
                 act_start=str(ad.drive_start)[11:16] if pd.notna(ad.drive_start) else "",
                 act_end=str(ad.drive_end)[11:16] if pd.notna(ad.drive_end) else "")
        for ph in ("stage1", "loop", "stage2"):
            g = act_p[(act_p.day == d) & (act_p.phase == ph)]
            r[f"act_{ph}_tel_km"] = float(g.km_telemetry.sum()) if len(g) else 0.0
        rows.append(r)
    t = pd.DataFrame(rows).set_index("day")
    t.to_csv(HERE / "plan_vs_actual_by_day.csv")
    return t


def plan_blocks(day, start_h):
    """Planned drive / stop blocks (hours since midnight); stops drawn at the end of each stage."""
    out, h = [], start_h
    for st in day["stages"]:
        el, stop = st["elapsed_s"] / 3600.0, st["stop_min"] / 60.0
        out.append((h, h + el - stop, "driving"))
        if stop > 0:
            out.append((h + el - stop, h + el, "stopped"))
        h += el
    return out


def fig_waterfall(t):
    steps = []
    for d in range(1, 9):
        short = t.plan_km[d] - t.actual_km[d]
        if d in (2, 3) and "asg_km" in t and pd.notna(t.asg_km.get(d)):
            route = t.plan_km[d] - t.asg_km[d]
            steps.append((f"D{d}\nroute", route, CAUSE_COLOR["Blind-route assumption"]))
            steps.append((f"D{d}\nfailures", short - route, CAUSE_COLOR[CAUSE[d][0]]))
        else:
            steps.append((f"D{d}", short, CAUSE_COLOR[CAUSE[d][0]]))
    start = float(t.plan_km.sum())
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(0, start, color=PLAN_C)
    ax.text(0, start + 25, f"{start:,.0f}", ha="center", fontweight="bold")
    level = start
    for i, (lab, delta, col) in enumerate(steps, 1):
        ax.bar(i, -delta, bottom=level, color=col)
        ax.text(i, max(level, level - delta) + 25, f"{-delta:+.0f}", ha="center", fontsize=9)
        level -= delta
    n = len(steps) + 1
    ax.bar(n, level, color=ACT_C)
    ax.text(n, level + 25, f"{level:,.1f}", ha="center", fontweight="bold")
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(["Plan"] + [s[0] for s in steps] + ["Driven"], fontsize=8.5)
    ax.set_ylabel("km")
    ax.set_ylim(0, start * 1.1)
    names = list(dict.fromkeys(CAUSE[d][0] for d in range(1, 9)))
    if any("route" in s[0] for s in steps):
        names.append("Blind-route assumption")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=CAUSE_COLOR[n_]) for n_ in names], names,
              fontsize=8, ncol=3, loc="upper right", frameon=False)
    ax.set_title("Deployed plan (km) to distance driven, by day and primary cause")
    fig.tight_layout()
    fig.savefig(HERE / "fig_waterfall.png")
    plt.close(fig)


def fig_phase_bars(t):
    fig, axs = plt.subplots(1, 8, figsize=(13, 4.2), sharey=True)
    cols = {"stage1": "#9bb7d4", "loop": "#e0b66b", "stage2": "#8fbf9f"}
    for d, ax in zip(range(1, 9), axs):
        plan = [t.plan_stage1[d], t.plan_loop[d], t.plan_stage2[d]]
        act = [t[f"act_{p}_tel_km"][d] for p in ("stage1", "loop", "stage2")]
        unlogged = max(t.actual_km[d] - sum(act), 0.0)
        for x, vals in ((0, plan), (1, act + [unlogged])):
            b = 0
            for k, v in enumerate(vals):
                c = list(cols.values())[k] if k < 3 else "#cfcfcf"
                ax.bar(x, v, bottom=b, color=c, width=0.8, edgecolor="white", linewidth=0.5)
                b += v
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["plan", "driven"], fontsize=8)
        ax.set_title(f"Day {d}", fontsize=9)
    axs[0].set_ylabel("km")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in list(cols.values()) + ["#cfcfcf"]]
    fig.legend(h, ["stage 1", "loops", "stage 2", "driven, not logged"], ncol=4, loc="lower center", frameon=False)
    fig.suptitle("Distance by stage: plan vs telemetry (grey = driven during logger drop-outs / unlabelled)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(HERE / "fig_phase_distance.png")
    plt.close(fig)


def fig_routes():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    spec = {
        2: dict(model=[("stage 1", 71.5), ("10 loops x 9.1", 91.0), ("stage 2", 231.0)],
                given=[("stage 1", 71.4), ("loop 34.9", 34.9), ("stage 2", 230.6)]),
        3: dict(model=[("stage 1", 0.0), ("stage 2", 212.6), ("6 loops x 37.4", 224.1)],
                given=[("stage 1", 117.2), ("loop 16.4", 16.4), ("stage 2", 102.7)]),
    }
    for ax, d in zip(axs, (2, 3)):
        for y, key in ((1, "model"), (0, "given")):
            left = 0
            for lab, km in spec[d][key]:
                if km <= 0:
                    continue
                ax.barh(y, km, left=left, color="#cfcfcf" if "loop" in lab else "#9bb7d4", edgecolor="white")
                if km > 60:
                    ax.text(left + km / 2, y, f"{lab}\n{km:.0f} km", ha="center", va="center", fontsize=8)
                else:
                    ax.annotate(lab, (left + km / 2, y + 0.4), ha="center", va="bottom", fontsize=7.5)
                left += km
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["issued route\n(one loop pass)", "model's\nroute"])
        ax.set_title(f"Day {d}" + (" (half blind)" if d == 2 else " (blind)"))
        ax.set_xlabel("km")
    fig.suptitle("Route the model planned on vs route issued")
    fig.tight_layout()
    fig.savefig(HERE / "fig_routes_d2_d3.png")
    plt.close(fig)


def _legend_states(ax, ncol=4, y=1.07, plan_label="planned stops (control stop + loop stops)"):
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#4f7d5a", "#5b5b5b", "#c2542d", "#e4e4e4")]
    ax.legend(h, ["driving", plan_label, "stopped (telemetry)", "no telemetry"], ncol=ncol,
              loc="upper center", bbox_to_anchor=(0.5, y), frameon=False, fontsize=8)


def fig_timeline(plan):
    tl = pd.read_csv(HERE / "timeline_runs.csv")
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for d in range(1, 9):
        y = (8 - d) * 3
        for a, b, st in plan_blocks(plan["days"][str(d)], 9.0 if d == 1 else 8.0):
            ax.barh(y + 1, b - a, left=a, height=0.8, color=STATE_COL["driving"] if st == "driving" else "#5b5b5b")
        for _, r in tl[tl.day == d].iterrows():
            ax.barh(y, r.end_h - r.start_h, left=r.start_h, height=0.8, color=STATE_COL[r.state])
        ax.text(5.6, y + 0.5, f"Day {d}", ha="right", va="center", fontweight="bold")
    ax.axvline(17, color="black", lw=0.8, ls="--")
    ax.set_xlim(4.6, 17.6)
    ax.set_yticks([])
    ax.set_xticks(range(6, 18))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(6, 18)], fontsize=8)
    ax.grid(axis="y", visible=False)
    _legend_states(ax)
    ax.set_title("Day timeline: top bar = plan, bottom bar = what the logger saw (dashed = 17:00 close)", pad=22)
    fig.tight_layout()
    fig.savefig(HERE / "fig_timeline.png")
    plt.close(fig)


def fig_timeline_d23(plan, asg):
    if not asg:
        return
    tl = pd.read_csv(HERE / "timeline_runs.csv")
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
    for ax, d in zip(axs, (2, 3)):
        for y, lab, src in ((2, "plan (blind route)", plan), (1, "re-plan (issued route)", asg)):
            for a, b, st in plan_blocks(src["days"][str(d)], 8.0):
                ax.barh(y, b - a, left=a, height=0.7, color=STATE_COL["driving"] if st == "driving" else "#5b5b5b")
            ax.text(5.4, y, lab, ha="right", va="center", fontsize=8)
        for _, r in tl[tl.day == d].iterrows():
            ax.barh(0, r.end_h - r.start_h, left=r.start_h, height=0.7, color=STATE_COL[r.state])
        ax.text(5.4, 0, "driven", ha="right", va="center", fontsize=8)
        ax.axvline(17, color="black", lw=0.8, ls="--")
        ax.set_yticks([])
        ax.set_title(f"Day {d}", loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axs[1].set_xlim(3.4, 17.6)
    axs[1].set_xticks(range(6, 18))
    axs[1].set_xticklabels([f"{h:02d}:00" for h in range(6, 18)], fontsize=8)
    _legend_states(axs[0], y=1.4, plan_label="planned stops")
    fig.tight_layout()
    fig.savefig(HERE / "fig_timeline_d2_d3.png")
    plt.close(fig)


def fig_speed(t):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(1, 9)
    w = 0.38
    ax.bar(x - w / 2, t.plan_speed, w, color=PLAN_C, label="plan (day average)")
    ax.bar(x + w / 2, t.act_avg_moving.fillna(0), w, color=ACT_C, label="driven (avg while moving)")
    ax.scatter(x + w / 2, t.act_p95.fillna(0), color="black", s=14, zorder=3, label="driven 95th percentile")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d}" for d in x])
    ax.set_ylabel("km/h")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    ax.set_title("Speed: planned vs driven")
    fig.tight_layout()
    fig.savefig(HERE / "fig_speed.png")
    plt.close(fig)


def fig_energy(t):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(1, 9)
    w = 0.38
    axs[0].bar(x - w / 2, t.plan_wh_per_km, w, color=PLAN_C, label="plan")
    axs[0].bar(x + w / 2, t.act_wh_per_km.fillna(0), w, color=ACT_C, label="driven")
    axs[0].set_title("Motor energy per km (Wh/km, net of regen)")
    axs[0].legend(frameon=False)
    axs[1].bar(x, t.act_solar_w.fillna(0), 0.6, color=ACT_C)
    axs[1].set_title("Array power logged (W, all four MPPT inputs)")
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels([f"D{d}" for d in x])
    fig.tight_layout()
    fig.savefig(HERE / "fig_energy.png")
    plt.close(fig)


def main():
    plan, asg = load_plans()
    t = build_table(plan, asg)
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(t.round(1).T.to_string())
    fig_waterfall(t)
    fig_phase_bars(t)
    fig_routes()
    fig_timeline(plan)
    fig_timeline_d23(plan, asg)
    fig_speed(t)
    fig_energy(t)


if __name__ == "__main__":
    main()
