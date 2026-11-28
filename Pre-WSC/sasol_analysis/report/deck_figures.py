"""Light-card figures for the slide deck (Agnirath look). Same CSVs and model outputs as the report."""
import glob
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PVA = HERE.parent / "plan_vs_actual"
LSTM = HERE.parent / "lstm_soc"
MODEL = REPO / "Model"
OUT = HERE / "deck_fig"
OUT.mkdir(exist_ok=True)

ORANGE, DEEP, PEACH, INK, GREY, LGREY = "#F26B1D", "#B8440E", "#F9B98C", "#262626", "#8A8A8A", "#D4D4D4"
plt.rcParams.update({
    "font.family": "Lato", "font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9A9A9A", "axes.linewidth": 1.0, "axes.labelcolor": "#444", "text.color": INK,
    "xtick.color": "#444", "ytick.color": "#444", "legend.frameon": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": "#ECECEC", "grid.linewidth": 1.0, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 14,
})


def save(fig, name):
    fig.tight_layout(pad=1.6)
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


# ---------- route ----------
def stage_files():
    """Planned (published + probable Day 3) stage files in race order, loops excluded."""
    files = sorted(glob.glob(str(MODEL / "data/processed/*.save")))
    out = []
    for f in files:
        d = json.load(open(f))
        name = Path(f).name
        if "Loop" in name:
            continue
        day = 3 if name.startswith("Day 3") else int(d["folder_name"].split()[1])
        stage = 1 if "Stage 1" in name else 2
        out.append((day, stage, d["profile"]))
    return sorted(out, key=lambda x: (x[0], x[1]))


def route_map():
    import shapefile
    shp = shapefile.Reader("/tmp/gpd/geopandas/datasets/naturalearth_lowres/naturalearth_lowres.shp")
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    ax.set_facecolor("#FFF7EE")
    for sr in shp.shapeRecords():
        rec = sr.record.as_dict()
        sa = rec["name"] == "South Africa"
        pts = sr.shape.points
        parts = list(sr.shape.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            ring = np.array(pts[a:b])
            ax.add_patch(Polygon(ring, closed=True, fc="#FDBE7B" if sa else "#FBE7D2",
                                 ec="#E9A262" if sa else "#F1D3B4", lw=1.2))
    for day, stage, p in stage_files():
        c = np.array(p["Coordinates"])
        ax.plot(c[:, 1], c[:, 0], color=DEEP, lw=3.2, solid_capstyle="round", zorder=3)
    towns = json.load(open(MODEL / "data/raw/points.json"))
    keep = {"Sasolburg": "Sasolburg", "Zeerust": "Zeerust", "Vryburg": "Vryburg", "Kimberley": "Kimberley",
            "Upington": "Upington", "Springbok": "Springbok", "Clanwilliam": "Clanwilliam", "Paarl": "Paarl"}
    for t in towns:
        lab = next((v for k, v in keep.items() if t["note"].startswith(k)), None)
        if lab is None:
            continue
        ax.scatter(t["lon"], t["lat"], s=70, color="white", ec=DEEP, lw=2.2, zorder=4)
        dx, ha = (0.35, "left") if lab not in ("Springbok", "Clanwilliam", "Paarl", "Upington") else (-0.35, "right")
        if lab == "Upington":
            dx, ha = (0.0, "center")
        dy = 0.45 if lab in ("Upington", "Zeerust") else 0.0
        if lab == "Zeerust":
            dx, ha = (0.0, "center")
        lab = {"Sasolburg": "Sasolburg  (start)", "Paarl": "Paarl  (finish)"}.get(lab, lab)
        ax.text(t["lon"] + dx, t["lat"] + dy, lab, ha=ha, va="center", fontsize=15, color="#5A2A0A", zorder=5)
    ax.set_xlim(16.2, 30.0); ax.set_ylim(-35.2, -23.6)
    ax.set_aspect(1 / np.cos(np.radians(29)))
    ax.axis("off")
    save(fig, "route_map.png")


def altitude():
    xs, ys, ticks = [], [], []
    x0 = 0.0
    for day, stage, p in stage_files():
        d = np.array(p["Distance"]); a = np.array(p["Altitude"])
        if stage == 1:
            ticks.append((x0, f"D{day}"))
        xs.append(x0 + d); ys.append(a); x0 += d[-1]
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    x = np.concatenate(xs); y = np.concatenate(ys)
    ax.fill_between(x, 0, y, color=PEACH, alpha=.55, lw=0)
    ax.plot(x, y, color=DEEP, lw=1.8)
    for xt, lab in ticks:
        ax.axvline(xt, color=LGREY, lw=1, zorder=0)
        ax.text(xt + 12, 2120, lab, fontsize=14, color=GREY, va="center")
    ax.set_ylim(0, 2250); ax.set_yticks([0, 500, 1000, 1500, 2000]); ax.set_xlim(0, x[-1])
    ax.set_xlabel("route distance, stages only (km)"); ax.set_ylabel("altitude (m)")
    save(fig, "altitude.png")


# ---------- the plan ----------
def plan_profile():
    d = json.load(open(MODEL / "output/strategy_aryaman_idk.json"))
    days = d["days"] if isinstance(d["days"], list) else list(d["days"].values())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    x0 = 0.0
    for i, day in enumerate(days):
        for st in day["stages"]:
            tr = st["trace"]
            x = x0 + np.array(tr["distance_km"])
            a1.plot(x, tr["velocity_kmh"], color=ORANGE, lw=1.4)
            a2.plot(x, tr["soc_pct"], color=INK, lw=1.5)
            x0 = x[-1]
        for ax in (a1, a2):
            ax.axvline(x0, color=LGREY, lw=1, zorder=0)
        a1.text(x0 - day["distance_km"] / 2, 92, f"D{i + 1}", ha="center", fontsize=14, color=GREY)
    a2.axhline(20, color=ORANGE, ls="--", lw=1.4)
    a2.text(30, 23, "SOC floor 20 %", color=DEEP, fontsize=13)
    a1.set_ylim(30, 100); a1.set_ylabel("speed (km/h)")
    a2.set_ylim(0, 105); a2.set_ylabel("battery SOC (%)")
    a2.set_xlabel("cumulative distance (km)"); a2.set_xlim(0, x0)
    save(fig, "plan_profile.png")


# ---------- race vs plan ----------
def waterfall():
    from matplotlib.patches import Patch
    t = pd.read_csv(PVA / "plan_vs_actual_by_day.csv").set_index("day")
    route = {d: t.asg_km[d] - t.blind_km[d] for d in (2, 3)}
    steps = [("Planned", 3202.0, "total")]
    steps += [(f"Day {d}\nreroute", route[d], "route") for d in (2, 3)]
    for d in range(1, 9):
        steps.append((f"Day {d}", t.actual_km[d] - (t.plan_km[d] + route.get(d, 0.0)), "loss"))
    steps.append(("Driven", float(t.actual_km.sum()), "total"))
    fig, ax = plt.subplots(figsize=(12.0, 4.9))
    run = 0.0
    for i, (lab, v, kind) in enumerate(steps):
        if kind == "total":
            ax.bar(i, v, color=INK, width=.64)
            ax.text(i, v + 55, f"{v:.0f} km", ha="center", fontsize=15, weight="bold")
            run = v
        else:
            ax.bar(i, abs(v), bottom=run + min(v, 0), color=LGREY if kind == "route" else ORANGE, width=.64)
            ax.text(i, run + max(v, 0) + 55, f"{v:+.0f}".replace("-", "\u2212"), ha="center", fontsize=14, color="#444")
            run += v
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[0] for s in steps], fontsize=13)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("distance (km)"); ax.set_ylim(0, 3700)
    ax.legend(handles=[Patch(color=INK, label="planned / driven total"),
                       Patch(color=LGREY, label="route changed by the organisers"),
                       Patch(color=ORANGE, label="km not driven: electrical and mechanical faults")],
              fontsize=13, loc="upper right", ncol=1)
    save(fig, "waterfall.png")


def climb_eta():
    g = pd.read_csv(PVA / "separate_by_grade.csv")
    lab = ["descent", "flat", "climb\n0.5–1.5 %", "climb\n> 1.5 %"]
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(x - .19, g.eta_needed_track_roadload, .36, color=LGREY, label="track-test Crr, CdA")
    ax.bar(x + .19, g.eta_needed_fitted_roadload, .36, color=ORANGE, label="race-fitted Crr, CdA")
    for i, v in enumerate(g.eta_needed_fitted_roadload):
        ax.text(i + .19, v + .02, f"{v:.2f}", ha="center", fontsize=14, color=DEEP, weight="bold")
    ax.axhline(0.95, color=INK, ls="--", lw=1.6)
    ax.text(3.5, 0.97, "assumed 0.95", fontsize=14, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=14); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("drivetrain efficiency needed"); ax.set_ylim(0, 1.18)
    ax.legend(fontsize=14, loc="upper left", ncol=2)
    save(fig, "climb_eta.png")


def shading():
    s = pd.read_csv(PVA / "shading_by_relaz.csv")
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    sty = {"A": (GREY, "-", "o", "MPPT A, rear left"), "B": (LGREY, "-", "o", "MPPT B, rear right"),
           "C": (DEEP, "--", "s", "MPPT C, front right"), "D": (ORANGE, "-", "s", "MPPT D, front left")}
    for c, (col, ls, mk, lab) in sty.items():
        ax.plot(s.bin, s[f"{c}_mean"], ls=ls, marker=mk, color=col, lw=2.4, ms=7, label=lab)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["behind", "sun on left", "ahead", "sun on right", "behind"])
    ax.set_ylabel("output / best MPPT"); ax.set_ylim(0, 1.3); ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.legend(fontsize=13, ncol=2, loc="upper center")
    ax.set_ylim(0, 1.45)
    save(fig, "shading.png")


def energy_balance():
    e = pd.read_csv(PVA / "energy_balance_by_stage.csv")
    e = e[e.day.isin([1, 2, 3, 6, 7])].reset_index(drop=True)
    lab = [f"D{d}\n{dict(stage1='S1', stage2='S2', loop='loop')[p]}" for d, p in zip(e.day, e.phase)]
    x = np.arange(len(e))
    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    ax.bar(x - .27, -e.measured_net_wh / 1000, .27, color=INK, label="measured")
    ax.bar(x, -e.original_net_wh / 1000, .27, color=LGREY, label="original model")
    ax.bar(x + .27, -e.refined_net_wh / 1000, .27, color=ORANGE, label="refined model")
    ax.axhline(0, color="#777", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=13); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("kWh used")
    ax.legend(fontsize=14, ncol=3, loc="upper right")
    save(fig, "energy_balance.png")


def d2s2():
    e = pd.read_csv(PVA / "energy_balance_by_stage.csv")
    r = e[(e.day == 2) & (e.phase == "stage2")].iloc[0]
    vals = [-r.measured_net_wh / 1000, -r.original_net_wh / 1000, -r.refined_net_wh / 1000]
    labs = ["measured\non the car", "original\nmodel", "refined\nmodel"]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.bar(range(3), vals, color=[INK, LGREY, ORANGE], width=.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.06, f"{v:.2f} kWh", ha="center", fontsize=17, weight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=15); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("battery energy used (kWh)"); ax.set_ylim(0, 3.3)
    save(fig, "d2s2.png")


def sensitivity():
    s = pd.read_csv(PVA / "sensitivity_energy.csv")
    s["mag"] = (s.energy_change_pct_plus10.abs() + s.energy_change_pct_minus10.abs()) / 2
    s = s.sort_values("mag")
    names = {"drivetrain efficiency": "drivetrain efficiency", "array efficiency (solar offset)": "array efficiency",
             "mass": "mass", "Crr": "Crr", "CdA": "CdA"}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    y = np.arange(len(s))
    cols = [ORANGE if n == "drivetrain efficiency" else LGREY for n in s.parameter]
    ax.barh(y, s.mag, color=cols, height=.6)
    for i, v in enumerate(s.mag):
        ax.text(v + .2, i, f"{v:.1f} %", va="center", fontsize=14)
    ax.set_yticks(y); ax.set_yticklabels([names[n] for n in s.parameter])
    ax.set_xlabel("energy change for a 10 % change"); ax.set_xlim(0, 12)
    ax.grid(axis="x", color="#ECECEC"); ax.grid(axis="y", visible=False)
    save(fig, "sensitivity.png")


def lstm_rmse():
    r = pd.read_csv(LSTM / "results/summary.csv")
    r = r[r.method.isin(["Voltage lookup (OCV + IR)", "Coulomb counting (live start)", "LSTM"])]
    lab = {"Voltage lookup (OCV + IR)": "voltage\nlookup", "Coulomb counting (live start)": "coulomb\ncounting",
           "LSTM": "LSTM"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.arange(len(r))
    ax.bar(x - .18, r.rmse, .34, color=[ORANGE if m == "LSTM" else LGREY for m in r.method], label="RMS error")
    ax.bar(x + .18, r.max_abs, .34, color=[DEEP if m == "LSTM" else GREY for m in r.method], label="worst case")
    for i, (a, b) in enumerate(zip(r.rmse, r.max_abs)):
        ax.text(i - .18, a + .3, f"{a:.1f}", ha="center", fontsize=14)
        ax.text(i + .18, b + .3, f"{b:.1f}", ha="center", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels([lab[m] for m in r.method]); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("SOC error (%)"); ax.set_ylim(0, 21)
    ax.legend(fontsize=13, loc="upper center", ncol=2)
    save(fig, "lstm_rmse.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["route_map", "altitude", "plan_profile", "waterfall", "climb_eta", "shading",
                             "energy_balance", "d2s2", "sensitivity", "lstm_rmse"]
    for w in which:
        globals()[w]()
    print(sorted(p.name for p in OUT.glob("*.png")))
