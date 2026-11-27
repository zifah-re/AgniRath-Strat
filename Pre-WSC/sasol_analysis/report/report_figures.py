"""Figures for the final report, drawn from the committed analysis CSVs (no new analysis here)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PVA = HERE.parent / "plan_vs_actual"
OUT = HERE / "fig"
OUT.mkdir(exist_ok=True)
DARK, MID, LIGHT, GREY, RED = "#365F91", "#4F81BD", "#B8CCE4", "#8C8C8C", "#C0504D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555", "axes.linewidth": .7,
                     "xtick.color": "#333", "ytick.color": "#333", "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300)
    plt.close(fig)


def waterfall():
    t = pd.read_csv(PVA / "plan_vs_actual_by_day.csv").set_index("day")
    steps = [("Deployed plan", 3202.0, "total")]
    route = {d: t.asg_km[d] - t.blind_km[d] for d in (2, 3)}
    steps += [(f"Day {d} route as issued", route[d], "route") for d in (2, 3)]
    for d in range(1, 9):
        planned = t.plan_km[d] + route.get(d, 0.0)
        steps.append((f"Day {d}", t.actual_km[d] - planned, "loss"))
    steps.append(("Driven", float(t.actual_km.sum()), "total"))
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    run = 0.0
    for i, (lab, v, kind) in enumerate(steps):
        if kind == "total":
            ax.bar(i, v, color=DARK, width=.65)
            ax.text(i, v + 40, f"{v:.0f}", ha="center", fontsize=7.5)
            run = v
        else:
            lo = run + min(v, 0)
            ax.bar(i, abs(v), bottom=lo, color=MID if kind == "route" else RED, width=.65)
            ax.text(i, run + max(v, 0) + 40, f"{v:+.0f}", ha="center", fontsize=7)
            run += v
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("distance (km)")
    ax.set_ylim(0, 3600)
    save(fig, "waterfall.png")


def climb_eta():
    g = pd.read_csv(PVA / "separate_by_grade.csv")
    lab = ["descent\n< -0.5 %", "flat\n|g| < 0.5 %", "climb\n0.5-1.5 %", "climb\n> 1.5 %"]
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    ax.bar(x - .18, g.eta_needed_track_roadload, .36, color=GREY, label="with track-test Crr and CdA")
    ax.bar(x + .18, g.eta_needed_fitted_roadload, .36, color=MID, label="with Crr and CdA fitted from the race data")
    ax.axhline(0.95, color=DARK, ls="--", lw=.9)
    ax.text(3.45, 0.965, "assumed 0.95", fontsize=7, color=DARK, ha="right", va="bottom")
    for i, n in enumerate(g.n):
        ax.text(i, 0.03, f"{n} windows", ha="center", fontsize=6.5, color="white")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.5)
    ax.set_ylabel("drivetrain efficiency\nneeded to match energy"); ax.set_ylim(0, 1.25)
    ax.legend(fontsize=7, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.02))
    save(fig, "climb_eta.png")


def shading():
    s = pd.read_csv(PVA / "shading_by_relaz.csv")
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    sty = {"A": (LIGHT, "o", "A (rear left)"), "B": ("#E6B9B8", "o", "B (rear right)"),
           "C": (RED, "s", "C (front right)"), "D": (DARK, "s", "D (front left)")}
    for c, (col, mk, lab) in sty.items():
        ax.plot(s.bin, s[f"{c}_mean"], marker=mk, color=col, lw=1.3, ms=4, label=lab)
    ax.set_xticks(range(-180, 181, 45))
    ax.set_xlabel("sun direction relative to the car (deg): 0 ahead, +90 on the right, -90 on the left")
    ax.set_ylabel("string output per m²\n/ best string"); ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7, ncol=4, loc="lower center")
    save(fig, "shading.png")


def energy_balance():
    e = pd.read_csv(PVA / "energy_balance_by_stage.csv")
    e = e[e.day.isin([1, 2, 3, 6, 7])].reset_index(drop=True)
    lab = [f"D{d} {dict(stage1='S1', stage2='S2', loop='loop')[p]}" for d, p in zip(e.day, e.phase)]
    x = np.arange(len(e))
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.bar(x - .27, -e.measured_net_wh / 1000, .27, color=DARK, label="measured")
    ax.bar(x, -e.original_net_wh / 1000, .27, color=GREY, label="original model")
    ax.bar(x + .27, -e.refined_net_wh / 1000, .27, color=MID, label="refined model")
    ax.axhline(0, color="#333", lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.5)
    ax.set_ylabel("battery energy used (kWh)\n(negative = charged)")
    ax.legend(fontsize=7, ncol=1, loc="upper right")
    save(fig, "energy_balance.png")


if __name__ == "__main__":
    waterfall(); climb_eta(); shading(); energy_balance()
    print(sorted(p.name for p in OUT.glob("*.png")))
