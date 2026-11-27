"""Dark versions of the report figures for the slide deck (same CSVs, no new analysis)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PVA = HERE.parent / "plan_vs_actual"
OUT = HERE / "slide_fig"
OUT.mkdir(exist_ok=True)

BG, INK, MUTED, LINE, GREY, RED = "#15181C", "#ECE8E1", "#A4A9B0", "#30353C", "#6E747C", "#E04B43"
plt.rcParams.update({
    "font.family": "IBM Plex Sans", "font.size": 19, "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": LINE,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2, "legend.frameon": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": LINE, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=100)
    plt.close(fig)


def waterfall():
    t = pd.read_csv(PVA / "plan_vs_actual_by_day.csv").set_index("day")
    route = {d: t.asg_km[d] - t.blind_km[d] for d in (2, 3)}
    steps = [("Plan", 3202.0, "total")]
    steps += [(f"D{d} route", route[d], "route") for d in (2, 3)]
    for d in range(1, 9):
        steps.append((f"Day {d}", t.actual_km[d] - (t.plan_km[d] + route.get(d, 0.0)), "loss"))
    steps.append(("Driven", float(t.actual_km.sum()), "total"))
    fig, ax = plt.subplots(figsize=(16.64, 6.4))
    run = 0.0
    for i, (lab, v, kind) in enumerate(steps):
        if kind == "total":
            ax.bar(i, v, color=INK, width=.62)
            ax.text(i, v + 60, f"{v:.0f}", ha="center", fontsize=21, color=INK, weight="medium")
            run = v
        else:
            ax.bar(i, abs(v), bottom=run + min(v, 0), color=GREY if kind == "route" else RED, width=.62)
            ax.text(i, run + max(v, 0) + 60, f"{v:+.0f}".replace("-", "−"), ha="center", fontsize=18, color=MUTED)
            run += v
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps])
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_ylabel("km")
    ax.set_ylim(0, 3600)
    save(fig, "waterfall.png")


def climb_eta():
    g = pd.read_csv(PVA / "separate_by_grade.csv")
    lab = ["descent", "flat", "climb 0.5–1.5 %", "climb > 1.5 %"]
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    ax.bar(x - .19, g.eta_needed_track_roadload, .36, color=GREY, label="track-test Crr, CdA")
    ax.bar(x + .19, g.eta_needed_fitted_roadload, .36, color=INK, label="race-fitted Crr, CdA")
    ax.axhline(0.95, color=RED, ls="--", lw=2)
    ax.text(3.55, 0.97, "assumed 0.95", fontsize=18, color=RED, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=17)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_ylabel("drivetrain efficiency needed"); ax.set_ylim(0, 1.15)
    ax.legend(fontsize=17, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
    save(fig, "climb_eta.png")


def shading():
    s = pd.read_csv(PVA / "shading_by_relaz.csv")
    fig, ax = plt.subplots(figsize=(16.64, 6.0))
    sty = {"A": (MUTED, "-", "o", "A  rear left"), "B": (GREY, "-", "o", "B  rear right"),
           "C": (RED, "--", "s", "C  front right"), "D": (RED, "-", "s", "D  front left")}
    for c, (col, ls, mk, lab) in sty.items():
        ax.plot(s.bin, s[f"{c}_mean"], ls=ls, marker=mk, color=col, lw=2.6, ms=8, label=lab)
    ax.set_xticks(range(-180, 181, 45))
    ax.set_xticklabels(["−180", "−135", "−90\nsun on left", "−45", "0\nahead", "45", "90\nsun on right", "135", "180"])
    ax.set_xlabel("sun direction relative to the car (deg)")
    ax.set_ylabel("output / best string"); ax.set_ylim(0, 1.25); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.legend(fontsize=17, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "shading.png")


def energy_balance():
    e = pd.read_csv(PVA / "energy_balance_by_stage.csv")
    e = e[e.day.isin([1, 2, 3, 6, 7])].reset_index(drop=True)
    lab = [f"D{d} {dict(stage1='S1', stage2='S2', loop='loop')[p]}" for d, p in zip(e.day, e.phase)]
    x = np.arange(len(e))
    fig, ax = plt.subplots(figsize=(16.64, 6.2))
    ax.bar(x - .27, -e.measured_net_wh / 1000, .27, color=INK, label="measured")
    ax.bar(x, -e.original_net_wh / 1000, .27, color=GREY, label="original model")
    ax.bar(x + .27, -e.refined_net_wh / 1000, .27, color=RED, label="refined model")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=17)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_ylabel("battery energy used (kWh)")
    ax.legend(fontsize=17, ncol=3, loc="upper right")
    save(fig, "energy_balance.png")


if __name__ == "__main__":
    waterfall(); climb_eta(); shading(); energy_balance()
    print(sorted(p.name for p in OUT.glob("*.png")))
