"""
Plots the stitched Sasol day files (cleaned_logs/dayN.jsonl), one figure per day per plot.

Usage:
  python plotter.py                         all days, all plots, saved to plots/
  python plotter.py --days 1 3              only some days
  python plotter.py --plots velocity        only some plots
  python plotter.py --show                  also open the figures

Adding a plot: write a function (axes, records, day) -> None that draws on the list of
axes (one per panel, sharing the time axis), then register it in PLOTS with a name, a
title and its panel height ratios. Styling, race-window shading and saving are handled
for every plot by draw_day().
"""

import argparse
import json
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from stitch_days import GAP_S, MAIN_MARKER, OUT_DIR as LOG_DIR, RACE_DAYS, REPO_ROOT, SAST

PLOT_DIR = LOG_DIR.parent / "plots"

# Solcast forecasts saved by the Dashboard, one file per route section, each holding a
# 5-min series for every point along that section. Where a section was re-fetched, the
# "_2" revision (the one the strategy scripts used) replaces the original.
SOLAR_DIR = REPO_ROOT / "Dashboard" / "Solar"
FORECAST_GLOBS = {
    1: ["mean_2026*_Day 1 _*.jsonl"],
    2: ["mean_2026*_Day 2 Half Blind_*.jsonl", "mean_SSC ROUTE FINAL_Day 2*.jsonl"],
    3: ["mean_Day3_*.jsonl"],
    4: ["mean_2026*_Day 4_*.jsonl"],
    5: ["mean_2026*_Day 5 _*.jsonl"],
    6: ["mean_2026*_Day 6 _*.jsonl", "mean_Day 6_One way*.jsonl"],
    7: ["mean_2026*_Day 7_*.jsonl"],
    8: ["mean_2026*_Day 8_*.jsonl"],
}

# Array model, same as the Dashboard day strategy scripts: flat deck sees GHI, MPPT D
# (canopy-shaded) delivers a quarter of a clean channel.
ARRAY_AREA_M2 = 5.78
ARRAY_EFFICIENCY = 0.24
MPPT_D_SHADE_FACTOR = 0.25
GHI_TO_W = ARRAY_AREA_M2 * ARRAY_EFFICIENCY * (3 + MPPT_D_SHADE_FACTOR) / 4

MPPT_CHANNELS = "ABCD"
POWER_BIN_S = 60  # MPPT power is averaged into bins this long before plotting

# AgniRath / Claude palette: near-black background, warm orange data, cream text
COLORS = {
    "background": "#141413",
    "panel": "#1F1E1D",
    "text": "#FAF9F5",
    "muted": "#B0AEA5",
    "grid": "#3D3D3A",
    "orange": "#D97757",
    "orange_light": "#F0A285",
    "outside_window": "#2A2927",
    # categorical slots 1-4 (dark-surface steps), one per MPPT channel
    "mppt_A": "#3987E5",
    "mppt_B": "#D95926",
    "mppt_C": "#199E70",
    "mppt_D": "#C98500",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["background"],
    "axes.facecolor": COLORS["panel"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["text"],
    "axes.titlecolor": COLORS["text"],
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.6,
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "text.color": COLORS["text"],
    "legend.facecolor": COLORS["panel"],
    "legend.edgecolor": COLORS["grid"],
    "font.size": 10,
})


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
def load_day(day):
    """All records of a stitched day, each with a parsed SAST datetime in r['t']."""
    records = []
    with open(LOG_DIR / f"day{day}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["t"] = datetime.fromisoformat(r["_rx_time"])
            records.append(r)
    return records


def series(records, key):
    """Times and values of one field, with NaN inserted at gaps > GAP_S so lines break there."""
    times, values = [], []
    for r in records:
        if key not in r:
            continue
        if times and (r["t"] - times[-1]).total_seconds() > GAP_S:
            times.append(times[-1])
            values.append(np.nan)
        times.append(r["t"])
        values.append(r[key])
    return times, values


def binned(times, values, bin_s):
    """Means of values over consecutive bin_s-long time bins; empty bins become NaN so lines break."""
    if not times:
        return [], np.array([])
    t0 = times[0]
    idx = np.array([int((t - t0).total_seconds() // bin_s) for t in times])
    vals = np.asarray(values, dtype=float)
    sums = np.bincount(idx, weights=vals)
    counts = np.bincount(idx)
    means = np.full(len(sums), np.nan)
    means[counts > 0] = sums[counts > 0] / counts[counts > 0]
    centres = [t0 + timedelta(seconds=(i + 0.5) * bin_s) for i in range(len(sums))]
    return centres, means


def forecast_files(day):
    files = {p for pattern in FORECAST_GLOBS[day] for p in SOLAR_DIR.glob(pattern)}
    revised = {p for p in files if p.stem.endswith("_2")}
    superseded = {p.with_name(p.stem[:-2] + p.suffix) for p in revised}
    return sorted(files - superseded)


def load_forecast_ghi(day):
    """Forecast GHI over every route point of the day: (times SAST, min, median, max).
    Each Solcast value is a 5-min period average, plotted at the middle of its period."""
    by_time = {}
    for path in forecast_files(day):
        with open(path, encoding="utf-8") as f:
            for point in json.load(f):
                for row in point["data"]:
                    by_time.setdefault(row["period_end"], []).append(row["ghi"])
    times = sorted(by_time)
    ghi = [by_time[t] for t in times]
    times = [datetime.fromisoformat(t).astimezone(SAST) - timedelta(minutes=2.5) for t in times]
    return (times, np.array([min(g) for g in ghi]), np.array([np.median(g) for g in ghi]),
            np.array([max(g) for g in ghi]))


# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------
def plot_velocity(axes, records, day):
    ax = axes[0]
    times, v = series([r for r in records if MAIN_MARKER in r], "Vehicle_Velocity")
    ax.plot(times, np.array(v) * 3.6, color=COLORS["orange"], linewidth=1.0)  # m/s -> km/h
    ax.set_ylabel("Vehicle velocity (km/h)")


def plot_solar(axes, records, day):
    """Top: forecast GHI along the day's route. Bottom: MPPT input power per channel and
    in total, against the power the array model expects from the forecast median."""
    ax_ghi, ax_p = axes
    f_times, g_min, g_med, g_max = load_forecast_ghi(day)

    ax_ghi.fill_between(f_times, g_min, g_max, color=COLORS["muted"], alpha=0.25, linewidth=0,
                        label="range over route points")
    ax_ghi.plot(f_times, g_med, color=COLORS["text"], linewidth=1.5, label="median over route")
    ax_ghi.set_ylabel("Forecast GHI (W/m²)")
    ax_ghi.set_ylim(bottom=0)
    ax_ghi.legend(loc="upper right", fontsize=8)

    main = [r for r in records if MAIN_MARKER in r]
    total = None
    for c in MPPT_CHANNELS:
        for r in main:  # input side; clip the sensor offset that reads slightly negative at idle
            r[f"_P_{c}"] = max(0.0, r[f"Input_Voltage_{c}"] * r[f"Input_Current_{c}"])
        times, p = series(main, f"_P_{c}")
        bt, bp = binned(times, p, POWER_BIN_S)
        ax_p.plot(bt, bp, color=COLORS[f"mppt_{c}"], linewidth=1.0, label=f"MPPT {c}")
        total = bp if total is None else total + bp
    ax_p.plot(bt, total, color=COLORS["text"], linewidth=2.0, label="Total (A+B+C+D)")
    ax_p.plot(f_times, g_med * GHI_TO_W, color=COLORS["orange_light"], linewidth=1.5, linestyle="--",
              label=f"Expected from forecast median ({ARRAY_AREA_M2} m², {ARRAY_EFFICIENCY:.0%}, D×{MPPT_D_SHADE_FACTOR})")
    ax_p.set_ylabel(f"MPPT input power (W, {POWER_BIN_S} s mean)")
    ax_p.set_ylim(bottom=0)
    ax_p.legend(loc="upper right", fontsize=8, ncol=3)


def plot_mppt_vi(axes, records, day):
    """Input voltage and input current of each MPPT channel. Current follows irradiance;
    voltage follows panel temperature and where the tracker sits on the I-V curve."""
    ax_v, ax_i = axes
    main = [r for r in records if MAIN_MARKER in r]
    for c in MPPT_CHANNELS:
        for ax, key in ((ax_v, f"Input_Voltage_{c}"), (ax_i, f"Input_Current_{c}")):
            bt, bv = binned(*series(main, key), POWER_BIN_S)
            ax.plot(bt, bv, color=COLORS[f"mppt_{c}"], linewidth=1.0, label=f"MPPT {c}")
    ax_v.set_ylabel(f"Input voltage (V, {POWER_BIN_S} s mean)")
    ax_i.set_ylabel(f"Input current (A, {POWER_BIN_S} s mean)")
    for ax in axes:
        ax.set_ylim(bottom=0)
    ax_v.legend(loc="lower right", fontsize=8, ncol=4)


PLOTS = {
    "velocity": ("Velocity vs time", plot_velocity, [1]),
    "solar": ("Forecast irradiance and MPPT power vs time", plot_solar, [1, 2]),
    "mppt_vi": ("MPPT input voltage and current vs time", plot_mppt_vi, [1, 1]),
}

# Plots that are not against time, drawn by their own function: (day, records, show) -> path
IV_POWER_GUIDES_W = (100, 200, 300, 400)


# ---------------------------------------------------------
# Figure framing
# ---------------------------------------------------------
def draw_day(day, name, records, show):
    date, h0, h1 = RACE_DAYS[day]
    title, plot_fn, heights = PLOTS[name]
    w0 = datetime.fromisoformat(f"{date}T{h0:02d}:00:00+02:00")
    w1 = datetime.fromisoformat(f"{date}T{h1:02d}:00:00+02:00")

    fig, axes = plt.subplots(len(heights), 1, figsize=(14, 2.5 + 2.5 * sum(heights)), sharex=True,
                             height_ratios=heights, squeeze=False)
    axes = list(axes[:, 0])
    plot_fn(axes, records, day)

    # Show the whole race window, plus any data recorded outside it
    x0 = min(w0, records[0]["t"]) if records else w0
    x1 = max(w1, records[-1]["t"]) if records else w1
    for ax in axes:
        ax.set_xlim(x0, x1)
        for a, b in ((x0, w0), (w1, x1)):
            if a < b:
                ax.axvspan(a, b, color=COLORS["outside_window"], zorder=0)
        for edge in (w0, w1):
            ax.axvline(edge, color=COLORS["orange_light"], linestyle="--", linewidth=0.8, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax = axes[-1]
    ax.xaxis.set_major_locator(mdates.HourLocator(tz=SAST))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45], tz=SAST))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=SAST))
    ax.set_xlabel("Time (SAST)")
    axes[0].set_title(f"Day {day} ({date}): {title}", loc="left", fontsize=13, fontweight="bold")
    axes[0].text(1.0, 1.02, f"race window {h0:02d}:00-{h1:02d}:00 (dashed); gaps > {GAP_S} s left blank",
                 transform=axes[0].transAxes, ha="right", va="bottom", color=COLORS["muted"], fontsize=8)

    fig.tight_layout()
    PLOT_DIR.mkdir(exist_ok=True)
    out = PLOT_DIR / f"day{day}_{name}.png"
    fig.savefig(out, dpi=150)
    if not show:
        plt.close(fig)
    return out


def draw_iv(day, records, show):
    """Operating points of each MPPT channel (input V vs I) inside the race window, with
    constant-power curves. A healthy channel clusters near its max-power voltage; shading
    shows up as a second cluster at higher voltage and low current."""
    date, h0, h1 = RACE_DAYS[day]
    in_window = [r for r in records if MAIN_MARKER in r and h0 <= r["t"].hour < h1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharex=True, sharey=True)
    volts = [r[f"Input_Voltage_{c}"] for r in in_window for c in MPPT_CHANNELS] or [30, 80]
    v_min, v_max = np.percentile(volts, 0.5) - 3, max(volts) * 1.03  # zoom past the rare near-0 V samples
    i_max = max((r[f"Input_Current_{c}"] for r in in_window for c in MPPT_CHANNELS), default=8) * 1.05
    v_grid = np.linspace(v_min, v_max, 200)
    for ax, c in zip(axes, MPPT_CHANNELS):
        for p in IV_POWER_GUIDES_W:
            ax.plot(v_grid, p / v_grid, color=COLORS["grid"], linewidth=0.8, zorder=1)
            ax.text(v_max * 0.98, p / (v_max * 0.98), f"{p} W", color=COLORS["muted"], fontsize=7,
                    ha="right", va="bottom")
        ax.scatter([r[f"Input_Voltage_{c}"] for r in in_window], [r[f"Input_Current_{c}"] for r in in_window],
                   s=3, alpha=0.15, color=COLORS[f"mppt_{c}"], linewidths=0, zorder=2)
        ax.set_title(f"MPPT {c}", loc="left", fontsize=11)
        ax.set_xlabel("Input voltage (V)")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Input current (A)")
    axes[0].set_xlim(v_min, v_max)
    axes[0].set_ylim(0, i_max)
    fig.suptitle(f"Day {day} ({date}): MPPT operating points (input I vs V)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.99, 0.95, f"every main packet {h0:02d}:00-{h1:02d}:00; grey curves are constant power",
             ha="right", color=COLORS["muted"], fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    PLOT_DIR.mkdir(exist_ok=True)
    out = PLOT_DIR / f"day{day}_iv.png"
    fig.savefig(out, dpi=150)
    if not show:
        plt.close(fig)
    return out


OTHER_PLOTS = {"iv": draw_iv}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    all_plots = list(PLOTS) + list(OTHER_PLOTS)
    parser.add_argument("--days", type=int, nargs="+", default=list(RACE_DAYS), choices=list(RACE_DAYS))
    parser.add_argument("--plots", nargs="+", default=all_plots, choices=all_plots)
    parser.add_argument("--show", action="store_true", help="open the figures after saving")
    args = parser.parse_args()

    for day in args.days:
        records = load_day(day)
        for name in args.plots:
            out = draw_day(day, name, records, args.show) if name in PLOTS else OTHER_PLOTS[name](day, records, args.show)
            print(f"Day {day}: {name} -> {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
