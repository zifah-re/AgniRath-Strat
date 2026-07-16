"""
plot_solar.py
-------------
Reads solar_input_*.jsonl files (fields: air_temp, dni, ghi, period_end),
converts period_end UTC → IST, computes seconds since midnight IST,
and plots all three metrics vs time of day.

Usage:
    python plot_solar.py                          # auto-discovers solar_input_*.jsonl in cwd
    python plot_solar.py file1.jsonl file2.jsonl  # explicit files
    python plot_solar.py --output-dir plots/   # save PNGs into a subfolder
"""

import json
import sys
import glob
import argparse
from pathlib import Path
from datetime import timezone, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# ── Constants ─────────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))
FIELDS = ["ghi", "dni", "air_temp"]
FIELD_LABELS = {
    "ghi": "GHI — Global Horizontal Irradiance (W/m²)",
    "dni": "DNI — Direct Normal Irradiance (W/m²)",
    "air_temp": "Air Temperature (°C)",
}
FIELD_COLORS = ["#e07b00", "#1a6eb5", "#d63b3b"]  # orange, blue, red


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Add ist_seconds (seconds since midnight IST) and ist_date columns."""
    # period_end already has +00:00 offset; parse directly
    dt_utc = pd.to_datetime(df["period_end"], utc=True)
    dt_ist = dt_utc.dt.tz_convert(IST)

    midnight_ist = dt_ist.dt.normalize()  # 00:00:00 IST of that day
    df["period_end"]=dt_ist #changing utc to ist
    df["ist_seconds"] = (dt_ist - midnight_ist).dt.total_seconds()
    df["ist_date"] = dt_ist.dt.strftime("%d %b %Y")
    df["ist_time_label"] = dt_ist.dt.strftime("%H:%M")
    return df


def seconds_to_hhmm(sec, _pos=None):
    """Tick formatter: seconds-since-midnight → HH:MM."""
    sec = int(sec)
    h, m = divmod(sec // 60, 60)
    return f"{h:02d}:{m:02d}"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot solar JSONL files vs IST time of day")
    parser.add_argument("files", nargs="*", help="JSONL paths (default: auto-discover)")
    parser.add_argument("--output-dir", default=".", help="Directory to save plots (default: current dir)")
    args = parser.parse_args()

    paths = args.files or sorted(glob.glob("solar_input_*.jsonl"))
    if not paths:
        print("No JSONL files found. Pass file paths or run from the folder containing them.")
        sys.exit(1)

    print(f"Loading {len(paths)} file(s): {[Path(p).name for p in paths]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        df = load_jsonl(p)
        df = parse_timestamps(df)
        df.sort_values("ist_seconds", inplace=True)

        date_label = df["ist_date"].iloc[0]
        stem = Path(p).stem

        # ── Figure: 3 subplots (GHI, DNI, Air Temp) ──────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
        fig.patch.set_facecolor("#f9f9f9")

        for ax, field, color in zip(axes, FIELDS, FIELD_COLORS):
            ax.set_facecolor("#ffffff")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#cccccc")
            ax.spines[["top", "right"]].set_visible(False)

            ax.plot(
                df["ist_seconds"],
                df[field],
                color=color,
                linewidth=1.8,
                alpha=0.9,
                marker="o" if len(df) < 30 else None,
                markersize=3,
            )
            ax.set_ylabel(FIELD_LABELS[field], fontsize=9, labelpad=8)

        # ── Shared X-axis ─────────────────────────────────────────────────────
        ax_bottom = axes[-1]
        ax_bottom.set_xlabel("Time of Day (IST)", fontsize=10, labelpad=8)
        ax_bottom.xaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_hhmm))
        ax_bottom.xaxis.set_major_locator(ticker.MultipleLocator(3600))   # every 1 hr
        ax_bottom.xaxis.set_minor_locator(ticker.MultipleLocator(1800))   # every 30 min
        plt.setp(ax_bottom.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

        fig.suptitle(f"Solar Irradiance & Temperature  |  {date_label}",
                     fontsize=13, fontweight="bold", y=1.005)

        plt.tight_layout(h_pad=1.5)
        out_path = output_dir / f"{stem}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()