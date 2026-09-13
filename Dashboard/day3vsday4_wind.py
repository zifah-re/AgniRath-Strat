"""
wind_analysis.py -- Day 3 vs Day 4 wind-speed comparison (Sasol Solar Challenge)

Compares the mean_*.jsonl weather forecast files, leg-by-leg (Stage 1 vs
Stage 1, Stage 2 vs Stage 2, Loop vs Loop), using the wind_speed_10m /
wind_direction_10m fields already in each file.

Run from inside your Dashboard/ folder:
    python wind_analysis.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

DASH = Path(".")          # <- run this from inside Dashboard/
SOLAR = DASH / "Solar"
LOCAL_TZ = timezone(timedelta(hours=2))  # SAST, UTC+2

# Matched leg-by-leg pairs: (label, Day3 file, Day4 file)
PAIRS = [
    ("Stage 1",
     SOLAR / "mean_Day3_Stage 1_Stage 1.jsonl",
     SOLAR / "mean_2026 Sasol Solar Challenge Route (Publish)_Day 4_13 Sept Stage 1 Kimberley to Postmasburg.jsonl"),
    ("Stage 2",
     SOLAR / "mean_Day3_Stage 2_Stage 2.jsonl",
     SOLAR / "mean_2026 Sasol Solar Challenge Route (Publish)_Day 4_13 Sept Stage 2 Postmasburg to Olifantshoek.jsonl"),
    ("Loop",
     SOLAR / "mean_Day3_Loop_Jan Kempdorp Loop.jsonl",
     SOLAR / "mean_2026 Sasol Solar Challenge Route (Publish)_Day 4_Postmasburg Loop.jsonl"),
]

# Race-window filter: only count daylight/race hours, not the overnight
# forecast hours the files also carry.
RACE_HOUR_START = 6
RACE_HOUR_END = 18


def load(path: Path) -> dict:
    with open(path) as f:
        payload = json.load(f)[0]
    times = [datetime.fromisoformat(r["period_end"]).astimezone(LOCAL_TZ) for r in payload["data"]]
    ws = np.array([r["wind_speed_10m"] for r in payload["data"]], dtype=float)
    wd = np.array([r["wind_direction_10m"] for r in payload["data"]], dtype=float)
    return {"lat": payload["lat"], "lon": payload["lon"], "times": times, "ws": ws, "wd": wd}


def race_window_mask(times: list[datetime]) -> np.ndarray:
    return np.array([RACE_HOUR_START <= t.hour <= RACE_HOUR_END for t in times])


def circular_mean_deg(deg: np.ndarray) -> float:
    """Mean wind direction, handling the 0/360 wraparound correctly."""
    rad = np.radians(deg)
    return float(np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360)


def summarize(label: str, data: dict) -> dict:
    mask = race_window_mask(data["times"])
    ws = data["ws"][mask]
    wd = data["wd"][mask]
    return {
        "label": label,
        "lat": data["lat"], "lon": data["lon"],
        "mean_ws": float(ws.mean()),
        "max_ws": float(ws.max()),
        "min_ws": float(ws.min()),
        "std_ws": float(ws.std()),
        "mean_wd": circular_mean_deg(wd),
    }


def print_hourly_table(day3: dict, day4: dict, leg_label: str) -> None:
    print(f"\n  Hourly wind speed (m/s), local time -- {leg_label}")
    print(f"  {'time':>6s}  {'Day3':>6s}  {'Day4':>6s}  {'Day4-Day3':>10s}")
    t3, ws3 = day3["times"], day3["ws"]
    t4, ws4 = day4["times"], day4["ws"]
    for hh in range(RACE_HOUR_START, RACE_HOUR_END + 1):
        # nearest sample to the top of each hour, per file
        i3 = min(range(len(t3)), key=lambda i: abs(t3[i].hour * 60 + t3[i].minute - hh * 60))
        i4 = min(range(len(t4)), key=lambda i: abs(t4[i].hour * 60 + t4[i].minute - hh * 60))
        v3, v4 = ws3[i3], ws4[i4]
        print(f"  {hh:02d}:00  {v3:6.2f}  {v4:6.2f}  {v4 - v3:+10.2f}")


def main() -> None:
    print("=" * 78)
    print(f"WIND SPEED COMPARISON: Day 3 (09-12) vs Day 4 (09-13), race window "
          f"{RACE_HOUR_START:02d}:00-{RACE_HOUR_END:02d}:00 local")
    print("=" * 78)

    rows = []
    for leg_label, f3, f4 in PAIRS:
        d3 = load(f3)
        d4 = load(f4)
        s3 = summarize(f"Day3 {leg_label}", d3)
        s4 = summarize(f"Day4 {leg_label}", d4)
        rows.append((leg_label, s3, s4))

        print(f"\n--- {leg_label} ---")
        print(f"  Day3: lat={s3['lat']:8.3f} lon={s3['lon']:8.3f}  "
              f"mean={s3['mean_ws']:5.2f} m/s  max={s3['max_ws']:5.2f}  "
              f"min={s3['min_ws']:5.2f}  std={s3['std_ws']:5.2f}  "
              f"mean_dir={s3['mean_wd']:5.1f} deg")
        print(f"  Day4: lat={s4['lat']:8.3f} lon={s4['lon']:8.3f}  "
              f"mean={s4['mean_ws']:5.2f} m/s  max={s4['max_ws']:5.2f}  "
              f"min={s4['min_ws']:5.2f}  std={s4['std_ws']:5.2f}  "
              f"mean_dir={s4['mean_wd']:5.1f} deg")
        pct = (s4["mean_ws"] / s3["mean_ws"] - 1.0) * 100.0
        print(f"  -> Day4 is {pct:+.1f}% vs Day3 on mean wind speed "
              f"({s4['mean_ws'] - s3['mean_ws']:+.2f} m/s)")

        print_hourly_table(d3, d4, leg_label)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'Leg':10s} {'Day3 mean':>10s} {'Day4 mean':>10s} {'Day4/Day3':>10s} {'Day3 dir':>9s} {'Day4 dir':>9s}")
    for leg_label, s3, s4 in rows:
        ratio = s4["mean_ws"] / s3["mean_ws"]
        print(f"{leg_label:10s} {s3['mean_ws']:10.2f} {s4['mean_ws']:10.2f} {ratio:9.2f}x "
              f"{s3['mean_wd']:8.1f}\u00b0 {s4['mean_wd']:8.1f}\u00b0")

    print("\nNote: this is wind speed/direction only. Whether higher wind on Day 4 "
          "actually costs more energy depends on heading vs wind direction "
          "(headwind vs tailwind vs crosswind) along each leg, which needs the "
          "route bearing at each point -- not modeled here. A uniform Cd*A drag "
          "increase from higher ambient wind is a reasonable first-order flag, "
          "but a proper number needs relative airspeed (vehicle vector + wind "
          "vector), not just wind speed magnitude.")


if __name__ == "__main__":
    main()