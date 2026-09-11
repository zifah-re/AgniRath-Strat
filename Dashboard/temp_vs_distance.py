"""Plot air temperature vs distance for a single race stage.

This combines two pieces of dashboard data:

  * a ``Solar/mean_*.jsonl`` file: a list of waypoints spaced along a route,
    each holding a 5-minute-resolution forecast time series (``air_temp``,
    UTC timestamps) at that waypoint's (lat, lon).
  * a strategy-output json (e.g. ``output/strategy_aryaman_idk.json``): for
    a given day + stage, its ``trace`` holds ``distance_km`` and
    ``velocity_kmh`` arrays describing how fast the car is planned to be
    moving at every point along the stage.

Pipeline:
  1. Convert each Solar waypoint's (lat, lon) into a cumulative distance
     along the route, using the great-circle distance between consecutive
     waypoints.
  2. Use the strategy velocity profile to turn "distance travelled" into
     "elapsed time since the stage started" (dt = d(distance) / speed),
     then add a stage start time to get an absolute UTC timestamp for every
     distance sample in the trace.
  3. For each (distance, absolute_time) pair in the trace, interpolate the
     forecast air temperature: first in time (between the two nearest
     5-minute forecast samples at a waypoint), then in space (between the
     two waypoints that bracket that distance).
  4. Plot / save temperature vs distance.

Example:
    python temp_vs_distance.py \
        "Solar/mean_2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 1 Boiketlong to Rustenburg.jsonl" \
        --strategy output/strategy_aryaman_idk.json --day 1 --stage stage1 \
        --start-time 2026-09-10T07:00:00+00:00 \
        --output day1_stage1_temp_vs_distance.png

If ``--start-time`` is omitted, the first timestamp found in the Solar file
is used (i.e. it assumes the stage starts right at the beginning of the
forecast window).
"""

from __future__ import annotations

import argparse
import bisect
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from geopy.distance import geodesic

# ---------------------------------------------------------------------------
# Solar waypoint loading
# ---------------------------------------------------------------------------


def load_solar_waypoints(path: Path) -> list[dict]:
    """Read a Solar/mean_*.jsonl file into a list of waypoint dicts.

    Despite the .jsonl extension, these files are a single JSON array, not
    newline-delimited JSON, so we just json.load() the whole thing.
    """
    try:
        waypoints = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"Solar file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path} is not valid JSON: {error}") from error

    if not isinstance(waypoints, list) or not waypoints:
        raise SystemExit(f"{path} did not contain a non-empty list of waypoints")

    for waypoint in waypoints:
        if "lat" not in waypoint or "lon" not in waypoint or "data" not in waypoint:
            raise SystemExit(f"{path} has a waypoint missing lat/lon/data: {waypoint}")
        if not waypoint["data"]:
            raise SystemExit(f"{path} has a waypoint with no forecast samples")

    return waypoints


def waypoint_distances_km(waypoints: list[dict]) -> list[float]:
    """Cumulative great-circle distance (km) along the chain of waypoints."""
    distances = [0.0]
    for prev, curr in zip(waypoints, waypoints[1:]):
        step_km = geodesic((prev["lat"], prev["lon"]), (curr["lat"], curr["lon"])).km
        distances.append(distances[-1] + step_km)
    return distances


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def temp_at_time(waypoint: dict, t: datetime) -> float:
    """Linearly interpolate a waypoint's air_temp forecast at time t.

    Clips to the first/last sample if t falls outside the forecast window.
    """
    samples = waypoint["data"]
    times = [parse_timestamp(s["period_end"]) for s in samples]

    if t <= times[0]:
        return samples[0]["air_temp"]
    if t >= times[-1]:
        return samples[-1]["air_temp"]

    idx = bisect.bisect_right(times, t) - 1
    t0, t1 = times[idx], times[idx + 1]
    temp0, temp1 = samples[idx]["air_temp"], samples[idx + 1]["air_temp"]

    span = (t1 - t0).total_seconds()
    frac = (t - t0).total_seconds() / span if span > 0 else 0.0
    return temp0 + frac * (temp1 - temp0)


# ---------------------------------------------------------------------------
# Strategy velocity-profile loading
# ---------------------------------------------------------------------------


def load_velocity_trace(strategy_path: Path, day: str, stage: str) -> tuple[list[float], list[float]]:
    """Return (distance_km, velocity_kmh) trace arrays for a day + stage."""
    try:
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"Strategy file not found: {strategy_path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"{strategy_path} is not valid JSON: {error}") from error

    days = strategy.get("days", {})
    if day not in days:
        raise SystemExit(f"Day {day!r} not found in {strategy_path}. Available days: {sorted(days)}")

    day_data = days[day]
    if stage not in day_data:
        available = day_data.get("stage_names", [])
        raise SystemExit(f"Stage {stage!r} not found for day {day}. Available stages: {available}")

    trace = day_data[stage].get("trace")
    if not trace or "distance_km" not in trace or "velocity_kmh" not in trace:
        raise SystemExit(f"Day {day} / stage {stage} has no usable trace with distance_km + velocity_kmh")

    return trace["distance_km"], trace["velocity_kmh"]


def elapsed_seconds(distance_km: list[float], velocity_kmh: list[float]) -> list[float]:
    """Integrate the velocity profile into elapsed time (s) since the stage start."""
    elapsed = [0.0]
    for i in range(1, len(distance_km)):
        d_dist = distance_km[i] - distance_km[i - 1]
        if d_dist <= 0:
            elapsed.append(elapsed[-1])
            continue
        avg_v = (velocity_kmh[i] + velocity_kmh[i - 1]) / 2
        if avg_v <= 0:
            avg_v = max(velocity_kmh[i], velocity_kmh[i - 1], 1.0)
        elapsed.append(elapsed[-1] + (d_dist / avg_v) * 3600.0)
    return elapsed


# ---------------------------------------------------------------------------
# Combine: distance -> temperature
# ---------------------------------------------------------------------------


def build_temperature_curve(
    waypoints: list[dict],
    waypoint_dist_km: list[float],
    trace_distance_km: list[float],
    trace_velocity_kmh: list[float],
    start_time: datetime,
) -> tuple[list[float], list[float], list[datetime]]:
    """Return (distance_km, air_temp, absolute_time) aligned to the trace."""
    elapsed = elapsed_seconds(trace_distance_km, trace_velocity_kmh)
    abs_times = [start_time + timedelta(seconds=s) for s in elapsed]

    temps: list[float] = []
    for dist, t in zip(trace_distance_km, abs_times):
        if dist <= waypoint_dist_km[0]:
            temps.append(temp_at_time(waypoints[0], t))
            continue
        if dist >= waypoint_dist_km[-1]:
            temps.append(temp_at_time(waypoints[-1], t))
            continue

        idx = bisect.bisect_right(waypoint_dist_km, dist) - 1
        d0, d1 = waypoint_dist_km[idx], waypoint_dist_km[idx + 1]
        frac = (dist - d0) / (d1 - d0) if d1 > d0 else 0.0

        temp0 = temp_at_time(waypoints[idx], t)
        temp1 = temp_at_time(waypoints[idx + 1], t)
        temps.append(temp0 + frac * (temp1 - temp0))

    return trace_distance_km, temps, abs_times


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("solar_file", type=Path, help="Path to a Solar/mean_*.jsonl file for one stage")
    parser.add_argument(
        "--strategy",
        type=Path,
        default=Path("output/strategy_aryaman_idk.json"),
        help="Path to the strategy output json holding the velocity profile (default: %(default)s)",
    )
    parser.add_argument("--day", default="1", help="Day key in the strategy json's 'days' dict (default: 1)")
    parser.add_argument(
        "--stage",
        default="stage1",
        choices=["stage1", "loop", "stage2"],
        help="Which stage's trace to use (default: stage1)",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="ISO timestamp (with UTC offset) for the start of the stage, "
        "e.g. 2026-09-10T07:00:00+00:00. Defaults to the first timestamp in the Solar file.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Save the plot to this file instead of showing it")
    args = parser.parse_args()

    waypoints = load_solar_waypoints(args.solar_file)
    waypoint_dist_km = waypoint_distances_km(waypoints)

    trace_distance_km, trace_velocity_kmh = load_velocity_trace(args.strategy, args.day, args.stage)

    if args.start_time:
        start_time = parse_timestamp(args.start_time)
    else:
        start_time = parse_timestamp(waypoints[0]["data"][0]["period_end"])
        print(f"No --start-time given, defaulting to first Solar timestamp: {start_time.isoformat()}")

    distance_km, air_temp, abs_times = build_temperature_curve(
        waypoints, waypoint_dist_km, trace_distance_km, trace_velocity_kmh, start_time
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distance_km, air_temp, color="tab:red", linewidth=1.5)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Air temperature (\u00b0C)")
    ax.set_title(f"{args.solar_file.name}\nDay {args.day} / {args.stage} temperature vs distance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()