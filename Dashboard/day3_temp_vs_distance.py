"""Plot temperature vs distance for a chained set of stages (e.g. Day 3: Stage 1 ->
control stop -> Loop -> Stage 2), assuming a constant cruising speed since no MPC /
strategy velocity profile exists for these stages yet.

For each leg:
  - A route file (.kml, .kml.save, or .gpx) supplies the distance axis: its own
    point-to-point cumulative distance, used directly (no resampling -- air_temp only
    varies smoothly at the km/hour scale anyway, so the route's native point spacing
    is more than fine enough).
  - A constant speed (--speed, default 60 km/h) turns that distance axis into elapsed
    time since the leg started.
  - A Solar mean_openmeteo_*.jsonl file (produced separately by
    route_temperature_forecast.py) supplies the air_temp forecast to interpolate
    against, exactly like temp_vs_distance.py does.

Legs are chained in time:
  Stage 1 starts at --start-time.
  The Loop starts --control-stop-hours (default 1.0) after Stage 1 finishes.
  Stage 2 starts immediately when the Loop finishes (no additional gap).

Usage:
    python day3_temp_vs_distance.py \
        --stage1-route "Saves/Day3_Stage 1_Stage 1.kml.save" \
        --stage1-solar "Solar/mean_openmeteo_Day3_Stage1.jsonl" \
        --loop-route   "Saves/Day3_Loop_Jan Kempdorp Loop.kml.save" \
        --loop-solar   "Solar/mean_openmeteo_Day3_Loop.jsonl" \
        --stage2-route "Saves/Day3_Stage 2_Stage 2.kml.save" \
        --stage2-solar "Solar/mean_openmeteo_Day3_Stage2.jsonl" \
        --start-time 2026-09-12T06:00:00+00:00 \
        --speed 60 --control-stop-hours 1 \
        --outdir day3_plots

Each of the three Solar files must already exist -- generate them first with
route_temperature_forecast.py (one call per route file).
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
# Route parsing (kept in sync with route_temperature_forecast.py)
# ---------------------------------------------------------------------------


def parse_kml_save_route(path: Path) -> list[tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    coords = (data.get("profile") or {}).get("Coordinates")
    if not coords:
        raise SystemExit(f"No profile.Coordinates found in {path}")
    return [(lat, lon) for lat, lon in coords]


def parse_kml_route(path: Path) -> list[tuple[float, float]]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(path.read_text(encoding="utf-8"))
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    points: list[tuple[float, float]] = []
    for coords_el in root.iter("coordinates"):
        text = (coords_el.text or "").strip()
        if not text:
            continue
        for triple in text.split():
            parts = triple.split(",")
            if len(parts) < 2:
                continue
            lon, lat = float(parts[0]), float(parts[1])
            points.append((lat, lon))

    if not points:
        raise SystemExit(f"No <coordinates> found in {path}")
    return points


def parse_gpx_route(path: Path) -> list[tuple[float, float]]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(path.read_text(encoding="utf-8"))
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    points = []
    for pt in list(root.iter("trkpt")) + list(root.iter("wpt")):
        lat, lon = pt.get("lat"), pt.get("lon")
        if lat is not None and lon is not None:
            points.append((float(lat), float(lon)))

    if not points:
        raise SystemExit(f"No trkpt/wpt found in {path}")
    return points


def load_route_points(path: Path) -> list[tuple[float, float]]:
    name_lower = path.name.lower()
    if name_lower.endswith(".kml.save"):
        return parse_kml_save_route(path)
    if name_lower.endswith(".kml"):
        return parse_kml_route(path)
    if name_lower.endswith(".gpx"):
        return parse_gpx_route(path)
    raise SystemExit(f"Unsupported route file type: {path.name} (use .kml, .kml.save, or .gpx)")


def route_distance_km(points: list[tuple[float, float]]) -> list[float]:
    distances = [0.0]
    for prev, curr in zip(points, points[1:]):
        distances.append(distances[-1] + geodesic(prev, curr).km)
    return distances


# ---------------------------------------------------------------------------
# Solar waypoint loading (kept in sync with temp_vs_distance.py)
# ---------------------------------------------------------------------------


def load_solar_waypoints(path: Path) -> list[dict]:
    waypoints = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(waypoints, list) or not waypoints:
        raise SystemExit(f"{path} did not contain a non-empty list of waypoints")
    return waypoints


def waypoint_distances_km(waypoints: list[dict]) -> list[float]:
    distances = [0.0]
    for prev, curr in zip(waypoints, waypoints[1:]):
        step_km = geodesic((prev["lat"], prev["lon"]), (curr["lat"], curr["lon"])).km
        distances.append(distances[-1] + step_km)
    return distances


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def temp_at_time(waypoint: dict, t: datetime) -> float:
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
# Constant-speed distance -> temperature
# ---------------------------------------------------------------------------


def build_constant_speed_curve(
    waypoints: list[dict],
    waypoint_dist_km: list[float],
    trace_distance_km: list[float],
    speed_kmh: float,
    start_time: datetime,
) -> tuple[list[float], list[float], list[datetime], datetime]:
    """Return (distance_km, air_temp, absolute_time, finish_time) for a constant-speed leg."""
    abs_times = [start_time + timedelta(hours=d / speed_kmh) for d in trace_distance_km]

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

    finish_time = abs_times[-1]
    return trace_distance_km, temps, abs_times, finish_time


def run_leg(
    name: str,
    route_path: Path,
    solar_path: Path,
    speed_kmh: float,
    start_time: datetime,
    outdir: Path,
) -> datetime:
    """Build + plot one leg, return its finish time so the caller can chain the next leg."""
    points = load_route_points(route_path)
    trace_distance_km = route_distance_km(points)

    waypoints = load_solar_waypoints(solar_path)
    waypoint_dist_km = waypoint_distances_km(waypoints)

    distance_km, air_temp, abs_times, finish_time = build_constant_speed_curve(
        waypoints, waypoint_dist_km, trace_distance_km, speed_kmh, start_time
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distance_km, air_temp, color="tab:red", linewidth=1.5)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Air temperature (\u00b0C)")
    ax.set_title(
        f"{name}: temperature vs distance\n"
        f"start {start_time.isoformat()}  |  finish {finish_time.isoformat()}  |  {speed_kmh:.0f} km/h constant"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{name}_temp_vs_distance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(
        f"{name}: {distance_km[-1]:.1f} km, start {start_time.isoformat()}, "
        f"finish {finish_time.isoformat()} -> saved {out_path}"
    )
    return finish_time


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1-route", type=Path, required=True)
    parser.add_argument("--stage1-solar", type=Path, required=True)
    parser.add_argument("--loop-route", type=Path, required=True)
    parser.add_argument("--loop-solar", type=Path, required=True)
    parser.add_argument("--stage2-route", type=Path, required=True)
    parser.add_argument("--stage2-solar", type=Path, required=True)
    parser.add_argument(
        "--start-time",
        required=True,
        help="ISO timestamp with UTC offset for Stage 1's start, e.g. 2026-09-12T06:00:00+00:00",
    )
    parser.add_argument("--speed", type=float, default=60.0, help="Constant cruising speed in km/h (default: 60)")
    parser.add_argument(
        "--control-stop-hours",
        type=float,
        default=1.0,
        help="Hold time at the control stop between Stage 1 finishing and the Loop starting (default: 1.0)",
    )
    parser.add_argument("--outdir", type=Path, default=Path("day3_plots"), help="Directory to save the 3 plots into")
    args = parser.parse_args()

    stage1_start = parse_timestamp(args.start_time)

    stage1_finish = run_leg("day3_stage1", args.stage1_route, args.stage1_solar, args.speed, stage1_start, args.outdir)

    loop_start = stage1_finish + timedelta(hours=args.control_stop_hours)
    print(f"Control stop: {args.control_stop_hours:.2f} h -> loop starts {loop_start.isoformat()}")
    loop_finish = run_leg("day3_loop", args.loop_route, args.loop_solar, args.speed, loop_start, args.outdir)

    stage2_start = loop_finish  # no gap: loop end == stage 2 start
    run_leg("day3_stage2", args.stage2_route, args.stage2_solar, args.speed, stage2_start, args.outdir)


if __name__ == "__main__":
    main()