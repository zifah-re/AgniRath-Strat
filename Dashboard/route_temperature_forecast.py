"""Fetch a temperature/irradiance forecast along a route from Open-Meteo (free, no API
key needed) and save it in the same schema as the Dashboard's Solar/mean_*.jsonl files,
so it's a drop-in, more-accurate alternative to the Solcast-derived air_temp values.

Pipeline:
  1. Parse an ordered list of (lat, lon) points out of a .kml or .gpx route file.
  2. Resample that polyline down to N evenly-spaced waypoints by distance (same idea
     as the existing Solar files, which use ~17 waypoints per stage).
  3. Send ONE batched request to Open-Meteo's forecast API for all waypoints at once
     (it supports comma-separated lat/lon lists and returns one result per location).
  4. Reshape the response into the {"lat", "lon", "data": [{"period_end", "air_temp",
     "dni", "ghi", "wind_speed_10m", "wind_direction_10m"}]} schema used elsewhere in
     the Dashboard.

Example:
    python route_temperature_forecast.py \
        "gpx/2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 1 Boiketlong to Rustenburg.gpx" \
        "Solar/mean_openmeteo_Day 1 Stage 1.jsonl" \
        --waypoints 17 --forecast-days 3 --model best_match

Forecasts only reach ~16 days out (Open-Meteo's limit), so this is for upcoming stages,
not past ones — use --start-date/--end-date to pin it to specific race dates within that
window instead of "the next N days from today".
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests
from geopy.distance import geodesic

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = [
    "temperature_2m",
    "direct_normal_irradiance",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
]

# ---------------------------------------------------------------------------
# Route parsing
# ---------------------------------------------------------------------------


def parse_kml_route(path: Path) -> list[tuple[float, float]]:
    """Extract an ordered list of (lat, lon) points from every LineString in a KML file."""
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
    """Extract an ordered list of (lat, lon) points from every trkpt/wpt in a GPX file."""
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
    suffix = path.suffix.lower()
    if suffix == ".kml":
        return parse_kml_route(path)
    if suffix == ".gpx":
        return parse_gpx_route(path)
    raise SystemExit(f"Unsupported route file type: {suffix} (use .kml or .gpx)")


# ---------------------------------------------------------------------------
# Resample the route to evenly spaced waypoints
# ---------------------------------------------------------------------------


def cumulative_distances_km(points: list[tuple[float, float]]) -> list[float]:
    distances = [0.0]
    for prev, curr in zip(points, points[1:]):
        distances.append(distances[-1] + geodesic(prev, curr).km)
    return distances


def resample_waypoints(points: list[tuple[float, float]], n_waypoints: int) -> list[tuple[float, float]]:
    """Pick n_waypoints evenly spaced by distance along the route (endpoints included)."""
    if n_waypoints < 2:
        raise SystemExit("--waypoints must be at least 2")

    distances = cumulative_distances_km(points)
    total = distances[-1]
    if total == 0:
        raise SystemExit("Route has zero length (all points identical)")

    targets = [total * i / (n_waypoints - 1) for i in range(n_waypoints)]
    waypoints = []
    seg = 0
    for target in targets:
        while seg < len(distances) - 2 and distances[seg + 1] < target:
            seg += 1
        d0, d1 = distances[seg], distances[seg + 1]
        frac = (target - d0) / (d1 - d0) if d1 > d0 else 0.0
        lat0, lon0 = points[seg]
        lat1, lon1 = points[seg + 1]
        waypoints.append((lat0 + frac * (lat1 - lat0), lon0 + frac * (lon1 - lon0)))
    return waypoints


# ---------------------------------------------------------------------------
# Open-Meteo forecast fetching
# ---------------------------------------------------------------------------


def fetch_forecast(
    waypoints: list[tuple[float, float]],
    forecast_days: int,
    start_date: str | None,
    end_date: str | None,
    model: str,
    timeout: float = 30.0,
) -> list[dict]:
    """One batched request for all waypoints; returns Open-Meteo's per-location list."""
    params = {
        "latitude": ",".join(str(lat) for lat, _ in waypoints),
        "longitude": ",".join(str(lon) for _, lon in waypoints),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "models": model,
    }
    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    else:
        params["forecast_days"] = forecast_days

    try:
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as error:
        raise SystemExit(f"Open-Meteo request failed: {error}") from error

    payload = response.json()
    # With a single location Open-Meteo returns one object; with several, a list.
    return payload if isinstance(payload, list) else [payload]


def normalize_time(value: str) -> str:
    """Turn Open-Meteo's '2026-09-10T07:00' into '2026-09-10T07:00:00+00:00'."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Convert to the Dashboard's Solar/mean_*.jsonl schema
# ---------------------------------------------------------------------------


def to_solar_schema(waypoints: list[tuple[float, float]], forecasts: list[dict]) -> list[dict]:
    if len(forecasts) != len(waypoints):
        raise SystemExit(
            f"Open-Meteo returned {len(forecasts)} location(s) but {len(waypoints)} waypoints were requested"
        )

    out = []
    for (lat, lon), forecast in zip(waypoints, forecasts):
        hourly = forecast.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [None] * len(times))
        dni = hourly.get("direct_normal_irradiance", [None] * len(times))
        ghi = hourly.get("shortwave_radiation", [None] * len(times))
        wind_speed = hourly.get("wind_speed_10m", [None] * len(times))
        wind_dir = hourly.get("wind_direction_10m", [None] * len(times))

        data = [
            {
                "period_end": normalize_time(t),
                "air_temp": temps[i],
                "dni": dni[i],
                "ghi": ghi[i],
                "wind_speed_10m": wind_speed[i],
                "wind_direction_10m": wind_dir[i],
            }
            for i, t in enumerate(times)
        ]
        out.append({"lat": lat, "lon": lon, "data": data})
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("route_file", type=Path, help="Route .kml or .gpx file")
    parser.add_argument("output", type=Path, help="Output path, e.g. Solar/mean_openmeteo_<stage>.jsonl")
    parser.add_argument("--waypoints", type=int, default=17, help="Number of evenly spaced sample points (default: 17)")
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=7,
        help="Days ahead to fetch if --start-date/--end-date aren't given (max 16, default: 7)",
    )
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD, restricts the forecast window (pair with --end-date)")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD, restricts the forecast window (pair with --start-date)")
    parser.add_argument(
        "--model",
        default="best_match",
        help="Open-Meteo weather model, e.g. best_match, ecmwf_ifs, gfs_seamless, icon_seamless (default: best_match)",
    )
    args = parser.parse_args()

    points = load_route_points(args.route_file)
    waypoints = resample_waypoints(points, args.waypoints)

    print(f"Resampled route into {len(waypoints)} waypoints; querying Open-Meteo ({args.model})...")
    forecasts = fetch_forecast(waypoints, args.forecast_days, args.start_date, args.end_date, args.model)

    solar_schema = to_solar_schema(waypoints, forecasts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(solar_schema, indent=4), encoding="utf-8")

    n_samples = len(solar_schema[0]["data"]) if solar_schema else 0
    print(f"Wrote {len(solar_schema)} waypoints x {n_samples} samples to {args.output}")


if __name__ == "__main__":
    main()