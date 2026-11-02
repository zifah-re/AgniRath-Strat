"""Download the historical Solcast samples for the 2025 WSC study.

The allocation intentionally counts *location/date* calls, rather than just
route locations: Day 1 has 16 calls and Days 2--5 have 32 calls each.
Run this once from this folder:

    python query_wsc_solar.py --execute

The script reads API keys from the existing team Solcast script without
executing it, so keys are not duplicated in this project or printed.
"""
from __future__ import annotations

import argparse
import ast
import json
import time as time_module
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from solcast import forecast, historic
from solcast.unmetered_locations import UNMETERED_LOCATIONS


HERE = Path(__file__).resolve().parent
SOLAR_DIR = HERE / "Solar25"
CHECKPOINT_PATH = SOLAR_DIR / ".wsc_query_checkpoint.json"
ACST = ZoneInfo("Australia/Darwin")  # ACST: UTC+09:30, no daylight saving
# The first configured key is known invalid. The two remaining keys each have
# 48 credits, so 10 locations × 9 date-batches = 90 requests is safe and even.
LOCATIONS_PER_DATE = 16

# The two dates are deliberately a sliding look-back window requested for the
# simulation.  There are nine location/date batches: 1 + 2 * 4 = 9.
QUERY_PLAN = {
    "Day1_WSC_Day 1.kml.save": (date(2025, 8, 24),),
    "Day2_WSC_Day 2.kml.save": (date(2025, 8, 24), date(2025, 8, 25)),
    "Day3_WSC_Day 3.kml.save": (date(2025, 8, 25), date(2025, 8, 26)),
    "Day4_WSC_Day 4.kml.save": (date(2025, 8, 26), date(2025, 8, 27)),
    "Day5_WSC_Day 5.kml.save": (date(2025, 8, 27), date(2025, 8, 28)),
}


def load_api_keys() -> list[str]:
    """Extract string API-key assignments from the existing team script safely."""
    key_script = HERE.parent.parent / "Group" / "Solcast Data comparison" / "solcast_scraping.py"
    tree = ast.parse(key_script.read_text(encoding="utf-8"), filename=str(key_script))
    old_keys = ["tqYlXYm5K8bVQ20coa5FJS-Wg5IHRAY8ac",
            "mdJAwni-zDwxDQZm1pRbuMog15flMNxC",
            "BIxzbRqwylC-hwNw1mHidlRYVBvOK9_g",
            "8ITfBFfjrwMZQeoCC9fLBnXqrCSGWIXm",]
    
    keys = ["F7s8YzfjgoNzcA1cX_2Gbqt5G8A3VsJp",
            "o2AmalzhezuK1qqOnkHOvHo67kB_UoSy",
            "x9F7mQ2pL8vR1tZ4aB6cY0wE3hJ5nU7dK",
            "E_rxZoBY9JeNAqjFmvoKNaJoara19FPa",
            "ztGpXqd6OoxwgiARn0pX7BKBu8_ZY3z8",
            "7d3D3jIfEE4bUJhEmKKgwyNquwR1Hr7M"]
    '''for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        if any(isinstance(target, ast.Name) and target.id.endswith("API_KEY") for target in node.targets):
            if node.value.value:
                keys.append(node.value.value)'''
    if not keys:
        raise RuntimeError(f"No API keys were found in {key_script}")
    if len(keys) < 3:
        raise RuntimeError("Expected the three team API-key assignments in the existing Solcast script")
    return keys


def route_samples(route_path: Path) -> list[tuple[float, float]]:
    profile = json.loads(route_path.read_text(encoding="utf-8"))["profile"]
    distances = np.asarray(profile["Distance"], dtype=float)
    coordinates = np.asarray(profile["Coordinates"], dtype=float)
    if distances[-1] <= 0 or len(coordinates) != len(distances):
        raise ValueError(f"Invalid distance/coordinate profile in {route_path.name}")

    # This follows Fallback Model/solar.py's evenly-spaced interior samples.
    query_distance = np.linspace(0.0, distances[-1], LOCATIONS_PER_DATE + 2)[1:-1]
    latitudes = np.interp(query_distance, distances, coordinates[:, 0])
    longitudes = np.interp(query_distance, distances, coordinates[:, 1])
    return [(float(lat), float(lon)) for lat, lon in zip(latitudes, longitudes)]


def fetch_one(api_key: str, lat: float, lon: float, query_date: date) -> list[dict]:
    # Cover the entire local calendar date.  The 08:00--17:00 driving window,
    # control-stop charging and evening hold are then available to the model.
    start = datetime.combine(query_date, time.min, tzinfo=ACST)
    end = datetime.combine(query_date, time.max, tzinfo=ACST)
    # The SDK sends api_key as an Authorization bearer header, never in a URL.
    response = historic.radiation_and_weather(
        latitude=lat, longitude=lon, start=start.isoformat(), end=end.isoformat(),
        period="PT5M", time_zone="utc", api_key=api_key,
    )
    rows = response.to_dict().get("estimated_actuals", [])
    wanted = ("period_end", "air_temp", "dni", "ghi", "wind_speed_10m", "wind_direction_10m")
    return [{name: row.get(name, 0.0) for name in wanted} for row in rows]


def check_keys() -> None:
    """Check key validity without consuming a transaction.

    Solcast documents this exact Sydney Opera House forecast coordinate as an
    unmetered location.  The returned X-RateLimit headers are a short-term
    rate-window limit, not a remaining-credit balance; Solcast has no public
    credit-balance endpoint.
    """
    sydney = UNMETERED_LOCATIONS['Sydney Opera House']
    for number, key in enumerate(load_api_keys(), start=1):
        try:
            response = forecast.radiation_and_weather(
                latitude=sydney['latitude'], longitude=sydney['longitude'], api_key=key,
            )
            response.to_dict()
            print(f"Active key {number}: HTTP {response.code}")
        except Exception as exc:
            print(f"Active key {number}: failed ({type(exc).__name__})")


def build_output_name(route_name: str) -> Path:
    return SOLAR_DIR / f"{Path(route_name).name.removesuffix('.kml.save')}_solar.json"


def call_id(route_name: str, sample_date: date, lat: float, lon: float) -> str:
    return f"{route_name}|{sample_date.isoformat()}|{lat:.9f}|{lon:.9f}"


def main(execute: bool) -> None:
    work = []
    for route_name, dates in QUERY_PLAN.items():
        route_path = HERE / route_name
        samples = route_samples(route_path)
        for sample_date in dates:
            work.extend((route_name, sample_date, lat, lon) for lat, lon in samples)

    print(f"Planned Solcast calls: {len(work)} (two active keys receive {len(work) // 2} calls each)")
    for route_name, dates in QUERY_PLAN.items():
        print(f"  {route_name}: {LOCATIONS_PER_DATE} locations × {len(dates)} date(s) = {LOCATIONS_PER_DATE * len(dates)}")
    if not execute:
        print(f"Dry run only. Add --execute to spend the {len(work)} historical-query calls.")
        return

    keys = load_api_keys()
    SOLAR_DIR.mkdir(exist_ok=True)
    checkpoint: dict[str, list[dict]] = {}
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        print(f"Resuming {len(checkpoint)} completed calls from checkpoint.")

    for call_number, (route_name, sample_date, lat, lon) in enumerate(work, start=1):
        identifier = call_id(route_name, sample_date, lat, lon)
        if identifier in checkpoint:
            continue
        print(f"[{call_number:03d}/{len(work)}] {route_name}, {sample_date}, {lat:.5f}, {lon:.5f}")
        # A brief retry handles transient limits without silently losing a point.
        for attempt in range(len(keys)):
            try:
                records = fetch_one(keys[(call_number - 1) % len(keys)], lat, lon, sample_date)
                checkpoint[identifier] = records
                CHECKPOINT_PATH.write_text(json.dumps(checkpoint), encoding="utf-8")
                break
            except Exception as exc:
                if attempt == len(keys) - 1:
                    # Do not chain the Requests exception: it can include the
                    # complete request URL in diagnostics on older clients.
                    raise RuntimeError(f"Call {call_number} failed permanently: {type(exc).__name__}") from None
                time_module.sleep(2 ** attempt)
        time_module.sleep(0.15)

    grouped: dict[str, dict[tuple[float, float], list[dict]]] = {name: {} for name in QUERY_PLAN}
    for route_name, sample_date, lat, lon in work:
        rows = checkpoint.get(call_id(route_name, sample_date, lat, lon))
        if rows:
            grouped[route_name].setdefault((lat, lon), []).extend(rows)
    for route_name, locations in grouped.items():
        payload = [
            {"lat": lat, "lon": lon, "data": sorted(rows, key=lambda row: row["period_end"])}
            for (lat, lon), rows in locations.items()
        ]
        output = build_output_name(route_name)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {len(payload)} locations to {output}")
    CHECKPOINT_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Perform the planned historical API requests")
    parser.add_argument("--check-keys", action="store_true", help="Check key validity at an unmetered Solcast location")
    arguments = parser.parse_args()
    if arguments.check_keys:
        check_keys()
    else:
        main(arguments.execute)
