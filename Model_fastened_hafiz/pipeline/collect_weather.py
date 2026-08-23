"""
pipeline/collect_weather.py — workplan 0.4: Solcast key test + the
forecast-vs-actual collection cron (longest lead-time item; starts first).

Usage:
    python -m pipeline.collect_weather test-keys
    python -m pipeline.collect_weather pull --points points.json --out data/raw
    (cron: run `pull` daily at 06:00 SAST; PMF pairs accumulate in data/raw/)

points.json: [{"id": 0, "lat": -26.81, "lon": 27.83}, ...]  — circle centres
for the route corridor. Until the route pipeline (block 2) generates real
circles, use a hand-placed set along the N12/N14 corridor.

Solcast auth: environment variable SOLCAST_API_KEY (never hardcode; the two
keys found in Dashboard/constants.py are TOMTOM keys, not Solcast — the
Solcast keys the senior mentioned must be located and exported before
`test-keys` will pass).

Wind source: Open-Meteo (no key needed) as the open-source default
(Plan v3 §6.2); the senior confirmed Solcast wind was never used.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request

SOLCAST_BASE = "https://api.solcast.com.au"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"


def _get(url: str, headers: dict | None = None, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def test_keys() -> bool:
    """Cheap live-key check against Solcast (workplan 0.4)."""
    key = os.environ.get("SOLCAST_API_KEY", "")
    if not key:
        print("SOLCAST_API_KEY not set — export it first.")
        return False
    url = (f"{SOLCAST_BASE}/data/forecast/radiation_and_weather?"
           f"latitude=-26.81&longitude=27.83&hours=1&output_parameters=ghi"
           f"&format=json&api_key={urllib.parse.quote(key)}")
    try:
        _get(url)
        print("Solcast key OK.")
        return True
    except Exception as e:  # noqa: BLE001 — report anything to the operator
        print(f"Solcast key FAILED: {e}")
        return False


def pull_solcast_forecast(lat: float, lon: float, hours: int = 72) -> list[dict]:
    key = os.environ["SOLCAST_API_KEY"]
    url = (f"{SOLCAST_BASE}/data/forecast/radiation_and_weather?"
           f"latitude={lat}&longitude={lon}&hours={hours}"
           f"&output_parameters=ghi,cloud_opacity&format=json"
           f"&api_key={urllib.parse.quote(key)}")
    return _get(url).get("forecasts", [])


def pull_solcast_actuals(lat: float, lon: float, hours: int = 24) -> list[dict]:
    """Estimated actuals — the other half of every PMF pair (Plan v3 §6.1)."""
    key = os.environ["SOLCAST_API_KEY"]
    url = (f"{SOLCAST_BASE}/data/live/radiation_and_weather?"
           f"latitude={lat}&longitude={lon}&hours={hours}"
           f"&output_parameters=ghi&format=json"
           f"&api_key={urllib.parse.quote(key)}")
    return _get(url).get("estimated_actuals", [])


def pull_open_meteo_wind(lat: float, lon: float) -> dict:
    url = (f"{OPEN_METEO_BASE}?latitude={lat}&longitude={lon}"
           f"&hourly=wind_speed_10m,wind_direction_10m"
           f"&forecast_days=3&wind_speed_unit=ms&timezone=Africa%2FJohannesburg")
    return _get(url)


def _append_rows(path: pathlib.Path, header: list[str], rows: list[list]):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def pull(points_json: str, out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    points = json.loads(pathlib.Path(points_json).read_text())
    stamp = dt.datetime.now().isoformat(timespec="seconds")

    for p in points:
        cid, lat, lon = p["id"], p["lat"], p["lon"]
        try:
            fc = pull_solcast_forecast(lat, lon)
            _append_rows(out / "solcast_forecast.csv",
                         ["pulled_at", "circle_id", "period_end", "ghi",
                          "cloud_opacity"],
                         [[stamp, cid, r.get("period_end"), r.get("ghi"),
                           r.get("cloud_opacity")] for r in fc])
            ac = pull_solcast_actuals(lat, lon)
            _append_rows(out / "solcast_actuals.csv",
                         ["pulled_at", "circle_id", "period_end", "ghi"],
                         [[stamp, cid, r.get("period_end"), r.get("ghi")]
                          for r in ac])
        except Exception as e:  # noqa: BLE001
            print(f"[circle {cid}] solcast pull failed: {e}", file=sys.stderr)
        try:
            wm = pull_open_meteo_wind(lat, lon)
            h = wm.get("hourly", {})
            times = h.get("time", [])
            _append_rows(out / "openmeteo_wind.csv",
                         ["pulled_at", "circle_id", "time", "speed_ms",
                          "dir_deg_from"],
                         [[stamp, cid, t, s, d] for t, s, d in
                          zip(times, h.get("wind_speed_10m", []),
                              h.get("wind_direction_10m", []))])
        except Exception as e:  # noqa: BLE001
            print(f"[circle {cid}] open-meteo pull failed: {e}", file=sys.stderr)
    print(f"pull complete -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test-keys")
    p = sub.add_parser("pull")
    p.add_argument("--points", required=True)
    p.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    if args.cmd == "test-keys":
        sys.exit(0 if test_keys() else 1)
    pull(args.points, args.out)


if __name__ == "__main__":
    main()
