"""
push_target_profile.py — feed the offline solver's strategy output into the
LIVE telemetry dashboard (main.py) as the "predicted" speed line.

WHAT THIS DOES
---------------
main.py's live dashboard reads current_data['profile']['TargetProfile'] to
compute metric['predicted'] (see main.py's update_processor, packet type
"A"). TargetProfile must be a list of [unix_time, speed_kmh] pairs, ONE PER
POINT of the currently-loaded route (current_data['profile']['Distance']) —
it's indexed positionally against the route, not queried by time.

This script:
  1. Loads output/strategy_<variant>.json (both variants supported; see
     --variant).
  2. Every poll interval, works out which race day "now" falls in from
     --race-start.
  3. Fetches whatever route is CURRENTLY loaded on the live dashboard
     (GET /api/data/profile) — the team loads a new KML each race day
     through the UI, so this script never assumes a fixed route; it just
     re-aligns to whatever's there.
  4. Interpolates that day's solved velocity profile (dashboard_trace:
     distance_m / time_s / v_kmh) onto the live route's Distance grid, turns
     time_s into real unix timestamps, and POSTs the result to
     /api/simulate as a "C" packet.
  5. Re-pushes automatically whenever the race day changes OR the loaded
     route changes (new KML uploaded) — so you start this once for the
     whole race and never touch it again; no manual re-run per day or per
     stage (stage1/loop/stage2 are already one continuous trace within a
     day, so there's nothing to switch mid-day either).

USAGE
-----
    python3 push_target_profile.py --variant aryaman --race-start 2026-09-13

    # both variants, race starts 13 Sept 2026, SAST (UTC+2), poll every 30s:
    python3 push_target_profile.py --variant aryaman \
        --race-start 2026-09-13 --tz-offset 2 --poll-interval 30

    # point at a live backend that isn't on localhost:8000:
    python3 push_target_profile.py --variant prahlad --race-start 2026-09-13 \
        --host 192.168.1.50 --port 8000

Requires: requests, numpy (both already used elsewhere in this project).
Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time

import numpy as np
import requests

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
N_RACE_DAYS = 8


def _log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_strategy(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "days" not in data:
        raise ValueError(f"{json_path} doesn't look like a strategy JSON (no 'days' key).")
    return data


def current_race_day(race_start_date: dt.date, tz_offset_hours: float) -> int | None:
    """1-indexed race day for 'now' in the race's local timezone, or None if
    we're before Day 1 or past Day 8."""
    now_local = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=tz_offset_hours)
    day_idx = (now_local.date() - race_start_date).days + 1
    if 1 <= day_idx <= N_RACE_DAYS:
        return day_idx
    return None


def day_midnight_epoch(race_start_date: dt.date, day_num: int, tz_offset_hours: float) -> float:
    """Unix epoch (UTC) for local midnight of the given race day."""
    local_midnight = dt.datetime.combine(
        race_start_date + dt.timedelta(days=day_num - 1), dt.time.min
    )
    utc_midnight = local_midnight - dt.timedelta(hours=tz_offset_hours)
    return utc_midnight.replace(tzinfo=dt.timezone.utc).timestamp()


def fetch_live_distance_km(base_url: str) -> np.ndarray | None:
    """Whatever route distance grid (km) is currently loaded on the live
    dashboard. None if nothing's loaded yet."""
    try:
        r = requests.get(f"{base_url}/api/data/profile", timeout=5)
        r.raise_for_status()
        dist = r.json().get("profile", {}).get("Distance", [])
    except Exception as e:
        _log(f"couldn't fetch live route profile: {e}")
        return None
    if not dist:
        return None
    return np.asarray(dist, dtype=float)


def build_target_profile(day: dict, live_distance_km: np.ndarray,
                          day_epoch0: float) -> list:
    """Interpolate this day's solved (distance_m, time_s, v_kmh) trace onto
    the live route's Distance grid (km), returning
    [[unix_time, speed_kmh], ...] aligned index-for-index with
    live_distance_km, as main.py's TargetProfile lookup requires."""
    tr = day["dashboard_trace"]
    solved_dist_km = np.asarray(tr["distance_m"], dtype=float) / 1000.0
    solved_t_s = np.asarray(tr["time_s"], dtype=float)
    solved_v_kmh = np.asarray(tr["v_kmh"], dtype=float)

    # Route Distance may run longer/shorter than the solved trace (different
    # source, different resolution) — clip live grid to the solved range so
    # we never extrapolate speed/time nonsensically past what was solved.
    lo, hi = solved_dist_km.min(), solved_dist_km.max()
    clipped = np.clip(live_distance_km, lo, hi)

    interp_v = np.interp(clipped, solved_dist_km, solved_v_kmh)
    interp_t = np.interp(clipped, solved_dist_km, solved_t_s)
    unix_times = day_epoch0 + interp_t

    return [[float(t), float(v)] for t, v in zip(unix_times, interp_v)]


def push_target_profile(base_url: str, target_profile: list) -> bool:
    try:
        r = requests.post(
            f"{base_url}/api/simulate",
            json={"type": "C", "TargetProfile": target_profile},
            timeout=5,
        )
        r.raise_for_status()
        return r.json().get("status") == "success"
    except Exception as e:
        _log(f"push failed: {e}")
        return False


def run(args: argparse.Namespace) -> None:
    base_url = f"http://{args.host}:{args.port}"
    race_start_date = dt.date.fromisoformat(args.race_start)

    strategies = {}
    for variant in args.variant:
        path = f"{args.json_dir}/strategy_{variant}_final.json"
        try:
            strategies[variant] = load_strategy(path)
            _log(f"loaded {path}")
        except Exception as e:
            _log(f"FATAL: couldn't load {path}: {e}")
            sys.exit(1)

    active_variant = args.variant[0]
    _log(f"using variant '{active_variant}' "
         f"({'only option' if len(args.variant) == 1 else 'first of ' + str(args.variant)})")

    last_pushed_day = None
    last_route_signature = None

    while True:
        day_num = current_race_day(race_start_date, args.tz_offset)

        if day_num is None:
            if last_pushed_day is not None:
                _log("outside race window (before Day 1 or after Day 8) — idle")
                last_pushed_day = None
            time.sleep(args.poll_interval)
            continue

        live_dist = fetch_live_distance_km(base_url)
        if live_dist is None:
            _log(f"Day {day_num}: no route currently loaded on dashboard — "
                 f"waiting for KML upload/render")
            time.sleep(args.poll_interval)
            continue

        route_signature = (len(live_dist), round(float(live_dist[0]), 3),
                            round(float(live_dist[-1]), 3))
        day_changed = day_num != last_pushed_day
        route_changed = route_signature != last_route_signature

        if not day_changed and not route_changed:
            time.sleep(args.poll_interval)
            continue

        strategy = strategies[active_variant]
        day_key = str(day_num)
        if day_key not in strategy["days"]:
            _log(f"Day {day_num} not present in strategy_{active_variant}.json — skipping")
            last_pushed_day = day_num
            time.sleep(args.poll_interval)
            continue

        day = strategy["days"][day_key]
        epoch0 = day_midnight_epoch(race_start_date, day_num, args.tz_offset)
        target_profile = build_target_profile(day, live_dist, epoch0)

        ok = push_target_profile(base_url, target_profile)
        if ok:
            reason = "day change" if day_changed else "route reload"
            _log(f"Day {day_num} [{active_variant}]: pushed TargetProfile "
                 f"({len(target_profile)} pts) — triggered by {reason}")
            last_pushed_day = day_num
            last_route_signature = route_signature
        else:
            _log(f"Day {day_num}: push failed, will retry next poll")

        time.sleep(args.poll_interval)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Push solver strategy output to the live dashboard as TargetProfile.")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--json-dir", default="output",
                     help="folder containing strategy_<variant>.json (default: output)")
    ap.add_argument("--variant", nargs="+", default=["aryaman", "prahlad"],
                     choices=["aryaman", "prahlad"],
                     help="which variant(s) to load; the FIRST one listed is used live "
                          "(default: both loaded, aryaman active)")
    ap.add_argument("--race-start", required=True,
                     help="calendar date (YYYY-MM-DD) of Day 1, in local race time")
    ap.add_argument("--tz-offset", type=float, default=2.0,
                     help="race-local timezone offset from UTC in hours (default: 2, SAST)")
    ap.add_argument("--poll-interval", type=float, default=30.0,
                     help="seconds between checks for day/route changes (default: 30)")
    args = ap.parse_args()

    _log(f"starting — race start {args.race_start}, tz UTC+{args.tz_offset}, "
         f"polling every {args.poll_interval}s")
    try:
        run(args)
    except KeyboardInterrupt:
        _log("stopped.")


if __name__ == "__main__":
    main()
