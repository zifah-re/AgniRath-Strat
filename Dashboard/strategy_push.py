"""
strategy_push.py — backend support for the Strategy page's "Push Strategy"
control.

WHY THIS EXISTS
----------------
Previously, getting a solved strategy (output/strategy_<variant>_fixed.json)
onto the live dashboard as the TargetProfile line meant running
push_target_profile.py as a separate, long-lived process that polled the
dashboard over HTTP. That's an extra terminal window, an extra thing that can
be left not-running, and an extra thing to explain to whoever's driving the
dashboard on race day.

This module does the same interpolation the script does, but in-process, on
demand: the Strategy page gets a "variant" + "day" picker and an Apply
button, which POSTs to /api/strategy/push (added in main.py) and the profile
is pushed immediately. No separate script needed.

INTERPOLATION APPROACH (unchanged from push_target_profile.py)
----------------------------------------------------------------
The solver's output (output/strategy_<variant>_fixed.json) is indexed by
DISTANCE (dashboard_trace.distance_m), each with a solved velocity (v_kmh)
and a solved elapsed time (time_s) at that distance. The Strategy page's
charts, however, plot everything against TIME (see the "Speed vs Time"
chart, which zips historic.Time_seconds against profile.TargetProfile).

To reconcile the two: for every point on the *live route's* distance grid
(current_data['profile']['Distance'], one entry per route point — this is
also what main.py indexes TargetProfile against), we use the solved trace's
own distance -> velocity and distance -> time_s mappings to look up (by
linear interpolation) the velocity and elapsed time at that exact distance.
The elapsed time is then converted to a wall-clock unix timestamp by adding
it to the epoch of local midnight on the day the strategy is being applied
for. That gives [[unix_time, speed_kmh], ...], aligned index-for-index with
the live route, which is exactly the shape main.py's TargetProfile lookup
and the Strategy page's time-axis chart both expect.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

STRATEGY_DIR = Path(__file__).resolve().parent / "output"
STRATEGY_FILE_SUFFIXES = ("_fixed.json", "_final.json", ".json")


class StrategyPushError(Exception):
    """Raised for any user-facing failure (bad variant, bad day, no route, etc)."""


def _safe_variant_slug(variant: str) -> str:
    """Keep this to alnum/-/_ so it can never be used to escape STRATEGY_DIR."""
    slug = "".join(c for c in variant if c.isalnum() or c in "-_")
    if not slug:
        raise StrategyPushError(f"Invalid strategy variant name: {variant!r}")
    return slug


def _strategy_paths(variant: str) -> list[Path]:
    slug = _safe_variant_slug(variant)
    return [
        STRATEGY_DIR / f"strategy_{slug}_final.json",
        STRATEGY_DIR / f"strategy_{slug}_fixed.json",
        STRATEGY_DIR / f"strategy_{slug}.json",
    ]


def _strategy_path(variant: str) -> Path:
    for path in _strategy_paths(variant):
        if path.exists():
            return path
    return _strategy_paths(variant)[0]


def list_strategy_variants() -> list[dict]:
    """Scan output/ for strategy_<variant>_fixed.json files and summarize
    what days/routes each one has, for the Strategy page's dropdowns."""
    variants = []
    if not STRATEGY_DIR.exists():
        return variants

    for path in sorted(STRATEGY_DIR.glob("strategy_*.json")):
        variant = path.stem[len("strategy_"):]
        for suffix in ("_final", "_fixed"):
            if variant.endswith(suffix):
                variant = variant[:-len(suffix)]
                break
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue  # skip unreadable/corrupt files rather than failing the whole list

        days = []
        for day_key, day in sorted(
            data.get("days", {}).items(), key=lambda kv: int(kv[0])
        ):
            days.append({
                "day": int(day_key),
                "route": day.get("route"),
                "distance_km": day.get("distance_km"),
                "speed_avg_kmh": day.get("speed_avg_kmh"),
            })

        variants.append({
            "variant": variant,
            "converged": data.get("converged"),
            "days": days,
        })

    return variants


def _load_strategy(variant: str) -> dict:
    path = _strategy_path(variant)
    if not path.exists():
        raise StrategyPushError(
            f"No strategy file found for variant '{variant}' (looked for {path.name})."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise StrategyPushError(f"Couldn't read {path.name}: {e}") from e
    if "days" not in data:
        raise StrategyPushError(f"{path.name} doesn't look like a strategy JSON (no 'days' key).")
    return data


def _get_day(data: dict, day: int) -> dict:
    day_key = str(day)
    if day_key not in data["days"]:
        available = ", ".join(sorted(data["days"].keys(), key=int))
        raise StrategyPushError(f"Day {day} not present in this strategy (available: {available}).")
    return data["days"][day_key]


def local_midnight_epoch(tz_offset_hours: float, for_date=None) -> float:
    """Unix epoch (UTC) for local midnight, in a timezone tz_offset_hours
    east of UTC. Defaults to 'today' in that timezone."""
    now_local = datetime.now(timezone.utc) + timedelta(hours=tz_offset_hours)
    if for_date is None:
        for_date = now_local.date()
    local_midnight_naive = datetime(for_date.year, for_date.month, for_date.day)
    utc_midnight = local_midnight_naive - timedelta(hours=tz_offset_hours)
    return utc_midnight.replace(tzinfo=timezone.utc).timestamp()


def build_target_profile(day: dict, live_distance_km: np.ndarray, epoch0: float) -> list:
    """Interpolate this day's solved (distance_m, time_s, v_kmh) trace onto
    the live route's Distance grid (km), returning
    [[unix_time, speed_kmh], ...] aligned index-for-index with
    live_distance_km — the exact shape main.py's TargetProfile lookup and
    the Strategy page's time-axis chart expect."""
    if "dashboard_trace" not in day:
        raise StrategyPushError("This day's strategy has no 'dashboard_trace' to interpolate from.")

    tr = day["dashboard_trace"]
    solved_dist_km = np.asarray(tr["distance_m"], dtype=float) / 1000.0
    solved_t_s = np.asarray(tr["time_s"], dtype=float)
    solved_v_kmh = np.asarray(tr["v_kmh"], dtype=float)

    if solved_dist_km.size == 0:
        raise StrategyPushError("This day's dashboard_trace is empty.")

    # Route Distance may run longer/shorter than the solved trace (different
    # source, different resolution) — clip live grid to the solved range so
    # we never extrapolate speed/time nonsensically past what was solved.
    lo, hi = float(solved_dist_km.min()), float(solved_dist_km.max())
    clipped = np.clip(live_distance_km, lo, hi)

    interp_v = np.interp(clipped, solved_dist_km, solved_v_kmh)
    interp_t = np.interp(clipped, solved_dist_km, solved_t_s)
    unix_times = epoch0 + interp_t

    return [[float(t), float(v)] for t, v in zip(unix_times, interp_v)]


def push_strategy_for_day(
    variant: str,
    day: int,
    live_distance_km: np.ndarray,
    tz_offset_hours: float,
) -> dict:
    """High-level entry point used by the /api/strategy/push route.
    Returns a small summary dict on success; raises StrategyPushError on any
    user-facing problem."""
    if live_distance_km is None or len(live_distance_km) == 0:
        raise StrategyPushError(
            "No route is currently loaded on the dashboard — load/select a KML first."
        )

    data = _load_strategy(variant)
    day_data = _get_day(data, day)
    epoch0 = local_midnight_epoch(tz_offset_hours)

    target_profile = build_target_profile(
        day_data, np.asarray(live_distance_km, dtype=float), epoch0
    )

    return {
        "target_profile": target_profile,
        "variant": variant,
        "day": day,
        "route": day_data.get("route"),
        "points": len(target_profile),
    }
