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

MANDATORY-STOP TIME (SR 2.28 control stop, SR 2.29 loop stops)
----------------------------------------------------------------
`dashboard_trace.time_s` turns out to be DRIVING time only — it stacks each
stage's own elapsed_s end-to-end with no allowance for the mandatory 30-min
control stop (SR 2.28.5) or the 5-min-before-every-loop-attempt stop
(SR 2.29.5, x n_loops). Its last value matches the day's eta_drive_only, not
its real eta. Stage 1 happens before any stop occurs so it's unaffected;
everything from the loop stage onward is increasingly too early.

Rather than duplicating race_config.py's regulation constants here (its own
docstring is explicit that regulation numbers should only live there), we
derive the correction from the solver's own per-stage 'stop_min' fields,
which are already regulation-correct (e.g. stage1.stop_min == 30 ==
CONTROL_STOP_DURATION_S/60; loop.stop_min == 35 == 5 x n_loops, matching
LOOP_STOP_DURATION_S). _stop_offset_at_km walks the day's stages in order and
adds each one's stop_min (in seconds) once cumulative distance passes it —
and, within the loop stage specifically, spreads its stop_min evenly across
n_loops so each individual lap gets its own 5-min addition at the point it
begins, rather than dumping all of it at once at the loop's start.
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


def _stop_events(
    day: dict,
    segments: list[dict],
    dashboard_dist_km: np.ndarray,
    dashboard_time_raw: np.ndarray,
) -> list[tuple[float, float, float]]:
    """Every mandatory stop for this day as (boundary_km, stop_start_s,
    stop_end_s), in chronological order.

    _stop_offset_at_km only corrected the *timing* of points after a stop —
    it never represented the stop itself, so a stage with stops inside it
    (only the loop stage, here) rendered as one continuous drive with no gap
    for each 5-min stop. This enumerates those gaps explicitly so callers can
    insert real (time, 0 km/h) points for each one.

    stop_start_s/stop_end_s are in the same "raw driving time + cumulative
    prior stop offset" units as build_target_profile's corrected solved_t_s
    (i.e. still need + epoch0 to become a wall-clock unix timestamp) —
    computed with a running cumulative offset so they line up exactly with
    what _stop_offset_at_km would report just before/after each stop.
    """
    events: list[tuple[float, float, float]] = []
    cum_offset = 0.0
    for seg in segments:
        stage = day.get(seg["key"]) or {}
        stop_s_total = (stage.get("stop_min") or 0.0) * 60.0
        if stop_s_total <= 0:
            continue

        if seg["key"] == "loop":
            n_loops = stage.get("n_loops") or 0
            seg_len = seg["end_km"] - seg["start_km"]
            if not n_loops or seg_len <= 0:
                continue
            lap_len = seg_len / n_loops
            per_lap_s = stop_s_total / n_loops
            for i in range(int(n_loops)):
                # The stop before lap i+1 happens at the start of that lap.
                boundary_km = seg["start_km"] + i * lap_len
                raw_t = float(np.interp(boundary_km, dashboard_dist_km, dashboard_time_raw))
                start_t = raw_t + cum_offset
                end_t = start_t + per_lap_s
                events.append((boundary_km, start_t, end_t))
                cum_offset += per_lap_s
        else:
            boundary_km = seg["end_km"]  # stop taken on arrival, before the next stage
            raw_t = float(np.interp(boundary_km, dashboard_dist_km, dashboard_time_raw))
            start_t = raw_t + cum_offset
            end_t = start_t + stop_s_total
            events.append((boundary_km, start_t, end_t))
            cum_offset += stop_s_total
    return events


def _stage_segments(day: dict) -> list[dict]:
    """Return this day's stages, in order, as segments of the day's
    cumulative dashboard_trace distance axis.

    The solver's per-day 'dashboard_trace' is a literal concatenation of
    day['stage1']['trace'] + day['loop']['trace'] + day['stage2']['trace']
    (in whatever order day['stage_names'] lists), each stage's own trace
    being LOCALLY zeroed (distance_km starts at 0 for that stage alone).
    So stage N's window inside the day-cumulative trace is
    [offset, offset + stage_N's own max local distance], where offset is
    the sum of all earlier stages' own max local distances.

    We use this purely to find *where* each stage lives inside
    dashboard_trace's cumulative axis — the actual speed/time values we
    interpolate still come from dashboard_trace itself (already correct,
    solver-computed), just sliced+rebased to this stage's window.
    """
    segments = []
    offset = 0.0
    for name in day.get("stage_names") or []:
        stage = day.get(name)
        if not stage or not (stage.get("trace") or {}).get("distance_km"):
            continue
        seg_len = float(stage["trace"]["distance_km"][-1])
        segments.append({
            "key": name,
            "label": _stage_label(name, stage),
            "start_km": offset,
            "end_km": offset + seg_len,
        })
        offset += seg_len
    return segments


def _stage_label(name: str, stage: dict) -> str:
    dist = stage.get("distance_km")
    n_loops = stage.get("n_loops") or 0
    if name == "loop" and n_loops:
        return f"Loop \u00d7{int(n_loops)} ({dist:.1f} km)" if dist is not None else f"Loop \u00d7{int(n_loops)}"
    pretty = f"Stage {name[len('stage'):]}" if name.startswith("stage") else name.capitalize()
    return f"{pretty} ({dist:.1f} km)" if dist is not None else pretty


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
            segments = [{
                "key": "full",
                "label": f"Full day ({day.get('distance_km', 0):.1f} km)" if day.get("distance_km") is not None else "Full day",
            }]
            for seg in _stage_segments(day):
                segments.append({"key": seg["key"], "label": seg["label"]})

            days.append({
                "day": int(day_key),
                "route": day.get("route"),
                "distance_km": day.get("distance_km"),
                "speed_avg_kmh": day.get("speed_avg_kmh"),
                "segments": segments,
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


def _stop_offset_at_km(cum_km: float, segments: list[dict], day: dict) -> float:
    """Seconds of mandatory-stop time (SR 2.28 control stop, SR 2.29 loop
    stops) that have already elapsed by the time the day's cumulative
    distance reaches cum_km.

    Walks the day's stages in the order they're actually driven. Each
    stage's own stop_min (minutes) is the solver's already-correct total for
    stops attached to that stage (e.g. stage1.stop_min=30 is the one SR
    2.28.5 control stop taken on arrival; loop.stop_min=35 is 7 x the SR
    2.29.5 5-min loop-attempt stop, one before each of the 7 laps) — see
    module docstring. Once cum_km is fully past a stage, that stage's whole
    stop_min counts. While cum_km is inside the loop stage specifically, its
    stop_min is spread evenly over n_loops (each lap being roughly equal
    length), so a lap deep in the loop gets progressively more of it rather
    than all of it landing at the loop's very first metre.
    """
    offset_s = 0.0
    for seg in segments:
        stage = day.get(seg["key"]) or {}
        stop_s = (stage.get("stop_min") or 0.0) * 60.0

        if cum_km <= seg["start_km"]:
            break

        if cum_km >= seg["end_km"]:
            offset_s += stop_s
            continue

        # cum_km lands inside this stage's own window.
        if seg["key"] == "loop" and stop_s > 0:
            n_loops = stage.get("n_loops") or 0
            seg_len = seg["end_km"] - seg["start_km"]
            if n_loops and seg_len > 0:
                lap_len = seg_len / n_loops
                laps_stopped_for = min(
                    int((cum_km - seg["start_km"]) // lap_len) + 1, n_loops
                )
                offset_s += (stop_s / n_loops) * laps_stopped_for
        break
    return offset_s


def build_target_profile(
    day: dict,
    live_distance_km: np.ndarray,
    epoch0: float,
    segment: str | None = None,
) -> list:
    """Interpolate this day's solved (distance_m, time_s, v_kmh) trace onto
    the live route's Distance grid (km), returning
    [[unix_time, speed_kmh], ...] aligned index-for-index with
    live_distance_km — the exact shape main.py's TargetProfile lookup and
    the Strategy page's time-axis chart expect.

    If `segment` names a specific stage (e.g. "stage1", "loop", "stage2"),
    the day's cumulative trace is first sliced down to that stage's own
    window and its distance column rebased to start at 0 — matching how a
    stage-only KML loads on the live dashboard (its own Distance array also
    starts at 0). Without this, a stage-only live route gets compared
    against the wrong stretch of the day's combined distance axis (e.g.
    Stage 2 gets matched against Stage 1's territory, since both start
    their live Distance count at 0 independently)."""
    if "dashboard_trace" not in day:
        raise StrategyPushError("This day's strategy has no 'dashboard_trace' to interpolate from.")

    tr = day["dashboard_trace"]
    solved_dist_km = np.asarray(tr["distance_m"], dtype=float) / 1000.0
    solved_v_kmh = np.asarray(tr["v_kmh"], dtype=float)

    if solved_dist_km.size == 0:
        raise StrategyPushError("This day's dashboard_trace is empty.")

    # dashboard_trace.time_s is DRIVING time only (see module docstring) —
    # add back the mandatory-stop time that's actually elapsed by each point,
    # so anything from the loop stage onward isn't reported hours too early.
    segments = _stage_segments(day)
    stop_offsets = np.array(
        [_stop_offset_at_km(d, segments, day) for d in solved_dist_km]
    )
    solved_t_s = np.asarray(tr["time_s"], dtype=float) + stop_offsets

    if segment and segment != "full":
        seg_bounds = {s["key"]: s for s in segments}
        if segment not in seg_bounds:
            available = ", ".join(sorted(seg_bounds.keys())) or "full"
            raise StrategyPushError(
                f"Segment '{segment}' not present for this day (available: {available}, full)."
            )
        b = seg_bounds[segment]
        mask = (solved_dist_km >= b["start_km"] - 1e-6) & (solved_dist_km <= b["end_km"] + 1e-6)
        if not mask.any():
            raise StrategyPushError(f"No solved trace points found for segment '{segment}'.")
        solved_dist_km = solved_dist_km[mask] - b["start_km"]  # rebase so this stage starts at 0
        # solved_t_s already has the correct absolute (stop-inclusive) elapsed
        # time baked in — do NOT rebase it, since it still needs to combine
        # with epoch0 into a real wall-clock timestamp for this stage's
        # actual time of day (e.g. Stage 2 really does start mid-afternoon).
        solved_t_s = solved_t_s[mask]
        solved_v_kmh = solved_v_kmh[mask]

    # Route Distance may run longer/shorter than the solved trace (different
    # source, different resolution) — clip live grid to the solved range so
    # we never extrapolate speed/time nonsensically past what was solved.
    lo, hi = float(solved_dist_km.min()), float(solved_dist_km.max())
    clipped = np.clip(live_distance_km, lo, hi)

    interp_v = np.interp(clipped, solved_dist_km, solved_v_kmh)
    interp_t = np.interp(clipped, solved_dist_km, solved_t_s)
    unix_times = epoch0 + interp_t

    return [[float(t), float(v)] for t, v in zip(unix_times, interp_v)]


def build_target_profile_chart(
    day: dict,
    epoch0: float,
    segment: str | None = None,
) -> list:
    """Build the full-resolution, time-ordered [[unix_time, speed_kmh], ...]
    series used ONLY by the Strategy page's chart (profile.TargetProfileChart)
    — NOT main.py's TargetProfile, which the MPC solver and the live
    "Predicted Speed" overlay both index point-for-point against the live
    route (main.py:614 does TargetProfile[i]/[i+1] where i is the live
    route's own point index), so it can never have extra points spliced in
    without corrupting both of those.

    Unlike build_target_profile, this ignores the live route's Distance grid
    entirely and uses the solved trace's own native points directly (already
    far denser than most live KML routes), PLUS explicit (time, 0 km/h)
    points at every mandatory stop from _stop_events — without these, a
    7-lap loop stage renders as one smooth ~15min drive instead of the real
    3+ hours with a 5-min dead stop before each lap.
    """
    if "dashboard_trace" not in day:
        raise StrategyPushError("This day's strategy has no 'dashboard_trace' to interpolate from.")

    tr = day["dashboard_trace"]
    full_dist_km = np.asarray(tr["distance_m"], dtype=float) / 1000.0
    full_v_kmh = np.asarray(tr["v_kmh"], dtype=float)
    full_raw_t_s = np.asarray(tr["time_s"], dtype=float)

    if full_dist_km.size == 0:
        raise StrategyPushError("This day's dashboard_trace is empty.")

    segments = _stage_segments(day)
    stop_offsets = np.array([_stop_offset_at_km(d, segments, day) for d in full_dist_km])
    full_t_s = full_raw_t_s + stop_offsets
    events = _stop_events(day, segments, full_dist_km, full_raw_t_s)

    dist_km, t_s, v_kmh = full_dist_km, full_t_s, full_v_kmh
    if segment and segment != "full":
        seg_bounds = {s["key"]: s for s in segments}
        if segment not in seg_bounds:
            available = ", ".join(sorted(seg_bounds.keys())) or "full"
            raise StrategyPushError(
                f"Segment '{segment}' not present for this day (available: {available}, full)."
            )
        b = seg_bounds[segment]
        mask = (dist_km >= b["start_km"] - 1e-6) & (dist_km <= b["end_km"] + 1e-6)
        if not mask.any():
            raise StrategyPushError(f"No solved trace points found for segment '{segment}'.")
        dist_km, t_s, v_kmh = dist_km[mask], t_s[mask], v_kmh[mask]
        events = [e for e in events if b["start_km"] - 1e-6 <= e[0] <= b["end_km"] + 1e-6]

    points = list(zip(t_s.tolist(), v_kmh.tolist()))
    for _boundary_km, start_s, end_s in events:
        points.append((start_s, 0.0))
        points.append((end_s, 0.0))

    points.sort(key=lambda p: p[0])
    return [[float(epoch0 + t), float(v)] for t, v in points]


def push_strategy_for_day(
    variant: str,
    day: int,
    live_distance_km: np.ndarray,
    tz_offset_hours: float,
    segment: str | None = None,
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
        day_data, np.asarray(live_distance_km, dtype=float), epoch0, segment=segment
    )
    target_profile_chart = build_target_profile_chart(day_data, epoch0, segment=segment)

    return {
        "target_profile": target_profile,
        "target_profile_chart": target_profile_chart,
        "variant": variant,
        "day": day,
        "segment": segment or "full",
        "route": day_data.get("route"),
        "points": len(target_profile),
    }