"""
race_config.py — Sasol Solar Challenge 2026 regulation-derived constants.

RULES OF THIS FILE (Plan v3 §4.1):
  * EVERY constant here cites the regulation clause it comes from
    (2026 Sasol Solar Challenge Sporting Regulations v1.0 + Technical
    Regulations v1.0, March 2025 — "SR x.y" / "TR x.y" in comments).
  * No regulation number may appear anywhere else in the codebase.
    Import from here.
  * If a value is uncertain it is marked  # TODO-VERIFY  with a reason.

Units convention (whole codebase): seconds, metres, m/s, watts, watt-hours,
kilograms, degrees for angles unless suffixed otherwise. Suffix units in
names where ambiguity is possible (_kmh, _wh, _pct, _s, _m).
"""

from __future__ import annotations

import datetime as _dt

# ---------------------------------------------------------------------------
# Race mode switch (Plan v3 §2.4). Core-team decision pending; build default
# is "loops" (the harder superset). Flipping this is the one-line change.
# ---------------------------------------------------------------------------
RACE_MODE: str = "loops"          # one of {"loops", "completion"}
_VALID_RACE_MODES = ("loops", "completion")
assert RACE_MODE in _VALID_RACE_MODES

# ---------------------------------------------------------------------------
# Calendar (SR 2.20 route notes + released 2026 route; day labels from KMZ)
# ---------------------------------------------------------------------------
RACE_YEAR = 2026
RACE_DAY_DATES = [                       # index 0 == Day 1
    _dt.date(2026, 9, 10),               # Day 1  Sasolburg -> Swartruggens
    _dt.date(2026, 9, 11),               # Day 2  HALF BLIND -> Vryburg
    _dt.date(2026, 9, 12),               # Day 3  FULL BLIND -> Kimberley
    _dt.date(2026, 9, 13),               # Day 4  -> Olifantshoek
    _dt.date(2026, 9, 14),               # Day 5  -> Augrabies
    _dt.date(2026, 9, 15),               # Day 6  -> Springbok
    _dt.date(2026, 9, 16),               # Day 7  -> Clanwilliam
    _dt.date(2026, 9, 17),               # Day 8  -> Paarl (timed finish)
]
N_RACE_DAYS = len(RACE_DAY_DATES)        # SR 2.20: multi-stage, 8 consecutive days
TIMEZONE = "Africa/Johannesburg"         # CAT / GMT+2 (SR 2.23 telemetry spec)

# ---------------------------------------------------------------------------
# Daily timing (SR 2.22)
# ---------------------------------------------------------------------------
START_TIME_DAY1_S = 9 * 3600             # SR 2.22.1: Day 1 official start 09H00
START_TIME_OTHER_S = 8 * 3600            # SR 2.22.2: Days 2-8 official start 08H00
FINISH_TIME_S = 17 * 3600                # SR 2.22.3: official finish 17H00 daily
FINISH_CUTOFF_ABS_S = 17 * 3600 + 30*60  # SR 2.30.2: absolute 17H30 parc-ferme
                                         #   cutoff; beyond -> severe penalties /
                                         #   possible exclusion
DAY8_TIMED_FINISH_S = 15 * 3600          # SR 2.22.4: Day 8 timed finish 15H00
DAY8_PROCEEDINGS_END_S = 17 * 3600       # SR 2.22.5: Day 8 finish-line
                                         #   proceedings end 17H00

BATTERY_UNSEAL_TIME_S = 6 * 3600         # SR 2.30.9: packs unsealed 06H00
BATTERY_SEAL_AFTER_FINISH_S = 5 * 60     # SR 2.30.7/2.30.8: pack disconnected &
                                         #   sealed within 5 min of finish line
ARRAY_DISCONNECT_AFTER_FINISH_S = 5 * 60 # SR 2.31.2: demonstrate array
                                         #   disconnected within 5 min in parc ferme

def day_start_time_s(day_index: int) -> int:
    """Official start time (seconds since midnight) for day_index (0-based)."""
    return START_TIME_DAY1_S if day_index == 0 else START_TIME_OTHER_S

def day_finish_time_s(day_index: int) -> int:
    """Official finish time (seconds since midnight) for day_index (0-based)."""
    return DAY8_TIMED_FINISH_S if day_index == N_RACE_DAYS - 1 else FINISH_TIME_S

# ---------------------------------------------------------------------------
# Late-finish penalty (SR 2.22.6): 1 min per minute (or part) up to and incl.
# 10 min late; each additional minute (or part) beyond 10 counts 2 min.
# Served at NEXT DAY's control stop (SR 2.22.7) -> couples into tomorrow's
# driving window (Plan v3 §2.2).  Worked example in regs: 17:13 -> 16 min.
# ---------------------------------------------------------------------------
LATE_PENALTY_BREAKPOINT_MIN = 10
LATE_PENALTY_RATE_BEYOND = 2

def late_finish_penalty_min(minutes_late: float) -> int:
    """Penalty minutes served next day at the control stop (SR 2.22.6/2.22.7).

    Any part of a minute counts as a full minute.
    """
    import math
    m = math.ceil(max(0.0, minutes_late))
    if m == 0:
        return 0
    if m <= LATE_PENALTY_BREAKPOINT_MIN:
        return m
    return LATE_PENALTY_BREAKPOINT_MIN + LATE_PENALTY_RATE_BEYOND * (
        m - LATE_PENALTY_BREAKPOINT_MIN
    )

EARLY_START_PENALTY_PER_MIN = 2          # SR 2.22.8: starting before official
                                         #   start: 2 min per offending minute

# ---------------------------------------------------------------------------
# Control stops (SR 2.28)
# ---------------------------------------------------------------------------
CONTROL_STOP_DURATION_S = 30 * 60        # SR 2.28.5: one mandatory 30-min stop/day
CONTROL_STOP_UNTOUCHABLE_S = 25 * 60     # SR 2.28.14: no team member may touch
                                         #   the vehicle for 25 min of the 30
# SR 2.28.6: the 30-min stop may be served at ANY chosen time during the day,
# provided enough time remains to reach the finish. SR 2.28.7: notify Control
# Stop Manager first. SR 2.28.13: arriving driver alone may reconfigure the
# vehicle for charging before timing starts -> CS solar capture is legal and
# belongs in the energy ledger (Plan v3 §2.2).

# ---------------------------------------------------------------------------
# Loop stops (SR 2.29)
# ---------------------------------------------------------------------------
LOOP_STOP_DURATION_S = 5 * 60            # SR 2.29.5: mandatory 5-min loop stop
                                         #   BEFORE EVERY loop attempt
LOOP_CRUISE_SPEED_MS = 55 / 3.6
# SR 2.29.6: loops optional; teams must declare intent to Loop Stop Manager
# each time (maps 1:1 to MPC commit/abort decision point, Plan v3 §2.2).
# SR 2.29.2: loop-km determine "most km clocked" -> 2026 Champions.

# ---------------------------------------------------------------------------
# Drivers & occupants (SR 2.24, SR pre-race 1.x)
# ---------------------------------------------------------------------------
DRIVER_SWAP_INTERVAL_S = 2 * 3600        # SR 2.24.4: drivers/passengers must
                                         #   change every two hours
DRIVER_MIN_MASS_KG = 80.0                # SR (scrutineering): driver <80 kg is
                                         #   ballasted up to 80 kg
# SR 2.24.5: a driver who drove >1 h may not drive a support vehicle within
# 2 h afterwards (ops constraint — runbook, not model).

# ---------------------------------------------------------------------------
# Trailering (SR 2.32) — Plan v3 §5
# ---------------------------------------------------------------------------
TRAILERING_MIN_SPEED_MS = 60.0 / 3.6     # SR 2.32.2: unable to maintain minimum
                                         #   60 km/h on open road -> must trailer
TRAILER_OFFLOAD_FINISH_RADIUS_M = 500.0  # SR 2.32.6: may offload/drive across
                                         #   the line only within 500 m of finish
# SR 2.32.5/2.32.7: trailer granularity is stage-level (start->CS, CS->finish);
# no offloading mid-stage once trailered.
# Scoring consequence (SR classification): teams that trailered at any point
# rank BELOW all non-trailered teams regardless of distance ("asterisk rule")
# -> trailering treated as near-lexicographic in all optimizers (Plan v3 §5).

# ---------------------------------------------------------------------------
# Penalty scale (SR 2.34) — for decision cards, not for planning to incur.
# ---------------------------------------------------------------------------
PENALTY_KM_UNINTENTIONAL_MAX = 50        # SR 2.34: unintentional, non-beneficial
PENALTY_KM_UNINT_BENEFICIAL_MAX = 100    # SR 2.34: unintentional but beneficial
PENALTY_KM_INTENTIONAL_MAX = 200         # SR 2.34: intentional (or DQ/exclusion)

# ---------------------------------------------------------------------------
# Special stages (SR 2.21) — Plan v3 §2.2
# ---------------------------------------------------------------------------
HALF_BLIND_DAY_INDEX = 1                 # Day 2 (route notes): loop info via
                                         #   Golden Envelope evening before
FULL_BLIND_DAY_INDEX = 2                 # Day 3: route+CS+loop via Golden Envelope
MARATHON_OPT_OUT_PENALTY_S = 3600        # SR 2.21 Marathon: opting out = 1 h penalty
MARATHON_EARLY_REMOVAL_PENALTY_KM = 150  # SR 2.21 Marathon: removing vehicle from
                                         #   parc ferme early = 150 km docked

# ---------------------------------------------------------------------------
# Vehicle technical limits used by strategy (TR)
# ---------------------------------------------------------------------------
TURNING_CIRCLE_RADIUS_M = 7.5            # TR: must turn within 7.5 m radius,
                                         #   measured to outer tyre track ->
                                         #   loop-turnaround audit (Plan v3 §4.2)
# TR storage allowances 2026 (context only; conformity is the car team's
# domain, not strategy's — Plan v3 §3):
#   4 m2 Challenger: 15 MJ;  6 m2 Challenger: 11 MJ.

# ---------------------------------------------------------------------------
# Route facts from the released 2026 route notes + KMZ (Plan v3 §2.1).
# Stage distances in km as published; loops as (name, km) tuples.
# Day 2 loop and all Day 3 values arrive via Golden Envelope -> placeholders
# generated by pipeline/blind_stage.py (do NOT hand-edit here on race night;
# the overnight rebuild writes data/processed/, not this file).
# ---------------------------------------------------------------------------
DAY_ROUTE_NOTES = [
    # day 1
    dict(stage1_km=172.7, loops=[("rustenburg_loop", 22.6)], stage2_km=65.6,
         start="Boiketlong Hall (Sasolburg)",
         control_stop="Rustenburg Highschool",
         finish="Swartruggens Dam"),
    # day 2 — HALF BLIND: loop unknown until Golden Envelope (SR 2.21)
    dict(stage1_km=71.5, loops=None, stage2_km=231.0,
         start="Swartruggens Dam", control_stop="Zeerust Highschool",
         finish="Kameelboom Lodge (Vryburg)"),
    # day 3 — FULL BLIND: everything unknown until Golden Envelope (SR 2.21)
    dict(stage1_km=None, loops=None, stage2_km=None,
         start="Kameelboom Lodge (Vryburg)", control_stop=None,
         finish="Kimberley Technical Highschool"),
    # day 4
    dict(stage1_km=197.0, loops=[("postmasburg_loop2", 21.0),
                                 ("postmasburg_loop1", 14.0)], stage2_km=63.3,
         start="Kimberley Technical Highschool",
         control_stop="Postmasburg Highschool",
         finish="Ranch Chalets (Olifantshoek)"),
    # day 5
    dict(stage1_km=178.0, loops=[("upington_loop1", 62.0),
                                 ("upington_loop2", 34.0)], stage2_km=114.0,
         start="Ranch Chalets (Olifantshoek)",
         control_stop="Upington Highschool",
         finish="Kameeldoring Campsite (Augrabies)"),
    # day 6 — control stop location == finish location
    dict(stage1_km=310.0, loops=[("springbok_loop", 18.2)], stage2_km=0.0,
         start="Kameeldoring Campsite (Augrabies)",
         control_stop="Namakwa Highschool (Springbok)",
         finish="Namakwa Highschool (Springbok)"),
    # day 7
    dict(stage1_km=261.0, loops=[("vanrhynsdorp_loop", 16.5)], stage2_km=80.9,
         start="Namakwa Highschool (Springbok)",
         control_stop="Vanrhynsdorp Highschool",
         finish="Augsburg Landbougimnasium (Clanwilliam)"),
    # day 8 — timed finish 15H00 (SR 2.22.4)
    dict(stage1_km=180.0, loops=[("ceres_loop", 21.8)], stage2_km=98.3,
         start="Augsburg Landbougimnasium (Clanwilliam)",
         control_stop="Charlie Hofmeyer Highschool (Ceres)",
         finish="Suid Agter Paarl Road (Paarl)"),
]
assert len(DAY_ROUTE_NOTES) == N_RACE_DAYS

# Placeholder length for unknown blind-day loops = mean of released loop
# lengths (Plan v3 §8 L1; notes doc idea v). Recomputed here so it tracks
# any route-note corrections automatically.
_released_loop_lengths = [
    km for d in DAY_ROUTE_NOTES if d["loops"]
    for (_, km) in d["loops"]
]
BLIND_LOOP_PLACEHOLDER_KM = sum(_released_loop_lengths) / len(_released_loop_lengths)

# Unplanned-stop time budget per day (Plan v3 §8 L2; Paper 1 Day-1 case study:
# three unplanned brief stops — traffic, driver changes — absent from the
# forecast). TODO-VERIFY: tune from our own test-run logs.
UNPLANNED_STOP_BUDGET_S = 8 * 60
