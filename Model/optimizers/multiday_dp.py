"""
optimizers/multiday_dp.py

L1 multi-day DP (Plan v3 S8): state = (day, SOC bucket), action = an
integer attempt count per NAMED loop that day (not one averaged loop --
see CHANGELOG), objective per race_config.RACE_MODE, morning-only
overnight charge (SR 2.30/2.31), late-finish penalty coupling (SR 2.22.7).

Backward induction follows Oosthuizen et al. 2018 ("Solar Electric
Vehicle Energy Optimization for the Sasol Solar Challenge 2018"), Sec.
III-D/E, Eq. 26: J_x(z_x) = min_u E{g_x + J_{x+1}(...)}, solved tail-first
(x = N-1, ..., 0) under Bellman's principle of optimality. The terminal
state carries no unique cost (paper Sec III-D, note after Eq. 26), so
V[n_days, :] = 0 and each day's own contribution is folded into the
summation exactly as the paper's simplified cost function does. The
paper's u_x (loop count per day) and its per-day favorability weighting
(Eq. 15-18) motivated treating loop *choice*, not just loop *count*, as
the actual decision variable here (Sec 2.3/8 of Master Plan v3 makes the
same point for our heterogeneous, named-loop route).

Mid-race re-plan: solve() takes start_day_index / remaining_km_today /
elapsed_s_today / control_stop_taken_today / extra_stoppage_s so the same
DP can resolve from wherever the car is now, not just Day 1. replan()
wraps that for "car got worse" cases; robustness_report() sweeps
DP_SOLAR_SCENARIOS x DP_STOPPAGE_SCENARIOS_S for a real completion
probability.

CHANGELOG (v2 -- fixes all 14 issues raised in the L1 code review, plus
one found during the rewrite itself):
  1. Fixed the `_day_slope_profile` call that omitted `loop_km` and
     crashed on the first transition (TypeError).
  2. Loop zones are now positioned after the day's real Stage-1 distance
     (`plan.stage1_km`), not hardcoded to km 0 -- see `_build_zones`.
  3. Named loops are no longer averaged into one generic loop. The action
     space is now a combination of per-named-loop attempt counts
     (`_enumerate_loop_combos`); Day 4/5's differently sized loops are
     genuinely distinct choices again.
  4. Turnaround time is now a distinct, explicitly config-sourced term
     (`rc.LOOP_TURNAROUND_S`, defaulted to 0.0 with a TODO -- see below)
     alongside loop-stop duration and drive time, per Master Plan Sec 2.3.
  5. Loop-zone driving now uses its own speed (`rc.LOOP_CRUISE_SPEED_MS`,
     placeholder-defaulted -- see below), separate from the day's solved
     base-route cruise speed, matching "drive time at loop speeds" in the
     Master Plan's loop-economics section.
  6. Late-finish penalty conversion (loops mode) now uses the actual
     local marginal km/s rate (today's own solved base speed for
     same-day/terminal penalties; the *cached* marginal rate of tomorrow's
     own chosen action, `R[d+1]`, for penalties carried into tomorrow's
     window) instead of a flat `car.v_max_ms`, which overstated the value
     of schedule slack.
  7. NEW (found during rewrite): the terminal day (Day 8) previously had
     its late-finish penalty silently zeroed (`d < n_days - 1` guarded the
     entire penalty block) because there is no "tomorrow" to serve it at.
     It is now charged directly against Day 8's own contribution instead
     of vanishing.
  8. RACE_MODE == "completion"'s zero-trailering requirement isn't solved
     here (no trailering data exists yet -- see the placeholder below),
     but the structure is set up so that once trailered stretches are
     excluded from distance/energy, a day that can't be completed without
     them simply becomes infeasible in the existing Bellman recursion --
     no extra state dimension needed.
  9. A day with no feasible starting SOC now reports `alpha_day_pct[d] =
     nan` and `day_feasible[d] = False`, instead of silently defaulting to
     `car.soc_max_pct` (which read as "just charge to max", indistinguishable
     from "nothing works").
  10. Stop-charging energy no longer assumes solar noon at route km 0 for
      every stop. Control-stop and per-loop stop energy are now each
      estimated at that stop's own approximate elapsed clock time and
      route position (see the estimation caveats inline -- this is still
      an L1-appropriate approximation, not L2/L3-grade telemetry).
  11. The bare `except Exception` around the physics call is narrowed to
      exceptions the simulator is expected to raise for constraint
      violations; if literally every attempted action raises, `solve()`
      now raises `RuntimeError` instead of silently reporting the whole
      race as infeasible (that pattern almost always means a wiring bug,
      not a real result).
  12. `robustness_report()` now validates that "p10" is actually a member
      of `sc.DP_SOLAR_SCENARIOS` before using it as the headline metric,
      instead of silently reporting completion_prob = 0.0 on a label
      mismatch.
  13. `wind_provider` is now threaded through to `simulate_constant_speed`
      (ASSUMPTION: this assumes forward_sim's signature accepts it --
      flag to whoever owns forward_sim.py if that hasn't landed yet).
  14. Added a config-driven `rc.CHARGING_MODE` switch (normal vs.
      extended_2h) per the team's binary charging-mode request --
      ASSUMPTION about its exact physical meaning; see the inline note
      where it's applied and confirm with the team.

Trailering (upcoming, not yet wired): `_apply_trailering_exclusion` is a
named, marked no-op hook at the exact point where a route's red-flagged
"must trailer" stretches (Master Plan Sec 5.1) should be stripped from a
day's zones before distance/energy accounting. Left as pass-through per
instruction -- do not implement the real exclusion logic here yet.

Still-open placeholders inherited/introduced by this rewrite (belong in
race_config.py / solver_config.py / car_config.py per the project's
config discipline; defined locally with getattr-fallbacks since they
aren't there yet):
  - rc.LOOP_TURNAROUND_S (defaults to 0.0 -- Sec 4.2 turn-audit not wired)
  - rc.LOOP_CRUISE_SPEED_MS (defaults to car.v_max_ms -- no turning-circle
    -derived loop speed cap available yet)
  - rc.CHARGING_MODE (defaults to "normal")
  - _MAX_REPS_PER_NAMED_LOOP, _EXTENDED_EVENING_CHARGE_S below

CHANGELOG (v3 -- bug review round 2, 4 issues found in v2):
  1. `_zone_slope_profile`/`_looped_zone_slope_profile` returned an empty
     profile when `route is None` (e.g. a blind day before its Golden
     Envelope KML lands), which `_simulate_day` then skipped entirely --
     silently zeroing that zone's energy consumption while still
     counting its distance. Both now fall back to a flat (0% slope)
     profile of the correct length instead of an empty one.
  2. `_adjust_plan_for_today` compared real odometer distance driven
     against the loop zone's single-pass linestring width to infer
     Stage-2 progress; once any loop was repeated more than once those
     two numbers are no longer on the same scale, and the subtraction
     could badly under- or over-count (even zero out) remaining Stage 2.
     Fixed to either use real per-loop attempt counts if the caller has
     them (new optional `loops_completed_today` param on solve()/
     replan()), or conservatively report Stage 2 as untouched rather than
     guess.
  3. `_simulate_day` advanced its clock by driving time only, never by
     the loop-stop + turnaround time between zones, so every zone after
     the first stop was queried for solar at an artificially early
     time-of-day. Loop attempts are now built as one zone EACH (instead
     of tiled into one call per named loop) with an explicit
     `pre_stop_s` gap the clock advances through before that attempt is
     simulated.
  4. `robustness_report`'s `worst_alpha` skipped `nan` alphas entirely
     when aggregating, so a day infeasible in every single scenario
     never got a key at all -- a `KeyError` waiting for any downstream
     code indexing it directly. `worst_alpha` is now pre-populated with
     `nan` for every day the sweep covers.

Out of scope for this pass (flagging rather than guessing):
  - Deliberately choosing MORE loop attempts than "always finish on time"
    and accepting a bigger late-finish penalty as a strategic trade is not
    modeled; the enumeration bound (`_enumerate_loop_combos`) stays at the
    same "always fits" ceiling the original code used. Expanding this is
    a real, separate design question (unbounded combinatorics), not one
    of the 14 flagged issues.
  - Whether `rc.FINISH_CUTOFF_ABS_S` should vary by day (Day 8's 15:00
    timed finish reads differently from Days 1-7's 17:30 absolute
    cutoff in the regs) is left as-is (single flat constant, matching the
    original interface) -- verify against SR 2.22 before relying on this
    for Day 8 specifically.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging

import numpy as np

from configs import race_config as rc
from configs import solver_config as sc
from configs.car_config import CarState
from core.battery import Battery
from simulator.forward_sim import simulate_constant_speed

logger = logging.getLogger(__name__)

_FULL_BLIND_DAY_KM_PRIOR = 230.0   # 2024 Vryburg->Kimberley prior, Day 3
_DP_COMPLETION_MARGIN_PCT = 5.0    # alpha_day buffer in RACE_MODE="completion"
_INFEASIBLE = -1.0e9               # sentinel, distinct from any realistic
                                   # (possibly penalty-discounted-negative)
                                   # value V/val can legitimately take

_N_SLOPE_SEG = 10                  # per-zone slope-profile resolution

_CHARGING_MODE_NORMAL = "normal"
_CHARGING_MODE_EXTENDED_2H = "extended_2h"
_EXTENDED_EVENING_CHARGE_S = 2.0 * 3600.0   # discussion1.txt's "two hour
                                            # charging case" -- see
                                            # CHANGELOG item 14


def _is_feasible(v: float) -> bool:
    return v > _INFEASIBLE / 2.0


def _loop_cruise_speed_ms(car: CarState) -> float:
    """Loop-zone driving speed, distinct from base-route cruise speed
    (Master Plan Sec 2.3: "drive time at loop speeds"). Real value should
    come from the turning-circle/lateral-accel audit (Sec 4.2, block 2.2);
    until that lands, default to the car's max speed so behaviour doesn't
    silently change for teams that haven't set this yet.
    """
    return max(getattr(rc, "LOOP_CRUISE_SPEED_MS", car.v_max_ms), 1e-6)


@dataclasses.dataclass(frozen=True)
class _DayPlan:
    """A day's route in driving order: Stage 1 -> named loop zone(s), in
    listed order -> Stage 2 (Table 1's column order). `loops` entries are
    (name, single-pass_km).
    """
    stage1_km: float
    stage2_km: float
    loops: tuple[tuple[str, float], ...]


def _get_day_plan(day_index: int) -> _DayPlan:
    """Nominal (not-yet-driven) route plan for a day, from route notes.

    Full-blind days (stage1_km is None) fall back to the distance-
    distribution prior as an undifferentiated Stage-1-only block (Master
    Plan Sec 7: "blind days as distance distributions ... re-solved the
    evening the Golden Envelope lands"). Days with no resolved loop yet
    (half- or full-blind) get one placeholder loop entry rather than
    silently losing the loop opportunity, matching the original code's
    intent.
    """
    note = rc.DAY_ROUTE_NOTES[day_index]
    if note["stage1_km"] is None:
        stage1_km, stage2_km = _FULL_BLIND_DAY_KM_PRIOR, 0.0
    else:
        stage1_km, stage2_km = note["stage1_km"], note["stage2_km"]
    loops = tuple(note["loops"]) if note["loops"] else (
        ("blind_loop_placeholder", rc.BLIND_LOOP_PLACEHOLDER_KM),
    )
    return _DayPlan(stage1_km, stage2_km, loops)


def _adjust_plan_for_today(plan: _DayPlan,
                           distance_done_km_today: float,
                           loops_completed_today: dict[str, int] | None = None
                           ) -> _DayPlan:
    """Reduce a day's plan for a same-day re-plan given km already driven.

    If the car hasn't yet reached the loop zone, Stage 1 shrinks and the
    full loop menu is still selectable. Once the car has passed Stage 1,
    SR 2.29's "intent declared before every attempt" means today's loop
    decision is already made and cannot be revisited -- only remaining
    Stage 2 counts.

    Bug (round 2, item 2): once ANY named loop has been repeated more
    than once, `distance_done_km_today` (real odometer distance, which
    grows by one loop-length PER REPEAT) and the loop zone's single-pass
    linestring width are no longer on the same scale -- comparing them
    directly attributed repeat-loop distance to Stage 2 and could zero it
    out even though the car hadn't reached Stage 2 at all. There is no
    way to reconstruct which specific loop(s) were repeated how many
    times from total distance alone (different loops have different
    lengths, so the decomposition isn't even unique in general).

    `loops_completed_today`, if the caller has it from telemetry, removes
    the ambiguity directly. Without it, this conservatively reports
    Stage 2 as entirely untouched (loops closed, full Stage 2 still
    ahead) rather than guessing -- possibly overstating remaining
    distance/time slightly, but never silently corrupting it the way the
    original subtraction could.
    """
    if distance_done_km_today <= plan.stage1_km:
        return _DayPlan(plan.stage1_km - distance_done_km_today,
                       plan.stage2_km, plan.loops)

    if loops_completed_today is not None:
        loop_drive_km = sum(
            loops_completed_today.get(name, 0) * km
            for name, km in plan.loops)
        stage2_done_km = max(
            0.0, distance_done_km_today - plan.stage1_km - loop_drive_km)
        return _DayPlan(0.0, max(0.0, plan.stage2_km - stage2_done_km), ())

    return _DayPlan(0.0, plan.stage2_km, ())


def _enumerate_loop_combos(loops: tuple[tuple[str, float], ...],
                           t_per_attempt_s: list[float],
                           t_loop_budget_upper_s: float):
    """Yield (reps, total_loop_km, total_loop_time_s) for every combination
    of per-named-loop attempt counts that could conceivably fit in the
    day's loop-time budget (computed assuming the base route is driven at
    v_max with zero lateness -- the same ceiling the pre-rewrite code
    used; see the module docstring's "out of scope" note on why this
    isn't extended to deliberately-late combos).
    """
    if not loops:
        yield (), 0.0, 0.0
        return
    per_loop_caps = []
    for t_attempt in t_per_attempt_s:
        cap = (int(t_loop_budget_upper_s // t_attempt)
              if t_attempt > 0 else 0)
        per_loop_caps.append(max(0, cap))
    for reps in itertools.product(*(range(cap + 1) for cap in per_loop_caps)):
        total_time_s = sum(n * t for n, t in zip(reps, t_per_attempt_s))
        if total_time_s > t_loop_budget_upper_s:
            continue
        total_km = sum(n * loops[i][1] for i, n in enumerate(reps))
        yield reps, total_km, total_time_s


def _zone_slope_profile(route, start_km: float, dist_km: float,
                        n_seg: int = _N_SLOPE_SEG):
    """Slope samples for a single, non-repeating stretch of route.

    `route is None` (no KMZ/parquet yet -- e.g. a blind day before its
    Golden Envelope lands) must NOT be treated the same as `dist_km <= 0`:
    a genuinely zero-length zone has no energy to account for, but a real
    zone with unknown topography still costs real energy to drive. Fall
    back to a flat (0% slope) assumption rather than returning an empty
    profile, which `_simulate_day` would otherwise skip entirely --
    silently zeroing that zone's energy consumption (fix, bug review
    round 2, item 1).
    """
    if dist_km <= 0:
        return np.zeros(0), 0.0
    if route is None:
        return np.zeros(n_seg), dist_km / n_seg
    xs = np.linspace(start_km * 1000.0, (start_km + dist_km) * 1000.0,
                     n_seg + 1)
    mid_xs = (xs[:-1] + xs[1:]) / 2.0
    return route.slope_pct_at(mid_xs), dist_km / n_seg


def _looped_zone_slope_profile(route, zone_start_km: float,
                               single_pass_km: float,
                               n_seg: int = _N_SLOPE_SEG):
    """Slope samples for ONE pass of a named loop attempt.

    Returns a single-attempt profile now, not a `reps`-tiled one: each
    attempt needs its own preceding stop/turnaround gap in the clock
    (SR 2.29 -- mandatory stop before EVERY attempt, not just the loop
    zone as a whole), which only works if `_simulate_day` sees each
    attempt as its own zone (fix, bug review round 2, item 3; see
    `_build_zones`). `route is None` falls back to flat terrain, same
    reasoning as `_zone_slope_profile` above.
    """
    if single_pass_km <= 0:
        return np.zeros(0), 0.0
    if route is None:
        return np.zeros(n_seg), single_pass_km / n_seg
    xs = np.linspace(zone_start_km * 1000.0,
                     (zone_start_km + single_pass_km) * 1000.0, n_seg + 1)
    mid_xs = (xs[:-1] + xs[1:]) / 2.0
    return route.slope_pct_at(mid_xs), single_pass_km / n_seg


def _apply_trailering_exclusion(route, zones):
    """PLACEHOLDER -- not wired yet.

    Once a route carries a red-flagged "must trailer" mask (Master Plan
    Sec 5.1, mechanism-A gradient-map output), this hook is where those
    stretches must be stripped out of each zone's slope profile / seg_len
    before simulate_constant_speed sees them, and the corresponding
    distance must be dropped from `dist_km` at the call site (see the
    comment next to `dist_km = ...` in solve()) so trailered km are
    counted as neither driven distance nor battery-relevant energy.

    Awaiting the trailering data structure from compliance/turn_audit.py /
    analysis/trailering.py. No-op for now -- do not implement the real
    exclusion logic here yet (per instruction).
    """
    return zones


def _build_zones(route, plan: _DayPlan, reps: tuple[int, ...],
                 route_offset_km: float, v_base_ms: float,
                 loop_speed_ms: float, loop_pre_attempt_stop_s: float):
    """Ordered (slope_pct, seg_len_km, v_ms, pre_stop_s) zones for one
    day's action, in driving order: Stage 1 -> each named loop's repeats,
    one zone PER ATTEMPT, in listed order -> Stage 2.

    Each loop-attempt zone carries `pre_stop_s` (SR 2.29's mandatory
    stop + turnaround BEFORE every attempt, including the first) so
    `_simulate_day` can advance the clock across it before querying
    solar for that attempt -- fixing the clock-drift bug where later
    zones (subsequent attempts, Stage 2) were queried at an
    artificially-early time-of-day (bug review round 2, item 3). Stage
    zones carry pre_stop_s=0.0: the once-per-day control stop is priced
    separately in solve() and does not interrupt this zone chain.

    Loop zones still advance the position cursor by their nominal
    single-pass width regardless of `reps` (the physical road is still
    there even on an attempt count of zero for that specific loop).
    """
    zones = []
    if plan.stage1_km > 0:
        slope, seg_len = _zone_slope_profile(route, route_offset_km,
                                             plan.stage1_km)
        zones.append((slope, seg_len, v_base_ms, 0.0))

    cursor_km = route_offset_km + plan.stage1_km
    for i, (_name, km_i) in enumerate(plan.loops):
        n_i = reps[i] if reps else 0
        for _attempt in range(n_i):
            slope, seg_len = _looped_zone_slope_profile(
                route, cursor_km, km_i)
            zones.append((slope, seg_len, loop_speed_ms,
                         loop_pre_attempt_stop_s))
        cursor_km += km_i

    if plan.stage2_km > 0:
        slope, seg_len = _zone_slope_profile(route, cursor_km,
                                             plan.stage2_km)
        zones.append((slope, seg_len, v_base_ms, 0.0))

    return zones


def _simulate_day(car: CarState, zones, solar_provider, wind_provider,
                  t0_s: float, start_soc_pct: float) -> float:
    """Run each zone through the physics engine in order, chaining SOC
    and elapsed clock time between zones (needed because the day is no
    longer one uniform speed -- CHANGELOG item 5).

    `pre_stop_s` (loop-stop + turnaround time immediately preceding this
    zone, SR 2.29) is added to the clock before the zone is simulated, so
    solar queries for later zones reflect real elapsed time-of-day
    instead of only counting driving time between zones (bug review
    round 2, item 3).
    """
    soc = start_soc_pct
    t_s = t0_s
    for slope_pct, seg_len_km, v_ms, pre_stop_s in zones:
        t_s += pre_stop_s
        if seg_len_km <= 0 or len(slope_pct) == 0:
            continue
        sim_out = simulate_constant_speed(
            car, slope_pct, seg_len_km, v_ms, solar_provider, wind_provider,
            t_s, soc)
        soc = sim_out["final_soc_pct"]
        dist_km = seg_len_km * len(slope_pct)
        t_s += (dist_km * 1000.0) / v_ms
    return soc


def solve(routes: list, car: CarState, solar_provider, wind_provider,
          start_soc_pct: float, *, start_day_index: int = 0,
          remaining_km_today: float | None = None,
          elapsed_s_today: float = 0.0,
          control_stop_taken_today: bool = False,
          extra_stoppage_s: float = 0.0,
          loops_completed_today: dict[str, int] | None = None) -> dict:
    """Backward Bellman DP over (day, SOC bucket).

    Defaults reproduce a full pre-race solve. Keyword args let it resolve
    from mid-race instead (see replan()/robustness_report()).
    extra_stoppage_s applies to every remaining day. loops_completed_today
    (optional, {loop_name: attempt_count} from telemetry) disambiguates
    today's Stage 2 progress once any loop has been repeated more than
    once -- see `_adjust_plan_for_today`; omit it if unknown, and today's
    Stage 2 will conservatively be treated as untouched.

    Returns dict(loop_plan, alpha_day_pct, day_feasible, feasible,
    start_day_index). loop_plan/alpha_day_pct/day_feasible are keyed by
    absolute day_index; loop_plan[d] is {loop_name: attempt_count}.
    """
    n_days = rc.N_RACE_DAYS
    if not (0 <= start_day_index < n_days):
        raise ValueError(f"start_day_index {start_day_index} out of range "
                        f"[0, {n_days})")

    soc_buckets = np.arange(car.soc_min_pct, car.soc_max_pct + 1e-9,
                            sc.DP_SOC_BUCKET_PCT)
    n_buckets = len(soc_buckets)

    # V[d, s]: optimal value-to-go from day d, SOC bucket s. Terminal
    # condition V[n_days, :] = 0 matches the paper's treatment of a
    # non-unique terminal cost (Sec III-D): each day's own gain/loss is
    # folded into the running sum instead.
    V = np.full((n_days + 1, n_buckets), _INFEASIBLE)
    V[n_days, :] = 0.0
    # R[d, s]: the marginal km/s rate of the chosen action at (d, s) --
    # cached so a penalty landing on day d's window (from day d-1's
    # lateness) can be converted at day d's own realized rate rather than
    # an arbitrary car.v_max_ms (CHANGELOG item 6).
    R = np.zeros((n_days + 1, n_buckets))

    policy_action: list[list[tuple[int, ...] | None]] = [
        [None] * n_buckets for _ in range(n_days)
    ]
    policy_next_soc = np.zeros((n_days, n_buckets), dtype=int)
    day_loops_used: dict[int, tuple[tuple[str, float], ...]] = {}

    sim_attempts = 0
    sim_errors = 0

    for d in range(n_days - 1, start_day_index - 1, -1):
        is_today = (d == start_day_index)
        nominal_plan = _get_day_plan(d)

        if is_today and remaining_km_today is not None:
            base_km_full = nominal_plan.stage1_km + nominal_plan.stage2_km
            distance_done_km_today = max(
                0.0, base_km_full - max(0.0, remaining_km_today))
            plan = _adjust_plan_for_today(
                nominal_plan, distance_done_km_today,
                loops_completed_today if is_today else None)
            route_offset_km = distance_done_km_today
        else:
            plan = nominal_plan
            route_offset_km = 0.0

        day_loops_used[d] = plan.loops
        route = routes[d] if routes and d < len(routes) else None

        t_window = rc.day_finish_time_s(d) - rc.day_start_time_s(d)
        if is_today:
            t_window = max(0.0, t_window - elapsed_s_today)

        t_stops_base = rc.UNPLANNED_STOP_BUDGET_S + extra_stoppage_s
        if not (is_today and control_stop_taken_today):
            t_stops_base += rc.CONTROL_STOP_DURATION_S

        loop_speed_ms = _loop_cruise_speed_ms(car)
        loop_turnaround_s = getattr(rc, "LOOP_TURNAROUND_S", 0.0)
        t_per_attempt_s = [
            (km * 1000.0) / loop_speed_ms
            + rc.LOOP_STOP_DURATION_S + loop_turnaround_s
            for _, km in plan.loops
        ]

        base_km_today = plan.stage1_km + plan.stage2_km
        t_base_at_vmax_s = (base_km_today * 1000.0) / car.v_max_ms
        t_loop_budget_upper_s = max(
            0.0, t_window - t_stops_base - t_base_at_vmax_s)

        if rc.RACE_MODE == "completion":
            # Loops off in completion mode (Master Plan Sec 2.4).
            combos = [((0,) * len(plan.loops), 0.0, 0.0)]
        else:
            combos = list(_enumerate_loop_combos(
                plan.loops, t_per_attempt_s, t_loop_budget_upper_s))

        is_terminal_day = (d == n_days - 1)

        for s_idx, start_soc in enumerate(soc_buckets):
            best_val = _INFEASIBLE
            best_reps: tuple[int, ...] | None = None
            best_next_s = -1
            best_rate_ms = 0.0

            for reps, loop_km_total, _loop_time_total_s in combos:
                n_attempts = sum(reps) if reps else 0
                t_stops = t_stops_base + n_attempts * (
                    rc.LOOP_STOP_DURATION_S + loop_turnaround_s)
                t_loop_drive_s = (
                    (loop_km_total * 1000.0) / loop_speed_ms
                    if loop_km_total > 0 else 0.0)
                t_available_for_base_s = t_window - t_stops - t_loop_drive_s

                if t_available_for_base_s < t_base_at_vmax_s:
                    late_s = t_base_at_vmax_s - t_available_for_base_s
                    if (rc.day_finish_time_s(d) + late_s
                            > rc.FINISH_CUTOFF_ABS_S):
                        continue
                    v_base_ms = car.v_max_ms
                    penalty_s = rc.late_finish_penalty_min(
                        late_s / 60.0) * 60.0
                else:
                    v_base_ms = (
                        (base_km_today * 1000.0)
                        / max(t_available_for_base_s, 1e-6)
                        if base_km_today > 0 else car.v_max_ms)
                    penalty_s = 0.0

                # dist_km: base + loop km actually driven this action.
                # Once trailering exclusion (placeholder above) is wired,
                # any red-flagged stretch inside these zones must be
                # subtracted here too, not just from the simulator input.
                dist_km = base_km_today + loop_km_total

                zones = _build_zones(
                    route, plan, reps, route_offset_km, v_base_ms,
                    loop_speed_ms, rc.LOOP_STOP_DURATION_S + loop_turnaround_s)
                zones = _apply_trailering_exclusion(route, zones)

                t0_s = (rc.day_start_time_s(d)
                       + (elapsed_s_today if is_today else 0.0))

                sim_attempts += 1
                try:
                    end_soc = _simulate_day(car, zones, solar_provider,
                                            wind_provider, t0_s, start_soc)
                except (ValueError, ArithmeticError) as exc:
                    # Narrowed from a bare `except Exception` (CHANGELOG
                    # item 11): only constraint-style rejections from the
                    # physics call are treated as "action infeasible".
                    sim_errors += 1
                    logger.debug("day %d combo %s rejected by simulator: %s",
                               d, reps, exc)
                    continue

                bat = Battery(car, end_soc)

                # --- Stop-charging energy (CHANGELOG item 10) ---
                # L1-appropriate approximation: estimate each stop's clock
                # time/position from the day's own solved geometry instead
                # of a fixed noon/km-0 guess. Still coarse (the exact
                # control-stop timing is an L2 decision variable per
                # Master Plan Sec 8) -- good enough for day-resolution
                # energy accounting, not a substitute for L2/L3 precision.
                elapsed_before_mid_s = (
                    (base_km_today / 2.0) * 1000.0 / max(v_base_ms, 1e-6)
                    if base_km_today > 0 else 0.0)
                t_cs_s = t0_s + elapsed_before_mid_s
                x_cs_km = route_offset_km + base_km_today / 2.0
                ghi_cs = solar_provider.ghi_wm2(t_cs_s, x_cs_km)
                p_solar_cs = (car.array_area_m2 * car.array_efficiency
                             * ghi_cs) - car.p_idle_w
                cs_active_s = (0.0 if (is_today and control_stop_taken_today)
                              else rc.CONTROL_STOP_DURATION_S)
                bat.apply_energy_wh((p_solar_cs * cs_active_s) / 3600.0)
                # Unplanned-stop budget: small aggregate buffer with no
                # single identifiable event -- reuses the same (t, x)
                # proxy as the control stop; a day-representative estimate
                # is as precise as this layer needs.
                bat.apply_energy_wh(
                    (p_solar_cs * (rc.UNPLANNED_STOP_BUDGET_S
                                  + extra_stoppage_s)) / 3600.0)

                cursor_km = route_offset_km + plan.stage1_km
                elapsed_before_loop_s = (
                    (plan.stage1_km * 1000.0) / max(v_base_ms, 1e-6)
                    if plan.stage1_km > 0 else 0.0)
                for i, (_name, km_i) in enumerate(plan.loops):
                    n_i = reps[i] if reps else 0
                    if n_i > 0:
                        t_loop_s = t0_s + elapsed_before_loop_s
                        ghi_loop = solar_provider.ghi_wm2(t_loop_s, cursor_km)
                        p_solar_loop = (car.array_area_m2
                                       * car.array_efficiency
                                       * ghi_loop) - car.p_idle_w
                        bat.apply_energy_wh(
                            (p_solar_loop * n_i
                            * rc.LOOP_STOP_DURATION_S) / 3600.0)
                    cursor_km += km_i

                if not bat.feasible():
                    continue

                if not is_terminal_day:
                    t_morn_start = rc.BATTERY_UNSEAL_TIME_S
                    t_morn_end = rc.day_start_time_s(d + 1)
                    morning_dur = max(0.0, t_morn_end - t_morn_start)
                    ghi_morn = solar_provider.ghi_wm2(
                        t_morn_start + morning_dur / 2.0, 0.0)
                    p_solar_morn = (car.array_area_m2 * car.array_efficiency
                                   * ghi_morn) - car.p_idle_w
                    bat.apply_energy_wh((p_solar_morn * morning_dur) / 3600.0)

                    if (getattr(rc, "CHARGING_MODE", _CHARGING_MODE_NORMAL)
                            == _CHARGING_MODE_EXTENDED_2H):
                        # ASSUMPTION (CHANGELOG item 14, discussion1.txt's
                        # binary charging-mode ask) -- interpreted here as
                        # an additional, discretionary up-to-2h static
                        # charge stacked on top of the mandatory
                        # morning-only recovery, taken near end-of-day
                        # before pack sealing (Master Plan Sec 2.2's
                        # pre-finish-line stationary-charge tactic is the
                        # closest reg-legal analogue). CONFIRM the exact
                        # mechanism this is meant to represent before
                        # relying on it.
                        t_eve_s = rc.day_finish_time_s(d) \
                            - _EXTENDED_EVENING_CHARGE_S / 2.0
                        ghi_eve = solar_provider.ghi_wm2(t_eve_s, dist_km)
                        p_solar_eve = (car.array_area_m2
                                      * car.array_efficiency
                                      * ghi_eve) - car.p_idle_w
                        bat.apply_energy_wh(
                            (p_solar_eve
                            * _EXTENDED_EVENING_CHARGE_S) / 3600.0)

                if not bat.feasible():
                    continue

                next_s_idx = int(np.clip(
                    np.searchsorted(soc_buckets, bat.soc_pct) - 1,
                    0, n_buckets - 1))

                if not is_terminal_day:
                    if not _is_feasible(V[d + 1][next_s_idx]):
                        continue
                    if penalty_s > 0.0:
                        # Tomorrow's OWN cached marginal rate, not today's
                        # v_max and not a flat constant (CHANGELOG item 6).
                        rate_ms = R[d + 1][next_s_idx]
                        if rc.RACE_MODE == "completion":
                            tmrw_window_s = max(
                                rc.day_finish_time_s(d + 1)
                                - rc.day_start_time_s(d + 1), 1.0)
                            penalty_loss = penalty_s / tmrw_window_s
                        else:
                            penalty_loss = (penalty_s * rate_ms) / 1000.0
                    else:
                        penalty_loss = 0.0

                    val = (V[d + 1][next_s_idx] + 1.0 - penalty_loss
                          if rc.RACE_MODE == "completion"
                          else dist_km + V[d + 1][next_s_idx] - penalty_loss)
                else:
                    # Terminal day (CHANGELOG item 7): no "tomorrow" to
                    # serve the penalty at (Day 8's 15:00 timed finish is
                    # the end of the race), so charge it directly against
                    # today's own contribution instead of silently
                    # dropping it.
                    if rc.RACE_MODE == "completion":
                        penalty_loss = penalty_s / max(t_window, 1.0)
                        val = 1.0 - penalty_loss
                    else:
                        penalty_loss_km = (penalty_s * v_base_ms) / 1000.0
                        val = dist_km - penalty_loss_km

                if val > best_val:
                    best_val = val
                    best_reps = reps
                    best_next_s = next_s_idx
                    best_rate_ms = v_base_ms

            V[d][s_idx] = best_val
            R[d][s_idx] = best_rate_ms
            policy_action[d][s_idx] = best_reps
            policy_next_soc[d][s_idx] = best_next_s

    if sim_attempts > 0 and sim_errors == sim_attempts:
        # Every single action raised -- almost certainly a simulator or
        # config wiring defect, not a genuinely all-infeasible race.
        # Reporting this as "day 8 is infeasible" would hide the real
        # problem (CHANGELOG item 11).
        raise RuntimeError(
            f"solve(): all {sim_attempts} simulate_constant_speed calls "
            f"raised -- refusing to report this as day-by-day "
            f"infeasibility. Check the simulator/config wiring.")

    alpha_day_pct: dict[int, float] = {}
    day_feasible: dict[int, bool] = {}
    for d in range(start_day_index, n_days):
        feasible_idx = np.where(V[d] > _INFEASIBLE / 2.0)[0]
        day_feasible[d] = bool(feasible_idx.size)
        alpha_day_pct[d] = (float(soc_buckets[feasible_idx[0]])
                           if feasible_idx.size else float("nan"))

    if rc.RACE_MODE == "completion":
        alpha_day_pct = {
            d: (float(np.clip(a + _DP_COMPLETION_MARGIN_PCT,
                             car.soc_min_pct, car.soc_max_pct))
                if day_feasible[d] else a)
            for d, a in alpha_day_pct.items()
        }

    loop_plan: dict[int, dict[str, int]] = {}
    curr_s = int(np.clip(np.searchsorted(soc_buckets, start_soc_pct) - 1,
                        0, n_buckets - 1))
    feasible = True
    for d in range(start_day_index, n_days):
        if not _is_feasible(V[d][curr_s]):
            loop_plan[d] = {}
            feasible = False
            break
        reps = policy_action[d][curr_s] or ()
        loops_this_day = day_loops_used[d]
        loop_plan[d] = ({name: n for (name, _km), n
                        in zip(loops_this_day, reps) if n > 0}
                       if reps else {})
        curr_s = policy_next_soc[d][curr_s]

    return dict(loop_plan=loop_plan, alpha_day_pct=alpha_day_pct,
               day_feasible=day_feasible, feasible=feasible,
               start_day_index=start_day_index)


def replan(routes: list, base_car: CarState, solar_provider, wind_provider,
          current_soc_pct: float, *, current_day_index: int,
          distance_done_km_today: float = 0.0,
          elapsed_s_today: float = 0.0,
          control_stop_taken_today: bool = False,
          capability_overrides: dict | None = None,
          loops_completed_today: dict[str, int] | None = None) -> dict:
    """Re-solve the rest of the race from where the car is now.

    capability_overrides is passed to dataclasses.replace(base_car, ...)
    for a degraded CarState (e.g. {"array_area_m2": 4.1}).
    loops_completed_today: see solve()'s docstring.
    """
    car = dataclasses.replace(base_car, **(capability_overrides or {}))
    nominal_plan = _get_day_plan(current_day_index)
    base_km_full = nominal_plan.stage1_km + nominal_plan.stage2_km
    remaining_km_today = max(0.0, base_km_full - distance_done_km_today)
    return solve(
        routes, car, solar_provider, wind_provider, current_soc_pct,
        start_day_index=current_day_index,
        remaining_km_today=remaining_km_today,
        elapsed_s_today=elapsed_s_today,
        control_stop_taken_today=control_stop_taken_today,
        loops_completed_today=loops_completed_today,
    )


def robustness_report(routes: list, car: CarState, solar_providers: dict,
                      wind_provider, start_soc_pct: float, *,
                      start_day_index: int = 0,
                      remaining_km_today: float | None = None,
                      elapsed_s_today: float = 0.0,
                      control_stop_taken_today: bool = False) -> dict:
    """Sweep DP_SOLAR_SCENARIOS x DP_STOPPAGE_SCENARIOS_S; return a real
    completion probability instead of one scenario's pass/fail.

    solar_providers needs one entry per DP_SOLAR_SCENARIOS label.
    """
    missing = [k for k in sc.DP_SOLAR_SCENARIOS if k not in solar_providers]
    if missing:
        raise ValueError(f"solar_providers missing scenario(s): {missing}")
    if "p10" not in sc.DP_SOLAR_SCENARIOS:
        # CHANGELOG item 12: without this check, a naming/casing mismatch
        # silently zeroed the headline completion_prob instead of erroring.
        raise ValueError(
            "'p10' is not in sc.DP_SOLAR_SCENARIOS -- the headline "
            "completion_prob has no scenario to compute from.")

    scenarios = []
    for solar_label in sc.DP_SOLAR_SCENARIOS:
        for stoppage_s in sc.DP_STOPPAGE_SCENARIOS_S:
            out = solve(
                routes, car, solar_providers[solar_label], wind_provider,
                start_soc_pct, start_day_index=start_day_index,
                remaining_km_today=remaining_km_today,
                elapsed_s_today=elapsed_s_today,
                control_stop_taken_today=control_stop_taken_today,
                extra_stoppage_s=stoppage_s,
            )
            scenarios.append(dict(solar=solar_label, stoppage_s=stoppage_s,
                                  feasible=out["feasible"],
                                  alpha_day_pct=out["alpha_day_pct"]))

    n = len(scenarios)
    n_feasible = sum(r["feasible"] for r in scenarios)

    # Headline completion_prob: feasibility fraction over the stoppage
    # sweep under worst-case P10 solar only (master plan spec), not an
    # average across all solar scenarios.
    p10 = [r for r in scenarios if r["solar"] == "p10"]
    n_p10 = len(p10)
    n_p10_feasible = sum(r["feasible"] for r in p10)

    # Pre-populate every day this sweep covers with nan so a day that is
    # infeasible in EVERY scenario still ends up with a key (nan) instead
    # of being silently absent -- the previous version's `continue` on
    # nan meant such a day never got a key at all, causing a KeyError for
    # any downstream consumer indexing worst_alpha[day] directly (bug
    # review round 2, item 4).
    worst_alpha: dict[int, float] = {
        d: float("nan") for d in range(start_day_index, rc.N_RACE_DAYS)
    }
    for r in scenarios:
        for day, alpha in r["alpha_day_pct"].items():
            if np.isnan(alpha):
                continue
            prev = worst_alpha[day]
            worst_alpha[day] = alpha if np.isnan(prev) else max(prev, alpha)

    return dict(
        completion_prob=(n_p10_feasible / n_p10) if n_p10 else 0.0,
        n_scenarios=n_p10, n_feasible=n_p10_feasible,
        n_scenarios_all=n, n_feasible_all=n_feasible,
        worst_case_alpha_day_pct=worst_alpha, scenarios=scenarios,
    )