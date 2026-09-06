"""
optimizers/singleday.py — L2 single-day velocity optimizer

L2 solves for the fastest feasible velocity profile for a committed route.
SOC is a hard feasibility constraint; solar curtailment is penalized so the
optimizer does not deliberately preserve a full battery while wasting useful
solar energy.

Requires scipy>=1.14 (pinned in Model/requirements.txt) for NonlinearConstraint
objects to work directly with method="SLSQP" in `minimize`.
"""

from __future__ import annotations

from tqdm import tqdm
import typing as _t
import numpy as np
import pandas as pd
import logging
from scipy.optimize import Bounds, NonlinearConstraint, differential_evolution, minimize

from configs.car_config import CarState
from configs import solver_config as SCFG
from configs import race_config
from core.route import Route
from core import physics

# Import the centralized forward integrator
from simulator import forward_sim

logger = logging.getLogger(__name__)

# ===============================================================================
# 0. Local config overrides
# ===============================================================================

CONTROL_SEGMENT_M = SCFG.CONTROL_SEGMENT_M

SHARP_TURN_HEADING_DELTA_DEG = 30.0      
SHARP_TURN_SPEED_LIMIT_KMH = 20.0       

# Hoisted from _splice_loops (was a local variable there) so
# apply_turn_speed_caps' loop-periodicity fix can use the EXACT same gap
# when computing the real-distance modulo period of a repeated lap
# (lap_len_m + this separator) — see _loop_local_turn_caps below.
_LOOP_SEPARATOR_M = 300.0

DE_POPSIZE = 8                            
DE_MAXITER = 60                           


# ===============================================================================
# 1. Sharp-turn speed caps
# ===============================================================================

# Cap on the physical length (m) a single flagged bearing-change point can
# "own" for the sharp-turn fraction below — see BUGFIX note in
# _sharp_turn_fraction. A real hairpin/sharp bend on this route class spans
# tens of metres, not kilometres.
MAX_TURN_SPAN_M = 50.0


def _turn_point_lengths(x: np.ndarray, bearing: np.ndarray,
                         heading_delta_threshold_deg: float,
                         max_turn_span_m: float = MAX_TURN_SPAN_M
                         ) -> np.ndarray:
    """Physical length (m) attributable to a sharp turn at each route point.

    Zero for points whose bearing change is below threshold. For flagged
    points, the length is half the gap to the previous point + half the gap
    to the next, CAPPED at max_turn_span_m so an isolated flagged point next
    to a large inter-point gap (sparse routing-API sampling on an otherwise
    straight stretch) can't inherit that whole gap as "turn length".
    """
    if len(x) == 0:
        return np.zeros(0)
    raw_delta = np.diff(bearing, prepend=bearing[0])
    wrapped = (raw_delta + 180.0) % 360.0 - 180.0
    sharp_point = np.abs(wrapped) >= heading_delta_threshold_deg
    if len(x) > 1:
        gap_prev = np.diff(x, prepend=x[0])
        gap_next = np.diff(x, append=x[-1])
        own_len = np.minimum((gap_prev + gap_next) / 2.0, max_turn_span_m)
    else:
        own_len = np.zeros(len(x))
    return np.where(sharp_point, own_len, 0.0)


def _sharp_turn_fraction(route: Route, seg_start_m: np.ndarray, seg_len_m: float,
                          heading_delta_threshold_deg: float) -> np.ndarray:
    """Physical-length fraction of each control segment flagged as a sharp
    turn — i.e. (metres of segment that are actually a sharp turn) / (segment
    length), NOT (count of flagged route points) / (count of route points).

    BUGFIX: the previous implementation divided the COUNT of flagged route
    points by the COUNT of route points landing in the segment. Route points
    come straight from the routing API's own profile output and are NOT
    evenly spaced — curvy stretches get sampled far more densely than
    straight ones by most routing APIs. That made the computed "fraction"
    track point DENSITY, not real curviness: a segment with only a handful
    of sparse points, one of which is flagged, could score frac >= 0.5 and
    collapse apply_turn_speed_caps' blended speed bound for the WHOLE 10km
    control segment down near SHARP_TURN_SPEED_LIMIT_KMH, even though the
    real turn physically spans a few tens of metres of that segment. This
    is the root cause of the "random" ~29-30 km/h 10km-segment crawls seen
    in production output that had no relationship to actual route curvature
    or SOC/energy pressure. Using real point spacing (_turn_point_lengths)
    and dividing by the segment's own physical length fixes this regardless
    of how densely any given stretch happens to be sampled.
    """
    if not route: return np.zeros(len(seg_start_m))
    x = route.df["distance_m"].to_numpy()
    bearing = route.df["bearing_deg"].to_numpy()
    flagged_len = _turn_point_lengths(x, bearing, heading_delta_threshold_deg)

    seg_end_m = seg_start_m + seg_len_m
    frac = np.zeros(len(seg_start_m))
    for i, (s, e) in enumerate(zip(seg_start_m, seg_end_m)):
        in_seg = (x >= s) & (x < e)
        if np.any(in_seg):
            frac[i] = float(np.sum(flagged_len[in_seg])) / seg_len_m
    return np.clip(frac, 0.0, 1.0)


def _loop_local_sharp_turn_frac(loop_df: pd.DataFrame, seg_len_m: float,
                                heading_delta_threshold_deg: float
                                ) -> tuple[np.ndarray, float]:
    """Sharp-turn fraction computed on ONE lap's own local geometry (loop-
    relative distance starting at 0), binned into fixed-size local control
    segments — evaluated ONCE per unique loop name, then tiled identically
    across every rep of that loop (see _loop_local_turn_caps).

    Root cause this fixes (workplan "loop velocity not periodic"): the old
    _sharp_turn_fraction rebinned bearing deltas over the FULL ABSOLUTE
    spliced-route position. That meant (a) the single synthetic separator
    row _splice_loops inserts between reps could manufacture a bogus heading
    discontinuity right at the lap seam, and (b) a real sharp turn near a lap
    boundary could land in control-segment N on lap 1 but N+1 on lap 2 purely
    from floating-point/offset drift between reps — a binning artifact, not a
    real speed limit changing. Computing the cap once on the lap's own local
    geometry and tiling it removes both: every rep sees the identical cap at
    the identical loop-relative position.
    """
    # BUGFIX (same root cause as _sharp_turn_fraction above): use physical
    # flagged-length / segment-length, not flagged-point-count / point-count,
    # so this loop-local fraction isn't at the mercy of how densely the
    # loop's own .save file happened to be sampled.
    if loop_df is None or len(loop_df) == 0:
        return np.zeros(1), 0.0
    x_local = loop_df["distance_m"].to_numpy()
    x_local = x_local - x_local[0]
    bearing = loop_df["bearing_deg"].to_numpy()
    flagged_len = _turn_point_lengths(x_local, bearing, heading_delta_threshold_deg)

    lap_len_m = float(x_local[-1]) if len(x_local) else 0.0
    n_local_segs = max(1, int(np.ceil(lap_len_m / seg_len_m))) if lap_len_m > 0 else 1
    frac = np.zeros(n_local_segs)
    for i in range(n_local_segs):
        s, e = i * seg_len_m, (i + 1) * seg_len_m
        in_seg = (x_local >= s) & (x_local < e)
        if np.any(in_seg):
            frac[i] = float(np.sum(flagged_len[in_seg])) / seg_len_m
    return np.clip(frac, 0.0, 1.0), lap_len_m


def _build_loop_local_turn_caps(loop_geoms: dict | None,
                                loops_committed: list[tuple[str, float]],
                                seg_len_m: float,
                                heading_delta_threshold_deg: float) -> dict:
    """One (local_frac_array, lap_len_m) pair per UNIQUE loop name in
    loops_committed, built from that loop's own geometry file (or a
    zero-turns synthetic fallback — flat placeholder legs have no real
    bearing data to flag turns from)."""
    caps: dict[str, tuple[np.ndarray, float]] = {}
    for name, km in loops_committed:
        if name in caps:
            continue
        geom = loop_geoms.get(name) if loop_geoms else None
        if geom is not None and len(geom) > 1:
            caps[name] = _loop_local_sharp_turn_frac(
                geom, seg_len_m, heading_delta_threshold_deg)
        else:
            caps[name] = (np.zeros(1), km * 1000.0)
    return caps


def _loop_occurrence_ranges(route: Route, name: str) -> list[tuple[float, float]]:
    """Exact (x_start_m, x_end_m) for every contiguous rep of loop `name`
    found directly in the spliced route's own seg_type column — NOT sampled
    at CONTROL_SEGMENT_M points. This is exact regardless of how a control
    segment's start happens to quantize against a lap boundary, which is
    what made the modulo-period approach still show occasional non-periodic
    dips right at the FIRST/LAST control segment of a lap (the segment whose
    CONTROL_SEGMENT_M window straddles the lap boundary): that boundary
    segment's single-point seg_type sample could land on the wrong side of
    the join, silently falling back to the buggy globally-binned fraction
    for exactly the segments most likely to sit near a real corner.
    """
    tag = f"loop_{name}"
    seg_type = route.df["seg_type"].to_numpy()
    x = route.df["distance_m"].to_numpy(dtype=float)
    is_loop = (seg_type == tag)
    ranges: list[tuple[float, float]] = []
    n = len(is_loop)
    i = 0
    while i < n:
        if is_loop[i]:
            j = i
            while j + 1 < n and is_loop[j + 1]:
                j += 1
            ranges.append((float(x[i]), float(x[j])))
            i = j + 1
        else:
            i += 1
    return ranges


def apply_turn_speed_caps(route: Route, v_max_kmh: np.ndarray,
                           seg_start_m: np.ndarray,
                           seg_len_m: float = CONTROL_SEGMENT_M,
                           heading_delta_threshold_deg: float = SHARP_TURN_HEADING_DELTA_DEG,
                           turn_speed_limit_kmh: float = SHARP_TURN_SPEED_LIMIT_KMH,
                           loop_geoms: dict | None = None,
                           loops_committed: list[tuple[str, float]] | None = None,
                           ) -> np.ndarray:
    """Blend v_max toward turn_speed_limit_kmh in proportion to how much of the
    segment is actually a sharp turn, instead of capping the whole segment for
    a single flagged point.

    loop_geoms / loops_committed (optional — pass the same values given to
    _splice_loops for this day): when a control segment falls inside a
    repeated loop, its fraction is OVERRIDDEN with a loop-local, tiled value
    (see _loop_local_sharp_turn_frac) instead of the globally-binned one, so
    the cap is periodic across reps. Tiling is keyed to each occurrence's
    OWN exact start position (_loop_occurrence_ranges, found directly in
    route.df's seg_type column) rather than an assumed constant lap period —
    this means every rep, INCLUDING its boundary control segments, maps to
    exactly the same loop-local index with zero drift. Omitting these keeps
    the prior (non-periodicity-corrected) global-binning behaviour for
    callers that haven't been updated — nothing regresses.
    """
    frac = _sharp_turn_fraction(route, seg_start_m, seg_len_m, heading_delta_threshold_deg)

    if route is not None and loops_committed:
        seg_start_m = np.asarray(seg_start_m, dtype=float)
        seg_mid_m = seg_start_m + seg_len_m / 2.0
        loop_caps = _build_loop_local_turn_caps(
            loop_geoms, loops_committed, seg_len_m, heading_delta_threshold_deg)
        for name in {n for n, _ in loops_committed}:
            local_frac, _lap_len_m = loop_caps.get(name, (np.zeros(1), 0.0))
            n_local = len(local_frac)
            for (rx0, rx1) in _loop_occurrence_ranges(route, name):
                # A control segment "belongs" to this occurrence if its
                # MIDPOINT falls inside the occurrence's exact range —
                # majority-overlap by construction, and unambiguous even for
                # the boundary segments a point-sample at seg_start_m alone
                # could misclassify.
                in_rep = (seg_mid_m >= rx0) & (seg_mid_m < rx1)
                if not np.any(in_rep):
                    continue
                local_pos = seg_start_m[in_rep] - rx0
                local_idx = np.clip(
                    np.floor(local_pos / seg_len_m).astype(int), 0, n_local - 1)
                frac[in_rep] = local_frac[local_idx]

    v_eff = 1.0 / (frac / turn_speed_limit_kmh + (1.0 - frac) / np.maximum(v_max_kmh, 1e-6))
    return v_eff


# ===============================================================================
# 1c. Sustained-power-aware speed caps (workplan fix — "no power-draw cap
# exists at all"). p_max_peak_w in core.physics.net_power is a POINTWISE
# clip on whatever speed forward_sim ends up driving; on its own that lets
# DE/GA/SLSQP propose a speed on a long climb that forward_sim will silently
# clip AFTER the fact (reactive, and inefficient — the optimizer thinks it's
# getting the mechanical power it asked for, then gets less, so its SOC/time
# tradeoff is wrong going in). This gives the optimizer the SAME budget
# up front, the same way apply_turn_speed_caps gives it the turn cap up
# front instead of letting forward_sim clip speed reactively at a corner.
# ===============================================================================

def _max_speed_for_power_budget(car: CarState, slope_pct: float,
                                 power_budget_w: float,
                                 v_lo_ms: float = 0.5,
                                 v_hi_ms: float | None = None,
                                 tol_ms: float = 0.05) -> float:
    """Largest steady-state speed (m/s) on this gradient whose electrical
    draw (core.physics.power_required_at_speed) does not exceed
    power_budget_w. Monotonic in v for any slope that actually draws power
    (drag + rolling + gravity all increase with v), so a simple bisection is
    exact and cheap — no need for scipy's general-purpose root finders here.
    """
    if v_hi_ms is None:
        v_hi_ms = car.v_max_ms
    slope_arr = np.array([slope_pct])

    def _draw_at(v_ms: float) -> float:
        return float(physics.power_required_at_speed(car, v_ms, slope_arr)[0])

    if _draw_at(v_hi_ms) <= power_budget_w:
        return float(v_hi_ms)          # whole speed range fits the budget
    if _draw_at(v_lo_ms) > power_budget_w:
        return float(v_lo_ms)          # even a crawl exceeds it (steep climb)

    lo, hi = v_lo_ms, v_hi_ms
    while (hi - lo) > tol_ms:
        mid = 0.5 * (lo + hi)
        if _draw_at(mid) <= power_budget_w:
            lo = mid
        else:
            hi = mid
    return float(lo)


def apply_sustained_power_caps(route: Route, car: CarState,
                               v_max_kmh: np.ndarray,
                               seg_start_m: np.ndarray) -> np.ndarray:
    """Cap each control segment's v_max so holding that speed over the
    segment's OWN average gradient would not exceed car.p_max_sustained_w.

    *** NOT called from solve() — see the comment at that call site. ***
    Shrinking the v_max BOUND this way turned out to be a bug, not a fix: it
    silently collapses the existing V_MAX_HARD_KMH=85 / CRUISE_SOFT_CAP_KMH=75
    two-tier scheme (configs/solver_config.py), which is deliberately
    implemented as bounds-ub=85 + a soft OBJECTIVE penalty above 70/75 — NOT
    as a shrunk bound — precisely so short high-power bursts stay available
    "if needed" (a loop, a cutoff, a SOC recovery) while the day-average
    stays in the comfortable 60-70 band. Because almost any real segment with
    a few percent of climb needs more than a few kW to hold 85 km/h, this
    function was pulling v_max down toward ~70 EVERYWHERE there was any
    grade at all — i.e. becoming the new de-facto hard ceiling instead of 85,
    exactly backwards from the intended "only if needed" design, and (worse)
    occasionally shrinking the day's feasible speed range enough that the
    terminal-SOC NonlinearConstraint became infeasible for SLSQP to satisfy
    (hence SOC finishing under the safety floor) and inflating solve() run
    time (DE/SLSQP fighting a badly-conditioned, unexpectedly narrow bound
    array instead of the wide one the objective's penalty terms were tuned
    against).

    Retained here as a standalone utility (e.g. for an analysis script that
    wants "what speed would the sustained cap alone allow on this grade") —
    the actual enforcement lives in _build_objective's sustained-power
    penalty term below, which penalizes res.sustained_power_over_budget_s
    (measured by simulator.forward_sim.SustainedPowerTracker over the
    ACTUAL chosen speed profile) the same way the existing speed-band and
    high-SOC terms work — pressure on the objective, not a hole cut in the
    search space.
    """
    sustained_cap_w = getattr(car, "p_max_sustained_w", None)
    if route is None or sustained_cap_w is None:
        return v_max_kmh
    slopes = route.slope_pct_array(np.asarray(seg_start_m, dtype=float))
    v_cap_ms = np.array([
        _max_speed_for_power_budget(car, float(s), float(sustained_cap_w))
        for s in slopes
    ])
    v_cap_kmh = v_cap_ms * 3.6
    return np.minimum(v_max_kmh, v_cap_kmh)


# ===============================================================================
# 1b. Loop splicing — ROOT-CAUSE FIX for the "free loop reps" bug
# ===============================================================================
#
# Previously, loops_committed only ever subtracted mandatory stop-time from
# the day's time budget (see solve()'s old allowed_time_s calc). The actual
# simulated route (passed into forward_sim) ALWAYS integrated only the base
# Stage1+Stage2 distance, regardless of how many loop reps were committed.
# That made extra reps a pure "free win" for Tier 3's DP, which scores combos
# as dist = base_km + reps*loop_km with no check against what singleday.solve
# actually simulated — so the optimizer kept picking absurd rep counts
# (14, 20, 22+), producing a "planning estimate" total distance (~5000km)
# with no relation to anything physically driven (~2000km), artificially
# depressed average speeds, and solar/motor/drain figures that couldn't
# reconcile.
#
# Fix: splice each committed loop rep's REAL geometry (or a flat synthetic
# fallback when no .save file exists for that loop) into the simulated route
# between Stage 1 and Stage 2 — matching _DayPlan's documented driving order
# (Stage 1 -> loop zone(s) -> Stage 2). Every rep now costs real integrated
# time/energy and is bounded by the same feasibility constraints as the rest
# of the day.

def _synthetic_loop_leg(km: float, route: Route | None) -> pd.DataFrame:
    """Flat-road fallback geometry for a loop with no matching .save file
    (confirmed to happen for some days — e.g. a 2nd named loop variant with
    no dedicated geometry file). 0% slope, full car-speed limit. Still costs
    real distance/time/energy in the simulation, unlike the old silent-no-op
    behaviour — it just can't reflect real terrain for that loop."""
    n = max(4, int(round(km * 1000.0 / 500.0)))  # ~500m sampling
    dist = np.linspace(0.0, km * 1000.0, n)
    if route is not None:
        try:
            last_lat, last_lon = route.latlon_at(route.total_m)
        except Exception:
            last_lat, last_lon = -26.2, 27.0
    else:
        last_lat, last_lon = -26.2, 27.0
    return pd.DataFrame({
        "distance_m": dist,
        "elevation_m": 0.0,
        "slope_pct": 0.0,
        "bearing_deg": 0.0,
        "lat": last_lat,
        "lon": last_lon,
        "v_max_ms": 90.0 / 3.6,
        "curvature_1pm": 0.0,
        "circle_id": 0,
        "red_flag_trailer": False,
        "control_stop": False,
        "seg_type": "loop_synthetic",
    })


def _splice_loops(route: Route, loop_geoms: dict | None,
                   loops_committed: list[tuple[str, float]],
                   stage1_km: float | None = None) -> Route:
    """Build the real simulated route for a day with committed loop reps.

    stage1_km: the day's REAL Stage-1 distance (plan.stage1_km — see
    trust_region.py's per-variant _DayPlan construction for Day 3, which
    already computes this correctly: 0.0 for Aryaman, a real value for
    Prahlad). This disambiguates two genuinely different route shapes that
    otherwise present IDENTICALLY as "stage1 bucket empty, stage2 bucket
    populated" from the dataframe alone, and previously got the SAME
    treatment (loop spliced AFTER the populated bucket) even though only
    one of them is actually correct that way:
      - Day 6-style mistagging: real Stage 1 content exists (stage1_km>0)
        but its source file got tagged "stage2" — the loop belongs AFTER
        that (mistagged) content, matching what _DayPlan expects.
      - A genuinely loop-first day (Day 3 Aryaman, stage1_km==0.0): there
        is NO Stage 1 leg at all — the loop belongs BEFORE whatever content
        sits in the "stage2" bucket. Splicing it after (the old behaviour)
        silently fed the optimizer a Stage2-then-loop route instead of the
        real loop-then-Stage2 route, corrupting that day's simulated
        solar/time/SOC curve (the driving order the physics integrates
        determines what time-of-day sun each segment sees).
    Passing stage1_km=None (old call sites, or genuinely unknown) falls
    back to the prior bucket-occupancy-only heuristic — Day-6-style
    behaviour — so nothing regresses for callers that haven't been updated.

    KNOWN SIMPLIFICATION: assumes the day's loops haven't started yet (fine
    for a full-day Tier 2 sample or the final extract_final_profiles pass,
    which cover the overwhelming majority of solve() calls). A mid-day
    replan that's already partway through a loop (dist_done_km landing
    inside a loop rep, not just Stage 1/2) is not specially handled here —
    it will treat any remaining committed reps as starting fresh from the
    current position, which is an approximation, not exact.

    Returns a NEW Route (does not mutate the input route). If
    loops_committed is empty or route is None, returns route unchanged.
    """
    if not loops_committed or route is None:
        return route

    base_df = route.df
    stage1 = base_df[base_df["seg_type"] == "stage1"].copy()
    stage2 = base_df[base_df["seg_type"] == "stage2"].copy()

    # CRASH FIX (Day 6) + DISAMBIGUATION FIX (Day 3 Aryaman): the original
    # guard only handled "both stage1 and stage2 empty". Some single-file
    # days get tagged "stage2" instead of "stage1" depending on the source
    # filename (Day 6: control-stop location == finish location, single
    # leg, no separate Stage 1 file to disambiguate the name against — its
    # file apparently reads as "Stage 2"). That left stage1 empty / stage2
    # populated, which the old code always resolved as "populated bucket is
    # the pre-loop content, loop after" — correct for Day 6, but WRONG for
    # Aryaman's Day 3, where stage1 is empty because there genuinely is no
    # Stage 1 (the day starts with the loop). Both cases look identical from
    # bucket occupancy alone; stage1_km is the real signal that tells them
    # apart.
    if len(stage1) == 0 and len(stage2) > 0:
        if stage1_km is not None and stage1_km < 1.0:
            # Genuinely no Stage 1 (e.g. Day 3 Aryaman) — loop comes FIRST.
            pre, post = base_df.iloc[0:0].copy(), stage2
        else:
            # stage1_km unknown (old call site) or real/nonzero (Day-6-style
            # mistagging) — keep the original crash-fix behaviour: treat the
            # populated bucket as the pre-loop content, loop after.
            pre, post = stage2, stage1
    elif len(stage1) == 0 and len(stage2) == 0:
        pre, post = base_df.copy(), base_df.iloc[0:0].copy()
    else:
        pre, post = stage1, stage2

    # blocks starts EMPTY (not [pre]) when pre has no rows, so the
    # loop-first case below doesn't anchor its first separator row on an
    # empty frame (blocks[-1].iloc[[-1]] on a 0-row pre would raise the same
    # IndexError the Day-6 crash fix already had to solve once).
    blocks = [pre] if len(pre) else []
    pre_end_m = float(pre["distance_m"].max()) if len(pre) else 0.0
    offset = pre_end_m

    def _separator_row(at_m: float) -> pd.DataFrame:
        # Anchor to whatever block was most recently appended — never assume
        # a specific named block ("stage1") is guaranteed non-empty (that
        # assumption is exactly what crashed on Day 6).
        row = blocks[-1].iloc[[-1]].copy()
        row["distance_m"] = at_m
        # BUGFIX: this used to be tagged plain "stage1" so forward_sim's
        # in_loop_zone flag (which gates the per-rep loop-stop solar credit)
        # correctly resets between reps. But _coarse_stage() in
        # trust_region.py buckets ANYTHING starting with "stage1" into the
        # reporting/plotting "stage1" stage — so with N committed reps, N-1
        # of these single-point separator rows got swept into the "stage1"
        # trace even though they sit deep inside (or after) the loop zone,
        # scattered every ~1 loop-length apart. That corrupted the "stage1"
        # stage's reported distance_km (it could read ~2x the real Stage 1
        # length) and interleaved stray, unrelated-time-of-day solar_w
        # samples into what should be a smooth morning ramp — the exact
        # "extra peak before/after the stop" artifact reported against the
        # dashboard's irradiance-vs-distance curve.
        # Fix: tag it "loop_separator" instead — _coarse_stage() already
        # buckets anything starting with "loop" as "loop" for reporting, so
        # this now correctly reports as part of the loop stage. forward_sim's
        # loop_stop_here test explicitly excludes "loop_separator" (see
        # simulator/forward_sim.py) so the per-rep credit-reset behavior is
        # unchanged — only the reporting bucket moves.
        row["seg_type"] = "loop_separator"
        return row

    for name, km in loops_committed:
        geom = loop_geoms.get(name) if loop_geoms else None
        if geom is not None and len(geom) > 0:
            leg = geom.copy()
            # Re-scale if the geometry file's own length differs materially
            # from the plan's nominal km for this loop (>50m mismatch).
            # Skip rescaling for the full-blind-day placeholder — its
            # "nominal km" is only a crude average-of-released-loops
            # estimate (race_config.BLIND_LOOP_PLACEHOLDER_KM), not a real
            # target length, so a real geometry file's actual length should
            # be trusted as-is rather than squeezed to match the estimate.
            file_len_m = float(leg["distance_m"].max()) or (km * 1000.0)
            is_placeholder_km = (name == "blind_loop_placeholder")
            if not is_placeholder_km and file_len_m > 0 and abs(file_len_m - km * 1000.0) > 50.0:
                scale = (km * 1000.0) / file_len_m
                leg["distance_m"] = leg["distance_m"] * scale
        else:
            leg = _synthetic_loop_leg(km, route)
        leg = leg.copy()
        leg["seg_type"] = f"loop_{name}"
        # Only insert a separator when there's real prior content to reset
        # away from (blocks non-empty). On a loop-first day (empty pre), the
        # very first loop leg starts immediately at x=0 with nothing before
        # it to separate from — a separator there would have nothing valid
        # to anchor its row on and isn't semantically needed anyway.
        if blocks:
            offset += _LOOP_SEPARATOR_M
            blocks.append(_separator_row(offset))
        leg["distance_m"] = leg["distance_m"] + offset
        offset = float(leg["distance_m"].max())
        blocks.append(leg)

    if len(post):
        s2 = post.copy()
        s2["distance_m"] = (s2["distance_m"] - pre_end_m) + offset
        blocks.append(s2)

    spliced = pd.concat(blocks, ignore_index=True)
    if "day" in base_df.columns and len(base_df):
        spliced["day"] = base_df["day"].iloc[0]
    return Route(spliced)


# ===============================================================================
# 2. Day-level evaluation
# ===============================================================================

class DayEvaluator:
    """Runs one candidate speed vector through physics + timing via forward_sim."""
    def __init__(self, route: Route, car: CarState, solar_provider,
                 wind_provider, t0_s: float, start_soc_pct: float,
                 seg_start_m: np.ndarray, seg_len_m: float = CONTROL_SEGMENT_M,
                 energy_grid_m: float = SCFG.ENERGY_GRID_M, *,
                 regen_cap_w: float | None = None,
                 cs_taken: bool = False,
                 loop_stop_duration_s: float | None = None,
                 unplanned_stop_budget_s: float | None = None):
        self.route = route
        self.car = car
        self.solar_provider = solar_provider
        self.wind_provider = wind_provider 
        self.t0_s = t0_s
        self.start_soc_pct = start_soc_pct
        self.seg_start_m = seg_start_m
        self.seg_len_m = seg_len_m
        self.energy_grid_m = energy_grid_m
        self.regen_cap_w = regen_cap_w
        self.cs_taken = cs_taken
        self.loop_stop_duration_s = loop_stop_duration_s
        self.unplanned_stop_budget_s = unplanned_stop_budget_s
        self._cache: dict[bytes, forward_sim.DayEvalResult] = {}

    def __call__(self, v_kmh: np.ndarray) -> forward_sim.DayEvalResult:
        key = np.asarray(v_kmh, dtype=float).round(6).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._simulate(np.asarray(v_kmh, dtype=float))
        self._cache[key] = result
        return result

    def _simulate(self, v_kmh: np.ndarray) -> forward_sim.DayEvalResult:
        return forward_sim.simulate_variable_speed(
            v_kmh=v_kmh, route=self.route, car=self.car,
            solar_provider=self.solar_provider, wind_provider=self.wind_provider,
            t0_s=self.t0_s, start_soc_pct=self.start_soc_pct,
            seg_start_m=self.seg_start_m, seg_len_m=self.seg_len_m,
            energy_grid_m=self.energy_grid_m,
            regen_cap_w=self.regen_cap_w, cs_taken=self.cs_taken,
            loop_stop_duration_s=self.loop_stop_duration_s,
            unplanned_stop_budget_s=self.unplanned_stop_budget_s,
        )


# ===============================================================================
# 3. Objective / constraints
# ===============================================================================

def _build_objective(evaluator: DayEvaluator) -> _t.Callable[[np.ndarray], float]:
    """Minimize time, with a secondary penalty for solar curtailment.

    The old objective maximized end-of-day SOC, so a 100% SOC solution was
    preferred even when it finished unnecessarily early. That is backwards
    for the race: the SOC requirement is a constraint, while time/distance is
    what the race strategy should optimize.

    ``SOLAR_UNDERUTIL_WEIGHT`` is expressed as equivalent seconds per wasted
    Wh, making the two terms dimensionally comparable.
    """
    _w = float(SCFG.SOLAR_UNDERUTIL_WEIGHT)
    # Battery-safety: seconds of penalty per (SOC-fraction·second) spent above
    # the safe band. Pushes the profile to draw the pack down (drive faster)
    # instead of coasting at ~100% — addresses both "we can't sit at full SOC"
    # and "the model under-drives / leaves speed on the table".
    _w_high = float(getattr(SCFG, "SOC_HIGH_PENALTY_WEIGHT", 0.0))
    # SOFT SPEED BAND (directive 22/08): a convex, two-zone penalty on the
    # per-segment TARGET speed that keeps the day-average in the 60-70 band with
    # a ~75 normal ceiling, while leaving the hard bound at ~85 for the rare
    # "only if needed" burst. Computed straight from the decision vector (the
    # reported average IS mean(v_kmh)), so it's smooth and SLSQP-friendly.
    #   * gentle quadratic above CRUISE_COMFORT_KMH  → free cruise settles ~71,
    #     average ~65-68 (reproduces the old hard-70 behaviour without a wall);
    #   * steep quadratic above CRUISE_SOFT_CAP_KMH  → 75-85 is expensive, taken
    #     only when a hard constraint (cutoff/SOC) or a loop's km makes it pay.
    _v_comfort = float(getattr(SCFG, "CRUISE_COMFORT_KMH", 70.0))
    _v_softcap = float(getattr(SCFG, "CRUISE_SOFT_CAP_KMH", 75.0))
    _w_comfort = float(getattr(SCFG, "SPEED_COMFORT_PENALTY_WEIGHT", 0.0))
    _w_softcap = float(getattr(SCFG, "SPEED_SOFTCAP_PENALTY_WEIGHT", 0.0))
    # Sustained-power soft penalty (bugfix — see apply_sustained_power_caps'
    # docstring for why this must NOT be a shrunk v_max bound). Weight is
    # equivalent seconds of objective cost per second the rolling-average
    # draw (simulator.forward_sim.SustainedPowerTracker) spends over
    # car.p_max_sustained_w — same units/pattern as high_soc_penalty_s below.
    _w_sustained = float(getattr(SCFG, "SUSTAINED_POWER_PENALTY_WEIGHT", 2.0))

    def _speed_band_penalty_s(v_kmh: np.ndarray) -> float:
        if _w_comfort <= 0.0 and _w_softcap <= 0.0:
            return 0.0
        v = np.asarray(v_kmh, dtype=float)
        over_comfort = np.maximum(0.0, v - _v_comfort)
        over_softcap = np.maximum(0.0, v - _v_softcap)
        return float(_w_comfort * np.sum(over_comfort ** 2)
                     + _w_softcap * np.sum(over_softcap ** 2))

    def _objective(v_kmh: np.ndarray) -> float:
        res = evaluator(v_kmh)
        solar_penalty_s = _w * float(res.solar_underutil_j) / 3600.0
        high_soc_penalty_s = _w_high * float(getattr(res, "soc_over_safe_pct_s", 0.0))
        speed_penalty_s = _speed_band_penalty_s(v_kmh)
        sustained_penalty_s = _w_sustained * float(
            getattr(res, "sustained_power_over_budget_s", 0.0))
        return (float(res.total_time_s) + solar_penalty_s
                + high_soc_penalty_s + speed_penalty_s + sustained_penalty_s)
    return _objective

def _terminal_soc_constraint(evaluator: DayEvaluator,
                              alpha_next_day_pct: float) -> NonlinearConstraint:
    return NonlinearConstraint(
        lambda v: evaluator(v).final_soc_pct - alpha_next_day_pct,
        lb=0.0, ub=np.inf,
    )

def _intraday_soc_floor_constraint(evaluator: DayEvaluator,
                                    soc_min_pct: float) -> NonlinearConstraint:
    """Hard floor on SOC at EVERY substep of the day, not just the endpoint.

    BUGFIX: only the terminal SOC (_terminal_soc_constraint above, and
    trust_region's end_soc >= soc_min_pct check) was ever constrained.
    core.battery.Battery.apply_energy_wh deliberately clips the physical SOC
    ledger to [0, soc_max_pct] — NOT [soc_min_pct, soc_max_pct] — leaving the
    soc_min_pct feasibility check to "the optimizer's/checker's job" (its own
    docstring). But nothing was actually doing that check intraday, so a
    profile could plan a mid-day dip to single digits, or literally 0% SOC
    (confirmed in production dashboard traces — a day cruising at 68 km/h
    with the pack pinned at 0.00% for several minutes before recovering by
    dusk), and still be reported "feasible" because the END-of-day number
    looked fine. A real pack cannot sustain multi-kW motor draw at 0% SOC;
    this is a physical/safety violation regardless of how the day finishes.
    Uses the SAME soc_pct_trace already computed by forward_sim for the
    dashboard, so there's no extra simulation cost.
    """
    def _min_soc_margin(v: np.ndarray) -> float:
        trace = evaluator(v).soc_pct_trace
        if trace.size == 0:
            return 0.0  # no per-substep trace available -> nothing to check
        return float(np.min(trace)) - soc_min_pct
    return NonlinearConstraint(_min_soc_margin, lb=0.0, ub=np.inf)

def _time_cutoff_constraint(evaluator: DayEvaluator,
                             allowed_time_s: float) -> NonlinearConstraint:
    return NonlinearConstraint(
        lambda v: allowed_time_s - evaluator(v).total_time_s,
        lb=0.0, ub=np.inf,
    )


# ===============================================================================
# 4. Swappable global search
# ===============================================================================

class GlobalSearchResult(_t.NamedTuple):
    x: np.ndarray
    fun: float
    method: str

class GlobalSearchStrategy(_t.Protocol):
    def search(self, objective: _t.Callable[[np.ndarray], float], bounds: Bounds,
               constraints: list[NonlinearConstraint],
               seed: int | None = None,
               extra_seeds: list[np.ndarray] | None = None) -> GlobalSearchResult: ...

class DifferentialEvolutionSearch:
    def __init__(self, popsize: int = DE_POPSIZE, maxiter: int = DE_MAXITER,
                 strategy: str = "best1bin", mutation=(0.5, 1.0),
                 recombination: float = 0.7):
        self.popsize = popsize
        self.maxiter = maxiter
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination

    def search(self, objective, bounds, constraints, seed=None,
               extra_seeds: list[np.ndarray] | None = None) -> GlobalSearchResult:
        init = "latinhypercube"
        if extra_seeds:
            # Build an explicit initial population: scipy's DE accepts an
            # (M, N) array for `init` directly. Fill the rest with a Latin
            # hypercube sample and slot the targeted high-speed seeds in.
            lb, ub = np.asarray(bounds.lb), np.asarray(bounds.ub)
            dim = lb.size
            pop_size = max(self.popsize * dim, len(extra_seeds) + 1)
            rng = np.random.default_rng(seed)
            base = rng.uniform(lb, ub, size=(pop_size, dim))
            for i, s in enumerate(extra_seeds[:pop_size]):
                base[i] = np.clip(s, lb, ub)
            init = base
        result = differential_evolution(
            objective, bounds,
            strategy=self.strategy, popsize=self.popsize, maxiter=self.maxiter,
            mutation=self.mutation, recombination=self.recombination,
            constraints=tuple(constraints), polish=False, seed=seed, tol=1e-6,
            init=init,
        )
        return GlobalSearchResult(x=result.x, fun=result.fun, method="de")

class GeneticAlgorithmSearch:
    def __init__(self, population: int = SCFG.GA_POPULATION,
                 generations: int = SCFG.GA_GENERATIONS,
                 mutation_kmh: float = SCFG.GA_MUTATION_KMH,
                 elite_frac: float = 0.1, tournament_k: int = 3,
                 penalty_weight: float = 1e6):
        self.population = population
        self.generations = generations
        self.mutation_kmh = mutation_kmh
        self.elite_frac = elite_frac
        self.tournament_k = tournament_k
        self.penalty_weight = penalty_weight

    def _penalized_fitness(self, objective, constraints, x: np.ndarray) -> float:
        val = objective(x)
        for c in constraints:
            g = np.atleast_1d(c.fun(x))
            lb = np.atleast_1d(c.lb)
            ub = np.atleast_1d(c.ub)
            violation = np.maximum(lb - g, 0.0) + np.maximum(g - ub, 0.0)
            val += self.penalty_weight * float(np.sum(violation))
        return val

    @staticmethod
    def _tournament(pop: np.ndarray, fitness: np.ndarray, rng, k: int) -> np.ndarray:
        idx = rng.integers(0, len(pop), size=k)
        return pop[idx[np.argmin(fitness[idx])]]

    def search(self, objective, bounds, constraints, seed=None,
               extra_seeds: list[np.ndarray] | None = None) -> GlobalSearchResult:
        rng = np.random.default_rng(seed)
        lb, ub = np.asarray(bounds.lb), np.asarray(bounds.ub)
        dim = lb.size

        pop = rng.uniform(lb, ub, size=(self.population, dim))
        # SEED the population near the speed band where the optimum now lives.
        # The race objective is "drive as fast as feasible subject to the SOC
        # floor", but pure-random GA init almost never samples a good cruise on
        # a ~25-dim box, so the search used to converge to a mediocre ~55 km/h
        # profile. We now seed at ABSOLUTE target speeds around the comfort band
        # (updated 22/08: the hard bound is ~85, so the old "fractions of ub"
        # seeds landed at ~85 — needlessly high now that a soft penalty holds
        # the band). One seed still pushes the soft-cap so constrained days that
        # genuinely need the top end start with it available; SLSQP + the speed
        # penalty pull everything back into 60-70 unless a constraint forces up.
        _vc = float(getattr(SCFG, "CRUISE_COMFORT_KMH", 70.0))
        _vs = float(getattr(SCFG, "CRUISE_SOFT_CAP_KMH", 75.0))
        seed_speeds = (_vs, _vc, _vc - 5.0, _vc - 12.0)   # e.g. 75, 70, 65, 58
        n_seed = min(self.population, len(seed_speeds))
        for i in range(n_seed):
            pop[i] = np.clip(np.full(dim, seed_speeds[i]), lb, ub)

        # BUGFIX (issue 4 — "barely goes above 70/75 even on days with clear
        # slack"): the seeds above are all UNIFORM comfort-band speeds. With
        # the steep quadratic softcap penalty starting right at
        # CRUISE_SOFT_CAP_KMH and a small population/generation budget, SLSQP
        # gradient-descends locally from whichever seed it's handed and almost
        # never discovers that pushing into 75-85 pays off — that benefit is
        # non-convex (it only pays off if it lets you actually beat a cutoff
        # or bank a loop), so a local search needs to START near that basin to
        # find it at all. `extra_seeds` (built by solve() from the day's real
        # allowed_time_s / distance and its loop count) drops in targeted,
        # non-uniform candidates — e.g. "the constant speed that exactly uses
        # the day's remaining time budget" — instead of leaving the optimizer
        # to stumble onto them by chance. Slotted in right after the uniform
        # seeds, before the random fill, and never at the cost of losing the
        # existing comfort-band seeds.
        if extra_seeds:
            n_extra = min(self.population - n_seed, len(extra_seeds))
            for j in range(n_extra):
                pop[n_seed + j] = np.clip(
                    np.asarray(extra_seeds[j], dtype=float).reshape(dim), lb, ub)

        fitness = np.array([self._penalized_fitness(objective, constraints, ind)
                             for ind in pop])
        n_elite = max(1, int(self.elite_frac * self.population))

        for gen in tqdm(range(self.generations), desc="GA gens", leave=False):

            order = np.argsort(fitness)
            pop, fitness = pop[order], fitness[order]
            new_pop = [pop[i].copy() for i in range(n_elite)]
            while len(new_pop) < self.population:
                p1 = self._tournament(pop, fitness, rng, self.tournament_k)
                p2 = self._tournament(pop, fitness, rng, self.tournament_k)
                alpha = rng.uniform(0.0, 1.0, size=dim)
                child = alpha * p1 + (1.0 - alpha) * p2
                mutate = rng.random(dim) < (1.0 / dim)
                child = child + mutate * rng.normal(0.0, self.mutation_kmh, size=dim)
                new_pop.append(np.clip(child, lb, ub))
            pop = np.array(new_pop)
            fitness = np.array([self._penalized_fitness(objective, constraints, ind)
                                 for ind in pop])

        best_i = int(np.argmin(fitness))
        best_x = pop[best_i]
        return GlobalSearchResult(x=best_x, fun=objective(best_x), method="ga")

GLOBAL_SEARCH_REGISTRY: dict[str, type] = {
    "de": DifferentialEvolutionSearch,
    "ga": GeneticAlgorithmSearch,
}

def get_global_search(method: str, **kwargs) -> GlobalSearchStrategy:
    try:
        cls = GLOBAL_SEARCH_REGISTRY[method]
    except KeyError:
        raise KeyError(f"Unknown global_method={method!r}")
    return cls(**kwargs)


# ===============================================================================
# 5. Integer km/h projection
# ===============================================================================

def project_to_integer_kmh(evaluator: DayEvaluator, v_kmh: np.ndarray,
                            v_max_kmh: np.ndarray, v_min_kmh: float = 5.0,
                            constraints: _t.Sequence[NonlinearConstraint] = (),
                            objective: _t.Callable[[np.ndarray], float] | None = None,
                            ) -> np.ndarray:
    """Project SLSQP's continuous solution to integer km/h without changing
    the optimization objective.

    The previous implementation silently re-optimized the rounded profile for
    *maximum final SOC*, which partially undid the L2 objective change.
    """
    if objective is None:
        objective = _build_objective(evaluator)

    v_int = np.clip(np.round(v_kmh), v_min_kmh, np.floor(v_max_kmh))

    def _feasible(v: np.ndarray) -> bool:
        return all(np.all(np.atleast_1d(c.fun(v)) >= -1e-6) for c in constraints)

    if not _feasible(v_int):
        v_int = np.clip(np.asarray(v_kmh, dtype=float), v_min_kmh, np.floor(v_max_kmh))

    best = v_int.copy()
    best_obj = float(objective(best)) if _feasible(best) else float('inf')
    for i in range(len(best)):
        for step in (+1.0, -1.0):
            cand = best.copy()
            cand[i] = np.clip(cand[i] + step, v_min_kmh, np.floor(v_max_kmh[i]))
            if cand[i] == best[i] or not _feasible(cand):
                continue
            obj = float(objective(cand))
            if obj < best_obj:
                best, best_obj = cand, obj
    return best


# ===============================================================================
# 6. solve() — frozen API
# ===============================================================================

def solve(route: Route, car: CarState, solar_provider, wind_provider,
          day_index: int, start_soc_pct: float, alpha_next_day_pct: float,
          loops_committed, global_method: str = "ga", seed: int | None = None,
          dist_done_km: float = 0.0, elapsed_s: float = 0.0, cs_taken: bool = False,
          loop_geoms: dict | None = None,
          stage1_km: float | None = None,
          penalty_stoppage_s: float = 0.0,
          **kwargs):

    # ROOT-CAUSE FIX: splice committed loop reps into the real simulated
    # route (see section 1b above) instead of the old behaviour where
    # loops_committed only subtracted stop-time and never touched the
    # physics at all. loop_geoms is {loop_name: DataFrame} for this day,
    # threaded down from trust_region.py -> tier2.py. If not provided
    # (e.g. an old call site not yet updated), loops still get a flat
    # synthetic geometry via _splice_loops rather than silently costing
    # nothing — real distance/time/energy either way.
    # stage1_km (plan.stage1_km) disambiguates loop-first days (e.g. Day 3
    # Aryaman, stage1_km==0.0) from Day-6-style mistagged-single-file days —
    # see _splice_loops' docstring. None (old call sites) keeps the prior
    # Day-6-only behaviour.
    sim_route = route
    if loops_committed and route is not None:
        sim_route = _splice_loops(route, loop_geoms, loops_committed,
                                   stage1_km=stage1_km)

    rem_m = (sim_route.total_m - dist_done_km * 1000.0) if sim_route else 0.0
    n_segments = max(1, int(np.ceil(rem_m / CONTROL_SEGMENT_M)))
    
    seg_start_m = (dist_done_km * 1000.0) + np.arange(n_segments) * CONTROL_SEGMENT_M

    v_max_kmh = sim_route.v_max_ms_at(seg_start_m) * 3.6 if sim_route else np.full(n_segments, car.v_max_ms * 3.6)
    if sim_route:
        # loop_geoms/loops_committed passed through so repeated-lap control
        # segments get the periodicity-corrected loop-local turn cap (see
        # apply_turn_speed_caps docstring) instead of the globally-binned one.
        v_max_kmh = apply_turn_speed_caps(sim_route, v_max_kmh, seg_start_m,
                                          loop_geoms=loop_geoms,
                                          loops_committed=loops_committed)
        # NOTE: sustained power is deliberately NOT enforced by shrinking
        # v_max_kmh here (see apply_sustained_power_caps' docstring for why
        # that was a bug: it collapses the intended V_MAX_HARD_KMH=85 /
        # CRUISE_SOFT_CAP_KMH=75 two-tier scheme by silently pulling the
        # actual upper BOUND itself down to ~70 on any real climb). It's
        # enforced instead as a soft objective penalty on
        # sustained_power_over_budget_s in _build_objective below — same
        # pattern as the existing speed-band and high-SOC penalties.

    # Enforce the CAR's physical max speed. The route's own speed-limit column
    # can read up to ~120 km/h, which was leaking into the bounds and letting
    # the optimizer "drive" faster than the car can (the 112 km/h you saw).
    v_max_kmh = np.minimum(v_max_kmh, car.v_max_ms * 3.6)
    # HARD SPEED CEILING (directive 22/08): the car may briefly touch ~85 km/h
    # "only if needed", so the hard bound is V_MAX_HARD_KMH (was a hard 70). The
    # 60-70 AVERAGE and the ~75 normal ceiling are enforced SOFTLY, by the
    # convex speed penalty in the objective (see _build_objective) — not by this
    # bound — so the optimizer can spend into 75-85 when a loop/cutoff/SOC makes
    # it worth the penalty, but is otherwise held in-band.
    _v_hard_kmh = getattr(SCFG, "V_MAX_HARD_KMH", None)
    if _v_hard_kmh:
        v_max_kmh = np.minimum(v_max_kmh, float(_v_hard_kmh))
    v_max_kmh = np.maximum(v_max_kmh, 5.0)
    bounds = Bounds(lb=np.full(n_segments, 5.0), ub=v_max_kmh)

    t0_s = race_config.day_start_time_s(day_index) + elapsed_s
    
    evaluator = DayEvaluator(sim_route, car, solar_provider, wind_provider,
                              t0_s=t0_s,
                              start_soc_pct=start_soc_pct,
                              seg_start_m=seg_start_m,
                              cs_taken=cs_taken)

    objective = _build_objective(evaluator)

    n_loops = len(loops_committed) if loops_committed else 0
    # Tier 1 parity on the stop-time budget: the control stop, the unplanned
    # stop budget, and each loop turnaround are all parked time the car is NOT
    # driving — subtract them all from the allowed drive window exactly like
    # tier1.guess_baseline's t_stops_base / pre_attempt_stop_s. forward_sim
    # credits the parked solar for the same windows (see its stop-time
    # charging block), so time budget and energy credit stay symmetric.
    # NOTE: this is stop-time only — the loop's actual DRIVING time is now
    # simulated for real via the sim_route splice above, so there's no
    # double-counting between "stopped at the loop" and "driving the loop".
    allowed_time_s = (
        (race_config.day_finish_time_s(day_index) - race_config.day_start_time_s(day_index))
        - elapsed_s
        - (0.0 if cs_taken else race_config.CONTROL_STOP_DURATION_S)
        - race_config.UNPLANNED_STOP_BUDGET_S
        - n_loops * (race_config.LOOP_STOP_DURATION_S
                     + getattr(race_config, "LOOP_TURNAROUND_S", 0.0))
        # FEATURE B (strategist directive 21/08): a late finish on the PREVIOUS
        # day incurs the SR 2.22.6 time penalty, served stationary at the start
        # of THIS day. That penalty is not just a reporting artifact — it is
        # parked time this car cannot drive, so it must shrink today's real
        # driving window exactly like any other stoppage. Threaded in from
        # trust_region.extract_final_profiles as the prior day's realized
        # late-finish penalty (seconds). Zero on Day 1 and on any day whose
        # predecessor finished on time.
        - max(0.0, penalty_stoppage_s)
    )

    constraints = [
        _terminal_soc_constraint(evaluator, alpha_next_day_pct),
        _time_cutoff_constraint(evaluator, allowed_time_s),
        _intraday_soc_floor_constraint(evaluator, car.soc_min_pct),
    ]

    # ------------------------------------------------------------------
    # Targeted high-speed GA/DE seeds for "worth-it" opportunities (issue 4
    # fix): instead of relying on random/uniform-comfort-band seeds to
    # stumble onto the cases where pushing past CRUISE_SOFT_CAP_KMH actually
    # pays off, build candidates that start EXACTLY where that basin is:
    #   1) the constant speed that exactly spends the day's whole remaining
    #      time budget over its remaining distance — the natural "how fast do
    #      we NEED to go" anchor. On a spacious day this sits below comfort
    #      and does no harm (SLSQP + the speed penalty pull it back down); on
    #      a genuinely tight day it lands above CRUISE_SOFT_CAP_KMH and gives
    #      the local search a starting point already inside the basin it
    #      would otherwise never find.
    #   2) the same speed with a small margin added, in case hitting the
    #      cutoff exactly still fails the intraday-SOC/terminal-SOC
    #      constraints and a bit more speed is what actually closes the gap.
    #   3) on loop days specifically (n_loops > 0 — the loop stop-time budget
    #      already eats into allowed_time_s, so these days are structurally
    #      the tightest), a seed pinned at the hard soft-cap so the "burst
    #      speed, bank time, fit the loop" strategy is represented too.
    # ------------------------------------------------------------------
    extra_seeds: list[np.ndarray] = []
    rem_km = rem_m / 1000.0
    if allowed_time_s > 0.0 and rem_km > 0.0:
        target_avg_kmh = rem_km / (allowed_time_s / 3600.0)
        extra_seeds.append(np.clip(np.full(n_segments, target_avg_kmh),
                                    bounds.lb, bounds.ub))
        extra_seeds.append(np.clip(np.full(n_segments, target_avg_kmh + 5.0),
                                    bounds.lb, bounds.ub))
    if n_loops > 0:
        _burst_kmh = float(getattr(SCFG, "CRUISE_SOFT_CAP_KMH", 75.0))
        extra_seeds.append(np.clip(np.full(n_segments, _burst_kmh),
                                    bounds.lb, bounds.ub))

    if "warm_start_kmh" in kwargs and kwargs["warm_start_kmh"] is not None:
        warm_x = np.asarray(kwargs["warm_start_kmh"], dtype=float).reshape(-1)
        if len(warm_x) == n_segments:
            global_result = GlobalSearchResult(
                x=np.clip(warm_x, bounds.lb, bounds.ub),
                fun=objective(warm_x), method="warm")
        elif len(warm_x) >= 2 and n_segments >= 1:
            # Loop counts change the route length/control-vector dimension.
            # Resample the previous profile instead of discarding it and
            # launching another expensive GA.
            old_u = np.linspace(0.0, 1.0, len(warm_x))
            new_u = np.linspace(0.0, 1.0, n_segments)
            seed_x = np.interp(new_u, old_u, warm_x)
            seed_x = np.clip(seed_x, bounds.lb, bounds.ub)
            global_result = GlobalSearchResult(
                x=seed_x, fun=objective(seed_x), method="warm-resampled")
        else:
            global_search = get_global_search(global_method)
            global_result = global_search.search(
                objective, bounds, constraints, seed=seed, extra_seeds=extra_seeds)
    else:
        global_search = get_global_search(global_method)
        global_result = global_search.search(
            objective, bounds, constraints, seed=seed, extra_seeds=extra_seeds)

    _iter_count = [0]
    def _cb(xk):
        _iter_count[0] += 1
        logger.info(f"SLSQP iter {_iter_count[0]}/{SCFG.SLSQP_MAX_ITER}")


    slsqp_result = minimize(
        objective, x0=global_result.x, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options=dict(maxiter=SCFG.SLSQP_MAX_ITER, ftol=SCFG.SLSQP_FTOL),
    )

    v_final_kmh = project_to_integer_kmh(
        evaluator, slsqp_result.x, v_max_kmh, constraints=constraints,
        objective=objective)
    final_eval = evaluator(v_final_kmh)

    return dict(
        v_kmh=v_final_kmh,
        seg_start_m=seg_start_m,
        final_soc_pct=final_eval.final_soc_pct,
        total_time_s=final_eval.total_time_s,
        t_s=final_eval.t_s,
        x_m=final_eval.x_m,
        driver_swaps=final_eval.driver_swaps,
        global_method=global_result.method,
        trailered_km=getattr(final_eval, 'trailered_km', 0.0),
        trailered_substeps=getattr(final_eval, 'trailered_substeps', 0),
        driven_km=getattr(final_eval, 'driven_km', 0.0),
        motor_energy_wh=getattr(final_eval, 'motor_energy_wh', 0.0),
        solar_energy_wh=getattr(final_eval, 'solar_energy_wh', 0.0),
        solar_underutil_wh=getattr(final_eval, 'solar_underutil_j', 0.0) / 3600.0,
        battery_delta_wh=(float(final_eval.final_soc_pct) - float(start_soc_pct))
                         * car.battery_nominal_wh / 100.0,
        # Per-substep dashboard traces (aligned 1:1 with t_s / x_m). Continuous
        # SOC / velocity / solar / gradient curves vs distance for the Dashboard;
        # the coarse per-segment v_kmh above stays the driver card.
        soc_over_safe_pct_s=getattr(final_eval, 'soc_over_safe_pct_s', 0.0),
        soc_pct_trace=getattr(final_eval, 'soc_pct_trace', np.array([])),
        v_kmh_trace=getattr(final_eval, 'v_kmh_trace', np.array([])),
        solar_w_trace=getattr(final_eval, 'solar_w_trace', np.array([])),
        slope_pct_trace=getattr(final_eval, 'slope_pct_trace', np.array([])),
        # Motor power vs distance (workplan fix) + sustained-power-cap
        # exposure (see simulator.forward_sim.SustainedPowerTracker).
        motor_w_trace=getattr(final_eval, 'motor_w_trace', np.array([])),
        sustained_power_over_budget_s=getattr(
            final_eval, 'sustained_power_over_budget_s', 0.0),
        # Per-point seg_type, for splitting the day into stage1/loop/stage2.
        seg_type_trace=getattr(final_eval, 'seg_type_trace', np.array([])),
        diagnostics=dict(
            global_fun=global_result.fun,
            slsqp_fun=slsqp_result.fun,
            slsqp_success=slsqp_result.success,
        ),
    )