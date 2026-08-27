"""
optimizers/hierarchical/trust_region.py — coarse-to-fine driver loop.
python -m optimizers.trust_region --variant aryaman/prahlad
remove --variant for both
"""

from __future__ import annotations

# ── BLAS thread pinning (must run before numpy/scipy import) ──────────────
# scipy.optimize.minimize / differential_evolution call BLAS (LAPACK)
# routines that default to OMP's auto thread count. Under Tier 2's
# thread-per-day parallelism that oversubscribes cores and stalls every
# solve. Pin each process to one BLAS thread so worker threads actually
# parallelize instead of thrashing. Keep this BEFORE any numpy/scipy
# import in this module (and ideally in the entrypoint too).
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")
del _os, _v

import logging
import numpy as np
import dataclasses
import glob
import pandas as pd

import random as _random
from configs import race_config as rc
from configs.car_config import CarState
from core.options import DailyBreakdown as _DailyBreakdown
from optimizers import singleday
from .tier1 import _get_day_plan
from . import tier1, tier2, tier3, _threads
from .tier1 import _adjust_plan_for_today

logger = logging.getLogger(__name__)

# Trust-region convergence. The SOC trajectory drift oscillates in the single
# digits after the first pass (tier1 baseline -> tier3 reallocation is the big
# jump), so a 5% window never triggered and every run burned the full 4
# iterations (~1 h/variant EACH just from that). A 12% window converges in ~2
# passes on real data — the residual few-% SOC drift is well within planning
# tolerance — and the hard cap keeps the worst case bounded so two variants
# finish inside the 2 h budget.
MAX_ITERS = 2
CONVERGENCE_WINDOW_PCT = 15.0

def _alpha_floors_from_traj(s1_pct: np.ndarray, car: CarState, start_day: int,
                            overnight_gains: dict | None = None) -> dict:
    """Compute end-of-day SOC floors for Tier 2.

    Tier 2's job is to BUILD SURROGATES by sampling the full feasible space.
    The alpha passed to singleday.solve is only a minimum-SOC guard — it
    should be as lenient as possible so combos aren't rejected prematurely.

    Tier 1's uniform-speed DP produces end-of-day SOCs (e.g. 26.8%) that
    singleday.solve — with variable speeds, turn caps and tighter time
    budgets — often can't replicate (27977s vs 27000s allowed). Using those
    as alpha floors causes ALL combos to be marked infeasible, leaving
    surrogates empty for 7/8 days.

    Fix: use soc_min_pct for every day. Tier 3 handles inter-day SOC
    allocation using the populated surrogates.
    """
    n = len(s1_pct)
    floors = {}
    for d in range(start_day, n):
        floors[d] = car.soc_min_pct
    return floors

def replan(routes: list, base_car: CarState, solar_providers: dict, wind_providers: dict,
           cur_soc_pct: float, *, cur_day: int, dist_done_km: float = 0.0,
           elapsed_s: float = 0.0, cs_taken: bool = False,
           loops_done: dict | None = None, overrides: dict | None = None,
           kml_paths: dict | None = None, **kwargs) -> dict:
    
    car = dataclasses.replace(base_car, **(overrides or {}))
    return optimize(routes, car, solar_providers, wind_providers, cur_soc_pct,
                    start_day=cur_day, dist_done_km=dist_done_km,
                    elapsed_s=elapsed_s, cs_taken=cs_taken, 
                    loops_done=loops_done, kml_paths=kml_paths, **kwargs)

def _downsample_trace_by_distance(x_m, series: dict, stride_m: float) -> dict:
    """Downsample per-substep traces to ~one point per `stride_m` metres.

    forward_sim integrates on the ~100 m energy grid, so a full day is a few
    thousand substeps — more than a dashboard needs and heavy in JSON. Pick
    representative points spaced by distance (not index) so the curves stay
    faithful regardless of how speed varied. Always keeps the first and last
    point. Returns {name: list}. Empty in -> empty out.
    """
    import numpy as _np
    x = _np.asarray(x_m, dtype=float)
    n = x.size
    if n == 0:
        return {k: [] for k in (["distance_m"] + list(series.keys()))}
    if n == 1 or stride_m <= 0:
        keep = _np.arange(n)
    else:
        keep = [0]
        last_x = x[0]
        for i in range(1, n):
            if x[i] - last_x >= stride_m:
                keep.append(i)
                last_x = x[i]
        if keep[-1] != n - 1:
            keep.append(n - 1)
        keep = _np.asarray(keep, dtype=int)
    out = {"distance_m": [round(float(v), 1) for v in x[keep]]}
    for name, arr in series.items():
        a = _np.asarray(arr, dtype=float)
        if a.size != n:
            out[name] = []
            continue
        out[name] = [round(float(v), 2) for v in a[keep]]
    return out


def _coarse_stage(seg_type: str) -> str:
    """Map a raw route seg_type to a coarse reporting stage.

    'stage1'      -> 'stage1'
    'loop_<name>' / 'loop_synthetic' -> 'loop'
    'stage2'      -> 'stage2'
    anything else -> the raw string (single-file days keep their own tag).
    """
    s = str(seg_type)
    if s.startswith("stage1"):
        return "stage1"
    if s.startswith("loop"):
        return "loop"
    if s.startswith("stage2"):
        return "stage2"
    return s


def _day_mandatory_stop_s(n_loops: int, cs_taken: bool = False) -> float:
    """Mandatory PARKED time that lands inside a race day and pushes the arrival
    clock later, but which forward_sim credits as parked-solar only (it never
    advances its drive clock). This is what must be ADDED to drive time to get
    the true ETA / elapsed race time:

      * the 30-min control stop (SR 2.28.5), unless it's already been taken
        (mid-day replan with cs_taken=True);
      * a 5-min mandatory stop per loop attempt (SR 2.29.5), plus any loop
        turnaround, times the number of committed loop reps.

    The 8-min UNPLANNED_STOP_BUDGET_S reserve is deliberately EXCLUDED — it's a
    contingency the optimizer holds against the 17:00 window, not a scheduled
    stop, so the planned ETA shouldn't bake it in (it just becomes headroom).
    """
    control = 0.0 if cs_taken else float(rc.CONTROL_STOP_DURATION_S)
    per_loop = float(rc.LOOP_STOP_DURATION_S) + float(getattr(rc, "LOOP_TURNAROUND_S", 0.0))
    return control + max(0, int(n_loops)) * per_loop


def _stage_breakdown(res: dict, tow_speed_kmh: float,
                     control_stop_s: float = 0.0,
                     loop_stop_total_s: float = 0.0,
                     stride_m: float = 250.0,
                     n_loop_reps: int = 0) -> list:
    """Split one day's full-resolution traces into per-stage summaries.

    Partitions the day (stage1 / loop / stage2, in the order they occur along
    the route) using the per-point seg_type trace from forward_sim, and reports
    the SAME metrics per stage as the day-level report: distance driven, avg/
    min/max speed, SOC start->end, solar captured, trailered km, elapsed & ETA.
    Loop reps are aggregated into one 'loop' stage. Days missing a stage simply
    omit it — no crash. Returns [] when the seg_type trace is unavailable (e.g.
    a synthetic-geometry day), so the caller can fall back to day-level only.
    """
    x = np.asarray(res.get("x_m", []), dtype=float)
    seg = np.asarray(res.get("seg_type_trace", []))
    if x.size == 0 or seg.size != x.size:
        return []
    soc = np.asarray(res.get("soc_pct_trace", []), dtype=float)
    v = np.asarray(res.get("v_kmh_trace", []), dtype=float)
    solar = np.asarray(res.get("solar_w_trace", []), dtype=float)
    t = np.asarray(res.get("t_s", []), dtype=float)
    have_soc = soc.size == x.size
    have_v = v.size == x.size
    have_solar = solar.size == x.size
    have_t = t.size == x.size

    coarse = np.array([_coarse_stage(s) for s in seg])
    # Preserve route order: the stage of the first point, then each new stage as
    # it first appears.
    order = []
    for c in coarse:
        if c not in order:
            order.append(c)

    stages = []
    # Running mandatory-stop offset so each stage's ETA is on the SAME true race
    # clock as the day-level ETA (drive time alone understates arrival — see
    # _day_mandatory_stop_s). The control stop is attributed to the first stage;
    # the loop stops to the loop stage. The final stage's ETA then equals the
    # corrected day ETA.
    _stop_accum_s = 0.0
    for name in order:
        idx = np.where(coarse == name)[0]
        if idx.size == 0:
            continue
        # Distance in this stage = sum of forward dx over points tagged here.
        # dx[i] is the step LEAVING point i; assign it to point i's stage.
        i0, i1 = idx[0], idx[-1]
        dx = np.diff(x)
        stage_mask = coarse[:-1] == name
        dist_m = float(np.sum(dx[stage_mask])) if dx.size else 0.0
        # Trailered points: on the tow (speed == tow speed AND no solar capture).
        trailered_km = 0.0
        driven_v = None
        if have_v:
            is_tow = np.isclose(v[:-1], tow_speed_kmh) & (
                (solar[:-1] == 0.0) if have_solar else True)
            trailered_km = float(np.sum(dx[stage_mask & is_tow])) / 1000.0 if dx.size else 0.0
            # Speed stats over the DRIVEN (non-tow) points in this stage.
            drv = idx[~np.isclose(v[idx], tow_speed_kmh)]
            driven_v = v[drv] if drv.size else v[idx]
        entry = {
            "stage": name,
            "distance_km": round(dist_m / 1000.0, 1),
            "trailered_km": round(trailered_km, 1),
            # Number of loop reps in this stage (only meaningful for 'loop').
            "n_loops": (int(n_loop_reps) if name == "loop" else 0),
        }
        if have_soc:
            entry["soc_start_pct"] = round(float(soc[i0]), 1)
            entry["soc_end_pct"] = round(float(soc[i1]), 1)
        if have_v and driven_v is not None and driven_v.size:
            entry["speed_avg_kmh"] = round(float(np.mean(driven_v)), 1)
            entry["speed_min_kmh"] = round(float(np.min(driven_v)), 1)
            entry["speed_max_kmh"] = round(float(np.max(driven_v)), 1)
        if have_solar:
            # Integrate solar power over this stage's substep dts (Wh).
            if have_t and t.size == x.size:
                dt = np.diff(t)
                wh = float(np.sum(solar[:-1][stage_mask] * dt[stage_mask])) / 3600.0
            else:
                wh = 0.0
            entry["solar_wh"] = round(wh, 0)
        if have_t:
            # Mandatory stops attributed to this stage (control → first stage,
            # loop stops → loop stage), added to the running clock so ETAs match
            # the day total.
            this_stop_s = 0.0
            if name == order[0]:
                this_stop_s += control_stop_s
            if name == "loop":
                this_stop_s += loop_stop_total_s
            _stop_accum_s += this_stop_s
            entry["stop_min"] = round(this_stop_s / 60.0, 0)
            entry["elapsed_s"] = round(float(t[i1] - t[i0]) + this_stop_s, 0)
            entry["eta"] = _clock_hhmm(float(t[i1]) + _stop_accum_s)

        # ── Per-stage plotting trace ────────────────────────────────────────
        # The dashboard plots each stage on its own, so every stage carries its
        # own downsampled curves (distance / velocity / solar / SOC / slope),
        # distance-indexed and reset to 0 at the stage start so stage1, loop and
        # stage2 can be drawn side by side. ~one point per stride_m metres.
        sl = np.asarray(res.get("slope_pct_trace", []), dtype=float)
        have_sl = sl.size == x.size
        xs = x[idx]
        keep = [0]
        _lastx = xs[0]
        for _j in range(1, xs.size):
            if xs[_j] - _lastx >= stride_m:
                keep.append(_j); _lastx = xs[_j]
        if keep[-1] != xs.size - 1:
            keep.append(xs.size - 1)
        gi = idx[np.asarray(keep, dtype=int)]      # global indices kept
        x0 = float(x[idx[0]])
        entry["trace"] = {
            "distance_km": [round((float(x[k]) - x0) / 1000.0, 3) for k in gi],
            "velocity_kmh": ([round(float(v[k]), 1) for k in gi] if have_v else []),
            "solar_w": ([round(float(solar[k]), 1) for k in gi] if have_solar else []),
            "soc_pct": ([round(float(soc[k]), 2) for k in gi] if have_soc else []),
            "slope_pct": ([round(float(sl[k]), 2) for k in gi] if have_sl else []),
        }
        stages.append(entry)
    return stages


def _clock_hhmm(abs_s: float) -> str:
    h = int(abs_s // 3600); m = int((abs_s % 3600) // 60)
    return f"{h:02d}:{m:02d}"


def _trailered_km_by_day(routes):
    """Return trailered km from the exact mask consumed by forward_sim."""
    out = {}
    for d, route in enumerate(routes or []):
        if route is None or not hasattr(route, "df") or len(route.df) < 2:
            out[d] = 0.0
            continue
        dist = route.df["distance_m"].to_numpy(dtype=float)
        mask = route.df["red_flag_trailer"].to_numpy(dtype=bool)
        out[d] = float(np.sum(np.diff(dist)[mask[:-1]])) / 1000.0
    return out


def optimize(routes: list, car: CarState, solar_providers: dict, wind_providers: dict,
             start_soc_pct: float = 100.0, *, start_day: int = 0,
             dist_done_km: float = 0.0, elapsed_s: float = 0.0,
                          cs_taken: bool = False, loops_done: dict | None = None,
             kml_paths: dict | None = None,
             loop_geoms_by_day: dict | None = None,
             plans_override: list | None = None, parallel: bool = True,
             n_workers: int | None = None, global_method: str = "ga",
             seed: int | None = None, max_iters: int = MAX_ITERS,
             window_pct: float = CONVERGENCE_WINDOW_PCT,
             tier1_baseline: dict | None = None) -> dict:

        # Audit fix: normalize the worker count exactly once, in one place.
    # tier2.sample_all_days re-clamps through the same worker_cap, so an
    # explicit n_workers is honored end-to-end (previously it was passed to
    # sample_all_days and silently swallowed by its **kwargs, so every run
    # used a hardcoded min(n_days, os.cpu_count()) instead of the caller's
    # parallel= setting).
    n_workers = _threads.worker_cap(n_workers)

    if tier1_baseline is not None:
        base = tier1_baseline
        logger.info("Using pre-computed Tier 1 baseline (skipping recomputation)")
    else:
        base = tier1.guess_baseline(routes, car, solar_providers, wind_providers,
                                    start_soc_pct, start_day=start_day,
                                    dist_done_km=dist_done_km, elapsed_s=elapsed_s,
                                    cs_taken=cs_taken, loops_done=loops_done,
                                    kml_paths=kml_paths,
                                    plans_override=plans_override)
    plans = plans_override if plans_override is not None else base["day_plans"]
    s_center = base["s0_pct"].copy()
    if not base["feasible"]:
        logger.warning("Tier 1 baseline infeasible from start_soc=%.1f%%", start_soc_pct)
        # Fill NaN entries so Tier 2 gets usable start-SOC guesses
        nan_mask = ~np.isfinite(s_center)
        if nan_mask.any():
            first_nan = int(np.argmax(nan_mask))
            last_known = float(s_center[first_nan - 1]) if first_nan > 0 else start_soc_pct
            n_nan = int(nan_mask.sum())
            target = max(car.soc_min_pct + 10.0, 50.0)
            fill = np.linspace(last_known, target, n_nan + 2)[1:-1]
            s_center[nan_mask] = fill
            logger.info("Filled %d NaN SOC entries: %s",
                        n_nan, [round(float(x), 1) for x in s_center])

    history = []
    result = None
    last_feasible = None

    for it in range(max_iters):
        # Alpha floors = soc_min_pct for all days.  Tier 2 samples the full
        # feasible space; Tier 3 handles multi-day SOC allocation.
        alpha_floors = _alpha_floors_from_traj(
            np.append(s_center, car.soc_min_pct)[: len(plans) + 1], car, start_day)

        logger.info("Tier2 it=%d alpha_floors (end-of-day targets): %s",
                    it, {d+1: f"{v:.1f}%" for d, v in alpha_floors.items()})
        logger.info("Tier2 it=%d s_center (start-of-day): %s",
                    it, [f"D{d+1}={v:.1f}%" for d, v in enumerate(s_center) if np.isfinite(v)])

        per_day = tier2.sample_all_days(
            routes, car, solar_providers, wind_providers, s_center, plans,
            alpha_floors, start_day=start_day, dist_done_km=dist_done_km,
            elapsed_s=elapsed_s, cs_taken=cs_taken, loops_done=loops_done,
            loop_geoms_by_day=loop_geoms_by_day,
            parallel=parallel, n_workers=n_workers, global_method=global_method, seed=seed)

        # Tier 2 surrogate diagnostic + Tier 1 fallback for empty days
        for d in range(start_day, len(plans)):
            dd = per_day.get(d)
            if dd is None:
                logger.warning("Tier2 Day %d: no data returned at all!", d + 1)
                dd = {}
            surr = dd.get("surrogates", {})
            n_keys = len(surr) if isinstance(surr, dict) else 0
            if n_keys > 0:
                logger.info("Tier2 Day %d: %d surrogate(s) populated", d + 1, n_keys)
            else:
                # ── Tier 1 fallback surrogate ──────────────────────────
                # singleday.solve failed for every combo at all offsets.
                # Build a synthetic linear surrogate from Tier 1's coarse
                # energy model so Tier 3 can still allocate this day.
                logger.warning("Tier2 Day %d: EMPTY — building Tier 1 fallback surrogate", d + 1)
                s0 = float(s_center[d])
                # Estimate end-of-day SOC from Tier 1 trajectory
                if d + 1 < len(s_center) and np.isfinite(s_center[d + 1]):
                    sp = solar_providers.get(d)
                    gain = tier1.overnight_soc_gain(car, sp, d) if sp else 0.0
                    end_est = float(s_center[d + 1]) - gain
                else:
                    end_est = car.soc_min_pct
                end_est = max(end_est, car.soc_min_pct)
                # Linear surrogate:  predict(s) = end_est + 1.0*(s - s0)
                reps = (0,) * len(plans[d].loops)
                fb = tier2.LinearSurrogate(
                    a=float(end_est), b=1.0, s0=s0, loop_km=0.0, reps=reps,
                    soc_lo=car.soc_min_pct, soc_hi=car.soc_max_pct)
                per_day[d] = {"surrogates": {reps: fb}, "s0_pct": s0,
                              "n_l2_solves": 0}
                logger.info("Tier2 Day %d: fallback a=%.1f b=%.2f s0=%.1f",
                            d + 1, end_est, 1.0, s0)

        trailered_km_by_day = _trailered_km_by_day(routes)
        if any(km > 0.0 for km in trailered_km_by_day.values()):
            logger.info("Tier3 credited trailered km by day: %s",
                        {d + 1: round(km, 1) for d, km in trailered_km_by_day.items() if km > 0.0})
        result = tier3.allocate(
            car, solar_providers, per_day, plans, start_soc_pct,
            start_day=start_day, trailered_km_by_day=trailered_km_by_day)
        s_refined = result["s1_pct"]

        drift = np.nanmax(np.abs(s_refined[start_day:] - s_center[start_day:]))
        history.append(dict(iteration=it, drift_pct=float(drift),
                            total_km=result["total_distance_km"], feasible=result["feasible"]))
        logger.info("trust-region it=%d drift=%.2f%% total=%.1f km feasible=%s",
                    it, drift, result["total_distance_km"], result["feasible"])

        # ROBUSTNESS: never ship an infeasible allocation when a feasible one
        # was already found. The trust-region trajectory can oscillate into an
        # infeasible SOC chain on a later pass (the refined s_center overshoots
        # somewhere); the old code broke and returned THAT infeasible result,
        # discarding a perfectly good earlier feasible plan. Keep the last
        # feasible result and fall back to it.
        if result["feasible"]:
            last_feasible = result
            if drift <= window_pct:
                return _package(result, plans, car, converged=True,
                                iterations=it + 1, history=history, start_day=start_day)
            s_center = s_refined.copy()
        else:
            logger.warning(
                "trust-region it=%d produced an INFEASIBLE allocation "
                "(total=%.1f km) — stopping and keeping the last feasible "
                "result.", it, result["total_distance_km"])
            break

    final = last_feasible if last_feasible is not None else result
    return _package(final, plans, car, converged=False,
                    iterations=len(history), history=history, start_day=start_day)

def _package(result: dict, plans: list, car: CarState, *, converged: bool,
             iterations: int, history: list, start_day: int) -> dict:
    if result is None:
        return dict(loop_plan={}, s_start_pct=None, total_distance_km=0.0,
                    converged=False, iterations=iterations, feasible=False,
                    alpha_day_pct={}, history=history, start_day_index=start_day)
    s1 = result["s1_pct"]
    alpha_day = {d: float(s1[d]) for d in range(start_day, len(plans)) if np.isfinite(s1[d])}
    return dict(
        loop_plan=result["loop_plan"],
        s_start_pct=s1,
        total_distance_km=result["total_distance_km"],
        converged=converged,
        iterations=iterations,
        feasible=result["feasible"],
        alpha_day_pct=alpha_day,
        history=history,
        start_day_index=start_day
    )

# ===========================================================================
# 1. Final Velocity Extraction (Post-Convergence)
# ===========================================================================

def extract_final_profiles(routes: list, base_car: CarState, solar_providers: dict,
                           wind_providers: dict, optimize_result: dict,
                           overrides: dict | None = None,
                           loop_geoms_by_day: dict | None = None,
                           plans_override: list | None = None,
                           breakdown_enabled: bool = False,
                           breakdown_seed: int | None = None) -> dict:
                           
    if not optimize_result.get("feasible"):
        logger.error("Cannot extract profiles from an infeasible result.")
        return {}

    car = dataclasses.replace(base_car, **(overrides or {}))
    final_race_plan = {}
    
    start_socs = optimize_result["s_start_pct"]
    loop_plan = optimize_result["loop_plan"]
    alpha_floors = optimize_result["alpha_day_pct"]
    start_day = optimize_result.get("start_day_index", 0)

    # PHYSICALLY-CONSISTENT SOC CHAIN. The old code simulated each day starting
    # from Tier 3's planned s1[d], which is a surrogate estimate — so the
    # reported day d+1 start (Tier 3) didn't match day d's ACTUAL simulated end,
    # making the SOC look like it dropped overnight. Here we chain forward: each
    # day starts at the PREVIOUS day's real end SOC plus the (safety-gated)
    # morning charge, so the trajectory the report shows actually reconciles and
    # the morning charge is visibly additive.
    cur_soc = (float(start_socs[start_day])
               if np.isfinite(start_socs[start_day]) else 100.0)

    # FEATURE B: the previous day's late-finish penalty (seconds) is served
    # stationary at the START of the current day. It both extends the morning
    # charge (already done via overnight_soc_gain extra_charge_s) AND eats into
    # the current day's legal driving window — so it is threaded forward into
    # singleday.solve as penalty_stoppage_s and pushes the finish clock later.
    # Zero for the first simulated day (nothing finished late before it).
    carryover_penalty_s = 0.0

    for d in range(start_day, len(routes)):
        s_start = cur_soc
        d_loops = loop_plan.get(d, {})

        solar_provider = solar_providers.get(d)
        wind_provider = wind_providers.get(d)

        # Terminal-SOC floor for day d = what day d+1 needs to START, minus the
        # overnight charge that lands before it — NOT day d's own start SOC.
        # The old code used alpha_floors[d] == s1[d] (day d's *start*), which
        # forced every day to end no lower than it began. That made the
        # optimizer drive slow and hoard charge to chase an unreachable
        # end-SOC (Day 1 literally can't end at its 100% start), which is the
        # root of the "50 km/h, never spends the battery" behaviour. Letting
        # day d spend down to exactly what tomorrow needs frees it to drive
        # faster and bank distance.
        if d + 1 < len(start_socs) and np.isfinite(start_socs[d + 1]):
            gain_next = tier1.overnight_soc_gain(car, solar_provider, d)
            alpha_next = max(car.soc_min_pct,
                             float(start_socs[d + 1]) - gain_next)
        else:
            alpha_next = car.soc_min_pct
        
        if plans_override is not None:
            if d >= len(plans_override):
                raise ValueError(f"plans_override has no plan for day index {d}")
            nom_plan = plans_override[d]
        else:
            nom_plan = _get_day_plan(d)
        loops_committed = []
        for name, km in nom_plan.loops:
            count = d_loops.get(name, 0)
            loops_committed.extend([(name, km)] * count)
            
        logger.info(f"Extracting exact profile for Day {d}...")
        
                        # Real per-loop geometry for this day so committed reps are actually
        # simulated (loaded once in __main__ via _load_loop_geometries).
        loop_geoms = (loop_geoms_by_day or {}).get(d)

        route_d = routes[d] if routes else None
        res = singleday.solve(
            route=route_d,
            car=car,
            solar_provider=solar_provider,
            wind_provider=wind_provider,
            day_index=d,
            start_soc_pct=s_start,
            alpha_next_day_pct=alpha_next,
            loops_committed=loops_committed,
            loop_geoms=loop_geoms,
            penalty_stoppage_s=carryover_penalty_s,
        )

        # ── HARD finish-feasibility backstop ─────────────────────────────
        # The Tier-2 surrogate that Tier 3 allocated from is sampled at a few
        # SOC offsets; the real chained start SOC here can differ, so the
        # allocated loop count can overshoot the day's absolute cutoff (this
        # is exactly how "8 loops, ETA 18:32" shipped). Never publish a plan
        # that finishes past the cutoff: drop the last-committed loop rep and
        # re-solve until it fits (or no loops remain). Each drop is one extra
        # solve, on the single final-extract pass only.
        # Enforce the SOFT finish limit (strategist directive 23/08): a normal
        # day must not be planned past 17:10, and Day 8 not past its hard 15:00
        # timed finish. 17:30 (day_finish_cutoff_s) remains the absolute rail,
        # but the backstop now drops loops to the tighter soft limit so the plan
        # never lands in the "dire only" 17:10-17:30 band. (In practice singleday
        # already targets the 17:00/15:00 on-time window, so days finish well
        # inside this; the tighter rail is a guarantee, not a distance cut.)
        cutoff_s = rc.day_finish_soft_limit_s(d)
        hard_cutoff_s = rc.day_finish_cutoff_s(d)
        day_start_s = rc.day_start_time_s(d)
        # Control stop always applies in a full-day strategy extract (it hasn't
        # been "taken" yet for a fresh day).
        _cs_taken_day = False
        # The TRUE finish clock is day_start + prior-day penalty hold (served at
        # the top of today) + drive time + the day's MANDATORY STOPS (control
        # stop + per-loop stops). forward_sim's total_time_s is drive-only — it
        # credits stop-solar but never advances its clock — so the stops must be
        # added here or the finish is understated by ~30 min + 5 min/loop.
        def _true_finish_s(_res, _loops):
            return (day_start_s + carryover_penalty_s
                    + float(_res.get("total_time_s", 0.0))
                    + _day_mandatory_stop_s(len(_loops), _cs_taken_day))
        _dropped = 0
        while loops_committed and _true_finish_s(res, loops_committed) > cutoff_s:
            loops_committed = loops_committed[:-1]
            _dropped += 1
            res = singleday.solve(
                route=route_d, car=car, solar_provider=solar_provider,
                wind_provider=wind_provider, day_index=d, start_soc_pct=s_start,
                alpha_next_day_pct=alpha_next, loops_committed=loops_committed,
                loop_geoms=loop_geoms, penalty_stoppage_s=carryover_penalty_s)
        if _dropped:
            finish_clk = _true_finish_s(res, loops_committed)
            logger.warning(
                "Day %d: dropped %d loop rep(s) to meet the %02d:%02d cutoff — "
                "final finish %02d:%02d with %d loop rep(s).",
                d + 1, _dropped, int(cutoff_s // 3600), int((cutoff_s % 3600) // 60),
                int(finish_clk // 3600), int((finish_clk % 3600) // 60),
                len(loops_committed))

        # Continuous distance-indexed dashboard trace (downsampled). The coarse
        # per-segment velocity_profile_kmh stays the driver card; this is the
        # smooth SOC/velocity/solar/gradient-vs-distance curve for the dashboard.
        from configs import solver_config as _sc
        dashboard_trace = _downsample_trace_by_distance(
            res.get("x_m", []),
            {
                "v_kmh": res.get("v_kmh_trace", []),
                "soc_pct": res.get("soc_pct_trace", []),
                "solar_w": res.get("solar_w_trace", []),
                "slope_pct": res.get("slope_pct_trace", []),
                "time_s": res.get("t_s", []),
            },
            getattr(_sc, "OUTPUT_TRACE_STRIDE_M", 250.0),
        )

        # Per-stage breakdown (stage1 / loop / stage2) from the full-resolution
        # traces — same metrics as the day, split by route stage. [] if the
        # seg_type trace is unavailable (synthetic day), so callers fall back to
        # day-level reporting.
        # Mandatory stop times for THIS day's final loop count (control stop +
        # per-loop stops) — added to drive time to get the true race clock.
        _n_loops_final = len(loops_committed)
        _control_stop_s = 0.0 if _cs_taken_day else float(rc.CONTROL_STOP_DURATION_S)
        _loop_stop_total_s = _n_loops_final * (
            float(rc.LOOP_STOP_DURATION_S) + float(getattr(rc, "LOOP_TURNAROUND_S", 0.0)))
        _stop_time_s = _control_stop_s + _loop_stop_total_s

        stage_breakdown = _stage_breakdown(
            res, getattr(_sc, "TRAILER_TOW_SPEED_KMH", 80.0),
            control_stop_s=_control_stop_s, loop_stop_total_s=_loop_stop_total_s,
            stride_m=getattr(_sc, "OUTPUT_TRACE_STRIDE_M", 250.0),
            n_loop_reps=_n_loops_final)

        # ── Advance the physical SOC chain to the next day ──────────────
        actual_end_soc = float(res.get("final_soc_pct", cur_soc))

        # FEATURE 2 (breakdown scenario, opt-in via --breakdown): exactly ONE
        # breakdown per day — a stationary stop of a sampled duration, NO charging
        # during it (pure lost time). Duration is drawn from a 0..1 h PDF scaled
        # by the day's average motor power (harder driving -> longer downtime;
        # see core.options.DailyBreakdown). It pushes the finish later and can
        # trigger the normal late penalty (Feature B) — the realistic what-if.
        breakdown_s = 0.0
        if breakdown_enabled:
            _drive_h = max(1e-6, float(res.get("total_time_s", 0.0)) / 3600.0)
            _avg_power_w = float(res.get("motor_energy_wh", 0.0)) / _drive_h
            _p_ref_w = float(getattr(car, "p_max_continuous_w", 3000.0))
            _bd = _DailyBreakdown(
                max_seconds=float(getattr(_sc, "BREAKDOWN_MAX_SECONDS", 3600.0)))
            _seed = (breakdown_seed if breakdown_seed is not None
                     else int(getattr(_sc, "BREAKDOWN_SEED", 20260823)))
            breakdown_s = _bd.sample_seconds(
                _avg_power_w, _p_ref_w, _random.Random(_seed + d))

        # TRUE absolute finish clock = day_start + prior-day penalty hold (served
        # at the top of today) + drive time + today's mandatory stops (+ breakdown
        # if the scenario is on). The stops are real elapsed race time that
        # forward_sim's drive-only total_time_s omits, so they must be added here.
        finish_abs_s = (day_start_s + carryover_penalty_s
                        + float(res.get("total_time_s", 0.0))
                        + _stop_time_s + breakdown_s)
        # Late-finish penalty (SR 2.22.6/7): minutes served stationary at the
        # NEXT day's control stop. The car captures solar during that hold, so
        # it buys back morning-charge time (06:30 -> 08:00 + penalty).
        on_time_s = rc.day_finish_time_s(d)
        minutes_late = max(0.0, (finish_abs_s - on_time_s) / 60.0)
        late_penalty_min = rc.late_finish_penalty_min(minutes_late)

        # FEATURE 1 (early-finish charging): if the day finishes before the 17:00
        # close, the panel keeps charging from the finish moment until 17:00 and
        # that energy banks into tomorrow. On by default; skipped on Day 8 and
        # when a breakdown/late finish already ran the day to/past the close.
        evening_gain_pct = 0.0
        if getattr(_sc, "EVENING_CHARGE_ENABLED", True):
            evening_gain_pct = tier1.evening_soc_gain(
                car, solar_provider, d, finish_abs_s, end_soc_pct=actual_end_soc)
        end_soc_after_evening = min(actual_end_soc + evening_gain_pct, car.soc_max_pct)

        # Morning charge onto tomorrow, gated by the SOC ENTERING THE NIGHT (end
        # of day + evening charge) so an evening top-up correctly suppresses the
        # morning charge, and extended by the late-penalty hold time.
        morning_gain_pct = tier1.overnight_soc_gain(
            car, solar_provider, d,
            prev_end_soc_pct=end_soc_after_evening,
            extra_charge_s=late_penalty_min * 60.0)
        next_start_soc = min(end_soc_after_evening + morning_gain_pct, car.soc_max_pct)

        final_race_plan[d] = {
            "start_soc_pct": s_start,
            "loops_committed": loops_committed,
            "end_soc_pct": res.get("final_soc_pct"),
            # FEATURE 1: end-of-day (finish->17:00) charge banked into tomorrow,
            # and the resulting SOC entering the night.
            "evening_charge_pct": round(evening_gain_pct, 2),
            "end_soc_after_evening_pct": round(end_soc_after_evening, 2),
            # FEATURE 2: this day's breakdown downtime (minutes), 0 unless the
            # --breakdown scenario is enabled.
            "breakdown_min": int(round(breakdown_s / 60.0)),
            "morning_charge_pct": round(morning_gain_pct, 2),
            "late_penalty_min": int(late_penalty_min),
            # FEATURE B: minutes of penalty inherited from the PREVIOUS day and
            # actually charged against today's driving window (0 on the first
            # day). Distinct from late_penalty_min, which is what THIS day hands
            # to tomorrow. Surfaced so the report can show the propagation.
            "inherited_penalty_min": int(round(carryover_penalty_s / 60.0)),
            "next_start_soc_pct": round(next_start_soc, 2),
            "velocity_profile_kmh": res.get("v_kmh"),
            "time_array_s": res.get("t_s"),
            "distance_array_m": res.get("x_m"),
            "total_time_s": res.get("total_time_s"),  # actual drive duration (not absolute)
            # Mandatory in-day parked time (control stop + per-loop stops) that
            # forward_sim omits from total_time_s. Added to drive time to get the
            # true ETA. Kept as explicit fields so the report/JSON can show the
            # drive-vs-stops split and the true finish never drifts from them.
            "stop_time_s": round(_stop_time_s, 0),
            "control_stop_s": round(_control_stop_s, 0),
            "loop_stop_s": round(_loop_stop_total_s, 0),
            "finish_abs_s": round(finish_abs_s, 0),
            "eta": _clock_hhmm(finish_abs_s),
            "trailered_km": res.get("trailered_km", 0.0),
            "trailered_substeps": res.get("trailered_substeps", 0),
            "driven_km": res.get("driven_km", 0.0),
            "motor_energy_wh": res.get("motor_energy_wh", 0.0),
            "solar_energy_wh": res.get("solar_energy_wh", 0.0),
            # Previously never forwarded -> the report defaulted it to 0.0 on
            # every day even when solar was genuinely clipped at the SOC ceiling.
            "solar_underutil_wh": res.get("solar_underutil_wh", 0.0),
            "dashboard_trace": dashboard_trace,
            "stages": stage_breakdown,
        }

        cur_soc = next_start_soc
        # Hand this day's late-finish penalty forward: it will be served
        # stationary at the top of tomorrow, shrinking tomorrow's drive window.
        carryover_penalty_s = late_penalty_min * 60.0

    return final_race_plan

# ===========================================================================
# 2. Fast Intra-Day Replan (Model Predictive Control / L2 Only)
# ===========================================================================

def fast_replan_today(route, base_car: CarState, solar_providers: dict, wind_providers: dict,
                      cur_soc_pct: float, target_end_soc_pct: float,
                      planned_loops: dict, cur_day: int, dist_done_km: float,
                      elapsed_s: float,              cs_taken: bool = False, loops_done: dict | None = None, 
                      loop_geoms: dict | None = None,
                      overrides: dict | None = None, **kwargs) -> dict:
                      
    car = dataclasses.replace(base_car, **(overrides or {}))
    solar_provider = solar_providers.get(cur_day)
    wind_provider = wind_providers.get(cur_day)
    
    nom_plan = _get_day_plan(cur_day)
    adjusted_plan = _adjust_plan_for_today(nom_plan, dist_done_km, loops_done)
    
    loops_done_dict = loops_done or {}
    loops_committed = []
    
    for name, km in nom_plan.loops:
        total_planned = planned_loops.get(name, 0)
        already_done = loops_done_dict.get(name, 0)
        remaining = max(0, total_planned - already_done)
        loops_committed.extend([(name, km)] * remaining)
        
    logger.info(f"Fast Replan Day {cur_day}: {dist_done_km:.1f}km done. "
                f"Targeting end SOC {target_end_soc_pct:.1f}%")

    res = singleday.solve(
        route=route,
        car=car,
        solar_provider=solar_provider,
        wind_provider=wind_provider,
        day_index=cur_day,
        start_soc_pct=cur_soc_pct,
        alpha_next_day_pct=target_end_soc_pct,
        loops_committed=loops_committed,
                dist_done_km=dist_done_km,
        elapsed_s=elapsed_s,
        cs_taken=cs_taken,
        loop_geoms=loop_geoms,
        **kwargs
    )
    
    if res.get("final_soc_pct", -float('inf')) < car.soc_min_pct:
        logger.warning("Fast Replan Failed: Cannot hit target SOC. Macro-Replan required.")
        return {"feasible": False}

    return {
        "feasible": True,
        "velocity_profile_kmh": res.get("v_kmh"),
        "predicted_end_soc_pct": res.get("final_soc_pct")
    }


# ===========================================================================
# 2b. INTRA-DAY resolve (single-day optimizer only) — the MPC-style re-plan
# ===========================================================================
#
# Team discussion (23/08, Diyaansh + Hafiz): the macro `resolve_from_actuals`
# is the night-time, complete-replan-from-tomorrow tool and does NOT need the
# time of day. The OTHER resolve is the intra-day one: "during the day, if we
# drift off too much, resolve for just that day alone" using ONLY the single-day
# optimizer — so it DOES take the time of day (how far into the day we are). This
# is that second tool: it re-plans the REMAINDER of the CURRENT day from where
# the car actually is (distance done, time of day, SOC now, loops already done),
# leaving every other day untouched. Cheap (one singleday.solve) and safe to run
# repeatedly as the day unfolds.

def resolve_intraday(
        route, base_car: CarState, solar_provider, wind_provider, *,
        day_index: int,
        cur_soc_pct: float,
        elapsed_s: float,
        dist_done_km: float = 0.0,
        loops_remaining: int | dict | None = None,
        cs_taken: bool = False,
        alpha_next_day_pct: float | None = None,
        loop_geoms: dict | None = None,
        solar_efficiency: float | None = None,
        car_overrides: dict | None = None,
        global_method: str = "ga",
        seed: int | None = None) -> dict:
    """Re-plan the rest of TODAY from the car's current state.

    Parameters
    ----------
    day_index     : 0-indexed race day currently being driven.
    cur_soc_pct   : SOC right now.
    elapsed_s     : seconds since this day's official start (the "time of day").
    dist_done_km  : distance already driven today.
    loops_remaining : loops still to attempt today — an int (that many reps of
                    the day's loop) or a {loop_name: reps} dict. None -> take the
                    plan's loops minus none (i.e. all of them still to do).
    cs_taken      : has today's 30-min control stop already been served?
    alpha_next_day_pct : end-of-day SOC floor to respect (defaults to the pack
                    floor — spend freely, this is a within-day correction).
    solar_efficiency, car_overrides : same overrides as the macro resolve, for a
                    measured efficiency / parameter change mid-day.

    Returns a dict: feasible, velocity_profile_kmh, end_soc_pct, eta (true finish
    incl. remaining stops), remaining_drive_s, remaining_km, stages (per-stage
    breakdown of the REMAINDER), plus the inputs echoed under 'actuals'.
    """
    from configs import solver_config as _sc
    fwd_overrides = dict(car_overrides or {})
    if solar_efficiency is not None:
        if not (0.0 < solar_efficiency <= 1.0):
            raise ValueError(f"solar_efficiency must be in (0,1]; got {solar_efficiency}")
        fwd_overrides["array_efficiency"] = float(solar_efficiency)
    car = dataclasses.replace(base_car, **fwd_overrides)

    # Resolve the loops still to run today.
    plan = _get_day_plan(day_index)
    loops_committed: list = []
    if loops_remaining is None:
        for name, km in plan.loops:
            loops_committed.append((name, km))
    elif isinstance(loops_remaining, dict):
        for name, km in plan.loops:
            loops_committed.extend([(name, km)] * int(loops_remaining.get(name, 0)))
    else:  # an int: that many reps of the day's (first) loop
        n = int(loops_remaining)
        if plan.loops and n > 0:
            name, km = plan.loops[0]
            loops_committed = [(name, km)] * n

    alpha = (float(alpha_next_day_pct) if alpha_next_day_pct is not None
             else float(car.soc_min_pct))

    logger.info("INTRA-DAY resolve: Day %d, %.1f km done, t=%s into day, SOC %.1f%%, "
                "%d loop rep(s) left", day_index + 1, dist_done_km,
                _clock_hhmm(rc.day_start_time_s(day_index) + elapsed_s),
                cur_soc_pct, len(loops_committed))

    res = singleday.solve(
        route=route, car=car, solar_provider=solar_provider,
        wind_provider=wind_provider, day_index=day_index,
        start_soc_pct=float(cur_soc_pct), alpha_next_day_pct=alpha,
        loops_committed=loops_committed, dist_done_km=dist_done_km,
        elapsed_s=elapsed_s, cs_taken=cs_taken, loop_geoms=loop_geoms,
        global_method=global_method, seed=seed)

    end_soc = float(res.get("final_soc_pct", cur_soc_pct))
    feasible = end_soc >= car.soc_min_pct - 1e-6

    # Remaining mandatory stops from NOW: control stop only if not yet taken,
    # plus 5 min per loop rep still to run.
    n_loops = len(loops_committed)
    control_left_s = 0.0 if cs_taken else float(rc.CONTROL_STOP_DURATION_S)
    loop_stop_s = n_loops * (float(rc.LOOP_STOP_DURATION_S)
                             + float(getattr(rc, "LOOP_TURNAROUND_S", 0.0)))
    remaining_drive_s = float(res.get("total_time_s", 0.0))
    # True finish = day start + time already elapsed + remaining drive + the
    # stops still to serve.
    finish_abs_s = (rc.day_start_time_s(day_index) + float(elapsed_s)
                    + remaining_drive_s + control_left_s + loop_stop_s)

    stages = _stage_breakdown(
        res, getattr(_sc, "TRAILER_TOW_SPEED_KMH", 80.0),
        control_stop_s=control_left_s, loop_stop_total_s=loop_stop_s,
        stride_m=getattr(_sc, "OUTPUT_TRACE_STRIDE_M", 250.0), n_loop_reps=n_loops)

    return {
        "feasible": bool(feasible),
        "velocity_profile_kmh": res.get("v_kmh"),
        "end_soc_pct": round(end_soc, 2),
        "remaining_drive_s": round(remaining_drive_s, 0),
        "remaining_km": round(float(res.get("driven_km", 0.0)), 1),
        "stop_time_s": round(control_left_s + loop_stop_s, 0),
        "finish_abs_s": round(finish_abs_s, 0),
        "eta": _clock_hhmm(finish_abs_s),
        "on_time": finish_abs_s <= rc.day_finish_soft_limit_s(day_index),
        "stages": stages,
        "actuals": {
            "day_index": day_index, "cur_soc_pct": float(cur_soc_pct),
            "elapsed_s": float(elapsed_s), "dist_done_km": float(dist_done_km),
            "loops_remaining": n_loops, "cs_taken": bool(cs_taken),
            "solar_efficiency_used": car.array_efficiency,
        },
    }


# ===========================================================================
# 3. Re-solve from realized actuals  (strategist directive 21/08 — "the feature")
# ===========================================================================
#
# "I should be able to rerun the model the night before day x+1 (or the night
#  after day x) inputting whatever performance parameters actually happened —
#  the distance and loops actually driven, the SOC actually reached, and the
#  solar efficiency we actually saw — with sensible defaults for the rest, and
#  have the remaining days re-optimized around that reality."
#
# This is the between-days macro re-solve. Days already completed are locked;
# everything from `resume_day` onward is re-optimized as a fresh multi-day plan
# that STARTS from the SOC you actually reached and uses the solar efficiency
# you actually measured going forward. It reuses the exact same optimize() ->
# extract_final_profiles() pipeline as a cold start, so a re-solve is just a
# cold solve with a different start day, start SOC and (optionally) car params.
#
# Why start SOC is the load-bearing input: each future day is planned fresh from
# its own route/weather, so yesterday's distance and loop count only influence
# tomorrow THROUGH the battery state you carried into it. Feeding the measured
# start SOC is therefore what actually re-anchors the plan; distance/loops-done
# are accepted too (recorded, and used if you re-solve mid-day) but do not
# retro-change a completed day's optimization.
#
# Solar efficiency: forward solar power is area * array_efficiency * GHI, so a
# measured effective efficiency (panel x real-world capture: dust, soiling,
# haze) is applied simply by overriding array_efficiency for the remaining days.
# Scaling efficiency by k is mathematically identical to scaling every future
# day's GHI by k, so this single knob captures "we're only getting 90% of the
# predicted solar from here on" exactly. Defaults to the car's nominal value.

def resolve_from_actuals(
        routes: list, base_car: CarState,
        solar_providers: dict, wind_providers: dict, *,
        resume_day: int,
        start_soc_pct: float,
        solar_efficiency: float | None = None,
        car_overrides: dict | None = None,
        actual_distance_km: float | None = None,
        actual_loops_done: dict | None = None,
        dist_done_km: float = 0.0,
        elapsed_s: float = 0.0,
        cs_taken: bool = False,
        loops_done: dict | None = None,
        kml_paths: dict | None = None,
        loop_geoms_by_day: dict | None = None,
        plans_override: list | None = None,
        tier1_baseline: dict | None = None,
        parallel: bool = True,
        n_workers: int | None = None,
        global_method: str = "ga",
        seed: int | None = None,
        max_iters: int = MAX_ITERS,
        breakdown_enabled: bool = False,
        breakdown_seed: int | None = None) -> dict:
    """Re-optimize days `resume_day`..end from what actually happened.

    Parameters
    ----------
    resume_day : int
        0-indexed first day to (re)plan. To replan the night AFTER Day 2 for
        Day 3 onward, pass resume_day=2 (Day 3). Everything before it is frozen.
    start_soc_pct : float
        The battery SOC (percent) actually available at the START of `resume_day`
        — i.e. the measured end-of-previous-day SOC plus whatever overnight/
        morning charge really landed. This is the primary reality anchor.
    solar_efficiency : float, optional
        Measured EFFECTIVE array efficiency (0..1) to use for all remaining days
        (e.g. 0.20 if soiling knocked the nominal 0.22 panel down). Defaults to
        the car's configured array_efficiency (no change).
    car_overrides : dict, optional
        Any other measured car parameters that changed (mass_kg, p_idle_w, ...).
        Sensible default: {} (keep the configured car). solar_efficiency, if
        given, is merged in as array_efficiency and wins over any array_efficiency
        placed here.
    actual_distance_km, actual_loops_done : optional
        Recorded for the returned audit block only — they describe COMPLETED days
        and do not alter the forward optimization (each future day is planned
        fresh). Provided so the caller can log "what happened" alongside "what's
        planned next" in one object.
    dist_done_km, elapsed_s, cs_taken, loops_done : optional
        Only for the rarer MID-day re-solve (you're partway through `resume_day`
        itself). Left at their between-days defaults (0 / 0 / False / None), the
        remaining days are planned from a clean morning start — the common case.
    The remaining parameters mirror optimize() and default sensibly, so the
    minimal call is:
        resolve_from_actuals(routes, car, solar_p, wind_p,
                             resume_day=2, start_soc_pct=58.0)

    Returns
    -------
    dict with keys:
        feasible       : bool
        result         : the raw optimize() result (or None if infeasible)
        profiles       : extract_final_profiles() output ({} if infeasible)
        forward_car    : the CarState actually used for the remaining days
        actuals        : echo of the realized inputs (audit trail)
    """
    if not (0 <= resume_day < len(routes)):
        raise ValueError(
            f"resume_day={resume_day} out of range 0..{len(routes) - 1}")

    # Build the forward car: nominal car + measured overrides + solar efficiency.
    fwd_overrides = dict(car_overrides or {})
    if solar_efficiency is not None:
        if not (0.0 < solar_efficiency <= 1.0):
            raise ValueError(
                f"solar_efficiency must be in (0, 1]; got {solar_efficiency}")
        fwd_overrides["array_efficiency"] = float(solar_efficiency)
    fwd_car = dataclasses.replace(base_car, **fwd_overrides)

    logger.info(
        "RESOLVE from actuals: resume Day %d at SOC=%.1f%%, array_eff=%.3f%s",
        resume_day + 1, start_soc_pct, fwd_car.array_efficiency,
        (f", +overrides {sorted(k for k in fwd_overrides if k != 'array_efficiency')}"
         if len(fwd_overrides) > 1 or (fwd_overrides and 'array_efficiency' not in fwd_overrides)
         else ""))

    # Recompute the Tier-1 baseline against the FORWARD car unless the caller
    # supplied one — array_efficiency (or mass, etc.) changed the energy balance,
    # so a baseline built on the old car would misseed the SOC window.
    result = optimize(
        routes=routes, car=fwd_car,
        solar_providers=solar_providers, wind_providers=wind_providers,
        start_soc_pct=float(start_soc_pct), start_day=resume_day,
        dist_done_km=dist_done_km, elapsed_s=elapsed_s, cs_taken=cs_taken,
        loops_done=loops_done, kml_paths=kml_paths,
        loop_geoms_by_day=loop_geoms_by_day, plans_override=plans_override,
        parallel=parallel, n_workers=n_workers, global_method=global_method,
        seed=seed, max_iters=max_iters, tier1_baseline=tier1_baseline)

    profiles = {}
    if result.get("feasible"):
        # Pass the forward car as base_car with no further overrides so the
        # extracted physics match the re-solve exactly.
        profiles = extract_final_profiles(
            routes, fwd_car, solar_providers, wind_providers, result,
            loop_geoms_by_day=loop_geoms_by_day, plans_override=plans_override,
            breakdown_enabled=breakdown_enabled, breakdown_seed=breakdown_seed)
    else:
        logger.warning(
            "RESOLVE infeasible from SOC=%.1f%% at Day %d — the measured start "
            "state can't legally finish the remaining route. Try a higher start "
            "SOC or confirm the efficiency input.", start_soc_pct, resume_day + 1)

    return {
        "feasible": bool(result.get("feasible")),
        "result": result if result.get("feasible") else None,
        "profiles": profiles,
        "forward_car": fwd_car,
        "actuals": {
            "resume_day_index": resume_day,
            "start_soc_pct": float(start_soc_pct),
            "solar_efficiency_used": fwd_car.array_efficiency,
            "car_overrides": fwd_overrides,
            "actual_distance_km": actual_distance_km,
            "actual_loops_done": actual_loops_done,
            "mid_day": bool(dist_done_km or elapsed_s or cs_taken or loops_done),
        },
    }

if __name__ == "__main__":
    import os
    import sys
    # DIAGNOSTIC/COMPAT: force the 'spawn' start method (Windows/macOS default)
    # even on Linux, to reproduce and test the Windows multiprocessing path.
    if os.environ.get("AGNIRATH_FORCE_SPAWN"):
        import multiprocessing as _mp_force
        _mp_force.set_start_method("spawn", force=True)
    import glob
    import logging
    import json
    import pandas as pd
    import numpy as np
    from configs.car_config import CarState
    from core.solar import HourlyJSONSolarProvider, GaussianProvider, FlooredSolarProvider
    from core.wind import HourlyJSONWindProvider, ConstantWindProvider
    from core.route import Route

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger("main")

    # FEATURE 2: enable the one-breakdown-per-day scenario for this whole run
    # with `--breakdown` (applies to every day, both variants). Off by default.
    _BREAKDOWN_ENABLED = ("--breakdown" in sys.argv[1:])
    if _BREAKDOWN_ENABLED:
        logger.info("Breakdown scenario ENABLED (--breakdown): one power-scaled "
                    "breakdown per day, no charging during it.")

    # ------------------------------------------------------------------ #
    # 1.  Car + directory setup
    # ------------------------------------------------------------------ #
    car = CarState()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir  = os.path.abspath(os.path.join(current_dir, "..", "data", "solar"))
    kml_dir   = os.path.abspath(os.path.join(current_dir, "..", "data", "shaded"))
    save_dir  = os.path.abspath(os.path.join(current_dir, "..", "data", "processed"))

    # ------------------------------------------------------------------ #
    # 2.  Helpers
    # ------------------------------------------------------------------ #
    def _route_sort_key(filepath):
        name = filepath.lower()
        if "stage 1" in name: return 1
        if "stage 2" in name: return 3
        # Dedicated loop files (contain "loop" but NOT "stage") sort last
        if "loop" in name and "stage" not in name: return 2
        return 4

    def _is_loop_file(filepath):
        """True only for DEDICATED loop files, not stage files with 'loop' in a place name."""
        bn = os.path.basename(filepath).lower()
        return "loop" in bn and "stage" not in bn

    def _seg_type_for_file(filepath):
        """Real per-file stage tag matching core/route.py's documented schema
        ('stage1' | 'loop_<name>' | 'stage2'). Every row previously got the
        literal string "stage" regardless of source file, which collapsed
        the trailered-mask's "whole STAGE if any point flagged" rule into an
        unintended "whole DAY" rule, and made it impossible to target the
        Day 7 Stage 2 / Day 8 Stage 1 trailering override at just one stage."""
        name = filepath.lower()
        if "stage 1" in name:
            return "stage1"
        if "stage 2" in name:
            return "stage2"
        if "loop" in name:
            base = os.path.splitext(os.path.basename(filepath))[0]
            return f"loop_{base}"
        return "stage1"  # single-file days (no explicit stage split)

    def _parse_route_file(filepath, day_num, seg_type=None):
        """Parse ONE route .save file into a DataFrame using the same schema
        as _load_route's per-file block — factored out so dedicated loop
        geometry files reuse the exact same parsing. seg_type defaults to the
        real per-file tag (stage1/stage2/loop_<base>) via _seg_type_for_file;
        loop geometry's own seg_type is irrelevant because singleday's
        _splice_loops() re-tags every spliced leg as 'loop_<name>'."""
        with open(filepath, "r", encoding="utf-8") as f:
            route_data = json.load(f)
        prof = route_data["profile"]
        dists    = [x * 1000.0 for x in prof["Distance"]]
        slopes   = prof["Gradient"]
        bearings = prof.get("Headings",   [0.0] * len(dists))
        alts     = prof.get("Altitude",   [0.0] * len(dists))
        lats     = [c[0] for c in prof["Coordinates"]]
        lons     = [c[1] for c in prof["Coordinates"]]
        # Clamp speed limits: some route data has erroneous 5 km/h entries.
        # Floor at 30 km/h (highway minimum) to avoid artificial slowdowns.
        v_maxs   = [max(v, 30.0) / 3.6 for v in prof["SpeedLimit"]]
        ml = min(len(dists), len(slopes), len(bearings),
                 len(alts), len(lats), len(lons), len(v_maxs))
        return pd.DataFrame({
            "distance_m":      dists[:ml],
            "elevation_m":     alts[:ml],
            "slope_pct":       slopes[:ml],
            "bearing_deg":     bearings[:ml],
            "lat":             lats[:ml],
            "lon":             lons[:ml],
            "v_max_ms":        v_maxs[:ml],
            "curvature_1pm":   0.0,
            "circle_id":       0,
            "red_flag_trailer": False,
            "control_stop":    False,
            "day":             day_num,
            "seg_type":        seg_type or _seg_type_for_file(filepath),
        })

    def _load_route(route_files, day_num):
        """Parse .save files into a single Route for one day.

        Dedicated loop .save files are EXCLUDED from the base route.
        A file is a loop file only if it contains 'loop' but NOT 'stage'
        in its name — this avoids false positives on South African place
        names like 'Loopspruit' that appear in stage file names.
        """
        route_files = sorted(route_files, key=_route_sort_key)
        # Remove dedicated loop files only.
        route_files = [f for f in route_files if not _is_loop_file(f)]
        day_dfs = []
        offset = 0.0
        for filepath in route_files:
            part_df = _parse_route_file(filepath, day_num)
            part_df["distance_m"] += offset
            offset = part_df["distance_m"].max()
            day_dfs.append(part_df)
        return Route(pd.concat(day_dfs, ignore_index=True))

    def _loop_name_tokens(loop_name):
        """Searchable city/place tokens for a plan loop name, generic words
        dropped. 'postmasburg_loop2' -> {'postmasburg'};
        'blind_loop_placeholder' -> set() (no usable token — falls through
        to closest-distance matching)."""
        drop = {"loop", "blind", "placeholder", "day", "stage"}
        tokens = set()
        for part in loop_name.replace("-", "_").split("_"):
            part = part.strip().lower()
            if part and part not in drop:
                tokens.add(part)
        return tokens

    def _norm_loop_filename(filepath):
        """Lowercase, punctuation-stripped basename used for token matching."""
        bn = os.path.splitext(os.path.basename(filepath))[0].lower()
        return "".join(ch for ch in bn if ch.isalnum())

    def _load_loop_geometries(route_files, plan, day_num):
        """Match each plan loop name to a DEDICATED loop .save file for this
        day and return {loop_name: DataFrame} of its real terrain/speed-limit
        geometry. Previously these loop files were discarded entirely and
        every committed rep got a flat synthetic leg — this is the piece that
        finally feeds real loop physics into Tier 2 and the final profiles.

        Matching is one-to-one (each loop file is consumed at most once):
          1. Preferred: token match — every non-generic word of the plan loop
             name must appear in the file's basename (e.g. 'springbok_loop'
             -> any file containing 'springbok'). Among token hits the file
             whose real length is closest to the plan's nominal km wins.
          2. Fallback: closest real driving distance to the nominal km among
             still-unused loop files (covers blind/placeholder names like
             Day 3's 'blind_loop_placeholder', whose real loop file is
             variant-specific).

        Plan loops left unmatched are logged — singleday's _splice_loops()
        falls back to a flat synthetic leg for them.
        """
        loop_files = [f for f in route_files if _is_loop_file(f)]
        if not loop_files or not (plan.loops or []):
            return {}

        parsed = []
        for fp in loop_files:
            try:
                df = _parse_route_file(fp, day_num)
            except Exception as exc:
                logger.warning("Day %d: skipping unparseable loop file %s (%s)",
                               day_num, os.path.basename(fp), exc)
                continue
            if df is None or len(df) == 0:
                continue
            parsed.append((fp, df, float(df["distance_m"].max()) / 1000.0))

        if not parsed:
            return {}

        geoms: dict = {}
        used: set = set()

        for name, nom_km in (plan.loops or []):
            cands = [(fp, df, km) for (fp, df, km) in parsed if fp not in used]
            if not cands:
                break
            match = None
            tokens = _loop_name_tokens(name)
            if tokens:
                hits = [(fp, df, km) for (fp, df, km) in cands
                        if all(tok in _norm_loop_filename(fp) for tok in tokens)]
                if hits:
                    match = min(hits, key=lambda t: abs(t[2] - nom_km))
            if match is None:
                # Closest real distance among all still-unused loop files.
                match = min(cands, key=lambda t: abs(t[2] - nom_km))
            fp, df, km = match
            used.add(fp)
            geoms[name] = df
            token_hit = tokens and all(tok in _norm_loop_filename(fp) for tok in tokens)
            logger.info("Day %d: loop '%s' (nom %.1f km) -> geometry %s (%.1f km%s)",
                        day_num, name, nom_km, os.path.basename(fp), km,
                        "" if token_hit else " [closest-distance fallback]")

        for name, nom_km in (plan.loops or []):
            if name not in geoms:
                logger.warning("Day %d: loop '%s' (%.1f km) has NO geometry file "
                               "— singleday will use a flat synthetic leg",
                               day_num, name, nom_km)
        return geoms

    def _apply_trailered_mask(route, kml_paths_for_day: dict | None, day_index: int):
        """Compute the real trailered mask (KML + hardcoded Day7/Day8 ground
        truth + whole-stage rounding) and write it onto route so the REAL
        optimizer (singleday.solve -> forward_sim -> route.red_flag_at)
        actually sees it. Previously this mask was only ever computed inside
        tier1.py's own coarse energy diagnostic and printed to the log --
        route.df["red_flag_trailer"] stayed False for every point on every
        day, so the report's TRAILER column and forward_sim's trailering-aware
        physics never fired, even on days confirmed fully trailered."""
        if route is None:
            return route
        mask = tier1.compute_trailered_mask_full(route, kml_paths_for_day, day_index)
        if mask.any():
            route.set_red_flag_mask(mask)
        return route

    # Minimum acceptable daytime-avg GHI for September in South Africa.
    # Historical data below this is almost certainly an anomalous cloudy day
    # and should be blended with the clear-sky Gaussian fallback.
    MIN_DAYTIME_GHI_WM2 = 400.0
    GAUSSIAN_BLEND_FRAC  = 0.5   # blend weight for Gaussian when JSON is below floor

    # Moved to core.solar.FlooredSolarProvider so it's importable/picklable in
    # spawned worker processes (the fastened Tier-2 process pool). Aliased here
    # so the call site below is unchanged.
    _FlooredSolarProvider = FlooredSolarProvider

    def _check_daytime_avg_ghi(weather_files):
        """Quick scan: average GHI during race hours (8-17h) across all nodes."""
        all_ghi = []
        for fp in weather_files:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for node in data:
                ghi = node["historical_weather"]["hourly"]["shortwave_radiation"]
                all_ghi.append(np.mean(ghi[8:17]))  # hours 8-16 (race window)
        return float(np.mean(all_ghi)) if all_ghi else 0.0

    def _load_weather(weather_files, route, day_num=None):
        """Load ALL weather JSONs for a day into solar + wind providers.

        If daytime-avg GHI is anomalously low (< MIN_DAYTIME_GHI_WM2),
        the solar provider is wrapped with a GaussianProvider floor.
        """
        if not weather_files:
            return GaussianProvider(), ConstantWindProvider(0.0, 0.0)

        json_prov = HourlyJSONSolarProvider(weather_files, route)
        wind_prov = HourlyJSONWindProvider(weather_files, route)

        avg_ghi = _check_daytime_avg_ghi(weather_files)
        if avg_ghi < MIN_DAYTIME_GHI_WM2:
            logger.warning(
                "Day %s: daytime avg GHI = %.0f W/m² (below %.0f floor) — "
                "blending with Gaussian clear-sky fallback",
                day_num or "?", avg_ghi, MIN_DAYTIME_GHI_WM2)
            json_prov = _FlooredSolarProvider(json_prov, GAUSSIAN_BLEND_FRAC)

        return json_prov, wind_prov

    # ------------------------------------------------------------------ #
    # 3.  Load Days 1-8 (Day 3 handled separately below)
    # ------------------------------------------------------------------ #
    routes         = {}
    solar_providers = {}
    wind_providers  = {}
    kml_paths       = {}
    # Real per-day loop geometry {day_index: {loop_name: DataFrame}}, loaded
    # from each day's dedicated loop .save files and threaded down to
    # Tier 2 / singleday.solve so committed loop reps are actually simulated.
    loop_geoms_by_day = {}

    for d in range(8):
        day_num = d + 1

        # --- Route ---
        route_files = glob.glob(os.path.join(save_dir, f"*Day {day_num}*.save"))
        if d == 2:
            # Day 3 loaded separately as multi-variant below
            continue
        if route_files:
            # Log which files are stage vs loop (dedicated loops excluded by _load_route)
            stage_files = [f for f in route_files if not _is_loop_file(f)]
            loop_files  = [f for f in route_files if _is_loop_file(f)]
            routes[d] = _load_route(route_files, day_num)
            logger.info("Day %d: route loaded (%d stage files, %d loop files excluded)",
                        day_num, len(stage_files), len(loop_files))
        else:
            logger.warning("Day %d: no route .save files found — flat fallback", day_num)
            routes[d] = None

        # --- Real loop geometry (dedicated loop .save files) ---
        # Day 3 (d==2) is variant-specific and handled in section 5.
        if d != 2:
            loop_geoms_by_day[d] = _load_loop_geometries(
                route_files, _get_day_plan(d), day_num)

        # --- Weather (ALL files for this day) ---
        weather_files = glob.glob(os.path.join(json_dir, f"*Day {day_num}*.json"))
        solar_providers[d], wind_providers[d] = _load_weather(weather_files, routes.get(d), day_num)
        if weather_files:
            logger.info("Day %d: loaded %d weather JSONs", day_num, len(weather_files))

        # --- KML trailering ---
        kml_files = glob.glob(os.path.join(kml_dir, f"*Day {day_num}*.kml"))
        kml_paths[d] = kml_files[0] if kml_files else None
        _apply_trailered_mask(routes.get(d), kml_paths, d)

    # ------------------------------------------------------------------ #
    def _build_day3_variant_plan(variant_route_files, variant_name, day_num=3):
        """Build Day 3 strictly from the released variant route files.

        Aryaman: Stage 2 + Loop, with NO Stage 1.
        Prahlad: Stage 1 + Stage 2 + Loop.
        """
        stage1_files, stage2_files, loop_files = [], [], []

        for fp in variant_route_files:
            bn = os.path.basename(fp).lower()
            if _is_loop_file(fp):
                loop_files.append(fp)
            elif "stage 1" in bn:
                stage1_files.append(fp)
            elif "stage 2" in bn:
                stage2_files.append(fp)

        variant = variant_name.lower()

        if variant == "aryaman":
            if stage1_files:
                raise ValueError(
                    "Day 3 Aryaman unexpectedly contains a Stage 1 file: "
                    f"{stage1_files}"
                )
            if len(stage2_files) != 1:
                raise ValueError(
                    "Day 3 Aryaman must contain exactly one Stage 2 file; "
                    f"found {len(stage2_files)}: {stage2_files}"
                )
            if len(loop_files) != 1:
                raise ValueError(
                    "Day 3 Aryaman must contain exactly one loop file; "
                    f"found {len(loop_files)}: {loop_files}"
                )
            stage1_km = 0.0
        elif variant == "prahlad":
            if len(stage1_files) != 1:
                raise ValueError(
                    "Day 3 Prahlad must contain exactly one Stage 1 file; "
                    f"found {len(stage1_files)}: {stage1_files}"
                )
            if len(stage2_files) != 1:
                raise ValueError(
                    "Day 3 Prahlad must contain exactly one Stage 2 file; "
                    f"found {len(stage2_files)}: {stage2_files}"
                )
            if len(loop_files) != 1:
                raise ValueError(
                    "Day 3 Prahlad must contain exactly one loop file; "
                    f"found {len(loop_files)}: {loop_files}"
                )
            stage1_km = _parse_route_file(
                stage1_files[0], day_num
            )["distance_m"].max() / 1000.0
        else:
            raise ValueError(f"Unknown Day 3 variant '{variant_name}'")

        stage2_km = _parse_route_file(
            stage2_files[0], day_num
        )["distance_m"].max() / 1000.0
        loop_km = _parse_route_file(
            loop_files[0], day_num
        )["distance_m"].max() / 1000.0

        plan = tier1._DayPlan(
            stage1_km=float(stage1_km),
            stage2_km=float(stage2_km),
            loops=(("day3_loop", float(loop_km)),),
        )
        logger.info(
            "Day 3 [%s] plan: Stage1=%.1f km, Stage2=%.1f km, Loop=%.1f km",
            variant_name, plan.stage1_km, plan.stage2_km, loop_km
        )
        return plan

    # 4.  Day 3 multi-variant discovery
    # ------------------------------------------------------------------ #
    day3_route_files = glob.glob(os.path.join(save_dir, "*Day 3*probables*.save")) or \
                       glob.glob(os.path.join(save_dir, "*Day 3*.save"))
    day3_weather_files = glob.glob(os.path.join(json_dir, "*Day 3*.json"))

    # Group by variant name (e.g. "Prahlad", "Aryaman")
    day3_variants = {}
    for fp in day3_route_files:
        bn = os.path.basename(fp).lower()
        if "prahlad" in bn:
            day3_variants.setdefault("prahlad", []).append(fp)
        elif "aryaman" in bn:
            day3_variants.setdefault("aryaman", []).append(fp)
        else:
            day3_variants.setdefault("unknown", []).append(fp)

    if not day3_variants:
        logger.warning("Day 3: no route files found — flat fallback for single variant")
        day3_variants = {"flat_fallback": []}

    logger.info("Day 3 variants discovered: %s", list(day3_variants.keys()))

    # Match weather files to each variant
    day3_variant_weather = {}
    for vname in day3_variants:
        matched = [f for f in day3_weather_files if vname.lower() in os.path.basename(f).lower()]
        if not matched:
            matched = day3_weather_files  # fallback: use all available
        day3_variant_weather[vname] = matched

    # ------------------------------------------------------------------ #
    # 5.  Run optimizer once per Day-3 variant, each with its OWN Tier 1
    #     baseline
    # ------------------------------------------------------------------ #
    # Tier 1's guess_baseline bakes in the full day-0..7 route + weather set
    # (the Day-3 variant is swapped into routes[2] before it runs), and the
    # resulting day_plans / s0_pct / feasible flag are all Day-3-dependent.
    # Reusing one shared baseline computed from the FIRST variant for every
    # other variant is wrong: different Day 3 route -> different energy
    # balance -> different s_center -> Tier 2 samples the wrong SOC window
    # and Tier 3 chains days around a Day-3-specific policy. Recompute per
    # variant so each result is self-consistent (cost: one ~10-min Tier 1
    # pass per variant).
    kml_files_d3 = glob.glob(os.path.join(kml_dir, "*Day 3*.kml"))

    # ------------------------------------------------------------------ #
    # 5a. RE-SOLVE-FROM-ACTUALS CLI  (Feature C — locally testable)
    # ------------------------------------------------------------------ #
    # Usage (skips the full 2-variant race run and re-optimizes just the
    # remaining days from a realized state you type in):
    #
    #   python -m optimizers.trust_region resolve \
    #       --resume-day 4 --start-soc 58 [--solar-eff 0.20] [--variant prahlad]
    #
    #   --resume-day  human day number (4 == Day 4) that you want to plan NEXT;
    #                 everything before it is treated as already run.
    #   --start-soc   the SOC (%) actually available at the start of that day.
    #   --solar-eff   OPTIONAL measured effective array efficiency (0..1) for the
    #                 remaining days; defaults to the car's nominal value.
    #   --variant     OPTIONAL Day-3 route variant to assume (default 'prahlad');
    #                 only matters if --resume-day <= 3.
    #
    # This is the exact optimize()->extract pipeline used for a cold start, so a
    # re-solve is just a cold solve from a different day/SOC/efficiency. Fast for
    # late resume days (only the tail is optimized).
    def _flag(name, cast, default=None):
        if name in sys.argv:
            return cast(sys.argv[sys.argv.index(name) + 1])
        return default

    # ------------------------------------------------------------------ #
    # INTRA-DAY resolve CLI (Feature 4 — single-day optimizer only)
    # ------------------------------------------------------------------ #
    #   python -m optimizers.trust_region resolve-intraday \
    #       --day 4 --soc 62 --time 11:30 [--dist-done 120] [--loops-left 3] \
    #       [--solar-eff 0.20] [--cs-taken] [--variant prahlad]
    # Re-plans the REMAINDER of the given day from the current state.
    if "resolve-intraday" in sys.argv[1:]:
        iday = _flag("--day", int, None)
        isoc = _flag("--soc", float, None)
        itime = _flag("--time", str, None)     # HH:MM clock, or minutes-into-day
        if iday is None or isoc is None or itime is None:
            logger.error("resolve-intraday needs --day <N> --soc <pct> --time <HH:MM|minutes>")
            sys.exit(2)
        didx = iday - 1
        if ":" in str(itime):
            _h, _m = map(int, str(itime).split(":"))
            elapsed = _h * 3600 + _m * 60 - rc.day_start_time_s(didx)
        else:
            elapsed = float(itime) * 60.0
        elapsed = max(0.0, float(elapsed))
        idist = _flag("--dist-done", float, 0.0)
        iloops = _flag("--loops-left", int, None)
        ieff = _flag("--solar-eff", float, None)
        ivariant = _flag("--variant", str, "prahlad")
        ics = ("--cs-taken" in sys.argv)

        # Materialise the day's route (Day 3 needs its variant).
        if didx == 2:
            vfiles = day3_variants.get(ivariant, next(iter(day3_variants.values())))
            routes[2] = _load_route(vfiles, 3) if vfiles else None
            _d3p = _build_day3_variant_plan(vfiles, ivariant, day_num=3)
            loop_geoms_by_day[2] = _load_loop_geometries(vfiles, _d3p, 3)
            solar_providers[2], wind_providers[2] = _load_weather(
                day3_variant_weather.get(ivariant, []), routes.get(2), 3)

        rr = resolve_intraday(
            routes.get(didx), car, solar_providers.get(didx), wind_providers.get(didx),
            day_index=didx, cur_soc_pct=isoc, elapsed_s=elapsed,
            dist_done_km=idist, loops_remaining=iloops, cs_taken=ics,
            solar_efficiency=ieff, loop_geoms=loop_geoms_by_day.get(didx))

        print("\n" + "=" * 64)
        print(f" INTRA-DAY RE-PLAN — Day {iday} from {itime} "
              f"(SOC {isoc:.0f}%, {idist:.0f} km done)")
        print("=" * 64)
        print(f"  feasible          : {rr['feasible']}")
        print(f"  remaining drive   : {rr['remaining_drive_s']/3600:.2f} h "
              f"({rr['remaining_km']:.1f} km)")
        print(f"  predicted end SOC : {rr['end_soc_pct']:.1f}%")
        print(f"  finish ETA        : {rr['eta']}  "
              f"({'on time' if rr['on_time'] else 'LATE — past soft limit'})")
        _vp = rr.get("velocity_profile_kmh")
        if _vp is not None and len(_vp):
            print(f"  avg target speed  : {float(np.mean(_vp)):.1f} km/h "
                  f"(max {float(np.max(_vp)):.0f})")
        for st in (rr.get("stages") or []):
            print(f"     · {st['stage']:<7} {st.get('distance_km',0):>6.1f} km  "
                  f"avg {st.get('speed_avg_kmh','?')}  ETA {st.get('eta','?')}  "
                  f"loops {st.get('n_loops',0)}")
        print("=" * 64 + "\n")
        sys.exit(0)

    if "resolve" in sys.argv[1:]:
        human_day = _flag("--resume-day", int, None)
        start_soc = _flag("--start-soc", float, None)
        if human_day is None or start_soc is None:
            logger.error("resolve needs --resume-day <N> and --start-soc <pct>")
            sys.exit(2)
        resume_idx = human_day - 1
        solar_eff = _flag("--solar-eff", float, None)
        variant_name = _flag("--variant", str, "prahlad")
        if "--mp" in sys.argv:  # test the fastened process-pool Tier 2 path
            from configs import solver_config as _sc_mp
            _sc_mp.TIER2_USE_PROCESS_POOL = True
            logger.info("resolve: TIER2_USE_PROCESS_POOL forced ON for this run")

        # Materialise a Day-3 route/plan/weather for the chosen variant so the
        # routes dict covers all 8 day slots (optimize/extract index 0..7).
        variant_route_files = day3_variants.get(
            variant_name, next(iter(day3_variants.values())))
        routes[2] = _load_route(variant_route_files, 3) if variant_route_files else None
        day3_plan = _build_day3_variant_plan(variant_route_files, variant_name, day_num=3)
        loop_geoms_by_day[2] = _load_loop_geometries(variant_route_files, day3_plan, 3)
        solar_providers[2], wind_providers[2] = _load_weather(
            day3_variant_weather.get(variant_name, []), routes.get(2), 3)
        kml_paths[2] = kml_files_d3[0] if kml_files_d3 else None
        _apply_trailered_mask(routes.get(2), kml_paths, 2)

        variant_plans = [_get_day_plan(d) for d in range(8) if d != 2]
        variant_plans.insert(2, day3_plan)

        logger.info("=" * 60)
        logger.info("RE-SOLVE FROM ACTUALS: Day %d onward, start SOC %.1f%%%s",
                    human_day, start_soc,
                    f", solar_eff={solar_eff}" if solar_eff is not None else "")
        logger.info("=" * 60)

        rr = resolve_from_actuals(
            routes, car, solar_providers, wind_providers,
            resume_day=resume_idx, start_soc_pct=start_soc,
            solar_efficiency=solar_eff,
            kml_paths=kml_paths, loop_geoms_by_day=loop_geoms_by_day,
            plans_override=variant_plans, parallel=True,
            breakdown_enabled=("--breakdown" in sys.argv[1:]))

        print("\n" + "=" * 64)
        print(f" RE-SOLVE RESULT — Day {human_day} onward "
              f"(variant={variant_name})")
        print("=" * 64)
        print(f" feasible          : {rr['feasible']}")
        print(f" start SOC          : {rr['actuals']['start_soc_pct']:.1f}%")
        print(f" array efficiency   : {rr['actuals']['solar_efficiency_used']:.3f}")
        if rr["feasible"]:
            profs = rr["profiles"]
            tot_driven = 0.0
            for d in sorted(profs):
                p = profs[d]
                driven = float(p.get("driven_km", 0.0))
                tot_driven += driven
                # TRUE finish incl. mandatory stops (control + per-loop) + any
                # inherited penalty hold — not drive time alone.
                fin = p.get("finish_abs_s")
                if fin is None:
                    fin = (rc.day_start_time_s(d) + float(p.get("total_time_s", 0.0))
                           + float(p.get("stop_time_s", _day_mandatory_stop_s(len(p.get("loops_committed") or []))))
                           + float(p.get("inherited_penalty_min", 0)) * 60.0)
                vprof = p.get("velocity_profile_kmh")
                _vp = np.asarray(vprof, dtype=float) if vprof is not None and len(vprof) else np.array([0.0])
                avg_v = float(_vp.mean()); max_v = float(_vp.max()); min_v = float(_vp.min())
                print(f"  Day {d+1}: start {p['start_soc_pct']:.1f}%  "
                      f"end {p['end_soc_pct']:.1f}%  driven {driven:.1f} km  "
                      f"avg {avg_v:.1f} (min {min_v:.0f}/max {max_v:.0f}) km/h  "
                      f"finish {int(fin//3600):02d}:{int((fin%3600)//60):02d}  "
                      f"loops {len(p.get('loops_committed', []))}  "
                      f"latepen {p.get('late_penalty_min',0)}m  "
                      f"inherited {p.get('inherited_penalty_min',0)}m")
                for st in (p.get("stages") or []):
                    print(f"        · {st['stage']:<7} {st.get('distance_km',0):>6.1f} km  "
                          f"SOC {st.get('soc_start_pct','?')}→{st.get('soc_end_pct','?')}%  "
                          f"avg {st.get('speed_avg_kmh','?')} (max {st.get('speed_max_kmh','?')})  "
                          f"solar {st.get('solar_wh','?')}Wh  "
                          f"trailer {st.get('trailered_km',0)}km  ETA {st.get('eta','?')}")
            print(f"  --> remaining driven distance: {tot_driven:.1f} km")
        else:
            print("  (infeasible from this state — raise start SOC or check "
                  "the efficiency input)")
        print("=" * 64 + "\n")
        if os.environ.get("AGNIRATH_DUMP_STAGES") and rr.get("feasible"):
            _dump = {}
            for _d, _p in (rr.get("profiles") or {}).items():
                _dump[str(_d + 1)] = {
                    "day": _d + 1, "eta": _p.get("eta"),
                    "end_soc_pct": _p.get("end_soc_pct"),
                    "evening_charge_pct": _p.get("evening_charge_pct"),
                    "end_soc_after_evening_pct": _p.get("end_soc_after_evening_pct"),
                    "next_start_soc_pct": _p.get("next_start_soc_pct"),
                    "breakdown_min": _p.get("breakdown_min"),
                    "late_penalty_min": _p.get("late_penalty_min"),
                    "morning_charge_pct": _p.get("morning_charge_pct"),
                }
            with open(os.environ["AGNIRATH_DUMP_STAGES"], "w") as _f:
                json.dump(_dump, _f, indent=2, default=str)
            logger.info("dumped stages -> %s", os.environ["AGNIRATH_DUMP_STAGES"])
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # FEATURE: --variant restricts this cold-start run to a single Day 3
    # route variant instead of always looping over every discovered one.
    # Skips the other variant's Tier-1 baseline + full optimize() pass
    # entirely, which is the expensive part (~1 wall-clock unit/variant),
    # so isolating one variant during iterative tuning avoids burning
    # time re-solving a variant you're not looking at.
    #
    #   python -m optimizers.trust_region --variant aryaman
    #   python -m optimizers.trust_region --variant prahlad
    #   python -m optimizers.trust_region                    # both (default)
    #
    # Uses the same lightweight _flag() sys.argv reader as the
    # resolve/resolve-intraday subcommands above (kept consistent rather
    # than introducing argparse, which doesn't compose cleanly with the
    # '"resolve" in sys.argv[1:]' style subcommand dispatch used here).
    _variant_choice = _flag("--variant", str, "both").lower()
    if _variant_choice not in ("prahlad", "aryaman", "both"):
        logger.error(
            "--variant must be one of: prahlad, aryaman, both (got %r)",
            _variant_choice)
        sys.exit(2)
    if _variant_choice != "both":
        if _variant_choice not in day3_variants:
            logger.error(
                "--variant %s requested but no matching Day 3 route files "
                "were discovered (found: %s)",
                _variant_choice, list(day3_variants.keys()))
            sys.exit(2)
        day3_variants = {_variant_choice: day3_variants[_variant_choice]}
        logger.info("Restricting cold-start run to Day 3 variant: %s",
                    _variant_choice)

    all_results = {}

    for variant_name, variant_route_files in day3_variants.items():
        logger.info("=" * 60)
        logger.info("RUNNING VARIANT: Day 3 = %s", variant_name)
        logger.info("=" * 60)

                # Build Day 3 route for this variant
        if variant_route_files:
            routes[2] = _load_route(variant_route_files, 3)
            logger.info("Day 3 [%s]: route loaded (%d files)", variant_name, len(variant_route_files))
        else:
            routes[2] = None

        # Day 3 is variant-specific: load ITS OWN dedicated loop geometry so
        # committed 'blind_loop_placeholder' reps simulate the real variant
        # loop terrain instead of a flat synthetic leg.
        day3_plan = _build_day3_variant_plan(
            variant_route_files, variant_name, day_num=3
        )
        loop_geoms_by_day[2] = _load_loop_geometries(
            variant_route_files, day3_plan, 3
        )

        # Same variant-specific plans are used by Tier 1, Tier 2/Tier 3,
        # final profile extraction and reporting.
        variant_plans = [_get_day_plan(d) for d in range(8) if d != 2]
        variant_plans.insert(2, day3_plan)
        if len(variant_plans) != rc.N_RACE_DAYS:
            raise RuntimeError(
                f"Day 3 variant plan has {len(variant_plans)} days; expected {rc.N_RACE_DAYS}"
            )

        # Build Day 3 weather for this variant
        weather_files = day3_variant_weather.get(variant_name, [])
        solar_providers[2], wind_providers[2] = _load_weather(weather_files, routes.get(2), 3)

                # KML for Day 3
        kml_paths[2] = kml_files_d3[0] if kml_files_d3 else None
        _apply_trailered_mask(routes.get(2), kml_paths, 2)

        # --- Tier 1 baseline for THIS variant (Day 3 already swapped in) ---
        logger.info("Computing Tier 1 baseline for variant '%s'...", variant_name)
        baseline = tier1.guess_baseline(
            routes, car, solar_providers, wind_providers, 100.0,
            start_day=0, kml_paths=kml_paths,
            plans_override=variant_plans)

        # --- Run trust-region optimizer (per-variant Tier 1 baseline) ---
        result = optimize(
            routes=routes,
            car=car,
            solar_providers=solar_providers,
            wind_providers=wind_providers,
            kml_paths=kml_paths,
            loop_geoms_by_day=loop_geoms_by_day,
            plans_override=variant_plans,
            start_soc_pct=100.0,
            start_day=0,
            parallel=True,
            max_iters=MAX_ITERS,
            tier1_baseline=baseline,
        )

        # --- Extract final speed profiles if feasible ---
        profiles = {}
        if result.get("feasible"):
            profiles = extract_final_profiles(
                routes, car, solar_providers, wind_providers, result,
                loop_geoms_by_day=loop_geoms_by_day,
                plans_override=variant_plans,
                breakdown_enabled=_BREAKDOWN_ENABLED)

        all_results[variant_name] = {"result": result, "profiles": profiles, "plans": variant_plans}

    # ------------------------------------------------------------------ #
    # 6.  Helpers for per-day strategy plan
    # ------------------------------------------------------------------ #
    # Route labels are the single source of truth in race_config.DAY_ROUTE_NOTES
    # (start → finish per the released 2026 route sheets). The old hardcoded
    # dict here was stale — e.g. Day 1 read "Johannesburg → Rustenburg" when the
    # real Day 1 is Sasolburg → Swartruggens (Rustenburg is only the control
    # stop). Build labels from the config so the report can never drift again.
    def _day_label(d_idx: int) -> str:
        try:
            note = rc.DAY_ROUTE_NOTES[d_idx]
            return f"{note['start']} → {note['finish']}"
        except Exception:
            return f"Day {d_idx + 1}"
    DAY_NAMES = {d: _day_label(d) for d in range(rc.N_RACE_DAYS)}

    def _hms(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h:02d}:{m:02d}"

    def _clock(abs_s: float) -> str:
        """Absolute seconds from midnight → HH:MM clock string."""
        return _hms(abs_s)

    # Car constants for output calculations.
    # FIX: these getattr() calls previously used attribute names that don't
    # exist on CarState ('solar_area', 'solar_eff', 'battery_capacity_wh'),
    # so every call silently fell through to the hardcoded default. That
    # made array_area_m2/battery figures coincidentally correct (they matched
    # their fallback defaults) but array_efficiency is really 0.22, not the
    # fallback 0.18 -- every "Solar input" figure was ~18% understated.
    _SOLAR_AREA  = car.array_area_m2
    _SOLAR_EFF   = car.array_efficiency
    _BATT_CAP_WH = car.battery_nominal_wh

    def _loop_km_for(plan, lp) -> float:
        """Committed loop km = Σ reps × official single-pass loop length."""
        return sum(cnt * km for (name, km) in plan.loops
                   for cnt in [lp.get(name, 0)]) if lp else 0.0

    def _trailered_km_for(d_idx, profiles) -> float:
        if profiles and d_idx in profiles:
            return float(profiles[d_idx].get("trailered_km", 0.0) or 0.0)
        return 0.0

    def _real_day_km(d_idx, plan, lp, profiles) -> float:
        """Distance credited to the race total, from the AUTHORITATIVE route
        table so the reported components always reconcile and match the
        released sheets:

            distance_km = stage1_km + stage2_km + committed_loop_km
                          − trailered_km        (SR asterisk rule)

        This is deliberately NOT the raw simulated driven_km: the forward
        integrator's driven_km differs from the published figure by the
        KML-linestring-vs-published rounding AND by the small synthetic
        loop-separator buffers spliced between repeated loop attempts
        (singleday._splice_loops), so it never reconciles cleanly with the
        headline stage/loop numbers. The simulated distance is still reported
        separately (distance_km_simulated) for transparency. Day 3 uses the
        variant plan's real geometry lengths, so it stays variant-specific.
        """
        loop_km = _loop_km_for(plan, lp)
        trailered_km = _trailered_km_for(d_idx, profiles)
        return max(0.0, plan.stage1_km + plan.stage2_km + loop_km - trailered_km)

    def _simulated_day_km(d_idx, profiles) -> float:
        """Raw driven distance from forward_sim (excludes trailered km)."""
        if profiles and d_idx in profiles:
            dk = profiles[d_idx].get("driven_km")
            if dk is not None and dk > 0:
                return float(dk)
        return 0.0

    def _estimate_solar_input_wh(solar_prov, day_index: int) -> float:
        """Rough total solar energy over the race window (Wh)."""
        if solar_prov is None:
            return 0.0
        t0 = rc.day_start_time_s(day_index)
        t1 = rc.day_finish_time_s(day_index)
        dt = 300.0  # 5-min steps
        total = 0.0
        t = t0
        while t < t1:
            ghi = solar_prov.ghi_wm2(t, 0.0)
            total += ghi * _SOLAR_AREA * _SOLAR_EFF * dt / 3600.0  # Wh, not joules
            t += dt
        return total

    # ------------------------------------------------------------------ #
    # 7.  Print full strategy plan for all variants
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE — ALL DAY 3 VARIANTS")
    print("=" * 70)

    for variant_name, data in all_results.items():
        result   = data["result"]
        profiles = data["profiles"]
        plans    = data["plans"]

        print(f"\n{'━' * 70}")
        print(f"  Day 3 variant: {variant_name.upper()}")
        print(f"{'━' * 70}")
        print(f"  Converged:  {result.get('converged')}")
        print(f"  Feasible:   {result.get('feasible')}")
        print(f"  Iterations: {result.get('iterations')}")
        # NOTE: real total (driven km only, trailered days excluded) is
        # printed after the per-day table below, once _real_day_km has run
        # for each day — the DP-internal total_distance_km here is the
        # allocator's own planning estimate (static plan tables, doesn't
        # know about trailering) and is kept only as a cross-check.
        print(f"  Allocator estimate (pre-cutoff ceiling): {result.get('total_distance_km', 0):.1f} km  "
              f"(actual counted total is below, after cutoff-feasibility drops)")

        if not result.get("feasible"):
            print("  ⚠ Infeasible — no per-day plan available")
            continue

        loop_plan = result.get("loop_plan", {})
        s_start = result.get("s_start_pct")

        print(f"\n  {'─' * 80}")
        print(f"  {'DAY':>5} │ {'ROUTE':<30} │ {'KM':>7} │ {'LOOPS':>5} │ {'SOC START→END':>14} │ {'ETA':>5} │ {'TRAILER':>8}")
        print(f"  {'─' * 80}")

        from collections import Counter as _Counter
        def _effective_lp(d_idx):
            """Actual loop reps driven — from the profile's loops_committed
            (which reflects any finish-backstop drops in extract), falling back
            to the allocator's loop_plan when no profile exists."""
            if profiles and d_idx in profiles and profiles[d_idx].get("loops_committed") is not None:
                return dict(_Counter(nm for nm, _km in profiles[d_idx]["loops_committed"]))
            return loop_plan.get(d_idx, {})

        _real_total_km = 0.0
        _total_trailered_km = 0.0
        for d_idx in range(8):
            plan = plans[d_idx]
            lp = _effective_lp(d_idx)
            n_loops = sum(lp.values()) if lp else 0
            loop_km = sum(cnt * km for (name, km) in plan.loops
                          for cnt in [lp.get(name, 0)])
            day_km = _real_day_km(d_idx, plan, lp, profiles)
            route_name = DAY_NAMES.get(d_idx, f"Day {d_idx + 1}")

            # SOC
            soc_start = (float(profiles[d_idx]["start_soc_pct"])
                         if profiles and d_idx in profiles and profiles[d_idx].get("start_soc_pct") is not None
                         else (float(s_start[d_idx]) if s_start is not None and np.isfinite(s_start[d_idx]) else 0.0))

            # End SOC + ETA from profiles if available
            if profiles and d_idx in profiles:
                p = profiles[d_idx]
                soc_end = p.get("end_soc_pct", 0.0)
                # Use total_time_s (drive duration) if available; fall back to
                # deriving from t_s (but t_s is ABSOLUTE clock time, not duration).
                drive_time = p.get("total_time_s")
                if drive_time is None:
                    t_arr = p.get("time_array_s")
                    if t_arr is not None and hasattr(t_arr, '__iter__'):
                        # t_s is absolute → duration = max(t_s) - day_start
                        drive_time = float(max(t_arr)) - rc.day_start_time_s(d_idx)
                    else:
                        drive_time = 0.0
            else:
                soc_end = 0.0
                drive_time = 0.0

            t_start_abs = rc.day_start_time_s(d_idx)
            # TRUE ETA = start + prior-day penalty hold + drive + mandatory stops
            # (control + per-loop). total_time_s is drive-only, so the stops must
            # be added or the arrival is understated by ~30 min + 5 min/loop.
            if profiles and d_idx in profiles:
                p = profiles[d_idx]
                _finish_abs = p.get("finish_abs_s")
                if _finish_abs is None:
                    _stop_s = float(p.get("stop_time_s", _day_mandatory_stop_s(n_loops)))
                    _inh_s = float(p.get("inherited_penalty_min", 0) or 0) * 60.0
                    _finish_abs = t_start_abs + drive_time + _stop_s + _inh_s
                eta_abs = float(_finish_abs)
            else:
                eta_abs = t_start_abs + drive_time
            eta_str = _clock(eta_abs) if drive_time > 0 else "—"

            trailer_km = 0.0
            if profiles and d_idx in profiles:
                trailer_km = profiles[d_idx].get("trailered_km", 0.0) or 0.0
            trailer_str = f"{trailer_km:.1f}km" if trailer_km > 0 else "—"
            _real_total_km += day_km
            _total_trailered_km += trailer_km
            print(f"  {d_idx+1:>5} │ {route_name:<30} │ {day_km:>6.1f} │ {n_loops:>5} │ "
                  f"{soc_start:>5.1f}% → {soc_end:>5.1f}% │ {eta_str:>5} │ {trailer_str:>8}")

        print(f"  {'─' * 80}")
        # Reconciled distance ledger (these now ADD UP):
        #   COUNTED (driven, scores)  +  TRAILERED (excluded)  =  GROUND COVERED
        # The DP planning estimate above is the allocator's PRE-cutoff figure
        # (before the finish-backstop drops loops that couldn't fit by 17:30),
        # so it is intentionally >= the counted total; it is a ceiling, not a
        # sum term.
        _dp_est = result.get('total_distance_km', 0) or 0.0
        _ground_covered = _real_total_km + _total_trailered_km
        print(f"  Counted distance (driven, scores):            {_real_total_km:>8.1f} km")
        if _total_trailered_km > 0:
            print(f"  Trailered (NOT counted, SR asterisk rule):    {_total_trailered_km:>8.1f} km")
            print(f"  = Ground covered (driven + trailered):        {_ground_covered:>8.1f} km")
        print(f"  (allocator pre-cutoff estimate was {_dp_est:.1f} km; "
              f"{max(0.0, _dp_est - _real_total_km):.1f} km of loops were dropped to finish by the cutoff)")

        # ── Detailed per-day strategy ──
        print(f"\n  {'═' * 66}")
        print("  DETAILED DAILY STRATEGY")
        print(f"  {'═' * 66}")

        for d_idx in range(8):
            plan = plans[d_idx]
            lp = _effective_lp(d_idx)
            n_loops = sum(lp.values()) if lp else 0
            loop_km = sum(cnt * km for (name, km) in plan.loops
                          for cnt in [lp.get(name, 0)])
            day_km = _real_day_km(d_idx, plan, lp, profiles)
            route_name = DAY_NAMES.get(d_idx, f"Day {d_idx + 1}")
            soc_start = (float(profiles[d_idx]["start_soc_pct"])
                         if profiles and d_idx in profiles and profiles[d_idx].get("start_soc_pct") is not None
                         else (float(s_start[d_idx]) if s_start is not None and np.isfinite(s_start[d_idx]) else 0.0))

            print(f"\n  ── Day {d_idx + 1}: {route_name} ──")

            # 1. Km planned + loops
            loop_detail = ", ".join(f"{name}×{cnt}" for name, cnt in lp.items()) if lp else "none"
            print(f"  1) Distance: {day_km:.1f} km | Loops: {loop_detail}")

            # 2. End of day SOC
            if profiles and d_idx in profiles:
                p = profiles[d_idx]
                soc_end = p.get("end_soc_pct", 0.0) or 0.0
                # total_time_s is the actual drive duration (not absolute clock)
                total_time = p.get("total_time_s")
                if total_time is None:
                    t_arr = p.get("time_array_s")
                    if t_arr is not None and hasattr(t_arr, '__iter__'):
                        total_time = float(max(t_arr)) - rc.day_start_time_s(d_idx)
                    else:
                        total_time = 0.0
                v_arr = p.get("velocity_profile_kmh")
            else:
                soc_end = 0.0
                total_time = 0.0
                v_arr = None
            print(f"  2) End-of-day SOC: {soc_end:.1f}%")

            # 3. SOC curve (key checkpoints)
            soc_drain_pct = soc_start - soc_end
            solar_wh = float((profiles.get(d_idx, {}) or {}).get("solar_energy_wh", 0.0) or 0.0)
            drain_wh = soc_drain_pct / 100.0 * _BATT_CAP_WH
            # Prefer the REAL simulated motor energy (integrated directly
            # from physics every substep in forward_sim.py) over the old
            # circular back-out (motor_wh = drain_wh + solar_wh), which was
            # mathematically forced to equal solar_wh whenever drain_wh
            # happened to be 0 -- exactly what happens on every day that
            # clips at the SOC ceiling (100% -> 100%), making the printed
            # figure meaningless on precisely those days.
            _real_motor_wh = (profiles.get(d_idx, {}).get("motor_energy_wh")
                               if profiles else None)
            motor_wh = _real_motor_wh if _real_motor_wh else (drain_wh + solar_wh)
            # Real SOC curve from the actual per-substep trace (not a linear
            # start->end approximation, which hid the mid-day peaks — e.g. a day
            # that briefly hit 100% and clipped solar looked like a smooth
            # decline). Sample ~9 points along the real trajectory.
            t_window = rc.day_finish_time_s(d_idx) - rc.day_start_time_s(d_idx)
            n_hours = max(1, int(t_window / 3600))
            _soc_tr = ((profiles.get(d_idx, {}) or {}).get("dashboard_trace", {}) or {}).get("soc_pct", [])
            if _soc_tr and len(_soc_tr) >= 2:
                _n = len(_soc_tr)
                _idxs = [int(round(i * (_n - 1) / 8)) for i in range(9)]
                soc_str = " → ".join(f"{float(_soc_tr[i]):.0f}%" for i in _idxs)
                _peak = max(float(x) for x in _soc_tr)
                soc_str += f"   (peak {_peak:.0f}%)"
                print(f"  3) SOC curve (real): {soc_str}")
            else:
                hourly_soc = [soc_start - (soc_drain_pct * h / n_hours) for h in range(n_hours + 1)]
                soc_str = " → ".join(f"{s:.0f}%" for s in hourly_soc[::max(1, len(hourly_soc)//5)])
                print(f"  3) SOC curve (approx): {soc_str}")

            # 4. Solar — GROSS captured by the panel vs what the battery could
            #    actually STORE. The difference is clipped at the SOC ceiling
            #    (the battery was already full — most acute on days that start
            #    near 100%). Showing all three is what makes the ledger
            #    reconcile: stored_solar - motor ≈ battery delta.
            underutil_wh = float((profiles.get(d_idx, {}) or {}).get("solar_underutil_wh", 0.0) or 0.0)
            stored_solar_wh = max(0.0, solar_wh - underutil_wh)
            if underutil_wh > 1.0:
                print(f"  4) Solar: {solar_wh:.0f} Wh gross | {underutil_wh:.0f} Wh WASTED at SOC ceiling "
                      f"| {stored_solar_wh:.0f} Wh stored")
            else:
                print(f"  4) Solar: {solar_wh:.0f} Wh captured (no ceiling clipping)")

            # 5. Energy consumption — reconciled ledger.
            #    stored_solar - motor should ≈ battery delta (-drain). Any
            #    residual is stationary-charge credit + charge/discharge
            #    efficiency, both small.
            net_wh = stored_solar_wh - motor_wh
            print(f"  5) Motor energy: {motor_wh:.0f} Wh | Battery {'drain' if drain_wh>=0 else 'GAIN'}: "
                  f"{abs(drain_wh):.0f} Wh ({-soc_drain_pct:+.1f}%)")
            print(f"     Ledger: stored solar {stored_solar_wh:.0f} − motor {motor_wh:.0f} = {net_wh:+.0f} Wh "
                  f"(≈ battery {'gain' if net_wh>=0 else 'drain'})")

            # 7. Morning charge onto the NEXT day (06:30 -> race start, safety-
            #    gated: skipped if this day ends high, capped to the safe band,
            #    extended by any late-finish penalty hold).
            _pdict = (profiles.get(d_idx, {}) or {}) if profiles else {}
            _morn = _pdict.get("morning_charge_pct")
            _latepen = _pdict.get("late_penalty_min", 0) or 0
            _nextstart = _pdict.get("next_start_soc_pct")
            if _morn is not None and d_idx < 7:
                if _morn <= 0.05:
                    reason = "skipped (pack already high — battery safety)" if soc_end >= 90.0 else "≈0 (little morning sun)"
                    print(f"  7) Morning charge -> Day {d_idx+2}: {reason}. Next day starts ~{_nextstart:.0f}%")
                else:
                    extra = f" (+ {_latepen} min penalty-hold charging)" if _latepen > 0 else ""
                    print(f"  7) Morning charge -> Day {d_idx+2}: +{_morn:.1f}% from 06:30{extra}. Next day starts ~{_nextstart:.0f}%")
            else:
                print(f"  7) Start strategy: Normal start at {_clock(rc.day_start_time_s(d_idx))}")

            # 8. ETA — TRUE arrival = start + prior-day penalty hold + drive +
            # mandatory stops (control + per-loop). total_time is drive-only.
            t_start_abs = rc.day_start_time_s(d_idx)
            if total_time > 0 and profiles and d_idx in profiles:
                p = profiles[d_idx]
                _stop_s = float(p.get("stop_time_s", _day_mandatory_stop_s(n_loops)))
                _inh_s = float(p.get("inherited_penalty_min", 0) or 0) * 60.0
                _finish_abs = p.get("finish_abs_s")
                if _finish_abs is None:
                    _finish_abs = t_start_abs + total_time + _stop_s + _inh_s
                _ctrl_s = float(p.get("control_stop_s", 0.0))
                _loop_s = float(p.get("loop_stop_s", 0.0))
                _inh_note = f" + {int(_inh_s//60)}min inherited penalty" if _inh_s > 0 else ""
                print(f"  8) ETA: {_clock(float(_finish_abs))}  "
                      f"(drive {_hms(total_time)} + stops {_hms(_stop_s)} "
                      f"[control {int(_ctrl_s//60)}m + loops {int(_loop_s//60)}m]{_inh_note})")
            elif total_time > 0:
                print(f"  8) ETA: {_clock(t_start_abs + total_time)} (drive time {_hms(total_time)})")
            else:
                avg_speed_kmh = day_km / (t_window / 3600) if t_window > 0 else 40.0
                est_time = day_km / avg_speed_kmh * 3600
                print(f"  8) ETA: ~{_clock(t_start_abs + est_time)} (est. drive time {_hms(est_time)})")

            # 9. Trailering
            if profiles and d_idx in profiles:
                t_km = profiles[d_idx].get("trailered_km", 0.0) or 0.0
                t_sub = profiles[d_idx].get("trailered_substeps", 0) or 0
                if t_km > 0:
                    print(f"  9) Trailered: {t_km:.1f} km ({t_sub} segments) — no power in/out")
                else:
                    print(f"  9) Trailered: none")
            else:
                print(f"  9) Trailered: n/a")

            # Speed summary
            if v_arr is not None:
                v_np = np.asarray(v_arr)
                print(f"      Speed: avg {v_np.mean():.1f} km/h, min {v_np.min():.1f}, max {v_np.max():.1f}")

    print("\n" + "=" * 70)

    # ------------------------------------------------------------------ #
    # 8.  Save durable JSON output for dashboard
    # ------------------------------------------------------------------ #
    import datetime as _dt

    output_dir = os.path.abspath(os.path.join(current_dir, "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    for variant_name, data in all_results.items():
        result   = data["result"]
        profiles = data["profiles"]
        plans    = data["plans"]

        json_out = {
            "variant": variant_name,
            "timestamp": _dt.datetime.now().isoformat(),
            "converged": result.get("converged"),
            "feasible": result.get("feasible"),
            "iterations": result.get("iterations"),
            "total_distance_km": result.get("total_distance_km", 0),
            "history": result.get("history", []),
            "days": {},
        }

        if result.get("feasible"):
            loop_plan = result.get("loop_plan", {})
            s_start = result.get("s_start_pct")
            _json_real_total_km = 0.0
            _json_total_trailered_km = 0.0

            from collections import Counter as _Counter2
            def _json_lp(d_idx):
                if profiles and d_idx in profiles and profiles[d_idx].get("loops_committed") is not None:
                    return dict(_Counter2(nm for nm, _km in profiles[d_idx]["loops_committed"]))
                return loop_plan.get(d_idx, {})

            for d_idx in range(8):
                plan = plans[d_idx]
                lp = _json_lp(d_idx)
                n_loops = sum(lp.values()) if lp else 0
                loop_km = sum(cnt * km for (name, km) in plan.loops
                              for cnt in [lp.get(name, 0)])
                day_km = _real_day_km(d_idx, plan, lp, profiles)
                soc_start = (float(profiles[d_idx]["start_soc_pct"])
                         if profiles and d_idx in profiles and profiles[d_idx].get("start_soc_pct") is not None
                         else (float(s_start[d_idx]) if s_start is not None and np.isfinite(s_start[d_idx]) else 0.0))

                trailered_km_day = _trailered_km_for(d_idx, profiles)
                day_data = {
                    "route": DAY_NAMES.get(d_idx, f"Day {d_idx + 1}"),
                    # Authoritative, reconciling distance:
                    #   distance_km == stage1_km + stage2_km + loop_km - trailered_km
                    "distance_km": round(day_km, 1),
                    "stage1_km": round(plan.stage1_km, 1),
                    "stage2_km": round(plan.stage2_km, 1),
                    "loop_km": round(loop_km, 1),
                    "trailered_km": round(trailered_km_day, 1),
                    # Raw forward_sim driven distance (KML linestring + loop
                    # separators) — for transparency; may differ from the
                    # published headline by ~1 km. Do NOT sum this for scoring.
                    "distance_km_simulated": round(_simulated_day_km(d_idx, profiles), 1),
                    "loops": {name: cnt for name, cnt in lp.items()} if lp else {},
                    "n_loops": n_loops,
                    "soc_start_pct": round(soc_start, 1),
                    "morning_charge_pct": (profiles.get(d_idx, {}) or {}).get("morning_charge_pct", 0.0),
                    "late_penalty_min": (profiles.get(d_idx, {}) or {}).get("late_penalty_min", 0),
                    "next_start_soc_pct": (profiles.get(d_idx, {}) or {}).get("next_start_soc_pct"),
                }

                solar_wh = float((profiles.get(d_idx, {}) or {}).get("solar_energy_wh", 0.0) or 0.0)
                underutil_wh = float((profiles.get(d_idx, {}) or {}).get("solar_underutil_wh", 0.0) or 0.0)
                day_data["solar_input_wh"] = round(solar_wh, 0)          # gross panel output
                day_data["solar_underutil_wh"] = round(underutil_wh, 0)  # clipped at SOC ceiling
                day_data["solar_stored_wh"] = round(max(0.0, solar_wh - underutil_wh), 0)  # actually banked

                if profiles and d_idx in profiles:
                    p = profiles[d_idx]
                    soc_end = p.get("end_soc_pct", 0.0) or 0.0
                    drive_t = p.get("total_time_s")
                    if drive_t is None:
                        t_arr = p.get("time_array_s")
                        if t_arr is not None and hasattr(t_arr, '__iter__'):
                            drive_t = float(max(t_arr)) - rc.day_start_time_s(d_idx)
                        else:
                            drive_t = 0.0
                    v_arr = p.get("velocity_profile_kmh")
                    day_data["soc_end_pct"] = round(soc_end, 1)
                    day_data["drive_time_s"] = round(drive_t, 0)
                    # ETA = start + prior-day penalty hold + drive + mandatory
                    # stops (control + per-loop). Prefer the extractor's true
                    # finish; fall back to computing it here for older profiles.
                    _stop_s = float(p.get("stop_time_s",
                        _day_mandatory_stop_s(len(p.get("loops_committed") or []))))
                    _inh_s = float(p.get("inherited_penalty_min", 0) or 0) * 60.0
                    _finish_abs = p.get("finish_abs_s")
                    if _finish_abs is None:
                        _finish_abs = rc.day_start_time_s(d_idx) + drive_t + _stop_s + _inh_s
                    day_data["stop_time_s"] = round(_stop_s, 0)
                    day_data["control_stop_s"] = round(float(p.get("control_stop_s", 0.0)), 0)
                    day_data["loop_stop_s"] = round(float(p.get("loop_stop_s", 0.0)), 0)
                    day_data["eta"] = _clock(float(_finish_abs))       # TRUE arrival (incl. stops)
                    day_data["eta_drive_only"] = _clock(rc.day_start_time_s(d_idx) + drive_t)
                    if v_arr is not None:
                        v_np = np.asarray(v_arr)
                        day_data["speed_avg_kmh"] = round(float(v_np.mean()), 1)
                        day_data["speed_min_kmh"] = round(float(v_np.min()), 1)
                        day_data["speed_max_kmh"] = round(float(v_np.max()), 1)
                        day_data["velocity_profile_kmh"] = [int(v) for v in v_arr]
                    # Continuous distance-indexed curves for the dashboard
                    # (SOC / velocity / solar / gradient vs distance). Coarse
                    # velocity_profile_kmh above stays the driver card.
                    dash = p.get("dashboard_trace")
                    if dash:
                        day_data["dashboard_trace"] = dash

                    # Per-stage breakdown (stage1 / loop / stage2). Same summary
                    # metrics as the day PLUS a per-stage plotting trace
                    # (distance/velocity/solar/soc/slope), split by route stage.
                    # Exposed two ways for the dashboard:
                    #   * "stages": the ordered list (as the route runs);
                    #   * "stage1"/"loop"/"stage2": direct keys, each the stage
                    #     dict or NULL when that stage doesn't exist that day
                    #     (e.g. no loop, or Day-3-aryaman with no stage1). So the
                    #     dashboard can always read day["stage1"] etc. safely.
                    _stage_list = p.get("stages", []) or []
                    day_data["stages"] = _stage_list
                    day_data["stage_names"] = [s["stage"] for s in _stage_list]
                    _by_name = {s["stage"]: s for s in _stage_list}
                    for _sk in ("stage1", "loop", "stage2"):
                        day_data[_sk] = _by_name.get(_sk)   # dict or None

                if profiles and d_idx in profiles:
                    day_data["trailered_km"] = round(profiles[d_idx].get("trailered_km", 0.0) or 0.0, 1)
                    day_data["trailered_substeps"] = profiles[d_idx].get("trailered_substeps", 0) or 0

                soc_drain = soc_start - day_data.get("soc_end_pct", soc_start)
                drain_wh = soc_drain / 100.0 * _BATT_CAP_WH
                day_data["battery_drain_wh"] = round(drain_wh, 0)
                day_data["battery_drain_pct"] = round(soc_drain, 1)
                # Prefer real simulated motor energy over the circular
                # back-out (see the print-loop fix above for why).
                _real_motor_wh_json = (profiles.get(d_idx, {}).get("motor_energy_wh")
                                        if profiles else None)
                day_data["motor_energy_wh"] = round(
                    float(_real_motor_wh_json) if _real_motor_wh_json is not None else 0.0, 0)

                _json_real_total_km += day_km
                _json_total_trailered_km += day_data.get("trailered_km", 0.0)

                json_out["days"][str(d_idx + 1)] = day_data

            # Real total: driven km only, trailered km excluded (SR asterisk
            # rule). Overwrite the DP planning estimate that was set above.
            json_out["total_distance_km"] = round(_json_real_total_km, 1)
            json_out["total_trailered_km"] = round(_json_total_trailered_km, 1)
            json_out["total_distance_km_dp_estimate"] = result.get("total_distance_km", 0)

        out_path = os.path.join(output_dir, f"strategy_{variant_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_out, f, indent=2, default=str)
        logger.info("Saved strategy → %s", out_path)

    logger.info("All variant results saved to %s/", output_dir)