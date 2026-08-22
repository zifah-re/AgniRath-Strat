"""
optimizers/hierarchical/tier2.py — Tier 2 high-fidelity local sampler.
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

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import inspect
import logging
import numpy as np
import os
import time

from configs import race_config as rc
from configs.car_config import CarState


from optimizers import singleday
from . import _threads
from .tier1 import _DayPlan
from .tier1 import relaxed_loop_combos

logger = logging.getLogger(__name__)

# surrogate covers every start SOC Tier 3 might need.  Tier 1's
# s_center can be far from what singleday.solve actually produces,
# so narrow offsets around s_center leave the surrogate window too
# tight for Tier 3 to chain days together.
SAMPLE_WINDOW_PCT = 5.0          # convergence drift threshold
MAX_COMBOS = 7

_L2_WARMSTART_KW = "warm_start_kmh"

def _ordered_combos(plan: _DayPlan, day_index: int, car: CarState, is_today: bool, elapsed_s: float):
    loop_speed_ms = max(getattr(rc, "LOOP_CRUISE_SPEED_MS", car.v_max_ms), 1e-6)
    turnaround_s = getattr(rc, "LOOP_TURNAROUND_S", 0.0)
    pre_attempt_stop_s = rc.LOOP_STOP_DURATION_S + turnaround_s
    t_window = rc.day_finish_time_s(day_index) - rc.day_start_time_s(day_index)
    if is_today:
        t_window -= elapsed_s
    t_stops_base = rc.UNPLANNED_STOP_BUDGET_S + rc.CONTROL_STOP_DURATION_S

    # Completion mode still has to optimize the actual race distance.
    # "completion" is a feasibility/finish requirement, not permission to
    # disable optional race loops. The old branch hard-coded zero loops, which
    # made every completion-mode day incapable of choosing a loop.
    combos = list(relaxed_loop_combos(
        plan, max(0.0, t_window), t_stops_base, loop_speed_ms, pre_attempt_stop_s))

    def _loop_time(item):
        reps, _km = item
        return sum(
            n * ((plan.loops[i][1] * 1000.0) / loop_speed_ms + pre_attempt_stop_s)
            for i, n in enumerate(reps)) if reps else 0.0

    combos_sorted = sorted(combos, key=_loop_time)
    if len(combos_sorted) <= MAX_COMBOS:
        return combos_sorted

    positive = [i for i, item in enumerate(combos_sorted) if sum(item[0]) > 0]
    required = [0, positive[0] if positive else 0, len(combos_sorted) - 1]
    interior = np.linspace(0, len(combos_sorted) - 1, MAX_COMBOS, dtype=int).tolist()
    idxs = sorted(set(required + interior))
    if len(idxs) < MAX_COMBOS:
        for i in range(len(combos_sorted)):
            if i not in idxs:
                idxs.append(i)
                if len(idxs) == MAX_COMBOS:
                    break
    return [combos_sorted[i] for i in sorted(idxs[:MAX_COMBOS])]


def _l2_result_feasible(res: dict, car: CarState, day_index: int) -> bool:
    if res is None: return False
    if res.get("final_soc_pct", -np.inf) < car.soc_min_pct: return False
    finish = rc.day_start_time_s(day_index) + res.get("total_time_s", np.inf)
    return finish <= rc.FINISH_CUTOFF_ABS_S

def _sweep_one_offset(task: dict) -> dict:
    route = task["route"]
    car: CarState = task["car"]
    solar_provider = task["solar_provider"]
    wind_provider = task["wind_provider"]
    day_index = task["day_index"]
    plan = task["plan"]
    start_soc = task["start_soc_pct"]
    alpha_next = task["alpha_next_day_pct"]
    ordered = task["ordered_combos"]
    global_method = task["global_method"]
    seed = task["seed"]
    # Real per-loop-name route geometry for this day (Root-cause fix: lets
    # singleday.solve() actually simulate committed loop reps instead of
    # only subtracting stop-time for them). May be None/missing entries —
    # singleday._splice_loops() falls back to a flat synthetic leg per name
    # that has no matching .save geometry.
    loop_geoms = task.get("loop_geoms")

    d_dist = task.get("dist_done_km", 0.0)
    d_elap = task.get("elapsed_s", 0.0)
    d_cs = task.get("cs_taken", False)

    out: dict[tuple, tuple] = {}
    underutil_by_combo: dict[tuple, float] = {}
    finish_by_combo: dict[tuple, float] = {}
    # Cross-offset warm-starting: accept an initial warm seed from a
    # previous offset's best result (avoids a full GA run on offset 2).
    warm = task.get("initial_warm_kmh")
    method_log: list[str] = []
    for reps, _loop_km in tqdm(ordered, desc=f"day {day_index} combos", leave=False):
        loops_committed = _reps_to_committed(plan, reps)

        kwargs = dict(global_method=global_method, seed=seed)
        if _L2_WARMSTART_KW is not None and warm is not None:
            kwargs[_L2_WARMSTART_KW] = warm

        t0 = time.perf_counter()
        res = singleday.solve(route, car, solar_provider, wind_provider,
                              day_index, start_soc, alpha_next,
                              loops_committed,
                              dist_done_km=d_dist, elapsed_s=d_elap, cs_taken=d_cs,
                              loop_geoms=loop_geoms,
                              **kwargs)
        dt = time.perf_counter() - t0

        used = str(res.get("global_method", "ga"))
        method_log.append(used)
        logger.info("  day %d  SOC=%.1f%%  reps=%s  method=%s  %.1fs",
                     day_index, start_soc, reps, used, dt)

        if not _l2_result_feasible(res, car, day_index):
            continue  # try other combos — don't assume monotonic infeasibility

        key = tuple(reps)
        out[key] = (start_soc, float(res["final_soc_pct"]))
        underutil_by_combo[key] = float(res.get("solar_underutil_wh", 0.0) or 0.0)
        # Absolute clock finish time (seconds since midnight) for this combo at
        # this start SOC. Threaded up into the surrogate so tier3 can price
        # late finishes (SR 2.22.6). total_time_s is the drive duration; adding
        # the day's start gives the wall-clock arrival used everywhere else.
        finish_by_combo[key] = (rc.day_start_time_s(day_index)
                                + float(res.get("total_time_s", 0.0) or 0.0))
        warm = res.get("v_kmh")

    return dict(offset_soc=start_soc, points=out,
                underutil_by_combo=underutil_by_combo,
                finish_by_combo=finish_by_combo,
                best_warm_kmh=warm, method_log=method_log)

def _reps_to_committed(plan: _DayPlan, reps: tuple[int, ...]):
    committed = []
    for i, (name, km) in enumerate(plan.loops):
        committed.extend([(name, km)] * (reps[i] if reps else 0))
    return committed

class LinearSurrogate:
    __slots__ = ("a", "b", "s0", "loop_km", "reps", "soc_lo", "soc_hi",
                 "xs", "ys", "wu", "fs")

    def __init__(self, a, b, s0, loop_km, reps, soc_lo, soc_hi,
                 xs=None, ys=None, wu=None, fs=None):
        self.a = a; self.b = b; self.s0 = s0
        self.loop_km = loop_km; self.reps = reps
        self.soc_lo = soc_lo; self.soc_hi = soc_hi
        self.xs = None if xs is None else np.asarray(xs, dtype=float)
        self.ys = None if ys is None else np.asarray(ys, dtype=float)
        self.wu = None if wu is None else np.asarray(wu, dtype=float)
        # Absolute clock finish time (s since midnight) per sampled start SOC.
        self.fs = None if fs is None else np.asarray(fs, dtype=float)

    def predict(self, start_soc: float) -> float:
        # Never extrapolate a single successful L2 sample with a fabricated
        # SOC slope.  A one-point surrogate is valid only at that sampled SOC
        # (in_window() enforces this); two or more samples use interpolation.
        if self.xs is None or self.ys is None:
            raise RuntimeError("surrogate has no successful L2 sample")
        return float(np.interp(start_soc, self.xs, self.ys))

    def predict_underutil(self, start_soc: float) -> float:
        if self.xs is None or self.wu is None:
            raise RuntimeError("surrogate has no successful L2 sample")
        return max(0.0, float(np.interp(start_soc, self.xs, self.wu)))

    def predict_finish_s(self, start_soc: float) -> float:
        """Absolute clock finish time (s since midnight) at this start SOC.

        Returns NaN when no finish time was recorded (e.g. a Tier-1 fallback
        surrogate) so tier3 can skip late-finish pricing rather than guess.
        """
        if self.xs is None or self.fs is None:
            return float("nan")
        return float(np.interp(start_soc, self.xs, self.fs))

    def in_window(self, start_soc: float) -> bool:
        return self.soc_lo - 1e-9 <= start_soc <= self.soc_hi + 1e-9


def _fit_surrogates(points_by_combo: dict, s0: float, plan: _DayPlan):
    surro = {}
    for reps, pts in points_by_combo.items():
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        wu = np.array([p[2] if len(p) >= 3 else 0.0 for p in pts], dtype=float)
        fs = np.array([p[3] if len(p) >= 4 else np.nan for p in pts], dtype=float)
        order = np.argsort(xs)
        xs, ys, wu, fs = xs[order], ys[order], wu[order], fs[order]
        if len(np.unique(xs)) >= 2:
            b, a_at_zero = np.polyfit(xs - s0, ys, 1)
            a = a_at_zero
        else:
            a, b = float(ys[0]), 0.0
        loop_km = sum((reps[i] if reps else 0) * km for i, (_n, km) in enumerate(plan.loops))
        surro[tuple(reps)] = LinearSurrogate(
            a=float(a), b=float(b), s0=s0, loop_km=loop_km, reps=tuple(reps),
            soc_lo=float(xs.min()), soc_hi=float(xs.max()),
            # Keep even a one-point sample.  It becomes a degenerate, exact-SOC
            # surrogate rather than silently falling back to a linear model.
            xs=xs, ys=ys, wu=wu, fs=fs)
    return surro


def sample_day(route, car: CarState, solar_provider, wind_provider,
               day_index: int, plan: _DayPlan, s0_pct: float,
               alpha_next_day_pct: float, *, parallel: bool = True,
               n_workers: int | None = None, global_method: str = "ga",
               seed: int | None = None,
               offsets_pct=None,
               is_today: bool = False, dist_done_km: float = 0.0,
               elapsed_s: float = 0.0, cs_taken: bool = False,
               loop_geoms: dict | None = None) -> dict:

    ordered = _ordered_combos(plan, day_index, car, is_today, elapsed_s)

    # Low/mid/high SOC samples. Three anchors are needed for COVERAGE, not
    # curvature: a combo whose low-SOC sample comes back infeasible collapses
    # a 2-point surrogate to a single valid SOC (in_window() only accepts the
    # exact sampled SOC), and Tier 3 then can't allocate that day unless the
    # chained SOC lands exactly there — which shows up as a spurious
    # "infeasible" whole-race result. The midpoint keeps a usable window even
    # when one endpoint drops out. (Runtime is recovered from the forward_sim
    # vectorization + 150 m grid instead, which don't cost feasibility.)
    lo = float(np.clip(car.soc_min_pct + 5.0, car.soc_min_pct, car.soc_max_pct))
    hi = float(car.soc_max_pct)
    mid = 0.5 * (lo + hi)
    start_socs = sorted(set([lo, mid, hi]))

    base_task = dict(route=route, car=car, solar_provider=solar_provider,
                     wind_provider=wind_provider, day_index=day_index, plan=plan,
                     alpha_next_day_pct=alpha_next_day_pct,
                     ordered_combos=ordered, global_method=global_method, seed=seed,
                     dist_done_km=dist_done_km, elapsed_s=elapsed_s, cs_taken=cs_taken,
                     loop_geoms=loop_geoms)

    # Cross-offset warm-starting: run first SOC offset, extract its best
    # velocity profile, then seed the second offset with it.  This lets
    # offset 2 skip the GA entirely and go straight to warm SLSQP — the
    # single biggest per-day speedup (~50-60% faster on offset 2).
    sweeps = []
    cross_warm = None
    for s in start_socs:
        task = {**base_task, "start_soc_pct": s}
        if cross_warm is not None:
            task["initial_warm_kmh"] = cross_warm
        sw = _sweep_one_offset(task)
        sweeps.append(sw)
        # Propagate best v_kmh to seed next offset.
        if sw.get("best_warm_kmh") is not None:
            cross_warm = sw["best_warm_kmh"]

    points_by_combo: dict[tuple, list] = {}
    seen_keys: set = set()
    n_solves = 0
    for sw in sweeps:
        wu_map = sw.get("underutil_by_combo", {})
        fs_map = sw.get("finish_by_combo", {})
        for reps, (ss, es) in sw["points"].items():
            key = (day_index, reps, round(ss, 1))
            if key in seen_keys: continue
            seen_keys.add(key)
            points_by_combo.setdefault(reps, []).append(
                (ss, es, wu_map.get(reps, 0.0), fs_map.get(reps, np.nan)))
            n_solves += 1

    surrogates = _fit_surrogates(points_by_combo, s0_pct, plan)
    if n_solves == 0:
        # Diagnostic for the empty-surrogate failure mode: every combo at
        # every sampled SOC offset came back infeasible. That leaves Tier 3
        # with nothing to allocate and trust_region falls back to the Tier 1
        # linear surrogate — log exactly how this happened so the fallback is
        # visible instead of silently hiding an optimizer-level problem.
        logger.warning(
            "Tier2 Day %d: 0 L2 solves succeeded at sampled SOCs %s — all "
            "combos infeasible (time/SOC). Surrogates empty; trust_region "
            "will fall back to the Tier 1 linear surrogate.",
            day_index, [round(float(s), 1) for s in start_socs])
    return dict(surrogates=surrogates, s0_pct=s0_pct, n_l2_solves=n_solves)


def sample_all_days(routes: list, car: CarState, solar_providers: dict, wind_providers: dict,
                    s0_traj: np.ndarray, plans: list, alpha_next_pct: dict,
                    start_day: int = 0, dist_done_km: float = 0.0, elapsed_s: float = 0.0,
                    cs_taken: bool = False, loops_done: dict | None = None,
                    n_workers: int | None = None,
                    loop_geoms_by_day: dict | None = None,
                    **kwargs) -> dict:
    """Sample all days.  Uses ThreadPoolExecutor to parallelize across days.

    loop_geoms_by_day: {day_index: {loop_name: DataFrame}} — real loop route
    geometry per day, loaded once in trust_region.py and threaded down here
    so every singleday.solve() call for that day can actually simulate
    committed loop reps (root-cause fix for the "free loop reps" bug — see
    singleday.py section 1b for the full explanation).

    NOTE on threading: forward_sim's integrator is a Python for-loop (holds
    the GIL), so threads give limited CPU parallelism.  The speedup comes
    from scipy's compiled SLSQP Fortran internals which DO release the GIL.
    Expect ~20-40% wall-time reduction on multi-core machines.  True CPU
    parallelism needs Cython forward_sim or picklable objects for mp — a
    separate effort.
    """
    from .tier1 import _adjust_plan_for_today

    def _prepare_and_sample(d: int) -> tuple[int, dict]:
        route = routes[d] if routes and d < len(routes) else None
        alpha = alpha_next_pct.get(d, car.soc_min_pct)
        if not np.isfinite(alpha):
            logger.warning("Day %d: alpha_next is NaN, falling back to soc_min=%.1f%%",
                           d, car.soc_min_pct)
            alpha = car.soc_min_pct

        solar_provider = solar_providers.get(d)
        wind_provider = wind_providers.get(d)

        is_today = (d == start_day)
        d_dist = dist_done_km if is_today else 0.0
        d_elap = elapsed_s if is_today else 0.0
        d_cs = cs_taken if is_today else False
        d_loops = loops_done if is_today else None

        nom_plan = plans[d]
        if is_today and d_dist > 0:
            plan_to_use = _adjust_plan_for_today(nom_plan, d_dist, d_loops)
        else:
            plan_to_use = nom_plan

        s0_d = float(s0_traj[d])
        if not np.isfinite(s0_d):
            logger.warning("Day %d: s0_traj is NaN, falling back to 50%%", d)
            s0_d = 50.0

        d_loop_geoms = (loop_geoms_by_day or {}).get(d)

        t0 = time.perf_counter()
        result = sample_day(
            route, car, solar_provider, wind_provider, d, plan_to_use,
            s0_d, alpha, is_today=is_today, dist_done_km=d_dist,
            elapsed_s=d_elap, cs_taken=d_cs, loop_geoms=d_loop_geoms, **kwargs)
        dt = time.perf_counter() - t0
        logger.info("Tier2 day %d done in %.1fs  (%d solves)",
                     d, dt, result.get("n_l2_solves", 0))
        return d, result

    day_indices = list(range(start_day, len(plans)))
    n_days = len(day_indices)
    # Use threads: share memory (no pickling), scipy SLSQP releases GIL.
    # Cap workers at number of days to avoid idle threads. Worker count is
    # normalized through optimizers._threads.worker_cap (the single source
    # of truth), so an explicit n_workers from the caller (trust_region's
    # parallel= setting) is honored and clamped to os.cpu_count() — before
    # this fix the caller's n_workers landed in **kwargs and was silently
    # dropped, so every run used the same hardcoded default.
    n_workers = min(n_days, _threads.worker_cap(n_workers) or os.cpu_count() or 4)

    per_day = {}
    if n_days <= 1 or n_workers <= 1:
        # Single day or single core — skip thread overhead.
        for d in tqdm(day_indices, desc="Tier2 days"):
            _, result = _prepare_and_sample(d)
            per_day[d] = result
    else:
        logger.info("Tier2: sampling %d days with %d threads", n_days, n_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_prepare_and_sample, d): d for d in day_indices}
            with tqdm(total=n_days, desc="Tier2 days") as pbar:
                for fut in as_completed(futures):
                    d, result = fut.result()
                    per_day[d] = result
                    pbar.update(1)

    return per_day