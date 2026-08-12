"""
optimizers/hierarchical/tier2.py — Tier 2 high-fidelity local sampler.
"""

from __future__ import annotations
from tqdm import tqdm

import inspect
import logging
import multiprocessing as mp
import numpy as np

from configs import race_config as rc
from configs.car_config import CarState
from optimizers import singleday
from .tier1 import _DayPlan
from .tier1 import relaxed_loop_combos 

logger = logging.getLogger(__name__)

SOC_OFFSETS_PCT = (-4.0,-2.0,0.0,2.0,4.0) 
SAMPLE_WINDOW_PCT = max(abs(o) for o in SOC_OFFSETS_PCT)

_L2_WARMSTART_KW = "warm_start_kmh"

def _ordered_combos(plan: _DayPlan, day_index: int, car: CarState, is_today: bool, elapsed_s: float):
    loop_speed_ms = max(getattr(rc, "LOOP_CRUISE_SPEED_MS", car.v_max_ms), 1e-6)
    turnaround_s = getattr(rc, "LOOP_TURNAROUND_S", 0.0)
    pre_attempt_stop_s = rc.LOOP_STOP_DURATION_S + turnaround_s
    t_window = rc.day_finish_time_s(day_index) - rc.day_start_time_s(day_index)
    if is_today:
        t_window -= elapsed_s
    t_stops_base = rc.UNPLANNED_STOP_BUDGET_S + rc.CONTROL_STOP_DURATION_S

    if rc.RACE_MODE == "completion":
        combos = [((0,) * len(plan.loops), 0.0)]
    else:
        combos = list(relaxed_loop_combos(
            plan, max(0.0, t_window), t_stops_base, loop_speed_ms, pre_attempt_stop_s))

    def _loop_time(item):
        reps, _km = item
        return sum(
            n * ((plan.loops[i][1] * 1000.0) / loop_speed_ms + pre_attempt_stop_s)
            for i, n in enumerate(reps)) if reps else 0.0

    return sorted(combos, key=_loop_time)

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
    
    d_dist = task.get("dist_done_km", 0.0)
    d_elap = task.get("elapsed_s", 0.0)
    d_cs = task.get("cs_taken", False)

    out: dict[tuple, tuple] = {}
    warm = None  
    for reps, _loop_km in tqdm(ordered, desc=f"day {day_index} combos", leave=False):
        loops_committed = _reps_to_committed(plan, reps)

        kwargs = dict(global_method=global_method, seed=seed)
        if _L2_WARMSTART_KW is not None and warm is not None:
            kwargs[_L2_WARMSTART_KW] = warm

        res = singleday.solve(route, car, solar_provider, wind_provider,
                              day_index, start_soc, alpha_next,
                              loops_committed, 
                              dist_done_km=d_dist, elapsed_s=d_elap, cs_taken=d_cs, 
                              **kwargs)

        if not _l2_result_feasible(res, car, day_index):
            break

        out[tuple(reps)] = (start_soc, float(res["final_soc_pct"]))
        warm = res.get("v_kmh")

    return dict(offset_soc=start_soc, points=out)

def _reps_to_committed(plan: _DayPlan, reps: tuple[int, ...]):
    committed = []
    for i, (name, km) in enumerate(plan.loops):
        committed.extend([(name, km)] * (reps[i] if reps else 0))
    return committed

class LinearSurrogate:
    __slots__ = ("a", "b", "s0", "loop_km", "reps", "soc_lo", "soc_hi")

    def __init__(self, a, b, s0, loop_km, reps, soc_lo, soc_hi):
        self.a = a; self.b = b; self.s0 = s0
        self.loop_km = loop_km; self.reps = reps
        self.soc_lo = soc_lo; self.soc_hi = soc_hi

    def predict(self, start_soc: float) -> float:
        return self.a + self.b * (start_soc - self.s0)

    def in_window(self, start_soc: float) -> bool:
        return self.soc_lo - 1e-9 <= start_soc <= self.soc_hi + 1e-9

def _fit_surrogates(points_by_combo: dict, s0: float, plan: _DayPlan):
    surro = {}
    for reps, pts in points_by_combo.items():
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        if len(np.unique(xs)) >= 2:
            b, a_at_zero = np.polyfit(xs - s0, ys, 1)  
            a = a_at_zero
        else:
            a, b = float(ys[0]), 0.0
        loop_km = sum((reps[i] if reps else 0) * km for i, (_n, km) in enumerate(plan.loops))
        surro[tuple(reps)] = LinearSurrogate(
            a=float(a), b=float(b), s0=s0, loop_km=loop_km, reps=tuple(reps),
            soc_lo=float(xs.min()), soc_hi=float(xs.max()))
    return surro

def sample_day(route, car: CarState, solar_provider, wind_provider,
               day_index: int, plan: _DayPlan, s0_pct: float,
               alpha_next_day_pct: float, *, parallel: bool = True,
               n_workers: int | None = None, global_method: str = "ga",
               seed: int | None = None,
               offsets_pct=SOC_OFFSETS_PCT, 
               is_today: bool = False, dist_done_km: float = 0.0, 
               elapsed_s: float = 0.0, cs_taken: bool = False) -> dict:

    ordered = _ordered_combos(plan, day_index, car, is_today, elapsed_s)
    start_socs = [float(np.clip(s0_pct + o, car.soc_min_pct, car.soc_max_pct))
                  for o in offsets_pct]

    tasks = [dict(route=route, car=car, solar_provider=solar_provider,
                  wind_provider=wind_provider, day_index=day_index, plan=plan,
                  start_soc_pct=s, alpha_next_day_pct=alpha_next_day_pct,
                  ordered_combos=ordered, global_method=global_method, seed=seed,
                  dist_done_km=dist_done_km, elapsed_s=elapsed_s, cs_taken=cs_taken)
             for s in start_socs]

    parallel = False   #Set this to TRUE if possible to use parallel processing

    if parallel and len(tasks) > 1:
        with mp.Pool(processes=n_workers or min(len(tasks), mp.cpu_count())) as pool:
            sweeps = pool.map(_sweep_one_offset, tasks)
    else:
        sweeps = [_sweep_one_offset(t) for t in tasks]

    points_by_combo: dict[tuple, list] = {}
    seen_keys: set = set()
    n_solves = 0
    for sw in sweeps:
        for reps, (ss, es) in sw["points"].items():
            key = (day_index, reps, round(ss, 1))
            if key in seen_keys: continue
            seen_keys.add(key)
            points_by_combo.setdefault(reps, []).append((ss, es))
            n_solves += 1

    surrogates = _fit_surrogates(points_by_combo, s0_pct, plan)
    return dict(surrogates=surrogates, s0_pct=s0_pct, n_l2_solves=n_solves)

def sample_all_days(routes: list, car: CarState, solar_providers: dict, wind_providers: dict,
                    s0_traj: np.ndarray, plans: list, alpha_next_pct: dict,
                    start_day: int = 0, dist_done_km: float = 0.0, elapsed_s: float = 0.0,
                    cs_taken: bool = False, loops_done: dict | None = None,
                    **kwargs) -> dict:
    per_day = {}
    for d in tqdm(range(start_day, len(plans)), desc="Tier2 days"):
        route = routes[d] if routes and d < len(routes) else None
        alpha = alpha_next_pct.get(d, car.soc_min_pct)

        solar_provider = solar_providers.get(d)
        wind_provider = wind_providers.get(d)
        
        is_today = (d == start_day)
        d_dist = dist_done_km if is_today else 0.0
        d_elap = elapsed_s if is_today else 0.0
        d_cs = cs_taken if is_today else False
        d_loops = loops_done if is_today else None
        
        nom_plan = plans[d]
        if is_today and d_dist > 0:
            from .tier1 import _adjust_plan_for_today
            plan_to_use = _adjust_plan_for_today(nom_plan, d_dist, d_loops)
        else:
            plan_to_use = nom_plan

        per_day[d] = sample_day(
            route, car, solar_provider, wind_provider, d, plan_to_use,
            float(s0_traj[d]), alpha, is_today=is_today, dist_done_km=d_dist, 
            elapsed_s=d_elap, cs_taken=d_cs, **kwargs)
    return per_day