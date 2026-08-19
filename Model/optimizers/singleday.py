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

# Import the centralized forward integrator
from simulator import forward_sim

logger = logging.getLogger(__name__)

# ===============================================================================
# 0. Local config overrides
# ===============================================================================

CONTROL_SEGMENT_M = SCFG.CONTROL_SEGMENT_M

SHARP_TURN_HEADING_DELTA_DEG = 30.0      
SHARP_TURN_SPEED_LIMIT_KMH = 20.0       

DE_POPSIZE = 8                            
DE_MAXITER = 60                           


# ===============================================================================
# 1. Sharp-turn speed caps
# ===============================================================================

def _sharp_turn_fraction(route: Route, seg_start_m: np.ndarray, seg_len_m: float,
                          heading_delta_threshold_deg: float) -> np.ndarray:
    """Fraction of each control segment's length flagged as a sharp turn."""
    if not route: return np.zeros(len(seg_start_m))
    x = route.df["distance_m"].to_numpy()
    bearing = route.df["bearing_deg"].to_numpy()
    raw_delta = np.diff(bearing, prepend=bearing[0])
    wrapped = (raw_delta + 180.0) % 360.0 - 180.0
    sharp_point = np.abs(wrapped) >= heading_delta_threshold_deg

    seg_end_m = seg_start_m + seg_len_m
    frac = np.zeros(len(seg_start_m))
    for i, (s, e) in enumerate(zip(seg_start_m, seg_end_m)):
        in_seg = (x >= s) & (x < e)
        n = int(np.sum(in_seg))
        if n > 0:
            frac[i] = float(np.sum(sharp_point[in_seg])) / n
    return frac


def apply_turn_speed_caps(route: Route, v_max_kmh: np.ndarray,
                           seg_start_m: np.ndarray,
                           seg_len_m: float = CONTROL_SEGMENT_M,
                           heading_delta_threshold_deg: float = SHARP_TURN_HEADING_DELTA_DEG,
                           turn_speed_limit_kmh: float = SHARP_TURN_SPEED_LIMIT_KMH,
                           ) -> np.ndarray:
    """Blend v_max toward turn_speed_limit_kmh in proportion to how much of the
    segment is actually a sharp turn, instead of capping the whole segment for
    a single flagged point."""
    frac = _sharp_turn_fraction(route, seg_start_m, seg_len_m, heading_delta_threshold_deg)
    v_eff = 1.0 / (frac / turn_speed_limit_kmh + (1.0 - frac) / np.maximum(v_max_kmh, 1e-6))
    return v_eff


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
                   loops_committed: list[tuple[str, float]]) -> Route:
    """Build the real simulated route for a day with committed loop reps.

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

    # CRASH FIX (Day 6): the old guard only handled "both stage1 and stage2
    # empty". Some single-file days get tagged "stage2" instead of "stage1"
    # depending on the source filename — Day 6's control-stop location ==
    # finish location (race_config.py), so it's a single leg with no
    # separate Stage 1 file to disambiguate the name against, and its file
    # apparently reads as "Stage 2". That left stage1 empty / stage2
    # populated, a case the old guard never caught, so the old code crashed
    # on stage1.iloc[-1] (IndexError: single positional indexer is
    # out-of-bounds) the moment a real combo (non-empty loops_committed) hit
    # this day. Fix: treat whichever block is actually populated as the
    # "pre-loop" content, instead of assuming stage1 specifically is always
    # the non-empty one.
    if len(stage1) == 0 and len(stage2) > 0:
        pre, post = stage2, stage1
    elif len(stage1) == 0 and len(stage2) == 0:
        pre, post = base_df.copy(), base_df.iloc[0:0].copy()
    else:
        pre, post = stage1, stage2

    blocks = [pre]
    pre_end_m = float(pre["distance_m"].max()) if len(pre) else 0.0
    offset = pre_end_m

    _LOOP_SEPARATOR_M = 300.0  # small buffer so each rep gets its own stop-dwell cycle

    def _separator_row(at_m: float) -> pd.DataFrame:
        # Anchor to whatever block was most recently appended — never assume
        # a specific named block ("stage1") is guaranteed non-empty (that
        # assumption is exactly what crashed on Day 6).
        row = blocks[-1].iloc[[-1]].copy()
        row["distance_m"] = at_m
        row["seg_type"] = "stage1"  # non-loop, resets the contiguous-zone flag
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

    def _objective(v_kmh: np.ndarray) -> float:
        res = evaluator(v_kmh)
        solar_penalty_s = _w * float(res.solar_underutil_j) / 3600.0
        return float(res.total_time_s) + solar_penalty_s
    return _objective

def _terminal_soc_constraint(evaluator: DayEvaluator,
                              alpha_next_day_pct: float) -> NonlinearConstraint:
    return NonlinearConstraint(
        lambda v: evaluator(v).final_soc_pct - alpha_next_day_pct,
        lb=0.0, ub=np.inf,
    )

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
               seed: int | None = None) -> GlobalSearchResult: ...

class DifferentialEvolutionSearch:
    def __init__(self, popsize: int = DE_POPSIZE, maxiter: int = DE_MAXITER,
                 strategy: str = "best1bin", mutation=(0.5, 1.0),
                 recombination: float = 0.7):
        self.popsize = popsize
        self.maxiter = maxiter
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination

    def search(self, objective, bounds, constraints, seed=None) -> GlobalSearchResult:
        result = differential_evolution(
            objective, bounds,
            strategy=self.strategy, popsize=self.popsize, maxiter=self.maxiter,
            mutation=self.mutation, recombination=self.recombination,
            constraints=tuple(constraints), polish=False, seed=seed, tol=1e-6,
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

    def search(self, objective, bounds, constraints, seed=None) -> GlobalSearchResult:
        rng = np.random.default_rng(seed)
        lb, ub = np.asarray(bounds.lb), np.asarray(bounds.ub)
        dim = lb.size

        pop = rng.uniform(lb, ub, size=(self.population, dim))
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
          **kwargs):

    # ROOT-CAUSE FIX: splice committed loop reps into the real simulated
    # route (see section 1b above) instead of the old behaviour where
    # loops_committed only subtracted stop-time and never touched the
    # physics at all. loop_geoms is {loop_name: DataFrame} for this day,
    # threaded down from trust_region.py -> tier2.py. If not provided
    # (e.g. an old call site not yet updated), loops still get a flat
    # synthetic geometry via _splice_loops rather than silently costing
    # nothing — real distance/time/energy either way.
    sim_route = route
    if loops_committed and route is not None:
        sim_route = _splice_loops(route, loop_geoms, loops_committed)

    rem_m = (sim_route.total_m - dist_done_km * 1000.0) if sim_route else 0.0
    n_segments = max(1, int(np.ceil(rem_m / CONTROL_SEGMENT_M)))
    
    seg_start_m = (dist_done_km * 1000.0) + np.arange(n_segments) * CONTROL_SEGMENT_M

    v_max_kmh = sim_route.v_max_ms_at(seg_start_m) * 3.6 if sim_route else np.full(n_segments, car.v_max_ms * 3.6)
    if sim_route:
        v_max_kmh = apply_turn_speed_caps(sim_route, v_max_kmh, seg_start_m)

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
    )  

    constraints = [
        _terminal_soc_constraint(evaluator, alpha_next_day_pct),
        _time_cutoff_constraint(evaluator, allowed_time_s),
    ]

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
            global_result = global_search.search(objective, bounds, constraints, seed=seed)
    else:
        global_search = get_global_search(global_method)
        global_result = global_search.search(objective, bounds, constraints, seed=seed)

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
        diagnostics=dict(
            global_fun=global_result.fun,
            slsqp_fun=slsqp_result.fun,
            slsqp_success=slsqp_result.success,
        ),
    )