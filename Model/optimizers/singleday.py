"""
optimizers/singleday.py — L2 single-day velocity optimizer

Mode-specific loss functions (charging/cruising/traffic) are intentionally
NOT included in this pass, per instruction — objective is plain end-of-day
SOC maximisation, matching Plan v3 §8's stated L2 objective.

Requires scipy>=1.14 (pinned in Model/requirements.txt) for NonlinearConstraint
objects to work directly with method="SLSQP" in `minimize`.
"""

from __future__ import annotations

from tqdm import tqdm
import typing as _t
import numpy as np
import logging
from scipy.optimize import Bounds, NonlinearConstraint, differential_evolution, minimize

from configs.car_config import CarState
from configs import solver_config as SCFG
from configs import race_config
from core.route import Route

# Import the centralized forward integrator
from simulator import forward_sim

logger = logging.getLogger(__name__)

# ===========================================================================
# 0. Local config overrides
# ===========================================================================

CONTROL_SEGMENT_M = SCFG.CONTROL_SEGMENT_M

SHARP_TURN_HEADING_DELTA_DEG = 30.0      
SHARP_TURN_SPEED_LIMIT_KMH = 20.0        

DE_POPSIZE = 8                            
DE_MAXITER = 60                           


# ===========================================================================
# 1. Sharp-turn speed caps
# ===========================================================================

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

# ===========================================================================
# 2. Day-level evaluation
# ===========================================================================

class DayEvaluator:
    """Runs one candidate speed vector through physics + timing via forward_sim."""
    def __init__(self, route: Route, car: CarState, solar_provider,
                 wind_provider, t0_s: float, start_soc_pct: float,
                 seg_start_m: np.ndarray, seg_len_m: float = CONTROL_SEGMENT_M,
                 energy_grid_m: float = SCFG.ENERGY_GRID_M):
        self.route = route
        self.car = car
        self.solar_provider = solar_provider
        self.wind_provider = wind_provider 
        self.t0_s = t0_s
        self.start_soc_pct = start_soc_pct
        self.seg_start_m = seg_start_m
        self.seg_len_m = seg_len_m
        self.energy_grid_m = energy_grid_m
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
            energy_grid_m=self.energy_grid_m
        )


# ===========================================================================
# 3. Objective / constraints
# ===========================================================================

def _build_objective(evaluator: DayEvaluator) -> _t.Callable[[np.ndarray], float]:
    def _objective(v_kmh: np.ndarray) -> float:
        return -evaluator(v_kmh).final_soc_pct
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


# ===========================================================================
# 4. Swappable global search
# ===========================================================================

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


# ===========================================================================
# 5. Integer km/h projection
# ===========================================================================

def project_to_integer_kmh(evaluator: DayEvaluator, v_kmh: np.ndarray,
                            v_max_kmh: np.ndarray, v_min_kmh: float = 5.0,
                            constraints: _t.Sequence[NonlinearConstraint] = (),
                            ) -> np.ndarray:
    v_int = np.clip(np.round(v_kmh), v_min_kmh, np.floor(v_max_kmh))

    def _feasible(v: np.ndarray) -> bool:
        return all(np.all(np.atleast_1d(c.fun(v)) >= -1e-6) for c in constraints)

    best = v_int.copy()
    best_obj = evaluator(best).final_soc_pct
    for i in range(len(best)):
        for step in (+1.0, -1.0):
            cand = best.copy()
            cand[i] = np.clip(cand[i] + step, v_min_kmh, np.floor(v_max_kmh[i]))
            if cand[i] == best[i] or not _feasible(cand):
                continue
            obj = evaluator(cand).final_soc_pct
            if obj > best_obj:
                best, best_obj = cand, obj
    return best


# ===========================================================================
# 6. solve() — frozen API
# ===========================================================================

def solve(route: Route, car: CarState, solar_provider, wind_provider,
          day_index: int, start_soc_pct: float, alpha_next_day_pct: float,
          loops_committed, global_method: str = "ga", seed: int | None = None,
          dist_done_km: float = 0.0, elapsed_s: float = 0.0, cs_taken: bool = False,
          **kwargs):
    
    rem_m = (route.total_m - dist_done_km * 1000.0) if route else 0.0
    n_segments = max(1, int(np.ceil(rem_m / CONTROL_SEGMENT_M)))
    
    seg_start_m = (dist_done_km * 1000.0) + np.arange(n_segments) * CONTROL_SEGMENT_M

    v_max_kmh = route.v_max_ms_at(seg_start_m) * 3.6 if route else np.full(n_segments, car.v_max_ms * 3.6)
    if route:
        v_max_kmh = apply_turn_speed_caps(route, v_max_kmh, seg_start_m)

    v_max_kmh = np.maximum(v_max_kmh, 5.0)    
    bounds = Bounds(lb=np.full(n_segments, 5.0), ub=v_max_kmh) 

    t0_s = race_config.day_start_time_s(day_index) + elapsed_s
    
    evaluator = DayEvaluator(route, car, solar_provider, wind_provider,
                              t0_s=t0_s,
                              start_soc_pct=start_soc_pct,
                              seg_start_m=seg_start_m)

    objective = _build_objective(evaluator)

    n_loops = len(loops_committed) if loops_committed else 0
    allowed_time_s = (
        (race_config.day_finish_time_s(day_index) - race_config.day_start_time_s(day_index))
        - elapsed_s
        - (0.0 if cs_taken else race_config.CONTROL_STOP_DURATION_S)
        - n_loops * race_config.LOOP_STOP_DURATION_S
    ) 

    constraints = [
        _terminal_soc_constraint(evaluator, alpha_next_day_pct),
        _time_cutoff_constraint(evaluator, allowed_time_s),
    ]

    if "warm_start_kmh" in kwargs and kwargs["warm_start_kmh"] is not None:
        warm_x = np.asarray(kwargs["warm_start_kmh"])
        if len(warm_x) == n_segments:
            global_result = GlobalSearchResult(x=warm_x, fun=objective(warm_x), method="warm")
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
        evaluator, slsqp_result.x, v_max_kmh, constraints=constraints)
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
        diagnostics=dict(
            global_fun=global_result.fun,
            slsqp_fun=slsqp_result.fun,
            slsqp_success=slsqp_result.success,
        ),
    )