"""
optimizers/singleday.py — L2 single-day velocity optimizer — REVIEW DRAFT v2
(block 5, owner: TBD). Mirrors the frozen `solve()` signature from the
existing stub.

Changes from v1 per Hafiz's review:
  i)   global search is swappable between differential_evolution and a
       custom GA (strategy pattern, GLOBAL_SEARCH_REGISTRY) instead of a
       hardcoded DE-only pipeline.
  ii)  control-segment resolution is 500 m (CONTROL_SEGMENT_M below),
       overriding solver_config.CONTROL_SEGMENT_M (5_000) — propose
       updating that shared constant in the merge PR instead of overriding
       it locally long-term.
  iii) mandatory driver/passenger swaps (SR 2.24.4, every 2 h) are modeled
       during integration: piggyback on a CS/loop stop when one coincides,
       otherwise cost a standalone stop.
  iv)  segments containing a rapid route-bearing change get an artificial
       20 km/h speed cap (sharp-turn realism) applied to their v_max before
       the solve, not learned by the optimizer.
  v)   final output is projected to integer km/h (cruise-control targets),
       with a light local search to recover SOC lost to rounding.

Mode-specific loss functions (charging/cruising/traffic) are intentionally
NOT included in this pass, per instruction — objective is plain end-of-day
SOC maximisation, matching Plan v3 §8's stated L2 objective.

Requires scipy>=1.14 (pinned in Model/requirements.txt) for NonlinearConstraint
objects to work directly with method="SLSQP" in `minimize`.
"""

from __future__ import annotations

import dataclasses
import typing as _t

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, differential_evolution, minimize

from configs.car_config import CarState
from configs import solver_config as SCFG
from configs import race_config
from core import physics
from core.battery import Battery
from core.route import Route
import random

# from simulator import forward_sim  # TODO: see module docstring (iii) —
# _simulate below is a real, working integrator, not a stub, but it duplicates
# what forward_sim.py is meant to own. Suggest merging this loop into
# forward_sim.py directly rather than Junior C writing a second one.


# ===========================================================================
# 0. Local config overrides — candidates for solver_config.py promotion
# ===========================================================================

CONTROL_SEGMENT_M = SCFG.CONTROL_SEGMENT_M //5                #Uses a kilometer resolution

SHARP_TURN_HEADING_DELTA_DEG = 30.0      # (iv) placeholder threshold, TODO-VERIFY
                                         # against real KMZ bearing noise/smoothing
SHARP_TURN_SPEED_LIMIT_KMH = 20.0        # as discussed

DE_POPSIZE = 8                            # (i) kept conservative — see perf note
DE_MAXITER = 60                           # in chat: DE popsize multiplies by
                                          # dimensionality, and 500 m resolution
                                          # means ~600 dims on a long day.
                                          # TODO-VERIFY / benchmark before race use.

_DRIVER_SWAP_STANDALONE_DURATION_S = race_config.LOOP_STOP_DURATION_S
# TODO-VERIFY: no dedicated "standalone swap duration" constant exists in
# race_config.py yet (only DRIVER_SWAP_INTERVAL_S, the 2 h cadence). Reusing
# LOOP_STOP_DURATION_S (5 min) as the closest existing mandatory-brief-stop
# analogue rather than inventing an unsourced number.


# ===========================================================================
# 1. Driver-swap scheduler (iii)
# ===========================================================================

class DriverSwapScheduler:
    """Tracks on-seat elapsed time and decides when SR 2.24.4 swaps land.

    Wall-clock elapsed time since the last swap (car is occupied during CS/
    loop stops too) is what's tracked — ASSUMPTION, not explicitly specified
    by the regs excerpt available; flag if "driving time only" was intended
    instead. A swap that coincides with an already-scheduled CS/loop stop
    (Plan v3: "swaps scheduled onto CS/loop stops where possible") costs
    nothing extra; one that doesn't costs a standalone stop.
    """

    def __init__(self, swap_interval_s: float = race_config.DRIVER_SWAP_INTERVAL_S,
                 standalone_duration_s: float = _DRIVER_SWAP_STANDALONE_DURATION_S):
        self.swap_interval_s = swap_interval_s
        self.standalone_duration_s = standalone_duration_s
        self._elapsed_since_last_swap_s = 0.0
        self.swap_log: list[dict] = []

    def advance(self, dt_s: float, t_now_s: float, x_m: float,
                coincides_with_stop: bool) -> float:
        """Advance the on-seat clock by dt_s; returns extra stoppage seconds
        to add to the day's total_time_s if a swap becomes due."""
        self._elapsed_since_last_swap_s += dt_s
        if self._elapsed_since_last_swap_s < self.swap_interval_s:
            return 0.0
        self._elapsed_since_last_swap_s = 0.0
        added_s = 0.0 if coincides_with_stop else self.standalone_duration_s
        self.swap_log.append(dict(t_s=t_now_s, x_m=x_m,
                                   piggybacked=coincides_with_stop,
                                   added_s=added_s))
        return added_s


def _is_mandatory_stop_zone(route: Route, x_m: float) -> bool:
    """Whether x_m falls inside a scheduled CS or loop-stop zone, so a swap
    can piggyback on it for free.

    Reaches into route.df directly (frozen schema: 'control_stop' bool,
    'seg_type' str) because Route doesn't currently expose a public accessor
    for either column — worth adding control_stop_at()/seg_type_at() to
    core/route.py for symmetry with slope_pct_at() etc.; not touching that
    shared/frozen-interface file in this review.
    """
    x = route.df["distance_m"].to_numpy()
    idx = min(int(np.searchsorted(x, x_m)), len(route.df) - 1)
    row = route.df.iloc[idx]
    return bool(row["control_stop"]) or str(row["seg_type"]).startswith("loop_")

def simulate_breakdown(p_net):
    inputs={"p_net":p_net}
    scenarios=[{"name":"Battery Failure","type":"Electrical","input":"p_net","duration":10*60,"prob": lambda s: 0 if s <= 2000 else (1.0 if s >= 4100 else 0.05 + 0.95 * ((s - 2000) / 2100) ** 3)}]
    seed=random.random()
    stop_time=0
    for scenario in scenarios:
        if seed < scenario["prob"](inputs[scenario["input"]]):
            stop_time+=scenario["duration"]
            break
    return stop_time

# ===========================================================================
# 2. Sharp-turn speed caps (iv)
# ===========================================================================

def _sharp_turn_mask(route: Route, seg_start_m: np.ndarray, seg_len_m: float,
                      heading_delta_threshold_deg: float) -> np.ndarray:
    """True for each control segment containing a rapid bearing change on
    the route's native grid (not just start/end of the 500 m segment, so a
    turn hidden mid-segment isn't missed by a coarse before/after check)."""
    x = route.df["distance_m"].to_numpy()
    bearing = route.df["bearing_deg"].to_numpy()
    raw_delta = np.diff(bearing, prepend=bearing[0])
    wrapped = (raw_delta + 180.0) % 360.0 - 180.0          # signed, [-180, 180]
    sharp_point = np.abs(wrapped) >= heading_delta_threshold_deg

    seg_end_m = seg_start_m + seg_len_m
    mask = np.zeros(len(seg_start_m), dtype=bool)
    for i, (s, e) in enumerate(zip(seg_start_m, seg_end_m)):
        in_seg = (x >= s) & (x < e)
        mask[i] = bool(np.any(sharp_point[in_seg]))
    return mask
    # NOTE: O(n_segments x n_route_points); a one-off cost at solve() setup,
    # not inside the objective loop, so left simple. Revisit with a
    # searchsorted bucketing pass if profiling ever flags it.


def apply_turn_speed_caps(route: Route, v_max_kmh: np.ndarray,
                           seg_start_m: np.ndarray,
                           seg_len_m: float = CONTROL_SEGMENT_M,
                           heading_delta_threshold_deg: float = SHARP_TURN_HEADING_DELTA_DEG,
                           turn_speed_limit_kmh: float = SHARP_TURN_SPEED_LIMIT_KMH,
                           ) -> np.ndarray:
    """Clamp v_max to turn_speed_limit_kmh wherever the route bearing turns
    sharply within a segment — "that's what the driver will realistically
    turn at," applied as a hard bound, not left for the optimizer to learn.

    Overlaps conceptually with route.v_max_ms_at()'s existing curvature_1pm
    -based turn cap (once pipeline/build_route.py's turn-cap layer is
    implemented) — worth reconciling the two later so sharp turns aren't
    penalised twice; implemented directly here per your instruction in the
    meantime.
    """
    sharp = _sharp_turn_mask(route, seg_start_m, seg_len_m, heading_delta_threshold_deg)
    return np.where(sharp, np.minimum(v_max_kmh, turn_speed_limit_kmh), v_max_kmh)


# ===========================================================================
# 3. Day-level evaluation
# ===========================================================================

@dataclasses.dataclass
class DayEvalResult:
    final_soc_pct: float
    total_time_s: float
    driver_swaps: list
    v_ms: np.ndarray


class DayEvaluator:
    """Runs one candidate speed vector through physics + timing.

    Memoised per-x: DE/GA/SLSQP all evaluate the objective and every
    constraint separately on the SAME x far more often than x changes.
    """

    def __init__(self, route: Route, car: CarState, solar_provider,
                 wind_provider, t0_s: float, start_soc_pct: float,
                 seg_start_m: np.ndarray, seg_len_m: float = CONTROL_SEGMENT_M,
                 energy_grid_m: float = SCFG.ENERGY_GRID_M):
        self.route = route
        self.car = car
        self.solar_provider = solar_provider
        self.wind_provider = wind_provider  # TODO: not yet wired into
        # physics.net_power's wind_along_ms — core/wind.py's exact call
        # convention wasn't in scope for this pass; left at the physics
        # default (0.0) rather than guessing the signature.
        self.t0_s = t0_s
        self.start_soc_pct = start_soc_pct
        self.seg_start_m = seg_start_m
        self.seg_len_m = seg_len_m
        self.energy_grid_m = energy_grid_m
        self._cache: dict[bytes, DayEvalResult] = {}

    def __call__(self, v_kmh: np.ndarray) -> DayEvalResult:
        key = np.asarray(v_kmh, dtype=float).round(6).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._simulate(np.asarray(v_kmh, dtype=float))
        self._cache[key] = result
        return result

    def _simulate(self, v_kmh: np.ndarray) -> DayEvalResult:
        """Real integrator (reuses core.physics.net_power + core.battery.Battery
        — no duplicated physics). See module docstring re: forward_sim overlap.

        Velocity is held per CONTROL_SEGMENT_M control segment; physics
        integrates on the finer ENERGY_GRID_M grid within each control
        segment (Plan v3 §8), so slope variation inside a 500 m segment is
        still captured.
        """
        battery = Battery(self.car, self.start_soc_pct)
        swap_scheduler = DriverSwapScheduler()
        t_s = float(self.t0_s)
        x_m = 0.0

        n_substeps = max(1, round(self.seg_len_m / self.energy_grid_m))
        substep_len_km = (self.seg_len_m / n_substeps) / 1000.0

        for v in v_kmh:
            v_ms = float(v) / 3.6
            for _ in range(n_substeps):
                slope = self.route.slope_pct_at(x_m)
                ghi = self.solar_provider.ghi_wm2(t_s, x_m)
                p_net, dt_s = physics.net_power(
                    self.car, v_ms, v_ms, slope, ghi, substep_len_km)
                battery.apply_energy_wh(float(p_net) * float(dt_s) / 3600.0)
                t_s += float(dt_s)
                x_m += substep_len_km * 1000.0

                stop_here = _is_mandatory_stop_zone(self.route, x_m)
                breakdown_time=simulate_breakdown(p_net)
                t_s += swap_scheduler.advance(
                    float(dt_s), t_s, x_m, coincides_with_stop=stop_here)
                t_s+=breakdown_time

        return DayEvalResult(
            final_soc_pct=battery.soc_pct,
            total_time_s=t_s - self.t0_s,
            driver_swaps=swap_scheduler.swap_log,
            v_ms=v_kmh / 3.6,
        )


# ===========================================================================
# 4. Objective / constraints
# ===========================================================================

def _build_objective(evaluator: DayEvaluator) -> _t.Callable[[np.ndarray], float]:
    """Maximise end-of-day SOC == minimise -SOC (Plan v3 §8 L2 objective)."""
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

# TODO: |a| <= a_max coupling constraint between consecutive control
# segments (Plan v3: <=0.5 m/s^2) — unchanged from v1, still not in scope
# for this pass.


# ===========================================================================
# 5. Swappable global search (i) — strategy pattern
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
    """Wraps scipy.optimize.differential_evolution.

    polish=False always: SLSQP is chained explicitly in solve() afterwards
    (full control over SLSQP_MAX_ITER/FTOL; keeps stages independently
    testable — DE's public `polish` kwarg is a bool, not an injectable
    SLSQP callable).
    """

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
    """Real-valued GA: tournament selection, arithmetic crossover, Gaussian
    mutation, elitism. Population stays fixed at `population` regardless of
    dimensionality (unlike DE's popsize*dim scaling) — the cheaper option of
    the two at 500 m resolution on long stages. Defaults match the notes doc
    (population=64, generations=50, mutation bump=+-10 km/h) via
    solver_config.py.

    Constraints have no native GA support, so violations are penalised into
    the fitness (standard exterior-penalty approach) — same NonlinearConstraint
    objects as DE, so both strategies share one constraint definition.
    """

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

        for _ in range(self.generations):
            order = np.argsort(fitness)
            pop, fitness = pop[order], fitness[order]
            new_pop = [pop[i].copy() for i in range(n_elite)]
            while len(new_pop) < self.population:
                p1 = self._tournament(pop, fitness, rng, self.tournament_k)
                p2 = self._tournament(pop, fitness, rng, self.tournament_k)
                alpha = rng.uniform(0.0, 1.0, size=dim)
                child = alpha * p1 + (1.0 - alpha) * p2       # arithmetic crossover
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
        raise KeyError(
            f"Unknown global_method={method!r}; choose from "
            f"{sorted(GLOBAL_SEARCH_REGISTRY)}."
        )
    return cls(**kwargs)


# ===========================================================================
# 6. Integer km/h projection (v)
# ===========================================================================

def project_to_integer_kmh(evaluator: DayEvaluator, v_kmh: np.ndarray,
                            v_max_kmh: np.ndarray, v_min_kmh: float = 5.0,
                            constraints: _t.Sequence[NonlinearConstraint] = (),
                            ) -> np.ndarray:
    """Round to integer km/h (cruise-control target: driver holds a whole
    number, +-1 km/h button), clipped so rounding never exceeds a segment's
    limit (turn caps included, since v_max_kmh already has them folded in).
    Follows with a single-pass greedy coordinate search to recover SOC lost
    to rounding, without breaking feasibility.

    PERF NOTE: each candidate re-runs the full day simulation, so this pass
    is O(n_segments^2) — fine for review/testing at ~600 segments, revisit
    (e.g. re-simulate only from segment i onward using cached battery state)
    if profiling flags it once forward_sim is wired in for real.
    """
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
# 7. solve() — frozen API
# ===========================================================================

def solve(route: Route, car: CarState, solar_provider, wind_provider,
          day_index: int, start_soc_pct: float, alpha_next_day_pct: float,
          loops_committed, global_method: str = "ga", seed: int | None = None):
    """Frozen API — signature unchanged from the existing stub, plus
    `global_method`/`seed` as optional kwargs (default "ga" — Plan v3's
    cited, evidence-backed choice — with "de" available on request; see
    review notes for why I didn't default to "de").

    Output: dict with the integer km/h velocity card + diagnostics. Exact
    shape TBD pending Diyaansh's dashboard-consumption format.
    """
    n_segments = int(np.ceil(route.total_m / CONTROL_SEGMENT_M))
    seg_start_m = np.arange(n_segments) * CONTROL_SEGMENT_M

    v_max_kmh = route.v_max_ms_at(seg_start_m) * 3.6
    v_max_kmh = apply_turn_speed_caps(route, v_max_kmh, seg_start_m)
    bounds = Bounds(lb=np.full(n_segments, 5.0), ub=v_max_kmh)  # 5 km/h floor placeholder

    evaluator = DayEvaluator(route, car, solar_provider, wind_provider,
                              t0_s=race_config.day_start_time_s(day_index),
                              start_soc_pct=start_soc_pct,
                              seg_start_m=seg_start_m)

    objective = _build_objective(evaluator)

    n_loops = len(loops_committed) if loops_committed else 0
    allowed_time_s = (
        (race_config.day_finish_time_s(day_index) - race_config.day_start_time_s(day_index))
        - race_config.CONTROL_STOP_DURATION_S
        - n_loops * race_config.LOOP_STOP_DURATION_S
    )  # approximation — driver-swap time isn't subtracted here since it's
       # already inside evaluator(...).total_time_s dynamically; not double-counted

    constraints = [
        _terminal_soc_constraint(evaluator, alpha_next_day_pct),
        _time_cutoff_constraint(evaluator, allowed_time_s),
    ]

    # --- stage 1: swappable global search (i) -------------------------------
    global_search = get_global_search(global_method)
    global_result = global_search.search(objective, bounds, constraints, seed=seed)

    # --- stage 2: SLSQP polish ------------------------------------------------
    slsqp_result = minimize(
        objective, x0=global_result.x, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options=dict(maxiter=SCFG.SLSQP_MAX_ITER, ftol=SCFG.SLSQP_FTOL),
    )

    # --- stage 3: integer km/h projection (v) ----------------------------------
    v_final_kmh = project_to_integer_kmh(
        evaluator, slsqp_result.x, v_max_kmh, constraints=constraints)
    final_eval = evaluator(v_final_kmh)

    return dict(
        v_kmh=v_final_kmh,
        seg_start_m=seg_start_m,
        final_soc_pct=final_eval.final_soc_pct,
        total_time_s=final_eval.total_time_s,
        driver_swaps=final_eval.driver_swaps,
        global_method=global_result.method,
        diagnostics=dict(
            global_fun=global_result.fun,
            slsqp_fun=slsqp_result.fun,
            slsqp_success=slsqp_result.success,
        ),
    )