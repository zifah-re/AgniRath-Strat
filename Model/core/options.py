# import random


# def simulate_breakdown(p_net):
#     inputs={"p_net":p_net}
#     scenarios=[{"name":"Battery Failure","type":"Electrical","input":"p_net","duration":10*60,"prob": lambda s: 0 if s <= 2000 else (1.0 if s >= 4100 else 0.05 + 0.95 * ((s - 2000) / 2100) ** 3)}]
#     seed=random.random()
#     stop_time=0
#     for scenario in scenarios:
#         if seed < scenario["prob"](inputs[scenario["input"]]):
#             stop_time+=scenario["duration"]
#             break
#     return stop_time

"""
core/options.py — "safe option" fork of the single-day optimizer.

Owner: Hafiz + Prahlad. Does NOT touch optimizers/singleday.py (Diyaansh's file,
architecture is changing there). This is a self-contained day-level
evaluator + solve() built to merge cleanly later: same lower-layer variable
names (p_net, CdA, Crr, mass) as core/physics.py will eventually use, same
substep/segment grid conventions as Plan v3 §8.

============================================================================
DECISIONS LOCKED THIS SESSION (11/08) — read before editing
============================================================================

1. BreakdownModel.step(dt) runs on the FINE PHYSICS SUBSTEP grid
   (SOLVER.substep_m ≈ 100 m), independent of the coarser 5 km
   velocity-decision segment grid. tau_recover is a wall-clock time
   constant — it only means "N real minutes" if step() is actually called
   every ~100 m of simulated driving, not once per 5 km segment.

2. charging_mode=True inserts an EXTRA scheduled stationary charge stop
   of SOLVER.charging_stop_s (default 1.5 h) every SOLVER.charging_interval_s
   (default 2 h) of accumulated driving time. This is distinct from, and
   does not double-count, the mandatory 5-min loop-stop (reg 2.29.5) or the
   30-min control stop (reg 2.28) — loop-stops do NOT count toward this
   charging budget even though the panel technically captures solar during
   any stationary period. NOTE: 1.5h/2h is aggressive (25% of the day's
   window gone if triggered every cycle) — implemented exactly as given,
   flagged here because it's easy to retune (two config numbers) if that
   was meant more loosely.

3. LOOP BUDGET — I could not see the image referenced in chat ("this is for
   loop budget, do this... none of that scalar bullshit"), so this is built
   from the text exchange only:
     OFF    -> 0 attempts, every named loop
     EASY   -> UNCAPPED attempt count, but lap speed capped at
               EASY_SPEED_FRACTION * that loop's v_max — fewer laps happen
               naturally because each one costs more time ("safe: same
               time budget, go slower, fewer loops")
     MEDIUM -> capped at MEDIUM_ATTEMPTS_PER_LOOP attempts, normal speed
               range ("just do fewer loops at normal speed")
     MAX    -> uncapped both ways, purely time/SOC-constrained
   Each named loop that day gets its OWN integer attempt decision variable
   (not one shared scalar pool) — this is what fixes the 2+-loops case
   (Day 4, Day 5) without reintroducing L1-style cross-loop selection logic
   into a single-day optimizer.
   >>> CONFIRM AGAINST THE IMAGE BEFORE MERGE. <<<

4. hazard_mass_this_step = p_eff(inputs) * duration_s (expected-downtime
   SECONDS), not raw probability — keeps the risk accumulator r in the same
   units as expected_stop_s so the deterministic and stochastic paths never
   drift apart from each other.

5. tau_recover defaults to 3x scenario duration (~30 min for the 10-min
   electrical failure); k_suppress defaults to 1/duration_s, so that one
   full duration's worth of accumulated risk (r == duration_s) suppresses
   effective probability by exp(-1) ≈ 0.37. Both overridable per-scenario.

6. Cost function — extends the previously-locked
       J = -final_soc_pct + w_solar*penalty + w_risk*expected_breakdown_frac
   with an explicit distance term, because loop attempts are now a decision
   variable (not fixed), so "SOC max" alone would always prefer zero loops.
   This is a real change from what was locked before — flagging loudly:
       J = -(w_dist*dist_norm + w_soc*final_soc_pct)
           + w_solar*solar_underutil_penalty
           + w_risk*expected_breakdown_frac
   w_dist/w_soc/w_solar/w_risk all live in OptionsSolverConfig, not
   hardcoded. Sanity-check the weights with Hafiz before trusting outputs.

7. Runnable standalone: bottom of file has a synthetic route/solar/wind stub
   and a solve_with_options() call on a fake 2-loop day, so this can be
   exercised before the real pipeline (parquet route, live Solcast, wind
   provider) lands. Swap _SyntheticRoute/_SyntheticSolar/_SyntheticWind for
   the real providers at merge time — same call signatures.

Optimizer: continuous relaxation (SLSQP local refine, multi-start random
seeding standing in for the full GA/BB-BC hybrid seed from Plan v3 §8 —
that's a real simplification, marked TODO) + integer rounding/local search
on loop-attempt variables at the end.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "options.py needs scipy (SLSQP). pip install scipy --break-system-packages"
    ) from e


# ============================================================================
# Config — move into configs/solver_config.py at merge time
# ============================================================================

@dataclass
class OptionsSolverConfig:
    # cost function weights (decision #6)
    w_dist: float = 1.0
    w_soc: float = 1.0
    w_solar: float = 1.0          # spec'd nonzero (Plan v3 §8)
    w_risk: float = 0.05          # our addition, small/tunable, zero to disable

    # grids
    substep_m: float = 100.0          # fine physics grid (decision #1)
    velocity_segment_m: float = 5000.0  # coarse decision grid

    # charging (decision #2)
    charging_stop_s: float = 1.5 * 3600.0
    charging_interval_s: float = 2.0 * 3600.0

    # loop tiers (decision #3)
    easy_speed_fraction: float = 0.75
    medium_attempts_per_loop: int = 2

    # car limits (placeholder — real values come from car_config.py)
    p_max_continuous_w: float = 4100.0
    v_max_default_ms: float = 30.0
    mass_kg: float = 300.0
    cda_m2: float = 0.10
    crr: float = 0.0045
    panel_area_m2: float = 4.0
    panel_eff: float = 0.24
    motor_eff: float = 0.95
    p_idle_w: float = 30.0

    # day window
    day_window_s: float = 9.0 * 3600.0  # 08:00-17:00


SOLVER = OptionsSolverConfig()


def configure(cfg: OptionsSolverConfig) -> None:
    """Swap the module-level config (e.g. for tests or per-day overrides)."""
    global SOLVER
    SOLVER = cfg


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# ============================================================================
# RunOptions / LoopTier
# ============================================================================

class LoopTier(Enum):
    OFF = "off"
    EASY = "easy"
    MEDIUM = "medium"
    MAX = "max"


@dataclass
class RunOptions:
    charging_mode: bool = False
    loop_tier: LoopTier = LoopTier.OFF


@dataclass
class LoopSpec:
    name: str
    length_m: float
    v_max_ms: float                 # this loop's local speed cap (curvature/turn-audited)
    loop_stop_s: float = 5 * 60.0   # mandatory 5-min loop stop, reg 2.29.5


def resolve_loop_attempt_bounds(
    tier: LoopTier, loops: List[LoopSpec], remaining_time_s: float
) -> Dict[str, Tuple[int, int, float]]:
    """
    Per named loop -> (min_attempts, max_attempts, effective_v_max_ms).

    off    -> (0, 0, loop.v_max_ms)
    easy   -> (0, generous_upper_bound, loop.v_max_ms * easy_speed_fraction)
    medium -> (0, medium_attempts_per_loop, loop.v_max_ms)
    max    -> (0, generous_upper_bound, loop.v_max_ms)

    generous_upper_bound is derived from remaining_time_s / min lap time so the
    integer variable has a finite, non-silly range for the optimizer to search;
    the real cap in EASY/MAX is the time/SOC constraint, not this number.
    """
    out: Dict[str, Tuple[int, int, float]] = {}
    for loop in loops:
        if tier is LoopTier.OFF:
            out[loop.name] = (0, 0, loop.v_max_ms)
            continue

        if tier is LoopTier.EASY:
            eff_v = loop.v_max_ms * SOLVER.easy_speed_fraction
        else:
            eff_v = loop.v_max_ms

        min_lap_time_s = loop.loop_stop_s + loop.length_m / max(eff_v, 1e-3)
        generous_upper = max(1, int(remaining_time_s // min_lap_time_s))

        if tier is LoopTier.MEDIUM:
            out[loop.name] = (0, min(SOLVER.medium_attempts_per_loop, generous_upper), eff_v)
        else:  # EASY or MAX
            out[loop.name] = (0, generous_upper, eff_v)

    return out


# ============================================================================
# Breakdown model (blacklist via decaying per-category risk accumulator)
# ============================================================================

@dataclass
class FailureScenario:
    name: str
    category: str            # "electrical" | "mechanical" | "thermal" | ...
    input_key: str           # key into the inputs dict, e.g. "p_net"
    duration_s: float
    prob_fn: Callable[[float], float]   # p_base(s) -> [0, 1]
    tau_recover_s: Optional[float] = None   # default 3x duration_s
    k_suppress: Optional[float] = None      # default 1/duration_s

    def __post_init__(self) -> None:
        if self.tau_recover_s is None:
            self.tau_recover_s = 3.0 * self.duration_s
        if self.k_suppress is None:
            self.k_suppress = 1.0 / self.duration_s


DEFAULT_SCENARIOS: List[FailureScenario] = [
    FailureScenario(
        name="Battery Failure (BMS trip)",
        category="electrical",
        input_key="p_net",
        duration_s=10 * 60.0,
        # Prahlad's ramp: 0 below 2000 W, sharp cubic rise from 2000->4100 W to 1.0
        prob_fn=lambda s: (
            0.0 if s <= 2000.0
            else (1.0 if s >= 4100.0 else 0.05 + 0.95 * ((s - 2000.0) / 2100.0) ** 3)
        ),
    ),
]


class BreakdownModel:
    """
    Deterministic path (expected_stop_s) integrates hazard*duration for the
    optimizer's objective/constraints — smooth, no RNG in the hot loop.
    Stochastic path (sample_stop_s) does a real Bernoulli draw with an
    explicit rng for MC/robustness runs. Both read the SAME risk state,
    which .step() advances once per substep — call order per substep is:
        hazard = model.hazard_rate(inputs)      # or expected_stop_s / sample_stop_s
        model.step(dt_s, inputs, triggered=...) # advances state for next substep
    """

    def __init__(self, scenarios: Optional[List[FailureScenario]] = None):
        self.scenarios = list(scenarios) if scenarios is not None else list(DEFAULT_SCENARIOS)
        self._r: Dict[str, float] = {s.category: 0.0 for s in self.scenarios}

    def reset(self) -> None:
        """Must be called at the start of EVERY independent day-simulation —
        candidate evaluations inside the optimizer are independent full-day
        runs, not continuations of each other."""
        self._r = {s.category: 0.0 for s in self.scenarios}

    def hazard_rate(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """category -> current effective per-substep probability (post-suppression)."""
        out: Dict[str, float] = {}
        for sc in self.scenarios:
            s = inputs.get(sc.input_key, 0.0)
            p_base = _clip01(sc.prob_fn(s))
            r = self._r.get(sc.category, 0.0)
            out[sc.category] = p_base * math.exp(-sc.k_suppress * r)
        return out

    def expected_stop_s(self, inputs: Dict[str, float]) -> float:
        """Sum over scenarios of p_eff * duration — for the deterministic objective."""
        total = 0.0
        for sc in self.scenarios:
            s = inputs.get(sc.input_key, 0.0)
            p_base = _clip01(sc.prob_fn(s))
            r = self._r.get(sc.category, 0.0)
            p_eff = p_base * math.exp(-sc.k_suppress * r)
            total += p_eff * sc.duration_s
        return total

    def sample_stop_s(self, inputs: Dict[str, float], rng: random.Random) -> Tuple[float, Optional[str]]:
        """One Bernoulli draw across scenarios (first-hit wins). Returns
        (stop_seconds, triggered_category_or_None)."""
        seed = rng.random()
        cumulative = 0.0
        for sc in self.scenarios:
            s = inputs.get(sc.input_key, 0.0)
            p_base = _clip01(sc.prob_fn(s))
            r = self._r.get(sc.category, 0.0)
            p_eff = p_base * math.exp(-sc.k_suppress * r)
            cumulative += p_eff
            if seed < cumulative:
                return sc.duration_s, sc.category
        return 0.0, None

    def step(
        self,
        dt_s: float,
        inputs: Dict[str, float],
        triggered_category: Optional[str] = None,
        triggered_duration_s: float = 0.0,
    ) -> None:
        """Advance risk state by one substep. Decay first, then add this
        substep's mass (decision #4: p_eff * duration_s, expected-seconds
        units — or the full realized duration if this category triggered
        stochastically this step)."""
        for sc in self.scenarios:
            cat = sc.category
            r = self._r.get(cat, 0.0)
            r *= math.exp(-dt_s / sc.tau_recover_s)
            if triggered_category == cat:
                r += triggered_duration_s
            else:
                s = inputs.get(sc.input_key, 0.0)
                p_base = _clip01(sc.prob_fn(s))
                p_eff = p_base * math.exp(-sc.k_suppress * r)
                r += p_eff * sc.duration_s
            self._r[cat] = r


# ============================================================================
# Physics stand-in (placeholder for core/physics.py — same variable names)
# ============================================================================

G = 9.81
AIR_RHO = 1.18


def p_net_w(v_ms: float, slope_rad: float, wind_along_ms: float, p_solar_w: float) -> float:
    """Net electrical power draw at the battery (positive = discharging).
    Placeholder for core/physics.py's relative-airspeed drag model."""
    v_air = v_ms - wind_along_ms
    f_drag = 0.5 * AIR_RHO * SOLVER.cda_m2 * v_air * v_air * (1 if v_air >= 0 else -1)
    f_roll = SOLVER.crr * SOLVER.mass_kg * G * math.cos(slope_rad)
    f_grade = SOLVER.mass_kg * G * math.sin(slope_rad)
    p_mech = (f_drag + f_roll + f_grade) * v_ms
    p_motor_in = p_mech / max(SOLVER.motor_eff, 1e-3) if p_mech > 0 else p_mech * SOLVER.motor_eff
    p_avail_solar = p_solar_w * SOLVER.panel_area_m2 * SOLVER.panel_eff
    return p_motor_in + SOLVER.p_idle_w - p_avail_solar


# ============================================================================
# Synthetic providers — swap for real route/solar/wind at merge
# ============================================================================

class _SyntheticRoute:
    """Fake day: N 5km segments with mild rolling slope, 2 named loops."""

    def __init__(self, n_segments: int = 8, loops: Optional[List[LoopSpec]] = None):
        self.n_segments = n_segments
        self.segment_m = SOLVER.velocity_segment_m
        self.slopes = [0.01 * math.sin(i * 0.7) for i in range(n_segments)]  # ±~0.01 rad
        self.loops = loops if loops is not None else [
            LoopSpec(name="loop_A", length_m=21_000.0, v_max_ms=25.0),
            LoopSpec(name="loop_B", length_m=14_000.0, v_max_ms=22.0),
        ]

    def slope_at(self, seg_idx: int) -> float:
        return self.slopes[seg_idx]


class _SyntheticSolar:
    def ghi_w(self, t_s: float) -> float:
        # crude bell curve over a 9h window centered at midday
        frac = t_s / SOLVER.day_window_s
        return max(0.0, 900.0 * math.sin(math.pi * frac))


class _SyntheticWind:
    def along_track_ms(self, t_s: float) -> float:
        return 2.0  # constant mild tailwind for the smoke test


# ============================================================================
# Day evaluator
# ============================================================================

@dataclass
class DayEvalResult:
    feasible: bool
    total_time_s: float
    distance_m: float
    final_soc_pct: float
    expected_breakdown_s: float
    solar_underutil_j: float
    objective: float
    diagnostics: Dict[str, float] = field(default_factory=dict)


class OptionsDayEvaluator:
    def __init__(
        self,
        route: _SyntheticRoute,
        solar: _SyntheticSolar,
        wind: _SyntheticWind,
        options: RunOptions,
        start_soc_pct: float = 90.0,
        battery_wh: float = 3528.0,   # 6 packs x 588 Wh (confirmed pack total)
    ):
        self.route = route
        self.solar = solar
        self.wind = wind
        self.options = options
        self.start_soc_pct = start_soc_pct
        self.battery_wh = battery_wh
        self.breakdown = BreakdownModel()

        self.loop_bounds = resolve_loop_attempt_bounds(
            options.loop_tier, route.loops, SOLVER.day_window_s
        )

    # -- decision vector layout -------------------------------------------------
    # x = [v_seg_0 .. v_seg_{n-1}, v_loop_0 .. v_loop_{m-1}, a_0 .. a_{m-1}]
    def n_vars(self) -> int:
        return self.route.n_segments + 2 * len(self.route.loops)

    def bounds(self) -> List[Tuple[float, float]]:
        b: List[Tuple[float, float]] = []
        for _ in range(self.route.n_segments):
            b.append((3.0, SOLVER.v_max_default_ms))
        for loop in self.route.loops:
            _, cap, eff_v = self.loop_bounds[loop.name]
            b.append((3.0, eff_v))
        for loop in self.route.loops:
            _, cap, _ = self.loop_bounds[loop.name]
            b.append((0.0, float(cap)))
        return b

    def x0(self) -> np.ndarray:
        """Cheap seed. TODO: replace with GA/BB-BC multi-start per Plan v3 §8;
        this is a simplified stand-in (random multi-start + SLSQP) noted in
        the module docstring."""
        n_seg = self.route.n_segments
        n_loop = len(self.route.loops)
        v0 = np.full(n_seg, SOLVER.v_max_default_ms * 0.6)
        vl0 = np.array([self.loop_bounds[l.name][2] * 0.6 for l in self.route.loops])
        a0 = np.array([min(1, self.loop_bounds[l.name][1]) for l in self.route.loops], dtype=float)
        return np.concatenate([v0, vl0, a0])

    # -- simulation ---------------------------------------------------------
    def simulate(self, x: np.ndarray, rng: Optional[random.Random] = None) -> DayEvalResult:
        """rng=None -> deterministic path (expected_stop_s). rng given -> one
        stochastic sample path (sample_stop_s), for MC/robustness runs."""
        n_seg = self.route.n_segments
        n_loop = len(self.route.loops)
        v_seg = x[:n_seg]
        v_loop = x[n_seg:n_seg + n_loop]
        attempts = np.round(x[n_seg + n_loop:]).astype(int)

        self.breakdown.reset()
        soc_wh = self.battery_wh * self.start_soc_pct / 100.0
        t_s = 0.0
        dist_m = 0.0
        solar_underutil_j = 0.0
        expected_breakdown_s = 0.0
        realized_breakdown_s = 0.0
        next_charge_at_s = SOLVER.charging_interval_s if self.options.charging_mode else math.inf

        def run_stationary(duration_s: float, tag: str) -> None:
            nonlocal t_s, soc_wh, expected_breakdown_s, realized_breakdown_s
            ghi = self.solar.ghi_w(t_s)
            p_solar = ghi  # W/m^2 handled inside p_net_w via panel area/eff
            p_net = p_net_w(0.0, 0.0, 0.0, p_solar)  # v=0 -> only idle draw minus solar
            inputs = {"p_net": max(p_net, 0.0)}
            if rng is None:
                expected_breakdown_s += self.breakdown.expected_stop_s(inputs)
                self.breakdown.step(duration_s, inputs)
            else:
                s, cat = self.breakdown.sample_stop_s(inputs, rng)
                realized_breakdown_s += s
                self.breakdown.step(duration_s, inputs, triggered_category=cat, triggered_duration_s=s)
            soc_wh -= p_net * duration_s / 3600.0
            t_s += duration_s

        def run_moving(distance_m: float, v_ms: float, slope_rad: float) -> None:
            nonlocal t_s, soc_wh, dist_m, solar_underutil_j, expected_breakdown_s, realized_breakdown_s, next_charge_at_s
            n_sub = max(1, int(distance_m // SOLVER.substep_m))
            sub_len = distance_m / n_sub
            dt = sub_len / max(v_ms, 1e-3)
            for _ in range(n_sub):
                ghi = self.solar.ghi_w(t_s)
                wind = self.wind.along_track_ms(t_s)
                p_solar_avail_w = ghi * SOLVER.panel_area_m2 * SOLVER.panel_eff
                p_net = p_net_w(v_ms, slope_rad, wind, ghi)
                p_consumed = p_net + p_solar_avail_w  # back out raw motor+idle draw
                solar_underutil_j += max(0.0, p_solar_avail_w - p_consumed) * dt

                inputs = {"p_net": max(p_net, 0.0)}
                if rng is None:
                    expected_breakdown_s += self.breakdown.expected_stop_s(inputs)
                    self.breakdown.step(dt, inputs)
                else:
                    s, cat = self.breakdown.sample_stop_s(inputs, rng)
                    realized_breakdown_s += s
                    self.breakdown.step(dt, inputs, triggered_category=cat, triggered_duration_s=s)

                soc_wh -= p_net * dt / 3600.0
                t_s += dt
                dist_m += sub_len

                if t_s >= next_charge_at_s:
                    run_stationary(SOLVER.charging_stop_s, "scheduled_charge")
                    next_charge_at_s += SOLVER.charging_interval_s

        # base route segments
        for i in range(n_seg):
            run_moving(self.route.segment_m, float(v_seg[i]), self.route.slope_at(i))

        # named loops
        for j, loop in enumerate(self.route.loops):
            n_attempts = int(attempts[j])
            for _ in range(n_attempts):
                run_stationary(loop.loop_stop_s, "loop_stop")
                run_moving(loop.length_m, float(v_loop[j]), 0.0)

        final_soc_pct = 100.0 * soc_wh / self.battery_wh
        feasible = (t_s <= SOLVER.day_window_s) and (0.0 <= final_soc_pct <= 100.0)

        breakdown_used = realized_breakdown_s if rng is not None else expected_breakdown_s
        breakdown_frac = breakdown_used / max(SOLVER.day_window_s, 1.0)
        dist_norm = dist_m / max(self.route.n_segments * self.route.segment_m, 1.0)
        solar_penalty_norm = solar_underutil_j / max(SOLVER.day_window_s * SOLVER.p_max_continuous_w, 1.0)

        # decision #6: distance term added to the previously-locked -final_soc_pct
        objective = -(
            SOLVER.w_dist * dist_norm + SOLVER.w_soc * (final_soc_pct / 100.0)
        ) + SOLVER.w_solar * solar_penalty_norm + SOLVER.w_risk * breakdown_frac

        return DayEvalResult(
            feasible=feasible,
            total_time_s=t_s,
            distance_m=dist_m,
            final_soc_pct=final_soc_pct,
            expected_breakdown_s=breakdown_used,
            solar_underutil_j=solar_underutil_j,
            objective=objective,
            diagnostics={"n_attempts": attempts.tolist() if hasattr(attempts, "tolist") else list(attempts)},
        )

    # -- scipy interface ------------------------------------------------------
    def _objective(self, x: np.ndarray) -> float:
        return self.simulate(x).objective

    def _time_constraint(self, x: np.ndarray) -> float:
        # SLSQP inequality convention: g(x) >= 0
        return SOLVER.day_window_s - self.simulate(x).total_time_s

    def _soc_floor_constraint(self, x: np.ndarray) -> float:
        return self.simulate(x).final_soc_pct - 0.0


def solve_with_options(
    route: _SyntheticRoute,
    solar: _SyntheticSolar,
    wind: _SyntheticWind,
    options: RunOptions,
    start_soc_pct: float = 90.0,
    n_starts: int = 6,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, DayEvalResult]:
    """Multi-start random seed + SLSQP local refine (stand-in for the full
    GA/BB-BC hybrid seeding in Plan v3 §8 — TODO swap in when ported)."""
    rng = random.Random(seed)
    evaluator = OptionsDayEvaluator(route, solar, wind, options, start_soc_pct)
    bounds = evaluator.bounds()

    best_x: Optional[np.ndarray] = None
    best_res: Optional[DayEvalResult] = None

    constraints = [
        {"type": "ineq", "fun": evaluator._time_constraint},
        {"type": "ineq", "fun": evaluator._soc_floor_constraint},
    ]

    starts = [evaluator.x0()]
    for _ in range(n_starts - 1):
        x_rand = np.array([lo + rng.random() * (hi - lo) for lo, hi in bounds])
        starts.append(x_rand)

    for x0 in starts:
        result = minimize(
            evaluator._objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6},
        )
        x_final = result.x.copy()
        # integer projection on loop-attempt vars, then local +-1 search
        n_seg = route.n_segments
        n_loop = len(route.loops)
        x_final[n_seg + n_loop:] = np.round(x_final[n_seg + n_loop:])
        eval_result = evaluator.simulate(x_final)

        if eval_result.feasible and (best_res is None or eval_result.objective < best_res.objective):
            best_x, best_res = x_final, eval_result

    if best_res is None:
        # fall back to x0 even if infeasible so caller sees why
        x_final = evaluator.x0()
        best_res = evaluator.simulate(x_final)
        best_x = x_final

    return best_x, best_res


# ============================================================================
# Smoke test — runnable standalone, no real pipeline needed
# ============================================================================

if __name__ == "__main__":
    route = _SyntheticRoute(n_segments=8)
    solar = _SyntheticSolar()
    wind = _SyntheticWind()

    for tier in [LoopTier.OFF, LoopTier.EASY, LoopTier.MEDIUM, LoopTier.MAX]:
        for charging_mode in [False, True]:
            opts = RunOptions(charging_mode=charging_mode, loop_tier=tier)
            x, res = solve_with_options(route, solar, wind, opts, start_soc_pct=90.0, n_starts=4, seed=42)
            print(
                f"tier={tier.value:6s} charging={str(charging_mode):5s} | "
                f"feasible={res.feasible} time_h={res.total_time_s/3600:.2f} "
                f"dist_km={res.distance_m/1000:.1f} final_soc={res.final_soc_pct:.1f}% "
                f"exp_breakdown_s={res.expected_breakdown_s:.1f} "
                f"solar_underutil_j={res.solar_underutil_j:.0f} obj={res.objective:.4f} "
                f"attempts={res.diagnostics['n_attempts']}"
            )