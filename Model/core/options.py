"""
core/options.py — BreakdownModel + loop/run-options support.

Owner: Hafiz + Prahlad.

============================================================================
LIVE CODE in this file (used by the real pipeline):
  - BreakdownModel (imported by simulator/forward_sim.py)
  - FailureScenario, DEFAULT_SCENARIOS
  - LoopTier, RunOptions, LoopSpec, resolve_loop_attempt_bounds

DEAD CODE REMOVED (12/08 cleanup):
  - p_net_w() — divergent physics stand-in with wrong constants
    (CdA=0.10, Crr=0.0045, panel=4m²/0.24eff, idle=30W).
    Real physics lives in core/physics.py.
  - OptionsSolverConfig physics constants — used only by p_net_w().
    The cost-function weights (w_dist, w_soc, w_solar, w_risk) are
    now in SolverWeights below, consumed by singleday.py's objective.
  - _SyntheticRoute, _SyntheticSolar, _SyntheticWind — test stubs.
  - OptionsDayEvaluator, solve_with_options — standalone solver that
    duplicated forward_sim.py + singleday.py. Tests exercise the real
    pipeline instead.
  - Smoke-test __main__ block.

DECISIONS LOCKED THIS SESSION (11/08) — still apply to BreakdownModel:
  1. BreakdownModel.step(dt) runs on the FINE PHYSICS SUBSTEP grid.
  4. hazard_mass = p_eff * duration_s (expected-seconds units).
  5. tau_recover = 3x duration; k_suppress = 1/duration.
============================================================================
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Cost-function weights — consumed by singleday._build_objective()
# ============================================================================

@dataclass
class SolverWeights:
    """Objective weights for the enriched cost function (decision #6).

    J = -final_soc_pct
        + w_solar * solar_underutil_penalty
        + w_risk  * expected_breakdown_frac
    """
    w_solar: float = 0.10         # solar underutilization penalty weight
    w_risk: float = 0.05          # breakdown risk weight (zero to disable)


WEIGHTS = SolverWeights()


def configure_weights(w: SolverWeights) -> None:
    """Swap the module-level weights (for tests or per-day overrides)."""
    global WEIGHTS
    WEIGHTS = w


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


# Default loop-tier tunables (previously buried in OptionsSolverConfig)
_EASY_SPEED_FRACTION = 0.75
_MEDIUM_ATTEMPTS_PER_LOOP = 2


def resolve_loop_attempt_bounds(
    tier: LoopTier, loops: List[LoopSpec], remaining_time_s: float
) -> Dict[str, Tuple[int, int, float]]:
    """
    Per named loop -> (min_attempts, max_attempts, effective_v_max_ms).

    off    -> (0, 0, loop.v_max_ms)
    easy   -> (0, generous_upper_bound, loop.v_max_ms * easy_speed_fraction)
    medium -> (0, medium_attempts_per_loop, loop.v_max_ms)
    max    -> (0, generous_upper_bound, loop.v_max_ms)
    """
    out: Dict[str, Tuple[int, int, float]] = {}
    for loop in loops:
        if tier is LoopTier.OFF:
            out[loop.name] = (0, 0, loop.v_max_ms)
            continue

        if tier is LoopTier.EASY:
            eff_v = loop.v_max_ms * _EASY_SPEED_FRACTION
        else:
            eff_v = loop.v_max_ms

        min_lap_time_s = loop.loop_stop_s + loop.length_m / max(eff_v, 1e-3)
        generous_upper = max(1, int(remaining_time_s // min_lap_time_s))

        if tier is LoopTier.MEDIUM:
            out[loop.name] = (0, min(_MEDIUM_ATTEMPTS_PER_LOOP, generous_upper), eff_v)
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