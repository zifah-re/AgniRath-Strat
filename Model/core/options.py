"""
core/options.py — breakdown / failure scenario models.

REVAMPED (23/08, strategist directive): the old file was a self-contained fork
of the single-day optimizer with charging-mode stops and easy/medium/max loop
tiers. None of that was used by the real pipeline (only `BreakdownModel` is
imported, by simulator/forward_sim.py), and the strategist asked to drop it:
  * charging during / around breakdowns — removed (no charging on a breakdown);
  * easy / medium / max run categorisation — removed;
  * synthetic route/solar/wind stand-ins and the fork `solve()` — removed.

What remains:
  * `BreakdownModel` — the per-substep hazard model forward_sim already imports
    (kept BIT-IDENTICAL so nothing downstream changes; it is gated OFF by
    default via solver_config.INCLUDE_BREAKDOWN_IN_TIME / SKIP_BREAKDOWN_WHEN_UNUSED).
  * `DailyBreakdown` — NEW: the realistic "one breakdown per day" scenario the
    strategist asked for. Exactly one breakdown per day, its duration drawn from
    a 0-to-60-minute PDF whose shape is driven by how much POWER the car is
    pulling (harder driving -> longer expected downtime). NO charging happens
    during it — it is pure lost time. Enabled per-run with `--breakdown` and
    applied to every day.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# ============================================================================
# Per-substep hazard model (unchanged — forward_sim imports this)
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
        # 0 below 2000 W, sharp cubic rise from 2000->4100 W to 1.0
        prob_fn=lambda s: (
            0.0 if s <= 2000.0
            else (1.0 if s >= 4100.0 else 0.05 + 0.95 * ((s - 2000.0) / 2100.0) ** 3)
        ),
    ),
]


class BreakdownModel:
    """Per-substep hazard model used (optionally) by forward_sim.

    Deterministic path (expected_stop_s) integrates hazard*duration for the
    optimizer's objective; stochastic path (sample_stop_s) does a real Bernoulli
    draw. Both read the same risk state, advanced by .step() once per substep.
    Kept verbatim from the pre-revamp file so forward_sim behaviour is unchanged
    (it is gated off by default via solver_config).
    """

    def __init__(self, scenarios: Optional[List[FailureScenario]] = None):
        self.scenarios = list(scenarios) if scenarios is not None else list(DEFAULT_SCENARIOS)
        self._r: Dict[str, float] = {s.category: 0.0 for s in self.scenarios}

    def reset(self) -> None:
        self._r = {s.category: 0.0 for s in self.scenarios}

    def hazard_rate(self, inputs: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for sc in self.scenarios:
            s = inputs.get(sc.input_key, 0.0)
            p_base = _clip01(sc.prob_fn(s))
            r = self._r.get(sc.category, 0.0)
            out[sc.category] = p_base * math.exp(-sc.k_suppress * r)
        return out

    def expected_stop_s(self, inputs: Dict[str, float]) -> float:
        total = 0.0
        for sc in self.scenarios:
            s = inputs.get(sc.input_key, 0.0)
            p_base = _clip01(sc.prob_fn(s))
            r = self._r.get(sc.category, 0.0)
            p_eff = p_base * math.exp(-sc.k_suppress * r)
            total += p_eff * sc.duration_s
        return total

    def sample_stop_s(self, inputs: Dict[str, float], rng: random.Random) -> Tuple[float, Optional[str]]:
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
# NEW: one-breakdown-per-day scenario (strategist directive 23/08)
# ============================================================================

@dataclass
class DailyBreakdown:
    """Exactly ONE breakdown per day, duration drawn from a 0..max_seconds PDF
    whose shape is set by how hard the car is driving that day.

    The idea (strategist): a breakdown is a realistic what-if we should be able
    to switch on for every day and see how the plan copes. It is NOT part of the
    optimizer's objective — it's a scenario overlay applied to the final plan:
    one stationary stop of a sampled duration, **no charging** during it (pure
    lost time), which pushes the finish clock later and can trigger the normal
    SR 2.22.6 late penalty (which then carries into the next day exactly like any
    other late finish).

    Duration PDF — a triangular distribution on [0, max_seconds]:
      * low mode  (short breakdowns) when the day's power draw is light;
      * mode rises toward max_seconds as the day's average power approaches the
        car's continuous power limit (harder driving -> longer expected downtime).
    Triangular is used because it is bounded (0..1 h exactly), needs only a mode,
    and its mean has a closed form so `expected_seconds` is exact for the
    deterministic/reporting path.

    Reproducible: pass a seeded `random.Random`. One draw per day (seed it with
    a per-day offset so different days get independent — but repeatable — draws).
    """
    max_seconds: float = 3600.0            # 1 hour hard cap (reached only at full power)
    mode_frac: float = 0.5                 # triangle peak, as a fraction of the
                                           # power-scaled upper bound (0.5 -> mean
                                           # sits at half the upper bound)

    def power_fraction(self, power_w: float, p_ref_w: float) -> float:
        """How stressed the drivetrain is, in [0, 1] — average (or peak) power
        as a fraction of the car's continuous power limit."""
        if p_ref_w <= 0.0:
            return 0.0
        return _clip01(power_w / p_ref_w)

    def _high_seconds(self, power_w: float, p_ref_w: float) -> float:
        """Worst-case breakdown duration for this power level. The whole 0..1 h
        range is SCALED by the power fraction, so a light day tops out at a few
        minutes and only a flat-out day can reach the full hour — 'duration based
        on the power we're consuming'."""
        return self.max_seconds * self.power_fraction(power_w, p_ref_w)

    def expected_seconds(self, power_w: float, p_ref_w: float) -> float:
        """Mean of the triangular(0, high, mode) PDF — deterministic estimate.
        With mode_frac=0.5 this is simply high/2 = max*frac/2."""
        high = self._high_seconds(power_w, p_ref_w)
        mode = high * self.mode_frac
        return (0.0 + high + mode) / 3.0

    def sample_seconds(self, power_w: float, p_ref_w: float,
                       rng: random.Random) -> float:
        """One stochastic breakdown-duration draw in [0, power-scaled high]."""
        high = self._high_seconds(power_w, p_ref_w)
        if high <= 0.0:
            return 0.0
        mode = high * self.mode_frac
        return float(rng.triangular(0.0, high, mode))
