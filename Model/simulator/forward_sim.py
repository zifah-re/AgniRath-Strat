"""
simulator/forward_sim.py — the forward integrator (owner: Junior C, Plan v3 §8).
Every optimizer candidate is evaluated through this module.

This module owns the variable-speed integration loop used by the high-fidelity
Tier 2 solver. It models mandatory driver swaps, battery breakdown probabilities,
and integrates physics on the high-resolution grid.

BREAKDOWN MODEL — wired to core/options.py (12/08 fix). This used to have its
own inline simulate_breakdown() calling raw random.random() every substep,
which meant every DE/GA/SLSQP candidate evaluation saw a different fitness for
the *same* speed vector — noise the optimizer reads as "this direction is bad"
even when it isn't. That copy is deleted. We now use core.options.BreakdownModel,
called the same way core/options.py's own BreakdownModel docstring requires:
rng=None -> deterministic expected_stop_s() (smooth, no RNG, safe inside an
optimizer loop); rng=<seeded random.Random> -> one stochastic sample_stop_s()
draw per substep, for MC/scenario/robustness runs only.

SOLAR UNDERUTILIZATION TRACKING (13/08 addition):
  solar_underutil_j tracks wasted solar capacity — energy the panel could
  produce but the battery can't absorb (SOC ceiling, or motor draw < solar
  output). This feeds into singleday.py's enriched cost function.
"""

from __future__ import annotations

import dataclasses
import random
import typing as _t
import numpy as np

from configs import race_config
from configs.car_config import CarState
from core import physics
from core.battery import Battery
from core.route import Route
from core import wind as wind_core
from core.options import BreakdownModel

_DRIVER_SWAP_STANDALONE_DURATION_S = race_config.LOOP_STOP_DURATION_S
# Speed at which the trailer/tow truck moves through red-flag segments (km/h).
_TRAILER_SPEED_KMH = 80.0


@dataclasses.dataclass
class DayEvalResult:
    final_soc_pct: float
    total_time_s: float
    driver_swaps: list
    v_ms: np.ndarray
    t_s: np.ndarray
    x_m: np.ndarray
    # Total breakdown stall time folded into total_time_s above.
    # rng=None runs -> sum of expected_stop_s per substep (deterministic).
    # rng=<Random> runs -> sum of realized sample_stop_s draws (stochastic).
    breakdown_s: float = 0.0
    # Only populated on stochastic (rng given) runs — one entry per substep
    # that actually triggered a failure. Empty on deterministic runs since
    # there's no discrete "triggered" event in the expected-value path.
    breakdown_log: list = dataclasses.field(default_factory=list)
    # Solar energy the panel could produce but the system couldn't absorb (J).
    # Non-zero when: battery at SOC ceiling, or motor draw << solar output
    # and the excess can't be stored. Used by singleday.py's cost function.
    solar_underutil_j: float = 0.0
    # Trailering: number of substeps and total distance on the trailer.
    trailered_substeps: int = 0
    trailered_km: float = 0.0


class DriverSwapScheduler:
    """Tracks on-seat elapsed time and decides when SR 2.24.4 swaps land."""

    def __init__(self, swap_interval_s: float = race_config.DRIVER_SWAP_INTERVAL_S,
                 standalone_duration_s: float = _DRIVER_SWAP_STANDALONE_DURATION_S):
        self.swap_interval_s = swap_interval_s
        self.standalone_duration_s = standalone_duration_s
        self._elapsed_since_last_swap_s = 0.0
        self.swap_log: list[dict] = []

    def advance(self, dt_s: float, t_now_s: float, x_m: float,
                coincides_with_stop: bool) -> float:
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
    """Checks if the car is currently in a CS or loop stop zone to piggyback swaps."""
    if not route:
        return False
    x = route.df["distance_m"].to_numpy()
    idx = min(int(np.searchsorted(x, x_m)), len(route.df) - 1)
    row = route.df.iloc[idx]
    return bool(row["control_stop"]) or str(row["seg_type"]).startswith("loop_")


def simulate_variable_speed(v_kmh: np.ndarray, route: Route, car: CarState,
                            solar_provider, wind_provider, t0_s: float,
                            start_soc_pct: float, seg_start_m: np.ndarray,
                            seg_len_m: float, energy_grid_m: float,
                            rng: _t.Optional[random.Random] = None) -> DayEvalResult:
    """
    The centralized real integrator (reuses core.physics.net_power + core.battery.Battery).
    Velocity is held per control segment; physics integrates on the finer
    energy grid within each control segment.

    rng: None (default) -> breakdown stall time is the deterministic expected
         value (BreakdownModel.expected_stop_s), so calling this twice with the
         same v_kmh gives the exact same result — required for DE/GA/SLSQP,
         which needs a stable objective to converge on, and for DayEvaluator's
         cache to stay valid. Pass a seeded random.Random() only for scenario/
         Monte-Carlo runs (simulator/scenarios.py) where you explicitly want a
         sampled outcome, not the average.
    """
    battery = Battery(car, start_soc_pct)
    swap_scheduler = DriverSwapScheduler()
    breakdown_model = BreakdownModel()
    breakdown_model.reset()  # each call is one independent day-sim, not a
                              # continuation of the previous candidate's risk state
    t_s = float(t0_s)
    x_m = float(seg_start_m[0]) if len(seg_start_m) > 0 else 0.0
    total_breakdown_s = 0.0
    breakdown_log: list[dict] = []
    solar_underutil_j = 0.0
    trailered_substeps = 0
    trailered_km_accum = 0.0

    n_substeps = max(1, round(seg_len_m / energy_grid_m))
    substep_len_km = (seg_len_m / n_substeps) / 1000.0

    t_array = []
    x_array = []

    for v in v_kmh:
        v_ms = float(v) / 3.6
        for _ in range(n_substeps):
            slope = route.slope_pct_at(x_m) if route else 0.0
            ghi = solar_provider.ghi_wm2(t_s, x_m)

            # --- Check if this grid point is on a trailered segment ---
            is_trailered = route.red_flag_at(x_m) if route else False

            if is_trailered:
                # Car is on the trailer: NO power input, NO power drain.
                # Time is dictated by trailer speed, not the optimizer's v_ms.
                trailer_v_ms = _TRAILER_SPEED_KMH / 3.6
                dt_s_step = float(substep_len_km * 1000.0 / trailer_v_ms)
                p_net = 0.0
                trailered_substeps += 1
                trailered_km_accum += substep_len_km
            else:
                p_net, dt_s_step = physics.net_power(
                    car, v_ms, v_ms, slope, ghi, substep_len_km)

            # --- Solar underutilization tracking ---
            # Skip entirely for trailered segments (no energy flow).
            if not is_trailered:
                # p_net = p_solar - p_electric - p_idle (core/physics.py convention)
                # Available solar power at this instant:
                p_solar_w = car.array_area_m2 * car.array_efficiency * ghi
                # Total consumption (motor + idle):
                p_consumed_w = p_solar_w - float(p_net)
                # If solar exceeds consumption AND battery is near full, excess is wasted:
                solar_excess_w = max(0.0, p_solar_w - p_consumed_w)
                # Check if battery can absorb the excess (SOC headroom)
                soc_headroom_wh = (car.soc_max_pct - battery.soc_pct) / 100.0 * car.battery_nominal_wh
                absorbable_wh = soc_headroom_wh  # how much the battery can still take
                excess_wh_this_step = solar_excess_w * float(dt_s_step) / 3600.0
                if excess_wh_this_step > absorbable_wh and absorbable_wh >= 0:
                    wasted_wh = excess_wh_this_step - absorbable_wh
                    solar_underutil_j += wasted_wh * 3600.0  # convert Wh -> J

            # p_net == 0 for trailered segments → no SOC change.
            battery.apply_energy_wh(float(p_net) * float(dt_s_step) / 3600.0)

            t_array.append(t_s)
            x_array.append(x_m)

            t_s += float(dt_s_step)
            x_m += substep_len_km * 1000.0

            stop_here = _is_mandatory_stop_zone(route, x_m)
            t_s += swap_scheduler.advance(
                float(dt_s_step), t_s, x_m, coincides_with_stop=stop_here)

            # BreakdownModel call — skip for trailered segments (motor off).
            if not is_trailered:
                # Input: motor ELECTRICAL DRAW, not physics.net_power's return.
                # DEFAULT_SCENARIOS' "p_net" threshold (2000-4100 W) is the
                # motor-power BMS-trip number. Back out from the solar term:
                # p_net = p_solar - p_electric - p_idle => p_electric = p_solar - p_net - p_idle
                motor_draw_w = max(0.0, p_solar_w - float(p_net) - car.p_idle_w)
                inputs = {"p_net": motor_draw_w}
                if rng is None:
                    breakdown_s = breakdown_model.expected_stop_s(inputs)
                    breakdown_model.step(float(dt_s_step), inputs)
                else:
                    breakdown_s, triggered_category = breakdown_model.sample_stop_s(inputs, rng)
                    breakdown_model.step(float(dt_s_step), inputs,
                                          triggered_category=triggered_category,
                                          triggered_duration_s=breakdown_s)
                    if triggered_category is not None:
                        breakdown_log.append(dict(t_s=t_s, x_m=x_m,
                                                   category=triggered_category,
                                                   duration_s=breakdown_s))
                total_breakdown_s += breakdown_s
                t_s += breakdown_s

    return DayEvalResult(
        final_soc_pct=battery.soc_pct,
        total_time_s=t_s - t0_s,
        driver_swaps=swap_scheduler.swap_log,
        v_ms=v_kmh / 3.6,
        t_s=np.array(t_array),
        x_m=np.array(x_array),
        breakdown_s=total_breakdown_s,
        breakdown_log=breakdown_log,
        solar_underutil_j=solar_underutil_j,
        trailered_substeps=trailered_substeps,
        trailered_km=trailered_km_accum,
    )