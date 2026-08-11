"""
simulator/forward_sim.py — the forward integrator (owner: Junior C, Plan v3 §8).
Every optimizer candidate is evaluated through this module.

This module owns the variable-speed integration loop used by the high-fidelity
Tier 2 solver. It models mandatory driver swaps, battery breakdown probabilities,
and integrates physics on the high-resolution grid.
"""

from __future__ import annotations

import dataclasses
import random
import numpy as np

from configs import race_config
from configs.car_config import CarState
from core import physics
from core.battery import Battery
from core.route import Route

_DRIVER_SWAP_STANDALONE_DURATION_S = race_config.LOOP_STOP_DURATION_S


@dataclasses.dataclass
class DayEvalResult:
    final_soc_pct: float
    total_time_s: float
    driver_swaps: list
    v_ms: np.ndarray
    t_s: np.ndarray
    x_m: np.ndarray


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


def simulate_breakdown(p_net: float) -> float:
    """Calculates stochastic electrical breakdown time loss based on power draw."""
    inputs = {"p_net": p_net}
    scenarios = [
        {"name": "Battery Failure", "type": "Electrical", "input": "p_net", "duration": 10 * 60, 
         "prob": lambda s: 0 if s <= 2000 else (1.0 if s >= 4100 else 0.05 + 0.95 * ((s - 2000) / 2100) ** 3)}
    ]
    seed = random.random()
    stop_time = 0.0
    for scenario in scenarios:
        if seed < scenario["prob"](inputs[scenario["input"]]):
            stop_time += scenario["duration"]
            break
    return stop_time


def simulate_variable_speed(v_kmh: np.ndarray, route: Route, car: CarState,
                            solar_provider, wind_provider, t0_s: float,
                            start_soc_pct: float, seg_start_m: np.ndarray,
                            seg_len_m: float, energy_grid_m: float) -> DayEvalResult:
    """
    The centralized real integrator (reuses core.physics.net_power + core.battery.Battery).
    Velocity is held per control segment; physics integrates on the finer
    energy grid within each control segment.
    """
    battery = Battery(car, start_soc_pct)
    swap_scheduler = DriverSwapScheduler()
    t_s = float(t0_s)
    x_m = float(seg_start_m[0]) if len(seg_start_m) > 0 else 0.0

    n_substeps = max(1, round(seg_len_m / energy_grid_m))
    substep_len_km = (seg_len_m / n_substeps) / 1000.0

    t_array = []
    x_array = []

    for v in v_kmh:
        v_ms = float(v) / 3.6
        for _ in range(n_substeps):
            slope = route.slope_pct_at(x_m) if route else 0.0
            ghi = solar_provider.ghi_wm2(t_s, x_m)
            
            p_net, dt_s = physics.net_power(
                car, v_ms, v_ms, slope, ghi, substep_len_km)
            battery.apply_energy_wh(float(p_net) * float(dt_s) / 3600.0)
            
            t_array.append(t_s)
            x_array.append(x_m)
            
            t_s += float(dt_s)
            x_m += substep_len_km * 1000.0

            stop_here = _is_mandatory_stop_zone(route, x_m)
            breakdown_time = simulate_breakdown(p_net)
            t_s += swap_scheduler.advance(
                float(dt_s), t_s, x_m, coincides_with_stop=stop_here)
            t_s += breakdown_time

    return DayEvalResult(
        final_soc_pct=battery.soc_pct,
        total_time_s=t_s - t0_s,
        driver_swaps=swap_scheduler.swap_log,
        v_ms=v_kmh / 3.6,
        t_s=np.array(t_array),
        x_m=np.array(x_array)
    )