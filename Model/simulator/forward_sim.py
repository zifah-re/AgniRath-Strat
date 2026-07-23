"""
simulator/forward_sim.py — THE forward integrator (block 3.1, owner:
Junior C). No private integrator copies anywhere else (Plan v3 §8).

Minimal working core below so blocks 0-1 tests can integrate energy today;
Junior C extends with stops, charging intervals, vehicle_state, telemetry
replay. Every optimizer candidate is evaluated through THIS module.
"""
from __future__ import annotations
import numpy as np
from core import physics
from core.battery import Battery


def simulate_constant_speed(car, route_slope_pct: np.ndarray,
                            seg_len_km: float, v_ms: float,
                            solar_provider, t0_s: float,
                            start_soc_pct: float = 100.0) -> dict:
    """Integrate a constant-speed run over uniform segments (smoke-level).

    Returns dict(t_s, soc_pct_series, final_soc_pct). Wind/geometry hooks
    arrive with blocks 4.x; this is the block-1 'Day-1 constant speed
    matches hand-calc' harness.
    """
    bat = Battery(car, start_soc_pct)
    t = float(t0_s)
    soc_series = [bat.soc_pct]
    n = len(route_slope_pct)
    for i in range(n):
        ghi = solar_provider.ghi_wm2(t, i * seg_len_km * 1000.0)
        p_net, dt_s = physics.net_power(
            car, v_ms, v_ms, route_slope_pct[i], ghi, seg_len_km)
        bat.apply_energy_wh(float(p_net) * float(dt_s) / 3600.0)
        t += float(dt_s)
        soc_series.append(bat.soc_pct)
    return dict(t_s=t, soc_pct_series=np.array(soc_series),
                final_soc_pct=bat.soc_pct)
