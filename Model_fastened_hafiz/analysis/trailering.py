"""
analysis/trailering.py — plan-time red-flag map (Plan v3 §5.1) — WORKING for
block 1; consumed by pipeline/build_route.py in block 2.2.

Red-flag criterion (SR 2.32.2 via race_config.TRAILERING_MIN_SPEED_MS):
grid points where holding 60 km/h needs more than the derated continuous
motor power. Uses core.physics.power_required_at_speed (2026 model); the
Kevin/Ramana motor-level model (physics.motor_power_kr) is the
high-fidelity cross-check for thermal margin on sustained climbs
(notes doc idea iv).
"""
from __future__ import annotations
import numpy as np
from configs.race_config import TRAILERING_MIN_SPEED_MS
from core import physics


def red_flags(car, slope_pct: np.ndarray,
              air_density: float = 1.2,
              headwind_ms: float = 0.0) -> np.ndarray:
    """Boolean red-flag array over gradient points (Plan v3 §5.1)."""
    p_req = physics.power_required_at_speed(
        car, TRAILERING_MIN_SPEED_MS, slope_pct, air_density,
        wind_along_ms=-abs(headwind_ms))
    limit = car.p_max_continuous_w * car.p_max_derating
    return p_req > limit
