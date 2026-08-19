"""
optimizers/hierarchical/tier3.py — Tier 3 algebraic multi-day allocator.
"""

from __future__ import annotations

import logging
import numpy as np

from configs import race_config as rc
from configs import solver_config as sc
from configs.car_config import CarState
from .tier1 import overnight_soc_gain, _completion_margin

logger = logging.getLogger(__name__)

def _base_km(plan, trailered_km: float = 0.0) -> float:
    """Base distance credited to the race, excluding mandatory trailering."""
    return max(0.0, plan.stage1_km + plan.stage2_km - float(trailered_km or 0.0))

def allocate(car: CarState, solar_providers: dict, per_day_samples: dict, plans: list,
             start_soc_pct: float, start_day: int = 0,
             trailered_km_by_day: dict[int, float] | None = None) -> dict:
    
    n_days = len(plans)
    soc_buckets = np.arange(car.soc_min_pct, car.soc_max_pct + 1e-9, sc.DP_SOC_BUCKET_PCT)
    nb = len(soc_buckets)
    completion = (rc.RACE_MODE == "completion")
    floor = car.soc_min_pct + (_completion_margin() if completion else 0.0)

    V = np.full((n_days + 1, nb), -np.inf)
    V[n_days, :] = 0.0
    policy_reps = [[None] * nb for _ in range(n_days)]
    policy_next = np.full((n_days, nb), -1, dtype=int)

    for d in range(n_days - 1, start_day - 1, -1):
        surro = per_day_samples[d]["surrogates"]
        solar_provider = solar_providers.get(d)
        gain = overnight_soc_gain(car, solar_provider, d)
        trailer_km = (trailered_km_by_day or {}).get(d, 0.0)
        base_km = _base_km(plans[d], trailer_km)

        for s_idx, s_start in enumerate(soc_buckets):
            best_val = -np.inf
            for reps, model in surro.items():
                if not model.in_window(s_start):
                    continue
                end_soc = model.predict(s_start)
                if end_soc < floor or end_soc > car.soc_max_pct + 1e-6:
                    continue

                next_soc = min(end_soc + gain, car.soc_max_pct)
                v_next = _interp(V[d + 1], soc_buckets, next_soc)
                if not np.isfinite(v_next):
                    continue

                dist = base_km + model.loop_km
                waste_wh = model.predict_underutil(s_start)
                waste_penalty_km = (
                    sc.SOLAR_UNDERUTIL_WEIGHT
                    * sc.DP_SOLAR_UNDERUTIL_EQ_SPEED_KMH
                    / 3600.0
                    * waste_wh
                )
                # Completion mode means the base race must remain feasible; it
                # must not turn the multi-day objective into "preserve SOC and
                # ignore distance". The competition objective is still distance
                # accumulated across days, with future SOC value as the coupling
                # term and solar curtailment as a secondary cost.
                val = dist + v_next - waste_penalty_km
                if val > best_val:
                    best_val = val
                    policy_reps[d][s_idx] = reps
                    policy_next[d][s_idx] = int(np.clip(
                        np.searchsorted(soc_buckets, next_soc) - 1, 0, nb - 1))
            V[d][s_idx] = best_val

    s1 = np.full(n_days, np.nan)
    loop_plan: dict[int, dict[str, int]] = {}
    total_km = 0.0
    feasible = True
    cur = float(np.clip(start_soc_pct, car.soc_min_pct, car.soc_max_pct))
    
    for d in range(start_day, n_days):
        s1[d] = cur
        s_idx = int(np.clip(np.searchsorted(soc_buckets, cur) - 1, 0, nb - 1))
        reps = policy_reps[d][s_idx]
        if reps is None or not np.isfinite(V[d][s_idx]):
            feasible = False
            loop_plan[d] = {}
            break
        model = per_day_samples[d]["surrogates"][reps]
        loop_plan[d] = {name: (reps[i] if reps else 0)
                        for i, (name, _km) in enumerate(plans[d].loops)
                        if reps and reps[i] > 0}
        trailer_km = (trailered_km_by_day or {}).get(d, 0.0)
        total_km += _base_km(plans[d], trailer_km) + model.loop_km
        end_soc = model.predict(cur)
        solar_provider = solar_providers.get(d)
        cur = min(end_soc + overnight_soc_gain(car, solar_provider, d), car.soc_max_pct)

    return dict(s1_pct=s1, loop_plan=loop_plan, total_distance_km=total_km, feasible=feasible)

def _interp(v_row: np.ndarray, buckets: np.ndarray, soc: float) -> float:
    finite = np.isfinite(v_row)
    if not finite.any(): return -np.inf
    return float(np.interp(soc, buckets[finite], v_row[finite]))