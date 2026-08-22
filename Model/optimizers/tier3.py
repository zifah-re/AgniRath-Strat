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


def _late_finish_penalty_km(model, s_start: float, day_index: int,
                            dist_km: float) -> float:
    """Km-equivalent cost of finishing after the day's on-time target.

    Implements the strategist directive (20/08): arriving by day_finish_time_s
    (17:00, or 15:00 on Day 8) is the target; finishing later costs the
    SR 2.22.6 penalty (served next day at the control stop). The penalty
    minutes are converted to a distance the car would otherwise have covered,
    using the combo's OWN realized average speed (finish - day_start over the
    distance driven) — the same "penalty * marginal rate" conversion tier1 and
    multiday_dp already use. Returns 0 when pricing is disabled, when the
    surrogate has no finish time (Tier-1 fallback), or when the combo is on
    time. Combos past the absolute cutoff are rejected upstream in tier2
    (_l2_result_feasible), so this only ever prices the [on_time, cutoff] band.
    """
    if not getattr(sc, "LATE_FINISH_PENALTY_ENABLED", True):
        return 0.0
    finish_s = model.predict_finish_s(s_start)
    if not np.isfinite(finish_s):
        return 0.0
    on_time_s = rc.day_finish_time_s(day_index)
    minutes_late = (finish_s - on_time_s) / 60.0
    if minutes_late <= 0.0:
        return 0.0
    pen_min = rc.late_finish_penalty_min(minutes_late)
    if pen_min <= 0:
        return 0.0
    day_start_s = rc.day_start_time_s(day_index)
    drive_hr = max((finish_s - day_start_s) / 3600.0, 1e-6)
    avg_speed_kmh = max(0.0, float(dist_km)) / drive_hr
    return (pen_min / 60.0) * avg_speed_kmh

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
                # Discount future SOC value so distance TODAY is preferred over
                # hoarding charge later days can't spend (see solver_config).
                v_next = float(getattr(sc, "DP_FUTURE_VALUE_DISCOUNT", 1.0)) * v_next

                dist = base_km + model.loop_km
                waste_wh = model.predict_underutil(s_start)
                waste_penalty_km = (
                    sc.SOLAR_UNDERUTIL_WEIGHT
                    * sc.DP_SOLAR_UNDERUTIL_EQ_SPEED_KMH
                    / 3600.0
                    * waste_wh
                )
                # Late-finish pricing (SR 2.22.6): arriving past the day's
                # on-time target (17:00, or 15:00 on Day 8) costs the penalty,
                # km-converted, so the allocator only runs late when a loop's
                # marginal distance beats that cost. v_next already values
                # banking SOC for a harder next day, so "arrive by 17:00 unless
                # the next day makes it worth it" falls out of the two terms.
                late_penalty_km = _late_finish_penalty_km(
                    model, s_start, d, dist)
                # Completion mode means the base race must remain feasible; it
                # must not turn the multi-day objective into "preserve SOC and
                # ignore distance". The competition objective is still distance
                # accumulated across days, with future SOC value as the coupling
                # term and solar curtailment as a secondary cost.
                val = dist + v_next - waste_penalty_km - late_penalty_km
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