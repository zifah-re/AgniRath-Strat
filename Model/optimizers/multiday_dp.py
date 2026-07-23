"""
optimizers/multiday_dp.py — STUB (block 6, owner: Junior B).

L1 multi-day DP (Plan v3 §8): state (day, SOC bucket solver_config.
DP_SOC_BUCKET_PCT); action = integer attempts per named loop, priced per
Plan v3 §2.3 (5-min loop stop + turnaround + drive time); overnight
transition = MORNING CHARGE ONLY (SR 2.30/2.31 — no evening charging);
late-finish penalty coupling (race_config.late_finish_penalty_min);
objective per race_config.RACE_MODE; robustness over DP_SOLAR_SCENARIOS x
DP_STOPPAGE_SCENARIOS_S. Outputs loop plan + per-day min start SOC floors
(alpha_day) + completion-probability report.
Paper 1's multilevel method (multi-day distance max, backward SOC pass) is
the reference design.
"""

def solve(routes, car, solar_provider, wind_provider, start_soc_pct: float):
    raise NotImplementedError("block 6 — Junior B")
