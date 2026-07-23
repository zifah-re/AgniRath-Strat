"""
optimizers/singleday.py — STUB (block 5, owner: Junior B).

L2 hybrid single-day optimizer (Plan v3 §8): GA seed (primary; EAFIT found
GA > BB-BC on 10D) -> SLSQP polish -> integer km/h projection + local
search (Paper 1: executable optimum). Objective: maximise end-of-day SOC at
fixed committed distance; constraints incl. terminal SOC >= alpha_day,
v <= v_max(x), |a| <= a_max, P <= derated P_max, timing (CS + loop stops +
swaps + race_config.UNPLANNED_STOP_BUDGET_S), solar under-utilisation
penalty (Paper 4: penalise Pc < Ps), end-of-day charge-vs-cross interval
variable. Solve budget: solver_config.L2_SOLVE_BUDGET_S.
"""

def solve(route, car, solar_provider, wind_provider, day_index: int,
          start_soc_pct: float, alpha_next_day_pct: float, loops_committed):
    raise NotImplementedError("block 5 — Junior B")
