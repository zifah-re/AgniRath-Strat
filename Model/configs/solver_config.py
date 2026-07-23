"""
solver_config.py — solver/horizon/grid tunables (Plan v3 §8, §7.1).

Nothing here cites a regulation; these are engineering knobs. Change freely
during tuning, but only HERE (no magic numbers in optimizer code).
"""

# ---- discretisation (Plan v3 §8) ------------------------------------------
CONTROL_SEGMENT_M = 5_000        # L2 velocity control resolution (Paper 1: 5 km)
ENERGY_GRID_M = 100              # fine grid for energy/physics integration
ROUTE_GRID_M = 10                # route parquet native grid

# ---- L1 multi-day DP -------------------------------------------------------
DP_SOC_BUCKET_PCT = 2.0          # SOC discretisation (Plan v3 §8 L1)
DP_SOLAR_SCENARIOS = ("p10", "p50", "p90")
DP_STOPPAGE_SCENARIOS_S = (0, 30 * 60, 60 * 60, 120 * 60)  # §7 time-loss inject

# ---- L2 single-day hybrid --------------------------------------------------
SEED_METHOD = "ga"               # "ga" primary (EAFIT: GA beat BB-BC on 10D),
                                 # "bbbc" fallback for diversity (Plan v3 §8)
GA_POPULATION = 64
GA_GENERATIONS = 50              # per notes doc (50 iterations)
GA_MUTATION_KMH = 10             # notes doc: +-10 km/h mutation bump
BBBC_POPULATION = 720            # per notes doc BB-BC description
SLSQP_MAX_ITER = 200
SLSQP_FTOL = 1e-6
INTEGER_PROJECT = True           # round to integer km/h + local search
L2_SOLVE_BUDGET_S = 120          # re-plan must complete < 2 min (Plan v3)

# ---- L3 MPC ----------------------------------------------------------------
MPC_HORIZON_STEPS = 10           # source: Dashboard mpc.py N=10 baseline
MPC_BACKEND = "slsqp"            # "slsqp" baseline | "ipopt" challenger; ipopt
                                 # promoted only if it wins the benchmark
                                 # (Plan v3 §8)
PERIODIC_RESOLVE_INTERVAL_S = 20 * 60    # §7.1 routine closed-loop re-solve
                                         # cadence (15-30 min band; tune)
REPLAN_SOC_DEVIATION_PCT = 5.0           # threshold trigger: |SOC - SOC*(x)|
REPLAN_SOC_DEVIATION_HOLD_S = 10 * 60    # ...sustained this long
REPLAN_FORECAST_REVISION_PCT = 15.0      # threshold trigger: Solcast revision

# ---- solar/wind spatial scheme (Plan v3 §6.1) ------------------------------
CIRCLE_TARGET_DIAMETER_KM = 25.0  # ~ irradiance decorrelation scale guess
                                  # (20-30 km band; tune vs collected data)

# ---- weather refresh cadence ----------------------------------------------
PULL_EVENING_LOCAL_H = 19         # evening pull -> tomorrow's DP/L2
PULL_MORNING_LOCAL_H = 6          # morning pull -> final fit before start
NOWCAST_INTERVAL_S = 30 * 60      # intra-day nowcast pulls (MPC only)
