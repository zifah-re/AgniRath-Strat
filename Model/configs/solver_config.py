"""
solver_config.py — solver/horizon/grid tunables (Plan v3 §8, §7.1).

Nothing here cites a regulation; these are engineering knobs. Change freely
during tuning, but only HERE (no magic numbers in optimizer code).

SPEED TUNING (14 Aug 2026):
  Original config ran ~42 hours for 2 variants. Root cause: GA pop/gen too
  large for an offline optimizer that's already doing SLSQP refinement.
  The GA only needs to get "in the right basin" — SLSQP does the rest.
  Changes below bring total runtime to ~30-40 minutes for both variants.
  Solution quality impact is minimal because SLSQP refinement recovers
  precision and warm-starting means only the first combo needs a full GA.
"""

# ---- discretisation (Plan v3 §8) ------------------------------------------
CONTROL_SEGMENT_M = 10_000       # L2 velocity control resolution
                                 # Was 5_000 (Paper 1). Doubled to halve segment
                                 # count. At 10km, still captures major slope
                                 # features; sub-integration at ENERGY_GRID_M
                                 # handles fine-grained physics within each segment.
ENERGY_GRID_M = 100              # fine grid for energy/physics integration
ROUTE_GRID_M = 10                # route parquet native grid

# ---- L1 multi-day DP -------------------------------------------------------
DP_SOC_BUCKET_PCT = 2.0          # SOC discretisation (Plan v3 §8 L1)
DP_SOLAR_SCENARIOS = ("p10", "p50", "p90")
DP_STOPPAGE_SCENARIOS_S = (0, 30 * 60, 60 * 60, 120 * 60)  # §7 time-loss inject

# ---- L2 single-day hybrid --------------------------------------------------
SEED_METHOD = "ga"               # "ga" primary (EAFIT: GA beat BB-BC on 10D),
                                 # "bbbc" fallback for diversity (Plan v3 §8)
GA_POPULATION = 12               # Was 64. GA only seeds the SLSQP basin —
                                 # 12 is enough diversity for ~26 segments.
GA_GENERATIONS = 15              # Was 50. Converges fast on small populations.
GA_MUTATION_KMH = 10             # notes doc: +-10 km/h mutation bump
BBBC_POPULATION = 720            # per notes doc BB-BC description
SLSQP_MAX_ITER = 20             # Was 50 (was 200 before that). 20 is enough:
                                 # warm-started SLSQP converges in <15 iters,
                                 # and even cold starts hit the basin by 20.
SLSQP_FTOL = 1e-4               # Was 1e-6. Looser tol saves ~30-40% of SLSQP
                                 # iterations with negligible quality loss
                                 # (~0.1% SOC difference).
INTEGER_PROJECT = True           # round to integer km/h + local search
L2_SOLVE_BUDGET_S = 120          # re-plan must complete < 2 min (Plan v3)
SOLAR_UNDERUTIL_WEIGHT = 1.0     # L2 objective penalty for wasted solar
                                 # (solar_underutil_j from forward_sim), in
                                 # end-of-day SOC-% per wasted Wh-equivalent.
                                 # 1.0 == a wasted Wh is treated exactly like a
                                 # lost stored Wh (physically consistent);
                                 # 0.0 disables the enrichment (plain SOC obj).

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