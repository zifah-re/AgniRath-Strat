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
ENERGY_GRID_M = 150              # fine grid for energy/physics integration.
                                 # Was 100. Coarsened to 150 m after the
                                 # forward_sim vectorization: on real routes the
                                 # end-of-day SOC/motor/solar differ <0.3% from
                                 # the 100 m grid (verified) while cutting substep
                                 # count ~1.5x — a real wall-clock win with
                                 # negligible accuracy loss (slopes vary on a
                                 # much longer length scale than 150 m).
ROUTE_GRID_M = 10                # route parquet native grid

# ---- L1 multi-day DP -------------------------------------------------------
DP_SOC_BUCKET_PCT = 2.0          # SOC discretisation (Plan v3 §8 L1)
DP_SOLAR_SCENARIOS = ("p10", "p50", "p90")
DP_STOPPAGE_SCENARIOS_S = (0, 30 * 60, 60 * 60, 120 * 60)  # §7 time-loss inject

# Realistic sustainable cruise for RESERVING base-route drive time when
# bounding how many loop attempts can fit in a day (relaxed_loop_combos).
# The old combo generator reserved ZERO time for the base Stage-1+Stage-2
# drive, so it emitted physically impossible loop counts (e.g. 8-9 attempts
# of a 40 km loop on top of a 220 km base) that could never finish by the
# cutoff — the allocator then happily picked them and the day "finished" at
# 18:30+. Reserving base_km / this_speed leaves only the genuinely spare time
# for loops. 65 km/h is the strategist's target sustainable cruise (fast
# enough to bank distance, slow enough to be energy-feasible on most days).
DP_BASE_PLANNING_SPEED_KMH = 65.0

# Future-SOC value discount in the Tier 3 allocator (0 < d <= 1). The DP's
# value-to-go term rewards ending a day with high SOC (more options tomorrow),
# but SOC banked beyond what later days can actually spend before their own
# ceilings clip is worthless — so the raw DP hoards charge and skips loops on
# good days (the "Day 5: 1 loop, 50 km/h, ends 99.8%" complaint). A discount
# < 1 makes distance banked TODAY worth marginally more than SOC carried to
# tomorrow, so the allocator takes a loop whenever its km beat the discounted
# future value. Strategist directive (21/08): "SOC conservation matters, but
# not at the cost of more distance." 1.0 = pure Bellman (hoards); lower =
# more aggressive distance. Keep gentle so late days stay feasible.
DP_FUTURE_VALUE_DISCOUNT = 0.93

# ---- Late-finish pricing in the Tier 3 allocator ---------------------------
# The strategist's directive (20/08): "arriving by 17:00 is good and must be
# followed more or less — only run past it if the extra distance is genuinely
# worth the next-day penalty." Tier 3 prices each candidate loop plan's
# finish time: arriving after day_finish_time_s (17:00, or 15:00 on Day 8)
# costs the SR 2.22.6 penalty, converted to a km-equivalent via the day's own
# realized average speed, and subtracted from that day's distance value. This
# makes the allocator stop adding loops around 17:00 and only exceed it when a
# loop's marginal km beats the penalty (the future-SOC value term already
# encodes "unless the next day needs the banked energy").
LATE_FINISH_PENALTY_ENABLED = True
# Combos finishing after this many minutes past the on-time target are still
# priced with the penalty; combos past the absolute cutoff are rejected
# upstream in tier2 (_l2_result_feasible). Kept explicit so the gradient and
# the hard gate are tuned in one place.
LATE_FINISH_MAX_LATE_MIN = 60.0

# ---- Dashboard continuous-output resolution --------------------------------
# The coarse per-control-segment velocity_profile_kmh stays the driver card
# (Plan v3 §8: one target speed per segment). For the DASHBOARD, forward_sim's
# fine per-substep traces (soc/velocity/solar/slope vs distance) are exported,
# downsampled to roughly one point per OUTPUT_TRACE_STRIDE_M metres so the JSON
# stays a sane size while the curves still read as continuous.
OUTPUT_TRACE_STRIDE_M = 250.0

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
DP_SOLAR_UNDERUTIL_EQ_SPEED_KMH = 65.0  # Tier-3 equivalent race speed for solar penalty
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