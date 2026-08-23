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
ENERGY_GRID_M = 200              # FASTENED (was 150): coarser substep, <0.3% SOC diff, ~1.3x fewer substeps.
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
DP_BASE_PLANNING_SPEED_KMH = 65.0   # strategist directive (21/08): the car
                                    # CANNOT sustain ~78 km/h — that was the
                                    # opposite extreme from the old ~48 km/h
                                    # crawl. The realistic sustainable cruise is
                                    # a 60-70 km/h AVERAGE, so the base route is
                                    # budgeted at 65 (mid-band) and the per-
                                    # segment target is hard-capped at
                                    # SUSTAINABLE_CRUISE_KMH below. Reserving
                                    # base time at the speed the car actually
                                    # drives keeps the loop budget honest.

# ---- Speed band: hard ceiling + soft comfort penalty (directive 22/08) ------
# UPDATED 22/08 after the strategist clarified: the 60-70 target was the
# *average*, NOT a hard instantaneous cap. The car CAN touch ~85 km/h briefly;
# it just must not be *encouraged* to live up there. So the old single hard cap
# at 70 is replaced by a two-part scheme that keeps the day-AVERAGE in 60-70
# while allowing short bursts to 85 only when they genuinely pay off:
#
#   1) V_MAX_HARD_KMH — the absolute ceiling the optimizer may never exceed
#      (bounds ub, on top of route speed-limit and car v_max). This is the
#      "only if needed" ceiling.
#   2) A soft, convex speed penalty added to the L2 objective (in singleday):
#        * gentle above CRUISE_COMFORT_KMH  — keeps the normal cruise in-band so
#          the average stays ~65-68 instead of drifting up to the ceiling;
#        * steep above CRUISE_SOFT_CAP_KMH  — makes anything above ~75 expensive,
#          so the car only reaches 75-85 when the time saved (fitting a loop,
#          meeting the cutoff/SOC) outweighs the penalty.
#   The penalty is a smooth function of the per-segment target speed, so SLSQP
#   still has clean gradients and convergence is unaffected.
#
# Net effect: normal driving sits in the 60-70 average band with a ~75 soft
# ceiling; 75-85 is available but rare and self-limiting. Raise the weights to
# pull the average down / discourage high speed harder; lower them to let the
# car use its top end more freely.
V_MAX_HARD_KMH = 85.0            # absolute instantaneous ceiling ("if needed")
CRUISE_SOFT_CAP_KMH = 75.0       # normal ceiling — steep penalty above this
CRUISE_COMFORT_KMH = 70.0        # free-cruise centre — gentle penalty above this
# Penalty weights: equivalent seconds of objective cost per (km/h above the
# threshold)². CRUISE_COMFORT is set at ~70 so the free-cruise optimum lands
# right where the old hard-70 cap put it (segments settle ~71, day-average
# ~65-68, comfortably in 60-70) — i.e. the great round-4 results are preserved,
# just no longer pinned to a wall — while the steep soft-cap term keeps 75-85
# rare and self-limiting, reached only when a loop/cutoff/SOC makes it worth it.
SPEED_COMFORT_PENALTY_WEIGHT = 3.0    # gentle zone (> CRUISE_COMFORT_KMH)
SPEED_SOFTCAP_PENALTY_WEIGHT = 30.0   # steep zone (> CRUISE_SOFT_CAP_KMH)

# ---- Trailer / tow logistics ----------------------------------------------
# Speed the tow vehicle moves the car through trailered stretches (mandatory
# trailering on Day 7 Stage 2 / Day 8 Stage 1, plus any red-flag safety zones).
# The car draws/stores NO energy while trailered (inert cargo), but this time
# IS counted into the day's ETA, so the value matters. Strategist-set (22/08):
# keep 80 km/h (a conservative highway tow) rather than 90.
TRAILER_TOW_SPEED_KMH = 80.0

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

# ---- Battery-safety / high-SOC discouragement (strategist directive 21/08) --
# "We cannot run the car at ~100% SOC for long — the pack will cook. Heavily
# discourage it, and if a day is predicted to END high, don't top the pack up
# the next morning." These knobs implement that as a soft band, NOT a hard cap
# (physics still clips at soc_max_pct):
#   * SOC_SAFE_MAX_PCT — the top of the comfortable band. Time spent above it
#     inside a day is penalized in the L2 objective (so the car drives faster /
#     spends charge instead of coasting at the ceiling), and the DP is penalized
#     for ENDING a day above it (so it prefers spending SOC on loops/distance).
#   * MORNING_CHARGE_SKIP_ABOVE_PCT — if the previous day is predicted to end
#     at/above this, the morning charge is skipped entirely (you don't need it
#     and topping a near-full pack is exactly the unsafe case). Between
#     SAFE_MAX and this, the morning charge is capped so start SOC never exceeds
#     SOC_SAFE_MAX_PCT.
SOC_SAFE_MAX_PCT = 90.0
MORNING_CHARGE_SKIP_ABOVE_PCT = 90.0
# L2 objective weight: equivalent seconds of "cost" per (SOC-fraction-second)
# spent above SOC_SAFE_MAX_PCT. Bigger = the optimizer works harder to keep the
# pack out of the danger band (drives faster to draw it down). Tune vs the time
# and solar-underutil terms (both also in seconds).
SOC_HIGH_PENALTY_WEIGHT = 4.0
# DP (Tier 3) penalty: km docked per SOC-% that a day ENDS above SOC_SAFE_MAX_PCT.
# Nudges the allocator toward more loops / higher speed instead of banking an
# unusable, unsafe charge surplus into the next day.
DP_HIGH_SOC_END_PENALTY_KM_PER_PCT = 1.5

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
GA_GENERATIONS = 12              # FASTENED (was 15): GA only seeds the SLSQP basin; SLSQP recovers precision.
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
# ---- Breakdown risk in the optimization objective (21/08 root-cause fix) ----
# The deterministic expected-breakdown-time term was dominating total_time_s and
# preventing the optimizer from ever driving fast (see diag: pure-drive time
# halves from 45->85 km/h but expected-breakdown time explodes, so the total the
# optimizer minimizes is flat). Breakdown is now a REPORTED risk, not a governor
# on speed. Set True only if you deliberately want the risk-averse behaviour.
INCLUDE_BREAKDOWN_IN_TIME = False

# PERFORMANCE knob (safe): when the deterministic objective does NOT fold
# breakdown into the clock (INCLUDE_BREAKDOWN_IN_TIME = False) and this run is
# deterministic (rng is None), the per-substep BreakdownModel calls compute a
# value that is then discarded — pure wasted work (~one Python call pair per
# substep, thousands of substeps per candidate, across every GA/SLSQP eval).
# Setting this True skips those calls entirely in exactly that case. It changes
# NO optimization result; the only visible effect is that the *reported*
# total_breakdown_s diagnostic reads 0 for deterministic runs (the risk figure
# only means something on the stochastic scenario/robustness runs anyway, which
# pass an rng and are never skipped). Left False here so the baseline model's
# reporting is unchanged; the hardware-fastened build flips it True.
SKIP_BREAKDOWN_WHEN_UNUSED = True   # FASTENED (was False): skip discarded breakdown calls (~1.23x/sim, bit-identical)

# PERFORMANCE knob (hardware-fastened build): Tier 2 samples the days in
# parallel. The baseline uses THREADS, which the GIL throttles because
# forward_sim's integrator is a Python loop — so multi-core scaling is poor
# (only scipy's compiled SLSQP releases the GIL). Setting this True switches the
# day-level fan-out to PROCESSES (ProcessPoolExecutor), giving true multi-core
# scaling: on an 8-16 core machine the Tier 2 phase — the dominant cost — drops
# roughly in proportion to min(n_days, cores). All the objects that cross the
# process boundary (Route, CarState, the spline/KDTree weather providers, the
# loop-geometry DataFrames) are picklable, so results are identical; only the
# wall-clock changes. Left False in the baseline (thread path, unchanged); the
# fastened build flips it True. See Model_fastened_hafiz/README_PERF.md.
TIER2_USE_PROCESS_POOL = True   # FASTENED (was False): true multi-core Tier 2 (biggest win on 8-16 cores)

# ---- Early-finish (evening) charging (strategist directive 23/08) ----------
# If a day finishes before the 17:00 close, the panel keeps charging the pack
# from the finish moment until 17:00 and that energy banks into the next day.
# Modelled in extract_final_profiles via tier1.evening_soc_gain. On by default
# (it's physically real free energy); set False to reproduce the old behaviour
# where an early finish banked nothing.
EVENING_CHARGE_ENABLED = True

# ---- One-breakdown-per-day scenario (strategist directive 23/08) -----------
# Enabled per-run with the `--breakdown` CLI flag (off by default). When on,
# EVERY day gets exactly one breakdown: a stationary stop whose duration is
# drawn from a 0..BREAKDOWN_MAX_SECONDS PDF scaled by the day's power draw (see
# core.options.DailyBreakdown). NO charging happens during it — pure lost time,
# which pushes the finish later and can trigger the normal SR 2.22.6 late
# penalty (carried into the next day like any late finish). Deterministic given
# BREAKDOWN_SEED (each day seeded with the base seed + its index).
BREAKDOWN_MAX_SECONDS = 3600.0     # 1 hour hard cap on a single day's breakdown
BREAKDOWN_SEED = 20260823          # base RNG seed for reproducible breakdown draws

