import numpy as np

# ==========================================
# CAR CONSTANTS
# ==========================================
MASS_KG = 300.0
G_MS2 = 9.81
CRR = 0.007
CDA_M2 = 0.16
AIR_DENSITY = 1.2
ARRAY_AREA_M2 = 5.95
ARRAY_EFFICIENCY = 0.18

# Drivetrain
MOTOR_EFF = 0.95
REGEN_EFF = 0.70
P_AUX = 70.0  # Constant idle power loss (Watts)
P_MAX = 5000.0  # Assumed max continuous motor draw (Watts)

# Battery
BATTERY_WH = 588.0 * 6
BATTERY_JOULES = BATTERY_WH * 3600
SOC_MIN_PCT = 20.0
SOC_MAX_PCT = 100.0

# Speed Limits
V_MAX_MS = 85.0 / 3.6
V_MIN_MS = 20.0 / 3.6
A_MAX_MS = 0.5

# ==========================================
# RACE CONSTANTS
# ==========================================
# Timing (Seconds past midnight)
START_TIME_DAY1_S = 9 * 3600
START_TIME_OTHER_S = 8 * 3600
FINISH_TIME_S = 17 * 3600
FINISH_CUTOFF_ABS_S = 17 * 3600 + 30 * 60
DAY8_TIMED_FINISH_S = 15 * 3600
MORNING_CHARGE_START_S = 6 * 3600

# Stops
CONTROL_STOP_DURATION_S = 30 * 60
CONTROL_STOP_PARC_FERME_S = 25 * 60
LOOP_STOP_DURATION_S = 5 * 60
DRIVER_SWAP_INTERVAL_S = 120 * 60

# Regulations
LATE_PENALTY_THRESHOLD_MIN = 10
LATE_PENALTY_MULTIPLIER_1 = 1
LATE_PENALTY_MULTIPLIER_2 = 2
TRAILER_OFFLOAD_DIST_M = 500.0

# Route Notes (Subset defining stage structure & loop dists)
DAY_ROUTE_NOTES = [
    dict(stage1_km=172.7, loops=[("rustenburg_loop", 22.6)], stage2_km=65.6),
    dict(stage1_km=71.5, loops=None, stage2_km=231.0),
    dict(stage1_km=None, loops=None, stage2_km=None), # Blind
    dict(stage1_km=197.0, loops=[("postmasburg_loop2", 21.0), ("postmasburg_loop1", 14.0)], stage2_km=63.3),
    dict(stage1_km=178.0, loops=[("upington_loop1", 62.0), ("upington_loop2", 34.0)], stage2_km=114.0),
    dict(stage1_km=310.0, loops=[("springbok_loop", 18.2)], stage2_km=0.0),
    dict(stage1_km=261.0, loops=[("vanrhynsdorp_loop", 16.5)], stage2_km=80.9),
    dict(stage1_km=180.0, loops=[("ceres_loop", 21.8)], stage2_km=98.3)
]

# ==========================================
# SOLVER CONSTANTS
# ==========================================
SLSQP_FTOL = 1e-6
SLSQP_EPS = 1.49e-08
MAX_ITER = 500
DS_RESOLUTION_M = 5000.0  # 5 km high-resolution route intervals