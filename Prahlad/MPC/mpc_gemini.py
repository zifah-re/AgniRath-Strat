import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ==========================================
# 1. VEHICLE CONFIGURATION & CONSTANTS
# ==========================================
MASS = 300.0          # Total car + driver mass (kg)
CDA = 0.16            # Aerodynamic drag area (Cd * A)
CRR = 0.007           # Rolling resistance coefficient
RHO = 1.2             # Air density (kg/m^3)
G = 9.81              # Gravity (m/s^2)

# Efficiencies (From user specs)
SOLAR_AREA = 6.0      # m^2
SOLAR_EFF = 0.22      # 22%
MOTOR_EFF = 0.95      # 95%
REGEN_EFF = 0.70      # 70%
POWER_LOSS = 70

BATT_CAPACITY_WH = 3528.0  # 3.5 kWh Battery Pack
DT = 120.0                 # 2-minute steps (120 seconds per step)
N = 10                     # 10-step look-ahead horizon (20 minutes total)

# ==========================================
# 2. PHYSICS ENGINE & POWER EQUATIONS
# ==========================================
def calculate_net_power(v_current, v_next, slope_rad, solar_irradiance):
    # Force required to move the car based on physics
    f_drag = 0.5 * RHO * CDA * (v_current ** 2)
    f_rolling = MASS * G * CRR * np.cos(slope_rad)
    f_gravity = MASS * G * np.sin(slope_rad)
    f_acceleration = MASS * (v_next - v_current) / DT
    
    f_total = f_drag + f_rolling + f_gravity + f_acceleration
    
    # 1. Solar Input Power
    p_solar = SOLAR_AREA * SOLAR_EFF * solar_irradiance
    
    # 2. Mechanical Power to Electrical Power conversion
    p_mech = f_total * v_current
    
    if p_mech >= 0:
        # Motor drawing power from battery
        p_electric = p_mech / MOTOR_EFF
    else:
        # Regenerative braking pushing power back into battery
        p_electric = p_mech * REGEN_EFF
        
    # Net Electrical Power flow out of/into battery
    net_power_watts = p_solar - p_electric - POWER_LOSS
    return net_power_watts

# ==========================================
# 3. MPC OBJECTIVE FUNCTION
# ==========================================
def mpc_cost_function(v_horizon, current_soc, current_v, target_profile, terrain_profile, solar_profile):
    cost = 0.0
    soc = current_soc
    v_prev = current_v
    
    for i in range(N):
        v_next = v_horizon[i]
        
        # Calculate battery power dynamics for this step
        p_net = calculate_net_power(v_prev, v_next, terrain_profile[i], solar_profile[i])
        
        # Convert Watt-seconds to Wh and update SoC percentage
        energy_change_wh = (p_net * DT) / 3600.0
        soc += (energy_change_wh / BATT_CAPACITY_WH) * 100.0
        
        # PENALTY 1: Staying close to strategic target velocity
        cost += 1.0 * (v_next - target_profile[i]) ** 2
        
        # PENALTY 2: Severe penalty for draining battery below safe limits
        if soc < 20.0:
            cost += 1000.0 * (20.0 - soc) ** 2
            
        # PENALTY 3: Driver comfort / drivetrain smoothing
        cost += 0.5 * (v_next - v_prev) ** 2
        
        v_prev = v_next
        
    return cost

# ==========================================
# 4. SIMULATION ENVIRONMENT RUN
# ==========================================
# Let's simulate a 2-hour race segment (60 steps of 2-mins)
sim_duration_steps = 60
time_axis = np.arange(sim_duration_steps) * (DT / 60.0) # In minutes

# Environmental Conditions across the entire race segment
# Imagine a rolling hill terrain profile (slopes in radians)
race_terrain = 0.02 * np.sin(np.linspace(0, 4 * np.pi, sim_duration_steps))
# Passing clouds drop solar irradiance from 900 down to 200 W/m^2 midway
race_solar = np.ones(sim_duration_steps) * 900.0
race_solar[25:40] = 200.0 

# High-level Strategic target velocity profile (m/s) from baseline strategy
# Cruising at ~72 km/h (20 m/s) but drops to 15 m/s later on schedule
strategic_v_targets = np.ones(sim_duration_steps) * 20.0
strategic_v_targets[30:] = 15.0

# Telemetry tracking logs
history_v = [18.0] # Starting race speed (m/s)
history_soc = [80.0] # Starting battery State of Charge (%)

print("Deploying Solar MPC...")

for step in range(sim_duration_steps - 1):
    # Grab current telemetry variables
    current_v = history_v[-1]
    current_soc = history_soc[-1]
    
    # Extract upcoming environmental segments for the short 10-step horizon window
    if step + N <= sim_duration_steps:
        target_window = strategic_v_targets[step:step+N]
        terrain_window = race_terrain[step:step+N]
        solar_window = race_solar[step:step+N]
    else:
        # Pad window arrays if nearing final race boundaries
        pad_size = N - (sim_duration_steps - step)
        target_window = np.pad(strategic_v_targets[step:], (0, pad_size), 'edge')
        terrain_window = np.pad(race_terrain[step:], (0, pad_size), 'edge')
        solar_window = np.pad(race_solar[step:], (0, pad_size), 'edge')
        
    # Bounds: Solar car speed range (0 to 35 m/s)
    speed_bounds = [(16.67, 25.0) for _ in range(N)]
    u_guess = np.ones(N) * current_v
    
    # Run tactical localized window optimization
    result = minimize(
        mpc_cost_function, u_guess, 
        args=(current_soc, current_v, target_window, terrain_window, solar_window),
        bounds=speed_bounds, method='SLSQP'
    )
    
    # Extract tactical target command for the driver
    optimal_next_v = result.x[0]
    
    # Evaluate physics to get actual real-world telemetry results
    p_net_actual = calculate_net_power(current_v, optimal_next_v, race_terrain[step], race_solar[step])
    next_soc = current_soc + ((p_net_actual * DT) / 3600.0 / BATT_CAPACITY_WH) * 100.0
    
    # Record data
    history_v.append(optimal_next_v)
    history_soc.append(next_soc)

print("Race simulation run successful.")

# ==========================================
# 5. DATA VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time_axis, history_v, 'b-o', label='MPC Recommended Velocity')
plt.plot(time_axis, strategic_v_targets, 'r--', label='Strategic Baseline Target')
plt.ylabel('Velocity (m/s)')
plt.title('Solar Car Closed-Loop Tactical MPC Response')
plt.grid(True); plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_axis, history_soc, 'g-', linewidth=2, label='Battery Pack SoC (%)')
plt.axhline(20.0, color='r', linestyle=':', label='Critical Low Limit')
plt.ylabel('Battery SoC (%)')
plt.grid(True); plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_axis, race_solar, 'y-', label='Solar Irradiance (W/m²)')
plt.fill_between(time_axis, race_terrain*5000, color='gray', alpha=0.3, label='Terrain Slope Elevation')
plt.ylabel('Environment Context')
plt.xlabel('Race Time Elapsed (Minutes)')
plt.grid(True); plt.legend()

plt.tight_layout()
plt.show()