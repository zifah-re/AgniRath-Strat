# mpc.py
import numpy as np
from scipy.optimize import minimize

# ==========================================
# VEHICLE CONFIGURATION & CONSTANTS
# ==========================================
MASS = 300.0          # Total car + driver mass (kg)
CDA = 0.16            # Aerodynamic drag area (Cd * A)
CRR = 0.007           # Rolling resistance coefficient
RHO = 1.2             # Air density (kg/m^3)
G = 9.81              # Gravity (m/s^2)

SOLAR_AREA = 6.0      # m^2
SOLAR_EFF = 0.22      # 22%
MOTOR_EFF = 0.95      # 95%
REGEN_EFF = 0.70      # 70%
POWER_LOSS = 70.0

BATT_CAPACITY_WH = 3528.0  # Battery Pack Capacity
DT = 120.0                 # 2-minute steps
N = 10                     # 10-step horizon

def calculate_net_power(v_current, v_next, slope_rad, solar_irradiance):
    f_drag = 0.5 * RHO * CDA * (v_current ** 2)
    f_rolling = MASS * G * CRR * np.cos(slope_rad)
    f_gravity = MASS * G * np.sin(slope_rad)
    f_acceleration = MASS * (v_next - v_current) / DT
    
    f_total = f_drag + f_rolling + f_gravity + f_acceleration
    p_solar = SOLAR_AREA * SOLAR_EFF * solar_irradiance
    p_mech = f_total * v_current
    
    if p_mech >= 0:
        p_electric = p_mech / MOTOR_EFF
    else:
        p_electric = p_mech * REGEN_EFF
        
    net_power_watts = p_solar - p_electric - POWER_LOSS
    return net_power_watts

def mpc_cost_function(v_horizon, current_soc, current_v, target_profile, terrain_profile, solar_profile):
    cost = 0.0
    soc = current_soc
    v_prev = current_v
    
    for i in range(N):
        v_next = v_horizon[i]
        p_net = calculate_net_power(v_prev, v_next, terrain_profile[i], solar_profile[i])
        
        energy_change_wh = (p_net * DT) / 3600.0
        soc += (energy_change_wh / BATT_CAPACITY_WH) * 100.0
        
        # Penalties
        cost += 1.0 * (v_next - target_profile[i]) ** 2
        if soc < 20.0:
            cost += 1000.0 * (20.0 - soc) ** 2
        cost += 0.5 * (v_next - v_prev) ** 2
        
        v_prev = v_next
        
    return cost

def compute_optimal_velocity(current_v, current_soc, target_profile, terrain_profile, solar_profile):
    """
    Executes the optimization window. 
    Expects arrays of length N for target, terrain, and solar profiles.
    """
    # Ensure inputs are correctly bounded and padded to length N
    def pad_or_truncate(arr, default_val):
        arr = list(arr) if arr is not None else []
        if len(arr) < N:
            return np.pad(arr, (0, N - len(arr)), 'constant', constant_values=default_val)
        return np.array(arr[:N])

    targets = pad_or_truncate(target_profile, current_v)
    terrain = pad_or_truncate(terrain_profile, 0.0)
    solar = pad_or_truncate(solar_profile, 500.0)

    speed_bounds = [(16.67, 25.0) for _ in range(N)] # Bounds between ~60 and 90 km/h
    u_guess = np.ones(N) * current_v
    
    result = minimize(
        mpc_cost_function, u_guess, 
        args=(current_soc, current_v, targets, terrain, solar),
        bounds=speed_bounds, method='SLSQP'
    )
    
    if result.success:
        return float(result.x[0])
    return current_v # Fallback to current speed if solver fails