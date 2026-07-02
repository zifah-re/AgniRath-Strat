from helper import get_current_state, get_profile
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
N = 10                     # 10-step horizon

def pad_or_truncate(arr, default_val):
    arr = list(arr) if arr is not None else []
    if len(arr) < N:
        return np.pad(arr, (0, N - len(arr)), 'constant', constant_values=default_val)
    return np.array(arr[:N])

def slice_profiles(profile,distance_profile,d_current,default_val):
    profile_distance=distance_profile[-1]
    distance=d_current%profile_distance
    i=-1
    while i<len(distance_profile)-1 and distance>=distance_profile[i+1]:
        i+=1
    f=(distance-distance_profile[i])/(distance_profile[i+1]-distance_profile[i])
    val_1=profile[i]
    profile=profile[i+1:len(profile)]
    profile=[val_1+f*(profile[0]-val_1)]+profile
    return pad_or_truncate(profile,default_val)

def calculate_net_power(v_current, v_next, slope_rad, solar_irradiance,seg_len):
    f_drag = 0.5 * RHO * CDA * (v_current ** 2)
    f_rolling = MASS * G * CRR * np.cos(slope_rad)
    f_gravity = MASS * G * np.sin(slope_rad)
    dt=(seg_len/v_current)*3600
    f_acceleration = MASS * (v_next - v_current) / dt
    
    f_total = f_drag + f_rolling + f_gravity + f_acceleration
    p_solar = SOLAR_AREA * SOLAR_EFF * solar_irradiance
    p_mech = f_total * v_current
    
    if p_mech >= 0:
        p_electric = p_mech / MOTOR_EFF
    else:
        p_electric = p_mech * REGEN_EFF
        
    net_power_watts = p_solar - p_electric - POWER_LOSS
    return net_power_watts

def mpc_cost_function(v_horizon, current_soc, current_v, target_profile, terrain_profile, solar_profile,distance_profile):
    cost = 0.0
    soc = current_soc
    v_prev = current_v
    
    for i in range(1,N+1):
        v_next = v_horizon[i-1]
        seg_len=distance_profile[i]-distance_profile[i-1]
        p_net = calculate_net_power(v_prev, v_next, terrain_profile[i-1], solar_profile[i-1],seg_len)
        dt=seg_len/v_prev
        energy_change_wh = (p_net * dt)
        soc += (energy_change_wh / BATT_CAPACITY_WH) * 100.0
        
        # Penalties
        cost += 1.0 * (v_next - target_profile[i-1]) ** 2
        if soc < 20.0:
            cost += 1000.0 * (20.0 - soc) ** 2
        cost += 0.5 * (v_next - v_prev) ** 2
        
        v_prev = v_next
        
    return cost

def compute_optimal_velocity(current_v, current_soc, targets, terrain, solar,distance):
    """
    Executes the optimization window. 
    Expects arrays of length N for target, terrain, and solar profiles.
    """

    speed_bounds = [(16.67, 25.0) for _ in range(N)] # Bounds between ~60 and 90 km/h
    u_guess = np.ones(N) * current_v
    
    result = minimize(
        mpc_cost_function, u_guess, 
        args=(current_soc, current_v, targets, terrain, solar,distance),
        bounds=speed_bounds, method='SLSQP'
    )
    
    if result.success:
        return result.x
    return u_guess # Fallback to current speed if solver fails

def main():
    results = get_current_state()
    current_speed = results['Speed']
    current_speed*=(5/18)  
    current_soc = results['SoC']
    current_distance = results['Distance']

    profiles = get_profile(["Gradient", "SpeedProfile", "SolarIrradiance","TargetProfile","Distance"])
    distance_profile=profiles.get("Distance")
    terrain_profile = profiles.get("Gradient", [0.0]*len(distance_profile))
    target_profile = profiles.get("TargetProfile", [current_speed]*len(distance_profile))
    solar_profile = profiles.get("SolarIrradiance", [500.0]*len(distance_profile))
    

    terrain_profile=slice_profiles(terrain_profile,distance_profile,current_distance,0)
    target_profile=slice_profiles(target_profile,distance_profile,current_distance,current_speed)
    target_profile=target_profile*(5/18)
    solar_profile=slice_profiles(solar_profile,distance_profile,current_distance,500)
    distance_profile=slice_profiles(distance_profile,distance_profile,current_distance,0)

    return compute_optimal_velocity(current_speed, current_soc, target_profile, terrain_profile, solar_profile,distance_profile)
