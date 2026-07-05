from helper import get_current_state, get_profile
import numpy as np
from scipy.optimize import minimize
from solar_table import SolarIrradiance
import pvlib
import pandas as pd
from numba import njit

# ==========================================
# VEHICLE CONFIGURATION & CONSTANTS
# ==========================================
MASS = 300.0          # Total car + driver mass (kg)
CDA = 0.16            # Aerodynamic drag area (Cd * A)
CRR = 0.007           # Rolling resistance coefficient
RHO = 1.2             # Air density (kg/m^3)
G = 9.81              # Gravity (m/s^2)

SOLAR_AREA = 5.95     # m^2
SOLAR_EFF = 0.18      # 18%
MOTOR_EFF = 0.95      # 95%
REGEN_EFF = 0.70      # 70%
POWER_LOSS = 70.0
PANEL_TILT = 4
ALBEDO = 0.2

BATT_CAPACITY_WH = 3528.0  # Battery Pack Capacity
N = 10                     # 10-step horizon

def pad_or_truncate(arr, default_val):
    arr = list(arr) if arr is not None else []
    if len(arr) < N:
        return np.pad(arr, (0, 2*N + 1 - len(arr)), 'constant', constant_values=default_val)
    return np.array(arr[:2*N+1])

def slice_profiles(profile, distance_profile, d_current, default_val):
    profile_distance = distance_profile[-1]
    distance = d_current % profile_distance
    i = -1
    while i < len(distance_profile) - 1 and distance >= distance_profile[i+1]:
        i += 1
    f = (distance - distance_profile[i]) / (distance_profile[i+1] - distance_profile[i])
    val_1 = profile[i]
    profile = profile[i+1:len(profile)]
    if not isinstance(profile[0], (list, tuple)):
        profile = [val_1 + f * (profile[0] - val_1)] + profile
    else:
        profile = [list(val_1[j] + f * (profile[0][j] - val_1[j]) for j in range(len(profile[0])))] + profile
    if default_val == "last":
        return pad_or_truncate(profile, profile[-1])
    return pad_or_truncate(profile, default_val)

@njit
def calculate_net_power(v_current, v_next, slope, solar_irradiance, seg_len):
    grad = slope / 100
    f_drag = 0.5 * RHO * CDA * (v_current ** 2)
    f_rolling = MASS * G * CRR * (1 - (grad**2)/2)
    f_gravity = MASS * G * grad
    
    # Avoid division by zero if car stops
    v_safe = max(v_current, 0.1)
    dt = max((seg_len / v_safe) * 1000, 0.01)
    
    f_acceleration = MASS * (v_next - v_current) / dt
    f_total = f_drag + f_rolling + f_gravity + f_acceleration
    p_solar = SOLAR_AREA * SOLAR_EFF * solar_irradiance
    p_mech = f_total * v_current
    
    if p_mech >= 0:
        p_electric = p_mech / MOTOR_EFF
    else:
        p_electric = p_mech * REGEN_EFF
        
    net_power_watts = p_solar - p_electric - POWER_LOSS
    return net_power_watts, dt

# --- NEW SPEEDUP: FAST PRE-COMPUTED TRACKING IRRADIANCE ---
def precompute_solar_gti_factors(time_base, coords_list, heading_profile, altitude_profile):
    """
    Computes solar positions for all horizon steps simultaneously using full vectorization.
    NO loops required.
    """
    coords_arr = np.array(coords_list)
    lats = coords_arr[:, 0]
    lons = coords_arr[:, 1]
    
    tz_times = pd.to_datetime(time_base, unit='s').tz_localize('UTC')
    
    solpos = pvlib.solarposition.get_solarposition(
        tz_times, 
        lats, 
        lons, 
        altitude=np.array(altitude_profile)
    )
    
    apparent_zenith = solpos['apparent_zenith'].values
    azimuth = solpos['azimuth'].values
    zenith_rad = np.radians(apparent_zenith)
    
    aoi = pvlib.irradiance.aoi(
        PANEL_TILT, 
        np.array(heading_profile), 
        apparent_zenith, 
        azimuth
    )
    
    tilt_rad = np.radians(PANEL_TILT)
    sky_factor = (1 + np.cos(tilt_rad)) / 2
    ground_factor = (1 - np.cos(tilt_rad)) / 2
    
    a_headings = np.cos(np.radians(aoi)) - (np.cos(zenith_rad) * sky_factor)
    
    b_constants = np.full(len(time_base), sky_factor + ALBEDO * ground_factor)
    
    return a_headings, b_constants

def mpc_cost_function(v_horizon, current_soc, current_v, target_profile, terrain_profile, distance_profile, a_headings, b_constants, solar_profile):
    cost = 0.0
    soc = current_soc
    v_prev = current_v
    
    for i in range(1, N + 1):
        v_next = v_horizon[i - 1]
        seg_len = distance_profile[i] - distance_profile[i - 1]
        
        # Pull precomputed values instantly using fast array lookups
        solar_irradiance = a_headings[i - 1] * solar_profile[i - 1][0] + b_constants[i - 1] * solar_profile[i - 1][1]
        
        p_net, dt = calculate_net_power(v_prev, v_next, terrain_profile[i - 1], solar_irradiance, seg_len)
        
        energy_change_wh = (p_net * dt) / 3600
        soc += (energy_change_wh / BATT_CAPACITY_WH) * 100.0
        
        # Penalties
        cost += 1.0 * (v_next - target_profile[i - 1]) ** 2
        if soc < 20.0:
            cost += 1000.0 * (20.0 - soc) ** 2
        cost += 0.5 * (v_next - v_prev) ** 2
        v_prev = v_next
        
    return cost

def compute_optimal_velocity(current_v, current_soc, current_time, targets, terrain, altitude, heading, coords, solar, distance):
    history_v = [current_v]
    history_soc = [current_soc]
    dt_array = [current_time]
    
    for i in range(1, N + 1):
        speed_bounds = [(0.1, 25.0) for _ in range(N)] # Lower bound 0.1 to avoid division by zero
        u_guess = np.ones(N) * current_v
        
        # --- PRECOMPUTE THE TIME WINDOW ---
        # Calculate a_headings and b_constants for the horizon profile chunk
        chunk_coords = coords[i - 1 : i + N - 1]
        chunk_heading = heading[i - 1 : i + N - 1]
        chunk_altitude = altitude[i - 1 : i + N - 1]
        
        # 1. Project an estimated timeline into the future based on your current speed
        estimated_time = dt_array[-1]
        estimated_timestamps = []

        for j in range(N):
            estimated_timestamps.append(estimated_time)
            seg_len = distance[i + j] - distance[i + j - 1]
            # Estimate how long it takes to cross this segment using current velocity
            dt_est = max((seg_len / max(targets[i+j-1], 0.1)) * 1000, 0.01)
            estimated_time += dt_est

        # 2. Re-calculate solar angles at the base time (perfectly fine since angles shift slowly)
        a_headings, b_constants = precompute_solar_gti_factors(estimated_timestamps, chunk_coords, chunk_heading, chunk_altitude)

        # 3. INTERPOLATE DNI/GHI matching each step's projected future timestamp!
        solar_profile = np.array([solar[chunk_coords[j], estimated_timestamps[j]].data(['dni','ghi']) for j in range(N)])
        
        result = minimize(
            mpc_cost_function, u_guess, 
            args=(history_soc[-1], history_v[-1], targets[i - 1 : i + N - 1], terrain[i - 1 : i + N - 1], distance[i - 1 : i + N], a_headings, b_constants, solar_profile),
            bounds=speed_bounds, method='SLSQP'
        )
        
        if result.success:
            history_v.append(float(result.x[0]))
        else:
            history_v.append(float(u_guess[0]))
            
        seg_len = distance[i] - distance[i - 1]
        
        # Re-apply using the optimal selected step
        solar_irradiance = a_headings[0] * solar_profile[0][0] + b_constants[0] * solar_profile[0][1]
        p_net_actual, dt = calculate_net_power(history_v[-2], history_v[-1], terrain[i - 1], solar_irradiance, seg_len)
        
        dt_array.append(dt_array[-1] + dt)
        history_soc.append(current_soc + ((p_net_actual * dt) / 3600.0 / BATT_CAPACITY_WH) * 100.0)
        
    history_v = np.array(history_v)
    return list(zip(dt_array, (history_v * (18 / 5)).tolist()))

def main(results=None, profiles=None):
    if not results:
        results = get_current_state()
    current_speed = results['Speed']
    current_soc = results['SoC']
    current_distance = results['Distance']
    current_time = results['Time_seconds']
    
    if not profiles:
        profiles = get_profile(["Gradient", "SpeedProfile", "SolarIrradiance", "TargetProfile", "Distance"])
        
    distance_profile = profiles.get("Distance")
    terrain_profile = profiles.get("Gradient", [0.0]*len(distance_profile)) or [0.0]*len(distance_profile)
    altitude_profile = profiles.get("Altitude", [0.0]*len(distance_profile)) or [0.0]*len(distance_profile)
    heading_profile = profiles.get("Headings", [0.0]*len(distance_profile)) or [0.0]*len(distance_profile)
    target_profile = profiles.get("TargetProfile", [current_speed]*len(distance_profile)) or [current_speed]*len(distance_profile)
    solar_profile = SolarIrradiance(profiles.get("SolarIrradiance", [500.0]*len(distance_profile)) or [500.0]*len(distance_profile))
    coords = profiles.get("Coordinates", [(0,0)]*len(distance_profile)) or [(0,0)]*len(distance_profile)
    
    if isinstance(target_profile[0], (tuple, list)):
        target_profile = [i for _, i in target_profile]
        
    terrain_profile = slice_profiles(terrain_profile, distance_profile, current_distance, 0)
    altitude_profile = slice_profiles(altitude_profile, distance_profile, current_distance, 0)
    heading_profile = slice_profiles(heading_profile, distance_profile, current_distance, 0)
    target_profile = slice_profiles(target_profile, distance_profile, current_distance, current_speed)
    target_profile = target_profile * (5 / 18)
    coords = slice_profiles(coords, distance_profile, current_distance, (0, 0))
    distance_profile = slice_profiles(distance_profile, distance_profile, current_distance, 0)
    
    current_speed *= 5 / 18
    return compute_optimal_velocity(current_speed, current_soc, current_time, target_profile, terrain_profile, altitude_profile, heading_profile, coords, solar_profile, distance_profile)