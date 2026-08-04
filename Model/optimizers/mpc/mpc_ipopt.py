from helper import get_current_state, get_profile
import numpy as np
import casadi as ca
from solar_table import SolarIrradiance
import pvlib
import pandas as pd
from constants import MASS, CDA, CRR, RHO, G, SOLAR_AREA, SOLAR_EFF, MOTOR_EFF, REGEN_EFF, POWER_LOSS, PANEL_TILT, ALBEDO, BATTERY_CAPACITY_WH, BATTERY_CAPACITY_AH, N

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


def calculate_net_power(v_current, v_next, slope, solar_irradiance, seg_len, is_casadi=False):
    grad = slope / 100
    f_drag = 0.5 * RHO * CDA * (v_current ** 2)
    f_rolling = MASS * G * CRR * (1 - (grad**2)/2)
    f_gravity = MASS * G * grad
    
    if is_casadi:
        v_safe = ca.fmax(v_current, 0.1)
        dt = ca.fmax((seg_len / v_safe) * 1000, 0.01)
    else:
        v_safe = max(v_current, 0.1)
        dt = max((seg_len / v_safe) * 1000, 0.01)
    
    f_acceleration = MASS * (v_next - v_current) / dt
    f_total = f_drag + f_rolling + f_gravity + f_acceleration
    p_solar = SOLAR_AREA * SOLAR_EFF * solar_irradiance
    p_mech = f_total * v_current
    
    if is_casadi:
        p_electric = ca.if_else(p_mech >= 0, p_mech / MOTOR_EFF, p_mech * REGEN_EFF)
    else:
        p_electric = p_mech / MOTOR_EFF if p_mech >= 0 else p_mech * REGEN_EFF
        
    net_power_watts = p_solar - p_electric - POWER_LOSS
    return net_power_watts, dt


def precompute_solar_gti_factors(time_base, coords_list, heading_profile, altitude_profile):
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


# ==========================================
# CASADI SOLVER
# ==========================================
v_opt = ca.MX.sym('v_opt', N)
p_v_init = ca.MX.sym('p_v_init')
p_soc_init = ca.MX.sym('p_soc_init')
p_targets = ca.MX.sym('p_targets', N)
p_terrain = ca.MX.sym('p_terrain', N)
p_seg_lens = ca.MX.sym('p_seg_lens', N)
p_a_headings = ca.MX.sym('p_a_headings', N)
p_b_constants = ca.MX.sym('p_b_constants', N)
p_solar_dni = ca.MX.sym('p_solar_dni', N)
p_solar_ghi = ca.MX.sym('p_solar_ghi', N)

_cost = 0.0
_soc = p_soc_init
_v_prev = p_v_init

for j in range(N):
    _v_next = v_opt[j]
    _seg_len = p_seg_lens[j]
    _solar_irradiance = p_a_headings[j] * p_solar_dni[j] + p_b_constants[j] * p_solar_ghi[j]
    
    _p_net, _dt = calculate_net_power(_v_prev, _v_next, p_terrain[j], _solar_irradiance, _seg_len, is_casadi=True)
    
    _energy_change_wh = (_p_net * _dt) / 3600
    _soc += (_energy_change_wh / BATTERY_CAPACITY_WH) * 100.0
    
    _cost += 1.0 * (_v_next - p_targets[j]) ** 2
    _cost += ca.if_else(_soc < 20.0, 1000.0 * (20.0 - _soc) ** 2, 0.0)
    _cost += 0.5 * (_v_next - _v_prev) ** 2
    _v_prev = _v_next

_p = ca.vertcat(p_v_init, p_soc_init, p_targets, p_terrain, p_seg_lens, p_a_headings, p_b_constants, p_solar_dni, p_solar_ghi)
_nlp = {'x': v_opt, 'f': _cost, 'p': _p}
_opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
GLOBAL_SOLVER = ca.nlpsol('solver', 'ipopt', _nlp, _opts)


# ==========================================
# EXECUTION
# ==========================================
def compute_optimal_velocity(current_v, current_soc, current_time, targets, terrain, altitude, heading, coords, solar, distance):
    history_v = [current_v]
    history_soc = [current_soc]
    dt_array = [current_time]
    
    lbx = [0.1] * N
    ubx = [25.0] * N
    
    for i in range(1, N + 1):
        u_guess = np.ones(N) * history_v[-1]
        
        chunk_coords = coords[i - 1 : i + N - 1]
        chunk_heading = heading[i - 1 : i + N - 1]
        chunk_altitude = altitude[i - 1 : i + N - 1]
        
        estimated_time = dt_array[-1]
        estimated_timestamps = []
        seg_lens_num = []

        for j in range(N):
            estimated_timestamps.append(estimated_time)
            seg_len = distance[i + j] - distance[i + j - 1]
            seg_lens_num.append(seg_len)
            dt_est = max((seg_len / max(targets[i+j-1], 0.1)) * 1000, 0.01)
            estimated_time += dt_est

        a_headings, b_constants = precompute_solar_gti_factors(estimated_timestamps, chunk_coords, chunk_heading, chunk_altitude)
        solar_profile = np.array([solar[chunk_coords[j], estimated_timestamps[j]].data(['dni','ghi']) for j in range(N)])
        
        solar_dni_num = solar_profile[:, 0]
        solar_ghi_num = solar_profile[:, 1]
        
        p_num = ca.vertcat(
            history_v[-1], 
            history_soc[-1], 
            targets[i - 1 : i + N - 1], 
            terrain[i - 1 : i + N - 1], 
            seg_lens_num, 
            a_headings, 
            b_constants, 
            solar_dni_num, 
            solar_ghi_num
        )
        
        try:
            sol = GLOBAL_SOLVER(x0=u_guess.tolist(), lbx=lbx, ubx=ubx, p=p_num)
            v_sol = np.array(sol['x']).flatten()
            history_v.append(float(v_sol[0]))
        except Exception:
            history_v.append(float(u_guess[0]))
            
        seg_len_actual = distance[i] - distance[i - 1]
        solar_irradiance_actual = a_headings[0] * solar_dni_num[0] + b_constants[0] * solar_ghi_num[0]
        
        p_net_actual, dt_actual = calculate_net_power(history_v[-2], history_v[-1], terrain[i - 1], solar_irradiance_actual, seg_len_actual, is_casadi=False)
        
        dt_array.append(dt_array[-1] + dt_actual)
        history_soc.append(history_soc[-1] + ((p_net_actual * dt_actual) / 3600.0 / BATTERY_CAPACITY_WH) * 100.0)
        
    history_v = np.array(history_v)
    return list(zip(dt_array, (history_v * (18 / 5)).tolist()))


def main(results=None, profiles=None):
    if not results:
        results = get_current_state()
    current_speed = results['Speed']
    current_soc = (results['SoC'] / BATTERY_CAPACITY_AH) * 100.0 
    current_distance = results['Distance']
    current_time = results['Time_seconds']
    
    if not profiles:
        profiles = get_profile(["Gradient", "SpeedProfile", "SolarIrradiance", "TargetProfile", "Distance", "Altitude", "Headings", "Coordinates"])
        
    distance_profile = profiles.get("Distance")
    terrain_profile = profiles.get("Gradient", [0.0]*len(distance_profile)) or [0.0]*len(distance_profile)
    altitude_profile = profiles.get("Altitude", [0.0]*len(distance_profile)) or [0.0]*len(distance_profile)
    heading_profile = profiles.get("Headings", [0.0]*len(distance_profile)) or [0.0]*len(distance_profile)
    target_profile = profiles.get("TargetProfile", [current_speed]*len(distance_profile)) or [current_speed]*len(distance_profile)
    solar_profile = SolarIrradiance(profiles.get("SolarIrradiance", [500.0]*len(distance_profile)) or [500.0]*len(distance_profile))
    coords = profiles.get("Coordinates", [(0,0)]*len(distance_profile)) or [(0,0)]*len(distance_profile)
    
    if len(target_profile) > 0 and isinstance(target_profile[0], (tuple, list)):
        target_profile = [i for _, i in target_profile]
    elif len(target_profile) == 0:
        target_profile = [current_speed] * len(distance_profile)
        
    terrain_profile = slice_profiles(terrain_profile, distance_profile, current_distance, 0)
    altitude_profile = slice_profiles(altitude_profile, distance_profile, current_distance, 0)
    heading_profile = slice_profiles(heading_profile, distance_profile, current_distance, 0)
    target_profile = slice_profiles(target_profile, distance_profile, current_distance, current_speed)
    target_profile = target_profile * (5 / 18)
    coords = slice_profiles(coords, distance_profile, current_distance, (0, 0))
    distance_profile = slice_profiles(distance_profile, distance_profile, current_distance, 0)
    
    current_speed *= 5 / 18
    return compute_optimal_velocity(current_speed, current_soc, current_time, target_profile, terrain_profile, altitude_profile, heading_profile, coords, solar_profile, distance_profile)