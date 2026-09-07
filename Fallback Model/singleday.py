import json
import time as time_module
from datetime import datetime, date, time
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import minimize
from solar_table import SolarIrradiance
from tqdm import tqdm

class OptimizerTimeout(Exception):
    pass

# ----------------- CONSTANTS ----------------- #
MASS_KG = 320.0
G_MS2 = 9.81
CRR = 0.007
CDA_M2 = 0.16
AIR_DENSITY = 1.2
ARRAY_AREA_M2 = 5.78
ARRAY_EFFICIENCY = 0.21
PANEL_TILT = 4.0
ALBEDO = 0.2

GAMMA_TEMP_COEFF = 0.004
T_NOCT = 45.0
K_CONVECTIVE_COOLING = 0.03
T_AMBIENT_DEFAULT = 28.0

MOTOR_EFF = 0.95
REGEN_EFF = 0.70
P_AUX = 5.0

BATTERY_WH = 588.0 * 6
SOC_MIN = 20.0
SOC_MAX = 95.0
V_MAX_MS = 85.0 / 3.6
V_MIN_MS = 30.0 / 3.6
MAX_POWER_LIMIT_W = 4000.0

CONTROL_STOP_S = 30 * 60
LOOP_STOP_S = 5 * 60
EOD_CUTOFF_HOUR = 17
SA_TZ = ZoneInfo("Africa/Johannesburg")

# ----------------- SOLAR & POWER FUNCTIONS ----------------- #
def precompute_solar_gti(time_base, coords, headings, altitudes):
    time_len = len(time_base)
    coords_arr = np.array(coords)
    lats = coords_arr[:, 0] if coords_arr.ndim == 2 else np.full(time_len, coords_arr[0])
    lons = coords_arr[:, 1] if coords_arr.ndim == 2 else np.full(time_len, coords_arr[1])
    alt_arr = np.array(altitudes) if len(np.atleast_1d(altitudes)) == time_len else np.full(time_len, altitudes)

    tz_times = pd.to_datetime(time_base, unit='s', utc=True).tz_convert(SA_TZ)
    solpos = pvlib.solarposition.get_solarposition(tz_times, lats, lons, altitude=alt_arr)
    apparent_zenith = solpos['apparent_zenith'].values
    azimuth = solpos['azimuth'].values

    headings_arr = np.array(headings)
    aoi = pvlib.irradiance.aoi(PANEL_TILT, headings_arr, apparent_zenith, azimuth)

    tilt_rad = np.radians(PANEL_TILT)
    zenith_rad = np.radians(apparent_zenith)
    sky_factor = (1.0 + np.cos(tilt_rad)) / 2.0
    ground_factor = (1.0 - np.cos(tilt_rad)) / 2.0

    a = np.cos(np.radians(aoi)) - (np.cos(zenith_rad) * sky_factor)
    b = np.full(time_len, sky_factor + ALBEDO * ground_factor)
    return a, b

def get_solar_power(time_base, coords, headings, altitudes, solar_obj, v_ms=0.0):
    a, b = precompute_solar_gti(time_base, coords, headings, altitudes)
    
    weather_data = solar_obj[(coords, time_base)].data(["dni", "ghi", "air_temp"])
    if len(weather_data) == 3:
        dni, ghi, air_temp = weather_data
    else:
        dni, ghi = weather_data
        air_temp = np.full(len(time_base), T_AMBIENT_DEFAULT)
        
    gti = np.maximum(0.0, a * dni + b * ghi)
    stc_power = gti * ARRAY_EFFICIENCY * ARRAY_AREA_M2
    
    irradiance_wm2 = stc_power / (ARRAY_AREA_M2 * ARRAY_EFFICIENCY + 1e-6)
    convective_cooling = np.exp(-K_CONVECTIVE_COOLING * v_ms)
    t_cell = air_temp + (irradiance_wm2 / 800.0) * (T_NOCT - 20.0) * convective_cooling
    temp_derating = np.clip(1.0 - GAMMA_TEMP_COEFF * (t_cell - 25.0), 0.70, 1.05)
    
    return stc_power * temp_derating

def calc_stationary_charge(start_ts, end_ts, coord, heading, alt, solar_obj):
    if start_ts >= end_ts:
        return 0.0
    dt = 60.0
    t_base = np.arange(start_ts, end_ts, dt)
    if len(t_base) == 0:
        return 0.0
    coords = np.tile(coord, (len(t_base), 1))
    headings = np.full(len(t_base), heading)
    p_solar = get_solar_power(t_base, coords, headings, alt, solar_obj, v_ms=0.0)
    net_p = np.maximum(0.0, p_solar - 10.0)
    energy_wh = np.sum(net_p * dt) / 3600.0
    return (energy_wh / BATTERY_WH) * 100.0

def precompute_evening_lut(start_time_ts, eod_cutoff_ts, coord, heading, alt, solar_obj):
    if start_time_ts >= eod_cutoff_ts:
        return np.array([start_time_ts, eod_cutoff_ts]), np.array([0.0, 0.0])
    
    dt = 30.0
    t_grid = np.arange(start_time_ts, eod_cutoff_ts + dt, dt)
    if len(t_grid) < 2:
        return np.array([start_time_ts, eod_cutoff_ts]), np.array([0.0, 0.0])

    coords = np.tile(coord, (len(t_grid), 1))
    headings = np.full(len(t_grid), heading)
    p_solar = get_solar_power(t_grid, coords, headings, alt, solar_obj, v_ms=0.0)
    net_p = np.maximum(0.0, p_solar - 10.0)

    e_steps = (net_p[:-1] + net_p[1:]) * 0.5 * dt / 3600.0
    rem_energy = np.zeros(len(t_grid))
    rem_energy[:-1] = np.cumsum(e_steps[::-1])[::-1]
    rem_soc_pct = (rem_energy / BATTERY_WH) * 100.0

    return t_grid, rem_soc_pct

# ----------------- ROUTE DISCRETIZATION ----------------- #
def resample_stage(profile_dict, dx=10.0):
    d_orig = np.array(profile_dict['Distance']) * 1000.0
    coords = np.array(profile_dict['Coordinates'])
    headings = np.array(profile_dict['Headings'])
    altitudes = np.array(profile_dict['Altitude'])
    gradients = np.array(profile_dict['Gradient'])
    speed_limits = np.array(profile_dict['SpeedLimit'])

    min_len = min(len(d_orig), len(coords), len(headings), len(altitudes), len(gradients), len(speed_limits))
    d_orig = d_orig[:min_len]
    coords = coords[:min_len]
    headings = headings[:min_len]
    altitudes = altitudes[:min_len]
    gradients = gradients[:min_len]
    speed_limits = speed_limits[:min_len]

    d_orig, unique_idx = np.unique(d_orig, return_index=True)
    coords = coords[unique_idx]
    headings = headings[unique_idx]
    altitudes = altitudes[unique_idx]
    gradients = gradients[unique_idx]
    speed_limits = speed_limits[unique_idx]

    total_dist = d_orig[-1]
    n_segments = int(np.floor(total_dist / dx))
    d_sim = np.arange(n_segments) * dx

    lats = np.interp(d_sim, d_orig, coords[:, 0])
    lons = np.interp(d_sim, d_orig, coords[:, 1])
    h = np.interp(d_sim, d_orig, headings)
    alt = np.interp(d_sim, d_orig, altitudes)
    grad = np.interp(d_sim, d_orig, gradients) / 100.0

    indices = np.searchsorted(d_orig, d_sim, side='right') - 1
    indices = np.clip(indices, 0, len(speed_limits) - 1)
    limits_sim = speed_limits[indices]
    limits_sim = np.maximum(limits_sim, V_MIN_MS * 3.6)

    return {
        'coords': np.column_stack((lats, lons)),
        'headings': h,
        'altitudes': alt,
        'gradients': grad,
        'speed_limits': limits_sim,
        'n_segments': n_segments
    }

def build_day_route(s1_profile, loop_profile, s2_profile, n_loops):
    s1 = resample_stage(s1_profile)
    loop = resample_stage(loop_profile) if (n_loops > 0 and loop_profile) else None
    s2 = resample_stage(s2_profile) if s2_profile else None

    coords = [s1['coords']]
    headings = [s1['headings']]
    altitudes = [s1['altitudes']]
    gradients = [s1['gradients']]
    speed_limits = [s1['speed_limits']]
    delays = np.zeros(s1['n_segments'])
    delays[-1] = CONTROL_STOP_S

    if n_loops > 0 and loop:
        for k in range(n_loops):
            coords.append(loop['coords'])
            headings.append(loop['headings'])
            altitudes.append(loop['altitudes'])
            gradients.append(loop['gradients'])
            speed_limits.append(loop['speed_limits'])
            l_delays = np.zeros(loop['n_segments'])
            l_delays[-1] = LOOP_STOP_S
            delays = np.concatenate((delays, l_delays))

    if s2:
        coords.append(s2['coords'])
        headings.append(s2['headings'])
        altitudes.append(s2['altitudes'])
        gradients.append(s2['gradients'])
        speed_limits.append(s2['speed_limits'])
        delays = np.concatenate((delays, np.zeros(s2['n_segments'])))
        
    grad_concat = np.concatenate(gradients)
    f_roll_grav = MASS_KG * G_MS2 * CRR * (1.0 - (grad_concat ** 2) / 2.0) + MASS_KG * G_MS2 * grad_concat

    return {
        'coords': np.vstack(coords),
        'headings': np.concatenate(headings),
        'altitudes': np.concatenate(altitudes),
        'gradients': grad_concat,
        'f_roll_grav': f_roll_grav,
        'speed_limits': np.concatenate(speed_limits),
        'delays': delays,
        'n_segments': len(delays)
    }

# ----------------- SIMULATION & OPTIMIZER ----------------- #
def simulate_day_fast(v_opt_arr, route, start_time_ts, soc_start, precomputed_p_solar, precomputed_stop_gains, evening_lut, eod_cutoff_ts):
    n_sim = route['n_segments']
    dx = 10.0
    RATIO = 100
    V_FLOOR_MS = 1.0    
    V_CEIL_MS = 60.0 
    
    v_sim = np.repeat(v_opt_arr, RATIO)[:n_sim]
    np.clip(v_sim, V_FLOOR_MS, V_CEIL_MS, out=v_sim)
    
    dt_drive = dx / v_sim
    finish_t = start_time_ts + np.sum(dt_drive) + np.sum(route['delays'])
    
    p_mech = v_sim * v_sim
    p_mech *= (0.5 * AIR_DENSITY * CDA_M2)
    p_mech += route['f_roll_grav']
    p_mech *= v_sim

    p_drivetrain = np.where(p_mech >= 0, p_mech / MOTOR_EFF, p_mech * REGEN_EFF)
    
    energy_coeff = 100.0 / (BATTERY_WH * 3600.0)
    d_soc_drive = precomputed_p_solar - p_drivetrain
    d_soc_drive -= P_AUX
    d_soc_drive *= dt_drive
    d_soc_drive *= energy_coeff

    stop_idx = np.flatnonzero(precomputed_stop_gains > 0)
    if len(stop_idx) > 0:
        d_soc_drive[stop_idx] += precomputed_stop_gains[stop_idx]
    
    unclipped_soc = soc_start + np.cumsum(d_soc_drive)
    overshoot = np.maximum(0.0, unclipped_soc - SOC_MAX)
    lost_energy = np.maximum.accumulate(overshoot)
    
    soc = unclipped_soc - lost_energy

    evening_gain = 0.0
    if evening_lut is not None:
        evening_gain = float(np.interp(finish_t, evening_lut[0], evening_lut[1]))
        
    final_soc = min(soc[-1] + evening_gain, SOC_MAX)
    unclipped_final_soc = unclipped_soc[-1] + evening_gain

    return final_soc, finish_t, soc, p_drivetrain, unclipped_soc, unclipped_final_soc

def simulate_day_fast_batch(V, route, start_time_ts, soc_start, precomputed_p_solar, precomputed_stop_gains, evening_lut, eod_cutoff_ts):
    V = np.atleast_2d(V)
    n_sim = route['n_segments']
    dx = 10.0
    RATIO = 100
    V_FLOOR_MS = 1.0
    V_CEIL_MS = 60.0

    v_sim = np.repeat(V, RATIO, axis=1)[:, :n_sim]
    np.clip(v_sim, V_FLOOR_MS, V_CEIL_MS, out=v_sim)

    dt_drive = dx / v_sim
    finish_t = start_time_ts + np.sum(dt_drive, axis=1) + np.sum(route['delays'])

    p_mech = v_sim * v_sim
    p_mech *= (0.5 * AIR_DENSITY * CDA_M2)
    p_mech += route['f_roll_grav'][None, :]
    p_mech *= v_sim

    p_drivetrain = np.where(p_mech >= 0, p_mech / MOTOR_EFF, p_mech * REGEN_EFF)
    
    energy_coeff = 100.0 / (BATTERY_WH * 3600.0)
    d_soc_drive = precomputed_p_solar[None, :] - p_drivetrain
    d_soc_drive -= P_AUX
    d_soc_drive *= dt_drive
    d_soc_drive *= energy_coeff

    stop_idx = np.flatnonzero(precomputed_stop_gains > 0)
    if len(stop_idx) > 0:
        d_soc_drive[:, stop_idx] += precomputed_stop_gains[stop_idx]
    
    unclipped_soc = soc_start + np.cumsum(d_soc_drive, axis=1)
    overshoot = np.maximum(0.0, unclipped_soc - SOC_MAX)
    lost_energy = np.maximum.accumulate(overshoot, axis=1)
    
    soc = unclipped_soc - lost_energy
    
    if evening_lut is not None:
        evening_gains = np.interp(finish_t, evening_lut[0], evening_lut[1])
        final_soc = np.minimum(soc[:, -1] + evening_gains, SOC_MAX)
        unclipped_final_soc = unclipped_soc[:, -1] + evening_gains
    else:
        final_soc = soc[:, -1].copy()
        unclipped_final_soc = unclipped_soc[:, -1].copy()

    return final_soc, finish_t, soc, p_drivetrain, unclipped_soc, unclipped_final_soc

def _fd_perturbation_batch(v):
    steps = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(v))
    V = np.tile(v, (len(v) + 1, 1))
    V[1:, :] += np.diag(steps)
    return V, steps

def optimize_single_day(route, start_time_ts, soc_start, target_eod_soc, v_guess_kmh, solar_obj, eod_cutoff_ts, w1=1.0, w2=100.0, w3=5000.0, max_time_s=None):    
    n_sim = route['n_segments']
    RATIO = 100
    n_opt = int(np.ceil(n_sim / RATIO))
    
    v_guess_ms = v_guess_kmh / 3.6
    V_ref_sq = v_guess_ms ** 2
    x0 = np.full(n_opt, v_guess_ms)

    v_sim_baseline = np.repeat(x0, RATIO)[:n_sim]
    dt_baseline = 10.0 / v_sim_baseline
    times_baseline = start_time_ts + np.cumsum(dt_baseline) + np.cumsum(route['delays'])
    baseline_p_solar = get_solar_power(times_baseline, route['coords'], route['headings'], route['altitudes'], solar_obj, v_ms=v_guess_ms)
    
    stop_gains = np.zeros(n_sim)
    delay_indices = np.flatnonzero(route['delays'] > 0)
    for i in delay_indices:
        stop_gains[i] = calc_stationary_charge(
            times_baseline[i], times_baseline[i] + route['delays'][i], 
            route['coords'][i], route['headings'][i], route['altitudes'][i], solar_obj
        )

    evening_lut = precompute_evening_lut(
        start_time_ts, eod_cutoff_ts,
        route['coords'][-1], route['headings'][-1], route['altitudes'][-1], solar_obj
    )

    v_limits_10m = route['speed_limits']
    v_limits_1km = np.array([np.min(v_limits_10m[i*RATIO:(i+1)*RATIO]) for i in range(n_opt)])

    v_upper_bounds = np.minimum(V_MAX_MS, v_limits_1km / 3.6)
    v_lower_bounds = np.minimum(V_MIN_MS, v_upper_bounds)
    bounds = [(lb, ub) for lb, ub in zip(v_lower_bounds, v_upper_bounds)]
    
    sim_cache = {
        'v': None, 'V': None, 'steps': None, 'final_soc': None, 
        'finish_t': None, 'soc_hist': None, 'p_drive': None, 'unclipped_soc': None, 'unclipped_final_soc': None
    }

    def get_cached_batch_sim(v):
        if sim_cache['v'] is not None and np.array_equal(v, sim_cache['v']):
            return sim_cache['V'], sim_cache['steps'], sim_cache['final_soc'], sim_cache['finish_t'], sim_cache['soc_hist'], sim_cache['p_drive'], sim_cache['unclipped_soc'], sim_cache['unclipped_final_soc']
        
        V, steps = _fd_perturbation_batch(v)
        final_soc, finish_t, soc_hist, p_drive, unclipped_soc, unclipped_final_soc = simulate_day_fast_batch(
            V, route, start_time_ts, soc_start, baseline_p_solar, stop_gains, evening_lut, eod_cutoff_ts
        )
        sim_cache.update({'v': np.copy(v), 'V': V, 'steps': steps, 'final_soc': final_soc, 'finish_t': finish_t, 'soc_hist': soc_hist, 'p_drive': p_drive, 'unclipped_soc': unclipped_soc, 'unclipped_final_soc': unclipped_final_soc})
        return V, steps, final_soc, finish_t, soc_hist, p_drive, unclipped_soc, unclipped_final_soc

    single_cache = {
        'v': None, 'final_soc': None, 'finish_t': None, 'soc_hist': None, 
        'p_drive': None, 'unclipped_soc': None, 'unclipped_final_soc': None
    }

    def get_cached_single_sim(v):
        if single_cache['v'] is not None and np.array_equal(v, single_cache['v']):
            return single_cache['final_soc'], single_cache['finish_t'], single_cache['soc_hist'], single_cache['p_drive'], single_cache['unclipped_soc'], single_cache['unclipped_final_soc']

        final_soc, finish_t, soc_hist, p_drive, unclipped_soc, unclipped_final_soc = simulate_day_fast(
            v, route, start_time_ts, soc_start, baseline_p_solar, stop_gains, evening_lut, eod_cutoff_ts
        )
        single_cache.update({'v': np.copy(v), 'final_soc': final_soc, 'finish_t': finish_t, 'soc_hist': soc_hist, 'p_drive': p_drive, 'unclipped_soc': unclipped_soc, 'unclipped_final_soc': unclipped_final_soc})
        return final_soc, finish_t, soc_hist, p_drive, unclipped_soc, unclipped_final_soc

    def objective(v):
        _, _, _, p_drive, _, unclipped_f_soc = get_cached_single_sim(v)
        j_pacer = w1 * np.mean(((v - v_guess_ms) ** 2) / V_ref_sq)
        j_soc = w2 * (((unclipped_f_soc - target_eod_soc) / 100.0) ** 2)
        power_excess = np.maximum(0.0, p_drive - 3800.0) / 200.0
        j_power = w3 * np.mean(power_excess ** 2)
        return 1000.0 * (j_pacer + j_soc + j_power)

    def objective_grad(v):
        V, steps, _, _, _, p_drive, _, unclipped_f_soc = get_cached_batch_sim(v)
        j_pacer = w1 * np.mean(((V - v_guess_ms) ** 2) / V_ref_sq, axis=1)
        j_soc = w2 * (((unclipped_f_soc - target_eod_soc) / 100.0) ** 2)
        power_excess = np.maximum(0.0, p_drive - 3800.0) / 200.0
        j_power = w3 * np.mean(power_excess ** 2, axis=1)
        objs = 1000.0 * (j_pacer + j_soc + j_power)
        return (objs[1:] - objs[0]) / steps
    
    SOFT_K = 1.5
    def _softmin(rows, k=SOFT_K):
        rows = np.atleast_2d(rows)
        row_mins = np.min(rows, axis=1, keepdims=True)
        return row_mins.flatten() - np.log(np.sum(np.exp(-k * (rows - row_mins)), axis=1)) / k

    def _softmax(rows, k=SOFT_K):
        rows = np.atleast_2d(rows)
        row_maxs = np.max(rows, axis=1, keepdims=True)
        return row_maxs.flatten() + np.log(np.sum(np.exp(k * (rows - row_maxs)), axis=1)) / k

    def ineq_soc_min(v):
        _, _, soc_hist, _, _, _ = get_cached_single_sim(v)
        return _softmin(soc_hist)[0] - SOC_MIN

    def jac_ineq_soc_min(v):
        _, steps, _, _, soc_hist, _, _, _ = get_cached_batch_sim(v)
        min_soc = _softmin(soc_hist)
        return (min_soc[1:] - min_soc[0]) / steps

    def ineq_soc_max(v):
        _, _, _, _, unclipped_soc, _ = get_cached_single_sim(v)
        return SOC_MAX - _softmax(unclipped_soc)[0]

    def jac_ineq_soc_max(v):
        _, steps, _, _, _, _, unclipped_soc, _ = get_cached_batch_sim(v)
        max_soc = _softmax(unclipped_soc)
        return -(max_soc[1:] - max_soc[0]) / steps

    def ineq_time_max(v):
        _, finish_t, _, _, _, _ = get_cached_single_sim(v)
        return eod_cutoff_ts - finish_t

    def jac_ineq_time_max(v):
        _, steps, _, finish_t, _, _, _, _ = get_cached_batch_sim(v)
        return -(finish_t[1:] - finish_t[0]) / steps

    constraints = [
        {'type': 'ineq', 'fun': ineq_soc_min, 'jac': jac_ineq_soc_min},      
        {'type': 'ineq', 'fun': ineq_soc_max, 'jac': jac_ineq_soc_max},        
        {'type': 'ineq', 'fun': ineq_time_max, 'jac': jac_ineq_time_max},    
    ]
    
    MAXITER = 100
    pbar = tqdm(total=MAXITER, desc="SLSQP Iterations", unit="iter")
    best_xk = {'x': x0.copy(), 'iter': 0}
    start_wall = time_module.time()

    def progress_tracker(xk):
        pbar.update(1)
        best_xk['x'] = xk.copy()
        best_xk['iter'] += 1
        elapsed = time_module.time() - start_wall
        current_mean_kmh = np.mean(xk) * 3.6
        pbar.set_postfix({'Mean Speed': f"{current_mean_kmh:.1f} km/h", 'Elapsed': f"{elapsed:.0f}s"})
        if max_time_s is not None and elapsed > max_time_s:
            raise OptimizerTimeout(f"time budget exceeded")

    try:
        res = minimize(
            objective, x0, jac=objective_grad, method='SLSQP', bounds=bounds, 
            constraints=constraints, options={'ftol': 1e-4 , 'maxiter': MAXITER, 'disp': False}, callback=progress_tracker
        )
    except (OptimizerTimeout, KeyboardInterrupt) as e:
        class _PartialResult: pass
        res = _PartialResult()
        res.x = np.clip(best_xk['x'], v_lower_bounds, v_upper_bounds)
        res.success = False 
        res.status = -1 
        res.message = "Interrupted by user" if isinstance(e, KeyboardInterrupt) else str(e)
        res.nit = best_xk['iter']
        res.fun = objective(res.x)
        print("\n[!] Interruption caught. Saving partial best profile...")
    
    pbar.close()
    res.x = np.clip(res.x, v_lower_bounds, v_upper_bounds)
    res.fun = objective(res.x)

    final_soc_val, finish_t_val, soc, p_drive_final, _, unclipped_f_soc = simulate_day_fast(res.x, route, start_time_ts, soc_start, baseline_p_solar, stop_gains, evening_lut, eod_cutoff_ts)
    print(
        f"[Solver Diagnostic] status={res.status} success={res.success} msg='{res.message}' iters={res.nit}\n"
        f"  Target 17:00 SoC: {target_eod_soc:.2f}% | Math SoC (Unclipped): {unclipped_f_soc:.2f}%\n"
        f"  Peak SoC Hit: {np.max(soc):.2f}% | Min SoC Hit: {np.min(soc):.2f}%\n"
        f"  Peak Motor Power: {np.max(p_drive_final):.1f} W | Finish Slack: {(eod_cutoff_ts - finish_t_val) / 3600.0:.2f} hrs"
    )

    return res, baseline_p_solar, stop_gains, v_limits_1km, evening_lut

def resolve(current_km, current_time_ts, current_soc, s1_profile, loop_profile, s2_profile, manual_target_loops, target_eod_soc, v_guess_kmh, solar_obj, race_date, day_no, w1=1.0, w2=100.0, w3=10.0, max_time_s=None):
    eod_cutoff_ts = datetime.combine(race_date, time(EOD_CUTOFF_HOUR, 0), tzinfo=SA_TZ).timestamp()
    full_route = build_day_route(s1_profile, loop_profile, s2_profile, manual_target_loops)

    k_curr = int(np.floor(current_km * 100))
    k_curr = min(k_curr, full_route['n_segments'] - 1)

    rem_route = {
        'coords': full_route['coords'][k_curr:],
        'headings': full_route['headings'][k_curr:],
        'altitudes': full_route['altitudes'][k_curr:],
        'gradients': full_route['gradients'][k_curr:],
        'f_roll_grav': full_route['f_roll_grav'][k_curr:],
        'speed_limits': full_route['speed_limits'][k_curr:],
        'delays': full_route['delays'][k_curr:],
        'n_segments': full_route['n_segments'] - k_curr
    }

    print(f"Re-solving from km {current_km:.1f} ({rem_route['n_segments'] / 100:.1f} km left) | Target Loops: {manual_target_loops}")

    res, baseline_p_solar, stop_gains, v_limits_1km, evening_lut = optimize_single_day(
        rem_route, current_time_ts, current_soc, target_eod_soc, v_guess_kmh, solar_obj, eod_cutoff_ts, w1=w1, w2=w2, w3=w3, max_time_s=max_time_s
    )

    if res.success or res.status == 9 or res.status == 3 or res.status == -1 or res.status == 8:
        opt_v_ms = res.x
        opt_v_kmh = opt_v_ms * 3.6
        quantized_kmh = np.minimum(np.round(opt_v_kmh), v_limits_1km)
        quantized_ms = quantized_kmh / 3.6
        
        final_soc, finish_t, soc_history, p_drivetrain, _, _ = simulate_day_fast(
            quantized_ms, rem_route, current_time_ts, current_soc, baseline_p_solar, stop_gains, evening_lut, eod_cutoff_ts
        )

        quantized_kmh_sim = np.repeat(quantized_kmh, 100)[:rem_route['n_segments']]
        quantized_ms_sim = np.repeat(quantized_ms, 100)[:rem_route['n_segments']]
        limits_kmh_sim = np.repeat(v_limits_1km, 100)[:rem_route['n_segments']]

        dt_drive = 10.0 / quantized_ms_sim
        times = current_time_ts + np.cumsum(dt_drive) + np.cumsum(rem_route['delays'])

        save_dir = Path("Fallback Model/velocity_profiles")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if current_km > 0:
            file_name = f"optimized_day_{day_no}_resolve.npz"
        else:
            file_name = f"optimized_day_{day_no}.npz"
            
        save_path = save_dir / file_name
        
        np.savez(
            save_path, speeds_kmh=quantized_kmh_sim, speed_limits_kmh=limits_kmh_sim,
            soc=soc_history, power_w=p_drivetrain, start_km=current_km,
            times=times, eod_cutoff_ts=eod_cutoff_ts, final_soc=final_soc           
        )

        finish_str = datetime.fromtimestamp(finish_t, tz=SA_TZ).strftime("%H:%M:%S")
        print(f"Re-solve Succeeded! Finish Time: {finish_str} | Final physical 17:00 SoC: {final_soc:.2f}% | Peak SoC: {np.max(soc_history):.2f}%")
        return quantized_kmh
    else:
        print(f"Re-solve Failed: {res.message}")
        return np.full(int(np.ceil(rem_route['n_segments'] / 100)), v_guess_kmh)

if __name__ == "__main__":
    DAY_NO = 2
    day_str = f"Day {DAY_NO}"

    DAYWISE_FILES = {
        "Day 1": {"date":date(2026,9,10),"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 1 Boiketlong to Rustenburg","l":"2026 Sasol Solar Challenge Route (Publish)_Day 1 _Rustenburg Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 2 Rustenburg to Swartruggens"},
        "Day 2": {"date":date(2026,9,11),"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 2 Half Blind_11 Sept Stage 1 Swart Ruggens to Zeerust","l":"SSC ROUTE FINAL_Day 2 Half Blind_Day 2 Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 2 Half Blind_11 Sept Stage 2 Zeerust to Vryburg"},
        "Day 3": {"date":date(2026,9,12),"s1":"Day 3 probables_Probable Prahlad Route_Stage 1", "l":"Day 3 probables_Probable Prahlad Route_Day 3 Loop","s2":"Day 3 probables_Probable Prahlad Route_Stage 2" },
        "Day 4": {"date":date(2026,9,13),"s1":"2026 Sasol Solar Challenge Route (Publish)_Day 4_13 Sept Stage 1 Kimberley to Postmasburg","l":"2026 Sasol Solar Challenge Route (Publish)_Day 4_Postmasburg Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 4_13 Sept Stage 2 Postmasburg to Olifantshoek"},
        "Day 5": {"date":date(2026,9,14),"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 1 Olifantshoek to Upington","l":"2026 Sasol Solar Challenge Route (Publish)_Day 5 _Upington Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 2 Upington to Augrabies"},
        "Day 6": {"date":date(2026,9,15),"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 6 _15 Sept Stage 1 Augrabies to Springbok","l":"2026 Sasol Solar Challenge Route (Publish)_Day 6 _Springbok Loop","s2":None},
        "Day 7": {"date":date(2026,9,16),"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 7_16 Sept Stage 1 Springbok to Van Rhynsdorp","l":"2026 Sasol Solar Challenge Route (Publish)_Day 7_Van Rhynsdorp Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 7_16 Sept Stage 2 Van Rhynsdorp to Clanwilliam"},
        "Day 8": {"date":date(2026,9,17),"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 8_17 Sept Stage 1 Clanwilliam to Ceres","l":"2026 Sasol Solar Challenge Route (Publish)_Day 8_Ceres Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 8_17 Sept Stage 2 Ceres to Paarl"}
    }

    START_SOCS = [95.0, 75.03, 64.57, 73.37, 63.41, 54.16, 60.48, 61.80]
    V_GUESSES = [60.0, 64.0, 54.0, 52.0, 59.0, 53.0, 54.0, 54.0]
    LOOPS = [7, 9 , 5, 9 ,3 ,5 , 5 , 4]

    race_date = DAYWISE_FILES[day_str]["date"]
    s1_name = DAYWISE_FILES[day_str]["s1"]
    l_name = DAYWISE_FILES[day_str]["l"]
    s2_name = DAYWISE_FILES[day_str]["s2"]

    def load_profile(name):
        if not name: return None
        with open(f"Fallback Model/Saves/{name}.kml.save", 'r') as f:
            return json.load(f)['profile']

    s1_profile = load_profile(s1_name)
    loop_profile = load_profile(l_name)
    s2_profile = load_profile(s2_name)

    weather_filename = l_name if l_name else s1_name
    with open(f"Fallback Model/Solar_real/mean_{weather_filename}.jsonl", 'r') as f:
        weather_data = json.load(f)
    solar_obj = SolarIrradiance(weather_data, "period_end", "PT5M", 6)

    start_hour = 9 if DAY_NO == 1 else 8
    morning_ts = datetime.combine(race_date, time(6, 0), tzinfo=SA_TZ).timestamp()
    start_time_ts = datetime.combine(race_date, time(start_hour, 0), tzinfo=SA_TZ).timestamp()
    
    current_soc = START_SOCS[DAY_NO - 1]
    
    if DAY_NO != 1:
        morning_gain = calc_stationary_charge(morning_ts, start_time_ts, s1_profile['Coordinates'][0], s1_profile['Headings'][0], s1_profile['Altitude'][0], solar_obj)
        current_soc = min(SOC_MAX, current_soc + morning_gain)
        print(f"Morning charge complete. Starting day {DAY_NO} with SoC: {current_soc:.2f}%")

    if DAY_NO < len(START_SOCS):
        target_1700_soc = START_SOCS[DAY_NO]
    else:
        target_1700_soc = 20.0 

    print(f"Dynamic Target Locked: Aiming for exactly {target_1700_soc:.2f}% at 17:00 SAST.")

    optimized_speeds = resolve(
        current_km=0.0,
        current_time_ts=start_time_ts,
        current_soc=current_soc,
        s1_profile=s1_profile,
        loop_profile=loop_profile,
        s2_profile=s2_profile,
        manual_target_loops=LOOPS[DAY_NO -1],
        target_eod_soc=target_1700_soc,
        v_guess_kmh=V_GUESSES[DAY_NO - 1],
        solar_obj=solar_obj,
        race_date=race_date,
        day_no=DAY_NO,
        w1=1.0,        
        w2=100.0,
        w3=5000.0 
    )