# singleday.py
import json
from datetime import datetime, date, time
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import minimize
from solar_table import SolarIrradiance
from tqdm import tqdm

# ----------------- CONSTANTS ----------------- #
MASS_KG = 300.0
G_MS2 = 9.81
CRR = 0.007
CDA_M2 = 0.16
AIR_DENSITY = 1.2
ARRAY_AREA_M2 = 5.95
ARRAY_EFFICIENCY = 0.18
PANEL_TILT = 4.0
ALBEDO = 0.2

MOTOR_EFF = 0.95
REGEN_EFF = 0.70
P_AUX = 50.0

BATTERY_WH = 588.0 * 6
SOC_MIN = 20.0
SOC_MAX = 95.0
V_MAX_MS = 85.0 / 3.6
V_MIN_MS = 30.0 / 3.6

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

def get_solar_power(time_base, coords, headings, altitudes, solar_obj):
    a, b = precompute_solar_gti(time_base, coords, headings, altitudes)
    dni, ghi = solar_obj[(coords, time_base)].data(["dni", "ghi"])
    gti = np.maximum(0.0, a * dni + b * ghi)
    return gti * ARRAY_EFFICIENCY * ARRAY_AREA_M2

def calc_stationary_charge(start_ts, end_ts, coord, heading, alt, solar_obj):
    if start_ts >= end_ts:
        return 0.0
    dt = 60.0
    t_base = np.arange(start_ts, end_ts, dt)
    if len(t_base) == 0:
        return 0.0
    coords = np.tile(coord, (len(t_base), 1))
    headings = np.full(len(t_base), heading)
    p_solar = get_solar_power(t_base, coords, headings, alt, solar_obj)
    net_p = np.maximum(0.0, p_solar - 10.0)
    energy_wh = np.sum(net_p * dt) / 3600.0
    return (energy_wh / BATTERY_WH) * 100.0

# ----------------- ROUTE DISCRETIZATION ----------------- #
def resample_stage(profile_dict, dx=10.0):
    d_orig = np.array(profile_dict['Distance']) * 1000.0
    coords = np.array(profile_dict['Coordinates'])
    headings = np.array(profile_dict['Headings'])
    altitudes = np.array(profile_dict['Altitude'])
    gradients = np.array(profile_dict['Gradient'])

    min_len = min(len(d_orig), len(coords), len(headings), len(altitudes), len(gradients))
    d_orig = d_orig[:min_len]
    coords = coords[:min_len]
    headings = headings[:min_len]
    altitudes = altitudes[:min_len]
    gradients = gradients[:min_len]

    d_orig, unique_idx = np.unique(d_orig, return_index=True)
    coords = coords[unique_idx]
    headings = headings[unique_idx]
    altitudes = altitudes[unique_idx]
    gradients = gradients[unique_idx]

    total_dist = d_orig[-1]
    n_segments = int(np.floor(total_dist / dx))
    d_sim = np.arange(n_segments) * dx

    lats = np.interp(d_sim, d_orig, coords[:, 0])
    lons = np.interp(d_sim, d_orig, coords[:, 1])
    h = np.interp(d_sim, d_orig, headings)
    alt = np.interp(d_sim, d_orig, altitudes)
    grad = np.interp(d_sim, d_orig, gradients) / 100.0

    return {
        'coords': np.column_stack((lats, lons)),
        'headings': h,
        'altitudes': alt,
        'gradients': grad,
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
    delays = np.zeros(s1['n_segments'])
    delays[-1] = CONTROL_STOP_S

    if n_loops > 0 and loop:
        for k in range(n_loops):
            coords.append(loop['coords'])
            headings.append(loop['headings'])
            altitudes.append(loop['altitudes'])
            gradients.append(loop['gradients'])
            l_delays = np.zeros(loop['n_segments'])
            l_delays[-1] = LOOP_STOP_S
            delays = np.concatenate((delays, l_delays))

    if s2:
        coords.append(s2['coords'])
        headings.append(s2['headings'])
        altitudes.append(s2['altitudes'])
        gradients.append(s2['gradients'])
        delays = np.concatenate((delays, np.zeros(s2['n_segments'])))

    return {
        'coords': np.vstack(coords),
        'headings': np.concatenate(headings),
        'altitudes': np.concatenate(altitudes),
        'gradients': np.concatenate(gradients),
        'delays': delays,
        'n_segments': len(delays)
    }

# ----------------- SIMULATION & OPTIMIZER ----------------- #
def simulate_day_fast(v_opt_arr, route, start_time_ts, soc_start, precomputed_p_solar, precomputed_stop_gains, solar_obj, eod_cutoff_ts):
    n_sim = route['n_segments']
    dx = 10.0
    RATIO = 100
    
    v_sim = np.repeat(v_opt_arr, RATIO)[:n_sim]
    
    dt_drive = dx / v_sim
    times = start_time_ts + np.cumsum(dt_drive) + np.cumsum(route['delays'])
    
    grad = route['gradients']
    f_drag = 0.5 * AIR_DENSITY * CDA_M2 * (v_sim ** 2)
    f_roll = MASS_KG * G_MS2 * CRR * (1.0 - (grad ** 2) / 2.0)
    f_grav = MASS_KG * G_MS2 * grad
    p_mech = (f_drag + f_roll + f_grav) * v_sim

    p_drivetrain = np.where(p_mech >= 0, p_mech / MOTOR_EFF, p_mech * REGEN_EFF)
    p_elec = precomputed_p_solar - p_drivetrain - P_AUX
    d_soc_drive = (p_elec * dt_drive) / (BATTERY_WH * 3600.0) * 100.0

    soc = np.empty(n_sim)
    stop_idx = np.flatnonzero(precomputed_stop_gains > 0)
    base = soc_start
    start_idx = 0
    for j in stop_idx:
        seg_csum = np.cumsum(d_soc_drive[start_idx:j + 1])
        soc[start_idx:j + 1] = base + seg_csum
        soc[j] = min(SOC_MAX, soc[j] + precomputed_stop_gains[j])
        base = soc[j]
        start_idx = j + 1
    if start_idx < n_sim:
        seg_csum = np.cumsum(d_soc_drive[start_idx:n_sim])
        soc[start_idx:n_sim] = base + seg_csum

    final_soc = soc[-1]
    finish_t = times[-1]
    
    if finish_t < eod_cutoff_ts:
        evening_gain = calc_stationary_charge(finish_t, eod_cutoff_ts, route['coords'][-1], route['headings'][-1], route['altitudes'][-1], solar_obj)
        final_soc = min(SOC_MAX, final_soc + evening_gain)

    return final_soc, finish_t, soc, p_mech

def simulate_day_fast_batch(V, route, start_time_ts, soc_start, precomputed_p_solar, precomputed_stop_gains, solar_obj, eod_cutoff_ts):

    V = np.atleast_2d(V)
    B, n_opt = V.shape
    n_sim = route['n_segments']
    dx = 10.0
    RATIO = 100

    v_sim = np.repeat(V, RATIO, axis=1)[:, :n_sim]

    dt_drive = dx / v_sim
    times = start_time_ts + np.cumsum(dt_drive, axis=1) + np.cumsum(route['delays'])[None, :]

    grad = route['gradients'][None, :]
    f_drag = 0.5 * AIR_DENSITY * CDA_M2 * (v_sim ** 2)
    f_roll = MASS_KG * G_MS2 * CRR * (1.0 - (grad ** 2) / 2.0)
    f_grav = MASS_KG * G_MS2 * grad
    p_mech = (f_drag + f_roll + f_grav) * v_sim

    p_drivetrain = np.where(p_mech >= 0, p_mech / MOTOR_EFF, p_mech * REGEN_EFF)
    p_elec = precomputed_p_solar[None, :] - p_drivetrain - P_AUX
    d_soc_drive = (p_elec * dt_drive) / (BATTERY_WH * 3600.0) * 100.0

    soc = np.empty((B, n_sim))
    stop_idx = np.flatnonzero(precomputed_stop_gains > 0)
    base = np.full(B, soc_start, dtype=float)
    start_idx = 0
    for j in stop_idx:
        seg_csum = np.cumsum(d_soc_drive[:, start_idx:j + 1], axis=1)
        soc[:, start_idx:j + 1] = base[:, None] + seg_csum
        soc[:, j] = np.minimum(SOC_MAX, soc[:, j] + precomputed_stop_gains[j])
        base = soc[:, j]
        start_idx = j + 1
    if start_idx < n_sim:
        seg_csum = np.cumsum(d_soc_drive[:, start_idx:n_sim], axis=1)
        soc[:, start_idx:n_sim] = base[:, None] + seg_csum

    final_soc = soc[:, -1].copy()
    finish_t = times[:, -1]

    for b in range(B):
        if finish_t[b] < eod_cutoff_ts:
            evening_gain = calc_stationary_charge(
                finish_t[b], eod_cutoff_ts,
                route['coords'][-1], route['headings'][-1], route['altitudes'][-1],
                solar_obj
            )
            final_soc[b] = min(SOC_MAX, final_soc[b] + evening_gain)

    return final_soc, finish_t, soc

def _fd_perturbation_batch(v):

    steps = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(v))
    V = np.tile(v, (len(v) + 1, 1))
    V[1:, :] += np.diag(steps)
    return V, steps

def optimize_single_day(route, start_time_ts, soc_start, target_eod_soc, v_guess_kmh, solar_obj, eod_cutoff_ts, speed_limits=None, w1=1.0, w2=0.1, w3=2.0):
    n_sim = route['n_segments']
    RATIO = 100
    n_opt = int(np.ceil(n_sim / RATIO))
    
    v_guess_ms = v_guess_kmh / 3.6
    x0 = np.full(n_opt, v_guess_ms)

    # PRECOMPUTE SOLAR TO UNBLOCK SLSQP
    v_sim_baseline = np.repeat(x0, RATIO)[:n_sim]
    dt_baseline = 10.0 / v_sim_baseline
    times_baseline = start_time_ts + np.cumsum(dt_baseline) + np.cumsum(route['delays'])
    baseline_p_solar = get_solar_power(times_baseline, route['coords'], route['headings'], route['altitudes'], solar_obj)
    
    stop_gains = np.zeros(n_sim)
    for i in range(n_sim):
        if route['delays'][i] > 0:
            stop_gains[i] = calc_stationary_charge(times_baseline[i], times_baseline[i] + route['delays'][i], route['coords'][i], route['headings'][i], route['altitudes'][i], solar_obj)

    v_upper_bounds = np.full(n_opt, V_MAX_MS)
    if speed_limits is not None:
        v_upper_bounds = np.minimum(v_upper_bounds, np.array(speed_limits) / 3.6)
    bounds = [(V_MIN_MS, v_upper_bounds[i]) for i in range(n_opt)]

    v_low = 50.0 / 3.6
    v_high = 75.0 / 3.6

    T_ref = n_opt * (1000.0 / v_guess_ms) 
    V_ref_sq = v_guess_ms ** 2 

    def objective(v):
        j_time = w1 * (np.sum(1000.0 / v) / T_ref)
        j_smooth = w2 * (np.sum((v[1:] - v[:-1]) ** 2) / V_ref_sq)
        j_band = w3 * ((np.sum(np.maximum(0.0, v - v_high) ** 2) + np.sum(np.maximum(0.0, v_low - v) ** 2)) / V_ref_sq)
        
        return 10000*(j_time + j_smooth + j_band)

    def objective_grad(v):

        grad_time = w1 * (-1000.0 / v ** 2) / T_ref

        diff = v[1:] - v[:-1]
        grad_smooth = np.zeros_like(v)
        grad_smooth[:-1] += -2.0 * diff
        grad_smooth[1:] += 2.0 * diff
        grad_smooth *= w2 / V_ref_sq

        grad_band = w3 / V_ref_sq * (
            2.0 * np.maximum(0.0, v - v_high) - 2.0 * np.maximum(0.0, v_low - v)
        )

        return 10000.0 * (grad_time + grad_smooth + grad_band)

    def eq_soc(v):
        final_soc, _, _, _ = simulate_day_fast(v, route, start_time_ts, soc_start, baseline_p_solar, stop_gains, solar_obj, eod_cutoff_ts)
        return final_soc - target_eod_soc

    def jac_eq_soc(v):
        V, steps = _fd_perturbation_batch(v)
        final_soc, _, _ = simulate_day_fast_batch(V, route, start_time_ts, soc_start, baseline_p_solar, stop_gains, solar_obj, eod_cutoff_ts)
        return (final_soc[1:] - final_soc[0]) / steps

    def ineq_finish_time(v):
        _, finish_t, _, _ = simulate_day_fast(v, route, start_time_ts, soc_start, baseline_p_solar, stop_gains, solar_obj, eod_cutoff_ts)
        return eod_cutoff_ts - finish_t

    def jac_ineq_finish_time(v):
        V, steps = _fd_perturbation_batch(v)
        _, finish_t, _ = simulate_day_fast_batch(V, route, start_time_ts, soc_start, baseline_p_solar, stop_gains, solar_obj, eod_cutoff_ts)
        return -(finish_t[1:] - finish_t[0]) / steps

    def ineq_soc_min(v):
        _, _, soc_history, _ = simulate_day_fast(v, route, start_time_ts, soc_start, baseline_p_solar, stop_gains, solar_obj, eod_cutoff_ts)
        return np.min(soc_history) - SOC_MIN

    def jac_ineq_soc_min(v):
        V, steps = _fd_perturbation_batch(v)
        _, _, soc_hist = simulate_day_fast_batch(V, route, start_time_ts, soc_start, baseline_p_solar, stop_gains, solar_obj, eod_cutoff_ts)
        min_soc = np.min(soc_hist, axis=1)
        return (min_soc[1:] - min_soc[0]) / steps

    constraints = [
        {'type': 'ineq', 'fun': eq_soc, 'jac': jac_eq_soc},
        {'type': 'ineq', 'fun': ineq_finish_time, 'jac': jac_ineq_finish_time},
        {'type': 'ineq', 'fun': ineq_soc_min, 'jac': jac_ineq_soc_min},
    ]

    pbar = tqdm(total=50, desc="SLSQP Iterations", unit="iter")

    def progress_tracker(xk):
        pbar.update(1)
        current_mean_kmh = np.mean(xk) * 3.6
        pbar.set_postfix({'Mean Speed': f"{current_mean_kmh:.1f} km/h"})

    res = minimize(
        objective, 
        x0, 
        jac=objective_grad,
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints, 
        options={'ftol': 1e-3, 'maxiter': 50, 'disp': False},
        callback=progress_tracker
    )
    
    pbar.close()

    return res, baseline_p_solar, stop_gains

def resolve(current_km, current_time_ts, current_soc, s1_profile, loop_profile, s2_profile, loops_completed, manual_target_loops, target_eod_soc, v_guess_kmh, solar_obj, race_date, day_no):
    eod_cutoff_ts = datetime.combine(race_date, time(EOD_CUTOFF_HOUR, 0), tzinfo=SA_TZ).timestamp()
    full_route = build_day_route(s1_profile, loop_profile, s2_profile, manual_target_loops)

    k_curr = int(np.floor(current_km * 100))
    k_curr = min(k_curr, full_route['n_segments'] - 1)

    rem_route = {
        'coords': full_route['coords'][k_curr:],
        'headings': full_route['headings'][k_curr:],
        'altitudes': full_route['altitudes'][k_curr:],
        'gradients': full_route['gradients'][k_curr:],
        'delays': full_route['delays'][k_curr:],
        'n_segments': full_route['n_segments'] - k_curr
    }

    print(f"Re-solving from km {current_km:.1f} ({rem_route['n_segments'] / 100:.1f} km left) | Target Loops: {manual_target_loops}")

    res, baseline_p_solar, stop_gains = optimize_single_day(rem_route, current_time_ts, current_soc, target_eod_soc, v_guess_kmh, solar_obj, eod_cutoff_ts)

    if res.success or res.status == 9:
        opt_v_ms = res.x
        opt_v_kmh = opt_v_ms * 3.6
        
        # --- 1D Error Diffusion (Smart Quantization) ---
        quantized_kmh = np.zeros_like(opt_v_kmh)
        carry_error = 0.0
        for i in range(len(opt_v_kmh)):
            target = opt_v_kmh[i] + carry_error
            quantized_kmh[i] = np.round(target)
            carry_error = target - quantized_kmh[i]
            
        quantized_ms = quantized_kmh / 3.6
        
        # Run final simulation using the quantized integer speeds
        final_soc, finish_t, soc_history, p_mech = simulate_day_fast(quantized_ms, rem_route, current_time_ts, current_soc, baseline_p_solar, stop_gains, solar_obj, eod_cutoff_ts)

        # Stretch the quantized 1km speeds to match the 10m physics array for the dashboard
        quantized_kmh_sim = np.repeat(quantized_kmh, 100)[:rem_route['n_segments']]
        quantized_ms_sim = np.repeat(quantized_ms, 100)[:rem_route['n_segments']]

        # Calculate exact timestamps for the high-res dashboard
        dt_drive = 10.0 / quantized_ms_sim
        times = current_time_ts + np.cumsum(dt_drive) + np.cumsum(rem_route['delays'])

        save_dir = Path("Fallback Model/velocity_profiles")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"optimized_day_{day_no}.npz"
        
        np.savez(
            save_path, 
            speeds_kmh=quantized_kmh_sim, 
            soc=soc_history, 
            power_w=p_mech, 
            start_km=current_km,
            times=times,                  
            eod_cutoff_ts=eod_cutoff_ts,  
            final_soc=final_soc           
        )

        finish_str = datetime.fromtimestamp(finish_t, tz=SA_TZ).strftime("%H:%M:%S")
        print(f"Re-solve Succeeded! Finish Time: {finish_str} | Final EoD SoC: {final_soc:.2f}% | Avg Speed: {np.mean(quantized_kmh):.2f} km/h")
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

    START_SOCS = [95.0, 82.0, 63.0, 53.0, 67.0, 67.0, 53.0, 53.0]
    V_GUESSES = [70.0, 57.0, 60.0, 55.0, 60.0, 55.0, 55.0, 50.0]

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

    # ACCURATE WEATHER BINDING
    weather_filename = l_name if l_name else s1_name
    with open(f"Fallback Model/Solar_Processed/mean_{weather_filename}.jsonl", 'r') as f:
        weather_data = json.load(f)
    solar_obj = SolarIrradiance(weather_data, "period_end", "PT5M", 6)

    # MORNING CHARGING
    start_hour = 9 if DAY_NO == 1 else 8
    morning_ts = datetime.combine(race_date, time(6, 0), tzinfo=SA_TZ).timestamp()
    start_time_ts = datetime.combine(race_date, time(start_hour, 0), tzinfo=SA_TZ).timestamp()
    
    current_soc = START_SOCS[DAY_NO - 1]
    if DAY_NO != 1:
        morning_gain = calc_stationary_charge(morning_ts, start_time_ts, s1_profile['Coordinates'][0], s1_profile['Headings'][0], s1_profile['Altitude'][0], solar_obj)
        current_soc = min(SOC_MAX, current_soc + morning_gain)
        print(f"Morning charge complete. Starting day with SoC: {current_soc:.2f}%")

optimized_speeds = resolve(
        current_km=0.0,
        current_time_ts=start_time_ts,
        current_soc=82.0,
        s1_profile=s1_profile,
        loop_profile=loop_profile,
        s2_profile=s2_profile,
        loops_completed=0,
        manual_target_loops=10,
        target_eod_soc=63.0,
        v_guess_kmh=V_GUESSES[DAY_NO - 1],
        solar_obj=solar_obj,
        race_date=race_date,
        day_no=DAY_NO
    )