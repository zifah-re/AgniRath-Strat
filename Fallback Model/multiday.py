import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, time, date
from zoneinfo import ZoneInfo
from functools import lru_cache
from tqdm import tqdm
from scipy.optimize import differential_evolution
import pvlib
import matplotlib.pyplot as plt
from solar_table import SolarIrradiance

# ______ CAR CONSTANTS ______ #
MASS_KG = 300.0
G_MS2 = 9.81
CRR = 0.007
CDA_M2 = 0.16
AIR_DENSITY = 1.2
ARRAY_AREA_M2 = 5.95
ARRAY_EFFICIENCY = 0.18
PANEL_TILT = 4
ALBEDO = 0.2

# PV Thermal & Incline Physics Constants
GAMMA_TEMP_COEFF = 0.004      # 0.4% loss per degree C above 25°C
T_NOCT = 45.0                  # Nominal Operating Cell Temp (°C)
T_AMBIENT_DEFAULT = 28.0       # Fallback ambient temperature (°C)
K_CONVECTIVE_COOLING = 0.03    # Airflow cooling decay constant per m/s

# Drivetrain
MOTOR_EFF = 0.95
REGEN_EFF = 0.70
P_LOSS = 70.0

# Battery & Power Constraints
BATTERY_WH = 588.0 * 6
SOC_MIN_PCT = 20.0
SOC_MAX_PCT = 95.0
MAX_POWER_LIMIT_W = 4000.0  # Hard motor electrical draw limit (4 kW)

# Speed/Accel
V_MAX_MS = 85.0 / 3.6
A_MAX_MS = 0.5

# ____ ROUTE & TIME CONFIG ____ #
SA_TZ = ZoneInfo("Africa/Johannesburg")

DAY_DISTANCES = {
    "Day 1": {"s1": 172.7, "l": 22.6, "s2": 65.6},
    "Day 2": {"s1": 71.5, "l": 18.5, "s2": 231.0},
    "Day 3": {"s1": 8.0, "l": 39.9, "s2": 208.0},
    "Day 4": {"s1": 197.0, "l": 21.0, "s2": 63.3},
    "Day 5": {"s1": 178.0, "l": 60.7, "s2": 114.0},
    "Day 6": {"s1": 310.0, "l": 18.2, "s2": 0.0},
    "Day 7": {"s1": 261.0, "l": 16.5, "s2": 80.9},
    "Day 8": {"s1": 180.0, "l": 21.8, "s2": 98.3}
}

DAYWISE_FILES = {
    "Day 1": {"date": date(2026, 9, 10), "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 1 Boiketlong to Rustenburg", "l": "2026 Sasol Solar Challenge Route (Publish)_Day 1 _Rustenburg Loop", "s2": "2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 2 Rustenburg to Swartruggens"},
    "Day 2": {"date": date(2026, 9, 11), "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 2 Half Blind_11 Sept Stage 1 Swart Ruggens to Zeerust", "l": "SSC ROUTE FINAL_Day 2 Half Blind_Day 2 Loop", "s2": "2026 Sasol Solar Challenge Route (Publish)_Day 2 Half Blind_11 Sept Stage 2 Zeerust to Vryburg"},
    "Day 3": {"date": date(2026, 9, 12), "s1": "Day 3 probables_Probable Prahlad Route_Stage 1", "l": "Day 3 probables_Probable Prahlad Route_Day 3 Loop", "s2": "Day 3 probables_Probable Prahlad Route_Stage 2"},
    "Day 4": {"date": date(2026, 9, 13), "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 4_13 Sept Stage 1 Kimberley to Postmasburg", "l": "2026 Sasol Solar Challenge Route (Publish)_Day 4_Postmasburg Loop", "s2": "2026 Sasol Solar Challenge Route (Publish)_Day 4_13 Sept Stage 2 Postmasburg to Olifantshoek"},
    "Day 5": {"date": date(2026, 9, 14), "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 1 Olifantshoek to Upington", "l": "2026 Sasol Solar Challenge Route (Publish)_Day 5 _Upington Loop", "s2": "2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 2 Upington to Augrabies"},
    "Day 6": {"date": date(2026, 9, 15), "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 6 _15 Sept Stage 1 Augrabies to Springbok", "l": "2026 Sasol Solar Challenge Route (Publish)_Day 6 _Springbok Loop", "s2": None},
    "Day 7": {"date": date(2026, 9, 16), "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 7_16 Sept Stage 1 Springbok to Van Rhynsdorp", "l": "2026 Sasol Solar Challenge Route (Publish)_Day 7_Van Rhynsdorp Loop", "s2": "2026 Sasol Solar Challenge Route (Publish)_Day 7_16 Sept Stage 2 Van Rhynsdorp to Clanwilliam"},
    "Day 8": {"date": date(2026, 9, 17), "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 8_17 Sept Stage 1 Clanwilliam to Ceres", "l": "2026 Sasol Solar Challenge Route (Publish)_Day 8_Ceres Loop", "s2": "2026 Sasol Solar Challenge Route (Publish)_Day 8_17 Sept Stage 2 Ceres to Paarl"}
}

# ____ CACHING & FAST LOOKUPS ____ #
weather_folder = Path(r'Solar_Processed')
CACHE_FILE = "solar_grid_cache.pkl"

SOLAR_GRID_CACHE = {}
ROUTE_CACHE = {}

@lru_cache(maxsize=None)
def extractSolarData(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

@lru_cache(maxsize=1)
def get_weather_logs():
    weather_logs = {}
    for json_file in weather_folder.glob('*.jsonl'):
        stage_data_raw = extractSolarData(str(json_file))
        stage_data = {}
        for point in stage_data_raw:
            lat = point['lat']
            long = point['lon']
            dump = point['data']
            stage_data[(lat, long)] = dump
        weather_logs[json_file.stem] = SolarIrradiance(stage_data, "period_end", "PT5M", 6)
    return weather_logs

def ensure_solar_cache():
    global SOLAR_GRID_CACHE, ROUTE_CACHE
    if SOLAR_GRID_CACHE and ROUTE_CACHE:
        return

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
            SOLAR_GRID_CACHE = data["grid"]
            ROUTE_CACHE = data["route"]
    else:
        build_solar_cache()

def build_solar_cache():
    global SOLAR_GRID_CACHE, ROUTE_CACHE
    print("Pre-computing solar GTI & dynamic air_temp grids across all routes...")
    weather_logs = get_weather_logs()
    
    t_grid = np.arange(5.5 * 3600, 18.5 * 3600, 300.0)
    n_times = len(t_grid)
    
    unique_stages = []
    for day, files in DAYWISE_FILES.items():
        for key in ["s1", "l", "s2"]:
            fname = files.get(key)
            if fname and fname not in unique_stages:
                unique_stages.append(fname)

    sky_factor = (1 + np.cos(np.radians(PANEL_TILT))) / 2.0
    ground_factor = (1 - np.cos(np.radians(PANEL_TILT))) / 2.0
    b_constants = sky_factor + ALBEDO * ground_factor

    for fname in tqdm(unique_stages, desc="Caching Solar & Temp Grids"):
        if fname in SOLAR_GRID_CACHE:
            continue
        
        route_data = extractSolarData(f"Saves/{fname}.kml.save")["profile"]
        ROUTE_CACHE[fname] = route_data
        
        coords = np.array(route_data['Coordinates'])
        n_points = len(coords)
        
        headings = np.array(route_data['Headings'])
        altitude = np.array(route_data['Altitude'])
        
        if len(headings) < n_points:
            headings = np.pad(headings, (0, n_points - len(headings)), mode='edge')
        elif len(headings) > n_points:
            headings = headings[:n_points]
            
        if np.isscalar(altitude) or len(np.atleast_1d(altitude)) == 1:
            altitude = np.full(n_points, altitude)
        elif len(altitude) < n_points:
            altitude = np.pad(altitude, (0, n_points - len(altitude)), mode='edge')
        elif len(altitude) > n_points:
            altitude = altitude[:n_points]

        solar_obj = weather_logs[f'mean_{fname}']
        
        ref_date = date(2026, 9, 10)
        for day, files in DAYWISE_FILES.items():
            if fname in [files.get("s1"), files.get("l"), files.get("s2")]:
                ref_date = files["date"]
                break
                
        dt_base = datetime.combine(ref_date, time(0, 0), tzinfo=SA_TZ).timestamp()
        gti_matrix = np.zeros((n_points, n_times))
        temp_matrix = np.zeros((n_points, n_times))
        
        mid_coord = tuple(coords[n_points // 2])
        mean_lat = float(np.mean(coords[:, 0]))
        mean_lon = float(np.mean(coords[:, 1]))
        mean_alt = float(np.mean(altitude))
        
        tz_times = pd.to_datetime(dt_base + t_grid, unit='s', utc=True).tz_convert('Africa/Johannesburg')
        solpos = pvlib.solarposition.get_solarposition(tz_times, mean_lat, mean_lon, altitude=mean_alt)
        
        zenith_arr = solpos['apparent_zenith'].values
        azimuth_arr = solpos['azimuth'].values
        zenith_rad = np.radians(zenith_arr)
        
        for j, t_sec in enumerate(t_grid):
            t_epoch = dt_base + t_sec
            aoi = pvlib.irradiance.aoi(PANEL_TILT, headings, zenith_arr[j], azimuth_arr[j])
            a_factor = np.cos(np.radians(aoi)) - (np.cos(zenith_rad[j]) * sky_factor)
            
            weather_data = solar_obj[(mid_coord, t_epoch)]
            dni, ghi, air_temp = weather_data.data(["dni", "ghi", "air_temp"])
            
            gti_matrix[:, j] = (a_factor * dni + b_constants * ghi) * ARRAY_EFFICIENCY * ARRAY_AREA_M2
            
            # 2D Spatial-Temporal temperature storage
            if np.isscalar(air_temp):
                temp_matrix[:, j] = float(air_temp)
            else:
                air_temp_arr = np.atleast_1d(air_temp)
                if len(air_temp_arr) == n_points:
                    temp_matrix[:, j] = air_temp_arr
                else:
                    temp_matrix[:, j] = float(np.mean(air_temp_arr))
            
        SOLAR_GRID_CACHE[fname] = {
            "t_grid": t_grid,
            "matrix": gti_matrix,
            "temp_matrix": temp_matrix
        }
        
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"grid": SOLAR_GRID_CACHE, "route": ROUTE_CACHE}, f)
    print("Pre-computation complete & saved to disk!\n")

def solar_fast(time_base, fname, v_actual=None, gradient=None):
    """
    Advanced Solar Power Computation featuring:
    1. Pitch-adjusted effective panel tilt over gradients.
    2. 2D spatial-temporal weather air_temp for cell heating & thermal derating.
    3. Vehicle airspeed convective cooling.
    """
    ensure_solar_cache()
    cache = SOLAR_GRID_CACHE[fname]
    t_grid = cache["t_grid"]
    matrix = cache["matrix"]
    
    t_seconds = (time_base + 7200.0) % 86400
    n_points = len(time_base)
    
    # 1. Base Global Tilted Irradiance Power (Watts at STC: 25°C)
    if matrix.shape[0] == n_points:
        raw_solar_pwr = np.empty(n_points)
        for i in range(n_points):
            raw_solar_pwr[i] = np.interp(t_seconds[i], t_grid, matrix[i])
    else:
        raw_solar_pwr = np.interp(t_seconds, t_grid, matrix[0])

    # 2. Road Pitch Adjustment
    if gradient is not None and len(gradient) == n_points:
        pitch_rad = np.arctan(gradient / 100.0)
        pitch_corr_factor = np.cos(pitch_rad + np.radians(PANEL_TILT)) / np.cos(np.radians(PANEL_TILT))
        pitch_corr_factor = np.clip(pitch_corr_factor, 0.75, 1.25)
        raw_solar_pwr *= pitch_corr_factor

    # 3. Dynamic Ambient Temperature Lookup
    if "temp_matrix" in cache:
        temp_mat = cache["temp_matrix"]
        if temp_mat.shape[0] == n_points:
            t_ambient = np.empty(n_points)
            for i in range(n_points):
                t_ambient[i] = np.interp(t_seconds[i], t_grid, temp_mat[i])
        else:
            t_ambient = np.interp(t_seconds, t_grid, temp_mat[0])
    elif "temp_grid" in cache:
        t_ambient = np.interp(t_seconds, t_grid, cache["temp_grid"])
    else:
        t_ambient = np.full(n_points, T_AMBIENT_DEFAULT)

    # 4. PV Cell Temperature & Thermal Efficiency Derating
    if v_actual is None or len(v_actual) != n_points:
        v_actual = np.full(n_points, 55.0 / 3.6)
        
    irradiance_wm2 = raw_solar_pwr / (ARRAY_AREA_M2 * ARRAY_EFFICIENCY + 1e-6)
    
    convective_cooling = np.exp(-K_CONVECTIVE_COOLING * v_actual)
    t_cell = t_ambient + (irradiance_wm2 / 800.0) * (T_NOCT - 20.0) * convective_cooling
    
    temp_derating = np.clip(1.0 - GAMMA_TEMP_COEFF * (t_cell - 25.0), 0.70, 1.05)
    
    return raw_solar_pwr * temp_derating

# ____ POWER MODEL & ROBUST SPEED LIMITER ____ #
def enforce_power_limit(v_target_ms, grad_pct):
    grad = grad_pct / 100.0
    A = 0.5 * AIR_DENSITY * CDA_M2
    f_roll = MASS_KG * G_MS2 * CRR * (1.0 - (grad ** 2) / 2.0)
    f_grav = MASS_KG * G_MS2 * grad
    B = f_roll + f_grav
    
    p_mech_target = (A * (v_target_ms ** 2) + B) * v_target_ms
    p_max_mech = MAX_POWER_LIMIT_W * MOTOR_EFF
    
    mask = p_mech_target > p_max_mech
    if not np.any(mask):
        return v_target_ms
    
    v_actual = np.copy(v_target_ms)
    v_sub = v_actual[mask]
    B_sub = B[mask]
    
    for _ in range(8):
        f_val = A * (v_sub ** 3) + B_sub * v_sub - p_max_mech
        f_prime = 3.0 * A * (v_sub ** 2) + B_sub
        f_prime = np.where(np.abs(f_prime) < 1e-6, 1e-6, f_prime)
        v_sub = v_sub - f_val / f_prime

    v_actual[mask] = np.maximum(v_sub, 5.0 / 3.6)
    return v_actual

def net_power(v, grad, solar_pwr):
    grad = grad / 100.0
    f_drag = 0.5 * AIR_DENSITY * CDA_M2 * (v)**2
    f_roll = MASS_KG * G_MS2 * CRR * (1.0 - (grad ** 2) / 2.0)
    f_grav = MASS_KG * G_MS2 * grad

    f_total = f_drag + f_roll + f_grav
    p_mech = f_total * v

    p_motor_draw = np.where(p_mech > 0, np.minimum(p_mech / MOTOR_EFF, MAX_POWER_LIMIT_W), 0.0)
    p_mech_eff = np.where(p_mech < 0, p_mech * REGEN_EFF, p_mech / MOTOR_EFF)
    p_electric = solar_pwr - p_mech_eff - P_LOSS
    
    return p_electric, p_motor_draw

def stage_soc_profile(v_ms_target, fname, start_date, start_time, soc_start):
    ensure_solar_cache()
    route = ROUTE_CACHE[fname]
    distances = np.array(route['Distance']) * 1000.0
    grad = np.array(route['Gradient'])
    
    v_actual = enforce_power_limit(np.full(len(grad), v_ms_target), grad)
    
    dx = np.diff(distances)
    v_avg = np.maximum(0.5 * (v_actual[:-1] + v_actual[1:]), 1.0 / 3.6)
    dt = np.concatenate(([0], dx / v_avg))
    time_base = start_time + dt.cumsum()
    
    solar_irr = solar_fast(time_base, fname, v_actual=v_actual, gradient=grad)
    power, p_motor_draw = net_power(v_actual, grad, solar_irr)
    energy = power * dt
    soc = soc_start + (np.cumsum(energy / (BATTERY_WH * 3600.0))) * 100.0
    
    max_stage_power = float(np.max(p_motor_draw))
    
    return soc, power, time_base, max_stage_power

def stitch_loops(n, t_start, soc_start, solar_obj, coords, altitude, headings, distances, fname, start_date, v_loop_kmh=55.0):
    loop_start = t_start
    loop_soc_start = soc_start
    soc_profile = []
    end_time = t_start
    max_loop_power = 0.0

    for i in range(n):
        soc, _, end_time, stage_pwr = stage_soc_profile(v_loop_kmh / 3.6, fname, start_date, loop_start, loop_soc_start)
        max_loop_power = max(max_loop_power, stage_pwr)
        
        loop_end_ts = float(end_time[-1])
        loop_next_start_ts = loop_end_ts + 300.0
        
        loop_soc_start = charged(soc[-1], loop_end_ts, loop_next_start_ts, coords[0], headings[0], altitude[0], solar_obj, fname)
        loop_start = loop_next_start_ts
        soc_profile.append(soc)

    if len(soc_profile) > 0:
        return np.concatenate(soc_profile), end_time, max_loop_power
    else:
        return np.array([loop_soc_start]), np.array([t_start]), 0.0

def charged(soc, start_ts, end_ts, coords, heading, altitude, solar_obj, fname=None):
    if start_ts >= end_ts:
        return soc

    dt = 300.0
    time_base = np.arange(start_ts, end_ts, dt)
    n_points = len(time_base)

    if n_points == 0:
        return soc

    if fname and fname in SOLAR_GRID_CACHE:
        solar_pwr = solar_fast(time_base, fname, v_actual=np.zeros(n_points), gradient=np.zeros(n_points))
    else:
        ensure_solar_cache()
        if fname in SOLAR_GRID_CACHE:
            solar_pwr = solar_fast(time_base, fname, v_actual=np.zeros(n_points), gradient=np.zeros(n_points))
        else:
            solar_pwr = np.zeros(n_points)

    net_pwr = np.maximum(solar_pwr - 10.0, 0.0)
    energy_wh = np.sum(net_pwr * dt) / 3600.0
    soc_gained = (energy_wh / BATTERY_WH) * 100.0
    return min(SOC_MAX_PCT, soc + soc_gained)

# ____ DIAGNOSTICS & PLOTTING ____ #
def diagnose_solar_physics(fname, v_kmh=60.0, start_hour=9.0):
    """
    Diagnostic plot showing intermediate solar physics variables.
    
    Parameters:
    - fname: Route filename stage
    - v_kmh: Vehicle cruising speed (km/h)
    - start_hour: Departure time in 24h float format (e.g. 9.0 = 9:00 AM)
    """
    ensure_solar_cache()
    route = ROUTE_CACHE[fname]
    grad = np.array(route['Gradient'])
    n_points = len(grad)
    
    # Set departure time based on stage schedule (Day 1 Stage 1 starts at 9:00 AM)
    start_time = start_hour * 3600.0 
    v_actual = enforce_power_limit(np.full(n_points, v_kmh / 3.6), grad)
    
    dx = np.diff(np.array(route['Distance']) * 1000.0)
    v_avg = np.maximum(0.5 * (v_actual[:-1] + v_actual[1:]), 1.0 / 3.6)
    dt = np.concatenate(([0], dx / v_avg))
    time_base = start_time + dt.cumsum()
    
    cache = SOLAR_GRID_CACHE[fname]
    t_seconds = (time_base + 7200.0) % 86400
    
    # 1. Base Global Tilted Irradiance Power (STC 25°C)
    if cache["matrix"].shape[0] == n_points:
        stc_power = np.array([np.interp(t_seconds[i], cache["t_grid"], cache["matrix"][i]) for i in range(n_points)])
    else:
        stc_power = np.interp(t_seconds, cache["t_grid"], cache["matrix"][0])

    # 2. Ambient Air Temperature
    if "temp_matrix" in cache:
        temp_mat = cache["temp_matrix"]
        if temp_mat.shape[0] == n_points:
            t_ambient = np.array([np.interp(t_seconds[i], cache["t_grid"], temp_mat[i]) for i in range(n_points)])
        else:
            t_ambient = np.interp(t_seconds, cache["t_grid"], temp_mat[0])
    elif "temp_grid" in cache:
        t_ambient = np.interp(t_seconds, cache["t_grid"], cache["temp_grid"])
    else:
        t_ambient = np.full(n_points, T_AMBIENT_DEFAULT)
    
    # 3. Dynamic Cell Heating & Airflow Cooling
    irradiance_wm2 = stc_power / (ARRAY_AREA_M2 * ARRAY_EFFICIENCY + 1e-6)
    convective_cooling = np.exp(-K_CONVECTIVE_COOLING * v_actual)
    t_cell = t_ambient + (irradiance_wm2 / 800.0) * (T_NOCT - 20.0) * convective_cooling
    temp_derating = np.clip(1.0 - GAMMA_TEMP_COEFF * (t_cell - 25.0), 0.70, 1.05)
    
    # 4. Incline Pitch Correction
    pitch_rad = np.arctan(grad / 100.0)
    pitch_factor = np.clip(np.cos(pitch_rad + np.radians(PANEL_TILT)) / np.cos(np.radians(PANEL_TILT)), 0.75, 1.25)
    adjusted_power = stc_power * pitch_factor * temp_derating

    # --- PLOTTING ---
    dist_km = np.array(route['Distance'])
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    start_time_str = f"{int(start_hour):02d}:{int((start_hour%1)*60):02d}"
    end_hour_val = time_base[-1] / 3600.0
    end_time_str = f"{int(end_hour_val):02d}:{int((end_hour_val%1)*60):02d}"
    
    axs[0].plot(dist_km, stc_power, 'g--', label='Baseline Power (STC 25°C)')
    axs[0].plot(dist_km, adjusted_power, 'b-', label='Real-World Thermal & Pitch Power')
    axs[0].set_ylabel('Solar Power (W)')
    axs[0].set_title(f'Solar Physics Breakdown @ {v_kmh} km/h ({start_time_str} to {end_time_str}): {fname}', fontweight='bold')
    axs[0].legend(loc='upper left')
    axs[0].grid(True)
    
    axs[1].plot(dist_km, t_cell, 'r-', label='Cell Temp (°C)')
    axs[1].plot(dist_km, t_ambient, 'k:', label='Ambient Air Temp (°C)')
    axs[1].axhline(25, color='green', linestyle=':', label='STC Temp (25°C)')
    axs[1].set_ylabel('Temp (°C)')
    axs[1].legend(loc='upper left')
    axs[1].grid(True)
    
    axs[2].plot(dist_km, temp_derating * 100.0, 'orange')
    axs[2].set_ylabel('Efficiency (%)')
    axs[2].set_title('Thermal Efficiency Derating (relative to 100% at 25°C)', fontsize=10)
    axs[2].grid(True)
    
    power_lost = stc_power - adjusted_power
    axs[3].plot(dist_km, power_lost, 'purple')
    axs[3].set_ylabel('Power Lost (W)')
    axs[3].set_xlabel('Stage Distance (km)')
    axs[3].set_title('Net Overheating & Pitch Power Losses', fontsize=10)
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.savefig("solar_physics_diagnostics.png", dpi=300)
    plt.show()

# ____ SIMULATION ENGINE ____ #
def stitchAllDays(n=None, v=None):
    day_names = [f"Day {i}" for i in range(1, 9)]

    if n is None:
        n = [6, 4, 3, 5, 3, 2, 1, 1]

    v_map = {
        "Day 1": [v[0][0], (v[0][0] + v[0][1]) / 2.0, v[0][1]],
        "Day 2": [v[1][0], (v[1][0] + v[1][1]) / 2.0, v[1][1]],
        "Day 3": [v[2][0], (v[2][0] + v[2][1]) / 2.0, v[2][1]],
        "Day 4": [v[3][0], (v[3][0] + v[3][1]) / 2.0, v[3][1]],
        "Day 5": [v[4][0], (v[4][0] + v[4][1]) / 2.0, v[4][1]],
        "Day 6": [v[5][0], (v[5][0] + v[5][1]) / 2.0, v[5][1]],
        "Day 7": [v[6][0], (v[6][0] + v[6][1]) / 2.0, v[6][1]],
        "Day 8": [v[7][0], (v[7][0] + v[7][1]) / 2.0, v[7][1]],
    }

    soc = 95.0
    x_all = []
    y_all = []
    day_end_times = []
    cumulative_distance = 0.0
    weather_logs = get_weather_logs()
    min_soc_recorded = soc
    total_overtime_sec = 0.0
    max_power_recorded_w = 0.0

    for day_idx, day in enumerate(day_names):
        day_no = day_idx + 1
        start_date = DAYWISE_FILES[day]["date"]

        s1_name = DAYWISE_FILES[day]["s1"]
        route_s1 = ROUTE_CACHE.get(s1_name) or extractSolarData(f"Saves/{s1_name}.kml.save")["profile"]
        distances_s1 = np.array(route_s1["Distance"])
        s1_solar_obj = weather_logs[f"mean_{s1_name}"]

        l_name = DAYWISE_FILES[day]["l"]
        l_dist = DAY_DISTANCES[day]["l"]
        solar_obj_l = weather_logs[f"mean_{l_name}"]
        route_loop = ROUTE_CACHE.get(l_name) or extractSolarData(f"Saves/{l_name}.kml.save")["profile"]
        coords_l, headings_l, altitude_l = (
            np.array(route_loop["Coordinates"]),
            np.array(route_loop["Headings"]),
            np.array(route_loop["Altitude"]),
        )
        single_loop_dist = np.array(route_loop["Distance"])

        s2_name = DAYWISE_FILES[day].get("s2")
        try:
            route_s2 = ROUTE_CACHE.get(s2_name) or extractSolarData(f"Saves/{s2_name}.kml.save")["profile"]
            distances_s2 = np.array(route_s2["Distance"])
            s2_solar_obj = weather_logs[f"mean_{s2_name}"]
            has_s2 = True
        except Exception:
            route_s2 = None
            distances_s2 = np.array([0.0])
            s2_solar_obj = None
            has_s2 = False

        # Morning Charging
        stage_start_hour = 9 if day_no == 1 else 8
        morning_start_ts = datetime.combine(start_date, time(6, 0), tzinfo=SA_TZ).timestamp()
        morning_end_ts = datetime.combine(start_date, time(stage_start_hour, 0), tzinfo=SA_TZ).timestamp()

        m_coords = route_s1["Coordinates"][0]
        m_heading = route_s1["Headings"][0]
        m_alt = route_s1["Altitude"][0]

        soc = charged(soc, morning_start_ts, morning_end_ts, m_coords, m_heading, m_alt, s1_solar_obj, s1_name)
        min_soc_recorded = min(min_soc_recorded, soc)

        start_time_ts = morning_end_ts

        v_s1_ms = v_map[day][0] / 3.6
        soc_s1, power_s1, time_s1, pwr_s1 = stage_soc_profile(v_s1_ms, s1_name, start_date, start_time_ts, soc)
        s1_end_time = float(time_s1[-1])
        min_soc_recorded = min(min_soc_recorded, float(np.min(soc_s1)))
        max_power_recorded_w = max(max_power_recorded_w, pwr_s1)

        last_coords = route_s1["Coordinates"][-1]
        last_heading = route_s1["Headings"][-1]
        last_alt = route_s1["Altitude"][-1]
        last_solar_obj = s1_solar_obj
        last_fname = s1_name

        cs_start_ts = s1_end_time
        cs_end_ts = cs_start_ts + 1800.0 
        soc_before_cs = soc_s1[-1]
        soc_after_cs = charged(soc_before_cs, cs_start_ts, cs_end_ts, last_coords, last_heading, last_alt, last_solar_obj, s1_name)
        min_soc_recorded = min(min_soc_recorded, soc_after_cs)

        current_time = cs_end_ts
        current_soc = soc_after_cs
        n_loops = n[day_idx]
        v_loop_kmh = v_map[day][1]

        if n_loops > 0:
            loop_soc, current_time, pwr_loop = stitch_loops(
                n_loops, current_time, current_soc, solar_obj_l,
                coords_l, altitude_l, headings_l, l_dist, l_name, start_date, v_loop_kmh=v_loop_kmh
            )
            current_time = float(current_time[-1])
            current_soc = float(loop_soc[-1])
            min_soc_recorded = min(min_soc_recorded, float(np.min(loop_soc)))
            max_power_recorded_w = max(max_power_recorded_w, pwr_loop)
        else:
            loop_soc = np.array([])

        if has_s2 and s2_name is not None:
            v_s2_ms = v_map[day][2] / 3.6
            soc_s2, power_s2, time_s2, pwr_s2 = stage_soc_profile(v_s2_ms, s2_name, start_date, current_time, soc_start=current_soc)
            day_end_time = float(time_s2[-1])
            last_coords = route_s2["Coordinates"][-1]
            last_heading = route_s2["Headings"][-1]
            last_alt = route_s2["Altitude"][-1]
            last_solar_obj = s2_solar_obj
            last_fname = s2_name
            min_soc_recorded = min(min_soc_recorded, float(np.min(soc_s2)))
            max_power_recorded_w = max(max_power_recorded_w, pwr_s2)
        else:
            soc_s2 = np.array([])
            day_end_time = current_time

        day_end_times.append(day_end_time)

        # Daily Hard Deadline
        target_cutoff_hour = 15 if day_no == 8 else 17
        cutoff_ts = datetime.combine(start_date, time(target_cutoff_hour, 0), tzinfo=SA_TZ).timestamp()
        if day_end_time > cutoff_ts:
            total_overtime_sec += (day_end_time - cutoff_ts)

        if n_loops > 0:
            distances_loops_stacked = np.concatenate([single_loop_dist + k * l_dist for k in range(n_loops)])
            x_loops = distances_s1[-1] + distances_loops_stacked
            x_s2 = x_loops[-1] + distances_s2 if (has_s2 and len(soc_s2) > 0) else np.array([])
            x_day = np.concatenate([x for x in (distances_s1, x_loops, x_s2) if len(x) > 0])
            y_day = np.concatenate([y for y in (soc_s1, loop_soc, soc_s2) if len(y) > 0])
        else:
            x_s2 = distances_s1[-1] + distances_s2 if (has_s2 and len(soc_s2) > 0) else np.array([])
            x_day = np.concatenate([x for x in (distances_s1, x_s2) if len(x) > 0])
            y_day = np.concatenate([y for y in (soc_s1, soc_s2) if len(y) > 0])

        x_all.append(cumulative_distance + x_day)
        y_all.append(y_day)
        cumulative_distance += x_day[-1]
        final_soc = y_day[-1] if len(y_day) > 0 else current_soc

        # Evening Charging up to 17:00
        lock_ts = datetime.combine(start_date, time(17, 0), tzinfo=SA_TZ).timestamp()
        if day_end_time < lock_ts:
            final_soc = charged(final_soc, day_end_time, lock_ts, last_coords, last_heading, last_alt, last_solar_obj, last_fname)

        soc = final_soc
        min_soc_recorded = min(min_soc_recorded, soc)

    return x_all, y_all, min_soc_recorded, day_end_times, total_overtime_sec, max_power_recorded_w

# ---------------------------------------------------------
# OBJECTIVE FUNCTION & DE OPTIMIZER
# ---------------------------------------------------------
def objective(x):
    v_flat = x[:16]
    n = [int(val) for val in x[16:]]
    v = [(int(v_flat[i]), int(v_flat[i + 1])) for i in range(0, 16, 2)]

    res = stitchAllDays(n, v)
    if res is False or res is None:
        return 1e6

    x_all, soc, min_soc, _, overtime_sec, max_power_w = res
    total_distance = float(x_all[-1][-1])

    penalty = 0.0

    if min_soc < SOC_MIN_PCT:
        penalty += (SOC_MIN_PCT - min_soc) * 1000.0

    if overtime_sec > 0:
        penalty += (overtime_sec / 60.0) * 100.0

    if max_power_w > MAX_POWER_LIMIT_W:
        penalty += (max_power_w - MAX_POWER_LIMIT_W) * 50.0

    return -total_distance + penalty

def plot_soc_profile(optimal_v, optimal_n):
    print("\nGenerating SOC plot...")
    x_all, y_all, _, _, _, max_pwr = stitchAllDays(optimal_n, optimal_v)
    
    plt.figure(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for day_idx in range(len(x_all)):
        plt.plot(
            x_all[day_idx], 
            y_all[day_idx], 
            label=f"Day {day_idx + 1}", 
            color=colors[day_idx], 
            linewidth=2
        )
        if day_idx < len(x_all) - 1:
            day_end_x = x_all[day_idx][-1]
            plt.axvline(x=day_end_x, color='gray', linestyle=':', alpha=0.6)

    plt.axhline(y=SOC_MIN_PCT, color='red', linestyle='--', label=f'Min SOC Limit ({SOC_MIN_PCT:.0f}%)')
    plt.axhline(y=SOC_MAX_PCT, color='green', linestyle='--', label=f'Max SOC ({SOC_MAX_PCT:.0f}%)')
    
    plt.title(f"8-Day Thermal & Slope-Aware SOC Profile (Peak Power: {max_pwr:.0f} W)", fontsize=14, fontweight='bold')
    plt.xlabel("Cumulative Distance (km)", fontsize=12)
    plt.ylabel("State of Charge (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="lower left", bbox_to_anchor=(1, 0.1), frameon=True)
    plt.tight_layout()
    
    plt.savefig("optimal_soc_profile.png", dpi=300)
    print("Plot saved as 'optimal_soc_profile.png'.")
    plt.show()

def run_joint_optimization():
    speed_bounds = [(45, 75)] * 16
    loop_bounds = [(1, 10)] * 8  
    bounds = speed_bounds + loop_bounds
    integrality = [True] * 24

    v_init = [65, 60, 60, 55, 60, 55, 55, 55, 60, 55, 55, 50, 55, 50, 50, 50]
    n_init = [6, 4, 3, 5, 3, 2, 1, 1]
    x0_guess = np.array(v_init + n_init)

    max_generations = 100
    pbar = tqdm(total=max_generations, desc="Joint Optimization (v & n)", unit="gen")

    def callback(xk, convergence):
        pbar.update(1)
        pbar.set_postfix({"Convergence": f"{convergence:.3f}"})

    result = differential_evolution(
        objective,
        bounds,
        x0=x0_guess,
        integrality=integrality,
        popsize=10,             
        maxiter=max_generations,
        workers=8,             
        updating='deferred',
        mutation=(0.5, 1.2),
        recombination=0.9,
        callback=callback,
        disp=False
    )

    pbar.close()

    best_x = result.x
    optimal_v = [(int(best_x[i]), int(best_x[i + 1])) for i in range(0, 16, 2)]
    optimal_n = [int(val) for val in best_x[16:]]

    x_all, _, min_soc, day_end_times, overtime_sec, max_pwr = stitchAllDays(optimal_n, optimal_v)
    best_dist = float(x_all[-1][-1])

    print("\n--- Thermal & Slope-Aware Optimization Complete ---")
    print(f"Maximized Distance: {best_dist:.2f} km")
    print(f"Peak Motor Power Draw: {max_pwr:.1f} W (Limit: {MAX_POWER_LIMIT_W:.0f} W)")
    print(f"Minimum Battery SOC: {min_soc:.1f}%\n")
    print("Optimal Schedule (SAST Local Times):")
    for day in range(8):
        end_time_str = datetime.fromtimestamp(day_end_times[day], tz=SA_TZ).strftime("%H:%M:%S SAST")
        print(f"  Day {day+1}: Loops = {optimal_n[day]}, Stage Speeds = {optimal_v[day]} km/h | Finish Time: {end_time_str}")

    return optimal_v, optimal_n, best_dist



# Run diagnostic starting at 9:00 AM for Day 1 Stage 1
if __name__ == '__main__':
    ensure_solar_cache()
    s1_file = DAYWISE_FILES["Day 1"]["s1"]
    diagnose_solar_physics(s1_file, v_kmh=60.0, start_hour=9.0)  # Run full optimization
    optimal_v, optimal_n, best_dist = run_joint_optimization()
    plot_soc_profile(optimal_v, optimal_n)