"""
Solar Car CasADi IPOPT MPC & Solar Modeling
Author: Diyaansh K.
"""

import pandas as pd
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import pvlib
import time
from scipy.ndimage import uniform_filter1d

# ==========================================
# VEHICLE & ENVIRONMENT CONFIGURATION
# ==========================================
MASS = 300.0          
CDA = 0.15            
CRR = 0.005           
RHO = 1.2             
G = 9.81              

SOLAR_AREA = 6.0      
SOLAR_EFF = 0.20      
MOTOR_EFF = 0.95      
REGEN_EFF = 0.70      
BATT_CAPACITY_WH = 3528.0  
DT = 1.0     
N = 10       

# Solar Location & Panel Specs
LAT = 13.037206951836724
LON = 79.89299347657726
PANEL_TILT = 4    
ALBEDO = 0.2

# Fallback start time
RACE_START_TIME = "2026-05-03 10:00:00+05:30" 

def run_mpc_simulation(telemetry_file, solar_file):
    
    df_tel = pd.read_json(telemetry_file, lines=True)
    df_tel['Vehicle_Velocity'] = df_tel['Vehicle_Velocity'].replace([np.inf, -np.inf], np.nan)
    actual_v = df_tel['Vehicle_Velocity'].interpolate(method='linear').to_numpy()
    
    if '_rx_time' in df_tel.columns:
        df_tel['time'] = pd.to_datetime(df_tel['_rx_time'], utc=False)
    else:
        df_tel['time'] = pd.date_range(start=RACE_START_TIME, periods=len(actual_v), freq=f'{int(DT)}S')
        
    df_tel_time = pd.DataFrame({'time': df_tel['time'], 'velocity': actual_v})
    
    offline_v = uniform_filter1d(actual_v, size=20)

    df_sol = pd.read_json(solar_file, lines=True)
    df_sol['period_end'] = pd.to_datetime(df_sol['period_end'])
    
    solpos = pvlib.solarposition.get_solarposition(df_sol['period_end'], LAT, LON)

    b_constant = ((1 + np.cos(np.radians(PANEL_TILT))) / 2) + ALBEDO * ((1 - np.cos(np.radians(PANEL_TILT))) / 2)
    a_variants = []
    for heading in [0, 90, 180, 270]:  
        aoi = pvlib.irradiance.aoi(PANEL_TILT, heading, solpos['apparent_zenith'], solpos['azimuth'])
        a_heading = np.cos(np.radians(aoi)) - (np.cos(np.radians(solpos['apparent_zenith'])) * ((1 + np.cos(np.radians(PANEL_TILT))) / 2))
        a_variants.append(a_heading)
    df_sol['A_coeff'] = np.mean(a_variants, axis=0)
    df_sol['B_coeff'] = b_constant
    df_sol.loc[(solpos['zenith'] >= 90).values, ['A_coeff', 'B_coeff']] = 0
    df_sol['gti'] = (df_sol['A_coeff'] * df_sol['dni']) + (df_sol['B_coeff'] * df_sol['ghi'])
    
    df_sol['gti'] = uniform_filter1d(df_sol['gti'], size=30)
    
    df_sol = df_sol.sort_values('period_end')
    merged = pd.merge_asof(df_tel_time.sort_values('time'), df_sol[['period_end', 'gti']], 
                           left_on='time', right_on='period_end', direction='nearest')
    solar_gti_aligned = merged['gti'].values
    
    v_opt = ca.MX.sym('v', N)
    soc_opt = ca.MX.sym('soc', N)
    v_ref_p = ca.MX.sym('v_ref_p', N)
    solar_p = ca.MX.sym('solar_p', N)
    v_prev_p = ca.MX.sym('v_prev_p')
    soc_prev_p = ca.MX.sym('soc_prev_p')
    
    cost = 0.0
    g_vec = []
    soc_current, v_current = soc_prev_p, v_prev_p
    
    for i in range(N):
        # Physics (Same as before)
        f_drag = 0.5 * RHO * CDA * (v_opt[i] ** 2)
        f_rolling = MASS * G * CRR
        f_accel = MASS * (v_opt[i] - v_current) / DT
        p_mech = (f_drag + f_rolling + f_accel) * v_opt[i]
        p_elec = ca.if_else(p_mech >= 0, p_mech / MOTOR_EFF, p_mech * REGEN_EFF)
        energy_change_wh = (SOLAR_AREA * SOLAR_EFF * solar_p[i] - p_elec) * DT / 3600.0
        soc_next = soc_current + (energy_change_wh / BATT_CAPACITY_WH) * 100.0
        g_vec.append(soc_opt[i] - soc_next)
        
        cost += 1.0 * ca.sumsqr(v_opt[i] - v_ref_p[i])
        # 2. SOC Floor Penalty
        cost += 1000.0 * ca.sumsqr(ca.fmax(20.0 - soc_opt[i], 0))
        # 3. Velocity Continuity
        cost += 0.5 * ca.sumsqr(v_opt[i] - v_current)
        soc_current = soc_opt[i]
        v_current = v_opt[i]
        
    X = ca.vertcat(v_opt, soc_opt)
    P = ca.vertcat(v_ref_p, solar_p, v_prev_p, soc_prev_p)
    G_eq = ca.vertcat(*g_vec)
    
    nlp = {'x': X, 'f': cost, 'g': G_eq, 'p': P}
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, opts)
    
    # ==========================================
    # 5. EXECUTE ROLLING HORIZON LOOP
    # ==========================================
    mpc_v = []
    mpc_soc = []
    mpc_log = []
    current_soc = 95.0  
    current_v = actual_v[0]
    
    sim_steps = len(actual_v) - N
    
    print(f"\n{'='*50}\n  MPC SOLVER STARTING \n{'='*50}")
    
    for step in range(sim_steps):
        t0 = time.perf_counter()
        
        window_v_ref = offline_v[step:step+N]
        window_solar = solar_gti_aligned[step:step+N]
        
        p_val = ca.vertcat(window_v_ref, window_solar, current_v, current_soc)
        x0_guess = list(window_v_ref) + [current_soc] * N
        
        res = solver(x0=x0_guess, p=p_val, lbg=0, ubg=0)
        solve_time_ms = (time.perf_counter() - t0) * 1000
        
        sol_x = res['x'].full().flatten()
        optimal_v = sol_x[0]
        next_soc = sol_x[N]
        
        # Logging
        if step % 60 == 0:
            print(f"[{int(step/60)}m] Speed: {optimal_v*3.6:.1f} km/h | SOC: {current_soc:.2f}% | Solve: {solve_time_ms:.1f}ms")
        
        mpc_log.append({
            'elapsed_t_s': step * DT,
            'actual_spd_kmh': optimal_v * 3.6,
            'soc_pct': current_soc,
            'solve_ms': solve_time_ms
        })
        
        mpc_v.append(optimal_v)
        mpc_soc.append(current_soc)
        
        current_v = optimal_v
        current_soc = next_soc
        
    # Save Logs
    pd.DataFrame(mpc_log).to_csv('mpc_testing.csv', index=False)
    print(f"\n{'='*50}\n  Simulation complete. Logs saved to mpc_testing.csv\n{'='*50}")
    
    mpc_v = np.pad(mpc_v, (0, N), mode='edge')
    mpc_soc = np.pad(mpc_soc, (0, N), mode='edge')
    
    # ==========================================
    # 6. RENDER VISUALIZATION
    # ==========================================
    timestamps = np.arange(len(actual_v)) * DT
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
    
    # Top Panel: Velocity (km/h)
    ax1.plot(timestamps, offline_v * 3.6, color='#555555', linewidth=1.5, linestyle='--', label='Offline Plan')
    ax1.plot(timestamps, actual_v * 3.6, color='#E87722', linewidth=1.0, alpha=0.6, label='Actual Velocity')
    ax1.plot(timestamps, mpc_v * 3.6, color='#3FA1DF', linewidth=2.0, label='MPC Recommended Target')
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title('Velocity Tracking, Solar Input, and Battery SOC', fontsize=12, pad=15)
    ax1.legend(frameon=False, loc='upper right')
    
    # Middle Panel: Solar Irradiance
    ax2.plot(timestamps, solar_gti_aligned, color='#F5C518', linewidth=1.5, label='Estimated Panel GTI')
    ax2.fill_between(timestamps, solar_gti_aligned, color='#F5C518', alpha=0.2)
    ax2.set_ylabel('Irradiance (W/m²)')
    ax2.legend(frameon=False, loc='upper right')
    
    # Bottom Panel: Battery SOC
    ax3.plot(timestamps, mpc_soc, color='#4CAF50', linewidth=2.0, label='MPC SOC')
    ax3.axhline(20.0, color='#E24B4A', linestyle=':', linewidth=1.5, label='20% Critical Floor')
    ax3.set_xlabel('Mission Time (Seconds)')
    ax3.set_ylabel('SOC (%)')
    ax3.set_ylim(0, 100) 
    ax3.legend(frameon=False, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('mpc_testing.png', dpi=150, bbox_inches='tight')
    plt.close() 

    return actual_v, offline_v, mpc_v

if __name__ == "__main__":
    actual, offline, mpc = run_mpc_simulation(r'sasol\mpc_solver\Dinesh2-8_T5.jsonl', r'sasol\mpc_solver\solar_input_03-05-2026.jsonl')