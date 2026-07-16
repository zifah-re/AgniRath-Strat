# Track_Gradient_fixed.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re

# ── 1. LOAD & CLEAN ──────────────────────────────────────────────────────────
df = pd.read_csv("AllRuns_cleaned.csv")

drop_columns = [
    c for c in df.columns
    if re.search(
        r'^CMU|Timestamp|Flag\d+|.*Current.*|.*Voltage.*'
        r'|^Controller|^Mosfet|^Cabin|^Cell|Temp$|itude$'
        r'|^acc|Speed|error|Brake_Status',
        c
    )
]
df = df.drop(columns=drop_columns)
df['_time_sec'] = pd.to_numeric(df['_time_sec'], errors='coerce')
df = df.dropna(subset=['_time_sec']).reset_index(drop=True)

# ── 2. MASK CAR-OFF SECTIONS ─────────────────────────────────────────────────

dt    = df['_time_sec'].diff().fillna(0.0).clip(lower=0).values

jerk = df['Acceleration'].diff().fillna(0.0).values

JERK_THRESH = 1e-3   
MIN_INTERP_LEN = 30     

jerk_zero = np.abs(jerk) < JERK_THRESH

car_off = np.zeros(len(df), dtype=bool)
i = 0
while i < len(jerk_zero):
    if jerk_zero[i]:
        j = i
        while j < len(jerk_zero) and jerk_zero[j]:
            j += 1
        if (j - i) >= MIN_INTERP_LEN:
            car_off[i:j] = True
        i = j
    else:
        i += 1

df['Car_Off'] = car_off
n_off = car_off.sum()
print(f"Car-off rows masked: {n_off} ({100*n_off/len(df):.1f}%)")

# ── 3. GRADIENT via P(v) baseline ────────────────────────────────────────────
eta_motor = 0.95
m   = 310
g   = 9.81
k   = 0.098
crr = 0.007

v = df['Vehicle_Velocity'] / 3.6          
a = df['Acceleration']     / 3.6          
P_wheel = (df['Output_Power']-50) * eta_motor    

P_excess = P_wheel - m*a*v - k*v**3
P_excess[df['Car_Off']] = np.nan

V_BIN_WIDTH = 0.5   
v_edges = np.arange(0, v.max() + V_BIN_WIDTH, V_BIN_WIDTH)
v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

baseline = np.full(len(v_edges)-1, np.nan)
for b in range(len(baseline)):
    in_bin = (~df['Car_Off']) & (v >= v_edges[b]) & (v < v_edges[b + 1])
    if in_bin.sum()>5:
        baseline[b] = np.nanmedian(P_excess[in_bin])

from scipy.interpolate import interp1d
valid_bins  = np.isfinite(baseline)
P_flat_interp = interp1d(
    v_centers[valid_bins], baseline[valid_bins],
    kind='linear', bounds_error=False,
    fill_value=(baseline[valid_bins][0], baseline[valid_bins][-1])
)
P_flat = pd.Series(P_flat_interp(v.values), index=df.index)

fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(v[~df['Car_Off']], P_excess[~df['Car_Off']],
           s=0.5, alpha=0.1, color='steelblue', label='P_excess samples')
ax.plot(v_centers[valid_bins], baseline[valid_bins],
        color='crimson', lw=2, label='Median baseline P_flat(v)')
ax.set_xlabel('Velocity (m/s)'); ax.set_ylabel('P_excess (W)')
ax.set_title('P_excess vs velocity — baseline extraction')
ax.legend(); plt.tight_layout(); plt.show()

mg = m*g
R = P_excess - P_flat
r = R / (mg * v)                   
Q = r + crr                        

A_q  =  1 + crr**2
B_q  = -2 * Q
C_q  =  Q**2 - crr**2

disc = B_q**2 - 4 * A_q * C_q

sin_theta = np.full(len(df), np.nan)
ok = np.isfinite(disc.values) & (disc.values >= 0) & (~df['Car_Off'].values) & (v.values > 0.5)

s1 = (-B_q[ok] + np.sqrt(disc[ok])) / (2 * A_q)
s2 = (-B_q[ok] - np.sqrt(disc[ok])) / (2 * A_q)

sin_theta[ok] = np.where(np.abs(s1) <= np.abs(s2), s1, s2)
sin_theta     = np.clip(sin_theta, -1.0, 1.0)

df['Gradient'] = np.arcsin(sin_theta)   # radians

# Smooth
SMOOTH_WIN = 11
df['Gradient_smooth'] = (
    df['Gradient']
    .rolling(SMOOTH_WIN, center=True, min_periods=1)
    .median()
)

# ── 4. LAP SEGMENTATION ───────────────────────────────────────────────────────

v_active = df['Vehicle_Velocity'].copy()
v_active[car_off] = 0.0   

median_dt = np.median(dt[dt > 0])
fs = 1.0 / median_dt                     
print(f"Sampling rate: {fs:.2f} Hz")

MIN_LAP_TIME_S  = 60     
MIN_LAP_SAMPLES = int(MIN_LAP_TIME_S * fs)

VALLEY_PROMINENCE_KMH = 15   

valleys, _ = find_peaks(
    -v_active.values,
    distance   = MIN_LAP_SAMPLES,
    prominence = VALLEY_PROMINENCE_KMH,
)
print(f"Valleys (lap boundaries) found: {len(valleys)}")

lap_boundaries = [0] + [int(v) for v in valleys] + [len(df) - 1]
lap_boundaries = sorted(set(lap_boundaries))

df['Lap'] = 0
for lap_num, (start, end) in enumerate(
        zip(lap_boundaries[:-1], lap_boundaries[1:]), start=1):
    df.iloc[start:end, df.columns.get_loc('Lap')] = lap_num

num_laps = df['Lap'].max()
print(f"Laps segmented: {num_laps}")

# ── 5. GRADIENT vs DISTANCE PER LAP → AVERAGED ACROSS LAPS ──────────────────

dt_series = df['_time_sec'].diff().fillna(0.0).clip(lower=0)

lap_distances = []
for lap_num in range(1, num_laps + 1):
    mask = (df['Lap'] == lap_num) & (~df['Car_Off'])
    if mask.sum() < 10:
        continue
    ds = (v[mask] * dt_series[mask]).sum()
    lap_distances.append(ds)

median_lap_dist = np.median(lap_distances)
N_BINS = 200    
bin_size = median_lap_dist / N_BINS
bin_edges = np.arange(0, median_lap_dist + bin_size, bin_size)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

print(f"Median lap distance: {median_lap_dist:.1f} m  |  bin size: {bin_size:.2f} m")

all_lap_profiles = []    

for lap_num in range(1, num_laps + 1):
    mask = (df['Lap'] == lap_num) & (~df['Car_Off'])
    sub  = df.loc[mask]
    if len(sub) < 10:
        continue

    v_lap    = v.loc[mask].values
    grad_lap = df['Gradient_smooth'].loc[mask].values
    dt_lap   = dt_series.loc[mask].values

    cum_dist = np.cumsum(v_lap * dt_lap)   # distance within lap
    valid = np.isfinite(grad_lap) & (cum_dist <= median_lap_dist * 1.15)

    if valid.sum() < 10:
        continue

    bin_means = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        in_bin = valid & (cum_dist >= bin_edges[b]) & (cum_dist < bin_edges[b+1])
        if in_bin.sum() > 0:
            bin_means[b] = np.nanmean(grad_lap[in_bin])

    all_lap_profiles.append(bin_means)

profile_matrix = np.vstack(all_lap_profiles)          
avg_gradient = np.nanmean(profile_matrix, axis=0)   
std_gradient = np.nanstd(profile_matrix,  axis=0)

# ── 6. PLOTS ──────────────────────────────────────────────────────────────────
t_rel = df['_time_sec'] - df['_time_sec'].iloc[0]

fig, ax = plt.subplots(figsize=(25, 6))
ax.plot(t_rel, v, lw=0.7, color='steelblue', label='Velocity (m/s)')
for i, b in enumerate(lap_boundaries[1:-1]):
    ax.axvline(t_rel.iloc[b], color='tomato', lw=0.8, alpha=0.7,
               label='Lap boundary' if i == 0 else '')

in_off = False
for idx in range(len(df)):
    if car_off[idx] and not in_off:
        x0 = t_rel.iloc[idx]; in_off = True
    elif not car_off[idx] and in_off:
        ax.axvspan(x0, t_rel.iloc[idx], color='gray', alpha=0.25,
                   label='Car off' if 'Car off' not in [l.get_label() for l in ax.lines] else '')
        in_off = False
ax.set_xlabel('Time (s)'); ax.set_ylabel('Velocity (m/s)')
ax.set_title('Velocity with lap boundaries and car-off regions')
ax.legend(); plt.tight_layout(); plt.show()

overall_avg_deg = float(np.nanmean(np.degrees(avg_gradient)))

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(bin_centers, np.degrees(avg_gradient), color='darkorange', lw=1.5,
        label='Mean gradient across laps')
ax.fill_between(
    bin_centers,
    np.degrees(avg_gradient - std_gradient),
    np.degrees(avg_gradient + std_gradient),
    alpha=0.25, color='darkorange', label='±1σ'
)
ax.axhline(0, color='k', lw=0.6, ls='--')
ax.axhline(overall_avg_deg, color='crimson', lw=1.0, ls=':',
           label=f'Lap avg = {overall_avg_deg:.3f}°')
ax.text(
    bin_centers[-1], overall_avg_deg,
    f'  avg = {overall_avg_deg:.3f}°',
    color='crimson', va='bottom', ha='right', fontsize=10
)
ax.set_xlabel('Distance along lap (m)')
ax.set_ylabel('Gradient (°)')
ax.set_title('Track gradient vs distance — averaged across all laps')
ax.legend(); plt.tight_layout(); plt.show()