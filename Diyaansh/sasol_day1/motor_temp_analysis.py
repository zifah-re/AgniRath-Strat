import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------
# 1. Hardcoded File Mapping
# ---------------------------------------------------------
file_mapping = {
    "Stage 1": [
        "day1stage1firstfile.jsonl",
        "day1stage1file2beforebalastopping.jsonl", 
        "day1stage1thirdfile.jsonl",
        "day1stage1thirdfile2.jsonl",
        "day1stage1fourthfile.jsonl",
        "day1stage1fifthfile.jsonl",
        "day1stage1sixthfile.jsonl"
    ],
    "Loop": [
        "day1loop.jsonl"
    ],
    "Stage 2": [
        "day1stage2.jsonl"
    ]
}

TEMP_KEY = 'HeatSink_Temp' 

# ---------------------------------------------------------
# 2. Load and Tag Data
# ---------------------------------------------------------
raw_data = []

for stage_name, files in file_mapping.items():
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        
                        # Parse the time immediately
                        dt = datetime.fromisoformat(row['_rx_time'])
                        
                        # Apply your manual shifts ONLY to thirdfile2
                        if "thirdfile2" in filepath:
                            if dt.hour == 8:
                                dt += timedelta(hours=2)
                            elif dt.hour == 14:
                                dt -= timedelta(hours=3, minutes=30)
                        
                        # Strip the timezone off EVERYTHING so the sort function is forced to use face-value time
                        dt = dt.replace(tzinfo=None)
                        row['_rx_time'] = dt.isoformat()
                        
                        row['stage_label'] = stage_name
                        raw_data.append(row)
        except FileNotFoundError:
            print(f"Warning: Could not find {filepath}. Skipping.")

# ---------------------------------------------------------
# 3. Sort Chronologically
# ---------------------------------------------------------
raw_data.sort(key=lambda x: datetime.fromisoformat(x['_rx_time']))

# ---------------------------------------------------------
# 4. Process Data & Forward-Fill Missing Keys
# ---------------------------------------------------------
current_temp = None
current_velocity = 0.0
current_power = None

times = []
temps = []
distances = []
power_spikes_count = []
stage_labels_tracked = []

current_distance = 0.0
cumulative_spikes = 0
prev_time = None
prev_power = None

# Threshold: any gap larger than 10 seconds is considered a stop/blackout
MAX_GAP_SECONDS = 10.0 

for row in raw_data:
    if TEMP_KEY in row:
        current_temp = row[TEMP_KEY]
    if 'Vehicle_Velocity' in row:
        current_velocity = row['Vehicle_Velocity']
    if 'Bus_Voltage' in row and 'Bus_Current' in row:
        current_power = row['Bus_Voltage'] * row['Bus_Current']

    if current_temp is None or current_power is None:
        continue
    current_time = datetime.fromisoformat(row['_rx_time'])

    if prev_time is not None:
        dt_seconds = (current_time - prev_time).total_seconds()
        
        # Only integrate distance and spikes if the telemetry gap is normal
        if 0 < dt_seconds < MAX_GAP_SECONDS:
            # Divide by 1000 to convert meters to kilometers
            current_distance += (current_velocity * dt_seconds) / 1000.0
            
            if prev_power is not None and abs(current_power - prev_power) > 1000:
                cumulative_spikes += 1

    times.append(current_time)
    temps.append(current_temp)
    distances.append(current_distance)
    power_spikes_count.append(cumulative_spikes)
    stage_labels_tracked.append(row['stage_label'])
    
    prev_time = current_time
    prev_power = current_power

# ---------------------------------------------------------
# 5. Calculate Stage Boundaries for Shading
# ---------------------------------------------------------
bounds = {}
for t, d, label in zip(times, distances, stage_labels_tracked):
    if label not in bounds:
        bounds[label] = {'t_min': t, 't_max': t, 'd_min': d, 'd_max': d}
    else:
        bounds[label]['t_max'] = t
        bounds[label]['d_max'] = d

# ---------------------------------------------------------
# 6. Minimalist Plotting (Original 3 Subplots)
# ---------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

# Restored color scheme
stage_colors = {'Stage 1': 'lightgrey', 'Loop': 'lightblue', 'Stage 2': 'lightgreen'}

def style_ax(ax, title, ylabel, xlabel):
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

# Plot A: Temps vs Time
ax1.plot(times, temps, color='#E63946', linewidth=1.5)
style_ax(ax1, f'{TEMP_KEY} vs. Time', 'Temp (°C)', 'Time')

# Plot B: Temps vs Distance
ax2.plot(distances, temps, color='#1D3557', linewidth=1.5)
style_ax(ax2, f'{TEMP_KEY} vs. Distance', 'Temp (°C)', 'Distance (km)')

# Plot C: Temps vs Power Spikes (Scatter)
ax3.scatter(power_spikes_count, temps, alpha=0.6, color='#F4A261', edgecolors='none', s=15)
style_ax(ax3, f'{TEMP_KEY} vs. Cumulative Power Spikes (>1kW)', 'Temp (°C)', 'Number of >1kW Power Changes')

# Apply background shading
for label, span in bounds.items():
    color = stage_colors.get(label, 'white')
    ax1.axvspan(span['t_min'], span['t_max'], color=color, alpha=0.3, label=label, zorder=0)
    ax2.axvspan(span['d_min'], span['d_max'], color=color, alpha=0.3, zorder=0)

ax1.legend(loc='upper left', frameon=False, fontsize=10)


# ---------------------------------------------------------
# 7. Extra Section: Motor Temperature vs. Route Gradient (Fixed Alpha=0.9)
# ---------------------------------------------------------
route_save_files = {
    "Stage 1": "2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 1 Boiketlong to Rustenburg.kml.save",
    "Loop": "your_loop_save_filename_here.save",       # Replace with actual Loop .save filename
    "Stage 2": "your_stage2_save_filename_here.save"   # Replace with actual Stage 2 .save filename
}

all_gradients = []
all_temps = []

for stage_name, filename in route_save_files.items():
    try:
        with open(filename, 'r') as f:
            route_save_data = json.load(f)
        
        route_distances = np.array(route_save_data['profile']['Distance'])
        route_gradients = np.array(route_save_data['profile']['Gradient'])
        
        stage_indices = [i for i, label in enumerate(stage_labels_tracked) if label == stage_name]
        stage_temps = [temps[i] for i in stage_indices]
        
        if stage_temps:
            n_st = len(stage_temps)
            for idx, t_val in zip(range(n_st), stage_temps):
                progress = idx / max(1, n_st - 1)
                target_dist = progress * route_distances[-1]
                grad = np.interp(target_dist, route_distances, route_gradients)
                all_gradients.append(grad)
                all_temps.append(t_val)
                
    except FileNotFoundError:
        pass

if all_gradients and all_temps:
    x = np.array(all_gradients)
    y = np.array(all_temps)

    fig_grad, ax_grad = plt.subplots(figsize=(10, 5), constrained_layout=True)
    
    # Scatter plot with uniform color and fixed alpha=0.9
    ax_grad.scatter(x, y, color='#1D3557', alpha=0.9, edgecolors='none', s=15)
    style_ax(ax_grad, f'{TEMP_KEY} vs. Route Gradient', 'Temp (°C)', 'Gradient (%)')

plt.show()