# visualize_day.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from zoneinfo import ZoneInfo

def main():
    DAY_NO = 2
    SA_TZ = ZoneInfo("Africa/Johannesburg") # Defined the timezone variable
    
    # 1. Load the data
    file_path = f"Fallback Model/velocity_profiles/optimized_day_{DAY_NO}.npz"
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Run singleday.py first.")
        return

    speeds = data['speeds_kmh']
    soc = data['soc']
    power = data['power_w']
    start_km = data['start_km']
    
    # Load new time variables
    times = data['times']
    eod_cutoff_ts = float(data['eod_cutoff_ts'])
    final_soc = float(data['final_soc'])

    # 2. Create arrays for plotting (multiplied by 0.01 to match 10m physics resolution)
    distance = start_km + np.arange(len(speeds)) * 0.01
    
    # Build SoC & Time arrays (Append idle charging tail if finished early)
    finish_t = times[-1]
    finish_soc = soc[-1]
    
    if finish_t < eod_cutoff_ts:
        # Generate 50 points to smoothly plot the car sitting in the sun until 17:00
        idle_times = np.linspace(finish_t, eod_cutoff_ts, 50)
        idle_socs = np.linspace(finish_soc, final_soc, 50) 
        plot_times_ts = np.concatenate([times, idle_times])
        plot_socs = np.concatenate([soc, idle_socs])
    else:
        plot_times_ts = times
        plot_socs = soc
        
    # Convert timestamps to matplotlib-compatible datetime objects WITH TIMEZONE
    plot_times = [datetime.fromtimestamp(ts, tz=SA_TZ) for ts in plot_times_ts]

    # 3. Setup the dashboard (Removed sharex=True to decouple Time and Distance)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.canvas.manager.set_window_title('Driver Telemetry Overview')

    # Top: Velocity Profile (vs Distance)
    axes[0].plot(distance, speeds, color='dodgerblue', linewidth=2)
    axes[0].axhline(y=75, color='red', linestyle='--', alpha=0.5, label='Ideal Ceiling (75 km/h)')
    axes[0].axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Floor (50 km/h)')
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].set_title('Optimized Velocity Profile')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Middle: State of Charge (vs Time)
    axes[1].plot(plot_times, plot_socs, color='tomato', linewidth=2)
    
    if finish_t < eod_cutoff_ts:
         # Added timezone to the vertical finish line timestamp
         axes[1].axvline(x=datetime.fromtimestamp(finish_t, tz=SA_TZ), color='gray', linestyle=':', label='Crossed Finish Line / Parked')
         
    axes[1].axhline(y=20, color='black', linestyle='--', alpha=0.7, label='Safety Min (20%)')
    
    # Bound Matplotlib's internal date formatter to the exact timezone
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=SA_TZ))
    axes[1].set_ylabel('State of Charge (%)')
    axes[1].set_xlabel('Time of Day')
    axes[1].set_title('Battery SoC Progression')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Bottom: Power Consumption (vs Distance)
    axes[2].plot(distance, power, color='seagreen', linewidth=2)
    axes[2].axhline(y=0, color='black', linewidth=1)
    axes[2].set_ylabel('Mechanical Power (W)')
    axes[2].set_xlabel('Distance (km)')
    axes[2].set_title('Traction Power (Positive = Motoring, Negative = Regen)')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Manually link ONLY the Top and Bottom graphs to share the Distance x-axis
    axes[0].sharex(axes[2])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()