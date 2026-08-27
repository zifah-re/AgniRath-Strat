import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Load the data
    DAY_NO = 2
    
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

    # 2. Create x-axis distance array (1 element per km)
    distance = start_km + np.arange(len(speeds))

    # 3. Setup the dashboard
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.canvas.manager.set_window_title('Driver Telemetry Overview')

    # Top: Velocity Profile
    axes[0].plot(distance, speeds, color='dodgerblue', linewidth=2)
    axes[0].axhline(y=75, color='red', linestyle='--', alpha=0.5, label='Ideal Ceiling (75 km/h)')
    axes[0].axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Floor (60 km/h)')
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].set_title('Optimized Velocity Profile')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Middle: State of Charge (SoC)
    axes[1].plot(distance, soc, color='tomato', linewidth=2)
    axes[1].axhline(y=20, color='black', linestyle='--', alpha=0.7, label='Safety Min (20%)')
    axes[1].set_ylabel('State of Charge (%)')
    axes[1].set_title('Battery SoC Progression')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Bottom: Power Consumption
    axes[2].plot(distance, power, color='seagreen', linewidth=2)
    axes[2].axhline(y=0, color='black', linewidth=1)
    axes[2].set_ylabel('Mechanical Power (W)')
    axes[2].set_xlabel('Distance (km)')
    axes[2].set_title('Traction Power (Positive = Motoring, Negative = Regen)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()