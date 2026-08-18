import json
import numpy as np
from pathlib import Path
import constants as const
from single_day import SingleDayOptimizer
from multi_day import MultiDayOptimizer
import matplotlib.pyplot as plt
import constants as const

class DataIngestionPipeline:
    def __init__(self):
        self.base_path = Path("data")
        self.weather_path = self.base_path / "weather"
        self.route_path = self.base_path / "route"
        self.trailering_path = self.base_path / "trailering"

    def get_file(self, path_dir, pattern):
        """ Helper to glob a file and return the first match. """
        files = list(path_dir.glob(pattern))
        return files[0] if files else None

    def parse_save_file(self, filepath):
        """ Parses the nested profile dictionary from the .save JSON file. """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        profile = data["profile"]
        
        dist_raw = np.array(profile["Distance"]) * 1000.0 # km to m
        dx = np.diff(dist_raw, prepend=0)
        dx[dx < 0] = 0 
        
        gradient = np.array(profile["Gradient"])
        speed_limit = np.array(profile["SpeedLimit"]) / 3.6 # km/h to m/s
        heading = np.array(profile.get("Headings", np.zeros(len(dx))))
        
        # Safeguard: Force all arrays to perfectly match the length of dx
        def align_array(arr, target_length):
            if len(arr) < target_length:
                return np.pad(arr, (0, target_length - len(arr)), mode='edge')
            return arr[:target_length]
        
        return {
            "dx": dx,
            "gradient": align_array(gradient, len(dx)),
            "speed_limit": align_array(speed_limit, len(dx)),
            "heading": align_array(heading, len(dx))
        }

    def parse_weather(self, filepath):
        """ Parses the .json Meteomatics forecast. """
        with open(filepath, 'r') as f:
            weather_nodes = json.load(f)
            
        ghi_array = []
        wind_speed_array = []
        wind_dir_array = []
        
        # Grab midday averages from the hourly arrays for approx initialization
        for node in weather_nodes:
            hw = node["historical_weather"]["hourly"]
            ghi_array.append(np.mean(hw["shortwave_radiation"][8:16]))
            wind_speed_array.append(np.mean(hw["wind_speed_10m"][8:16]) / 3.6)
            wind_dir_array.append(np.mean(hw["wind_direction_10m"][8:16]))
            
        return np.array(ghi_array), np.array(wind_speed_array), np.array(wind_dir_array)

    def calculate_relative_wind(self, headings, wind_speeds, wind_dirs):
        """ Converts absolute wind to the effective longitudinal headwind/tailwind. """
        relative_angles = np.radians(wind_dirs - headings)
        return wind_speeds * np.cos(relative_angles)

    def downsample_data(self, dx, gradient, caps, v_wind, ghi):
        """ Compresses raw, high-resolution route arrays into solver-friendly chunks. """
        # Fallback to 5000.0 meters if the exact constant name differs in your file
        chunk_target = getattr(const, 'CHUNK_DIST_M', 5000.0) 
        
        new_dx, new_grad, new_caps, new_wind, new_ghi = [], [], [], [], []
        
        temp_dx = 0.0
        temp_grad, temp_caps, temp_wind, temp_ghi = [], [], [], []
        
        for i in range(len(dx)):
            temp_dx += dx[i]
            temp_grad.append(gradient[i])
            temp_caps.append(caps[i])
            temp_wind.append(v_wind[i])
            temp_ghi.append(ghi[i])
            
            # When we hit the chunk distance, or reach the very last element
            if temp_dx >= chunk_target or i == len(dx) - 1:
                if temp_dx > 0:
                    new_dx.append(temp_dx)
                    new_grad.append(np.mean(temp_grad))
                    # We use mean for speed limits so a single 0 km/h stop sign 
                    # doesn't force the solver to drive the entire 5km chunk at 0 km/h
                    new_caps.append(np.mean(temp_caps)) 
                    new_wind.append(np.mean(temp_wind))
                    new_ghi.append(np.mean(temp_ghi))
                
                # Reset for the next chunk
                temp_dx = 0.0
                temp_grad, temp_caps, temp_wind, temp_ghi = [], [], [], []
                
        return np.array(new_dx), np.array(new_grad), np.array(new_caps), np.array(new_wind), np.array(new_ghi)

import matplotlib.pyplot as plt
import constants as const

def plot_day_performance(day_label, dx, opt_v, v_cap, ghi, gradients, v_wind):
    """Generates a minimalist stacked dashboard of the day's performance."""
    
    cumulative_dist_m = np.cumsum(dx)
    dist_km = cumulative_dist_m / 1000.0

    # 1. Convert gradients to radians for the plot's physics calculation!
    theta_rad = np.arctan(gradients / 100.0)
    
    # 2. Reconstruct instantaneous Motor Power correctly
    f_aero = 0.5 * const.AIR_DENSITY * const.CDA_M2 * (opt_v + v_wind)*np.abs(opt_v + v_wind)
    f_roll = const.CRR * const.MASS_KG * const.G_MS2 * np.cos(theta_rad)
    f_grav = const.MASS_KG * const.G_MS2 * np.sin(theta_rad)
    power_motor_w = (f_aero + f_roll + f_grav) * opt_v

    opt_v_kmh = opt_v * 3.6
    v_cap_kmh = v_cap * 3.6

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Optimization Results: {day_label}', fontsize=16, fontweight='bold')
    
    for ax in (ax1, ax2, ax3):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    ax1.plot(dist_km, v_cap_kmh, color='#d3d3d3', linewidth=2, label='Speed Limit (km/h)')
    ax1.plot(dist_km, opt_v_kmh, color='#2c3e50', linewidth=2.5, label='Target Speed (km/h)')
    ax1.set_ylabel('Speed (km/h)', fontsize=11)
    ax1.legend(frameon=False, loc='upper right')

    ax2.plot(dist_km, ghi, color='#f39c12', linewidth=2)
    ax2.fill_between(dist_km, ghi, color='#f39c12', alpha=0.15)
    ax2.set_ylabel('Solar GHI (W/m²)', fontsize=11)

    # 3. Do not multiply gradients by 100 here, they are already percentages
    ax3.fill_between(dist_km, gradients, color='#bdc3c7', alpha=0.4, label='Gradient (%)')
    ax3.set_ylabel('Gradient (%)', color='#7f8c8d', fontsize=11)
    ax3.tick_params(axis='y', labelcolor='#7f8c8d')
    
    ax3_twin = ax3.twinx()
    ax3_twin.spines['top'].set_visible(False)
    ax3_twin.plot(dist_km, power_motor_w / 1000.0, color='#e74c3c', linewidth=2, label='Motor Power (kW)')
    ax3_twin.set_ylabel('Motor Power (kW)', color='#e74c3c', fontsize=11)
    ax3_twin.tick_params(axis='y', labelcolor='#e74c3c')

    ax3.set_xlabel('Cumulative Distance (km)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def main():
    pipeline = DataIngestionPipeline()

    def route_data_provider(day_index, num_loops):
        """
        Dynamically globs and concatenates Stage 1 + Loops + Stage 2 for the requested day.
        """
        day_num = day_index + 1
        print(f"Fetching Day {day_num}, Loops: {num_loops}")
        start_time_s = const.START_TIME_DAY1_S if day_index == 0 else const.START_TIME_OTHER_S
        allowed_time_s = const.FINISH_TIME_S - start_time_s - const.CONTROL_STOP_DURATION_S - (num_loops * const.LOOP_STOP_DURATION_S)

        # 1. Glob the correct files based on naming conventions
        if day_num == 2:
            # Day 2: Half Blind. Hardcoded to match the Swart Ruggens name.
            s1_file = pipeline.get_file(pipeline.route_path, "*Day 2 Half Blind*Stage 1*.kml.save")
            # Assuming Stage 2 exists with a similar name based on what you said
            s2_file = pipeline.get_file(pipeline.route_path, "*Day 2 Half Blind*Stage 2*.kml.save") 
            loop_file = None  # Hardcoded to None so glob doesn't crash trying to find it
            w_file = pipeline.get_file(pipeline.weather_path, "*Day 2*.json")
            
        elif day_num == 3:
            # Day 3: Full Blind / Probables. 
            s1_file = None  # Assuming no Stage 1 for the full blind
            s2_file = None  # Assuming no Stage 2 for the full blind
            loop_file = pipeline.get_file(pipeline.route_path, "*Day 3 probables*Loop*.kml.save")
            w_file = pipeline.get_file(pipeline.weather_path, "*Day 3*.json")
            
        else:
            # Standard naming for all other days
            s1_file = pipeline.get_file(pipeline.route_path, f"*Day {day_num}*Stage 1*.kml.save")
            s2_file = pipeline.get_file(pipeline.route_path, f"*Day {day_num}*Stage 2*.kml.save")
            loop_file = pipeline.get_file(pipeline.route_path, f"*Day {day_num}*Loop*.kml.save")
            w_file = pipeline.get_file(pipeline.weather_path, f"*Day {day_num}*.json")

        dx_all, grad_all, caps_all, head_all = [], [], [], []

        def add_data(f_path):
            if f_path and f_path.exists():
                data = pipeline.parse_save_file(f_path)
                dx_all.extend(data["dx"])
                grad_all.extend(data["gradient"])
                caps_all.extend(data["speed_limit"])
                head_all.extend(data["heading"])
            else:
                # Fallback to prevent crash if file is missing
                dx_all.extend([5000.0, 5000.0])
                grad_all.extend([0.0, 0.0])
                caps_all.extend([const.V_MAX_MS, const.V_MAX_MS])
                head_all.extend([0.0, 0.0])

        # 2. Stitch the route together sequentially
        add_data(s1_file)
        for _ in range(num_loops):
            add_data(loop_file)
        add_data(s2_file)

        dx = np.array(dx_all)
        gradients = np.array(grad_all)
        caps = np.array(caps_all)
        headings = np.array(head_all)

        # 3. Handle the weather array mappings
        if w_file and w_file.exists():
            ghi, ws, wd = pipeline.parse_weather(w_file)
            
            # Align array lengths: slice if weather is longer, edge-pad if route is longer
            if len(ghi) >= len(dx):
                ghi_cut = ghi[:len(dx)]
                ws_cut = ws[:len(dx)]
                wd_cut = wd[:len(dx)]
            else:
                pad_width = (0, len(dx) - len(ghi))
                ghi_cut = np.pad(ghi, pad_width, mode='edge')
                ws_cut = np.pad(ws, pad_width, mode='edge')
                wd_cut = np.pad(wd, pad_width, mode='edge')
        else:
            # Fallback dummy weather if JSON is missing
            ghi_cut = np.full(len(dx), 800.0)
            ws_cut = np.zeros(len(dx))
            wd_cut = np.zeros(len(dx))

        v_wind = pipeline.calculate_relative_wind(headings, ws_cut, wd_cut)
        dx, gradients, caps, v_wind, ghi_cut = pipeline.downsample_data(
            dx, gradients, caps, v_wind, ghi_cut
        )

        return dx, gradients, caps, v_wind, ghi_cut, allowed_time_s


    # =========================================================
    # EXECUTION
    # =========================================================
    
    print("Executing Multi-Day DP Optimizer (This evaluates combinations of Stage 1 + Loops + Stage 2)...")
    multi_day = MultiDayOptimizer(const.DAY_ROUTE_NOTES, route_data_provider)
    optimal_loops, alpha_day = multi_day.optimize_schedule()
    
    # Cap alpha_day display at 100.0% to mask overcharging artifacts
    clamped_alpha = [min(100.0, max(0.0, a)) for a in alpha_day]
    
    print(f"\nOptimal Loop Schedule (Days 1-8): {optimal_loops}")
    print(f"Minimum Starting SoC (alpha) per day: {[round(a, 2) for a in clamped_alpha]}%")

    print("\nExecuting Single-Day SLSQP Optimizer for Day 1 based on optimal loop count...")
    day1_data = route_data_provider(day_index=0, num_loops=optimal_loops[0])
    single_day = SingleDayOptimizer(*day1_data,desc="[Final Pass - Day 1]")
    opt_velocity, total_energy = single_day.run_optimization()
    
    print("SLSQP Completed.")
    print(f"Mean optimal velocity (Day 1): {np.mean(opt_velocity) * 3.6:.2f} km/h")
    print(f"Total energy used: {total_energy:.2f} Joules")
    dx, gradients, caps, v_wind, ghi_cut, _ = day1_data
    
    # Generate the dashboard
    plot_day_performance("Day 1 (Final Pass)", dx, opt_velocity, caps, ghi_cut, gradients, v_wind)


if __name__ == "__main__":
    main()