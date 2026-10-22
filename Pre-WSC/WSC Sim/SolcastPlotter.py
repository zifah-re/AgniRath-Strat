"""SolcastPlotter.py - Solar Irradiance Tracker & Background Plotter"""
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import the environment configurations and loaders directly from your optimizer
try:
    from optimizer import (
        load_route, load_solar_fields, field_for_distance,
        RACE_START_DATE, CHARGE_START, CHARGE_END, DRIVE_START, DRIVE_END,
        RACE_FINISH_KM, CONTROL_STOPS_KM, CONTROL_STOP_SECONDS,
        DRIVER_CHANGES_KM, DRIVER_CHANGE_SECONDS, ACST
    )
except ImportError:
    print("[!] Error: Could not import from 'optimizer.py'. Make sure this script is in the same folder and the optimizer is named 'optimizer.py'.")
    exit(1)


def run_car_tracer():
    """
    Drives a virtual car along the route using a 10-segment velocity array.
    Obeys drive windows and control stops, and logs the exact GHI/DNI 
    hitting the car at its specific coordinate and time.
    """
    print("[*] Initializing Route and Solar fields for the tracer...")
    route = load_route()
    fields, race_start_date = load_solar_fields()

    # =========================================================================
    # PLACEHOLDER 10-SEGMENT VELOCITY ARRAY
    # Change these 10 values to hardcode your optimal segment speeds (km/h).
    # The route (3022 km) is automatically divided into 10 equal 302.2 km chunks.
    # =========================================================================
    speeds_kmh = np.array([
        64.1429, 52.0024, 52.2445, 52.9752, 56.5386, 
        52.2393, 55.3030, 56.5571, 53.5341, 67.6225
    ])
    segment_width = RACE_FINISH_KM / len(speeds_kmh)

    distance = 0.0
    now = datetime.combine(race_start_date, CHARGE_START, tzinfo=ACST)
    
    history_t = []
    history_ghi = []
    history_dni = []
    
    pending_delay = 0.0
    dt_sec = 300.0  # 5-minute time steps to perfectly match Solcast resolution
    
    control_stops = list(CONTROL_STOPS_KM.values())
    driver_changes = DRIVER_CHANGES_KM

    print("[*] Running chronological simulation tracer...")
    while distance < RACE_FINISH_KM:
        # 1. Determine current geographic coordinate
        idx = np.searchsorted(route.distance_km, distance)
        if idx == 0:
            coord = route.coords[0]
        elif idx >= len(route.distance_km):
            coord = route.coords[-1]
        else:
            d0, d1 = route.distance_km[idx-1], route.distance_km[idx]
            c0, c1 = route.coords[idx-1], route.coords[idx]
            frac = (distance - d0) / (d1 - d0) if d1 > d0 else 0
            coord = c0 + frac * (c1 - c0)
            
        # 2. Extract Solar Data as the car sees it
        field, weather_ts = field_for_distance(distance, coord, now, fields)
        res = field[(coord, weather_ts.timestamp())]
        ghi_raw, dni_raw = res.data(["ghi", "dni"])
        
        history_t.append(now)
        history_ghi.append(float(np.asarray(ghi_raw).item()) if ghi_raw else 0.0)
        history_dni.append(float(np.asarray(dni_raw).item()) if dni_raw else 0.0)
        
        # 3. Move car if inside the driving window
        is_drive_time = DRIVE_START <= now.time() < DRIVE_END
        if is_drive_time:
            if pending_delay > 0:
                delay_served = min(dt_sec, pending_delay)
                pending_delay -= delay_served
                move_sec = dt_sec - delay_served
            else:
                move_sec = dt_sec
                
            if move_sec > 0:
                # Look up speed for current distance segment
                seg_idx = min(int(distance / segment_width), len(speeds_kmh) - 1)
                speed_ms = (speeds_kmh[seg_idx] * 1000.0) / 3600.0
                
                next_dist = distance + (speed_ms * move_sec) / 1000.0
                
                # Check for control stops and driver swaps
                for stop in control_stops:
                    if distance < stop <= next_dist:
                        next_dist = stop
                        pending_delay += CONTROL_STOP_SECONDS
                        break
                for stop in driver_changes:
                    if distance < stop <= next_dist:
                        next_dist = stop
                        pending_delay += DRIVER_CHANGE_SECONDS
                        break
                        
                distance = next_dist
        
        # 4. Advance clock by 5 minutes
        now += timedelta(seconds=dt_sec)
        
        # 5. Handle overnight jump
        if now.time() >= CHARGE_END:
            # Anchor plot at zero before night
            history_t.append(now)
            history_ghi.append(0.0)
            history_dni.append(0.0)
            
            # Jump to next morning 06:15
            now = datetime.combine(now.date() + timedelta(days=1), CHARGE_START, tzinfo=ACST)
            
            # Anchor plot at zero after night
            history_t.append(now)
            history_ghi.append(0.0)
            history_dni.append(0.0)
            
    print("[*] Tracer finished.")
    return history_t, history_ghi, history_dni


def process_and_plot_solar_data(folder_path):
    # Find ALL day JSON files
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in directory: {folder_path}")
        return

    plt.figure(figsize=(15, 7))

    print("[*] Plotting background raw data...")
    bg_dni_labeled = False
    bg_ghi_labeled = False

    for file_path in json_files:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        for location in data:
            time_series = location.get("data", [])
            if not time_series:
                continue

            df = pd.DataFrame(time_series)
            
            # Convert UTC to ACST to align exactly with the race simulation clock
            df['period_end'] = pd.to_datetime(df['period_end']).dt.tz_convert(ACST)

            # Plot background with severe transparency (alpha=0.1) and solid lines
            lbl_dni = "Background DNI (All Coords)" if not bg_dni_labeled else ""
            lbl_ghi = "Background GHI (All Coords)" if not bg_ghi_labeled else ""
            
            plt.plot(df['period_end'], df['dni'], color='cornflowerblue', alpha=0.08, linestyle='-', label=lbl_dni)
            plt.plot(df['period_end'], df['ghi'], color='sandybrown', alpha=0.08, linestyle='-', label=lbl_ghi)
            
            bg_dni_labeled = True
            bg_ghi_labeled = True

    # Run the tracer to get the car's actual experienced solar
    car_t, car_ghi, car_dni = run_car_tracer()

    print("[*] Plotting car's experienced solar...")
    # Plot the foreground data overlaid boldly
    plt.plot(car_t, car_dni, color='blue', linewidth=2.5, linestyle='-', label="Car Experienced DNI", zorder=10)
    plt.plot(car_t, car_ghi, color='darkorange', linewidth=2.5, linestyle='-', label="Car Experienced GHI", zorder=10)

    # Formatting the plot
    plt.title("Solar Irradiance: Global Forecast vs. Car's Experienced Trajectory")
    plt.xlabel("Race Time (ACST)")
    plt.ylabel("Irradiance (W/m²)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    from pathlib import Path
    
    # Force the script to look in the exact folder where this Python file lives
    HERE = Path(__file__).resolve().parent
    folder_directory = HERE / "Solar" 
    
    process_and_plot_solar_data(str(folder_directory))