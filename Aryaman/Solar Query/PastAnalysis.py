import os
import re
import json
import math
import requests
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from geopy.distance import geodesic

from solar_table import SolarIrradiance

# --- CONSTANTS & CONFIGURATION ---
CONSTANT_R_MAX_KM = 5.0  # Constant radius (Discretization step = 2 * R_max = 20 km)
DEFAULT_STAGE_HOURS = 8.0
RACE_START_DATE = datetime(2025, 9, 10).date()
STAGE_START_TIME_DAY1 = time(9, 0, 0)
STAGE_START_TIME_OTHER = time(8, 0, 0)
KML_FILENAME_HINT = "2024 Sasol Solar Challenge Route (Publish).kml"
PLOTS_OUTPUT_DIR = "solar_plots_2024"

# Set to True for testing before Aug 27, 2026 (uses 2025 Open-Meteo historical archive)
USE_OPEN_METEO_SANDBOX = True


# --- 1. KML PARSING & ROUTE STITCHING ---
def parse_day_number(day_label: str) -> int:
    """Extracts integer day number from folder labels (e.g., 'Day 1' -> 1, 'Day_02' -> 2)."""
    match = re.search(r'\d+', day_label)
    return int(match.group()) if match else 1


def parse_kml_day_folders(kml_path: str) -> dict:
    """
    Parses a KML file looking for Folder elements matching day names,
    extracting LineString coordinates as (lat, lon) tuples.
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # KML files usually have a namespace
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # Handle both namespaced and non-namespaced KMLs
    def find_all(element, tag):
        res = element.findall(f".//kml:{tag}", ns)
        if not res:
            res = element.findall(f".//{tag}")
        return res

    day_segments = {}

    folders = find_all(root, 'Folder')
    for folder in folders:
        name_elem = folder.find('kml:name', ns) if folder.find('kml:name', ns) is not None else folder.find('name')
        folder_name = name_elem.text.strip() if name_elem is not None and name_elem.text else "Day 1"

        # Look for coordinates inside LineStrings
        coords_list = []
        linestrings = find_all(folder, 'LineString')
        
        for ls in linestrings:
            coord_elem = ls.find('kml:coordinates', ns) if ls.find('kml:coordinates', ns) is not None else ls.find('coordinates')
            if coord_elem is not None and coord_elem.text:
                raw_coords = coord_elem.text.strip().split()
                segment = []
                for c in raw_coords:
                    parts = c.split(',')
                    if len(parts) >= 2:
                        lon, lat = float(parts[0]), float(parts[1])
                        segment.append((lat, lon))
                if segment:
                    coords_list.append(segment)

        if coords_list:
            day_segments[folder_name] = coords_list

    # Fallback if no Folders were defined: look for root LineStrings
    if not day_segments:
        all_coords = []
        for ls in find_all(root, 'LineString'):
            coord_elem = ls.find('kml:coordinates', ns) if ls.find('kml:coordinates', ns) is not None else ls.find('coordinates')
            if coord_elem is not None and coord_elem.text:
                segment = []
                for c in coord_elem.text.strip().split():
                    parts = c.split(',')
                    if len(parts) >= 2:
                        segment.append((float(parts[1]), float(parts[0])))
                if segment:
                    all_coords.append(segment)
        if all_coords:
            day_segments["Day 1"] = all_coords

    return day_segments


def stitch_day_segments(day_route_segments: dict) -> dict:
    """Flattens and stitches multiple line segments in a day into a single continuous list of coordinates."""
    stitched = {}
    for day_label, segments in day_route_segments.items():
        flat_coords = []
        for seg in segments:
            flat_coords.extend(seg)
        if flat_coords:
            stitched[day_label] = flat_coords
    return stitched


# --- 2. HAVERSINE HELPER ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# --- 3. WEATHER FETCHING & MAPPER ---
def _fetch_records(lat, lon):
    if USE_OPEN_METEO_SANDBOX:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2025-09-10",
            "end_date": "2025-09-17",
            "hourly": "shortwave_radiation,direct_normal_irradiance,wind_speed_10m,wind_direction_10m",
            "timezone": "UTC"
        }
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        data = res.json()
        
        records = []
        for i, t_str in enumerate(data["hourly"]["time"]):
            records.append({
                "period_end": f"{t_str}:00Z",
                "ghi": data["hourly"]["shortwave_radiation"][i],
                "dni": data["hourly"]["direct_normal_irradiance"][i],
                "wind_speed_10m": data["hourly"]["wind_speed_10m"][i],
                "wind_direction_10m": data["hourly"]["wind_direction_10m"][i]
            })
        return records
    else:
        pass


def fetch_stage_solar_irradiance(coords, stage_date, stage_start_time, tz):
    records_by_point = {}
    sampled_coords = coords[::max(1, len(coords) // 10)] 
    
    for lat, lon in sampled_coords:
        records = _fetch_records(lat, lon)
        records_by_point[(lat, lon)] = records

    interval = "PT1H" if USE_OPEN_METEO_SANDBOX else "PT30M"
    solar_irr = SolarIrradiance(records_by_point, time_key="period_end", interval=interval)
    return solar_irr, records_by_point


def fetch_local_weather(solar_irr, lat: float, lon: float, arrival_utc: datetime):
    res = solar_irr[(lat, lon), arrival_utc.isoformat()]
    ghi = res.data("ghi")
    dni = res.data("dni")
    wind_spd = res.data("wind_speed_10m")
    wind_dir = res.data("wind_direction_10m")
    return ghi, dni, wind_spd, wind_dir


# --- 4. ROUTE DISCRETIZATION ---
def discretize_route(coords, R_max, velocity_kmh):
    discretized = []
    current_center = coords[0]
    dist_since_last = 0.0
    cumulative_dist = 0.0

    discretized.append({"lat": current_center[0], "lon": current_center[1], "cum_dist": 0.0, "eta_hours": 0.0})

    for i in range(1, len(coords)):
        step_dist = haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        dist_since_last += step_dist
        cumulative_dist += step_dist

        if dist_since_last >= (2 * R_max):
            current_center = coords[i]
            discretized.append({
                "lat": current_center[0],
                "lon": current_center[1],
                "cum_dist": cumulative_dist,
                "eta_hours": cumulative_dist / velocity_kmh if velocity_kmh > 0 else 0.0,
            })
            dist_since_last = 0.0

    return discretized


# --- 5. PLOTTING FUNCTION ---
def plot_solar_irradiance_profile(day_label, stage_date, waypoints_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    distances = [wp["cum_dist_km"] for wp in waypoints_data]
    ghis = [wp["ghi_wm2"] for wp in waypoints_data]
    dnis = [wp["dni_wm2"] for wp in waypoints_data]
    etas = [wp["eta_hours"] for wp in waypoints_data]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    line1 = ax1.plot(distances, ghis, color="#ff7f0e", marker="o", linewidth=2, label="GHI (W/m²)")
    line2 = ax1.plot(distances, dnis, color="#d62728", marker="s", linestyle="--", linewidth=2, label="DNI (W/m²)")
    
    ax1.set_xlabel("Cumulative Distance along Stage (km)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Solar Irradiance (W/m²)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Solar Irradiance Profile — {day_label} ({stage_date.isoformat()})", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    if distances:
        max_dist = max(distances)
        max_eta = max(etas) if etas else 0.0
        tick_locations = ax1.get_xticks()
        ax2.set_xticks(tick_locations)
        ax2.set_xticklabels([f"{(d / max_dist * max_eta):.1f}h" if max_dist > 0 else "0h" for d in tick_locations])
        ax2.set_xlabel("Elapsed Time (Hours)", fontsize=10, color="gray")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)
    
    fig.tight_layout()
    plot_filename = os.path.join(output_dir, f"{day_label}_solar_profile.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)
    print(f"  [+] Solar plot saved to: {plot_filename}")


# --- 6. MAIN EXECUTION ---
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kml_file = os.path.join(script_dir, KML_FILENAME_HINT)

    if not os.path.exists(kml_file):
        print(f"KML file not found: {kml_file}")
        return

    day_route_segments = parse_kml_day_folders(kml_file)
    if not day_route_segments:
        print("No main routes found in KML.")
        return

    stage_groups = stitch_day_segments(day_route_segments)
    day_numbers = {day_label: parse_day_number(day_label) for day_label in stage_groups}
    stage_dates = {
        day_label: RACE_START_DATE + timedelta(days=day_numbers[day_label] - 1)
        for day_label in stage_groups
    }

    print(f"Found {len(stage_groups)} stage(s) with a defined route: {list(stage_groups.keys())}\n")

    sast_tz = ZoneInfo("Africa/Johannesburg")
    json_output_data = {}

    for day_label, coords in stage_groups.items():
        if len(coords) < 2:
            continue

        day_number = day_numbers[day_label]
        stage_date = stage_dates[day_label]

        print(f"Processing stage {day_label} ({stage_date.isoformat()}) using constant R_max = {CONSTANT_R_MAX_KM} km...")

        total_dist = sum(
            haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
            for i in range(1, len(coords))
        )
        velocity = total_dist / DEFAULT_STAGE_HOURS
        stage_start_time = STAGE_START_TIME_DAY1 if day_number == 1 else STAGE_START_TIME_OTHER

        try:
            solar_irr, records_by_point = fetch_stage_solar_irradiance(
                coords, stage_date, stage_start_time, sast_tz
            )
        except Exception as e:
            print(f"  [!] {day_label}: weather data fetch failed ({e}); skipping stage.")
            continue

        r_max_km = CONSTANT_R_MAX_KM
        waypoints = discretize_route(coords, r_max_km, velocity)

        stage_start_sast = datetime.combine(stage_date, stage_start_time, tzinfo=sast_tz)
        stage_waypoints_data = []

        for wp in waypoints:
            arrival_sast = stage_start_sast + timedelta(hours=wp["eta_hours"])
            arrival_utc = arrival_sast.astimezone(ZoneInfo("UTC"))

            ghi, dni, wind_spd, wind_dir = fetch_local_weather(solar_irr, wp["lat"], wp["lon"], arrival_utc)

            stage_waypoints_data.append({
                "cum_dist_km": round(wp["cum_dist"], 2),
                "eta_hours": round(wp["eta_hours"], 2),
                "latitude": round(wp["lat"], 5),
                "longitude": round(wp["lon"], 5),
                "ghi_wm2": round(ghi, 2),
                "dni_wm2": round(dni, 2),
                "wind_speed_ms": round(wind_spd, 2),
                "wind_direction_deg": round(wind_dir, 2)
            })

        plot_solar_irradiance_profile(
            day_label, 
            stage_date, 
            stage_waypoints_data, 
            os.path.join(script_dir, PLOTS_OUTPUT_DIR)
        )

        time_taken = waypoints[-1]["eta_hours"] if waypoints else 0.0

        json_output_data[day_label] = {
            "day_number": day_number,
            "date": stage_date.isoformat(),
            "start_time_sast": stage_start_time.strftime("%H:%M:%S"),
            "distance_covered_km": round(total_dist, 2),
            "time_taken_hours": round(time_taken, 2),
            "r_max_km": r_max_km,
            "r_max_source": "constant",
            "waypoints": stage_waypoints_data
        }

    json_filename = "race_weather_data_2024.json"
    with open(json_filename, "w") as json_file:
        json.dump(json_output_data, json_file, indent=4)

    print(f"\nWeather and route data successfully exported to {json_filename}")
    print(f"All solar profile plots saved in folder: '{PLOTS_OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()