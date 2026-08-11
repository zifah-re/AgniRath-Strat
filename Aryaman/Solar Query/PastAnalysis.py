import json
import os
import re
import time
from datetime import datetime, timedelta
import requests

# 1. Path & Resume Setup
input_dir = (
    r"C:\Users\aryam\Desktop\Aryaman\Agnirath\AgniRath-Strat\Model\data\processed"
)
solar_dir = os.path.join(input_dir, "solar")
os.makedirs(solar_dir, exist_ok=True)

BASE_DATE = datetime(2025, 9, 10)

# Set your resume checkpoint: Day 7, Stage 2
START_DAY = 7
START_STAGE = 2


def parse_file_info(filename):
    """Extracts Day, Stage, and calculated target date from the filename."""
    day_match = re.search(r"day\s*(\d+)", filename, re.IGNORECASE)
    stage_match = re.search(r"stage\s*(\d+)", filename, re.IGNORECASE)

    if not day_match:
        return None

    day_num = int(day_match.group(1))
    stage_num = int(stage_match.group(1)) if stage_match else 1
    calculated_date = (BASE_DATE + timedelta(days=day_num - 1)).strftime(
        "%Y-%m-%d"
    )

    return {
        "day": day_num,
        "stage": stage_num,
        "date": calculated_date,
    }


def fetch_historical_solar_wind(lat, lon, target_date):
    """Fetches historical weather for a specific date from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": target_date,
        "end_date": target_date,
        "hourly": [
            "shortwave_radiation",
            "direct_normal_irradiance",
            "diffuse_radiation",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        "timezone": "Africa/Johannesburg",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"\nError fetching data for ({lat}, {lon}): {e}")
        return None


# 2. Gather and sort files by (Day, Stage)
kml_files = []
for filename in os.listdir(input_dir):
    if filename.endswith(".kml.save"):
        info = parse_file_info(filename)
        if info:
            kml_files.append((filename, info))

# Sort chronologically by Day then Stage
kml_files.sort(key=lambda x: (x[1]["day"], x[1]["stage"]))

# 3. Process remaining files starting from Day 7, Stage 2
for filename, info in kml_files:
    day, stage, target_date = info["day"], info["stage"], info["date"]

    # Filter out files prior to Day 7 Stage 2
    if (day, stage) < (START_DAY, START_STAGE):
        print(f"Skipping (Already Processed): Day {day} Stage {stage} - {filename}")
        continue

    output_filename = f"{os.path.splitext(filename)[0]}_historical_solar.json"
    output_path = os.path.join(solar_dir, output_filename)

    # Skip if file was fully downloaded previously
    if os.path.exists(output_path):
        print(f"Skipping (Output file exists): {output_filename}")
        continue

    print(f"\nProcessing File: {filename}")
    print(f"Executing: Day {day}, Stage {stage} -> Target Date: {target_date}")

    try:
        with open(
            os.path.join(input_dir, filename), "r", encoding="utf-8"
        ) as f:
            data = json.loads(f.read())

        coordinates = data.get("profile", {}).get("Coordinates", [])

        if not coordinates:
            print(f"Warning: No coordinates found in {filename}. Skipping.")
            continue

        print(f"Found {len(coordinates)} route coordinates.")
        route_results = []

        for idx, (lat, lon) in enumerate(coordinates):
            print(
                f"[{idx + 1}/{len(coordinates)}] Fetching Lat: {lat}, Lon: {lon}...",
                end="\r",
            )

            weather = fetch_historical_solar_wind(lat, lon, target_date)
            if weather:
                route_results.append(
                    {
                        "index": idx,
                        "latitude": lat,
                        "longitude": lon,
                        "target_date": target_date,
                        "historical_weather": weather,
                    }
                )

            time.sleep(0.05)

        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(route_results, out_file, indent=2)

        print(f"\nSuccessfully saved: {output_path}")

    except Exception as e:
        print(f"\nError processing {filename}: {e}")