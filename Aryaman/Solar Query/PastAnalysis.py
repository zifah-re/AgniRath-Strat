import json
import os
import re
import time
from datetime import datetime, timedelta
import requests

# 1. Path Setup
input_dir = (
    r"C:\Users\aryam\Desktop\Aryaman\Agnirath\AgniRath-Strat\Model\data\processed"
)
solar_dir = os.path.join(input_dir, "solar")
os.makedirs(solar_dir, exist_ok=True)

# Base date: Day 1 corresponds to September 10, 2025
BASE_DATE = datetime(2025, 9, 10)


def extract_date_from_filename(filename):
    """Parses 'Day X' from filename and calculates date offset from Sept 10, 2025."""
    match = re.search(r"day\s*(\d+)", filename, re.IGNORECASE)
    if match:
        day_num = int(match.group(1))
        # Day 1 -> Sept 10 (+0 days), Day 2 -> Sept 11 (+1 day), etc.
        calculated_date = BASE_DATE + timedelta(days=day_num - 1)
        return calculated_date.strftime("%Y-%m-%d"), day_num
    return None, None


def fetch_historical_solar_wind(lat, lon, target_date):
    """Fetches historical weather for a specific single date from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": target_date,
        "end_date": target_date,
        "hourly": [
            "shortwave_radiation",  # GHI (W/m²)
            "direct_normal_irradiance",  # DNI (W/m²)
            "diffuse_radiation",  # DHI (W/m²)
            "wind_speed_10m",  # Wind Speed (km/h)
            "wind_direction_10m",  # Wind Direction (degrees)
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


# 2. Process all route files in directory
for filename in os.listdir(input_dir):
    if filename.endswith(".kml.save"):
        target_date, day_number = extract_date_from_filename(filename)

        if not target_date:
            print(
                f"Skipping {filename}: Unable to parse 'Day X' from filename."
            )
            continue

        file_path = os.path.join(input_dir, filename)
        output_filename = f"{os.path.splitext(filename)[0]}_historical_solar.json"
        output_path = os.path.join(solar_dir, output_filename)

        print(f"\nProcessing File: {filename}")
        print(f"Identified: Day {day_number} -> Date: {target_date}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())

            coordinates = data.get("profile", {}).get("Coordinates", [])

            if not coordinates:
                print(f"Warning: No coordinates found in {filename}. Skipping.")
                continue

            print(f"Found {len(coordinates)} route coordinates.")
            route_results = []

            # Loop through coordinates
            for idx, (lat, lon) in enumerate(coordinates[::10]):
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

            # Save the resulting weather payload
            with open(output_path, "w", encoding="utf-8") as out_file:
                json.dump(route_results, out_file, indent=2)

            print(f"\nSuccessfully saved to: {output_path}")

        except Exception as e:
            print(f"\nError processing {filename}: {e}")