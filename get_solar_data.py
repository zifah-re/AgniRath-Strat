# from solcast import live

# response=live.radiation_and_weather(
#     latitude=-25.682306,
#     longitude=28.130098,
#     output_parameters=['air_temp','dni','ghi','wind_speed_10m','wind_direction_10m'],
#     period="PT5M",
#     terrain_shading=True,
#     )
# df=response.to_pandas()
# print(df)

import argparse
import os
import pandas as pd
from datetime import datetime
from solcast import live, forecast

def valid_datetime(s):
    """Parses a string into a timezone-naive datetime object for uniform local filtering."""
    try:
        dt = datetime.fromisoformat(s.replace(' ', 'T'))
        # Strip timezone info so it's a completely clean, comparable timestamp
        return dt.replace(tzinfo=None)
    except ValueError:
        msg = f"Not a valid datetime: '{s}'. Use ISO format (YYYY-MM-DDTHH:MM:SSZ)."
        raise argparse.ArgumentTypeError(msg)

def main():
    parser = argparse.ArgumentParser(description="Fetch Solcast Live or Forecast weather and radiation data.")
    
    # Required arguments
    parser.add_argument('--mode', choices=['live', 'forecast'], required=True, help="API endpoint mode to query.")
    parser.add_argument('--lat', type=float, required=True, help="Latitude of the location.")
    parser.add_argument('--lon', type=float, required=True, help="Longitude of the location.")
    
    # API Key argument
    parser.add_argument('--api-key', type=str, help="Solcast API Key.")
    
    # Optional timeframe filters
    parser.add_argument('--start', type=valid_datetime, help="Start time (e.g., 2026-09-03T08:39:59)")
    parser.add_argument('--end', type=valid_datetime, help="End time (e.g., 2026-09-03T09:03:24)")
    
    # Configurations
    parser.add_argument('--period', type=str, default="PT5M", help="Time granularity period.")
    parser.add_argument('--terrain', action='store_true', default=True, help="Enable terrain shading.")
    parser.add_argument('--no-terrain', action='store_false', dest='terrain', help="Disable terrain shading.")

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("SOLCAST_API_KEY")
    if not api_key:
        print("Error: Solcast API key missing!")
        return

    params = {
        "latitude": args.lat,
        "longitude": args.lon,
        "output_parameters": ['air_temp', 'dni', 'ghi', 'wind_speed_10m', 'wind_direction_10m'],
        "period": args.period,
        "terrain_shading": args.terrain,
        "api_key": api_key,
    }

    try:
        if args.mode == 'live':
            print(f"Fetching LIVE data...")
            response = live.radiation_and_weather(**params)
        else:
            print(f"Fetching FORECAST data...")
            response = forecast.radiation_and_weather(**params)
        
        df = response.to_pandas()
        
        if df.empty:
            print("No data returned from Solcast API.")
            return

        # Strip the timezone from the dataframe index to allow accurate matching
        df.index = df.index.tz_localize(None)

        # Apply filtering if time window arguments exist
        if args.start or args.end:
            orig_len = len(df)
            if args.start:
                df = df[df.index >= args.start]
            if args.end:
                df = df[df.index <= args.end]
            
            # Catch-all if your window is strictly in the past and returned nothing
            if df.empty:
                print(f"\n⚠️ Warning: Your filter window ({args.start} to {args.end}) returned 0 records.")
                print(f"The API payload currently holds timestamps from {df.index.min()} to {df.index.max()}.")
                print("Showing unfiltered raw payload instead:")
                df = response.to_pandas()
                df.index = df.index.tz_localize(None)

        print(f"\n--- Query Results ({len(df)} rows) ---")
        print(df)
            
    except Exception as e:
        print(f"Error querying Solcast API: {e}")

if __name__ == "__main__":
    main()
