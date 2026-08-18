import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# data = [...] 

# Extract hourly wind speeds across all indices
weather_file = r"data\weather\2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 1 Boiketlong to Rustenburg.kml_historical_solar.json"

with open(weather_file, 'r') as f:
     data = json.load(f)

hourly_records = []
for item in data:
    weather = item.get("historical_weather", {})
    hourly = weather.get("hourly", {})
    times = hourly.get("time", [])
    wind_speeds = hourly.get("wind_speed_10m", [])
    
    for t, ws in zip(times, wind_speeds):
        hourly_records.append({"time": t, "wind_speed": ws})

# Convert to DataFrame
df = pd.DataFrame(hourly_records)

# Convert time to datetime and group by hour to find the mean wind speed along the route at each hour
df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.strftime('%H:%M')
mean_wind = df.groupby('hour')['wind_speed'].mean().reset_index()

# Plotting the wind speed across the day
plt.figure(figsize=(12, 6))
plt.plot(mean_wind['hour'], mean_wind['wind_speed'], marker='o', color='teal', linewidth=2, label='Average Wind Speed (km/h)')

plt.title('Average Wind Speed Across the Day (2025-09-10)', fontsize=14, fontweight='bold')
plt.xlabel('Time of Day', fontsize=12)
plt.ylabel('Wind Speed at 10m (km/h)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()