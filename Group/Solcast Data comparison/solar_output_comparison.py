import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Load Telemetry Data
tel_file_name = input("File name: ")
df = pd.read_json(f"Logs/{tel_file_name}", lines=True, convert_dates=False)

# Convert telemetry time column to datetime (normalize to UTC for safe comparison)
df['_rx_time'] = pd.to_datetime(df['_rx_time'], utc=False)

# Extract date from the first row for the solar file name
date = df['_rx_time'].iloc[0].strftime("%d-%m-%Y")

# 2. Load Solar Input Data
solar_file = f"solar_input_{date}.jsonl"
t_df = pd.read_json(solar_file, lines=True, convert_dates=False)
t_df['period_end'] = pd.to_datetime(t_df['period_end'], utc=False)

# Clean telemetry data
df = df.dropna(subset=["Output_Voltage_A"])

# 3. Capture exact Start and End bounds from df
t_start = df['_rx_time'].iloc[0]
t_end = df['_rx_time'].iloc[-1]

# Slice t_df to fit exactly inside the telemetry session window
t_df_sliced = t_df[(t_df['period_end'] >= t_start) & (t_df['period_end'] <= t_end)]

# Calculate solar output power
df['solar_o'] = (df['Output_Voltage_A'] * df['Output_Current_A'] +
                 df['Output_Voltage_B'] * df['Output_Current_B'] +
                 df['Output_Voltage_C'] * df['Output_Current_C'] +
                 df['Output_Voltage_D'] * df['Output_Current_D'])

# 4. Plotting both on the same window (2 rows, 1 column of subplots)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Top Graph: Telemetry Calculated Output
ax1.plot(df['_rx_time'], df['solar_o'], label='Calculated Solar Output', color='orangered', linewidth=1.5)
ax1.set_title(f'Telemetry Session Data ({date})')
ax1.set_ylabel('Total Output (Watts)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Bottom Graph: Sliced Solar Input (GHI / DNI)
ax2.plot(t_df_sliced['period_end'], t_df_sliced['ghi']*5.95*0.24, label='GHI', color='royalblue')
ax2.plot(t_df_sliced['period_end'], t_df_sliced['dni']*6*0.24, label='DNI', color='forestgreen')
ax2.set_title('Reference Solar Irradiance (Sliced to Match Session)')
ax2.set_ylabel('Irradiance (Watts)')
ax2.set_xlabel('Time (IST)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Adjust layout so labels don't overlap
plt.tight_layout()
plt.show()