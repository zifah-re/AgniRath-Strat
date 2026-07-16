import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURATION & INPUTS ---
LAT = 13.037206951836724
LON = 79.89299347657726
PANEL_TILT = 4    # Tilt angle of panels from flat ground
ALBEDO = 0.2

# 1. Load Telemetry Data
def main(tel_file_name):
    df = pd.read_json(f"Logs/{tel_file_name}", lines=True, convert_dates=False)
    df['_rx_time'] = pd.to_datetime(df['_rx_time'], utc=False)
    df = df.dropna(subset=["Output_Voltage_A"])

    # Calculate actual raw solar output power from the 4 MPPT channels
    df['solar_o'] = (df['Output_Voltage_A'] * df['Output_Current_A'] +
                    df['Output_Voltage_B'] * df['Output_Current_B'] +
                    df['Output_Voltage_C'] * df['Output_Current_C'] +
                    df['Output_Voltage_D'] * df['Output_Current_D'])

    # Extract date for the matching solar input file
    date = df['_rx_time'].iloc[0].strftime("%d-%m-%Y")

    # 2. Load Solar Input Reference Data
    solar_file = f"solar_input_{date}.jsonl"
    t_df = pd.read_json(solar_file, lines=True, convert_dates=False)
    t_df['period_end'] = pd.to_datetime(t_df['period_end'], utc=False)

    # 3. Simulate A and B Coefficients Over the Timeline
    # To account for the car spinning through laps, we average the A coefficient across all compass headings
    solpos = pvlib.solarposition.get_solarposition(t_df['period_end'], LAT, LON)
    zenith_rad = np.radians(solpos['apparent_zenith'])
    tilt_rad = np.radians(PANEL_TILT)

    # Compute constant B
    b_constant = ((1 + np.cos(tilt_rad)) / 2) + ALBEDO * ((1 - np.cos(tilt_rad)) / 2)

    # Average A over 4 cardinal headings to simulate the continuous looping rotation of the laps
    a_variants = []
    for heading in [0, 90, 180, 270]:  # North, East, South, West
        aoi = pvlib.irradiance.aoi(PANEL_TILT, heading, solpos['apparent_zenith'], solpos['azimuth'])
        a_heading = np.cos(np.radians(aoi)) - (np.cos(zenith_rad) * ((1 + np.cos(tilt_rad)) / 2))
        a_variants.append(a_heading)

    a_lap_averaged = np.mean(a_variants, axis=0)

    # Add coefficients directly to the solar reference dataframe
    t_df['A_coeff'] = a_lap_averaged
    t_df['B_coeff'] = b_constant
    # Mask nighttime
    t_df.loc[(solpos['zenith'] >= 90).values, ['A_coeff', 'B_coeff']] = 0

    # Calculate raw instantaneous GTI received on a moving panel plane
    t_df['gti'] = (t_df['A_coeff'] * t_df['dni']) + (t_df['B_coeff'] * t_df['ghi'])
    t_df['gti'] *=5.95*0.24
    # 4. Merge Datasets onto the Telemetry Timeline
    df = df.sort_values('_rx_time')
    t_df = t_df.sort_values('period_end')

    analysis_df = pd.merge_asof(
        df, 
        t_df[['period_end', 'gti']], 
        left_on='_rx_time', 
        right_on='period_end', 
        direction='nearest'
    )

    # Set time as index so we can easily use time-based rolling windows
    analysis_df = analysis_df.set_index('_rx_time')

    # 5. Apply the 5-Minute Rolling Average Smoothing
    # '5min' rolling window ensures lap variations (3 mins) disappear into the trend
    analysis_df['solar_o_smooth'] = analysis_df['solar_o'].rolling('5min', min_periods=1).mean()
    analysis_df['gti_smooth'] = analysis_df['gti'].rolling('5min', min_periods=1).mean()

    # 6. Calculate the Ratio (Smoothed Generated Power / Smoothed Received GTI)
    # This represents your real-world operational efficiency factor over time
    analysis_df['power_to_gti_ratio'] = analysis_df['solar_o_smooth'] / analysis_df['gti_smooth']
    mean_ratio = analysis_df['power_to_gti_ratio'].mean()
    
    return (analysis_df,date,mean_ratio)
    
if __name__ == "__main__":
    tel_file_name = input("File name: ")
    analysis_df,date,mean_ratio=main(tel_file_name)
    print(f"Calculated solar panel efficiency is {0.24*mean_ratio*100:.2f}%")
    # 7. Plotting All Three Metrics Together (3 subplots in 1 window)
    # 7. Plotting All Three Metrics Together (3 subplots in 1 window)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Top Graph: Solar Output (Raw vs Smoothed)
    ax1.plot(analysis_df.index, analysis_df['solar_o'], color='coral', alpha=0.3, label='Raw Telemetry Power')
    ax1.plot(analysis_df.index, analysis_df['solar_o_smooth'], color='orangered', lw=2, label='Smoothed Power (5-min Moving Avg)')
    ax1.set_title(f'Solar Car Telemetry Power Output ({date})')
    ax1.set_ylabel('Watts')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Middle Graph: Calculated GTI (Smoothed)
    ax2.plot(analysis_df.index, analysis_df['gti_smooth'], color='gold', lw=2, label='Lap-Averaged GTI (5-min Moving Avg)')
    ax2.set_title('Estimated Global Tilted Irradiance (GTI) Incident on Panels x 5.95 x 0.24')
    ax2.set_ylabel('Watts')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    # Bottom Graph: The Ratio (System Efficiency Coefficient) with Mean Line
    mean_ratio = analysis_df['power_to_gti_ratio'].mean()  # Calculates the overall session average ratio

    ax3.plot(analysis_df.index, analysis_df['power_to_gti_ratio'], color='mediumpurple', lw=2, label='System Factor Ratio (Power / GTI)')
    ax3.axhline(mean_ratio, color='thistle', linestyle='--', lw=2, label=f'Mean Ratio ({mean_ratio:.3f})')

    ax3.set_title('System Performance Ratio (System Scaling Factor)')
    ax3.set_ylabel('Ratio Index')
    ax3.set_xlabel('Time of Day')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend()
    plt.tight_layout()
    plt.show()