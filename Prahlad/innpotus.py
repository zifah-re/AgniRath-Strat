import numpy as np
import pandas as pd
import pvlib
from pvlib import solarposition, irradiance

def calculate_efficiency_from_location(
    latitude: float,
    longitude: float,
    datetime_str: str,
    dni: float,
    ghi: float,
    measured_power_watts: float,
    area: float = 6.0,
    altitude: float = 0.0,
    albedo: float = 0.2,
    model: str = 'isotropic'
) -> dict:
    """
    Calculates solar position (zenith & azimuth), total incident power, and 
    system/panel conversion efficiency for a panel perpendicular to the sun.

    Parameters:
    -----------
    latitude : float
        Latitude in decimal degrees (e.g., -34.9285 for Adelaide).
    longitude : float
        Longitude in decimal degrees (e.g., 138.6007 for Adelaide).
    datetime_str : str
        ISO 8601 formatted datetime string with ACST timezone offset 
        (e.g., '2026-03-21 17:00:00+09:30').
    dni : float
        Direct Normal Irradiance in W/m^2.
    ghi : float
        Global Horizontal Irradiance in W/m^2.
    measured_power_watts : float
        Actual measured electrical power output from the panel in Watts.
    area : float
        Surface area of the panel in m^2 (default 6.0 m^2).
    altitude : float
        Elevation above sea level in meters.
    albedo : float
        Ground reflectance coefficient (default 0.2).
    model : str
        Diffuse sky transpose model ('isotropic', 'haydavies', 'perez', etc.).

    Returns:
    --------
    dict containing solar position, POA components, incident power, and efficiency.
    """
    # Step 1: Parse timezone-aware datetime
    time = pd.Timestamp(datetime_str)

    # Step 2: Calculate exact solar position using pvlib
    solpos = solarposition.get_solarposition(
        time=time, 
        latitude=latitude, 
        longitude=longitude, 
        altitude=altitude
    )
    
    solar_zenith = float(solpos['zenith'].iloc[0])
    solar_azimuth = float(solpos['azimuth'].iloc[0])

    # If the sun is below the horizon, power generation is 0
    if solar_zenith >= 90.0:
        return {
            'Solar_Zenith_deg': solar_zenith,
            'Solar_Azimuth_deg': solar_azimuth,
            'Status': 'Sun is below the horizon (Nighttime)'
        }

    # Step 3: Align panel tilt and azimuth to point directly at the sun
    surface_tilt = solar_zenith
    surface_azimuth = solar_azimuth

    # Step 4: Calculate DHI without approximation: GHI = DNI * cos(zenith) + DHI
    zenith_rad = np.radians(solar_zenith)
    cos_zenith = np.cos(zenith_rad)
    dhi = max(0.0, ghi - dni * cos_zenith) if cos_zenith > 0 else ghi

    # Step 5: POA Irradiance components
    poa_components = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        albedo=albedo,
        model=model
    )

    poa_global = float(poa_components['poa_global'])

    # Step 6: Incident Solar Power hitting the 6 m^2 surface
    incident_power_watts = poa_global * area

    # Step 7: Calculate Module/System Efficiency
    efficiency_percent = (measured_power_watts / incident_power_watts) * 100.0

    return {
        'Timestamp': str(time),
        'Solar_Zenith_deg': solar_zenith,
        'Solar_Azimuth_deg': solar_azimuth,
        'Surface_Tilt_deg': surface_tilt,
        'Surface_Azimuth_deg': surface_azimuth,
        'DHI_W_m2': dhi,
        'POA_Global_Irradiance_W_m2': poa_global,
        'Total_Incident_Solar_Power_W': incident_power_watts,
        'Measured_Electrical_Power_W': measured_power_watts,
        'Calculated_Efficiency_Percent': efficiency_percent
    }


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Input Location: Adelaide, Australia (ACST timezone offset: +09:30)
    lat=-23.609869354457405
    lon= 133.8805538220748
    
    # Time: 5:00 PM ACST (+09:30 offset)
    time_acst = "2025-08-25 17:00:00+09:30"
    
    # Measured inputs
    dni_input = 739.0            # W/m^2
    ghi_input = 271.0            # W/m^2
    measured_watts = 1000.0       # Example measured power output in Watts

    results = calculate_efficiency_from_location(
        latitude=lat,
        longitude=lon,
        datetime_str=time_acst,
        dni=dni_input,
        ghi=ghi_input,
        measured_power_watts=measured_watts,
        area=6.0
    )

    print("--- Calculated Results ---")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: {val}")