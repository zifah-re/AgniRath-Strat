"""
core/solar.py — solar pipeline behind ONE frozen signature.

FROZEN INTERFACE (contract, README §Interfaces):
    provider.ghi_wm2(t_s: float, x_m: float) -> float
        t_s : seconds since local midnight of the race day
        x_m : distance along the day's route (metres, from day start line)

Providers:
      GaussianProvider          seniors' fallback day-shape.
      PVLibClearSkyProvider     pvlib clear-sky fallback.
      HourlyJSONSolarProvider   Primary provider using cubic spline interpolation 
                                over hourly JSON weather data.
"""

from __future__ import annotations

import json
import math
import pathlib
import typing as _t
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# PMF correction
# ---------------------------------------------------------------------------
PMF_TABLE = {
    ("Ia", 0.0, 2.0, "offpeak"): 0.9393,    
    ("Ib", 0.0, 2.0, "midday"): 0.9532,     
    ("IIa", 3.0, 10.0, "allday"): 1.1013,
    ("IIb", 11.0, 20.0, "allday"): 0.9965,
    ("IIc", 21.0, 40.0, "allday"): 0.9226,
    ("IId", 41.0, 100.0, "allday"): 1.1019,
}

def pmf_correction_factor(tcc_daily_mean_pct: float, t_s: float) -> float:
    tcc = float(tcc_daily_mean_pct)
    hour = (t_s / 3600.0)
    if tcc <= 2.0:
        return PMF_TABLE[("Ib", 0.0, 2.0, "midday")] if 10.0 <= hour < 14.0 \
            else PMF_TABLE[("Ia", 0.0, 2.0, "offpeak")]
    if tcc <= 10.0:
        return PMF_TABLE[("IIa", 3.0, 10.0, "allday")]
    if tcc <= 20.0:
        return PMF_TABLE[("IIb", 11.0, 20.0, "allday")]
    if tcc <= 40.0:
        return PMF_TABLE[("IIc", 21.0, 40.0, "allday")]
    return PMF_TABLE[("IId", 41.0, 100.0, "allday")]

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class SolarProvider:
    def ghi_wm2(self, t_s: float, x_m: float) -> float:  # pragma: no cover
        raise NotImplementedError

class GaussianProvider(SolarProvider):
    PEAK_WM2 = 1073.099
    NOON_S = 43_200.0
    SIGMA_S = 11_600.0

    def ghi_wm2(self, t_s: float, x_m: float = 0.0) -> float:
        return float(self.PEAK_WM2
                     * math.exp(-0.5 * ((t_s - self.NOON_S) / self.SIGMA_S) ** 2))

class PVLibClearSkyProvider(SolarProvider):
    def __init__(self, lat: float, lon: float, date_iso: str,
                 tz: str = "Africa/Johannesburg"):
        import pandas as pd
        import pvlib  
        self._pvlib = pvlib
        self._times = pd.date_range(f"{date_iso} 00:00", f"{date_iso} 23:59",
                                    freq="1min", tz=tz)
        loc = pvlib.location.Location(lat, lon, tz=tz)
        self._ghi = loc.get_clearsky(self._times, model="ineichen")["ghi"].values

    def ghi_wm2(self, t_s: float, x_m: float = 0.0) -> float:
        idx = int(np.clip(t_s // 60, 0, len(self._ghi) - 1))
        return float(self._ghi[idx])

class HourlyJSONSolarProvider(SolarProvider):
    """
    Consumes a point-by-point JSON containing hourly weather arrays.
    Maps the solver's x_m distance request to a lat/lon coordinate, then 
    uses a KDTree to find the nearest weather node.
    """
    def __init__(self, json_paths: list[str] | str, route):
        # Convert to list if a single string is passed
        if isinstance(json_paths, str):
            json_paths = [json_paths]
            
        self.route = route
        self.t_s_array = np.arange(24) * 3600.0  
        
        coords = []
        self.spline_models = []
        
        # Loop through EVERY file for this day and extract the nodes
        for jp in json_paths:
            with open(jp, 'r', encoding="utf-8") as f:
                data = json.load(f)
                
            for node in data:
                coords.append([node["latitude"], node["longitude"]])
                ghi_array = np.array(node["historical_weather"]["hourly"]["shortwave_radiation"], dtype=float)
                spline_fit = CubicSpline(self.t_s_array, ghi_array, bc_type='natural')
                self.spline_models.append(spline_fit)
            
        self.tree = cKDTree(np.array(coords))

    def ghi_wm2(self, t_s: float, x_m: float = 0.0) -> float:
        if self.route is None:
            idx = len(self.spline_models) // 2
        else:
            lat, lon = self.route.latlon_at(x_m)
            _, idx = self.tree.query([lat, lon])
            
        ghi_fitted = float(self.spline_models[idx](t_s))
        return max(0.0, ghi_fitted)

def best_available_provider(**pvlib_kwargs) -> SolarProvider:
    try:
        return PVLibClearSkyProvider(**pvlib_kwargs)
    except Exception:
        return GaussianProvider()

# ---------------------------------------------------------------------------
# Incidence geometry
# ---------------------------------------------------------------------------

def solar_declination_deg(day_of_year: int) -> float:
    return 23.45 * math.sin(math.radians(360.0 * (284 + day_of_year) / 365.0))

def equation_of_time_min(day_of_year: int) -> float:
    b = math.radians((day_of_year - 1) * 360.0 / 365.0)
    return 229.2 * (0.000075 + 0.001868 * math.cos(b) - 0.032077 * math.sin(b)
                    - 0.014615 * math.cos(2 * b) - 0.04089 * math.sin(2 * b))

def hour_angle_deg(t_solar_s: float) -> float:
    return (t_solar_s / 3600.0 - 12.0) * 15.0

def solar_time_s(t_clock_s: float, lon_deg: float, day_of_year: int,
                 tz_hours: float = 2.0) -> float:
    lst = 15.0 * tz_hours
    corr_min = 4.0 * (lon_deg - lst) + equation_of_time_min(day_of_year)
    return t_clock_s + corr_min * 60.0

def cos_incidence(
    lat_deg: float, decl_deg: float, omega_deg: float,
    tilt_deg: float, azimuth_deg: float,
) -> float:
    phi = math.radians(lat_deg)
    d = math.radians(decl_deg)
    w = math.radians(omega_deg)
    b = math.radians(tilt_deg)
    g = math.radians(azimuth_deg)
    return (
        math.sin(phi) * math.sin(d) * math.cos(b)
        - math.cos(phi) * math.sin(d) * math.sin(b) * math.cos(g)
        + math.cos(phi) * math.cos(d) * math.cos(w) * math.cos(b)
        + math.sin(phi) * math.cos(d) * math.cos(w) * math.sin(b) * math.cos(g)
        + math.cos(d) * math.sin(w) * math.sin(b) * math.sin(g)
    )

def cos_zenith(lat_deg: float, decl_deg: float, omega_deg: float) -> float:
    phi = math.radians(lat_deg)
    d = math.radians(decl_deg)
    w = math.radians(omega_deg)
    return math.cos(phi) * math.cos(d) * math.cos(w) + math.sin(phi) * math.sin(d)

def slope_geometry_factor(
    lat_deg: float, day_of_year: int, t_solar_s: float,
    road_slope_pct: float, route_bearing_deg: float,
    panel_tilt_base_deg: float = 4.0,
) -> float:
    decl = solar_declination_deg(day_of_year)
    omega = hour_angle_deg(t_solar_s)
    cz = cos_zenith(lat_deg, decl, omega)
    if cz <= 0.0:
        return 0.0                       
    road_pitch_deg = math.degrees(math.atan(road_slope_pct / 100.0))
    tilt = panel_tilt_base_deg + road_pitch_deg
    sign = 1.0 if tilt >= 0 else -1.0
    tilt_abs = abs(tilt)
    gamma = -route_bearing_deg * sign
    gamma = ((gamma + 180.0) % 360.0) - 180.0
    ci = cos_incidence(lat_deg, decl, omega, tilt_abs, gamma)
    return float(np.clip(ci / cz, 0.0, 2.0))