"""
core/wind.py — wind pipeline behind ONE frozen signature.

FROZEN INTERFACE (contract, README §Interfaces):
    provider.wind(t_s: float, x_m: float) -> (speed_ms, dir_deg_from)

Providers:
    ConstantWindProvider     trivial/fallback + scenario injection.
    HourlyJSONWindProvider   Primary provider using linear interpolation 
                             over hourly JSON weather data.
"""

from __future__ import annotations

import math
import pathlib
import json
import typing as _t
import numpy as np
from scipy.spatial import cKDTree

class WindProvider:
    def wind(self, t_s: float, x_m: float) -> tuple[float, float]:
        """Return (speed_ms, dir_deg_from)."""  # pragma: no cover
        raise NotImplementedError

class ConstantWindProvider(WindProvider):
    def __init__(self, speed_ms: float = 0.0, dir_deg_from: float = 0.0):
        self._s = float(speed_ms)
        self._d = float(dir_deg_from)

    def wind(self, t_s: float, x_m: float) -> tuple[float, float]:
        return self._s, self._d

class HourlyJSONWindProvider(WindProvider):
    """
    Consumes a point-by-point JSON containing hourly weather arrays.
    Maps x_m to the nearest spatial node and converts wind speed to m/s.
    """
    def __init__(self, json_paths: list[str] | str, route):
        if isinstance(json_paths, str):
            json_paths = [json_paths]
            
        self.route = route
        self.t_s_array = np.arange(24) * 3600.0
        
        coords = []
        self.speed_matrix = []
        self.dir_matrix = []
        
        # Loop through EVERY file for this day and extract the nodes
        for jp in json_paths:
            with open(jp, 'r', encoding="utf-8") as f:
                data = json.load(f)
                
            for node in data:
                coords.append([node["latitude"], node["longitude"]])
                speed_kmh = np.array(node["historical_weather"]["hourly"]["wind_speed_10m"], dtype=float)
                self.speed_matrix.append(speed_kmh / 3.6)
                self.dir_matrix.append(node["historical_weather"]["hourly"]["wind_direction_10m"])
            
        self.tree = cKDTree(np.array(coords))
        self.speed_matrix = np.array(self.speed_matrix, dtype=float)
        self.dir_matrix = np.array(self.dir_matrix, dtype=float)

    def wind(self, t_s: float, x_m: float = 0.0) -> tuple[float, float]:
        if self.route is None:
            idx = len(self.speed_matrix) // 2
        else:
            lat, lon = self.route.latlon_at(x_m)
            _, idx = self.tree.query([lat, lon])
            
        speed_array = self.speed_matrix[idx]
        dir_array = self.dir_matrix[idx]
        
        speed_ms = float(np.interp(t_s, self.t_s_array, speed_array))
        dir_deg = float(np.interp(t_s, self.t_s_array, dir_array))
        return speed_ms, dir_deg

# ---------------------------------------------------------------------------
# Decomposition helpers
# ---------------------------------------------------------------------------

def along_track_ms(speed_ms: float, dir_deg_from: float,
                   bearing_deg: float) -> float:
    to_deg = (dir_deg_from + 180.0) % 360.0
    return speed_ms * math.cos(math.radians(bearing_deg - to_deg))

def relative_wind(v_car_ms: float, speed_ms: float, dir_deg_from: float,
                  bearing_deg: float) -> tuple[float, float]:
    to_rad = math.radians((dir_deg_from + 180.0) % 360.0)
    br = math.radians(bearing_deg)
    wx, wy = speed_ms * math.sin(to_rad), speed_ms * math.cos(to_rad)
    cx, cy = v_car_ms * math.sin(br), v_car_ms * math.cos(br)
    rx, ry = wx - cx, wy - cy                    
    mag = math.hypot(rx, ry)
    if mag < 1e-9:
        return 0.0, 0.0
    ax, ay = -rx, -ry
    hx, hy = math.sin(br), math.cos(br)
    cospsi = max(-1.0, min(1.0, (ax * hx + ay * hy) / mag))
    return mag, math.degrees(math.acos(cospsi))