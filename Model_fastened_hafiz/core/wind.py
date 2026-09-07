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

from core import solcast_time as _solcast_time

class WindProvider:
    def wind(self, t_s: float, x_m: float) -> tuple[float, float]:
        """Return (speed_ms, dir_deg_from)."""  # pragma: no cover
        raise NotImplementedError

    def wind_array(self, t_s, x_m) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized wind for parallel (t_s, x_m) arrays -> (speed_ms,
        dir_deg_from) arrays. Same exact-per-point default-loop contract as
        SolarProvider.ghi_wm2_array; numpy providers override it."""
        t_s = np.asarray(t_s, dtype=float)
        x_m = np.asarray(x_m, dtype=float)
        if t_s.shape != x_m.shape:
            raise ValueError("t_s and x_m must have identical shapes")
        spd = np.empty(t_s.shape, dtype=float)
        d = np.empty(t_s.shape, dtype=float)
        for i, (a, b) in enumerate(zip(t_s.ravel(), x_m.ravel())):
            spd.ravel()[i], d.ravel()[i] = self.wind(float(a), float(b))
        return spd, d

class ConstantWindProvider(WindProvider):
    def __init__(self, speed_ms: float = 0.0, dir_deg_from: float = 0.0):
        self._s = float(speed_ms)
        self._d = float(dir_deg_from)

    def wind(self, t_s: float, x_m: float) -> tuple[float, float]:
        return self._s, self._d

    def wind_array(self, t_s, x_m) -> tuple[np.ndarray, np.ndarray]:
        shape = np.broadcast(np.asarray(t_s, dtype=float),
                             np.asarray(x_m, dtype=float)).shape
        return (np.full(shape, self._s, dtype=float),
                np.full(shape, self._d, dtype=float))

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

    def node_index_array(self, x_m: np.ndarray) -> np.ndarray:
        """Weather-node index per position, one batched KDTree query."""
        x_m = np.asarray(x_m, dtype=float)
        if self.route is None:
            return np.full(np.shape(x_m), len(self.speed_matrix) // 2,
                           dtype=np.intp)
        lats, lons = self.route.latlon_array(x_m)
        _, idx = self.tree.query(
            np.column_stack([lats.ravel(), lons.ravel()]))
        return idx.reshape(np.shape(x_m)).astype(np.intp)

    def wind_at_node(self, t_s: float, node_index: int) -> tuple[float, float]:
        """Scalar wind at a pre-resolved weather node (no KDTree query)."""
        speed_ms = float(np.interp(t_s, self.t_s_array,
                                   self.speed_matrix[int(node_index)]))
        dir_deg = float(np.interp(t_s, self.t_s_array,
                                  self.dir_matrix[int(node_index)]))
        return speed_ms, dir_deg

    def wind_array(self, t_s, x_m) -> tuple[np.ndarray, np.ndarray]:
        t_s = np.asarray(t_s, dtype=float)
        x_m = np.asarray(x_m, dtype=float)
        if t_s.shape != x_m.shape:
            raise ValueError("t_s and x_m must have identical shapes")
        node_idx = self.node_index_array(x_m)
        spd = np.empty(t_s.shape, dtype=float)
        d = np.empty(t_s.shape, dtype=float)
        for n in np.unique(node_idx):
            mask = node_idx == n
            spd[mask] = np.interp(t_s[mask], self.t_s_array,
                                  self.speed_matrix[int(n)])
            d[mask] = np.interp(t_s[mask], self.t_s_array,
                                self.dir_matrix[int(n)])
        return spd, d

# ---------------------------------------------------------------------------
# Decomposition helpers
# ---------------------------------------------------------------------------

class RealSolcastWindProvider(WindProvider):
    """
    Same ACTUAL race-week Solcast schema as core.solar.RealSolcastSolarProvider
    (see that class's docstring): node keys "lat"/"lon", 5-minute UTC-stamped
    samples under "data", "wind_speed_10m"/"wind_direction_10m" per sample.

    Unlike HourlyJSONWindProvider's old Open-Meteo source, Solcast reports
    wind_speed_10m directly in m/s — no /3.6 km/h->m/s conversion here.

    Outside the recorded forecast window, np.interp naturally clamps to the
    nearest edge sample rather than extrapolating (matches np.interp's
    default behaviour, and wind has no "night" concept the way GHI does).
    """
    def __init__(self, json_paths: list[str] | str, route):
        if isinstance(json_paths, str):
            json_paths = [json_paths]

        self.route = route
        coords = []
        self.t_arrays = []
        self.speed_arrays = []
        self.dir_arrays = []

        for jp in json_paths:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)

            for node in data:
                coords.append([node["lat"], node["lon"]])
                samples = node["data"]
                t_local_s = np.array(
                    [_solcast_time.period_end_to_local_s(s["period_end"])
                     for s in samples], dtype=float)
                speed_ms = np.array(
                    [float(s.get("wind_speed_10m", 0.0) or 0.0)
                     for s in samples], dtype=float)
                dir_deg = np.array(
                    [float(s.get("wind_direction_10m", 0.0) or 0.0)
                     for s in samples], dtype=float)
                order = np.argsort(t_local_s)
                self.t_arrays.append(t_local_s[order])
                self.speed_arrays.append(speed_ms[order])
                self.dir_arrays.append(dir_deg[order])

        self.tree = cKDTree(np.array(coords))

    def wind(self, t_s: float, x_m: float = 0.0) -> tuple[float, float]:
        if self.route is None:
            idx = len(self.speed_arrays) // 2
        else:
            lat, lon = self.route.latlon_at(x_m)
            _, idx = self.tree.query([lat, lon])

        speed_ms = float(np.interp(t_s, self.t_arrays[idx], self.speed_arrays[idx]))
        dir_deg = float(np.interp(t_s, self.t_arrays[idx], self.dir_arrays[idx]))
        return speed_ms, dir_deg

    def node_index_array(self, x_m: np.ndarray) -> np.ndarray:
        x_m = np.asarray(x_m, dtype=float)
        if self.route is None:
            return np.full(np.shape(x_m), len(self.speed_arrays) // 2,
                           dtype=np.intp)
        lats, lons = self.route.latlon_array(x_m)
        _, idx = self.tree.query(
            np.column_stack([lats.ravel(), lons.ravel()]))
        return idx.reshape(np.shape(x_m)).astype(np.intp)

    def wind_at_node(self, t_s: float, node_index: int) -> tuple[float, float]:
        idx = int(node_index)
        speed_ms = float(np.interp(t_s, self.t_arrays[idx], self.speed_arrays[idx]))
        dir_deg = float(np.interp(t_s, self.t_arrays[idx], self.dir_arrays[idx]))
        return speed_ms, dir_deg

    def wind_array(self, t_s, x_m) -> tuple[np.ndarray, np.ndarray]:
        t_s = np.asarray(t_s, dtype=float)
        x_m = np.asarray(x_m, dtype=float)
        if t_s.shape != x_m.shape:
            raise ValueError("t_s and x_m must have identical shapes")
        node_idx = self.node_index_array(x_m)
        spd = np.empty(t_s.shape, dtype=float)
        d = np.empty(t_s.shape, dtype=float)
        for n in np.unique(node_idx):
            mask = node_idx == n
            spd[mask] = np.interp(t_s[mask], self.t_arrays[int(n)],
                                  self.speed_arrays[int(n)])
            d[mask] = np.interp(t_s[mask], self.t_arrays[int(n)],
                                self.dir_arrays[int(n)])
        return spd, d


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