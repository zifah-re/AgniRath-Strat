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
import pandas as pd
import pvlib
import typing as _t
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline

from core import solcast_time as _solcast_time

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

    def ghi_wm2_array(self, t_s, x_m) -> np.ndarray:
        """Vectorized GHI for parallel (t_s, x_m) arrays.

        Audit fix 1: callers that can batch positions+times (forward_sim
        segment precomputation, Tier 1 coarse scans, scenarios) should use
        this instead of per-point ghi_wm2() calls through the Python API.
        The default implementation is an exact per-point loop so any
        third-party/legacy provider without its own vectorized path still
        works — it is just slower than the numpy overrides below.
        """
        t_s = np.asarray(t_s, dtype=float)
        x_m = np.asarray(x_m, dtype=float)
        if t_s.shape != x_m.shape:
            raise ValueError("t_s and x_m must have identical shapes")
        return np.array([self.ghi_wm2(float(a), float(b))
                         for a, b in zip(t_s.ravel(), x_m.ravel())],
                        dtype=float).reshape(t_s.shape)

class GaussianProvider(SolarProvider):
    PEAK_WM2 = 1073.099
    NOON_S = 43_200.0
    SIGMA_S = 11_600.0

    def ghi_wm2(self, t_s: float, x_m: float = 0.0) -> float:
        return float(self.PEAK_WM2
                     * math.exp(-0.5 * ((t_s - self.NOON_S) / self.SIGMA_S) ** 2))

    def ghi_wm2_array(self, t_s, x_m=0.0) -> np.ndarray:
        t_s = np.asarray(t_s, dtype=float)
        return self.PEAK_WM2 * np.exp(-0.5 * ((t_s - self.NOON_S)
                                              / self.SIGMA_S) ** 2)

class PVLibClearSkyProvider(SolarProvider):
    def __init__(self, lat: float, lon: float, date_iso: str,
                 tz: str = "Africa/Johannesburg"): 
        self._pvlib = pvlib
        self._times = pd.date_range(f"{date_iso} 00:00", f"{date_iso} 23:59",
                                    freq="1min", tz=tz)
        loc = pvlib.location.Location(lat, lon, tz=tz)
        self._ghi = loc.get_clearsky(self._times, model="ineichen")["ghi"].values

    def ghi_wm2(self, t_s: float, x_m: float = 0.0) -> float:
        idx = int(np.clip(t_s // 60, 0, len(self._ghi) - 1))
        return float(self._ghi[idx])

    def ghi_wm2_array(self, t_s, x_m=0.0) -> np.ndarray:
        t_s = np.asarray(t_s, dtype=float)
        idx = np.clip(t_s // 60, 0, len(self._ghi) - 1).astype(np.intp)
        return self._ghi[idx]

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

    def node_index_array(self, x_m: np.ndarray) -> np.ndarray:
        """Weather-node index for every position in x_m, in one batched
        lat/lon KDTree query (vectorized route lookup). This is the
        pre-resolved lookup forward_sim uses so the integrator never re-runs
        a KDTree query inside the substep loop."""
        x_m = np.asarray(x_m, dtype=float)
        if self.route is None:
            return np.full(np.shape(x_m), len(self.spline_models) // 2,
                           dtype=np.intp)
        lats, lons = self.route.latlon_array(x_m)
        _, idx = self.tree.query(
            np.column_stack([lats.ravel(), lons.ravel()]))
        return idx.reshape(np.shape(x_m)).astype(np.intp)

    def ghi_wm2_at_node(self, t_s: float, node_index: int) -> float:
        """Scalar GHI at a pre-resolved weather node — the hot path after
        node_index_array() removed the lat/lon + KDTree lookup."""
        return max(0.0, float(self.spline_models[int(node_index)](t_s)))

    def ghi_wm2_array(self, t_s, x_m) -> np.ndarray:
        t_s = np.asarray(t_s, dtype=float)
        x_m = np.asarray(x_m, dtype=float)
        if t_s.shape != x_m.shape:
            raise ValueError("t_s and x_m must have identical shapes")
        node_idx = self.node_index_array(x_m)
        out = np.empty(np.shape(x_m), dtype=float)
        # One spline evaluation per weather node (vectorized over the
        # positions assigned to that node) instead of one per position.
        for n in np.unique(node_idx):
            mask = node_idx == n
            out[mask] = self.spline_models[int(n)](t_s[mask])
        return np.clip(out, 0.0, None)

class RealSolcastSolarProvider(SolarProvider):
    """
    Consumes the ACTUAL race-week Solcast forecast (data/Solar_real/mean_*.jsonl):

        [{"lat": ..., "lon": ...,
          "data": [{"period_end": "2026-09-10T07:00:00+00:00",
                     "ghi": 461, "dni": 437, "air_temp": 20,
                     "wind_speed_10m": 2.3, "wind_direction_10m": 307}, ...]}]

    This is a DIFFERENT schema from HourlyJSONSolarProvider's typical-year
    file (node keys "latitude"/"longitude", hourly GHI nested under
    historical_weather.hourly.shortwave_radiation, 24 local-naive hourly
    buckets). The real data is instead: node keys "lat"/"lon", 5-minute
    resolution, UTC timestamps, and only covers the actual forecast daylight
    window (no synthetic night-time zero padding to 24h).

    FROZEN INTERFACE preserved exactly: ghi_wm2(t_s, x_m) still takes t_s as
    LOCAL seconds since midnight of the race day and x_m as route distance.
    node_index_array / ghi_wm2_at_node / ghi_wm2_array are also implemented
    so forward_sim's fast path (see simulator/forward_sim.py) is exercised
    exactly as it is for HourlyJSONSolarProvider — no perf regression.
    """
    def __init__(self, json_paths: list[str] | str, route):
        if isinstance(json_paths, str):
            json_paths = [json_paths]

        self.route = route
        coords = []
        self.spline_models = []
        # Per-node (t_min_s, t_max_s): outside this window the real forecast
        # simply has no sample (pre-dawn / post-dusk are not requested from
        # Solcast at all, unlike the old data's explicit 0.0 night buckets),
        # so a query outside the window returns night (0.0) rather than
        # extrapolating the spline off the end of its data.
        self.t_bounds = []

        for jp in json_paths:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)

            for node in data:
                coords.append([node["lat"], node["lon"]])
                samples = node["data"]
                t_local_s = np.array(
                    [_solcast_time.period_end_to_local_s(s["period_end"])
                     for s in samples], dtype=float)
                ghi = np.array(
                    [float(s.get("ghi", 0.0) or 0.0) for s in samples],
                    dtype=float)
                order = np.argsort(t_local_s)
                t_local_s, ghi = t_local_s[order], ghi[order]
                spline_fit = CubicSpline(t_local_s, ghi, bc_type="natural")
                self.spline_models.append(spline_fit)
                self.t_bounds.append((float(t_local_s[0]), float(t_local_s[-1])))

        self.tree = cKDTree(np.array(coords))

    def _in_window(self, t_s: float, idx: int) -> bool:
        t_min, t_max = self.t_bounds[idx]
        return t_min <= t_s <= t_max

    def ghi_wm2(self, t_s: float, x_m: float = 0.0) -> float:
        if self.route is None:
            idx = len(self.spline_models) // 2
        else:
            lat, lon = self.route.latlon_at(x_m)
            _, idx = self.tree.query([lat, lon])

        if not self._in_window(t_s, idx):
            return 0.0
        return max(0.0, float(self.spline_models[idx](t_s)))

    def node_index_array(self, x_m: np.ndarray) -> np.ndarray:
        x_m = np.asarray(x_m, dtype=float)
        if self.route is None:
            return np.full(np.shape(x_m), len(self.spline_models) // 2,
                           dtype=np.intp)
        lats, lons = self.route.latlon_array(x_m)
        _, idx = self.tree.query(
            np.column_stack([lats.ravel(), lons.ravel()]))
        return idx.reshape(np.shape(x_m)).astype(np.intp)

    def ghi_wm2_at_node(self, t_s: float, node_index: int) -> float:
        idx = int(node_index)
        if not self._in_window(t_s, idx):
            return 0.0
        return max(0.0, float(self.spline_models[idx](t_s)))

    def ghi_wm2_array(self, t_s, x_m) -> np.ndarray:
        t_s = np.asarray(t_s, dtype=float)
        x_m = np.asarray(x_m, dtype=float)
        if t_s.shape != x_m.shape:
            raise ValueError("t_s and x_m must have identical shapes")
        node_idx = self.node_index_array(x_m)
        out = np.zeros(np.shape(x_m), dtype=float)
        for n in np.unique(node_idx):
            mask = node_idx == n
            t_min, t_max = self.t_bounds[int(n)]
            vals = self.spline_models[int(n)](t_s[mask])
            in_win = (t_s[mask] >= t_min) & (t_s[mask] <= t_max)
            out[mask] = np.where(in_win, np.clip(vals, 0.0, None), 0.0)
        return out


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

class FlooredSolarProvider:
    """Wraps a JSON solar provider with a GaussianProvider clear-sky floor.

    If the JSON day has anomalously low solar (avg daytime GHI below a
    threshold, decided by the caller), ghi_wm2 returns
    ``max(json_ghi, blend_frac * gaussian_ghi)`` — preventing a single
    bad-weather historical year from making the whole race infeasible while
    still respecting realistic cloudy-day data.

    MOVED here VERBATIM from trust_region.py's ``__main__`` block (22/08) so it
    is a top-level, importable, PICKLABLE class. Under the 'spawn' multiprocessing
    start method (Windows/macOS default) worker processes re-import modules by
    name; a class defined inside ``if __name__ == '__main__'`` does not exist
    there, so the fastened Tier-2 process pool died with a BrokenProcessPool /
    "cannot pickle" error on Windows. As a module-level class it pickles and
    unpickles cleanly under every start method.

    IMPORTANT: this is a BARE class (it deliberately does NOT subclass
    SolarProvider) and exposes exactly the same methods the original did —
    ghi_wm2 / node_index_array / ghi_wm2_at_node, and NO ghi_wm2_array. That
    matters: forward_sim._ghi_segment picks ghi_wm2_array first when present, so
    adding one (or inheriting SolarProvider's default) would switch a floored
    day onto a different GHI code path and could change its numbers. Keeping the
    surface identical guarantees byte-for-byte the same results as before.
    """
    def __init__(self, json_prov, blend_frac: float = 0.5):
        self._json = json_prov
        self._gauss = GaussianProvider()
        self._blend = blend_frac

    def ghi_wm2(self, t_s: float, x_m: float = 0.0) -> float:
        j = self._json.ghi_wm2(t_s, x_m)
        g = self._gauss.ghi_wm2(t_s, x_m)
        return max(j, self._blend * g)

    def node_index_array(self, x_m):
        return self._json.node_index_array(x_m)

    def ghi_wm2_at_node(self, t_s: float, node_index: int) -> float:
        j = self._json.ghi_wm2_at_node(t_s, node_index)
        g = self._gauss.ghi_wm2(t_s, 0.0)  # Gaussian floor is position-independent
        return max(j, self._blend * g)