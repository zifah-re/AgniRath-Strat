"""
core/route.py — route file contract + loader (workplan 0.1 frozen schema).

FROZEN SCHEMA — data/processed/route_day{d}.parquet, one row per ~10 m grid
point (Plan v3 §8 DATA layer). Columns (all required):

    distance_m        float   cumulative along-route distance from day start
    lat, lon          float   WGS84
    elevation_m       float   smoothed elevation
    slope_pct         float   smoothed signed gradient, percent
    bearing_deg       float   route bearing at point, deg clockwise from N
    curvature_1pm     float   |d(bearing)/ds| in 1/m (turn caps input)
    v_max_ms          float   min(TomTom limit, turn cap, car max, COC)
    circle_id         int     solar/wind query circle (Plan v3 §6.1)
    seg_type          str     'stage1' | 'loop_<name>' | 'stage2'
    red_flag_trailer  bool    P_req(60km/h) > derated P_max (Plan v3 §5.1)
    control_stop      bool    grid point is within the CS zone
    day               int     1-based race day

Produced ONLY by pipeline/build_route.py (block 2). This module gives every
other layer a typed read API so nobody re-parses parquet ad hoc.
"""

from __future__ import annotations

import pathlib

import numpy as np

REQUIRED_COLUMNS = [
    "distance_m", "lat", "lon", "elevation_m", "slope_pct", "bearing_deg",
    "curvature_1pm", "v_max_ms", "circle_id", "seg_type",
    "red_flag_trailer", "control_stop", "day",
]


class Route:
    """Thin typed wrapper over one day's route dataframe."""

    def __init__(self, df):
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"route file missing columns: {missing}")
        self.df = df.sort_values("distance_m").reset_index(drop=True)
        self._x = self.df["distance_m"].to_numpy()

    # -- factory ----------------------------------------------------------
    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Route":
        import pandas as pd
        return cls(pd.read_parquet(path))

    # -- lookups (all by distance x_m; nearest-grid semantics) ------------
    def _idx(self, x_m: float | np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self._x, x_m), 0, len(self._x) - 1)

    def slope_pct_at(self, x_m):
        return self.df["slope_pct"].to_numpy()[self._idx(x_m)]

    def bearing_deg_at(self, x_m):
        return self.df["bearing_deg"].to_numpy()[self._idx(x_m)]

    def v_max_ms_at(self, x_m):
        return self.df["v_max_ms"].to_numpy()[self._idx(x_m)]

    def circle_id_at(self, x_m) -> int:
        return int(self.df["circle_id"].to_numpy()[self._idx(x_m)])

    def latlon_at(self, x_m):
        i = self._idx(x_m)
        return (self.df["lat"].to_numpy()[i], self.df["lon"].to_numpy()[i])

    def red_flag_at(self, x_m) -> bool:
        return bool(self.df["red_flag_trailer"].to_numpy()[self._idx(x_m)])

    def red_flag_at(self, x_m: float) -> bool:
        if "red_flag_trailer" not in self.df.columns:
            return False
        x = self.df["distance_m"].to_numpy()
        idx = min(int(np.searchsorted(x, x_m)), len(self.df) - 1)
        return bool(self.df.iloc[idx]["red_flag_trailer"])
 

    @property
    def total_m(self) -> float:
        return float(self._x[-1])
