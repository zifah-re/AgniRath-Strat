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
        # Cache columns once. forward_sim calls these per PHYSICS SUBSTEP
        # (ENERGY_GRID_M-spaced -> up to ~100 calls per 10km control segment,
        # times every GA/SLSQP candidate evaluation). Re-pulling .to_numpy()/
        # .iloc[] from the DataFrame on every call was a measurable per-call
        # cost multiplied by hundreds of thousands of calls per solve.
        self._slope = self.df["slope_pct"].to_numpy()
        self._bearing = self.df["bearing_deg"].to_numpy()
        self._v_max = self.df["v_max_ms"].to_numpy()
        self._circle_id = self.df["circle_id"].to_numpy()
        self._lat = self.df["lat"].to_numpy()
        self._lon = self.df["lon"].to_numpy()
        self._red_flag = self.df["red_flag_trailer"].to_numpy()
        self._control_stop = self.df["control_stop"].to_numpy()
        self._seg_type = self.df["seg_type"].to_numpy()

    # -- factory ----------------------------------------------------------
    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Route":
        import pandas as pd
        return cls(pd.read_parquet(path))

    # -- lookups (all by distance x_m; nearest-grid semantics) ------------
    def _idx(self, x_m: float | np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self._x, x_m), 0, len(self._x) - 1)

    def slope_pct_at(self, x_m):
        return self._slope[self._idx(x_m)]

    def bearing_deg_at(self, x_m):
        return self._bearing[self._idx(x_m)]

    def v_max_ms_at(self, x_m):
        return self._v_max[self._idx(x_m)]

    def circle_id_at(self, x_m) -> int:
        return int(self._circle_id[self._idx(x_m)])

    def latlon_at(self, x_m):
        i = self._idx(x_m)
        return (self._lat[i], self._lon[i])

    def red_flag_at(self, x_m) -> bool:
        return bool(self._red_flag[self._idx(x_m)])

    def control_stop_at(self, x_m) -> bool:
        return bool(self._control_stop[self._idx(x_m)])

    def seg_type_at(self, x_m) -> str:
        return str(self._seg_type[self._idx(x_m)])

    def set_red_flag_mask(self, mask: np.ndarray) -> None:
        """Overwrite the trailered-segment mask in place (both the cached
        array red_flag_at() reads, and self.df, so anything reading the
        DataFrame directly stays consistent). See
        tier1.compute_trailered_mask_full / trust_region.py's route loading."""
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(self.df):
            raise ValueError(f"trailered mask length {len(mask)} != route length {len(self.df)}")
        self._red_flag = mask
        self.df["red_flag_trailer"] = mask

    @property
    def total_m(self) -> float:
        return float(self._x[-1])