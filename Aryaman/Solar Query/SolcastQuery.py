'''"""

cdsapiQuerySemivariance.py
---------------------------------------------------------------------------
Parses the Sasol Solar Challenge KML route, queries ERA5 (via cdsapi) for
GHI / DNI / 10 m wind for each stage, discretizes each stage into waypoints,
looks up local weather at each waypoint's arrival time, and exports
race_weather_data.json.

v1.2 CHANGE (this revision): KML parsing/day-grouping rewritten.

  Root cause of the "only Day 1 and Day 2 came out right" bug: the KML's
  route LineString placemarks are named things like
  "10 Sept Stage 1: Boiketlong to Rustenburg" and
  "11 Sept Stage 1: Swart Ruggens to Zeerust" - i.e. every day's route
  segments are labelled "Stage 1" / "Stage 2", never "Day N". The old
  _DAY_PATTERN regex `(?:day|stage)\s*0*(\d+)` matched the word "Stage" and
  captured the STAGE number (1 or 2), not the day number. That collapsed
  every day's "Stage 1" segment into one "Day 1" bucket and every day's
  "Stage 2" segment into one "Day 2" bucket, and no other keys were ever
  produced (there is no "Stage 3" anywhere in the file) - hence only
  "Day 1" / "Day 2" ever appeared in the output, each one a scrambled
  merge of unrelated days' segments stitched together.

  Fix: parse per KML *Folder* instead of scanning all Placemarks flat and
  grouping by a regex on the placemark name. The KML's real day boundaries
  are the Folder structure (Document > Folder "My Places" > Folder "Day N
  ..."), so `parse_kml_main_routes()` now walks those folders directly and
  keys results by the folder's actual day label. Loop placemarks are still
  excluded, and Day 3 ("Full Blind") is naturally dropped from the output
  since it has no LineString placemarks at all - by design, that stage has
  no prescribed route.

  A second, related bug this fix also addresses: `main()` previously
  assigned each stage a date via `RACE_START_DATE + timedelta(days=i)`
  where `i` was the *index* into the (regex-broken) stage dict. Once Day 3
  is correctly dropped for having no route, that index-based date math
  would silently shift every day from Day 4 onward back by one calendar
  day. Dates are now derived from the day number parsed out of each
  folder's actual label (e.g. "Day 4" -> 4 -> RACE_START_DATE + 3 days),
  so gaps like the missing Day 3 no longer cause drift.

Key assumptions / decisions (see chat response for full rationale):
  - Weather source stays cdsapi/ERA5 only, per explicit instruction in this
    task (a separate, older team note mentions Solcast; that applies to the
    Model/ pipeline's primary solar interface, not this script).
  - Clear-sky GHI is computed with pvlib's Ineichen model (matching the
    PVLibClearSkyProvider already used in Model/core/solar.py), with a
    guarded fallback to a simple Haurwitz clear-sky model if pvlib isn't
    installed, so the script still runs without a new hard dependency.
  - epsilon default is 0.1 (not 0.05) on the sqrt(gamma) <-> Kc-difference
    scale; see rationale in the accompanying explanation. Both are exposed
    as DEFAULTS below.
  - R_max is clamped to [MIN_R_MAX_KM, MAX_R_MAX_KM] as a sanity bound
    around Model/configs/solver_config.py's CIRCLE_TARGET_DIAMETER_KM
    (~20-30 km decorrelation-scale guess -> ~10-15 km radius).
---------------------------------------------------------------------------
"""

import math
import zipfile
import glob
import os
import re
import shutil
import statistics
import tempfile
import json
import warnings
import xml.etree.ElementTree as ET
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import cdsapi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit, brentq

try:
    import pvlib
    _PVLIB_AVAILABLE = True
except ImportError:
    _PVLIB_AVAILABLE = False
    warnings.warn(
        "pvlib not installed - falling back to an approximate Haurwitz "
        "clear-sky model for the semivariogram's Kc normalization. "
        "`pip install pvlib` to match Model/core/solar.py's Ineichen model."
    )

# ---------------------------------------------------------------------------
# DEFAULTS - edit these instead of being prompted at runtime.
# ---------------------------------------------------------------------------
KML_FILENAME_HINT = "2026 Sasol Solar Challenge Route (Publish).kml"

RACE_START_DATE = date(2025, 9, 10)

STAGE_START_TIME_DAY1 = time(9, 0, 0)    # Day 1 start, SAST
STAGE_START_TIME_OTHER = time(8, 0, 0)   # Days 2+ start, SAST
DEFAULT_STAGE_HOURS = 8.0                # used to derive a default cruising speed per stage
ERA5_AREA = [-22, 16, -35, 33]           # [north, west, south, east]

# --- Semivariogram-derived dynamic R_max (section 3.5) ---------------------
FALLBACK_R_MAX_KM = 10.0         # used only if a stage's variogram can't be fit
SEMIVARIOGRAM_EPSILON = 0.1      # tolerance on sqrt(gamma(h)); see module docstring
MIN_R_MAX_KM = 5.0               # sanity floor (avoid pathologically small spacing)
MAX_R_MAX_KM = 40.0              # sanity ceiling (avoid pathologically sparse waypoints)
BBOX_BUFFER_KM = 20.0            # buffer added around each stage's coord bbox when
                                  # pulling ERA5 grid points into the variogram
N_LAG_BINS = 12                  # distance bins for the empirical variogram
MIN_PAIRS_PER_BIN = 3            # bins with fewer pairs than this are dropped
MIN_GRID_POINTS_FOR_FIT = 6      # below this, skip fitting and use the fallback
MIN_CLEARSKY_GHI_WM2 = 50.0      # drop low-sun-angle samples (unstable Kc ratio)
SAVE_VARIOGRAM_PLOTS = True      # diagnostic PNG per stage, for QA / debugging
VARIOGRAM_PLOT_DIR = "variogram_diagnostics"

# Matches the day number out of a KML Folder label like "Day 4", "Day 2 Half
# Blind", "Day 3 Full Blind", "Day 1 " (trailing space), etc. Deliberately
# does NOT match "Stage N" - that word only ever appears on the individual
# route-segment Placemarks inside a day folder, not on the folder itself,
# but staying folder-scoped (rather than falling back to a looser regex)
# avoids resurrecting the original Stage/Day collision bug.
_DAY_FOLDER_PATTERN = re.compile(r"day\s*0*(\d+)", re.IGNORECASE)


# --- 1. KML PARSING, PER DAY FOLDER ---
def parse_kml_day_folders(kml_path):
    """Returns an ordered dict {day_label: [segment_coords, ...]} where
    day_label is the KML Folder's own name (e.g. 'Day 4', 'Day 2 Half
    Blind') and each segment_coords is a list of (lat, lon) tuples for one
    non-loop LineString placemark inside that folder, in document order.

    Walking Folders directly (rather than scanning every Placemark in the
    document and grouping by a regex on its name) is the fix for the
    Stage-N/Day-N collision described in the module docstring: the route
    LineStrings are only ever named "Stage 1"/"Stage 2", so grouping by
    name-regex silently merged every day's Stage 1 into one bucket and
    every day's Stage 2 into another. Grouping by the folder they actually
    live in has no such ambiguity.

    A day folder with no LineString placemarks (Day 3, "Full Blind" - only
    start/finish Points, no prescribed route) is simply omitted from the
    result. That's expected, not an error: Full Blind stages have no fixed
    route to discretize or fetch weather along.
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    day_routes = {}
    # Day folders are nested: Document > Folder "My Places" > Folder "Day N ..."
    for day_folder in root.findall(".//kml:Folder/kml:Folder", ns):
        name_elem = day_folder.find("kml:name", ns)
        day_label = name_elem.text.strip() if name_elem is not None else "Unknown"

        segments = []
        for placemark in day_folder.findall("kml:Placemark", ns):
            pname_elem = placemark.find("kml:name", ns)
            pname = pname_elem.text.strip() if pname_elem is not None else ""
            if "loop" in pname.lower():
                continue

            linestring = placemark.find("kml:LineString/kml:coordinates", ns)
            if linestring is None or not linestring.text:
                continue

            coords = []
            for pt in linestring.text.strip().split():
                lon, lat, *_ = map(float, pt.split(","))
                coords.append((lat, lon))
            segments.append(coords)

        if segments:
            day_routes[day_label] = segments
        else:
            print(f"  [i] {day_label}: no LineString route segments found "
                  f"(expected for a Full Blind stage) - skipping.")

    return day_routes


def parse_day_number(day_label):
    """Extracts the integer day number from a folder label like 'Day 4' or
    'Day 2 Half Blind'. Raises ValueError if the label doesn't contain a
    recognizable 'Day N' - fails loudly rather than silently mis-dating a
    stage, since date alignment depends entirely on this."""
    m = _DAY_FOLDER_PATTERN.search(day_label)
    if not m:
        raise ValueError(
            f"Could not parse a day number out of folder label {day_label!r}; "
            f"expected something matching 'Day <N>'."
        )
    return int(m.group(1))


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


# --- 2. STAGE STITCHING ---
def stitch_day_segments(day_routes):
    """Takes {day_label: [segment_coords, ...]} (already correctly grouped
    by parse_kml_day_folders) and concatenates each day's segments into one
    continuous coordinate list, splicing out a duplicated junction point
    when consecutive segments meet at (near) the same spot. Returned dict
    preserves day order and is keyed the same way as the input."""
    stitched = {}
    for day_label, segments in day_routes.items():
        combined = list(segments[0])
        for seg in segments[1:]:
            if not seg:
                continue
            if combined and haversine(*combined[-1], *seg[0]) < 0.05:
                combined.extend(seg[1:])
            else:
                combined.extend(seg)
        stitched[day_label] = combined

    # Sort by the actual parsed day number so downstream processing (and
    # the ERA5 date list) proceeds in true chronological stage order, even
    # though the KML folders already happen to be in this order.
    return dict(sorted(stitched.items(), key=lambda kv: parse_day_number(kv[0])))


# --- 3. GRIDDED DATA INGESTION ---
def _normalize_time_dim(ds):
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    return ds


def _dates_by_year_month(dates):
    groups = {}
    for d in dates:
        groups.setdefault((d.year, d.month), set()).add(d.day)
    return groups


def _accum_j_m2_to_w_m2(accum_j_m2):
    """ERA5 accumulates radiation in J/m^2 since the previous forecast step;
    divide by 3600 to get an average W/m^2 and floor at 0 (shared by the
    per-waypoint lookup and the semivariogram's grid extraction, so the
    conversion only lives in one place)."""
    return max(0.0, float(accum_j_m2) / 3600.0)


_REQUIRED_ERA5_VARS = ("ssrd", "fdir", "u10", "v10")


def _open_one_era5_file(nc_path):
    """Opens a single ERA5 netcdf file, normalizes the time dim, and drops
    the ensemble 'number' dim (reanalysis is a single deterministic member,
    so it's always length 1, but it changes the shape of every downstream
    .sel()/.interp() call if left in)."""
    ds = _normalize_time_dim(xr.open_dataset(nc_path, engine="netcdf4"))
    if "number" in ds.dims:
        ds = ds.squeeze("number", drop=True)
    return ds


def _open_and_merge_era5_download(tmp_path):
    """Opens whatever CDS actually returned - a single NetCDF, or a ZIP -
    and returns one Dataset with all four requested variables.

    IMPORTANT: the current CDS/ECMWF Datastores backend splits a single
    request into MULTIPLE NetCDF files inside the zip when it mixes
    instantaneous variables (u10/v10, tagged stepType=instant) and
    accumulated variables (ssrd/fdir, tagged stepType=accum) - each file
    only contains its own subset of variables. Taking only nc_files[0]
    silently drops the other file's variables, which is why every waypoint
    lookup used to fail downstream with "No variable named 'ssrd'" instead
    of failing loudly at download time. Fix: open every .nc file found and
    xr.merge() them; the coordinates (time/lat/lon) are identical across
    the split files because they came from the same request.
    """
    extract_dir = None
    try:
        if zipfile.is_zipfile(tmp_path):
            print("  [i] Downloaded file is a ZIP archive. Extracting NetCDF file(s)...")
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                zf.extractall(extract_dir)
                nc_files = sorted(
                    os.path.join(extract_dir, f) for f in zf.namelist() if f.endswith('.nc')
                )
            if not nc_files:
                raise RuntimeError("No .nc file found inside the downloaded ZIP archive.")
            if len(nc_files) > 1:
                print(f"  [i] ZIP contained {len(nc_files)} NetCDF files "
                      f"(e.g. instant/accum split) - merging all of them.")

            sub_datasets = [_open_one_era5_file(p) for p in nc_files]
            ds = xr.merge(sub_datasets, compat="override", join="exact") if len(sub_datasets) > 1 \
                else sub_datasets[0]
        else:
            ds = _open_one_era5_file(tmp_path)

        missing = [v for v in _REQUIRED_ERA5_VARS if v not in ds.variables]
        if missing:
            raise RuntimeError(
                f"ERA5 download is missing variable(s) {missing} after merge. "
                f"Dataset actually contains: {list(ds.data_vars)}. This means "
                f"CDS split the response into more files than were merged, or "
                f"the request itself didn't include these variables - check "
                f"the 'variable' list in download_era5_weather_grid()."
            )

        ds.load()
        return ds
    finally:
        if extract_dir is not None:
            shutil.rmtree(extract_dir, ignore_errors=True)


def download_era5_weather_grid(dates):
    c = cdsapi.Client()
    datasets = []
    for (year, month), days in _dates_by_year_month(dates).items():
        tmp_path = tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name
        try:
            print(f"Downloading ERA5 weather grid for {year}-{month:02d}, days {sorted(days)}...")
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": [
                        "surface_solar_radiation_downwards",           # GHI
                        "total_sky_direct_solar_radiation_at_surface", # DNI
                        "10m_u_component_of_wind",                     # U wind
                        "10m_v_component_of_wind"                      # V wind
                    ],
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": [f"{d:02d}" for d in sorted(days)],
                    "time": [f"{h:02d}:00" for h in range(4, 19)],
                    "area": ERA5_AREA,
                },
                tmp_path,
            )

            ds = _open_and_merge_era5_download(tmp_path)
            datasets.append(ds)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]


def fetch_local_weather(ds, lat, lon, arrival_time_utc):
    naive_time = arrival_time_utc.replace(tzinfo=None)
    try:
        point_data = ds.sel(latitude=lat, longitude=lon, method="nearest").interp(time=np.datetime64(naive_time))

        ghi_wm2 = _accum_j_m2_to_w_m2(point_data["ssrd"].values)
        dni_wm2 = _accum_j_m2_to_w_m2(point_data["fdir"].values)

        u10 = float(point_data["u10"].values)
        v10 = float(point_data["v10"].values)

        wind_speed = math.sqrt(u10**2 + v10**2)
        # Meteorological wind direction: angle wind is blowing FROM
        wind_dir = (270 - math.degrees(math.atan2(v10, u10))) % 360

        return ghi_wm2, dni_wm2, wind_speed, wind_dir
    except Exception as e:
        print(f"  [!] weather lookup failed at ({lat:.3f},{lon:.3f}) {arrival_time_utc}: {e}")
        return 0.0, 0.0, 0.0, 0.0


# --- 3.5 SPATIAL SEMIVARIOGRAM -> DYNAMIC R_MAX ---
#
# For each stage: pull the ERA5 grid points covering that stage's bounding
# box, convert GHI -> clear-sky index Kc (isolating cloud variability from
# the solar-elevation gradient that would otherwise dominate GHI(lat,lon)),
# build an empirical semivariogram gamma(h) of Kc over spatial lag h, fit an
# exponential/spherical model via NLS, and solve for the largest radius
# R_max where sqrt(gamma(R_max)) <= epsilon. That R_max becomes this stage's
# waypoint-spacing radius (replacing the fixed R_MAX_KM).

def _haurwitz_clearsky_ghi_wm2(lat, lon, time_utc):
    """Fallback clear-sky model (used only if pvlib is unavailable).

    Standard solar-position + Haurwitz clear-sky formula:
        GHI_cs = 1098 * sin(elev) * exp(-0.059 / sin(elev))   for elev > 0
    This is intentionally simple (no turbidity/aerosol data) - it exists so
    the semivariogram feature degrades gracefully rather than hard-failing
    when pvlib isn't installed. Prefer pvlib (Ineichen) when available, to
    stay consistent with Model/core/solar.py's PVLibClearSkyProvider.
    """
    day_of_year = time_utc.timetuple().tm_yday
    decl = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + day_of_year) / 365.0)))
    hour_utc = time_utc.hour + time_utc.minute / 60.0 + time_utc.second / 3600.0
    solar_time = hour_utc + lon / 15.0  # rough longitude-only correction (no EoT)
    hour_angle = math.radians(15.0 * (solar_time - 12.0))
    lat_rad = math.radians(lat)

    sin_elev = (math.sin(lat_rad) * math.sin(decl)
                + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle))
    if sin_elev <= 0.01:
        return 0.0
    return max(0.0, 1098.0 * sin_elev * math.exp(-0.059 / sin_elev))


def _pvlib_clearsky_ghi_series_wm2(lat, lon, times_utc):
    """One pvlib call per grid point covering all its timestamps (batched,
    not per-point-per-time, to keep this affordable per stage)."""
    loc = pvlib.location.Location(lat, lon, tz="UTC")
    times_idx = pd.DatetimeIndex(times_utc).tz_localize("UTC")
    cs = loc.get_clearsky(times_idx, model="ineichen")
    return cs["ghi"].values


def _clearsky_ghi_series_wm2(lat, lon, times_utc):
    if _PVLIB_AVAILABLE:
        return _pvlib_clearsky_ghi_series_wm2(lat, lon, times_utc)
    return np.array([_haurwitz_clearsky_ghi_wm2(lat, lon, t) for t in times_utc])


def _route_bbox(coords, buffer_km):
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    mean_lat = sum(lats) / len(lats)
    lat_buf = buffer_km / 111.0
    lon_buf = buffer_km / (111.0 * max(math.cos(math.radians(mean_lat)), 0.1))
    return (min(lats) - lat_buf, max(lats) + lat_buf,
            min(lons) - lon_buf, max(lons) + lon_buf)


def _era5_grid_points_in_bbox(ds, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    sel_lats = lats[(lats >= lat_min) & (lats <= lat_max)]
    sel_lons = lons[(lons >= lon_min) & (lons <= lon_max)]
    return [(float(la), float(lo)) for la in sel_lats for lo in sel_lons]


def _times_for_date(ds, target_date):
    times = pd.to_datetime(ds["time"].values)
    mask = times.date == target_date
    return times[mask]


def _build_kc_field(ds, grid_points, stage_date):
    """For each grid point, returns its ERA5 GHI series (matched to
    `times_used`) converted to clear-sky index Kc, after dropping
    low-elevation samples where the ratio is unstable. Points/times with no
    valid samples anywhere are dropped."""
    times_used = _times_for_date(ds, stage_date)
    if len(times_used) == 0:
        return [], [], [], np.array([])

    kept_lats, kept_lons, kept_series = [], [], []
    valid_mask = None

    for lat, lon in grid_points:
        try:
            point_ds = ds.sel(latitude=lat, longitude=lon, method="nearest").sel(
                time=times_used.values
            )
            ghi_actual = np.array(
                [_accum_j_m2_to_w_m2(v) for v in point_ds["ssrd"].values]
            )
        except Exception:
            continue

        ghi_clear = _clearsky_ghi_series_wm2(lat, lon, times_used)
        point_valid = ghi_clear >= MIN_CLEARSKY_GHI_WM2
        if not point_valid.any():
            continue

        kc = np.full(len(times_used), np.nan)
        kc[point_valid] = np.clip(ghi_actual[point_valid] / ghi_clear[point_valid], 0.0, 1.3)

        kept_lats.append(lat)
        kept_lons.append(lon)
        kept_series.append(kc)
        valid_mask = point_valid if valid_mask is None else (valid_mask & point_valid)

    if not kept_series:
        return [], [], [], times_used

    # Keep only timestamps valid (non-NaN) across every kept grid point, so
    # every point contributes to every lag pair with the same sample count.
    stacked = np.vstack(kept_series)
    common_valid = ~np.isnan(stacked).any(axis=0)
    if common_valid.sum() < 3:
        return [], [], [], times_used

    kept_series = [s[common_valid] for s in stacked]
    return kept_lats, kept_lons, kept_series, times_used[common_valid]


def _exponential_model(h, nugget, psill, rng):
    return nugget + psill * (1.0 - np.exp(-h / rng))


def _spherical_model(h, nugget, psill, rng):
    h = np.asarray(h, dtype=float)
    hs = np.clip(h / rng, 0.0, 1.0)
    return nugget + psill * (1.5 * hs - 0.5 * hs**3)


def _empirical_semivariogram(lats, lons, series, n_bins):
    n = len(lats)
    dists, sqdiffs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(lats[i], lons[i], lats[j], lons[j])
            diffs = (series[i] - series[j]) ** 2
            dists.extend([d] * len(diffs))
            sqdiffs.extend(diffs.tolist())

    dists = np.array(dists)
    sqdiffs = np.array(sqdiffs)
    if len(dists) == 0:
        return np.array([]), np.array([]), np.array([])

    max_lag = dists.max()
    edges = np.linspace(0, max_lag, n_bins + 1)
    h_mid, gamma, counts = [], [], []
    for k in range(n_bins):
        mask = (dists >= edges[k]) & (dists < edges[k + 1] if k < n_bins - 1
                                       else dists <= edges[k + 1])
        if mask.sum() >= MIN_PAIRS_PER_BIN:
            h_mid.append(0.5 * (edges[k] + edges[k + 1]))
            gamma.append(0.5 * np.mean(sqdiffs[mask]))
            counts.append(mask.sum())

    return np.array(h_mid), np.array(gamma), np.array(counts)


def _fit_variogram_model(h_mid, gamma, counts):
    """Fits both exponential and spherical models via NLS (scipy.optimize.
    curve_fit) and returns the one with lower weighted SSE. Returns None if
    neither converges."""
    nugget0 = max(float(gamma[0]), 1e-4)
    sill0 = max(float(gamma.max() - nugget0), 1e-4)
    range0 = float(h_mid[len(h_mid) // 2])
    bounds = ([0.0, 1e-6, 1e-3], [gamma.max(), gamma.max() * 2.0, h_mid.max() * 3.0])
    weights = 1.0 / np.sqrt(counts)

    best = None
    for name, model in (("exponential", _exponential_model), ("spherical", _spherical_model)):
        try:
            popt, _ = curve_fit(model, h_mid, gamma, p0=[nugget0, sill0, range0],
                                 sigma=weights, bounds=bounds, maxfev=5000)
            pred = model(h_mid, *popt)
            sse = float(np.sum(counts * (gamma - pred) ** 2))
            if best is None or sse < best[2]:
                best = (name, popt, sse)
        except Exception as e:
            print(f"    [!] {name} variogram fit failed: {e}")

    return best


def _solve_dynamic_r_max(model_name, popt, epsilon, max_h, min_h=0.5):
    model = _exponential_model if model_name == "exponential" else _spherical_model
    target = epsilon ** 2
    f = lambda h: model(h, *popt) - target

    if f(max_h) < 0:
        return max_h, "sill_below_tolerance_over_observed_range"
    if f(min_h) > 0:
        return min_h, "nugget_already_above_tolerance"

    root = brentq(f, min_h, max_h)
    return root, "solved"


def _plot_variogram(day_label, h_mid, gamma, counts, model_name, popt, r_max, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = _exponential_model if model_name == "exponential" else _spherical_model
    h_dense = np.linspace(0, h_mid.max(), 200)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(h_mid, gamma, s=np.clip(counts, 10, 200), color="steelblue",
                label="Empirical $\\gamma(h)$", zorder=3)
    plt.plot(h_dense, model(h_dense, *popt), color="firebrick",
              label=f"Fitted {model_name} model")
    plt.axvline(r_max, color="gray", linestyle="--", linewidth=1,
                label=f"$R_{{max}}$ = {r_max:.1f} km")
    plt.xlabel("Lag distance h (km)")
    plt.ylabel(r"Semivariance $\gamma(h)$ (Kc$^2$)")
    plt.title(f"Clear-sky index semivariogram - {day_label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = re.sub(r"[^\w\-]+", "_", day_label)
    plt.savefig(os.path.join(out_dir, f"variogram_{safe_name}.png"), dpi=120)
    plt.close()


def compute_stage_r_max(ds, coords, stage_date, day_label,
                         epsilon=SEMIVARIOGRAM_EPSILON):
    """Top-level orchestrator: returns (r_max_km, diagnostics_dict).
    Always returns a usable r_max_km - falls back to FALLBACK_R_MAX_KM with
    an explanatory status on any failure, so a bad fit for one stage never
    stops the run."""
    diagnostics = {
        "epsilon": epsilon,
        "status": None,
        "model": None,
        "n_grid_points": 0,
        "n_lag_bins": 0,
    }

    try:
        bbox = _route_bbox(coords, BBOX_BUFFER_KM)
        grid_points = _era5_grid_points_in_bbox(ds, bbox)
        diagnostics["n_grid_points"] = len(grid_points)

        if len(grid_points) < MIN_GRID_POINTS_FOR_FIT:
            diagnostics["status"] = "too_few_grid_points_fallback"
            print(f"  [i] {day_label}: only {len(grid_points)} ERA5 grid points in bbox "
                  f"(< {MIN_GRID_POINTS_FOR_FIT}); using fallback R_max="
                  f"{FALLBACK_R_MAX_KM:.1f} km.")
            return FALLBACK_R_MAX_KM, diagnostics

        lats, lons, series, times_used = _build_kc_field(ds, grid_points, stage_date)
        if len(lats) < MIN_GRID_POINTS_FOR_FIT:
            diagnostics["status"] = "too_few_valid_kc_samples_fallback"
            print(f"  [i] {day_label}: not enough valid Kc samples after sun-angle "
                  f"filtering; using fallback R_max={FALLBACK_R_MAX_KM:.1f} km.")
            return FALLBACK_R_MAX_KM, diagnostics

        h_mid, gamma, counts = _empirical_semivariogram(lats, lons, series, N_LAG_BINS)
        diagnostics["n_lag_bins"] = len(h_mid)
        if len(h_mid) < 3:
            diagnostics["status"] = "too_few_lag_bins_fallback"
            print(f"  [i] {day_label}: not enough populated lag bins; using fallback "
                  f"R_max={FALLBACK_R_MAX_KM:.1f} km.")
            return FALLBACK_R_MAX_KM, diagnostics

        fit = _fit_variogram_model(h_mid, gamma, counts)
        if fit is None:
            diagnostics["status"] = "fit_failed_fallback"
            print(f"  [i] {day_label}: variogram model fit did not converge; using "
                  f"fallback R_max={FALLBACK_R_MAX_KM:.1f} km.")
            return FALLBACK_R_MAX_KM, diagnostics

        model_name, popt, sse = fit
        r_max_raw, solve_status = _solve_dynamic_r_max(model_name, popt, epsilon, h_mid.max())
        r_max = float(np.clip(r_max_raw, MIN_R_MAX_KM, MAX_R_MAX_KM))

        diagnostics.update({
            "status": solve_status,
            "model": model_name,
            "nugget": float(popt[0]),
            "partial_sill": float(popt[1]),
            "range_km": float(popt[2]),
            "sse": sse,
            "r_max_raw_km": float(r_max_raw),
            "r_max_clamped_km": r_max,
        })

        print(f"  [i] {day_label}: {model_name} variogram fit "
              f"(nugget={popt[0]:.4f}, sill={popt[1]:.4f}, range={popt[2]:.1f} km) "
              f"-> R_max={r_max:.1f} km ({solve_status}, n_grid_points={len(lats)})")

        if SAVE_VARIOGRAM_PLOTS:
            try:
                _plot_variogram(day_label, h_mid, gamma, counts, model_name, popt,
                                 r_max, VARIOGRAM_PLOT_DIR)
            except Exception as e:
                print(f"    [!] variogram plot failed for {day_label}: {e}")

        return r_max, diagnostics

    except Exception as e:
        diagnostics["status"] = f"error_fallback: {e}"
        print(f"  [!] {day_label}: semivariogram R_max computation failed ({e}); "
              f"using fallback R_max={FALLBACK_R_MAX_KM:.1f} km.")
        return FALLBACK_R_MAX_KM, diagnostics


# --- 4. ROUTE DISCRETIZATION ---
def discretize_route(coords, R_max, velocity_kmh):
    discretized = []
    current_center = coords[0]
    dist_since_last = 0.0
    cumulative_dist = 0.0

    discretized.append({"lat": current_center[0], "lon": current_center[1], "cum_dist": 0.0, "eta_hours": 0.0})

    for i in range(1, len(coords)):
        step_dist = haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        dist_since_last += step_dist
        cumulative_dist += step_dist

        if dist_since_last >= (2 * R_max):
            current_center = coords[i]
            discretized.append({
                "lat": current_center[0],
                "lon": current_center[1],
                "cum_dist": cumulative_dist,
                "eta_hours": cumulative_dist / velocity_kmh,
            })
            dist_since_last = 0.0

    return discretized


# --- 5. MAIN EXECUTION ---
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kml_file = os.path.join(script_dir, KML_FILENAME_HINT)

    if not os.path.exists(kml_file):
        print(f"KML file not found: {kml_file}")
        return

    day_route_segments = parse_kml_day_folders(kml_file)
    if not day_route_segments:
        print("No main routes found in KML.")
        return

    # Keyed by real day label, in true chronological day-number order.
    # Day 3 (Full Blind) is simply absent - no route to stitch/discretize.
    stage_groups = stitch_day_segments(day_route_segments)

    # Dates are derived from each folder's own parsed day number, NOT from
    # a sequential enumerate() index - that's what silently shifted every
    # date from Day 4 onward when Day 3 (no route) was dropped in the old
    # code path. See module docstring, v1.2 CHANGE.
    day_numbers = {day_label: parse_day_number(day_label) for day_label in stage_groups}
    stage_dates = {
        day_label: RACE_START_DATE + timedelta(days=day_numbers[day_label] - 1)
        for day_label in stage_groups
    }

    print(f"Found {len(stage_groups)} stage(s) with a defined route "
          f"(of {len(day_route_segments)} day folder(s) parsed): "
          f"{list(stage_groups.keys())}\n")

    weather_grid_ds = download_era5_weather_grid(list(stage_dates.values()))
    sast_tz = ZoneInfo("Africa/Johannesburg")

    json_output_data = {}

    for day_label, coords in stage_groups.items():
        if len(coords) < 2:
            continue

        day_number = day_numbers[day_label]
        stage_date = stage_dates[day_label]

        print(f"Processing stitched stage: {day_label} ({stage_date.isoformat()})...")

        total_dist = sum(
            haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
            for i in range(1, len(coords))
        )
        velocity = total_dist / DEFAULT_STAGE_HOURS

        r_max_km, variogram_diagnostics = compute_stage_r_max(
            weather_grid_ds, coords, stage_date, day_label
        )
        waypoints = discretize_route(coords, r_max_km, velocity)

        # Apply conditional logic for Day 1 (9 AM) vs Day 2+ (8 AM), keyed
        # off the actual parsed day number rather than loop position.
        stage_start_time = STAGE_START_TIME_DAY1 if day_number == 1 else STAGE_START_TIME_OTHER
        stage_start_sast = datetime.combine(stage_date, stage_start_time, tzinfo=sast_tz)

        stage_waypoints_data = []

        for wp in waypoints:
            arrival_sast = stage_start_sast + timedelta(hours=wp["eta_hours"])
            arrival_utc = arrival_sast.astimezone(ZoneInfo("UTC"))

            ghi, dni, wind_spd, wind_dir = fetch_local_weather(weather_grid_ds, wp["lat"], wp["lon"], arrival_utc)

            stage_waypoints_data.append({
                "cum_dist_km": round(wp["cum_dist"], 2),
                "eta_hours": round(wp["eta_hours"], 2),
                "latitude": round(wp["lat"], 5),
                "longitude": round(wp["lon"], 5),
                "ghi_wm2": round(ghi, 2),
                "dni_wm2": round(dni, 2),
                "wind_speed_ms": round(wind_spd, 2),
                "wind_direction_deg": round(wind_dir, 2)
            })

        # The last waypoint's ETA represents the time taken based on the velocity assumption
        time_taken = waypoints[-1]["eta_hours"] if waypoints else 0.0

        json_output_data[day_label] = {
            "day_number": day_number,
            "date": stage_date.isoformat(),
            "start_time_sast": stage_start_time.strftime("%H:%M:%S"),
            "distance_covered_km": round(total_dist, 2),
            "time_taken_hours": round(time_taken, 2),
            "r_max_km": round(r_max_km, 2),
            "r_max_source": "semivariogram" if variogram_diagnostics.get("status") == "solved"
                             else "fallback",
            "semivariogram_diagnostics": variogram_diagnostics,
            "waypoints": stage_waypoints_data
        }

    # Output to JSON
    json_filename = "race_weather_data_2026.json"
    with open(json_filename, "w") as json_file:
        json.dump(json_output_data, json_file, indent=4)

    print(f"\nWeather and route data successfully exported to {json_filename}")

if __name__ == "__main__":
    main()'''

"""
cdsapiQuerySemivariance.py
---------------------------------------------------------------------------
Parses the Sasol Solar Challenge KML route, queries Solcast (Live and
Forecast plan) for GHI / DNI / 10 m wind for each stage, discretizes each
stage into waypoints, looks up local weather at each waypoint's arrival
time, and exports race_weather_data_2026.json.

v1.3 CHANGE (this revision): weather source switched from cdsapi/ERA5 to
the Solcast "Live and Forecast" API, and all time/space interpolation now
goes through solar_table.SolarIrradiance instead of hand-rolled xarray
.sel()/.interp() calls. solar_table.py must sit alongside this script.

  Why this isn't just a drop-in source swap: ERA5 gave us a free, dense,
  continuous grid to pull as many spatial samples as we wanted for the
  semivariogram at no extra cost. Solcast is metered per UNIQUE (lat, lon)
  LOCATION per day (Live and Forecast Max: 1500 locations/day, 1000
  req/min - see plan screenshot) - the number of timesteps or repeat
  queries against an already-queried location doesn't count further. So
  the query strategy changed shape, not just the transport:

  1. Per stage, `_generate_stage_sample_points()` picks a small, deliberate
     set of locations (VARIOGRAM_SAMPLE_BUDGET, default 15) - ~60% spaced
     along the route (by cumulative distance) and the rest jittered 5-20 km
     laterally off those, so the semivariogram gets pairs across a spread
     of distances rather than the ~1D distance distribution a purely
     along-route sample would give.
  2. Each of those locations is queried ONCE (`fetch_stage_solar_irradiance`),
     for the full remaining stage window at PT30M resolution. Requesting a
     longer time series or finer period costs nothing extra against the
     location cap, so there's no reason to skimp on it.
  3. Those SAME queried points serve double duty: their GHI series feeds
     the clear-sky-index semivariogram (as before, -> dynamic R_max), AND
     the raw records are handed straight to SolarIrradiance, which then
     answers every waypoint's weather lookup via nearest-queried-location +
     time interpolation - zero extra Solcast calls per waypoint. This
     mirrors Plan v3 section 6.1's "equal-area circle" scheme: query once
     at circle centers, every route point maps to its nearest circle.
  4. A module-level location budget tracker (`_solcast_budget_check`)
     refuses to exceed SOLCAST_LOCATION_BUDGET (set below the 1500/day cap
     with headroom) across the whole run, rather than silently overrunning
     the account's daily quota partway through a multi-stage run.

  Wind is now much simpler: Solcast returns wind_speed_10m/wind_direction_10m
  directly, so the old u10/v10 -> speed/direction atan2 conversion in
  fetch_local_weather is gone entirely.

  KML day-folder parsing (v1.2) is untouched by this revision.

Key assumptions / decisions (see chat response for full rationale):
  - Solcast's `forecast/radiation_and_weather` endpoint is used (not
    `live`), since every stage date this script is run for is in the
    future relative to "now". `live` is wired up (`SOLCAST_LIVE_ENDPOINT`)
    but unused by default - it exists for the intra-day nowcast refresh use
    case Plan v3 mentions, wiring that up is a separate task.
  - Solcast's documented forecast window is 1-168 hours ahead; a stage
    further out than that just gets clamped to 168h with a printed
    warning - Solcast won't have forecast data for it yet regardless of
    what this script requests, so re-running closer to the stage date is
    the actual fix, not something code can paper over.
  - `SOLCAST_API_KEY` is read from the environment - never hardcode it.
  - Clear-sky GHI (for the semivariogram's Kc normalization) is still
    computed independently via pvlib/Haurwitz, same as v1.1/v1.2 - that's
    an astronomical calculation, not tied to the weather data source.
  - epsilon default is 0.1; R_max still clamped to [MIN_R_MAX_KM,
    MAX_R_MAX_KM]; both unchanged from v1.2.
---------------------------------------------------------------------------
"""

import json
import math
import os
import re
import statistics
import warnings
import xml.etree.ElementTree as ET
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import curve_fit, brentq

from solar_table import SolarIrradiance

try:
    import pvlib
    _PVLIB_AVAILABLE = True
except ImportError:
    _PVLIB_AVAILABLE = False
    warnings.warn(
        "pvlib not installed - falling back to an approximate Haurwitz "
        "clear-sky model for the semivariogram's Kc normalization. "
        "`pip install pvlib` to match Model/core/solar.py's Ineichen model."
    )

# ---------------------------------------------------------------------------
# DEFAULTS - edit these instead of being prompted at runtime.
# ---------------------------------------------------------------------------
KML_FILENAME_HINT = "2026 Sasol Solar Challenge Route (Publish).kml"

RACE_START_DATE = date(2025, 9, 10)
# NOTE: the KML hint above says "2026 ... Route" but this date is 2025-09-10.
# Flagging rather than silently changing it - confirm which is right before
# this goes near real stage dates; if it's a typo it should be 2026-09-10.

STAGE_START_TIME_DAY1 = time(9, 0, 0)    # Day 1 start, SAST
STAGE_START_TIME_OTHER = time(8, 0, 0)   # Days 2+ start, SAST
DEFAULT_STAGE_HOURS = 8.0                # used to derive a default cruising speed per stage

# --- Solcast ("Live and Forecast Max") ---------------------------------
SOLCAST_API_BASE = "https://api.solcast.com.au"
SOLCAST_API_KEY = os.environ.get("SOLCAST_API_KEY")           # never hardcode
SOLCAST_FORECAST_ENDPOINT = "forecast"                        # used by default
SOLCAST_LIVE_ENDPOINT = "live"                                # available, unused by default
SOLCAST_OUTPUT_PARAMETERS = ("ghi", "dni", "wind_speed_10m", "wind_direction_10m")
SOLCAST_PERIOD = "PT30M"                                       # averaging interval
SOLCAST_MAX_FORECAST_HOURS = 168                               # documented API ceiling (7 days)
SOLCAST_REQUEST_TIMEOUT_S = 30
SOLCAST_LOCATION_BUDGET = 1400          # headroom under the 1500/day plan cap
_solcast_queried_locations = set()      # unique (lat,lon) queried so far this run

# --- Semivariogram-derived dynamic R_max (section 3.5) ---------------------
FALLBACK_R_MAX_KM = 10.0         # used only if a stage's variogram can't be fit
SEMIVARIOGRAM_EPSILON = 0.1      # tolerance on sqrt(gamma(h)); see module docstring
MIN_R_MAX_KM = 5.0               # sanity floor (avoid pathologically small spacing)
MAX_R_MAX_KM = 40.0              # sanity ceiling (avoid pathologically sparse waypoints)
VARIOGRAM_SAMPLE_BUDGET = 15     # Solcast locations queried per stage (see module docstring)
N_LAG_BINS = 10                  # distance bins for the empirical variogram (fewer points now)
MIN_PAIRS_PER_BIN = 3            # bins with fewer pairs than this are dropped
MIN_GRID_POINTS_FOR_FIT = 6      # below this, skip fitting and use the fallback
MIN_CLEARSKY_GHI_WM2 = 50.0      # drop low-sun-angle samples (unstable Kc ratio)
SAVE_VARIOGRAM_PLOTS = True      # diagnostic PNG per stage, for QA / debugging
VARIOGRAM_PLOT_DIR = "variogram_diagnostics"

# Matches the day number out of a KML Folder label like "Day 4", "Day 2 Half
# Blind", "Day 3 Full Blind", "Day 1 " (trailing space), etc. Deliberately
# does NOT match "Stage N" - that word only ever appears on the individual
# route-segment Placemarks inside a day folder, not on the folder itself,
# but staying folder-scoped (rather than falling back to a looser regex)
# avoids resurrecting the original Stage/Day collision bug.
_DAY_FOLDER_PATTERN = re.compile(r"day\s*0*(\d+)", re.IGNORECASE)


# --- 1. KML PARSING, PER DAY FOLDER --- (unchanged from v1.2)
def parse_kml_day_folders(kml_path):
    """Returns an ordered dict {day_label: [segment_coords, ...]} where
    day_label is the KML Folder's own name (e.g. 'Day 4', 'Day 2 Half
    Blind') and each segment_coords is a list of (lat, lon) tuples for one
    non-loop LineString placemark inside that folder, in document order.

    Walking Folders directly (rather than scanning every Placemark in the
    document and grouping by a regex on its name) is the fix for the
    Stage-N/Day-N collision described in the module docstring: the route
    LineStrings are only ever named "Stage 1"/"Stage 2", so grouping by
    name-regex silently merged every day's Stage 1 into one bucket and
    every day's Stage 2 into another. Grouping by the folder they actually
    live in has no such ambiguity.

    A day folder with no LineString placemarks (Day 3, "Full Blind" - only
    start/finish Points, no prescribed route) is simply omitted from the
    result. That's expected, not an error: Full Blind stages have no fixed
    route to discretize or fetch weather along.
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    day_routes = {}
    # Day folders are nested: Document > Folder "My Places" > Folder "Day N ..."
    for day_folder in root.findall(".//kml:Folder/kml:Folder", ns):
        name_elem = day_folder.find("kml:name", ns)
        day_label = name_elem.text.strip() if name_elem is not None else "Unknown"

        segments = []
        for placemark in day_folder.findall("kml:Placemark", ns):
            pname_elem = placemark.find("kml:name", ns)
            pname = pname_elem.text.strip() if pname_elem is not None else ""
            if "loop" in pname.lower():
                continue

            linestring = placemark.find("kml:LineString/kml:coordinates", ns)
            if linestring is None or not linestring.text:
                continue

            coords = []
            for pt in linestring.text.strip().split():
                lon, lat, *_ = map(float, pt.split(","))
                coords.append((lat, lon))
            segments.append(coords)

        if segments:
            day_routes[day_label] = segments
        else:
            print(f"  [i] {day_label}: no LineString route segments found "
                  f"(expected for a Full Blind stage) - skipping.")

    return day_routes


def parse_day_number(day_label):
    """Extracts the integer day number from a folder label like 'Day 4' or
    'Day 2 Half Blind'. Raises ValueError if the label doesn't contain a
    recognizable 'Day N' - fails loudly rather than silently mis-dating a
    stage, since date alignment depends entirely on this."""
    m = _DAY_FOLDER_PATTERN.search(day_label)
    if not m:
        raise ValueError(
            f"Could not parse a day number out of folder label {day_label!r}; "
            f"expected something matching 'Day <N>'."
        )
    return int(m.group(1))


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _destination_point(lat, lon, bearing_deg, distance_km):
    """Standard great-circle destination point formula. Used only to jitter
    variogram sample points laterally off the route (see module docstring);
    not a route/discretization primitive."""
    R = 6371.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    d_r = distance_km / R
    lat2 = math.asin(math.sin(lat1) * math.cos(d_r) + math.cos(lat1) * math.sin(d_r) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d_r) * math.cos(lat1),
                              math.cos(d_r) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


# --- 2. STAGE STITCHING --- (unchanged from v1.2)
def stitch_day_segments(day_routes):
    """Takes {day_label: [segment_coords, ...]} (already correctly grouped
    by parse_kml_day_folders) and concatenates each day's segments into one
    continuous coordinate list, splicing out a duplicated junction point
    when consecutive segments meet at (near) the same spot. Returned dict
    preserves day order and is keyed the same way as the input."""
    stitched = {}
    for day_label, segments in day_routes.items():
        combined = list(segments[0])
        for seg in segments[1:]:
            if not seg:
                continue
            if combined and haversine(*combined[-1], *seg[0]) < 0.05:
                combined.extend(seg[1:])
            else:
                combined.extend(seg)
        stitched[day_label] = combined

    # Sort by the actual parsed day number so downstream processing (and
    # the Solcast query order) proceeds in true chronological stage order,
    # even though the KML folders already happen to be in this order.
    return dict(sorted(stitched.items(), key=lambda kv: parse_day_number(kv[0])))


# --- 3. SOLCAST QUERYING ---
def _solcast_budget_check(lat, lon):
    """Tracks unique (lat, lon) locations queried this run against
    SOLCAST_LOCATION_BUDGET. Raises rather than silently blowing through
    the account's daily location cap partway through a multi-stage run -
    better to fail loudly on stage 6 of 8 than to quietly get truncated/
    throttled data for the rest of the run."""
    key = (round(lat, 4), round(lon, 4))
    if key not in _solcast_queried_locations:
        if len(_solcast_queried_locations) >= SOLCAST_LOCATION_BUDGET:
            raise RuntimeError(
                f"Solcast location budget ({SOLCAST_LOCATION_BUDGET}/day) would be "
                f"exceeded by querying {key}. Lower VARIOGRAM_SAMPLE_BUDGET, raise "
                f"SOLCAST_LOCATION_BUDGET if your plan has more headroom than "
                f"assumed, or split the run across multiple days."
            )
        _solcast_queried_locations.add(key)
    return key


def _fetch_solcast_records(lat, lon, hours, endpoint=SOLCAST_FORECAST_ENDPOINT,
                            period=SOLCAST_PERIOD, output_parameters=SOLCAST_OUTPUT_PARAMETERS):
    """One HTTP call to Solcast's {endpoint}/radiation_and_weather for a
    single location, returning the raw list of per-period records (each a
    dict with 'period_end' plus the requested output parameters) exactly
    as Solcast returns them - this shape is what SolarIrradiance expects.
    """
    if not SOLCAST_API_KEY:
        raise RuntimeError(
            "SOLCAST_API_KEY environment variable is not set. Export it before "
            "running this script - never hardcode a Solcast key in source."
        )

    url = f"{SOLCAST_API_BASE}/data/{endpoint}/radiation_and_weather"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hours": max(1, min(hours, SOLCAST_MAX_FORECAST_HOURS)),
        "period": period,
        "output_parameters": ",".join(output_parameters),
        "format": "json",
    }
    headers = {"Authorization": f"Bearer {SOLCAST_API_KEY}", "Accept": "application/json"}

    response = requests.get(url, params=params, headers=headers, timeout=SOLCAST_REQUEST_TIMEOUT_S)
    response.raise_for_status()
    payload = response.json()

    # forecast/live responses key their series differently ("forecasts" vs
    # "estimated_actuals") - accept either rather than hardcoding one.
    records = payload.get("forecasts")
    if records is None:
        records = payload.get("estimated_actuals")
    if records is None:
        raise RuntimeError(
            f"Unexpected Solcast response shape for {endpoint} at "
            f"({lat:.4f},{lon:.4f}): top-level keys were {list(payload.keys())}"
        )
    return records


def _solcast_hours_needed(stage_date, stage_start_time, sast_tz, buffer_hours=4.0):
    """How many hours ahead (from right now) to request so the whole stage
    window is covered, clamped to Solcast's documented 168h ceiling. Warns
    (doesn't fail) if the stage genuinely needs more - that's an inherent
    forecast-horizon limit, not something this script can work around; the
    real fix is re-running it closer to the stage date."""
    now_utc = datetime.now(ZoneInfo("UTC"))
    stage_start_sast = datetime.combine(stage_date, stage_start_time, tzinfo=sast_tz)
    stage_end_utc = (stage_start_sast + timedelta(hours=DEFAULT_STAGE_HOURS + buffer_hours)) \
        .astimezone(ZoneInfo("UTC"))
    delta_hours = (stage_end_utc - now_utc).total_seconds() / 3600.0

    if delta_hours > SOLCAST_MAX_FORECAST_HOURS:
        print(f"  [!] stage end is {delta_hours:.0f}h from now, beyond Solcast's "
              f"{SOLCAST_MAX_FORECAST_HOURS}h forecast horizon - requesting the max "
              f"({SOLCAST_MAX_FORECAST_HOURS}h). Forecast quality/coverage for the "
              f"tail of this stage will improve on a re-run closer to the date.")
    return max(1, min(math.ceil(delta_hours), SOLCAST_MAX_FORECAST_HOURS))


def _generate_stage_sample_points(coords, n_points, seed=None):
    """Chooses up to n_points (lat, lon) Solcast query locations for this
    stage: ~60% spaced evenly along the route (by cumulative distance -
    these double as candidate weather-lookup anchors for waypoints later),
    and the remainder placed as small lateral jitters (5-20 km, roughly
    the expected decorrelation scale) off those anchors, so the
    semivariogram sees pairs across a spread of distances rather than only
    the ~1D distances a purely along-route sample would give. Near-
    duplicate points are dropped so budget isn't wasted on them."""
    n_points = max(2, n_points)
    n_along = max(2, math.ceil(n_points * 0.6))
    n_jitter = max(0, n_points - n_along)

    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1]))
    total = cum[-1] if cum[-1] > 0 else 1.0

    rng = np.random.default_rng(seed)
    along_points = []
    for k in range(n_along):
        target_d = total * k / max(n_along - 1, 1)
        idx = min(range(len(cum)), key=lambda i: abs(cum[i] - target_d))
        along_points.append(coords[idx])

    jitter_points = []
    for k in range(n_jitter):
        base = along_points[k % len(along_points)]
        bearing = float(rng.uniform(0, 360))
        dist = float(rng.uniform(5.0, 20.0))
        jitter_points.append(_destination_point(base[0], base[1], bearing, dist))

    seen, deduped = set(), []
    for p in along_points + jitter_points:
        key = (round(p[0], 3), round(p[1], 3))
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def fetch_stage_solar_irradiance(coords, stage_date, stage_start_time, sast_tz,
                                  n_sample_points=VARIOGRAM_SAMPLE_BUDGET):
    """Queries Solcast once per chosen sample location for this stage and
    returns (solar_irr, records_by_point):
      - solar_irr: a SolarIrradiance built from every successfully-queried
        location, used for all downstream weather lookups (waypoints).
      - records_by_point: [(lat, lon, records), ...] for the same
        locations, used to build the semivariogram's Kc field.
    A location whose query fails is skipped (logged), not fatal - the
    stage still proceeds with whatever locations did succeed, subject to
    the usual MIN_GRID_POINTS_FOR_FIT fallback downstream."""
    hours = _solcast_hours_needed(stage_date, stage_start_time, sast_tz)
    sample_points = _generate_stage_sample_points(coords, n_sample_points)

    records_by_point = []
    solcast_entries = []
    for lat, lon in sample_points:
        _solcast_budget_check(lat, lon)
        try:
            records = _fetch_solcast_records(lat, lon, hours=hours)
        except Exception as e:
            print(f"  [!] Solcast query failed at ({lat:.4f},{lon:.4f}): {e}")
            continue
        if not records:
            print(f"  [!] Solcast returned no records at ({lat:.4f},{lon:.4f}); skipping.")
            continue
        records_by_point.append((lat, lon, records))
        solcast_entries.append({"lat": lat, "lon": lon, "data": records})

    if not solcast_entries:
        raise RuntimeError("No Solcast data could be retrieved for this stage "
                            "(every sampled location failed) - check SOLCAST_API_KEY "
                            "and connectivity before retrying.")

    solar_irr = SolarIrradiance(solcast_entries, time_key="period_end", interval=SOLCAST_PERIOD)
    return solar_irr, records_by_point


def fetch_local_weather(solar_irr, lat, lon, arrival_time_utc):
    """Looks up GHI/DNI/wind at (lat, lon, arrival_time_utc) via
    SolarIrradiance - nearest queried Solcast location, linearly
    interpolated in time. Replaces the old xarray .sel()/.interp() version;
    no more manual u10/v10 -> speed/direction math since Solcast already
    reports wind_speed_10m/wind_direction_10m directly."""
    try:
        proxy = solar_irr[(lat, lon), arrival_time_utc.isoformat()]
        ghi, dni, wind_speed, wind_dir = proxy.data(
            ["ghi", "dni", "wind_speed_10m", "wind_direction_10m"]
        )
        return (float(ghi or 0.0), float(dni or 0.0),
                float(wind_speed or 0.0), float(wind_dir or 0.0))
    except Exception as e:
        print(f"  [!] weather lookup failed at ({lat:.3f},{lon:.3f}) {arrival_time_utc}: {e}")
        return 0.0, 0.0, 0.0, 0.0


# --- 3.5 SPATIAL SEMIVARIOGRAM -> DYNAMIC R_MAX ---
#
# For each stage: take the Solcast-queried sample locations, convert GHI ->
# clear-sky index Kc (isolating cloud variability from the solar-elevation
# gradient that would otherwise dominate GHI(lat,lon)), build an empirical
# semivariogram gamma(h) of Kc over spatial lag h, fit an exponential/
# spherical model via NLS, and solve for the largest radius R_max where
# sqrt(gamma(R_max)) <= epsilon. That R_max becomes this stage's waypoint-
# spacing radius (replacing the fixed R_MAX_KM).

def _haurwitz_clearsky_ghi_wm2(lat, lon, time_utc):
    """Fallback clear-sky model (used only if pvlib is unavailable).

    Standard solar-position + Haurwitz clear-sky formula:
        GHI_cs = 1098 * sin(elev) * exp(-0.059 / sin(elev))   for elev > 0
    This is intentionally simple (no turbidity/aerosol data) - it exists so
    the semivariogram feature degrades gracefully rather than hard-failing
    when pvlib isn't installed. Prefer pvlib (Ineichen) when available, to
    stay consistent with Model/core/solar.py's PVLibClearSkyProvider.
    """
    day_of_year = time_utc.timetuple().tm_yday
    decl = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + day_of_year) / 365.0)))
    hour_utc = time_utc.hour + time_utc.minute / 60.0 + time_utc.second / 3600.0
    solar_time = hour_utc + lon / 15.0  # rough longitude-only correction (no EoT)
    hour_angle = math.radians(15.0 * (solar_time - 12.0))
    lat_rad = math.radians(lat)

    sin_elev = (math.sin(lat_rad) * math.sin(decl)
                + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle))
    if sin_elev <= 0.01:
        return 0.0
    return max(0.0, 1098.0 * sin_elev * math.exp(-0.059 / sin_elev))


def _pvlib_clearsky_ghi_series_wm2(lat, lon, times_utc):
    """One pvlib call per grid point covering all its timestamps (batched,
    not per-point-per-time, to keep this cheap)."""
    loc = pvlib.location.Location(lat, lon, tz="UTC")
    times_idx = pd.DatetimeIndex(times_utc)
    if times_idx.tz is None:
        times_idx = times_idx.tz_localize("UTC")
    cs = loc.get_clearsky(times_idx, model="ineichen")
    return cs["ghi"].values


def _clearsky_ghi_series_wm2(lat, lon, times_utc):
    if _PVLIB_AVAILABLE:
        return _pvlib_clearsky_ghi_series_wm2(lat, lon, times_utc)
    return np.array([_haurwitz_clearsky_ghi_wm2(lat, lon, t) for t in times_utc])


def _build_kc_field_from_records(records_by_point, stage_date):
    """Converts each sample location's raw Solcast records into a
    clear-sky index (Kc) series, keeps only timestamps that (a) fall on
    stage_date, (b) have a valid (non-low-sun) clear-sky reference at every
    kept location, then returns (lats, lons, kc_series, times_used) in the
    same shape the semivariogram fitting expects.

    Defensive against tiny per-location clock skew: intersects on exact
    matching timestamps across locations rather than assuming identical
    indices, since each location was queried in a separate HTTP call.
    """
    per_point_kc = {}   # (lat, lon) -> {iso_time: kc}
    for lat, lon, records in records_by_point:
        day_records = [r for r in records
                        if datetime.fromisoformat(r["period_end"]).date() == stage_date]
        if not day_records:
            continue

        times = [datetime.fromisoformat(r["period_end"]) for r in day_records]
        ghi_actual = np.array([float(r.get("ghi") or 0.0) for r in day_records])
        ghi_clear = _clearsky_ghi_series_wm2(lat, lon, times)

        valid = ghi_clear >= MIN_CLEARSKY_GHI_WM2
        if not valid.any():
            continue

        kc_by_time = {}
        for t, is_valid, actual, clear in zip(times, valid, ghi_actual, ghi_clear):
            if is_valid:
                kc_by_time[t.isoformat()] = float(np.clip(actual / clear, 0.0, 1.3))
        if kc_by_time:
            per_point_kc[(lat, lon)] = kc_by_time

    if len(per_point_kc) < MIN_GRID_POINTS_FOR_FIT:
        return [], [], [], []

    common_times = None
    for series in per_point_kc.values():
        keys = set(series.keys())
        common_times = keys if common_times is None else (common_times & keys)
    common_times = sorted(common_times or [])

    if len(common_times) < 3:
        return [], [], [], []

    lats, lons, kc_series = [], [], []
    for (lat, lon), series in per_point_kc.items():
        lats.append(lat)
        lons.append(lon)
        kc_series.append(np.array([series[t] for t in common_times]))

    return lats, lons, kc_series, common_times


def _exponential_model(h, nugget, psill, rng):
    return nugget + psill * (1.0 - np.exp(-h / rng))


def _spherical_model(h, nugget, psill, rng):
    h = np.asarray(h, dtype=float)
    hs = np.clip(h / rng, 0.0, 1.0)
    return nugget + psill * (1.5 * hs - 0.5 * hs**3)


def _empirical_semivariogram(lats, lons, series, n_bins):
    n = len(lats)
    dists, sqdiffs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(lats[i], lons[i], lats[j], lons[j])
            diffs = (series[i] - series[j]) ** 2
            dists.extend([d] * len(diffs))
            sqdiffs.extend(diffs.tolist())

    dists = np.array(dists)
    sqdiffs = np.array(sqdiffs)
    if len(dists) == 0:
        return np.array([]), np.array([]), np.array([])

    max_lag = dists.max()
    edges = np.linspace(0, max_lag, n_bins + 1)
    h_mid, gamma, counts = [], [], []
    for k in range(n_bins):
        mask = (dists >= edges[k]) & (dists < edges[k + 1] if k < n_bins - 1
                                       else dists <= edges[k + 1])
        if mask.sum() >= MIN_PAIRS_PER_BIN:
            h_mid.append(0.5 * (edges[k] + edges[k + 1]))
            gamma.append(0.5 * np.mean(sqdiffs[mask]))
            counts.append(mask.sum())

    return np.array(h_mid), np.array(gamma), np.array(counts)


def _fit_variogram_model(h_mid, gamma, counts):
    """Fits both exponential and spherical models via NLS (scipy.optimize.
    curve_fit) and returns the one with lower weighted SSE. Returns None if
    neither converges."""
    nugget0 = max(float(gamma[0]), 1e-4)
    sill0 = max(float(gamma.max() - nugget0), 1e-4)
    range0 = float(h_mid[len(h_mid) // 2])
    bounds = ([0.0, 1e-6, 1e-3], [gamma.max(), gamma.max() * 2.0, h_mid.max() * 3.0])
    weights = 1.0 / np.sqrt(counts)

    best = None
    for name, model in (("exponential", _exponential_model), ("spherical", _spherical_model)):
        try:
            popt, _ = curve_fit(model, h_mid, gamma, p0=[nugget0, sill0, range0],
                                 sigma=weights, bounds=bounds, maxfev=5000)
            pred = model(h_mid, *popt)
            sse = float(np.sum(counts * (gamma - pred) ** 2))
            if best is None or sse < best[2]:
                best = (name, popt, sse)
        except Exception as e:
            print(f"    [!] {name} variogram fit failed: {e}")

    return best


def _solve_dynamic_r_max(model_name, popt, epsilon, max_h, min_h=0.5):
    model = _exponential_model if model_name == "exponential" else _spherical_model
    target = epsilon ** 2
    f = lambda h: model(h, *popt) - target

    if f(max_h) < 0:
        return max_h, "sill_below_tolerance_over_observed_range"
    if f(min_h) > 0:
        return min_h, "nugget_already_above_tolerance"

    root = brentq(f, min_h, max_h)
    return root, "solved"


def _plot_variogram(day_label, h_mid, gamma, counts, model_name, popt, r_max, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = _exponential_model if model_name == "exponential" else _spherical_model
    h_dense = np.linspace(0, h_mid.max(), 200)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(h_mid, gamma, s=np.clip(counts, 10, 200), color="steelblue",
                label="Empirical $\\gamma(h)$", zorder=3)
    plt.plot(h_dense, model(h_dense, *popt), color="firebrick",
              label=f"Fitted {model_name} model")
    plt.axvline(r_max, color="gray", linestyle="--", linewidth=1,
                label=f"$R_{{max}}$ = {r_max:.1f} km")
    plt.xlabel("Lag distance h (km)")
    plt.ylabel(r"Semivariance $\gamma(h)$ (Kc$^2$)")
    plt.title(f"Clear-sky index semivariogram - {day_label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = re.sub(r"[^\w\-]+", "_", day_label)
    plt.savefig(os.path.join(out_dir, f"variogram_{safe_name}.png"), dpi=120)
    plt.close()


def compute_stage_r_max(records_by_point, stage_date, day_label,
                         epsilon=SEMIVARIOGRAM_EPSILON):
    """Top-level orchestrator: returns (r_max_km, diagnostics_dict).
    Always returns a usable r_max_km - falls back to FALLBACK_R_MAX_KM with
    an explanatory status on any failure, so a bad fit for one stage never
    stops the run."""
    diagnostics = {
        "epsilon": epsilon,
        "status": None,
        "model": None,
        "n_sample_points": len(records_by_point),
        "n_lag_bins": 0,
    }

    try:
        if len(records_by_point) < MIN_GRID_POINTS_FOR_FIT:
            diagnostics["status"] = "too_few_sample_points_fallback"
            print(f"  [i] {day_label}: only {len(records_by_point)} Solcast sample "
                  f"points (< {MIN_GRID_POINTS_FOR_FIT}); using fallback R_max="
                  f"{FALLBACK_R_MAX_KM:.1f} km.")
            return FALLBACK_R_MAX_KM, diagnostics

        lats, lons, series, times_used = _build_kc_field_from_records(records_by_point, stage_date)
        if len(lats) < MIN_GRID_POINTS_FOR_FIT:
            diagnostics["status"] = "too_few_valid_kc_samples_fallback"
            print(f"  [i] {day_label}: not enough valid Kc samples after sun-angle "
                  f"filtering; using fallback R_max={FALLBACK_R_MAX_KM:.1f} km.")
            return FALLBACK_R_MAX_KM, diagnostics

        h_mid, gamma, counts = _empirical_semivariogram(lats, lons, series, N_LAG_BINS)
        diagnostics["n_lag_bins"] = len(h_mid)
        if len(h_mid) < 3:
            diagnostics["status"] = "too_few_lag_bins_fallback"
            print(f"  [i] {day_label}: not enough populated lag bins; using fallback "
                  f"R_max={FALLBACK_R_MAX_KM:.1f} km.")
            return FALLBACK_R_MAX_KM, diagnostics

        fit = _fit_variogram_model(h_mid, gamma, counts)
        if fit is None:
            diagnostics["status"] = "fit_failed_fallback"
            print(f"  [i] {day_label}: variogram model fit did not converge; using "
                  f"fallback R_max={FALLBACK_R_MAX_KM:.1f} km.")
            return FALLBACK_R_MAX_KM, diagnostics

        model_name, popt, sse = fit
        r_max_raw, solve_status = _solve_dynamic_r_max(model_name, popt, epsilon, h_mid.max())
        r_max = float(np.clip(r_max_raw, MIN_R_MAX_KM, MAX_R_MAX_KM))

        diagnostics.update({
            "status": solve_status,
            "model": model_name,
            "nugget": float(popt[0]),
            "partial_sill": float(popt[1]),
            "range_km": float(popt[2]),
            "sse": sse,
            "r_max_raw_km": float(r_max_raw),
            "r_max_clamped_km": r_max,
        })

        print(f"  [i] {day_label}: {model_name} variogram fit "
              f"(nugget={popt[0]:.4f}, sill={popt[1]:.4f}, range={popt[2]:.1f} km) "
              f"-> R_max={r_max:.1f} km ({solve_status}, n_sample_points={len(lats)})")

        if SAVE_VARIOGRAM_PLOTS:
            try:
                _plot_variogram(day_label, h_mid, gamma, counts, model_name, popt,
                                 r_max, VARIOGRAM_PLOT_DIR)
            except Exception as e:
                print(f"    [!] variogram plot failed for {day_label}: {e}")

        return r_max, diagnostics

    except Exception as e:
        diagnostics["status"] = f"error_fallback: {e}"
        print(f"  [!] {day_label}: semivariogram R_max computation failed ({e}); "
              f"using fallback R_max={FALLBACK_R_MAX_KM:.1f} km.")
        return FALLBACK_R_MAX_KM, diagnostics


# --- 4. ROUTE DISCRETIZATION --- (unchanged)
def discretize_route(coords, R_max, velocity_kmh):
    discretized = []
    current_center = coords[0]
    dist_since_last = 0.0
    cumulative_dist = 0.0

    discretized.append({"lat": current_center[0], "lon": current_center[1], "cum_dist": 0.0, "eta_hours": 0.0})

    for i in range(1, len(coords)):
        step_dist = haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        dist_since_last += step_dist
        cumulative_dist += step_dist

        if dist_since_last >= (2 * R_max):
            current_center = coords[i]
            discretized.append({
                "lat": current_center[0],
                "lon": current_center[1],
                "cum_dist": cumulative_dist,
                "eta_hours": cumulative_dist / velocity_kmh,
            })
            dist_since_last = 0.0

    return discretized


# --- 5. MAIN EXECUTION ---
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kml_file = os.path.join(script_dir, KML_FILENAME_HINT)

    if not os.path.exists(kml_file):
        print(f"KML file not found: {kml_file}")
        return

    day_route_segments = parse_kml_day_folders(kml_file)
    if not day_route_segments:
        print("No main routes found in KML.")
        return

    # Keyed by real day label, in true chronological day-number order.
    # Day 3 (Full Blind) is simply absent - no route to stitch/discretize.
    stage_groups = stitch_day_segments(day_route_segments)

    # Dates are derived from each folder's own parsed day number, NOT from
    # a sequential enumerate() index - see v1.2 CHANGE for why that matters.
    day_numbers = {day_label: parse_day_number(day_label) for day_label in stage_groups}
    stage_dates = {
        day_label: RACE_START_DATE + timedelta(days=day_numbers[day_label] - 1)
        for day_label in stage_groups
    }

    print(f"Found {len(stage_groups)} stage(s) with a defined route "
          f"(of {len(day_route_segments)} day folder(s) parsed): "
          f"{list(stage_groups.keys())}\n")

    sast_tz = ZoneInfo("Africa/Johannesburg")
    json_output_data = {}

    for day_label, coords in stage_groups.items():
        if len(coords) < 2:
            continue

        day_number = day_numbers[day_label]
        stage_date = stage_dates[day_label]

        print(f"Processing stitched stage: {day_label} ({stage_date.isoformat()})...")

        total_dist = sum(
            haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
            for i in range(1, len(coords))
        )
        velocity = total_dist / DEFAULT_STAGE_HOURS

        stage_start_time = STAGE_START_TIME_DAY1 if day_number == 1 else STAGE_START_TIME_OTHER

        try:
            solar_irr, records_by_point = fetch_stage_solar_irradiance(
                coords, stage_date, stage_start_time, sast_tz
            )
        except Exception as e:
            print(f"  [!] {day_label}: could not fetch any Solcast data ({e}); skipping stage.")
            continue

        r_max_km, variogram_diagnostics = compute_stage_r_max(
            records_by_point, stage_date, day_label
        )
        waypoints = discretize_route(coords, r_max_km, velocity)

        stage_start_sast = datetime.combine(stage_date, stage_start_time, tzinfo=sast_tz)

        stage_waypoints_data = []

        for wp in waypoints:
            arrival_sast = stage_start_sast + timedelta(hours=wp["eta_hours"])
            arrival_utc = arrival_sast.astimezone(ZoneInfo("UTC"))

            ghi, dni, wind_spd, wind_dir = fetch_local_weather(solar_irr, wp["lat"], wp["lon"], arrival_utc)

            stage_waypoints_data.append({
                "cum_dist_km": round(wp["cum_dist"], 2),
                "eta_hours": round(wp["eta_hours"], 2),
                "latitude": round(wp["lat"], 5),
                "longitude": round(wp["lon"], 5),
                "ghi_wm2": round(ghi, 2),
                "dni_wm2": round(dni, 2),
                "wind_speed_ms": round(wind_spd, 2),
                "wind_direction_deg": round(wind_dir, 2)
            })

        # The last waypoint's ETA represents the time taken based on the velocity assumption
        time_taken = waypoints[-1]["eta_hours"] if waypoints else 0.0

        json_output_data[day_label] = {
            "day_number": day_number,
            "date": stage_date.isoformat(),
            "start_time_sast": stage_start_time.strftime("%H:%M:%S"),
            "distance_covered_km": round(total_dist, 2),
            "time_taken_hours": round(time_taken, 2),
            "r_max_km": round(r_max_km, 2),
            "r_max_source": "semivariogram" if variogram_diagnostics.get("status") == "solved"
                             else "fallback",
            "semivariogram_diagnostics": variogram_diagnostics,
            "waypoints": stage_waypoints_data
        }

    # Output to JSON
    json_filename = "race_weather_data_2026.json"
    with open(json_filename, "w") as json_file:
        json.dump(json_output_data, json_file, indent=4)

    print(f"\nWeather and route data successfully exported to {json_filename}")
    print(f"Solcast locations queried this run: {len(_solcast_queried_locations)} "
          f"/ {SOLCAST_LOCATION_BUDGET} budget.")

if __name__ == "__main__":
    main()