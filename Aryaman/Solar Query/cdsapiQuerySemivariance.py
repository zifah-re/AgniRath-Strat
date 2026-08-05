import math
import glob
import os
import re
import statistics
import tempfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import cdsapi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# DEFAULTS - edit these instead of being prompted at runtime.
# ---------------------------------------------------------------------------
KML_FILENAME_HINT = "2024 Sasol Solar Challenge Route (Publish).kml"

# 2024 Sasol Solar Challenge: Stage 1 was 13 Sept 2024. Stages are assumed to
# run on consecutive calendar days from here (Stage 1 -> RACE_START_DATE,
# Stage 2 -> +1 day, etc). VERIFY this against the real day-by-day schedule -
# rest days or reordering would silently point stages at the wrong ERA5 date.
RACE_START_DATE = date(2024, 9, 13)

STAGE_START_TIME = time(9, 0, 0)   # daily start, SAST
DEFAULT_STAGE_HOURS = 8.0          # used to derive a default cruising speed per stage
R_MAX_KM = 10.0                    # fixed waypoint spacing radius (see prior version)
ERA5_AREA = [-22, 16, -35, 33]     # [north, west, south, east]

# Placemark names matched against this pattern are grouped (and stitched)
# into a single stage, e.g. "Day 1 - Secunda to X" and "Day 1 - X to Y" both
# map to "Day 1". Names with no match are kept as their own single-segment
# stage, so this still works if the KML already has one placemark per day.
_DAY_PATTERN = re.compile(r"(?:day|stage)\s*0*(\d+)", re.IGNORECASE)


# --- 1. KML PARSING WITH LOOP FILTERING ---
def parse_kml_main_routes(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    namespace = {"kml": "http://www.opengis.net/kml/2.2"}

    main_routes = {}
    for placemark in root.findall(".//kml:Placemark", namespace):
        name_elem = placemark.find("kml:name", namespace)
        name = name_elem.text.strip() if name_elem is not None else "Unknown"

        if "loop" in name.lower():
            continue

        linestring = placemark.find(".//kml:LineString/kml:coordinates", namespace)
        if linestring is not None:
            coords_text = linestring.text.strip().split()
            coords = []
            for pt in coords_text:
                lon, lat, *_ = map(float, pt.split(","))
                coords.append((lat, lon))
            main_routes[name] = coords
    return main_routes


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


# --- 2. STAGE GROUPING / STITCHING ---
def group_stages_by_day(route_placemarks):
    """
    Groups KML placemarks into per-day stages and stitches multi-segment
    days into one continuous coordinate list (in placemark order), so a day
    made of several named legs becomes a single route + a single plot.
    """
    groups = {}
    order = []
    for name, coords in route_placemarks.items():
        match = _DAY_PATTERN.search(name)
        key = f"Day {int(match.group(1))}" if match else name
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(coords)

    stitched = {}
    for key in order:
        segments = groups[key]
        combined = list(segments[0])
        for seg in segments[1:]:
            if not seg:
                continue
            # Skip a duplicate junction point if consecutive segments share it.
            if combined and haversine(*combined[-1], *seg[0]) < 0.05:
                combined.extend(seg[1:])
            else:
                combined.extend(seg)
        stitched[key] = combined

    def sort_key(k):
        m = re.match(r"Day (\d+)$", k)
        return (0, int(m.group(1))) if m else (1, order.index(k))

    return {k: stitched[k] for k in sorted(stitched, key=sort_key)}


# --- 3. GRIDDED DATA INGESTION (no .nc file left on disk) ---
def _normalize_time_dim(ds):
    """Current CDS/ECMWF Datastores backend names the time coordinate
    'valid_time' instead of the legacy 'time'; normalize it."""
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    return ds


def _dates_by_year_month(dates):
    groups = {}
    for d in dates:
        groups.setdefault((d.year, d.month), set()).add(d.day)
    return groups


def download_era5_solar_grid(dates):
    """
    Downloads ERA5 GHI (ssrd) for all given calendar dates, grouped into as
    few CDS requests as possible (one per year/month), and returns a single
    in-memory xarray Dataset spanning all of them.

    CDS always writes its response to a file - there's no API for getting
    data purely in memory - so this downloads to a temp file and deletes it
    immediately after loading, rather than leaving a .nc file on disk.
    """
    c = cdsapi.Client()
    datasets = []
    for (year, month), days in _dates_by_year_month(dates).items():
        tmp_path = tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name
        try:
            print(f"Downloading ERA5 solar grid for {year}-{month:02d}, "
                  f"days {sorted(days)} (historical/reanalysis data)...")
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": "surface_solar_radiation_downwards",
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": [f"{d:02d}" for d in sorted(days)],
                    "time": [f"{h:02d}:00" for h in range(4, 19)],
                    "area": ERA5_AREA,
                },
                tmp_path,
            )
            ds = _normalize_time_dim(xr.open_dataset(tmp_path))
            ds.load()  # force full read into memory before deleting the file
            datasets.append(ds)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]


def fetch_local_irradiance(ds, lat, lon, arrival_time_utc):
    naive_time = arrival_time_utc.replace(tzinfo=None)
    try:
        val = (
            ds["ssrd"]
            .sel(latitude=lat, longitude=lon, method="nearest")
            .interp(time=np.datetime64(naive_time))
            .values
        )
        ghi_wm2 = float(val) / 3600.0
        return max(0, ghi_wm2)
    except Exception as e:
        print(f"  [!] irradiance lookup failed at ({lat:.3f},{lon:.3f}) "
              f"{arrival_time_utc}: {e}")
        return 0.0


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


def resolve_kml_path(preferred_name):
    if os.path.exists(preferred_name):
        return preferred_name
    candidates = glob.glob("*.kml")
    if len(candidates) == 1:
        print(f"'{preferred_name}' not found; using '{candidates[0]}' instead.")
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"'{preferred_name}' not found and multiple .kml files are present: "
            f"{candidates}. Set KML_FILENAME_HINT explicitly."
        )
    raise FileNotFoundError(f"No .kml file found (looked for '{preferred_name}').")


# --- 5. STATS ---
def summarize_stage(day_label, stage_date, eta_hours, ghis):
    ghis_arr = np.array(ghis, dtype=float)
    eta_arr = np.array(eta_hours, dtype=float)

    # Energy = integral of irradiance over elapsed time (Wh/m^2), i.e. total
    # insolation received per unit area over the course of the stage.
    energy_wh_m2 = float(np.trapz(ghis_arr, x=eta_arr)) if len(ghis_arr) > 1 else 0.0

    # Mode of a set of near-continuous floats is rarely meaningful as-is
    # (exact repeats are unlikely), so it's computed on values binned to the
    # nearest 5 W/m^2.
    binned = np.round(ghis_arr / 5.0) * 5.0
    mode_val = float(statistics.mode(binned)) if len(binned) else float("nan")

    return {
        "Stage": day_label,
        "Date": stage_date.isoformat(),
        "Waypoints": len(ghis),
        "Energy (kWh/m^2)": energy_wh_m2 / 1000.0,
        "Max GHI (W/m^2)": float(np.max(ghis_arr)) if len(ghis_arr) else float("nan"),
        "Mean GHI (W/m^2)": float(np.mean(ghis_arr)) if len(ghis_arr) else float("nan"),
        "Median GHI (W/m^2)": float(np.median(ghis_arr)) if len(ghis_arr) else float("nan"),
        "Mode GHI (W/m^2, 5W bins)": mode_val,
    }


def print_summary_table(rows):
    if not rows:
        print("No stage data to summarize.")
        return
    df = pd.DataFrame(rows).set_index("Stage")
    with pd.option_context("display.float_format", lambda x: f"{x:.1f}", "display.width", 120):
        print("\n" + "=" * 78)
        print("DAILY SUMMARY")
        print("=" * 78)
        print(df.to_string())
    print(f"\nTotal event energy input across all stages: "
          f"{df['Energy (kWh/m^2)'].sum():.2f} kWh/m^2")


# --- 6. MAIN EXECUTION (no interactive prompts) ---
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kml_file = os.path.join(script_dir, "2024 Sasol Solar Challenge Route (Publish).kml")
    
    route_placemarks = parse_kml_main_routes(kml_file)
    if not route_placemarks:
        print("No main routes found in KML.")
        return

    stage_groups = group_stages_by_day(route_placemarks)
    stage_dates = [RACE_START_DATE + timedelta(days=i) for i in range(len(stage_groups))]

    print(f"Found {len(stage_groups)} stage(s) after grouping/stitching: "
          f"{list(stage_groups.keys())}")
    print(f"Assuming consecutive dates starting {RACE_START_DATE.isoformat()} "
          f"-> verify against the actual event schedule.\n")

    solar_grid_ds = download_era5_solar_grid(stage_dates)
    sast_tz = ZoneInfo("Africa/Johannesburg")

    summary_rows = []

    for (day_label, coords), stage_date in zip(stage_groups.items(), stage_dates):
        if len(coords) < 2:
            continue

        print(f"\nProcessing stitched stage: {day_label} ({stage_date.isoformat()})...")

        total_dist = sum(
            haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
            for i in range(1, len(coords))
        )
        velocity = total_dist / DEFAULT_STAGE_HOURS

        waypoints = discretize_route(coords, R_MAX_KM, velocity)
        print(f"  {len(waypoints)} waypoints @ R_max={R_MAX_KM:.0f} km, "
              f"default speed {velocity:.1f} km/h over {total_dist:.1f} km.")

        stage_start_sast = datetime.combine(stage_date, STAGE_START_TIME, tzinfo=sast_tz)

        distances, ghis, eta_hours_list = [], [], []
        for wp in waypoints:
            arrival_sast = stage_start_sast + timedelta(hours=wp["eta_hours"])
            arrival_utc = arrival_sast.astimezone(ZoneInfo("UTC"))
            ghi = fetch_local_irradiance(solar_grid_ds, wp["lat"], wp["lon"], arrival_utc)

            distances.append(wp["cum_dist"])
            ghis.append(ghi)
            eta_hours_list.append(wp["eta_hours"])
            print(f"  Dist: {wp['cum_dist']:6.1f} km | ETA: {arrival_sast.strftime('%H:%M SAST')} | GHI: {ghi:5.1f} W/m²")

        summary_rows.append(summarize_stage(day_label, stage_date, eta_hours_list, ghis))

        plt.figure(figsize=(10, 5))
        plt.plot(distances, ghis, marker="o", color="gold", linestyle="-", linewidth=2)
        plt.fill_between(distances, ghis, color="yellow", alpha=0.3)
        if ghis:
            plt.axhline(np.mean(ghis), color="gray", linestyle="--", linewidth=1,
                         label=f"Mean = {np.mean(ghis):.0f} W/m²")
            plt.legend()
        plt.title(f"Stage Solar Profile - {day_label} ({stage_date.isoformat()}, "
                  f"start {STAGE_START_TIME.strftime('%H:%M')} SAST)")
        plt.xlabel("Route Distance (km)")
        plt.ylabel("Solar Irradiance (GHI) [W/m²]")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()