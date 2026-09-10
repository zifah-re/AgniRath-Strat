#!/usr/bin/env python3
"""
Fallback Model / Strategy Export Script
=======================================
Runs the Fallback Model (`Fallback Model/singleday.py`) across all 8 race days 
(or re-uses precomputed `.npz` files if available) and exports the full-plan strategy
to `output/strategy_fallback[_<tag>].json` following the schema defined in 
`Model/optimizers/trust_region.py`.
"""

import os
import sys
import argparse
import json
from datetime import datetime, date, time
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np

# Ensure root directory and Fallback Model directory are in Python path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "Fallback Model" else SCRIPT_DIR
FALLBACK_DIR = ROOT_DIR / "Fallback Model"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(FALLBACK_DIR) not in sys.path:
    sys.path.insert(0, str(FALLBACK_DIR))

# Import required public functions from Fallback Model/singleday.py
from singleday import (
    resolve,
    build_day_route,
    calc_stationary_charge,
    SA_TZ,
    BATTERY_WH,
    SOC_MAX,
    SOC_MIN,
    EOD_CUTOFF_HOUR,
)
from solar_table import SolarIrradiance

# ----------------- CONSTANTS & RACES DATA ----------------- #
DAYWISE_FILES = {
    "Day 1": {
        "date": date(2026, 9, 10),
        "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 1 Boiketlong to Rustenburg",
        "l": "2026 Sasol Solar Challenge Route (Publish)_Day 1 _Rustenburg Loop",
        "s2": "2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 2 Rustenburg to Swartruggens",
    },
    "Day 2": {
        "date": date(2026, 9, 11),
        "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 2 Half Blind_11 Sept Stage 1 Swart Ruggens to Zeerust",
        "l": "SSC ROUTE FINAL_Day 2 Half Blind_Day 2 Loop",
        "s2": "2026 Sasol Solar Challenge Route (Publish)_Day 2 Half Blind_11 Sept Stage 2 Zeerust to Vryburg",
    },
    "Day 3": {
        "date": date(2026, 9, 12),
        "s1": "Day 3 probables_Probable Prahlad Route_Stage 1",
        "l": "Day 3 probables_Probable Prahlad Route_Day 3 Loop",
        "s2": "Day 3 probables_Probable Prahlad Route_Stage 2",
    },
    "Day 4": {
        "date": date(2026, 9, 13),
        "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 4_13 Sept Stage 1 Kimberley to Postmasburg",
        "l": "2026 Sasol Solar Challenge Route (Publish)_Day 4_Postmasburg Loop",
        "s2": "2026 Sasol Solar Challenge Route (Publish)_Day 4_13 Sept Stage 2 Postmasburg to Olifantshoek",
    },
    "Day 5": {
        "date": date(2026, 9, 14),
        "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 1 Olifantshoek to Upington",
        "l": "2026 Sasol Solar Challenge Route (Publish)_Day 5 _Upington Loop",
        "s2": "2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 2 Upington to Augrabies",
    },
    "Day 6": {
        "date": date(2026, 9, 15),
        "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 6 _15 Sept Stage 1 Augrabies to Springbok",
        "l": "2026 Sasol Solar Challenge Route (Publish)_Day 6 _Springbok Loop",
        "s2": None,
    },
    "Day 7": {
        "date": date(2026, 9, 16),
        "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 7_16 Sept Stage 1 Springbok to Van Rhynsdorp",
        "l": "2026 Sasol Solar Challenge Route (Publish)_Day 7_Van Rhynsdorp Loop",
        "s2": "2026 Sasol Solar Challenge Route (Publish)_Day 7_16 Sept Stage 2 Van Rhynsdorp to Clanwilliam",
    },
    "Day 8": {
        "date": date(2026, 9, 17),
        "s1": "2026 Sasol Solar Challenge Route (Publish)_Day 8_17 Sept Stage 1 Clanwilliam to Ceres",
        "l": "2026 Sasol Solar Challenge Route (Publish)_Day 8_Ceres Loop",
        "s2": "2026 Sasol Solar Challenge Route (Publish)_Day 8_17 Sept Stage 2 Ceres to Paarl",
    },
}

START_SOCS = [95.0, 75.03, 64.57, 73.37, 63.41, 54.16, 60.48, 61.80]
V_GUESSES = [60.0, 64.0, 54.0, 52.0, 59.0, 53.0, 54.0, 54.0]
LOOPS = [7, 9, 5, 9, 3, 5, 5, 4]


def load_profile(name):
    """Load route stage JSON profile from Fallback Model/Saves."""
    if not name:
        return None
    file_path = FALLBACK_DIR / "Saves" / f"{name}.kml.save"
    if not file_path.exists():
        return None
    with open(file_path, "r") as f:
        return json.load(f)["profile"]


def load_solar_obj(weather_filename):
    """Load solar irradiance object from Fallback Model weather files."""
    file_path = FALLBACK_DIR / "Solar_real" / f"mean_{weather_filename}.jsonl"
    if not file_path.exists():
        return None
    with open(file_path, "r") as f:
        weather_data = json.load(f)
    return SolarIrradiance(weather_data, "period_end", "PT5M", 6)


def downsample_trace(
    times_s,
    distances_m,
    speeds_kmh,
    soc_pct,
    powers_w,
    slopes_pct=None,
    stride_m=250.0,
):
    """
    Downsamples full 10m simulation arrays to a spatial stride (~250m)
    matching trust_region.py's _downsample_trace_by_distance.
    """
    n = len(distances_m)
    if n == 0:
        return {
            "distance_km": [],
            "v_kmh": [],
            "soc_pct": [],
            "solar_w": [],
            "slope_pct": [],
            "motor_w": [],
            "time_s": [],
        }

    keep = [0]
    last_d = distances_m[0]
    for i in range(1, n):
        if distances_m[i] - last_d >= stride_m:
            keep.append(i)
            last_d = distances_m[i]
    if keep[-1] != n - 1:
        keep.append(n - 1)

    keep = np.array(keep, dtype=int)

    return {
        "distance_km": [round(float(distances_m[i]) / 1000.0, 3) for i in keep],
        "v_kmh": [round(float(speeds_kmh[i]), 1) for i in keep],
        "soc_pct": [round(float(soc_pct[i]), 2) for i in keep],
        # NOTE: Fallback Model doesn't output solar_w or slope_pct trace arrays in its npz output
        "solar_w": [0.0] * len(keep),
        "slope_pct": (
            [round(float(slopes_pct[i]), 2) for i in keep]
            if slopes_pct is not None
            else [0.0] * len(keep)
        ),
        "motor_w": [round(float(powers_w[i]), 1) for i in keep],
        "time_s": [round(float(times_s[i]), 1) for i in keep],
    }


def run_or_load_day(day_no, force_run=False):
    """Executes resolve() or loads existing .npz profile for a single day."""
    day_key = f"Day {day_no}"
    npz_path = (
        FALLBACK_DIR / "velocity_profiles" / f"optimized_day_{day_no}.npz"
    )

    race_date = DAYWISE_FILES[day_key]["date"]
    s1_name = DAYWISE_FILES[day_key]["s1"]
    l_name = DAYWISE_FILES[day_key]["l"]
    s2_name = DAYWISE_FILES[day_key]["s2"]

    s1_profile = load_profile(s1_name)
    loop_profile = load_profile(l_name)
    s2_profile = load_profile(s2_name)

    weather_filename = l_name if l_name else s1_name
    solar_obj = load_solar_obj(weather_filename)

    start_hour = 9 if day_no == 1 else 8
    morning_ts = datetime.combine(
        race_date, time(6, 0), tzinfo=SA_TZ
    ).timestamp()
    start_time_ts = datetime.combine(
        race_date, time(start_hour, 0), tzinfo=SA_TZ
    ).timestamp()

    current_soc = START_SOCS[day_no - 1]
    morning_gain = 0.0

    if day_no != 1 and solar_obj and s1_profile:
        morning_gain = calc_stationary_charge(
            morning_ts,
            start_time_ts,
            s1_profile["Coordinates"][0],
            s1_profile["Headings"][0],
            s1_profile["Altitude"][0],
            solar_obj,
        )
        current_soc = min(SOC_MAX, current_soc + morning_gain)

    if (not npz_path.exists()) or force_run:
        target_1700_soc = (
            START_SOCS[day_no] if day_no < len(START_SOCS) else 20.0
        )
        resolve(
            current_km=0.0,
            current_time_ts=start_time_ts,
            current_soc=current_soc,
            s1_profile=s1_profile,
            loop_profile=loop_profile,
            s2_profile=s2_profile,
            manual_target_loops=LOOPS[day_no - 1],
            target_eod_soc=target_1700_soc,
            v_guess_kmh=V_GUESSES[day_no - 1],
            solar_obj=solar_obj,
            race_date=race_date,
            day_no=day_no,
            w1=1.0,
            w2=100.0,
            w3=5000.0,
        )

    # Load saved results
    npz_data = np.load(npz_path)

    # Build full route geometry to extract distance and segment metadata
    full_route = build_day_route(
        s1_profile, loop_profile, s2_profile, LOOPS[day_no - 1]
    )

    return {
        "day_no": day_no,
        "date": race_date,
        "npz": npz_data,
        "full_route": full_route,
        "morning_gain_pct": morning_gain,
        "soc_start_pct": current_soc,
        "s1_name": s1_name,
        "l_name": l_name,
        "s2_name": s2_name,
    }


def format_hhmm(timestamp_s):
    """Format Unix epoch timestamp to HH:MM in SA_TZ."""
    dt = datetime.fromtimestamp(timestamp_s, tz=SA_TZ)
    return dt.strftime("%H:%M")


def main():
    parser = argparse.ArgumentParser(
        description="Export Fallback Model strategy to trust_region JSON schema."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="fallback",
        help="Variant name (default: fallback)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag appended to output JSON filename",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Re-run resolve() for all days even if .npz files exist",
    )
    args = parser.parse_args()

    days_dict = {}
    total_distance_km = 0.0

    for day_no in range(1, 9):
        print(f"Processing Day {day_no}...")
        day_info = run_or_load_day(day_no, force_run=args.force_run)
        npz = day_info["npz"]
        full_route = day_info["full_route"]

        speeds_kmh = npz["speeds_kmh"]
        soc_arr = npz["soc"]
        power_w = npz["power_w"]
        times = npz["times"]
        final_soc = float(npz["final_soc"])

        n_segments = len(speeds_kmh)
        # Distance array: 10m steps per segment
        distances_m = np.arange(n_segments) * 10.0
        dist_km = float(distances_m[-1] / 1000.0) if n_segments > 0 else 0.0
        total_distance_km += dist_km

        # Derive drive time & ETA
        drive_time_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        eta_str = format_hhmm(times[-1]) if len(times) > 0 else "00:00"

        # Derive energy stats (Power W * dt s -> Joules -> Wh)
        dt = 10.0 / (
            np.maximum(speeds_kmh, 1.0) / 3.6
        )  # dt per segment in seconds
        motor_energy_wh = float(
            np.sum(np.maximum(0.0, power_w) * dt) / 3600.0
        )

        # Battery drain calculation
        battery_drain_pct = float(day_info["soc_start_pct"] - final_soc)
        battery_drain_wh = (battery_drain_pct / 100.0) * BATTERY_WH

        # Build dashboard trace
        dashboard_trace = downsample_trace(
            times_s=times,
            distances_m=distances_m,
            speeds_kmh=speeds_kmh,
            soc_pct=soc_arr,
            powers_w=power_w,
            slopes_pct=full_route["gradients"] * 100.0,
            stride_m=250.0,
        )

        # Next start SOC (approximated based on next day's start)
        next_start = (
            START_SOCS[day_no] if day_no < len(START_SOCS) else final_soc
        )

        days_dict[str(day_no)] = {
            "route": f"Day {day_no} Full Route",
            "distance_km": round(dist_km, 1),
            # NOTE: Fallback Model combines stages into a single continuous array; stage-level distances not explicitly partitioned
            "stage1_km": 0.0,
            "stage2_km": 0.0,
            "loop_km": 0.0,
            # NOTE: Fallback Model does not model red-flag trailering or towing
            "trailered_km": 0.0,
            "distance_km_simulated": round(dist_km, 1),
            "loops": {f"Day_{day_no}_Loop": LOOPS[day_no - 1]},
            "n_loops": LOOPS[day_no - 1],
            "soc_start_pct": round(float(day_info["soc_start_pct"]), 2),
            "soc_end_pct": round(final_soc, 2),
            "next_start_soc_pct": round(float(next_start), 2),
            "morning_charge_pct": round(
                float(day_info["morning_gain_pct"]), 2
            ),
            # NOTE: Fallback Model does not compute evening charge gain explicitly
            "evening_charge_pct": 0.0,
            # NOTE: Fallback Model does not model late penalties or breakdowns
            "late_penalty_min": 0,
            "breakdown_min": 0,
            "inherited_penalty_min": 0,
            "speed_avg_kmh": round(float(np.mean(speeds_kmh)), 1),
            "speed_min_kmh": round(float(np.min(speeds_kmh)), 1),
            "speed_max_kmh": round(float(np.max(speeds_kmh)), 1),
            "drive_time_s": round(drive_time_s, 0),
            # NOTE: Mandatory stop times (control stops/loop stops) are embedded in delays array in build_day_route
            "stop_time_s": float(np.sum(full_route["delays"])),
            "control_stop_s": 1800.0,  # 30 minutes control stop
            "loop_stop_s": float(LOOPS[day_no - 1] * 300),  # 5 mins per loop
            "eta": eta_str,
            "eta_drive_only": format_hhmm(times[0] + drive_time_s),
            # NOTE: Detailed solar input components are not stored in npz output
            "solar_input_wh": 0.0,
            "solar_underutil_wh": 0.0,
            "solar_stored_wh": 0.0,
            "motor_energy_wh": round(motor_energy_wh, 1),
            "battery_drain_wh": round(battery_drain_wh, 1),
            "battery_drain_pct": round(battery_drain_pct, 2),
            "dashboard_trace": dashboard_trace,
            # NOTE: Fallback Model does not break down traces into discrete stages list
            "stages": [],
            "stage1": {},
            "loop": {},
            "stage2": {},
            "velocity_profile_kmh": [
                int(v) for v in npz.get("speeds_kmh", speeds_kmh)[::100]
            ],
        }

    # Assemble schema
    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_filename = f"strategy_{args.variant}{tag_suffix}.json"
    output_dir = ROOT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / out_filename

    strategy_data = {
        "variant": args.variant,
        "timestamp": datetime.now(tz=SA_TZ).isoformat(),
        "converged": True,
        "feasible": True,
        "iterations": 1,
        "total_distance_km": round(total_distance_km, 1),
        "history": [],
        "days": days_dict,
        # NOTE: Fallback Model does not support trailering
        "total_trailered_km": 0.0,
        "total_distance_km_dp_estimate": round(total_distance_km, 1),
    }

    with open(output_path, "w") as f:
        json.dump(strategy_data, f, indent=2)

    print(f"\nSuccessfully exported Fallback Model strategy to {output_path}")


if __name__ == "__main__":
    main()