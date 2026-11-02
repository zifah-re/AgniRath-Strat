#!/usr/bin/env python3
"""
optimize_rear_angle.py

Find the rear-panel tilt angle that maximises solar input for the car,
and compares it with the average solar position to evaluate diffuse effects.
"""

import argparse
import glob
import json
import os
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pvlib
from scipy.spatial import cKDTree

# ----------------------------- panel constants ----------------------------- #
TOTAL_AREA_M2 = 5.97
FRONT_FRACTION = 0.54       # flat, at the front
ALBEDO = 0.2
ARRAY_EFFICIENCY = 0.21     # only scales results; does not change the optimum


# ------------------------------- file loading ------------------------------ #
def find_file(data_dir, pattern):
    hits = sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"No file matching '{pattern}' under {data_dir}")
    return hits[0]


def load_route_segment(path, span_km):
    with open(path, "r", encoding="utf-8") as f:
        prof = json.load(f)["profile"]

    coords = np.array(prof["Coordinates"], dtype=float)
    dist = np.array(prof["Distance"], dtype=float)          # km
    head = np.array(prof["Headings"], dtype=float)
    alt = np.array(prof.get("Altitude", np.zeros(len(dist))), dtype=float)

    n = min(len(coords), len(dist), len(head), len(alt))
    coords, dist, head, alt = coords[:n], dist[:n], head[:n], alt[:n]

    total = dist[-1]
    mid = total / 2.0
    half = span_km / 2.0
    mask = (dist >= mid - half) & (dist <= mid + half)
    if mask.sum() < 2:
        raise ValueError(f"{path}: middle {span_km} km selection is empty "
                         f"(route length {total:.1f} km)")
    return dict(coords=coords[mask], headings=head[mask], altitude=alt[mask],
                dist=dist[mask], total_km=total)


def load_noon_irradiance(path, tz, hour, window_min):
    with open(path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    def epochs(node):
        return np.array([pd.Timestamp(d["period_end"]).timestamp()
                         for d in node["data"]])

    ep0 = epochs(nodes[0])
    local_date = datetime.fromtimestamp(float(np.median(ep0)), tz).date()
    target = datetime.combine(local_date, time(hour, 0), tzinfo=tz)
    t0 = target.timestamp()

    offsets = (np.linspace(-window_min, window_min, 13) * 60.0
               if window_min > 0 else np.array([0.0]))

    latlon, dni_out, ghi_out = [], [], []
    for node in nodes:
        ep = epochs(node)
        order = np.argsort(ep)
        ep = ep[order]
        dni = np.array([float(node["data"][i].get("dni") or 0.0) for i in order])
        ghi = np.array([float(node["data"][i].get("ghi") or 0.0) for i in order])
        latlon.append((node["lat"], node["lon"]))
        dni_out.append(np.mean(np.interp(t0 + offsets, ep, dni)))
        ghi_out.append(np.mean(np.interp(t0 + offsets, ep, ghi)))

    return (np.array(latlon), np.array(dni_out), np.array(ghi_out),
            local_date, target)


# --------------------------------- physics --------------------------------- #
def sun_position(target, lats, lons, alts):
    ts = pd.Timestamp(target)
    zen = np.empty(len(lats))
    az = np.empty(len(lats))
    idx = pd.DatetimeIndex([ts])
    for i in range(len(lats)):
        sp = pvlib.solarposition.get_solarposition(
            idx, float(lats[i]), float(lons[i]), altitude=float(alts[i]))
        zen[i] = sp["apparent_zenith"].iloc[0]
        az[i] = sp["azimuth"].iloc[0]
    return zen, az


def plane_irradiance(tilt, surf_az, zen, sun_az, dni, ghi):
    aoi = pvlib.irradiance.aoi(tilt, surf_az, zen, sun_az)
    cos_aoi = np.maximum(np.cos(np.radians(aoi)), 0.0)
    cos_zen = np.maximum(np.cos(np.radians(zen)), 0.0)
    beam = dni * cos_aoi
    dhi = np.maximum(ghi - dni * cos_zen, 0.0)
    tilt_r = np.radians(tilt)
    sky = dhi * (1.0 + np.cos(tilt_r)) / 2.0
    ground = ghi * ALBEDO * (1.0 - np.cos(tilt_r)) / 2.0
    out = beam + sky + ground
    return np.where(zen < 90.0, out, 0.0)


def sweep_day(seg, nodes_latlon, node_dni, node_ghi, target, angles):
    coords = seg["coords"]
    mean_lat = float(np.mean(coords[:, 0]))
    scale = np.cos(np.radians(mean_lat))
    tree = cKDTree(np.column_stack((nodes_latlon[:, 0], nodes_latlon[:, 1] * scale)))
    _, nn = tree.query(np.column_stack((coords[:, 0], coords[:, 1] * scale)))
    dni = node_dni[nn]
    ghi = node_ghi[nn]

    zen, sun_az = sun_position(target, coords[:, 0], coords[:, 1], seg["altitude"])
    head = seg["headings"]

    front = plane_irradiance(0.0, head, zen, sun_az, dni, ghi)
    area_front = TOTAL_AREA_M2 * FRONT_FRACTION
    area_rear = TOTAL_AREA_M2 * (1.0 - FRONT_FRACTION)

    result = np.empty(len(angles))
    for k, a in enumerate(angles):
        rear_az = (head + 180.0) % 360.0 if a >= 0 else head % 360.0
        rear = plane_irradiance(abs(a), rear_az, zen, sun_az, dni, ghi)
        result[k] = np.mean(area_front * front + area_rear * rear)

    info = dict(zen=zen, dni=dni, ghi=ghi, heading=head)
    return result, info


# ----------------------------------- main ---------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="folder containing the files")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--span-km", type=float, default=150.0)
    ap.add_argument("--hour", type=int, default=12, help="local hour of day")
    ap.add_argument("--window-min", type=float, default=0.0)
    ap.add_argument("--tz", default="Australia/Darwin")
    ap.add_argument("--angle-min", type=float, default=-20.0)
    ap.add_argument("--angle-max", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    tz = ZoneInfo(args.tz)
    angles = np.arange(args.angle_min, args.angle_max + 1e-9, args.step)

    per_day = {}
    all_zeniths = []
    all_dni = []
    all_ghi = []

    for d in range(1, args.days + 1):
        kml_path = find_file(args.data_dir, f"Day{d}_*.kml.save")
        sol_path = find_file(args.data_dir, f"Day{d}_*solar.json")
        print(f"Day {d}: {os.path.basename(kml_path)} | {os.path.basename(sol_path)}")

        seg = load_route_segment(kml_path, args.span_km)
        latlon, dni, ghi, local_date, target = load_noon_irradiance(
            sol_path, tz, args.hour, args.window_min)
        power, info = sweep_day(seg, latlon, dni, ghi, target, angles)
        per_day[d] = power

        all_zeniths.extend(info['zen'])
        all_dni.extend(info['dni'])
        all_ghi.extend(info['ghi'])

        best = angles[np.argmax(power)]
        flat = power[np.argmin(np.abs(angles))]
        print(f"   date {local_date}, mean zenith {np.mean(info['zen']):.1f} deg, "
              f"DNI {np.mean(info['dni']):.0f}, GHI {np.mean(info['ghi']):.0f} W/m2")
        print(f"   best rear angle {best:+.1f} deg -> {power.max():.0f} W incident\n")

    mean_power = np.mean(list(per_day.values()), axis=0)
    best_i = int(np.argmax(mean_power))
    best_angle = angles[best_i]
    flat_power = mean_power[np.argmin(np.abs(angles))]

    # Average solar metrics across all sampled points
    mean_zenith = np.mean(all_zeniths)
    mean_elevation = 90.0 - mean_zenith
    mean_dni = np.mean(all_dni)
    mean_ghi = np.mean(all_ghi)
    diffuse_fraction = np.mean(np.maximum(0.0, np.array(all_ghi) - np.array(all_dni) * np.cos(np.radians(all_zeniths)))) / np.mean(all_ghi)

    print("=" * 62)
    print(f"OPTIMIZATION & SOLAR DIAGNOSTICS ({args.days} days, {args.hour}:00 local)")
    print(f"   Optimal Rear Angle   : {best_angle:+.1f} deg")
    print(f"   Mean Sun Zenith      : {mean_zenith:.1f} deg (Elevation: {mean_elevation:.1f} deg)")
    print(f"   Mean DNI / GHI       : {mean_dni:.0f} / {mean_ghi:.0f} W/m²")
    print(f"   Est. Diffuse Fraction: {diffuse_fraction:.1%}")
    print(f"   Angle Difference     : {abs(best_angle - mean_zenith):.1f} deg deviation from zenith")
    print("=" * 62)


if __name__ == "__main__":
    main()