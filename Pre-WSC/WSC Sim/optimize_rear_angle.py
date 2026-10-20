#!/usr/bin/env python3
"""
optimize_rear_angle.py

Find the rear-panel tilt angle that maximises solar input for the car.

Panel model
-----------
  total area  : 5.97 m^2
  front 54 %  : flat (tilt 0 deg)
  rear  46 %  : inclined by `angle` (optimised)

Angle convention
----------------
  angle > 0 : rear panel slopes DOWN toward the back of the car
              (surface normal tilts toward the rear; azimuth = heading + 180)
  angle < 0 : rear panel slopes UP toward the back
              (surface normal tilts toward the front; azimuth = heading)

Data used
---------
  * Day{n}_*.kml.save   -> JSON with a "profile" dict
                           (Coordinates, Headings, Altitude, Distance[km])
  * Day{n}_*_solar.json -> list of nodes {"lat","lon","data":[{"period_end",
                           "dni","ghi",...}, ...]}, timestamps in UTC

For each of the 5 days:
  1. keep only the middle `--span-km` (default 150 km) of the route
  2. take DNI/GHI at 12:00 local time from the nearest solar node
  3. compute the sun position at each route point
  4. sweep the rear angle and average power over the route segment
Then average over all days and report the optimum.

Usage
-----
  python optimize_rear_angle.py --data-dir "Pre-WSC/WSC Sim" --plot
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
FRONT_FRACTION = 0.54      # flat, at the front
ALBEDO = 0.2
ARRAY_EFFICIENCY = 0.21    # only scales results; does not change the optimum


# ------------------------------- file loading ------------------------------ #
def find_file(data_dir, pattern):
    hits = sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"No file matching '{pattern}' under {data_dir}")
    return hits[0]


def load_route_segment(path, span_km):
    """Return dict of arrays for the middle `span_km` of the route."""
    with open(path, "r", encoding="utf-8") as f:
        prof = json.load(f)["profile"]

    coords = np.array(prof["Coordinates"], dtype=float)
    dist = np.array(prof["Distance"], dtype=float)           # km
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
    """Return (lat_lon[k,2], dni[k], ghi[k], local_date, target_datetime)."""
    with open(path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    # Local date of the data (from the first node's median timestamp)
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
        if t0 < ep[0] or t0 > ep[-1]:
            print(f"  [warn] {os.path.basename(path)}: {hour}:00 local outside "
                  f"data range for node ({node['lat']:.3f},{node['lon']:.3f})")
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
    """Isotropic-sky plane-of-array irradiance (W/m^2)."""
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
    """Mean incident solar power (W) over the segment for each rear angle."""
    coords = seg["coords"]
    mean_lat = float(np.mean(coords[:, 0]))
    scale = np.cos(np.radians(mean_lat))
    tree = cKDTree(np.column_stack((nodes_latlon[:, 0], nodes_latlon[:, 1] * scale)))
    _, nn = tree.query(np.column_stack((coords[:, 0], coords[:, 1] * scale)))
    dni = node_dni[nn]
    ghi = node_ghi[nn]

    zen, sun_az = sun_position(target, coords[:, 0], coords[:, 1], seg["altitude"])
    head = seg["headings"]

    front = plane_irradiance(0.0, head, zen, sun_az, dni, ghi)   # flat
    area_front = TOTAL_AREA_M2 * FRONT_FRACTION
    area_rear = TOTAL_AREA_M2 * (1.0 - FRONT_FRACTION)

    result = np.empty(len(angles))
    for k, a in enumerate(angles):
        rear_az = (head + 180.0) % 360.0 if a >= 0 else head % 360.0
        rear = plane_irradiance(abs(a), rear_az, zen, sun_az, dni, ghi)
        result[k] = np.mean(area_front * front + area_rear * rear)

    info = dict(zen=float(np.mean(zen)), dni=float(np.mean(dni)),
                ghi=float(np.mean(ghi)), heading=float(np.mean(head)))
    return result, info


# ----------------------------------- main ---------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="folder containing the files")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--span-km", type=float, default=150.0)
    ap.add_argument("--hour", type=int, default=12, help="local hour of day")
    ap.add_argument("--window-min", type=float, default=0.0,
                    help="average irradiance over +/- this many minutes around the hour")
    ap.add_argument("--tz", default="Australia/Darwin")
    ap.add_argument("--angle-min", type=float, default=-20.0)
    ap.add_argument("--angle-max", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    tz = ZoneInfo(args.tz)
    angles = np.arange(args.angle_min, args.angle_max + 1e-9, args.step)

    per_day = {}
    for d in range(1, args.days + 1):
        kml_path = find_file(args.data_dir, f"Day{d}_*.kml.save")
        sol_path = find_file(args.data_dir, f"Day{d}_*solar.json")
        print(f"Day {d}: {os.path.basename(kml_path)} | {os.path.basename(sol_path)}")

        seg = load_route_segment(kml_path, args.span_km)
        latlon, dni, ghi, local_date, target = load_noon_irradiance(
            sol_path, tz, args.hour, args.window_min)
        power, info = sweep_day(seg, latlon, dni, ghi, target, angles)
        per_day[d] = power

        best = angles[np.argmax(power)]
        flat = power[np.argmin(np.abs(angles))]
        print(f"   date {local_date}, {len(seg['coords'])} pts, "
              f"mean heading {info['heading']:.0f} deg, mean zenith {info['zen']:.1f} deg, "
              f"DNI {info['dni']:.0f}, GHI {info['ghi']:.0f} W/m2")
        print(f"   best rear angle {best:+.1f} deg -> {power.max():.0f} W incident "
              f"({100 * (power.max() / flat - 1):+.2f}% vs flat)\n")

    mean_power = np.mean(list(per_day.values()), axis=0)
    best_i = int(np.argmax(mean_power))
    best_angle = angles[best_i]
    flat_power = mean_power[np.argmin(np.abs(angles))]

    print("=" * 62)
    print(f"OPTIMAL REAR ANGLE (avg of {args.days} days, {args.hour}:00 local, "
          f"middle {args.span_km:.0f} km): {best_angle:+.1f} deg")
    print(f"  mean incident power : {mean_power[best_i]:.0f} W "
          f"(~{mean_power[best_i] * ARRAY_EFFICIENCY:.0f} W electrical @ "
          f"{ARRAY_EFFICIENCY:.0%} eff)")
    print(f"  fully flat panel    : {flat_power:.0f} W")
    print(f"  gain vs flat        : {100 * (mean_power[best_i] / flat_power - 1):+.2f}%")
    print("  (angle > 0 = rear slopes down toward back of car)")
    print("=" * 62)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5))
        for d, p in per_day.items():
            plt.plot(angles, p, alpha=0.5, label=f"Day {d}")
        plt.plot(angles, mean_power, "k", lw=2.5, label="Mean")
        plt.axvline(best_angle, color="r", ls="--",
                    label=f"Optimum {best_angle:+.1f} deg")
        plt.xlabel("Rear panel angle (deg, + = slopes down toward rear)")
        plt.ylabel("Mean incident solar power (W)")
        plt.title(f"Rear angle sweep, {args.hour}:00 local, middle {args.span_km:.0f} km")
        plt.grid(alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig("rear_angle_sweep.png", dpi=200)
        print("Saved rear_angle_sweep.png")
        plt.show()


if __name__ == "__main__":
    main()