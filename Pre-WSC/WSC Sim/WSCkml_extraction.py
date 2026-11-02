import json
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.signal import savgol_filter

KML_PATH = "WSC.kml"
ROUTE_NAME = "WSC Route"          # None -> use the longest LineString
OUT_JSON = "WSC.kml.save"

# Coords are in (lat, lon) format, altitude in meters, distance in km, gradient in percent, headings in degrees

bwsc_control_stops = {
    "Start": (-12.4628, 130.8418),
    "CS1": (-14.4647, 132.2636),
    "CS2": (-16.6744, 133.4078),
    "CS3": (-19.6483, 134.1903),
    "CS4": (-21.5244, 133.8828),
    "CS5": (-23.6980, 133.8807),
    "CS6": (-25.8411, 133.2981),
    "CS7": (-29.0136, 134.7544),
    "CS8": (-31.0222, 135.8286),
    "CS9": (-32.4938, 137.7628),
    "Finish": (-34.9285, 138.6007),
}

# ---------- Parse KML ----------
def load_route(path, route_name=None):
    root = ET.parse(path).getroot()
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    best = None
    for pm in root.iter("Placemark"):
        ls = pm.find("LineString/coordinates")
        if ls is None:
            continue
        name = pm.find("name").text if pm.find("name") is not None else ""
        pts = ls.text.split()
        if route_name and name == route_name:
            best = (name, pts)
            break
        if best is None or len(pts) > len(best[1]):
            best = (name, pts)
    if best is None:
        raise ValueError("No LineString found in KML")

    coords, alts = [], []
    for p in best[1]:
        lon, lat, *rest = p.split(",")
        coords.append((float(lat), float(lon)))
        alts.append(float(rest[0]) if rest else 0.0)
    print(f"Route '{best[0]}': {len(coords)} points")
    return coords, np.array(alts)


# ---------- Distance & heading ----------
def cumulative_distance_km(coords):
    c = np.asarray(coords)
    R = 6371.0088
    p1, p2 = np.radians(c[:-1, 0]), np.radians(c[1:, 0])
    dphi = p2 - p1
    dlmb = np.radians(c[1:, 1] - c[:-1, 1])
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    seg = 2 * R * np.arcsin(np.sqrt(a))
    return np.concatenate(([0.0], np.cumsum(seg)))


def headings_deg(coords):
    c = np.asarray(coords)
    lat1, lat2 = np.radians(c[:-1, 0]), np.radians(c[1:, 0])
    dlon = np.radians(c[1:, 1] - c[:-1, 1])
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    h = (np.degrees(np.arctan2(x, y)) + 360) % 360
    return np.append(h, h[-1])


# ---------- Gradient (same method as render_selected_track in main.py) ----------
def gradient_percent(distance_km, altitude_m):
    target_points = int((distance_km[-1] * 1000) // 100)
    target_idx = np.linspace(0, distance_km[-1], target_points)
    interp_alts = np.interp(target_idx, distance_km, altitude_m)
    slopes = savgol_filter(interp_alts, window_length=15, polyorder=3,
                           deriv=1, delta=100)
    return np.interp(distance_km, target_idx, slopes * 100)


if __name__ == "__main__":
    coords, altitude = load_route(KML_PATH, ROUTE_NAME)
    distance = cumulative_distance_km(coords)
    gradient = gradient_percent(distance, altitude)
    headings = headings_deg(coords)

    profile = {
        "Altitude": altitude.tolist(),
        "Gradient": gradient.tolist(),
        "Coordinates": [list(c) for c in coords],
        "Distance": distance.tolist(),
        "Headings": headings.tolist(),
    }
    with open(OUT_JSON, "w") as f:
        json.dump({"profile": profile}, f)

    print(f"Total distance: {distance[-1]:.1f} km")
    print(f"Altitude range: {altitude.min():.0f}–{altitude.max():.0f} m")
    print(f"Gradient range: {gradient.min():.2f}% to {gradient.max():.2f}%")

keys = profile.keys()
print(keys)
coords = profile["Coordinates"]
distance = profile["Distance"]
headings = profile["Headings"]
# print(f"Start (Darwin): {coords[0]}, End (Adelaide): {coords[-1]}\n")
# print(f'Distance samples: {distance[:10]}, {distance[-10:]}\n')
# print(f'Headings samples: {headings[:10]}, {headings[-10:]}')

distance = np.array(distance)
resolution = (np.diff(distance))
resolution = np.append(resolution, resolution[-1])  # Append last value to keep same length
# fig,ax = plt.subplots(figsize=(10, 6))
# ax.plot(distance, resolution, label="Distance Resolution")
# ax.set_xlabel("Distance (km)")
# ax.set_ylabel("Resolution (km)")
# ax.legend()
# box_props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5, edgecolor='gray')
# ax.text(0.05, 0.95, f"Mean: {np.mean(resolution):.1f}\nMax: {np.max(resolution):.1f}",transform=ax.transAxes,fontsize=12, verticalalignment='top', bbox=box_props)
# plt.show()
# print(f"Mean distance resolution: {np.mean(resolution):.2f} km")

cs_idx = []

for k,v in bwsc_control_stops.items():
    # Find the closest point in the route to the control stop
    distances = np.array([np.linalg.norm(np.array(c) - np.array(v)) for c in coords])
    closest_idx = np.argmin(distances)
    cs_idx.append(closest_idx)
    closest_coord = coords[closest_idx]
    closest_distance = distance[closest_idx]
    closest_heading = headings[closest_idx]
    print(f"{k}: Closest point at {closest_coord}, Distance: {closest_distance:.2f} km, Heading: {closest_heading:.2f}°, Distance to control stop: {distances[closest_idx]:.4f} km")

days = {f"Day {i+1}": coords[cs_idx[2*i]:cs_idx[2*i+1]] for i in range(len(cs_idx)//2)} # Assumed 2 Control stops a day
print(f"Total days: {len(days)}")
print(f"{days['Day 1']}")