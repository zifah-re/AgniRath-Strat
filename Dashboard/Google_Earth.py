import requests
import re
import sys
import blackboxprotobuf
import base64, struct
from geopy.distance import geodesic
from shapely.geometry import LineString
import numpy as np
import folium
from folium.plugins import Realtime
from folium import JsCode
from pathlib import Path

def main(route_name,new_coordinates):
    url="https://earth.google.com/web/"
    headers={
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-GB,en;q=0.9",
        "Host": "earth.google.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:151.0) Gecko/20100101 Firefox/151.0"
    }
    req=requests.get(url=url,headers=headers)
    if req.status_code!=200:
        sys.exit(0)
    data=req.text
    cookies=req.cookies
    data_version=re.search(r"data-version=\"([^\"]*)",data).group(1)
    url=f"https://earth.google.com/static/multi-threaded/versions/{data_version}/main.dart.js"
    req=requests.get(url=url,headers=headers,cookies=cookies)
    data=req.text
    key_list=re.findall(r"\"(AIzaSy[^\"]*)",data)
    key_list=set(key_list)
    key_list=list(key_list)
    client_token="EgsxMC4xMDkuODEuNRgDIgJJTg"
    server_token="CAMSgwENdYulhx_8yJsdA_eo9gMD2Z4qA-3TAgP6ogEDjt0MA921HgO0mgYD5NgWA6qWAwPdhAgDs9UFA5JuA43YBqYLsKUDA8WdAwOj3QED1-cGA8G4AwP4wA4D0OsBA53GAwOgngQD1pADA76kAwP87g8D85oAAxUKs76QELGqmRz03BGNQg=="
    
    original_line = LineString([(lon, lat) for lat, lon in new_coordinates])
    total_shapely_length = original_line.length
    google_matched_coordinates = []
    target_points = 999

    # 2. Distribute points proportionally across each individual segment
    for i in range(len(new_coordinates) - 1):
        p1 = new_coordinates[i]
        p2 = new_coordinates[i+1]
        
        # Create a mini-line for just this single segment
        segment = LineString([(p1[1], p1[0]), (p2[1], p2[0])])
        
        # Calculate how many of our 999 points belong to this segment's length
        proportion = segment.length / total_shapely_length
        num_points_for_segment = max(2, int(round(proportion * target_points)))
        
        # Interpolate just within this segment (keeps the original vertices locked!)
        seg_distances = np.linspace(0, segment.length, num_points_for_segment)
        
        for dist in seg_distances[:-1]: # Skip the last point to avoid duplicates with the next segment
            pt = segment.interpolate(dist)
            google_matched_coordinates.append((pt.y, pt.x))

    # 3. Explicitly append the absolute final destination point
    google_matched_coordinates.append(new_coordinates[-1])

    # 4. If rounding caused it to be slightly off from 999, trim or pad safely
    while len(google_matched_coordinates) > 999:
        # Remove a point from the middle to preserve start/end
        google_matched_coordinates.pop(len(google_matched_coordinates) // 2)
    
    payload = {
        '2': [
            {'1': lat, '2': lon} for lat, lon in google_matched_coordinates
        ]
    }
    message_type = {
        '2': {
            'type': 'message',
            'name': '',
            'message_typedef': {
                '1': {'type': 'double', 'name': ''},
                '2': {'type': 'double', 'name': ''}
            }
        }
    }
    encoded_payload = blackboxprotobuf.encode_message(payload, message_type=message_type)
    for api_key in key_list:
        url="https://earth-pa.clients6.google.com/v1/earth/elevation?alt=proto"
        headers={
            "Content-Type": "application/x-protobuf",
            "Origin": "https://earth.google.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:151.0) Gecko/20100101 Firefox/151.0",
            "Referer":"https://earth.google.com/",
            "X-Goog-Api-Key": api_key,
            "X-Goog-Earth-Client-Metadata": client_token,
            "X-Goog-Encode-Response-If-Executable": "base64"
            #"X-Server-Token": server_token
        }
        req=requests.post(url=url,data=encoded_payload,headers=headers)
        if not "blocked" in req.text.lower():
            break

    x_distances = [0.0]  # Start point is always 0 km/meters
    total_distance = 0.0

    for i in range(1, len(google_matched_coordinates)):
        # Calculate geodesic distance between consecutive pairs in kilometers
        segment_dist = geodesic(google_matched_coordinates[i-1], google_matched_coordinates[i]).kilometers
        total_distance += segment_dist
        x_distances.append(total_distance)

    # x_distances now holds the exact X-axis coordinates for your plot!
    raw = base64.b64decode(req.text.strip().replace("\n", ""))
    altitude_list=[]
    i = 0
    while i <= len(raw) - 8:
        val = struct.unpack('d', raw[i:i+8])[0]
        # Check for valid elevation range and explicitly skip zero data
        if 0.0 <= val < 9000.0:
            altitude_list.append(val)
            i += 8  # Jump ahead by a full 8-byte float chunk
        else:
            i += 1  # Slide forward to find the next valid stream alignment

    path_coordinates = google_matched_coordinates
    start_lat, start_lon = path_coordinates[0]
    end_lat, end_lon = path_coordinates[-1]

    # 1. Initialize the map centered at your starting coordinates
    m = folium.Map(
        location=[path_coordinates[len(path_coordinates)//2][0], path_coordinates[len(path_coordinates)//2][1]], 
        zoom_start=11,
        tiles=None  # We leave this blank to inject our custom layers manually below
    )
    # 2. Add Google Standard Roadmap Tiles as an alternative toggle option
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google Maps",
        name="Google Standard Roadmap",
        overlay=False,
        control=True
    ).add_to(m)

    # 3. Add Google Hybrid Tiles (Satellite + Road Names & Labels)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid",
        name="Google Satellite (With Roads)",
        overlay=False,
        control=True
    ).add_to(m)
    
    # 4. Draw your 999-point path line (A bright neon green pops nicely on both map styles)
    folium.PolyLine(
        locations=path_coordinates,
        color="#00ff66",       
        weight=5,
        opacity=0.9,
        tooltip=route_name
    ).add_to(m)

    # 5. Drop the Start/End markers
    folium.Marker([start_lat, start_lon], popup="Start Point", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([end_lat, end_lon], popup="End Point", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    # Read your downloaded file and encode it
    SCRIPT_DIR = Path(__file__).resolve().parent

    # 2. Point to the icon relative to this script
    ICON_PATH = SCRIPT_DIR / "prodbuild/navigation_arrow.svg"
    with open(ICON_PATH, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    # Create a data URI string
    # (Change 'image/svg+xml' to 'image/png' if using a PNG)
    base64_icon = f"data:image/svg+xml;base64,{encoded_string}"
    Realtime(
        "http://localhost:8000/api/live-car-gps",  # Your backend URL supplying the coordinate JSON
        get_feature_id="function(f) { return f.properties.id; }",
        point_to_layer=JsCode(f"""
            (f, latlng) => {{
                // 1. Calculate your heading angle
                let angle = (f.properties.bearing !== undefined) ? (f.properties.bearing - 45) : 0;
                
                // 2. Inject your base64 image inside a divIcon wrapper
                let carIcon = L.divIcon({{
                    html: `<img src="{base64_icon}" style="width: 24px; height: 24px; transform: rotate(${{angle}}deg); display: block;" />`,
                    className: 'custom-png-icon', // Keeps Leaflet from adding a white background box
                    iconSize: [24, 24],
                    iconAnchor: [12, 12]
                }});
                
                return L.marker(latlng, {{ icon: carIcon }});
            }}
        """),# 2. CRITICAL FIX: This runs on EVERY incoming real-time data update
        update_feature=JsCode("""
            (feature, oldLayer) => {
                if (!oldLayer) { return; }
                
                // Move coordinates natively
                oldLayer.setLatLng([feature.geometry.coordinates[1], feature.geometry.coordinates[0]]);
                
                if (feature.properties && feature.properties.bearing !== undefined) {
                    let angle = feature.properties.bearing - 45;
                    let container = oldLayer.getElement();
                    if (container) {
                        let img = container.querySelector('img');
                        if (img) {
                            img.style.setProperty('transform', `rotate(${angle}deg)`, 'important');
                        }
                    }
                }
                
                return oldLayer;
            }
        """),
        interval=1500,  # Auto-refresh interval in milliseconds (1.5 seconds)
    ).add_to(m)
    # 6. Include a Layer Control button to alternate views dynamically
    folium.LayerControl(position="topright").add_to(m)
    
    # Save the final product
    return m._repr_html_(),altitude_list,x_distances,google_matched_coordinates