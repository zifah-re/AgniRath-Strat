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
from traffic import main as traffic_main

def main(route_info,new_coordinates,relevant_points):
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
    key_list=[*{*key_list}]
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
    google_matched_coordinates,speed_limit,eta,d,d1=traffic_main(google_matched_coordinates)
    print(len(google_matched_coordinates))
    print(f"ETA: {eta//3600:.0f} Hour {(eta%3600)/60:.2f} Minutes")
    print(f"Total distance {d/1000:.2f}")
    print(f"Calculated distance {d1/1000:.2f}")
    splits=[]
    if len(google_matched_coordinates)>999:
        k=0
        j=0
        for i in range(int(len(google_matched_coordinates)//999)):
            j=min(999*(i+1),len(google_matched_coordinates))
            splits.append(google_matched_coordinates[k:j])
            k=j
        if k!=len(google_matched_coordinates):
            splits.append(google_matched_coordinates[k:len(google_matched_coordinates)])
    else:
        splits.append(google_matched_coordinates)
    altitude_list=[]
    for coordinates in splits:
        payload = {
            '2': [
                {'1': lat, '2': lon} for lat, lon in coordinates
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
        for i in range(len(key_list)):
            api_key=key_list[i]
            url="https://earth-pa.clients6.google.com/v1/earth/elevation?alt=proto"
            headers={
                "Content-Type": "application/x-protobuf",
                "Origin": "https://earth.google.com",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:151.0) Gecko/20100101 Firefox/151.0",
                "Referer":"https://earth.google.com/",
                "X-Goog-Api-Key": api_key,
                "X-Goog-Encode-Response-If-Executable": "base64"
            }
            req=requests.post(url=url,data=encoded_payload,headers=headers)
            if not "blocked" in req.text.lower():
                break
            else:
                print(f"Trying api key ({i+1}/{len(key_list)})")
        else:
            print("No valid api key found")

        # x_distances now holds the exact X-axis coordinates for your plot!
        raw = base64.b64decode(req.text.strip().replace("\n", ""))
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
    x_distances = [0.0]  # Start point is always 0 km/meters
    total_distance = 0.0

    for i in range(1, len(google_matched_coordinates)):
        # Calculate geodesic distance between consecutive pairs in kilometers
        segment_dist = geodesic(google_matched_coordinates[i-1], google_matched_coordinates[i]).kilometers
        total_distance += segment_dist
        x_distances.append(total_distance)

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
        max_zoom=20,
        overlay=False,
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m@221097413,traffic&x={x}&y={y}&z={z}",
        attr="Google Maps Traffic",
        name="Google Standard with Traffic",
        max_zoom=20,
        overlay=False,
        control=True
    ).add_to(m)
    
    # 3. Add Google Hybrid Tiles (Satellite + Road Names & Labels)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid",
        name="Google Satellite (With Roads)",
        max_zoom=20,
        overlay=False,
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y@221097413,traffic&x={x}&y={y}&z={z}",
        attr="Google Hybrid Traffic",
        name="Google Sattelite with Traffic",
        max_zoom=20,
        overlay=False,
        control=True
    ).add_to(m)
    
    # 4. Draw your 999-point path line (A bright neon green pops nicely on both map styles)
    folium.PolyLine(
        locations=path_coordinates,
        color=route_info["colour"],       
        weight=route_info["width"],
        opacity=route_info["opacity"],
        tooltip=route_info["name"]
    ).add_to(m)

    # 5. Drop the Start/End markers
    for point in relevant_points:
        folium.Marker([point["coordinates"][0], point["coordinates"][1]], popup=point["name"]+":\n"+point["description"] if point["description"] is not None else point["name"],icon=folium.CustomIcon(icon_image=point["url"],icon_size=(32,32),icon_anchor=(32-int(point["anchor"][0]),32-int(point["anchor"][1])),popup_anchor=(0,-32))).add_to(m)
    if len(relevant_points)==0:
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
    return m._repr_html_(),altitude_list,x_distances,google_matched_coordinates,speed_limit
if __name__=="__main__":
    coordinates=[(-26.51561060506015, 29.15905071175074), (-26.51561060506015, 29.15905071175074), (-26.51561060506015, 29.15905071175074), (-26.51601217650888, 29.15892328079135), (-26.51614254609568, 29.16092335445208), (-26.51621305166819, 29.16196244958119), (-26.51630538825174, 29.16313140980347), (-26.51641247051314, 29.16405568268156), (-26.51661571277459, 29.16475116143538), (-26.51707645104253, 29.16558756661656), (-26.51758945638141, 29.16650019079219), (-26.51551000621189, 29.16928905666451), (-26.51448324359882, 29.17070813278564), (-26.51201913121356, 29.17397281028781), (-26.50998420665888, 29.17650216243694), (-26.50818276801434, 29.17785492790017), (-26.50267440340641, 29.18078658689427), (-26.49552064138969, 29.18415047214351), (-26.4932287628756, 29.17952489605989), (-26.49037929618361, 29.17266026719786), (-26.4872855685723, 29.165348900667), (-26.48690703823739, 29.16320889606239), (-26.48544901323299, 29.14460619753257), (-26.47352416899055, 29.14586623622474), (-26.4649948777624, 29.14655580197222), (-26.46141271183122, 29.14683396646904), (-26.45726541515798, 29.14609709293942), (-26.45564406908822, 29.14636357417808), (-26.45435333933633, 29.14698612171006), (-26.45270284715653, 29.14799064668304), (-26.45241561016122, 29.1474625000189), (-26.45210280024332, 29.14712680746435), (-26.45086821071083, 29.14657049025843), (-26.44981076041877, 29.14586913731243), (-26.44966717941508, 29.14574656602631), (-26.44741736101432, 29.14400137472462), (-26.44549664806913, 29.14244543865783), (-26.44269813085265, 29.13994935488419), (-26.44103742533035, 29.1376008319677), (-26.43905173926475, 29.13326101404693), (-26.43752810967054, 29.12957829054423), (-26.43711851224293, 29.12770304450848), (-26.43705104143812, 29.12577591040332), (-26.4375101509793, 29.12060864956961), (-26.43742974553244, 29.11782831395491), (-26.43405110504559, 29.10448732622449), (-26.43092479654501, 29.09231185000948), (-26.42835122977216, 29.08189444853267), (-26.42587919055832, 29.07248171147186), (-26.42532619729885, 29.0676999076864), (-26.42494656132974, 29.06093299522433), (-26.42393850389036, 29.04733829450655), (-26.42336863125849, 29.04113501252515), (-26.42310931457374, 29.03640943592308), (-26.42198977067883, 29.03035931842517), (-26.41940216402596, 29.01895055765578), (-26.41658772878308, 29.00725744541327), (-26.413622058871, 28.9944882692941), (-26.4105257541463, 28.98543637344452), (-26.40623353463845, 28.97312738248096), (-26.40269561376254, 28.96307582284627), (-26.39929641192644, 28.95312464526524), (-26.39679133381799, 28.94255492505394), (-26.39448510536825, 28.93018744648437), (-26.39406638042541, 28.92569301202677), (-26.39449264671162, 28.91983045294424), (-26.39589038829273, 28.91521537500821), (-26.39801753815498, 28.90907331851402), (-26.40010356303154, 28.90245907671688), (-26.40146430533214, 28.89518159226631), (-26.40158424852752, 28.89009447855519), (-26.40115783049778, 28.8805657366483), (-26.39779490063417, 28.86247928374413), (-26.39514760685959, 28.85087123792897), (-26.38933944010553, 28.83242093624747), (-26.38372277136403, 28.81352816759069), (-26.37284282166573, 28.7808137932819), (-26.36904464793914, 28.75486656449307), (-26.3660362334067, 28.72369361201217), (-26.35853240808074, 28.65826921822705), (-26.3208676427085, 28.59270695403004), (-26.32073190867116, 28.59193020013897), (-26.32067888435585, 28.59092564083987), (-26.32013448302856, 28.59001365589035), (-26.32901906949195, 28.57544395056194), (-26.34182850337608, 28.56548112189677), (-26.36118172175404, 28.5507937362128), (-26.36503993999657, 28.54700726270758), (-26.37661191778599, 28.5247721573496), (-26.38151244662698, 28.51423513745923), (-26.38347834481209, 28.51114669075363), (-26.38638400985548, 28.50780150168003), (-26.39168700437152, 28.49446341504588), (-26.39312893752895, 28.49283725145925), (-26.40904145335723, 28.47746186545819), (-26.41561738138012, 28.46815252850903), (-26.41653650335648, 28.46557546426729), (-26.41537814271326, 28.45652628777339), (-26.42141353019552, 28.45589923786444), (-26.42478978834824, 28.45452623571128), (-26.42841202154511, 28.45122019569349), (-26.43307156752106, 28.4441208098157), (-26.44059283468613, 28.43628795019836), (-26.45981842625743, 28.42305658364536), (-26.4700721008577, 28.41556936356356), (-26.47402896825163, 28.41059081787499), (-26.47751977381122, 28.40344135066268), (-26.4807256935071, 28.39330295171281), (-26.4833969034747, 28.38803551619809), (-26.49132910600995, 28.37307378903221), (-26.49393259130607, 28.3672170065683), (-26.49436926852399, 28.36693437736358), (-26.4950563464969, 28.36677297787011), (-26.49614642745349, 28.36702997583308), (-26.49866869221546, 28.36787463749098), (-26.498812407454, 28.36565512055382), (-26.49885085850737, 28.36487118785316), (-26.49868782450891, 28.36487312293744)]