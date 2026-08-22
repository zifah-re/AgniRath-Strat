import numpy as np
import json 
from pathlib import Path
import datetime as _dt
from solar_table import SolarIrradiance,SolarResultProxy
import pvlib
import pandas as pd
from datetime import datetime, time, date
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from math import ceil,floor


#______CAR CONSTANTS_______#
#constants
MASS_KG = 300.0
G_MS2 = 9.81
CRR = 0.007
CDA_M2 = 0.16
AIR_DENSITY = 1.2
ARRAY_AREA_M2 = 5.95
ARRAY_EFFICIENCY = 0.18
PANEL_TILT=4
ALBEDO=0.2

#drivetrain
MOTOR_EFF = 0.95
REGEN_EFF = 0.70
P_LOSS = 70

#battery
BATTERY_WH = 588.0*6
SOC_MIN_PCT = 20.0
SOC_MAX_PCT = 95.0

#speed/accel

V_MAX_MS = 85.0/3.6
A_MAX_MS = 0.5

#____ROUTE______#
START_TIME_DAY1_S = 9 * 3600
START_TIME_OTHER_S = 8 * 3600
FINISH_TIME_S = 17 * 3600  
FINISH_CUTOFF_ABS_S = 17 * 3600 + 30*60

DAY8_TIMED_FINISH_S = 15 * 3600
DAY8_PROCEEDINGS_END_S = 17 * 3600

CONTROL_STOP_DURATION_S = 30 * 60
LOOP_STOP_DURATION_S = 5 * 60  

CONSTANT_VELOCITY_MS = 60/3.6
LOOP_CRUISE_SPEED_MS = 55/3.6

SA_TZ=ZoneInfo("Africa/Johannesburg")
HALF_BLIND_LOOP_PLACEHOLDER_KM = 14.0 #Change later somehow
DAY_DISTANCES={
    "Day 1": {"s1": 172.7, "l":22.6, "s2":65.6},
    "Day 2": {"s1": 71.5, "l":18.5, "s2":231.0},
    "Day 3": {"s1": 8.0, "l":39.9, "s2":208.0},
    "Day 4": {"s1": 197.0, "l":21.0, "s2":63.3},
    "Day 5": {"s1": 178.0, "l":60.7, "s2":114.0},
    "Day 6": {"s1": 310.0, "l":18.2, "s2":0.0},
    "Day 7": {"s1": 261.0, "l":16.5, "s2":80.9},
    "Day 8": {"s1": 180.0, "l":21.8, "s2":98.3}
}
DAY_ROUTE_NOTES = [
    # day 1
    dict(stage1_km=172.7, loops=[("rustenburg_loop", 22.6)], stage2_km=65.6,
         start="Boiketlong Hall (Sasolburg)",
         control_stop="Rustenburg Highschool",
         finish="Swartruggens Dam"),
    # day 2 — HALF BLIND: loop unknown until Golden Envelope (SR 2.21)
    dict(stage1_km=71.5, loops=None, stage2_km=231.0,
         start="Swartruggens Dam", control_stop="Zeerust Highschool",
         finish="Kameelboom Lodge (Vryburg)"),
    # day 3 — FULL BLIND: everything unknown until Golden Envelope (SR 2.21)
    dict(stage1_km=None, loops=None, stage2_km=None,
         start="Kameelboom Lodge (Vryburg)", control_stop=None,
         finish="Kimberley Technical Highschool"),
    # day 4
    dict(stage1_km=197.0, loops=[("postmasburg_loop2", 21.0),
                                 ("postmasburg_loop1", 14.0)], stage2_km=63.3,
         start="Kimberley Technical Highschool",
         control_stop="Postmasburg Highschool",
         finish="Ranch Chalets (Olifantshoek)"),
    # day 5
    dict(stage1_km=178.0, loops=[("upington_loop1", 62.0),
                                 ("upington_loop2", 34.0)], stage2_km=114.0,
         start="Ranch Chalets (Olifantshoek)",
         control_stop="Upington Highschool",
         finish="Kameeldoring Campsite (Augrabies)"),
    # day 6 — control stop location == finish location
    dict(stage1_km=310.0, loops=[("springbok_loop", 18.2)], stage2_km=0.0,
         start="Kameeldoring Campsite (Augrabies)",
         control_stop="Namakwa Highschool (Springbok)",
         finish="Namakwa Highschool (Springbok)"),
    # day 7
    dict(stage1_km=261.0, loops=[("vanrhynsdorp_loop", 16.5)], stage2_km=80.9,
         start="Namakwa Highschool (Springbok)",
         control_stop="Vanrhynsdorp Highschool",
         finish="Augsburg Landbougimnasium (Clanwilliam)"),
    # day 8 — timed finish 15H00 (SR 2.22.4)
    dict(stage1_km=180.0, loops=[("ceres_loop", 21.8)], stage2_km=98.3,
         start="Augsburg Landbougimnasium (Clanwilliam)",
         control_stop="Charlie Hofmeyer Highschool (Ceres)",
         finish="Suid Agter Paarl Road (Paarl)"),
]
DAYWISE_FILES={
    "Day 1": {"date":date(2026,9,10),"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 1 Boiketlong to Rustenburg","l":"2026 Sasol Solar Challenge Route (Publish)_Day 1 _Rustenburg Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 2 Rustenburg to Swartruggens"},
    "Day 2": {"date":date(2026,9,11),"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 2 Half Blind_11 Sept Stage 1 Swart Ruggens to Zeerust","l":"SSC ROUTE FINAL_Day 2 Half Blind_Day 2 Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 2 Half Blind_11 Sept Stage 2 Zeerust to Vryburg"},
    "Day 3": [{"date":date(2026,9,12),"s1": None,"l":"Day 3 probables_Probable Aryaman Day 3_Day 3 Loop","s2":"Day 3 probables_Probable Aryaman Day 3_Stage 2"},
              {"date":date(2026,9,12),"s1":"Day 3 probables_Probable Prahlad Route_Stage 1", "l":"Day 3 probables_Probable Prahlad Route_Day 3 Loop","s2":"Day 3 probables_Probable Prahlad Route_Stage 2" }][1],
    "Day 4": {"date":date(2026,9,13),"s1":"2026 Sasol Solar Challenge Route (Publish)_Day 4_13 Sept Stage 1 Kimberley to Postmasburg","l":"2026 Sasol Solar Challenge Route (Publish)_Day 4_Postmasburg Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 4_13 Sept Stage 2 Postmasburg to Olifantshoek"},
    "Day 5": {"date":date(2026,9,14),"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 1 Olifantshoek to Upington","l":"2026 Sasol Solar Challenge Route (Publish)_Day 5 _Upington Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 2 Upington to Augrabies"},
    "Day 6": {"date":date(2026,9,15),"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 6 _15 Sept Stage 1 Augrabies to Springbok","l":"2026 Sasol Solar Challenge Route (Publish)_Day 6 _Springbok Loop","s2":None},
    "Day 7": {"date":date(2026,9,16),"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 7_16 Sept Stage 1 Springbok to Van Rhynsdorp","l":"2026 Sasol Solar Challenge Route (Publish)_Day 7_Van Rhynsdorp Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 7_16 Sept Stage 2 Van Rhynsdorp to Clanwilliam"},
    "Day 8": {"date":date(2026,9,17),"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 8_17 Sept Stage 1 Clanwilliam to Ceres","l":"2026 Sasol Solar Challenge Route (Publish)_Day 8_Ceres Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 8_17 Sept Stage 2 Ceres to Paarl"}
}

#____ Solar and Wind_____#
weather_folder = Path(r'Solar_Processed')

def extractSolarData(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

weather_logs = {} # Dictionary of keys: stage names, values: dict of keys: (lat, long), values: dict of lists with solar data
stage_names = []

for json_file in weather_folder.glob('*.jsonl'):
    stage_names.append(json_file.stem)
    stage_data_raw = extractSolarData(json_file)
    stage_data = {}
    for point in stage_data_raw:
        lat = point['lat']
        long = point['lon']
        dump = point['data']
        stage_data[(lat, long)] = dump
    weather_logs[json_file.stem] = SolarIrradiance(stage_data,"period_end","PT5M",6)


#____ Power ____ #
def precompute_solar_gti_factors(time_base, coords_list, heading_profile, altitude_profile):
    """
    Computes solar positions for all horizon steps simultaneously using full vectorization.
    NO loops required.
    """
    time_len = len(time_base)
    coords_arr = np.array(coords_list)
    
    if coords_arr.ndim == 1:
        lats = np.full(time_len, coords_arr[0])
        lons = np.full(time_len, coords_arr[1])
    else:
        lats = coords_arr[:, 0]
        lons = coords_arr[:, 1]
        
    alt_arr = np.full(time_len, altitude_profile) if np.isscalar(altitude_profile) or len(np.atleast_1d(altitude_profile)) == 1 else np.array(altitude_profile)
    
    tz_times = pd.to_datetime(time_base, unit='s', utc=True).tz_convert('Africa/Johannesburg')
    
    solpos = pvlib.solarposition.get_solarposition(
        tz_times, 
        lats, 
        lons, 
        altitude=alt_arr
    )
    
    apparent_zenith = solpos['apparent_zenith'].values
    azimuth = solpos['azimuth'].values
    zenith_rad = np.radians(apparent_zenith)

    headings_arr = np.array(heading_profile)
    if len(headings_arr) == time_len - 1:
        headings_arr = np.append(headings_arr, headings_arr[-1])
    elif len(headings_arr) != time_len:
        headings_arr = np.resize(headings_arr, time_len)
        
    aoi = pvlib.irradiance.aoi(
        PANEL_TILT, 
        headings_arr, 
        apparent_zenith, 
        azimuth
    )
    
    tilt_rad = np.radians(PANEL_TILT)
    sky_factor = (1 + np.cos(tilt_rad)) / 2
    ground_factor = (1 - np.cos(tilt_rad)) / 2
    
    a_headings = np.cos(np.radians(aoi)) - (np.cos(zenith_rad) * sky_factor)
    b_constants = np.full(time_len, sky_factor + ALBEDO * ground_factor)
    
    return a_headings, b_constants

def solar(time_base,coords,headings,altitude,solar_obj):
    a,b=precompute_solar_gti_factors(time_base,coords,headings,altitude)
    dni,ghi=solar_obj[(coords,time_base)].data(["dni","ghi"])
    gti=a*dni + b*ghi
    solar_irr=gti*ARRAY_EFFICIENCY*ARRAY_AREA_M2
    return solar_irr

def net_power(v,grad,solar):
    grad=grad/100
    f_drag = 0.5 * AIR_DENSITY * CDA_M2 * (v)**2  #Get v_wind from solar/wind above
    f_roll = MASS_KG * G_MS2 * CRR * (1.0 - (grad ** 2) / 2.0) #Implement a def gradient fn
    f_grav = MASS_KG * G_MS2 * grad

    f_total = f_drag + f_roll + f_grav
    p_mech = f_total * v

    p_mech = np.where(p_mech < 0, p_mech * REGEN_EFF,  p_mech / MOTOR_EFF)
    p_electric = solar - p_mech - 50
    return p_electric

def stage_soc_profile(v,fname,start_date,start_time,soc_start):
    solar_obj=weather_logs[f'mean_{fname}']
    route=extractSolarData(f'Saves/{fname}.kml.save')['profile']
    coords,headings,altitude,distances,grad=np.array(route['Coordinates']),np.array(route['Headings']),np.array(route['Altitude']),np.array(route['Distance']),np.array(route['Gradient'])
    distances=distances*1000
    dx=np.diff(distances)
    dt=dx/v
    dt=np.concatenate(([0],dt))
    time_base=start_time + dt.cumsum()
    solar_irr=solar(time_base,coords,headings,altitude,solar_obj)
    v_array=np.full(len(time_base),v)
    power=net_power(v_array,grad,solar_irr)
    energy=power*np.concatenate((dx/v,[0]))
    soc=soc_start+(np.cumsum(energy/(BATTERY_WH*3600)))*100
    '''fig,ax=plt.subplots(1,3,figsize=(10,5))
    ax[0].plot(distances,power,color="steelblue")
    ax[0].set_title("Power")
    ax[1].plot(distances,soc,color="tomato")
    ax[1].set_title("SoC")
    ax[2].plot(distances,solar_irr,color="seagreen")
    ax[2].set_title("Solar")
    plt.show()'''
    return soc,power,time_base

    
def loops_range(d1,d2,dl,day_1=False):
    if day_1:
        return (0,ceil((8-(d1+d2)/75 - 30/60)/((dl/55)+5/60)))
    return (0,ceil((9-(d1+d2)/75 - 30/60)/((dl/55)+5/60)))

def no_of_loops(v1,v2,d1,d2,dl,day_1=False):
    if day_1:
        return floor((8-(d1)/v1-(d2)/v2 - 30/60)/((dl/55)+5/60))
    return floor((9 -d1/v1 - d2/v2 - 30/60)/((dl/55)+5/60))

def stitch_loops(n,t_start,soc_start,solar_obj,coords,altitude,headings,distances,fname,start_date):
    loop_start=t_start+30*60
    loop_soc_start=soc_start + (solar([t_start+15*60],[coords[0]],[headings[0]],[altitude[0]],solar_obj)/(BATTERY_WH*3600))[-1]*100*30*60
    soc_profile=[]
    end_time=loop_start
    for i in range(n):
        soc,_,end_time=stage_soc_profile(55/3.6,fname,start_date,loop_start,loop_soc_start)
        loop_start=end_time+5*60
        loop_soc_start=soc[-1] + (solar([end_time+2.5*60],[coords[0]],[headings[0]],[altitude[0]],solar_obj)/(BATTERY_WH*3600))[-1]*100*5*60
        soc_profile.append(soc)
    if len(soc_profile)>0:
        return np.concatenate(soc_profile),end_time
    else:
       return np.array([loop_soc_start]),end_time


def charged(soc, start_ts, end_ts, coords, heading, altitude, solar_obj):
    """Stationary charging model integrating solar yield over time."""
    if start_ts >= end_ts:
        return soc

    dt = 300.0  # 5-minute step in seconds
    time_base = np.arange(start_ts, end_ts, dt)
    n_points = len(time_base)

    if n_points == 0:
        return soc

    c_array = np.tile(coords, (n_points, 1)) if np.ndim(coords) == 1 else np.array([coords] * n_points)
    h_array = np.full(n_points, heading, dtype=float)
    a_array = np.full(n_points, altitude, dtype=float)

    solar_pwr = solar(time_base, c_array, h_array, a_array, solar_obj)
    net_pwr = np.maximum(solar_pwr - 10.0, 0.0) # 10 W loss
    energy_wh = np.sum(net_pwr * dt) / 3600.0
    soc_gained = (energy_wh / BATTERY_WH) * 100.0
    return min(SOC_MAX_PCT, soc + soc_gained)

def local_time_of_day(ts):
    return datetime.fromtimestamp(ts, tz=SA_TZ).time()

def stitchAllDays():
    """Simulates multi-day solar vehicle race profile across 8 days with daily summaries."""
    day_names = [f"Day {i}" for i in range(1, 9)]

    # Loop counts per day
    n = [0, 0, 0, 0, 0, 0, 0, 0]

    # Target speeds in km/h -> [stage1, loop, stage2]
    v = {
        "Day 1": [70, 55, 70],
        "Day 2": [60, 55, 60],
        "Day 3": [60, 55, 60],
        "Day 4": [55, 55, 55],
        "Day 5": [60, 55, 60],
        "Day 6": [55, 55, 55],
        "Day 7": [55, 55, 55],
        "Day 8": [50, 55, 50],
    }

    soc = 95.0  # Starting SoC on Day 1 morning (no charge added on Day 1 morning)
    x_all = []
    y_all = []
    day_boundaries = []
    cumulative_distance = 0.0

    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    
    print("\n" + "=" * 65)
    print("           8-DAY RACE SIMULATION SUMMARY")
    print("=" * 65)

    for day_idx, day in enumerate(day_names):
        day_no = day_idx + 1
        start_date = DAYWISE_FILES[day]["date"]

        s1_name = DAYWISE_FILES[day]["s1"]
        route_s1 = extractSolarData(f"Saves/{s1_name}.kml.save")["profile"]
        distances_s1 = np.array(route_s1["Distance"])
        s1_solar_obj = weather_logs[f"mean_{s1_name}"]

        l_name = DAYWISE_FILES[day]["l"]
        l_dist = DAY_DISTANCES[day]["l"]
        solar_obj_l = weather_logs[f"mean_{l_name}"]
        route_loop = extractSolarData(f"Saves/{l_name}.kml.save")["profile"]
        coords_l, headings_l, altitude_l = (
            np.array(route_loop["Coordinates"]),
            np.array(route_loop["Headings"]),
            np.array(route_loop["Altitude"]),
        )
        single_loop_dist = np.array(route_loop["Distance"])

        s2_name = DAYWISE_FILES[day].get("s2")
        try:
            route_s2 = extractSolarData(f"Saves/{s2_name}.kml.save")["profile"]
            distances_s2 = np.array(route_s2["Distance"])
            s2_solar_obj = weather_logs[f"mean_{s2_name}"]
            has_s2 = True
        except Exception:
            route_s2 = None
            distances_s2 = np.array([0.0])
            s2_solar_obj = None
            has_s2 = False


        # Morning Charging
        if day_no == 1:
            morning_soc_gained = 0.0
        else:
            m_coords = route_s1["Coordinates"][0]
            m_heading = route_s1["Headings"][0]
            m_alt = route_s1["Altitude"][0]

            morning_start_ts = datetime.combine(start_date, time(6, 0), tzinfo=SA_TZ).timestamp()
            morning_end_ts = datetime.combine(start_date, time(8, 0), tzinfo=SA_TZ).timestamp()

            soc_after_m_charge = charged(soc, morning_start_ts, morning_end_ts, m_coords, m_heading, m_alt, s1_solar_obj)
            morning_soc_gained = soc_after_m_charge - soc
            soc = soc_after_m_charge

        start_time_ts = datetime.combine(
            start_date,
            datetime.strptime("09:00:00" if day_no == 1 else "08:00:00", "%H:%M:%S").time(),
            tzinfo=SA_TZ,
        ).timestamp()

        v_s1_ms = v[day][0] / 3.6
        soc_s1, power_s1, time_s1 = stage_soc_profile(v_s1_ms, s1_name, start_date, start_time_ts, soc)
        s1_end_time = time_s1[-1]
        s1_end_str = datetime.fromtimestamp(s1_end_time, tz=SA_TZ).strftime("%H:%M:%S")

        last_coords = route_s1["Coordinates"][-1]
        last_heading = route_s1["Headings"][-1]
        last_alt = route_s1["Altitude"][-1]
        last_solar_obj = s1_solar_obj

        cs_start_ts = s1_end_time
        cs_end_ts = cs_start_ts + 1800.0 
        soc_before_cs = soc_s1[-1]
        soc_after_cs = charged(soc_before_cs, cs_start_ts, cs_end_ts, last_coords, last_heading, last_alt, last_solar_obj)
        cs_soc_gained = soc_after_cs - soc_before_cs

        current_time = cs_end_ts
        current_soc = soc_after_cs

        n_loops = n[day_idx]
        loop_soc_list = []
        loop_gaps_soc_gained = 0.0

        if n_loops > 0:
            v_loop_ms = v[day][1] / 3.6
            for k in range(n_loops):
                single_loop_soc, loop_drive_end_time = stitch_loops(
                    1,
                    current_time,
                    current_soc,
                    solar_obj_l,
                    coords_l,
                    altitude_l,
                    headings_l,
                    l_dist,
                    l_name,
                    start_date,
                    v_loop_ms
                )
                loop_soc_list.append(single_loop_soc)
                soc_post_drive = single_loop_soc[-1]

                # 5-minute gap stationary charge 
                gap_start_ts = loop_drive_end_time
                gap_end_ts = gap_start_ts + 300.0  

                last_coords = coords_l[-1]
                last_heading = headings_l[-1]
                last_alt = altitude_l[-1]
                last_solar_obj = solar_obj_l

                soc_post_gap = charged(
                    soc_post_drive, gap_start_ts, gap_end_ts, last_coords, last_heading, last_alt, last_solar_obj
                )
                loop_gaps_soc_gained += (soc_post_gap - soc_post_drive)

                current_soc = soc_post_gap
                current_time = gap_end_ts

            loop_soc = np.concatenate(loop_soc_list)
        else:
            loop_soc = np.array([])

        if has_s2 and s2_name is not None:
            s2_start_str = datetime.fromtimestamp(current_time, tz=SA_TZ).strftime("%H:%M:%S")
            v_s2_ms = v[day][2] / 3.6

            soc_s2, power_s2, time_s2 = stage_soc_profile(
                v_s2_ms, s2_name, start_date, current_time, soc_start=current_soc
            )
            day_end_time = time_s2[-1]
            s2_finish_str = datetime.fromtimestamp(day_end_time, tz=SA_TZ).strftime("%H:%M:%S")

            last_coords = route_s2["Coordinates"][-1]
            last_heading = route_s2["Headings"][-1]
            last_alt = route_s2["Altitude"][-1]
            last_solar_obj = s2_solar_obj
        else:
            soc_s2 = np.array([])
            day_end_time = current_time
            s2_start_str = datetime.fromtimestamp(current_time, tz=SA_TZ).strftime("%H:%M:%S")
            s2_finish_str = "N/A (No Stage 2)"

        if n_loops > 0:
            distances_loops_stacked = np.concatenate(
                [single_loop_dist + k * l_dist for k in range(n_loops)]
            )
            x_loops = distances_s1[-1] + distances_loops_stacked
            x_s2 = x_loops[-1] + distances_s2 if (has_s2 and len(soc_s2) > 0) else np.array([])

            x_day = np.concatenate([x for x in (distances_s1, x_loops, x_s2) if len(x) > 0])
            y_day = np.concatenate([y for y in (soc_s1, loop_soc, soc_s2) if len(y) > 0])
        else:
            x_s2 = distances_s1[-1] + distances_s2 if (has_s2 and len(soc_s2) > 0) else np.array([])

            x_day = np.concatenate([x for x in (distances_s1, x_s2) if len(x) > 0])
            y_day = np.concatenate([y for y in (soc_s1, soc_s2) if len(y) > 0])

        x_all.append(cumulative_distance + x_day)
        y_all.append(y_day)

        km_covered_today = x_day[-1]
        cumulative_distance += km_covered_today
        day_boundaries.append(cumulative_distance)

        final_soc = y_day[-1] if len(y_day) > 0 else current_soc

        evening_end_ts = datetime.combine(start_date, time(17, 0), tzinfo=SA_TZ).timestamp()
        soc_before_e_charge = final_soc

        if day_end_time < evening_end_ts:
            final_soc = charged(final_soc, day_end_time, evening_end_ts,
                                last_coords, last_heading, last_alt, last_solar_obj)

        evening_soc_gained = final_soc - soc_before_e_charge
        soc = final_soc

        print(f"\n--- {day} ({start_date}) ---")
        print(f"  • Distance Covered:           {km_covered_today:.2f} km")
        print(f"  • Stage 1 Finish Time:        {s1_end_str}")
        print(f"  • 30-Min Stop SoC Gain:       +{cs_soc_gained:.2f}%")
        if n_loops > 0:
            print(f"  • Loop Gaps SoC Gain:         +{loop_gaps_soc_gained:.2f}% ({n_loops} x 5 min gaps)")
        print(f"  • Stage 2 Start Time:         {s2_start_str}")
        print(f"  • Stage 2 Finish Time:        {s2_finish_str}")
        print(f"  • Morning SoC Gain:           +{morning_soc_gained:.2f}%")
        print(f"  • Evening SoC Gain:           +{evening_soc_gained:.2f}%")
        print(f"  • End of Day Final SoC:       {final_soc:.2f}%")

        if final_soc < SOC_MIN_PCT:
            print(f"\n[CRITICAL WARNING] {day}: SoC fell below minimum limit ({final_soc:.2f}%) — stopping simulation.")
            break

    print("\n" + "=" * 65)

    x_full = np.concatenate(x_all)
    y_full = np.concatenate(y_all)

    axes.plot(x_full, y_full, color="tomato", linewidth=2, label="SoC (%)")

    for b_idx, boundary in enumerate(day_boundaries[:-1]):
        axes.axvline(
            x=boundary,
            color="black",
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
            label="Day Separator" if b_idx == 0 else ""
        )

    axes.set_xlabel("Distance (km)")
    axes.set_ylabel("State of Charge (%)")
    axes.set_title("8-Day SoC Profile")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="upper right")
    plt.show()

    return x_full, y_full

def main():
    day_no = 6
    v1 = [70 / 3.6,60/3.6,60/3.6,55/3.6,60/3.6,55/3.6,55/3.6,50/3.6][day_no-1]
    v2 = [70 / 3.6,60/3.6,60/3.6,55/3.6,60/3.6,55/3.6,55/3.6,50/3.6][day_no-1]
    soc_start = [95,82,63,53,67,67,53,53][day_no-1]
    loop_bounds = []

    for day in [f"Day {day_no}"]:
        s1, l, s2,start_date = (
            DAY_DISTANCES[day]["s1"],
            DAY_DISTANCES[day]["l"],
            DAY_DISTANCES[day]["s2"],
            DAYWISE_FILES[day]["date"]
        )
        loop_bounds.append(
            np.arange(
                *loops_range(s1, s2, l, False if day != "Day 1" else True)
            )
        )
        print(f"At this speed {no_of_loops(v1*3.6,v2*3.6,s1,s2,l, False if day != "Day 1" else True)} loops are possible")
        end_soc_s1, power_s1, end_time = stage_soc_profile(
            v1,
            DAYWISE_FILES[day]["s1"],
            start_date,
            datetime.combine(
                start_date,
                datetime.strptime("09:00:00", "%H:%M:%S").time(),
                tzinfo=SA_TZ,
            ).timestamp()
            if day == "Day 1"
            else datetime.combine(
                start_date,
                datetime.strptime("08:00:00", "%H:%M:%S").time(),
                tzinfo=SA_TZ,
            ).timestamp(),
            soc_start,
        )

        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        
        solar_obj = weather_logs[f"mean_{DAYWISE_FILES[day]['l']}"]

        # Extract base route profiles once
        route_loop = extractSolarData(f"Saves/{DAYWISE_FILES[day]['l']}.kml.save")[
            "profile"
        ]
        coords, headings, altitude = (
            np.array(route_loop["Coordinates"]),
            np.array(route_loop["Headings"]),
            np.array(route_loop["Altitude"]),
        )
        single_loop_dist = np.array(route_loop["Distance"])

        route_s1 = extractSolarData(f"Saves/{DAYWISE_FILES[day]['s1']}.kml.save")[
            "profile"
        ]
        distances_s1 = np.array(route_s1["Distance"])

        try:
            route_s2 = extractSolarData(f"Saves/{DAYWISE_FILES[day]['s2']}.kml.save")[
                "profile"
            ]
            distances_s2 = np.array(route_s2["Distance"])
        except:
            distances_s2=np.array([0])

        for i in range(len(loop_bounds[-1])):
            n_loops = loop_bounds[-1][i]

            loop_soc, end_loop_time = stitch_loops(
            n_loops,
            end_time,
            end_soc_s1[-1],
            solar_obj,
            coords,
            altitude,
            headings,
            l,
            DAYWISE_FILES[day]["l"],
            start_date,
            )

            if len(loop_soc)>1 and not loop_soc[-1] < SOC_MIN_PCT:
                if DAYWISE_FILES[day]['s2'] is not None:
                    end_soc_s2, power_s2, day_end_time = stage_soc_profile(
                        v2,
                        DAYWISE_FILES[day]["s2"],
                        start_date,
                        end_loop_time,
                        soc_start=loop_soc[-1],
                    )
                else:
                    end_soc_s2=[loop_soc[-1]]
                    day_end_time=end_loop_time

                if not end_soc_s2[-1] < SOC_MIN_PCT:
                # --- CORRECT DISTANCE CONCATENATION ---
                    if n_loops > 0:
                        # Concatenate n copies of loop distance shifted by k * loop_length
                        distances_loops_stacked = np.concatenate(
                            [single_loop_dist + k * l for k in range(n_loops)]
                        )
                        x_loops = distances_s1[-1] + distances_loops_stacked
                        x_s2 = x_loops[-1] + distances_s2
                        x_total = np.concatenate((distances_s1, x_loops, x_s2))
                    else:
                        x_s2 = distances_s1[-1] + distances_s2
                        x_total = np.concatenate((distances_s1, x_s2))

                    y_total = np.concatenate((end_soc_s1, loop_soc, end_soc_s2))

                    axes.plot(x_total, y_total, label=f"{n_loops} Loops")
                else:
                    print(
                        f"{n_loops} loops failed on stage 2 with final SoC"
                        f" {end_soc_s2[-1]:.2f}%"
                    )
            elif len(loop_soc)==1:
                if DAYWISE_FILES[day]['s2'] is not None:
                    end_soc_s2, power_s2, day_end_time = stage_soc_profile(
                    v2,
                    DAYWISE_FILES[day]["s2"],
                    start_date,
                    end_loop_time,
                    soc_start=loop_soc[-1],
                    )
                else:
                    end_soc_s2=[loop_soc[-1]]
                    day_end_time=end_loop_time

                if not end_soc_s2[-1] < SOC_MIN_PCT:
                    # --- CORRECT DISTANCE CONCATENATION ---
                    
                    x_s2 = distances_s1[-1] + distances_s2
                    x_total = np.concatenate((distances_s1, x_s2))
                    
                    y_total = np.concatenate((end_soc_s1, end_soc_s2))
                    
                    axes.plot(x_total, y_total, label=f"{n_loops} Loops")
                else:
                    print(
                        f"{n_loops} loops failed on stage 2 with final SoC"
                        f" {end_soc_s2[-1]:.2f}%"
                    )
            
            else:
                print(
                    f"{n_loops} loops failed on loops with final SoC"
                    f" {loop_soc[-1]:.2f}%"
                )

        axes.set_xlabel("Distance (km)")
        axes.set_ylabel("State of Charge (%)")
        axes.set_title(f"SoC Profile - {day}")
        axes.legend()
        axes.grid(True)
        plt.show()
    
stitchAllDays()