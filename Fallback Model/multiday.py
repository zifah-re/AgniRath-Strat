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
SOC_MAX_PCT = 100.0

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
    "Day 2": {"s1": 71.5, "l":22.6, "s2":231.0},
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
    "Day 1": {"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 1 Boiketlong to Rustenburg","l":"2026 Sasol Solar Challenge Route (Publish)_Day 1 _Rustenburg Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 2 Rustenburg to Swartruggens"},
    "Day 2": {"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 2 Half Blind_11 Sept Stage 1 Swart Ruggens to Zeerust","l":None,"s2":"2026 Sasol Solar Challenge Route (Publish)_Day 2 Half Blind_11 Sept Stage 2 Zeerust to Vryburg"},
    "Day 3": [{"s1": None,"l":"Day 3 probables_Probable Aryaman Day 3_Day 3 Loop","s2":"Day 3 probables_Probable Aryaman Day 3_Stage 2"},
              {"s1":"Day 3 probables_Probable Prahlad Route_Stage 1", "l":"Day 3 probables_Probable Prahlad Route_Day 3 Loop","s2":"Day 3 probables_Probable Prahlad Route_Stage 2" }],
    "Day 4": {"s1":"2026 Sasol Solar Challenge Route (Publish)_Day 4_13 Sept Stage 1 Kimberley to Postmasburg","l":"2026 Sasol Solar Challenge Route (Publish)_Day 4_Postmasburg Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 4_13 Sept Stage 2 Postmasburg to Olifantshoek"},
    "Day 5": {"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 1 Olifantshoek to Upington","l":"2026 Sasol Solar Challenge Route (Publish)_Day 5 _Upington Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 5 _14 Sept Stage 2 Upington to Augrabies"},
    "Day 6": {"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 6 _15 Sept Stage 1 Augrabies to Springbok","l":"2026 Sasol Solar Challenge Route (Publish)_Day 6 _Springbok Loop","s2":None},
    "Day 7": {"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 7_16 Sept Stage 1 Springbok to Van Rhynsdorp","l":"2026 Sasol Solar Challenge Route (Publish)_Day 7_Van Rhynsdorp Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 7_16 Sept Stage 2 Van Rhynsdorp to Clanwilliam"},
    "Day 8": {"s1": "2026 Sasol Solar Challenge Route (Publish)_Day 8_17 Sept Stage 1 Clanwilliam to Ceres","l":"2026 Sasol Solar Challenge Route (Publish)_Day 8_Ceres Loop","s2":"2026 Sasol Solar Challenge Route (Publish)_Day 8_17 Sept Stage 2 Ceres to Paarl"}
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
    weather_logs[json_file.stem] = SolarIrradiance(stage_data,"period_end","PT5M")


#____ Power ____ #
def precompute_solar_gti_factors(time_base, coords_list, heading_profile, altitude_profile):
    """
    Computes solar positions for all horizon steps simultaneously using full vectorization.
    NO loops required.
    """
    coords_arr = np.array(coords_list)
    lats = coords_arr[:, 0]
    lons = coords_arr[:, 1]
    
    tz_times = pd.to_datetime(time_base, unit='s',utc=True).tz_convert('Africa/Johannesburg')
    
    solpos = pvlib.solarposition.get_solarposition(
        tz_times, 
        lats, 
        lons, 
        altitude=np.array(altitude_profile)
    )
    
    apparent_zenith = solpos['apparent_zenith'].values
    azimuth = solpos['azimuth'].values
    zenith_rad = np.radians(apparent_zenith)
    
    aoi = pvlib.irradiance.aoi(
        PANEL_TILT, 
        np.concatenate((np.array(heading_profile),[0])), 
        apparent_zenith, 
        azimuth
    )
    
    tilt_rad = np.radians(PANEL_TILT)
    sky_factor = (1 + np.cos(tilt_rad)) / 2
    ground_factor = (1 - np.cos(tilt_rad)) / 2
    
    a_headings = np.cos(np.radians(aoi)) - (np.cos(zenith_rad) * sky_factor)
    
    b_constants = np.full(len(time_base), sky_factor + ALBEDO * ground_factor)
    
    return a_headings, b_constants

def solar(time_base,coords,headings,altitude,solar_obj):
    a,b=precompute_solar_gti_factors(time_base,coords,headings,altitude)
    dni,ghi=solar_obj[((0,0),time_base)].data(["dni","ghi"])
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
    return soc,power

    
def loops_range(d1,d2,dl,day_1=False):
    if day_1:
        return (floor((8-(d1+d2)/55 - 30/60)/((dl/55)+5/60)),ceil((8-(d1+d2)/75 - 30/60)/((dl/55)+5/60)))
    return (floor((9-(d1+d2)/55 - 30/60)/((dl/55)+5/60)),ceil((9-(d1+d2)/75 - 30/60)/((dl/55)+5/60)))

def stitch_loops(n,t_start,soc_start,solar_obj,coords,altitude,headings,distances,fname,start_date):
    loop_start=t_start+30*60
    loop_soc_start=soc_start + (solar([t_start+15*60],[(0,0)],[headings[0]],[altitude[0]],solar_obj)/(BATTERY_WH*3600))[-1]*100*30*60
    soc_profile=[]
    for i in range(n):
        soc,_=stage_soc_profile(55/3.6,fname,start_date,loop_start,loop_soc_start)
        loop_start=t_start+5*60
        loop_soc_start=soc[-1] + (solar([t_start+2.5*60],[(0,0)],[headings[0]],[altitude[0]],solar_obj)/(BATTERY_WH*3600))[-1]*100*5*60
        soc_profile.append(soc)
    return np.concatenate(soc_profile)

def main():
    v1 = 55/3.6
    v2 = 75/3.6
    loop_bounds=[]
    for day in DAY_DISTANCES:
        s1,l,s2=DAY_DISTANCES[day]['s1'],DAY_DISTANCES[day]['l'],DAY_DISTANCES[day]['s2']
        loop_bounds.append(np.arange(*loops_range(s1,s2,l,False if day!="Day 1" else True)))

        end_soc_s1,power_s1 = stage_soc_profile(v1,DAYWISE_FILES[day]['s1'],start_date,start_time=START_TIME_DAY1_S if day=="Day 1" else START_TIME_OTHER_S,soc_start)

        fig,axes = plt.subplots(1,1,figsize=(10,5))

        
        for i in range(len(loop_bounds[-1])):
            loop_soc,loop_power=stitch_loops(loop_bounds[-1][i],end_soc_s1,solar_obj,coords,altitude,headings,l,DAYWISE_FILES[day]['l'],start_date)

            if not loop_soc[-1]<SOC_MIN_PCT:
                end_soc_s2,power_s2 = stage_soc_profile(v2,DAYWISE_FILES[day]['s2'],start_date,start_time=loop_bounds[-1][i]*60*60 + CONTROL_STOP_DURATION_S + START_TIME_DAY1_S if day=="Day 1" else START_TIME_OTHER_S,soc_start=loop_soc[-1])
                if not end_soc_s2[-1]<SOC_MIN_PCT:
                    loop_length = l
                    for k in range(loop_bounds[-1][i]):
                        loop_length = np.concatenate((loop_length,loop_length[-1]+l))
                    axes.plot(np.concatenate((s1,s1[-1]+loop_length,s1[-1]+loop_length[-1]+s2)),np.concatenate((end_soc_s1,loop_soc,end_soc_s2)),label=f"Loop {loop_bounds[-1][i]}")
                else: print(f"{loop_bounds[-1][i]} loops failed on stage 2 with final SoC {end_soc_s2[-1]:.2f}%")
            else: print(f"{loop_bounds[-1][i]} loops failed on loops with final SoC {loop_soc[-1]:.2f}%")

    
