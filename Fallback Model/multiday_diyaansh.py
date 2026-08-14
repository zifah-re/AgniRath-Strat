import numpy as np
import json 
from pathlib import Path
import datetime as _dt

#______CAR CONSTANTS_______#
#constants
MASS_KG = 300.0
G_MS2 = 9.81
CRR = 0.007
CDA_M2 = 0.16
AIR_DENSITY = 1.2
ARRAY_AREA_M2 = 5.95
ARRAY_EFFICIENCY = 0.18

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

HALF_BLIND_LOOP_PLACEHOLDER_KM = 14.0 #Change later somehow
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

#____ Solar and Wind_____#
weather_folder = Path(r'Model\data\solar')

def extractSolarData(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

weather_logs = {} # Dictionary of keys: stage names, values: dict of keys: (lat, long), values: dict of lists with solar data
stage_names = []

for json_file in weather_folder.glob('*.json'):
    stage_names.append(json_file.stem)
    stage_data_raw = extractSolarData(json_file)
    stage_data = {}
    for point in stage_data_raw:
        lat = point['historical_weather']['latitude']
        long = point['historical_weather']['longitude']

        dump = point['historical_weather']['hourly']
        stage_data[(lat, long)] = dump
        
    weather_logs[json_file.stem] = stage_data

print(weather_logs[stage_names[0]][list(weather_logs[stage_names[0]].keys())[0]])

#____ Power ____ #

def net_power( ):
    f_drag = 0.5 * AIR_DENSITY * CDA_M2 * (v - v_wind)**2  #Get v_wind from solar/wind above
    f_roll = MASS_KG * G_MS2 * CRR * (1.0 - (grad ** 2) / 2.0) #Implement a def gradient fn
    f_grav = MASS_KG * G_MS2 * grad
    f_acc = MASS_KG* (v_next - v) / dt_s #Implement v_array and get dt_s

    f_total = f_drag + f_roll + f_grav + f_acc
    p_mech = f_total * v

    p_electric = 
