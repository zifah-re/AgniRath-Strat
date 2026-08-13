import numpy as np
import json 
from pathlib import Path
import datetime as _dt

# CAR CONSTANTS #
#constants
MASS_KG = 300.0
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

# SOLVER CONFIGS #
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




folder = Path(r'Model\data\solar')

def extractSolarData(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

weather_logs = {} # Dictionary of keys: stage names, values: dict of keys: (lat, long), values: dict of lists with solar data
stage_names = []

for json_file in folder.glob('*.json'):
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
