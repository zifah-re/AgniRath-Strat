import numpy as np
import json 
from pathlib import Path

folder = Path('Model\data\solar')

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
