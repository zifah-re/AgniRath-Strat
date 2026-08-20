from datetime import datetime, time, date
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import numpy as np
import re

SA_TZ=ZoneInfo("Africa/Johannesburg")
DATES={
  "Day 1": date(2026,9,10),
  "Day 2": date(2026,9,11),
  "Day 3": date(2026,9,12),
  "Day 4": date(2026,9,13),
  "Day 5": date(2026,9,14),
  "Day 6": date(2026,9,15),
  "Day 7": date(2026,9,16),
  "Day 8": date(2026,9,17)
}

folder = Path(r'Fallback Model\Solar')
output_folder = Path(r'Fallback Model\Solar_Processed')
output_folder.mkdir(parents=True, exist_ok=True)


def extractSolarData(json_file):
  with open(json_file, 'r') as f:
    data = json.load(f)
  return data


def trimSolarData(solar_data, start_time=time(8, 0,tzinfo=SA_TZ), end_time=time(17, 0,tzinfo=SA_TZ)):
  trimmed_data = []
  for entry in solar_data:
    entry_time = datetime.fromisoformat(entry['period_end']).time()
    if start_time <= entry_time <= end_time:
      trimmed_data.append(entry)
  return trimmed_data


def meanSolar(solar_data, day, start_time=time(8, 0,tzinfo=SA_TZ), end_time=time(17, 0,tzinfo=SA_TZ)):
  time_dict = {
      f'{m // 60:02d}:{m % 60:02d}': []
      for m in range(
          start_time.hour * 60 + start_time.minute,
          end_time.hour * 60 + end_time.minute + 1,
          5,
      )
  }

  for entry in solar_data:
    entry_time = datetime.fromisoformat(entry['period_end']).astimezone(SA_TZ).time()
    if start_time <= entry_time <= end_time:
      time_str = f'{entry_time.hour:02d}:{entry_time.minute:02d}'
      if time_str in time_dict:
        time_dict[time_str].append(entry)

  for k, v in time_dict.items():
    if v:
      mean_entry = {
          key: float(np.mean([entry[key] for entry in v]))
          for key in v[0]
          if key != 'period_end'
      }
      parsed_time = datetime.strptime(f"{k}:00", "%H:%M:%S").time()
      # Combine today's date with the parsed time and attach SAST timezone
      dt = datetime.combine(DATES[day], parsed_time, tzinfo=SA_TZ)
      mean_entry["period_end"] = dt.isoformat()
      time_dict[k] = mean_entry
    else:
      time_dict[k] = None

  return [v for v in time_dict.values() if v is not None]


for json_file in folder.glob('*.jsonl'):
  file_name = json_file.stem

  stage_point_raw = extractSolarData(json_file)
  output_data=[]
  for item in stage_point_raw:
    stage_point_lat = item['lat']
    stage_point_long = item['lon']
    stage_point_data = item['data']
    day=re.search(r"Day [1-8]",json_file.name).group(0)
    if 'Day 1' in file_name:
      mean_data = meanSolar(stage_point_data, day, time(9, 0,tzinfo=SA_TZ))
    else:
      mean_data = meanSolar(stage_point_data, day)

    output_data.append({
        'lat': stage_point_lat,
        'lon': stage_point_long,
        'data': mean_data,
    })

  output_file = output_folder / f'mean_{json_file.name}'
  with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)