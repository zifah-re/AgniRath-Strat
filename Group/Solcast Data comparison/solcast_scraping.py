import requests
from datetime import datetime, time
from zoneinfo import ZoneInfo
import json

API_KEY = "AzBIT5zz2EEovhMXcHF7tQKUrteydtgL"
ARYAMAN_API_KEY="ncL0tw6DjwP8uZGbRHpHBCORHZMIsTef"
DIYAANSH_API_KEY="g4qlpjja-n3P0GKWMnG5jUelXmGGZXZH"
LAT = 13.037206951836724
LON = 79.89299347657726

if not API_KEY:
    API_KEY = input("Enter your API key: ")
if not LAT:
    LAT = float(input("Enter latitude: "))
if not LON:
    LON = float(input("Enter longitude: "))

start_date = input("Enter start date(DD-MM-YYYY): ")
parsed_date = datetime.strptime(start_date, "%d-%m-%Y").date()
india_tz = ZoneInfo("Asia/Kolkata")
local_start = datetime.combine(parsed_date, time(8, 0, 0, tzinfo=india_tz))
local_end = datetime.combine(parsed_date, time(17, 0, 0, tzinfo=india_tz))

utc_start_str = local_start.isoformat()
utc_end_str = local_end.isoformat()

url = "https://api.solcast.com.au/data/historic/radiation_and_weather"

query_params = {
    "latitude": LAT,
    "longitude": LON,
    "period": "PT5M",
    "start": utc_start_str,
    "end": utc_end_str,
    "format": "json",
    "time_zone": "utc",
    "api_key": API_KEY
}

response = requests.get(url, params=query_params)

if response.status_code == 200:
    data = json.loads(response.text)
    keep_columns = ["air_temp", "dni", "ghi", "period_end"]
    filtered_list = []
    for entry in data.get("estimated_actuals", []):
        filtered_entry = {key: entry[key] for key in keep_columns if key in entry}
        filtered_list.append(filtered_entry)
    output_file = f"solar_input_{start_date}.jsonl"
    with open(output_file, 'w') as file:
        for row in filtered_list:
            line = json.dumps(row)
            file.write(line + "\n")
            
    print(f"Successfully saved as {output_file}")
else:
    print("Error occurred")
    print(f"Status code {response.status_code}")
    print(response.text)