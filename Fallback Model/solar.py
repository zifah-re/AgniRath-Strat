from solcast import forecast
import glob
import argparse
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import os

SCRIPT_DIR= Path(__file__).resolve().parent

def main(file):
    with open(f"Fallback Model\\Saves\\{file}","r") as f:
        data=f.read()
        fname=os.path.basename(f.name)
    data=json.loads(data)
    distances=data['profile']['Distance']
    coords=np.array(data['profile']['Coordinates'])
    lats,lons=coords[:,0],coords[:,1]
    n=1
    query_dist=np.linspace(0.0,distances[-1],n)
    solar_data=[]
    lat_list,lon_list=np.interp(query_dist,distances,lats),np.interp(query_dist,distances,lons)
    for i in range(n):
        response=forecast.radiation_and_weather(
            latitude=lat_list[i],
            longitude=lon_list[i],
            output_parameters=['air_temp','dni','ghi','wind_speed_10m','wind_direction_10m'],
            period="PT5M",
            terrain_shading=True,
            hours=335
        )
        df=response.to_pandas()
        df.reset_index(inplace=True)
        df['period_end'] = df['period_end'].dt.strftime("%Y-%m-%dT%H:%M:%S%:z")
        record={"lat":lat_list[i],"lon":lon_list[i],"data":df.to_dict(orient="records")}
        solar_data.append(record)
    file_path=SCRIPT_DIR / "Solar" / (fname[:len(fname)-9]+".jsonl")
    with open(file_path,'w',encoding='utf-8') as f:
        json.dump(solar_data,f,indent=4)

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Get solar data for the given kml files")
    parser.add_argument("files", nargs="*", help="KML save paths (default: auto-discover)")
    parser.add_argument("--output-dir", default=r"Solar", help="Directory to save plots (default: current dir)")
    args = parser.parse_args()

    paths = args.files or sorted(glob.glob("Fallback Model\\Saves\\*.kml.save"))
    if not paths:
        print("No KML save files found. Pass file paths or run from the folder containing them.")
        sys.exit(1)

    print(f"Loading {len(paths)} file(s): {[Path(p).name for p in paths]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        main(Path(p).name)
    