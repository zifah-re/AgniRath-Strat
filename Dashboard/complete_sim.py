from real_sim import main as real_sim_main
from pathlib import Path
import pandas as pd
from helper import get_profile
import numpy as np
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
file_name=input("File name: ")
FILE_PATH=SCRIPT_DIR / "Logs" / file_name
df=pd.read_json(FILE_PATH,lines=True,convert_dates=False)
offline_model=df[['_rx_time','Vehicle_Velocity']]
offline_model.dropna(subset=['Vehicle_Velocity'],inplace=True)
distance_profle=get_profile(["Distance"])["Distance"]
distance_profle=np.array(distance_profle)*1000
offline_model["timestamp_secs"] = pd.to_datetime(offline_model["_rx_time"]).view("int64").astype("float") / 10**9
velocity_mps = offline_model["Vehicle_Velocity"]
dt = offline_model["timestamp_secs"].diff().fillna(0)
step_distances = 0.5 * (velocity_mps + velocity_mps.shift(0).fillna(0)) * dt
offline_model["distance"] = step_distances.cumsum()
interpolated_secs = np.interp(distance_profle, offline_model["distance"], offline_model["timestamp_secs"]).tolist()
interpolated_velocity = (np.interp(distance_profle, offline_model["distance"], offline_model["Vehicle_Velocity"])*(18/5))
target_profile=list(zip(interpolated_secs,interpolated_velocity))
URL="http://127.0.0.1:8000/api/simulate"
pkt={
    "type":"C",
    "TargetProfile":target_profile
}
print(isinstance(target_profile[0],tuple))
req=requests.post(URL,json=pkt)
real_sim_main(df=df)
