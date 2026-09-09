from helper import get_profile
import numpy as np

# TargetProfile is deliberately one-lap long so it stays index-aligned with
# the one-lap KML Distance array.  A pushed loop strategy also provides one
# such profile per solved lap; concatenate them here so the simulator drives
# through the complete strategy instead of stopping after lap one.
profiles = get_profile(["TargetProfile", "TargetProfilesByLap"])
lap_profiles = profiles.get("TargetProfilesByLap") or []
if lap_profiles:
    # The per-lap profiles contain the solved driving points.  Add explicit
    # stationary points at each boundary so the simulator also represents the
    # mandatory five-minute stop instead of teleporting into the next lap.
    frames = []
    for lap_index, lap in enumerate(lap_profiles):
        if lap_index and frames:
            frames.append((frames[-1][0], 0.0, True))
            frames.append((lap[0][0], 0.0, True))
        frames.extend((point[0], point[1], False) for point in lap)
    full_profile = np.asarray([[time_s, speed] for time_s, speed, _ in frames], dtype=float)
    stop_frames = np.asarray([is_stop for _, _, is_stop in frames], dtype=bool)
else:
    full_profile = np.asarray(profiles["TargetProfile"], dtype=float)
    stop_frames = np.zeros(len(full_profile), dtype=bool)

target_profile, time_profile = full_profile[:, 1], full_profile[:, 0]

noise_data=target_profile+np.random.uniform(-5,5,len(target_profile))
noise_data[stop_frames] = 0.0

import time
import json
import math
import random
import urllib.request
import urllib.error
from datetime import datetime, timezone
import pandas as pd
import requests
from pathlib import Path
from zoneinfo import ZoneInfo

SA_TZ=ZoneInfo("Africa/Johannesburg")
URL = "http://127.0.0.1:8000/api/simulate"

# Simulation state
state = {
    "Pack_Voltage": 118.0,
    "Speed": 0.0,
    "Time": 0.0,
    "Latitude": -12.446822,
    "Longitude": 130.907036
}

def generate_packet_a(speed,time):
    state["Time"] += 1.0
    
    # Simulate driving: erratic speed changes
    state["Speed"]=speed
    if state["Speed"] < 0: state["Speed"] = 0
    state['Speed']*=5/18
    # Motor uses highly erratic power (up to 150kW to make SOC drain visible)
    motor_power = max(0, state["Speed"] *18 *100.0/5) * random.uniform(0.1, 0.7)
        
    bus_voltage = 100.0 + random.uniform(-5, 5)
    bus_current = motor_power / bus_voltage
    
    # Solar provides erratic power
    solar_power = 1000 + 400.0 * math.sin(state["Time"] / 10.0) + random.uniform(-200, 200)
    if solar_power < 0: solar_power = 0
    mppt_power = solar_power / 4.0
    mppt_out_v = 100.0
    mppt_out_i = mppt_power / mppt_out_v
    
    # Battery dynamics: drops slightly every second to simulate discharge
    # Wait, the backend uses Coulomb counting now, but we still simulate Pack Voltage
    # because the initial SOC relies on it, and the backend might plot it.
    state["Pack_Voltage"] -= 0.005
    if state["Pack_Voltage"] < 70.0:
        state["Pack_Voltage"] = 118.0
        
    packet = {
        "type": "A",
        "Timestamp": datetime.fromtimestamp(time,tz=SA_TZ).isoformat(),
        "_rx_time": datetime.fromtimestamp(time,tz=SA_TZ).isoformat(),
        "SOC_Ah": 12000,
        "Pack_Voltage": state["Pack_Voltage"] * 1000, # backend divides by 1000
        "Pack_Current": ((motor_power - solar_power) / state["Pack_Voltage"]) * 1000,
        "Bus_Voltage": bus_voltage,
        "Bus_Current": bus_current,
        "Motor_Velocity": state["Speed"] * 10,
        "Vehicle_Velocity": state["Speed"],
        "Speed": state["Speed"],
        "PhaseC_Current": bus_current * 0.5,
        "PhaseB_Current": bus_current * 0.5,
        "Altitude": 15.0,
        "Latitude": state["Latitude"],
        "Longitude": state["Longitude"],
        "acc_X": random.uniform(-0.1, 0.1),
        "acc_Y": random.uniform(-0.1, 0.1),
    }
    
    # MPPTs
    for i, m in enumerate(['A', 'B', 'C', 'D']):
        packet[f"Input_Voltage_{m}"] = 50.0 + random.uniform(-2, 2)
        packet[f"Input_Current_{m}"] = (mppt_power / 50.0) + random.uniform(-0.1, 0.1)
        packet[f"Output_Voltage_{m}"] = mppt_out_v
        packet[f"Output_Current_{m}"] = mppt_out_i
        for j in range(1, 9):
            packet[f"MPPT_{m}_Flag{j}"] = False
            
    # Flags
    for j in range(1, 6): packet[f"Precharge_State_Flag{j}"] = False
    for j in range(1, 9): packet[f"Precharge_Contactor_Flag{j}"] = False
    for j in range(1, 14): packet[f"BMS_Flag{j}"] = False
    for j in range(1, 8): packet[f"MC_Limit_Flag{j}"] = False
    for j in range(1, 10): packet[f"MC_Error_Flag{j}"] = False
    
    return packet

def generate_packet_b():
    packet = {
        "type": "B",
        "Timestamp": datetime.now(timezone.utc).isoformat(),
        "Motor_Temp": 45.0 + random.uniform(-1, 1),
        "HeatSink_Temp": 40.0 + random.uniform(-1, 1),
        "DSP_Board_Temp": 35.0 + random.uniform(-1, 1),
        "Cabin_CO_Content": 1,
        "Cabin_CH4_Content": 2,
        "Cabin_NH3_Content": 3,
        "Cabin_NO2_Content": 4,
        "Cabin_O2_Content": 5,
        "Cabin_Temperature": 25,
        "Cabin_Pressure": 1013,
        "Cabin_CO2_Content": 400,
    }
    
    for i, m in enumerate(['A', 'B', 'C', 'D']):
        packet[f"Mosfet_Temp_{m}"] = 30.0 + random.uniform(-1, 1)
        packet[f"Controller_Temp_{m}"] = 32.0 + random.uniform(-1, 1)
        
    for i in range(1, 5):
        packet[f"CMU{i}_Temp"] = 25.0 + random.uniform(-0.5, 0.5)
        packet[f"Cell{i}_Temp"] = 26.0 + random.uniform(-0.5, 0.5)
        for j in range(8):
            packet[f"CMU{i}_Cell{j}_Voltage"] = 3.7 + random.uniform(-0.02, 0.02)
            
    return packet

def send_data(packet):
    req = urllib.request.Request(URL, data=json.dumps(packet).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            pass
        return True
    except Exception:
        return False

def main(v=None,t=None):
    print("="*50)
    print("🚀 CAR TELEMETRY SIMULATOR STARTED")
    print(f"📡 Sending mock data to {URL}")
    print("="*50)
    req = requests.get(url="http://127.0.0.1:8000/api/data/clear")
    for i in range(len(v)):
        try:
            pkt_a = generate_packet_a(v[i],t[i])
            pkt_b = generate_packet_b()
            
            success_a = send_data(pkt_a)
            success_b = send_data(pkt_b)
            
            if success_a and success_b:
                # Print occasionally so terminal isn't overwhelmed by 10Hz
                if random.random() < 0.1:
                    calc_motor_power = pkt_a['Bus_Current'] * pkt_a['Bus_Voltage']
                    print(f"[{datetime.fromtimestamp(t[i]).strftime('%H:%M:%S')}] SENT | Speed: {pkt_a['Speed']:.1f} km/h | Pack: {pkt_a['Pack_Voltage']/1000:.1f}V | Motor Pwr: {calc_motor_power/1000:.1f}kW")
            else:
                print(f"[{datetime.fromtimestamp(t[i]).strftime('%H:%M:%S')}] Connection refused. Is main.py running?")
            
        except Exception as e:
            print(f"Error: {e}")
            
        time.sleep(0.1)  # 10Hz update rate
        

if __name__ == "__main__":
    main(v=noise_data,t=time_profile)
