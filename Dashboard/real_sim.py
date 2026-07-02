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

URL = "http://127.0.0.1:8000/api/simulate"

# Simulation state
state = {
    "Pack_Voltage": 118.0,
    "Speed": 0.0,
    "Time": 0.0,
    "Latitude": -12.446822,
    "Longitude": 130.907036
}

def generate_packet_a():
    state["Time"] += 1.0
    
    # Simulate driving: erratic speed changes
    speed_target = 60.0 + 40.0 * math.sin(state["Time"] / 5.0) + random.uniform(-20, 20)
    state["Speed"] += (speed_target - state["Speed"]) * 0.5
    if state["Speed"] < 0: state["Speed"] = 0
    
    # Motor uses highly erratic power (up to 150kW to make SOC drain visible)
    motor_power = max(0, state["Speed"] * 100.0) * random.uniform(0.1, 5.0)
    if random.random() < 0.2:
        motor_power += random.uniform(50000, 150000) # Huge 150kW spikes to drain SOC fast
        
    bus_voltage = 100.0 + random.uniform(-5, 5)
    bus_current = motor_power / bus_voltage
    
    # Solar provides erratic power
    solar_power = 800.0 + 400.0 * math.sin(state["Time"] / 10.0) + random.uniform(-200, 200)
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
        "Timestamp": datetime.now(timezone.utc).isoformat(),
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

def main(df=None):
    print("="*50)
    print("🚀 CAR TELEMETRY SIMULATOR STARTED")
    print(f"📡 Sending mock data to {URL}")
    print("="*50)
    req = requests.get(url="http://127.0.0.1:8000/api/data/clear")
    if df is None:
        while True:
            try:
                pkt_a = generate_packet_a()
                pkt_b = generate_packet_b()
                
                success_a = send_data(pkt_a)
                success_b = send_data(pkt_b)
                
                if success_a and success_b:
                    # Print occasionally so terminal isn't overwhelmed by 10Hz
                    if random.random() < 0.1:
                        calc_motor_power = pkt_a['Bus_Current'] * pkt_a['Bus_Voltage']
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] SENT | Speed: {pkt_a['Speed']:.1f} km/h | Pack: {pkt_a['Pack_Voltage']/1000:.1f}V | Motor Pwr: {calc_motor_power/1000:.1f}kW")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection refused. Is main.py running?")
                
            except Exception as e:
                print(f"Error: {e}")
                
            time.sleep(0.1)  # 10Hz update rate
    else:
        for index, row in df.iterrows():
            try:
                if pd.notna(row['SOC_Ah']):
                    pkt_a = {
                        "type": "A",
                        "Timestamp": row['_rx_time']}
                    col_list=['SOC_Ah', 'Pack_Voltage', 'Pack_Current', 'Bus_Voltage', 'Bus_Current', 'Motor_Velocity', 'Vehicle_Velocity', 'PhaseC_Current', 'PhaseB_Current', 'Input_Voltage_A', 'Input_Current_A', 'Output_Voltage_A', 'Output_Current_A', 'Input_Voltage_B', 'Input_Current_B', 'Output_Voltage_B', 'Output_Current_B', 'Input_Voltage_C', 'Input_Current_C', 'Output_Voltage_C', 'Output_Current_C', 'Input_Voltage_D', 'Input_Current_D', 'Output_Voltage_D', 'Output_Current_D', 'Latitude', 'Longitude', 'Altitude', 'Speed', 'acc_X', 'acc_Y', 'acc_Z', 'Throttle_Perc', 'Brake_Status', 'Precharge_Contactor_Flag1', 'Precharge_Contactor_Flag2', 'Precharge_Contactor_Flag3', 'Precharge_Contactor_Flag4', 'Precharge_Contactor_Flag5', 'Precharge_Contactor_Flag6', 'Precharge_Contactor_Flag7', 'Precharge_Contactor_Flag8', 'Precharge_State_Flag1', 'Precharge_State_Flag2', 'Precharge_State_Flag3', 'Precharge_State_Flag4', 'Precharge_State_Flag5', 'BMS_Flag1', 'BMS_Flag2', 'BMS_Flag3', 'BMS_Flag4', 'BMS_Flag5', 'BMS_Flag6', 'BMS_Flag7', 'BMS_Flag8', 'BMS_Flag9', 'BMS_Flag10', 'BMS_Flag11', 'BMS_Flag12', 'BMS_Flag13', 'MC_Limit_Flag1', 'MC_Limit_Flag2', 'MC_Limit_Flag3', 'MC_Limit_Flag4', 'MC_Limit_Flag5', 'MC_Limit_Flag6', 'MC_Limit_Flag7', 'MC_Error_Flag1', 'MC_Error_Flag2', 'MC_Error_Flag3', 'MC_Error_Flag4', 'MC_Error_Flag5', 'MC_Error_Flag6', 'MC_Error_Flag7', 'MC_Error_Flag8', 'MC_Error_Flag9', 'MPPT_A_Flag1', 'MPPT_A_Flag2', 'MPPT_A_Flag3', 'MPPT_A_Flag4', 'MPPT_A_Flag5', 'MPPT_A_Flag6', 'MPPT_A_Flag7', 'MPPT_A_Flag8', 'MPPT_B_Flag1', 'MPPT_B_Flag2', 'MPPT_B_Flag3', 'MPPT_B_Flag4', 'MPPT_B_Flag5', 'MPPT_B_Flag6', 'MPPT_B_Flag7', 'MPPT_B_Flag8', 'MPPT_C_Flag1', 'MPPT_C_Flag2', 'MPPT_C_Flag3', 'MPPT_C_Flag4', 'MPPT_C_Flag5', 'MPPT_C_Flag6', 'MPPT_C_Flag7', 'MPPT_C_Flag8', 'MPPT_D_Flag1', 'MPPT_D_Flag2', 'MPPT_D_Flag3', 'MPPT_D_Flag4', 'MPPT_D_Flag5', 'MPPT_D_Flag6', 'MPPT_D_Flag7', 'MPPT_D_Flag8']
                    for col in col_list:
                        pkt_a[col]=row[col]
                    #pkt_b["Timestamp"]=row['_rx_time']
                    success=send_data(pkt_a)
                else:
                    pkt_b={
                        "type": "B",
                        "Timestamp":row['_rx_time']
                    }
                    col_list=['CMU1_Temp', 'Cell1_Temp', 'CMU2_Temp', 'Cell2_Temp', 'CMU3_Temp', 'Cell3_Temp', 'CMU4_Temp', 'Cell4_Temp', 'CMU5_Temp', 'Cell5_Temp', 'Motor_Temp', 'HeatSink_Temp', 'DSP_Board_Temp', 'Mosfet_Temp_A', 'Controller_Temp_A', 'Mosfet_Temp_B', 'Controller_Temp_B', 'Mosfet_Temp_C', 'Controller_Temp_C', 'Mosfet_Temp_D', 'Controller_Temp_D', 'CMU1_Cell0_Voltage', 'CMU1_Cell1_Voltage', 'CMU1_Cell2_Voltage', 'CMU1_Cell3_Voltage', 'CMU1_Cell4_Voltage', 'CMU1_Cell5_Voltage', 'CMU1_Cell6_Voltage', 'CMU1_Cell7_Voltage', 'CMU2_Cell0_Voltage', 'CMU2_Cell1_Voltage', 'CMU2_Cell2_Voltage', 'CMU2_Cell3_Voltage', 'CMU2_Cell4_Voltage', 'CMU2_Cell5_Voltage', 'CMU2_Cell6_Voltage', 'CMU2_Cell7_Voltage', 'CMU3_Cell0_Voltage', 'CMU3_Cell1_Voltage', 'CMU3_Cell2_Voltage', 'CMU3_Cell3_Voltage', 'CMU3_Cell4_Voltage', 'CMU3_Cell5_Voltage', 'CMU3_Cell6_Voltage', 'CMU3_Cell7_Voltage', 'CMU4_Cell0_Voltage', 'CMU4_Cell1_Voltage', 'CMU4_Cell2_Voltage', 'CMU4_Cell3_Voltage', 'CMU4_Cell4_Voltage', 'CMU4_Cell5_Voltage', 'CMU4_Cell6_Voltage', 'CMU4_Cell7_Voltage', 'CMU5_Cell0_Voltage', 'CMU5_Cell1_Voltage', 'CMU5_Cell2_Voltage', 'CMU5_Cell3_Voltage', 'CMU5_Cell4_Voltage', 'CMU5_Cell5_Voltage', 'CMU5_Cell6_Voltage', 'CMU5_Cell7_Voltage', 'Cabin_CO_Content', 'Cabin_CH4_Content', 'Cabin_NH3_Content', 'Cabin_NO2_Content', 'Cabin_O2_Content', 'Cabin_Temperature', 'Cabin_Pressure', 'Cabin_CO2_Content']
                    for col in col_list:
                        pkt_b[col]=row[col]
                    #pkt_a["Timestamp"]=row['_rx_time']
                    success=send_data(pkt_b)
                if success:
                    # Print occasionally so terminal isn't overwhelmed by 10Hz
                    if random.random() < 0.1:
                        calc_motor_power = pkt_a['Bus_Current'] * pkt_a['Bus_Voltage']
                        print(f"[{datetime.fromisoformat(pkt_a['Timestamp']).strftime('%H:%M:%S')}] SENT | Speed: {pkt_a['Vehicle_Velocity']*(18/5):.1f} km/h | Pack: {pkt_a['Pack_Voltage']/1000:.1f}V | Motor Pwr: {calc_motor_power/1000:.1f}kW")
                else:
                    print(f"[{datetime.fromisoformat(pkt_a['Timestamp']).strftime('%H:%M:%S')}] Connection refused. Is main.py running?")
            except Exception as e:
                    print(f"Error: {e}")
                    
            time.sleep(2)  # 10Hz update rate
            

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    file_name=input("File name: ")
    FILE_PATH=SCRIPT_DIR / "Logs" / file_name
    df=pd.read_json(FILE_PATH,lines=True,convert_dates=False)
    main(df=df)
