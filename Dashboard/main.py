from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from scipy.signal import savgol_filter
import asyncio
import json
import time
import math
import random
from datetime import datetime, timedelta, timezone
import uvicorn
from contextlib import asynccontextmanager
import threading
import traceback
from pprint import pprint 
import pandas as pd
import numpy as np
import copy

from downlink import main as run_downlink
import uuid
import xml.etree.ElementTree as ET
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from Google_Earth import main as maps_main

# Key Lists
PACKET_A_DIRECT_KEYS = ("SOC_Ah", "Pack_Voltage", "Pack_Current", "Bus_Voltage",
                        "Bus_Current", "Motor_Velocity", "PhaseC_Current",
                        "PhaseB_Current", "Speed",)
MPPT_NAMES = ('A', 'B', 'C', 'D')
MPPT_VALUE_KEYS = ("Input_Voltage", "Input_Current",
                        "Output_Voltage", "Output_Current")
MPPT_FLAG_NAMES = (
    'hw_overvolt', 'hw_overcurrent', None, 'under12v',
    'battery_full', 'battery_low',
    'mosfet_overheat', 'low_array_power'
)
CONTACTOR_FLAG_NAMES = (
    'contactor1_error', 'contactor2_error',
    'contactor1_output', 'contactor2_output',
    'contactor_supply',
    'contactor3_error', 'contactor3_output',
    None
)
CONTACTOR_FLAG_MAPPINGS = {
    "Precharge_Contactor_Flag3": 'contactor1_output',
    "Precharge_Contactor_Flag1": 'contactor1_error',
    "Precharge_Contactor_Flag4": 'contactor2_output',
    "Precharge_Contactor_Flag2": 'contactor2_error',
    "Precharge_Contactor_Flag7": 'contactor3_output',
    "Precharge_Contactor_Flag6": 'contactor3_error',
    "Precharge_Contactor_Flag5": 'contactor_supply',
    "Precharge_Contactor_Flag8": None,
    
}
CONTACTOR_STATE_MAPPINGS = {
    0: 'Error',
    1: 'Idle',
    2: 'Measure',
    3: 'Pre-charge',
    4: 'Run',
    5: 'Enable Pack'
}
BMS_FLAG_NAMES = (
    "cell_over_voltage", "cell_under_voltage", "cell_over_temp",
    "measurement_untrusted", "cmu_comm_timeout", "vehicle_comm_timeout",
    "bms_setup_mode", "cmu_can_status", "isolation_test_fail", "soc_invalid",
    "can_supply_low", "contactor_not_engaged", "extra_cell_detected",
)
MOTOR_LIMIT_NAMES = (
    'ipm_temp_limit', 'bus_voltage_lower_limit',
    'bus_voltage_upper_limit', 'bus_current_limit',
    'velocity_limit', 'motor_current_limit',
    'output_voltage_pwm_limit',
)
MOTOR_ERROR_NAMES = (
    'hardware_over_current','software_over_current', 'dc_bus_over_voltage', 
    'bad_motor_position', 'watchdog_reset', 'config_read_error',
    'rail_15v_uvlo', 'desaturation_fault', 'motor_over_speed', 
)
CABIN_FLAG_NAMES = (  ## TODO
    'Cabin_CO_Content', 'Cabin_CH4_Content',
    'Cabin_NH3_Content', 'Cabin_NO2_Content',
    'Cabin_O2_Content', 'Cabin_Temperature',
    'Cabin_Pressure', 'Cabin_CO2_Content',
)
PACKET_B_DIRECT_KEYS = ("Motor_Temp", "HeatSink_Temp", "DSP_Board_Temp",)


# Global state to store current data
current_data_default = {
    "metric": {
        'Pack_Voltage': 48.2,
        'SOC_Ah': 12000,
        'power_consumption': 1250.0,
        'solar_input': 450.0,
        'distance_travelled': 0.0,
        'Motor_Temp': 68.5,
        'Speed': 65.4,
        'predicted': 67.2,

        'Pack_Current': 46.26,
        'cmus': [{
            'temperature': 30.12,
            'cell_temperature': 30.12,
            'cell_voltages': [3.7 for _ in range(8)]
        } for _ in range(4)],
        'battery_ranges': {
            'min_temp': 0,
            'max_temp': 0,
            'min_volt': 0,
            'max_volt': 0,
        },
        'precharge_state': 0,
        'contactor_flags': {
            'contactor1_output': False,
            'contactor1_error': False,
            'contactor2_output': False,
            'contactor2_error': False,
            'contactor3_output': False,
            'contactor3_error': False,
            'contactor_supply': False,
        },
        'bmsFlags': {
            'cell_over_voltage': False,
            'cell_under_voltage': False,
            'cell_over_temp': False,
            'measurement_untrusted': False,
            'cmu_comm_timeout': False,
            'vehicle_comm_timeout': False,
            'bms_setup_mode': False,
            'cmu_can_status': False,
            'isolation_test_fail': False,
            'soc_invalid': False,
            'can_supply_low': False,
            'contactor_not_engaged': False,
            'extra_cell_detected': False,
        },


        'Motor_Velocity': 123,
        'Speed2': 75,
        'HeatSink_Temp': 31,
        'PhaseA_Current': 45.65,
        'PhaseB_Current': 45.6,
        'PhaseC_Current': 45.7,
        'Bus_Voltage': 50,
        'Bus_Current': 45,
        'Bus_Power': 50 * 45,
        'DSP_Board_Temp': 0,
        'MotorLimits': {
            'ipm_temp_limit': False,
            'bus_voltage_lower_limit': False,
            'bus_voltage_upper_limit': False,
            'bus_current_limit': False,
            'velocity_limit': False,
            'motor_current_limit': False,
            'output_voltage_pwm_limit': False,
        },
        'MotorErrors': {
            'motor_over_speed': False,
            'desaturation_fault': False,
            'rail_15v_uvlo': False,
            'config_read_error': False,
            'watchdog_reset': False,
            'bad_motor_position': False,
            'dc_bus_over_voltage': False,
            'software_over_current': False,
            'hardware_over_current': False,
        },
    
        'mppts': [{
            'Input_Voltage': 50,
            'Input_Current': 45,
            'Output_Voltage': 50,
            'Output_Current': 45,
            'Output_Power': 50 * 45,
            'efficiency': 98,

            'Mosfet_Temperature': 35,
            'MPPT_Temperature': 35,
            'flags': {
                'hw_overvolt': False,
                'hw_overcurrent':  False,
                'under12v': False,
                'low_array_power': False,
                'battery_full': False,
                'battery_low': False,
                'mosfet_overheat': False,
            }
        } for _ in range(4)],

        'CabinSensors': {
            'Cabin_CO_Content': 0.2,
            'Cabin_CH4_Content': 3,
            'Cabin_NH3_Content': 4,
            'Cabin_NO2_Content': 5,
            'Cabin_O2_Content': 6,
            'Cabin_Temperature': 35,
            'Cabin_Pressure': 76,
            'Cabin_CO2_Content': 2,
        }
    },
    "historic": {
        'Timestamps': [],
        'Speed': [],
        'Battery': [],
        'Power': [],
        'Solar': [],
        'Bus_Power': [],
        'Motor_Velocity': [],
        'Speed2': [],
        'PhaseA_Current': [],
        'PhaseB_Current': [],
        'PhaseC_Current': [],
        'HeatSink_Temp': [],
        'cmu1_temp': [], 'cmu1_cell_temp': [],
        'cmu2_temp': [], 'cmu2_cell_temp': [],
        'cmu3_temp': [], 'cmu3_cell_temp': [],
        'cmu4_temp': [], 'cmu4_cell_temp': [],
        'solar_input_voltage': [],
        'solar_output_power': [],
        'solar_mppt_A_Output_Voltage': [],
        'solar_mppt_B_Output_Voltage': [],
        'solar_mppt_C_Output_Voltage': [],
        'solar_mppt_D_Output_Voltage': [],
        'solar_mppt_A_Output_Power': [],
        'solar_mppt_B_Output_Power': [],
        'solar_mppt_C_Output_Power': [],
        'solar_mppt_D_Output_Power': [],
        'solar_mppt_A_Output_Current': [],
        'solar_mppt_B_Output_Current': [],
        'solar_mppt_C_Output_Current': [],
        'solar_mppt_D_Output_Current': [],
        'Acceleration': [],
        'Distance': [],
        'Altitude': [],
        'Latitudes': [-12.446822],
        'Longitudes': [130.907036],
    },
    "profile":{
        "Altitude": [],
        "Distance": [],
        "Gradient": []
    }
}
current_data = copy.deepcopy(current_data_default)
# In-memory session data store to hold file contents between requests
TRACK_SESSIONS = {}

# Reusable helper to parse KML data structure and extract structural folders
def analyze_kml_structure(kml_bytes: bytes):
    root = ET.fromstring(kml_bytes)
    
    # Strip namespaces
    for el in root.iter():
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]
            
    # Determine depth dynamically matching your logic
    depth = 1
    folder = True
    while folder:
        try:
            tag = root.find('Document/' + ('Folder/' * depth) * 1 + 'name')
            if tag is not None and len(tag.text):
                depth += 1
            else:
                depth -= 1
                folder = False
        except:
            depth -= 1
            folder = False

    # Extract all folders and their child placemarks
    available_folders = []
    folder_elements = root.findall('Document' + ('/Folder') * depth)
    
    for f_idx, f_el in enumerate(folder_elements):
        f_name = f_el.find('name')
        f_text = f_name.text if f_name is not None else f"Folder {f_idx + 1}"
        
        placemarks = []
        placemark_elements = f_el.findall('Placemark')
        for p_idx, p_el in enumerate(placemark_elements):
            p_name = p_el.find('name')
            p_text = p_name.text if p_name is not None else f"Placemark {p_idx + 1}"
            
            # Verify it contains line coordinates before listing
            if p_el.find("LineString/coordinates") is not None or p_el.find("Polygon/outerBoundaryIs/LinearRing/coordinates") is not None:
                placemarks.append({"index": p_idx, "name": p_text})
                
        available_folders.append({
            "index": f_idx,
            "name": f_text,
            "placemarks": placemarks
        })
        
    return root, available_folders

# WebSocket connection manager

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()

SOC_CURVE = [
    (115.36, 100.0),
    (114.38, 99.0),
    (113.40, 98.0),
    (112.70, 97.0),
    (112.00, 96.0),
    (111.44, 95.0),
    (110.88, 94.0),
    (110.60, 93.0),
    (110.32, 92.0),
    (110.04, 91.0),
    (109.76, 90.0),
    (109.48, 89.0),
    (109.20, 88.0),
    (108.92, 87.0),
    (108.64, 86.0),
    (108.36, 85.0),
    (108.08, 84.0),
    (107.80, 83.0),
    (107.52, 82.0),
    (107.24, 81.0),
    (106.96, 80.0),
    (106.75, 79.0),
    (106.54, 78.0),
    (106.33, 77.0),
    (106.12, 76.0),
    (105.91, 75.0),
    (105.70, 74.0),
    (105.49, 73.0),
    (105.28, 72.0),
    (105.07, 71.0),
    (104.86, 70.0),
    (104.65, 69.0),
    (104.44, 68.0),
    (104.16, 67.0),
    (103.88, 66.0),
    (103.60, 65.0),
    (103.32, 64.0),
    (103.04, 63.0),
    (102.76, 62.0),
    (102.48, 61.0),
    (102.20, 60.0),
    (101.92, 59.0),
    (101.64, 58.0),
    (101.36, 57.0),
    (101.08, 56.0),
    (100.80, 55.0),
    (100.52, 54.0),
    (100.24, 53.0),
    (99.96, 52.0),
    (99.61, 51.0),
    (99.26, 50.0),
    (98.91, 49.0),
    (98.56, 48.0),
    (98.21, 47.0),
    (97.86, 46.0),
    (97.51, 45.0),
    (97.16, 44.0),
    (96.81, 43.0),
    (96.46, 42.0),
    (96.11, 41.0),
    (95.76, 40.0),
    (95.41, 39.0),
    (95.06, 38.0),
    (94.71, 37.0),
    (94.36, 36.0),
    (93.94, 35.0),
    (93.52, 34.0),
    (93.10, 33.0),
    (92.68, 32.0),
    (92.26, 31.0),
    (91.84, 30.0),
    (91.42, 29.0),
    (91.00, 28.0),
    (90.51, 27.0),
    (90.02, 26.0),
    (89.53, 25.0),
    (89.04, 24.0),
    (88.55, 23.0),
    (88.06, 22.0),
    (87.57, 21.0),
    (87.08, 20.0),
    (86.52, 19.0),
    (85.96, 18.0),
    (85.40, 17.0),
    (84.84, 16.0),
    (84.14, 15.0),
    (83.44, 14.0),
    (82.74, 13.0),
    (82.04, 12.0),
    (81.20, 11.0),
    (80.36, 10.0),
    (79.52, 9.0),
    (78.68, 8.0),
    (77.56, 7.0),
    (76.44, 6.0),
    (74.90, 5.0),
    (73.36, 4.0),
    (71.40, 3.0),
    (69.44, 2.0),
    (69.02, 1.0),
    (68.60, 0.0),
]

def get_initial_soc(pack_voltage):
    if pack_voltage >= 115.36: return 100.0
    if pack_voltage <= 68.60: return 0.0
    
    for i in range(len(SOC_CURVE) - 1):
        v_upper, soc_upper = SOC_CURVE[i]
        v_lower, soc_lower = SOC_CURVE[i+1]
        if v_lower <= pack_voltage <= v_upper:
            ratio = (pack_voltage - v_lower) / (v_upper - v_lower)
            return soc_lower + ratio * (soc_upper - soc_lower)
    return 0.0

BATTERY_CAPACITY_WH = 3528.0  # Nominal capacity: 5.0Ah * 4.2V * 28S = 588Wh
BATTERY_CAPACITY_AH= 30.0 #5.0Ah * 28
# CMU reports cell voltage in mV; dashboard displays volts (valid ~2.5–4.2 V).
INVALID_CELL_MV = 10_000.0


def cell_voltage_mv_to_v(raw) -> float | None:
    """Convert telemetry mV to V; return None for missing/invalid cells."""
    try:
        mv = float(raw)
    except (TypeError, ValueError):
        return None
    if mv <= 0 or mv >= INVALID_CELL_MV:
        return None
    return mv / 1000.0

tracker_state = {
    "count": 0,
    "current_soc_percentage": None,
    "initial_SOC_Ah": None,
    "last_update_time": None
}
async def update_processor(queue: asyncio.Queue):
    """Background task that waits for data update events and broadcasts"""
    global tracker_state,current_data
    while True:
        try:
            (ptype, pdata) = await queue.get()
            # await asyncio.sleep(1)

            update_packet = {"type": "update", "historic": None}
            metric = current_data['metric']
            # ptype = 'None'
            # count += 1
            # metric['Speed'] = count % 10

            # if(count > 10):
            #     metric['bmsFlags']['cell_over_voltage'] = True
            # print( ptype, pdata)

            if ptype == "A":
                # Direct data
                for k in PACKET_A_DIRECT_KEYS:
                    metric[k] = pdata[k]
                # Convert velocity fields from m/s to km/h
                metric['Motor_Velocity'] = pdata['Motor_Velocity'] * 3.6
                metric['Speed'] = pdata['Vehicle_Velocity'] * 3.6
                
                # Divide Pack Voltage by 1000
                metric['Pack_Voltage'] = metric['Pack_Voltage'] / 1000.0
                
                # Direct data - reorganised
                mppts = []
                solar_o = 0
                solar_i_v = 0
                solar_ah_sum=0
                for i, old in zip(MPPT_NAMES, current_data['metric']['mppts']):
                    d = {k: pdata[k + f"_{i}"]
                        for k in MPPT_VALUE_KEYS}
                    
                    # Immeditate bugfix
                    # if(d['Input_Voltage'] < 0):
                    #     d['Input_Voltage'] = old['Input_Voltage']
                    
                    d['Output_Power'] = ds_o = d['Output_Voltage'] * d['Output_Current']
                    # d['Output_Power'] = ds_o = d['Input_Voltage'] * d['Input_Current']
                    solar_o +=  ds_o
                    solar_i_v += d['Input_Voltage']
            
                    d['efficiency'] = 100 * ds_o / max(d['Input_Voltage'] * d['Input_Current'], 0.00001)
                    d['efficiency'] = max(0, d['efficiency'])

                    d['Mosfet_Temperature'] = old['Mosfet_Temperature']
                    d['MPPT_Temperature'] = old['MPPT_Temperature']

                    d['flags'] = {
                        key: pdata[F"MPPT_{i}_Flag{j+1}"]
                        for j, key in enumerate(MPPT_FLAG_NAMES) if key
                    }
                    mppts.append(d)
                
                metric['mppts'] = mppts
                solar_i_v /= 4

                # Flags
                metric['precharge_state'] = sum((2**i) * pdata[f'Precharge_State_Flag{i+1}'] for i in range(0, 5))
                metric['contactor_flags'] = {
                    CONTACTOR_FLAG_MAPPINGS[key]: pdata[key]
                    for key, value in CONTACTOR_FLAG_MAPPINGS.items() if value
                }
                metric['bmsFlags'] = {
                    key: pdata[f"BMS_Flag{j+1}"]
                    for j, key in enumerate(BMS_FLAG_NAMES) if key
                }
                metric['MotorLimits'] = {
                    key: pdata[f"MC_Limit_Flag{j+1}"]
                    for j, key in enumerate(MOTOR_LIMIT_NAMES) if key          
                }
                metric['MotorErrors'] = {
                    key: pdata[f"MC_Error_Flag{j+1}"]
                    for j, key in enumerate(MOTOR_ERROR_NAMES) if key          
                }

                metric['PhaseA_Current'] = phase_a_current = (metric['PhaseB_Current'] + metric['PhaseC_Current']) / 2.0    #CHECK
                metric['Bus_Power'] = b_output_power = pdata['Bus_Voltage'] * pdata['Bus_Current']
                metric['power_consumption'] = output_power = b_output_power
                metric['Speed2'] = pdata['Vehicle_Velocity'] * 3.6
                metric['solar_input'] = solar_o

                metric['Motor_Temp'] = 0       
                raw_time = pdata.get('_rx_time', pdata.get('Timestamp'))
                if isinstance(raw_time, str):
                    try:
                        rx_dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                    except ValueError:
                        rx_dt = datetime.now().astimezone()
                else:
                    rx_dt = datetime.now().astimezone()
                if tracker_state['current_soc_percentage'] is None:
                    tracker_state['current_soc_percentage'] = get_initial_soc(metric['Pack_Voltage'])
                    #initial_soc_percentage=get_initial_soc(metric['Pack_Voltage'])
                    tracker_state['initial_SOC_Ah']=pdata["SOC_Ah"]
                    tracker_state['last_update_time'] = rx_dt
                else:
                    dt_seconds = (rx_dt - tracker_state['last_update_time']).total_seconds()
                    if dt_seconds < 0 or dt_seconds > 300:
                        metric['distance_travelled'] = 0.0
                        for k in current_data['historic']:
                            current_data['historic'][k] = []
                        tracker_state['last_update_time'] = rx_dt
                        dt_seconds = 0

                    if dt_seconds > 0:
                        net_power_watts = solar_o - b_output_power
                        energy_change_wh = net_power_watts * (dt_seconds / 3600.0)
                        soc_change = (energy_change_wh / BATTERY_CAPACITY_WH) * 100.0
                        tracker_state['current_soc_percentage'] += soc_change
                        tracker_state['current_soc_percentage'] = max(0.0, min(100.0, tracker_state['current_soc_percentage']))
                        '''solar_ah_sum += (solar_o*1000/pdata["Pack_Voltage"])*(dt_seconds/3600)
                        current_soc_percentage=initial_soc_percentage - (((pdata["SOC_Ah"]-initial_SOC_Ah)-solar_ah_sum)/BATTERY_CAPACITY_AH)*100'''
                        v = pdata.get('Vehicle_Velocity', 0)
                        distance_increment_km = (v * dt_seconds) / 1000.0
                        metric['distance_travelled'] += distance_increment_km
                    tracker_state["last_update_time"] = rx_dt

                metric['SOC_Ah'] = tracker_state["current_soc_percentage"]

                historic = {
                    'Timestamps': rx_dt.strftime('%H:%M:%S'),
                    'Speed': pdata['Vehicle_Velocity'] * 3.6,
                    'Battery': tracker_state["current_soc_percentage"],
                    'Power': output_power,
                    'Solar': solar_o,
                    'Bus_Power': b_output_power,
                    'Motor_Velocity': pdata['Motor_Velocity'] * 3.6,
                    'Speed2': pdata['Vehicle_Velocity'] * 3.6,

                    'PhaseA_Current': phase_a_current,
                    'PhaseB_Current': pdata['PhaseB_Current'],
                    'PhaseC_Current': pdata['PhaseC_Current'],
                    'HeatSink_Temp': metric.get('HeatSink_Temp', 0),

                    'cmu1_temp': metric['cmus'][0]['temperature'],
                    'cmu1_cell_temp': metric['cmus'][0]['cell_temperature'],
                    'cmu2_temp': metric['cmus'][1]['temperature'],
                    'cmu2_cell_temp': metric['cmus'][1]['cell_temperature'],
                    'cmu3_temp': metric['cmus'][2]['temperature'],
                    'cmu3_cell_temp': metric['cmus'][2]['cell_temperature'],
                    'cmu4_temp': metric['cmus'][3]['temperature'],
                    'cmu4_cell_temp': metric['cmus'][3]['cell_temperature'],

                    'solar_input_voltage': solar_i_v,
                    'solar_output_power': solar_o,
                    'solar_mppt_A_Output_Voltage': pdata['Output_Voltage_A'],
                    'solar_mppt_B_Output_Voltage': pdata['Output_Voltage_B'],
                    'solar_mppt_C_Output_Voltage': pdata['Output_Voltage_C'],
                    'solar_mppt_D_Output_Voltage': pdata['Output_Voltage_D'],
                    'solar_mppt_A_Output_Power': pdata['Output_Voltage_A'] * pdata['Output_Current_A'],
                    'solar_mppt_B_Output_Power': pdata['Output_Voltage_B'] * pdata['Output_Current_B'],
                    'solar_mppt_C_Output_Power': pdata['Output_Voltage_C'] * pdata['Output_Current_C'],
                    'solar_mppt_D_Output_Power': pdata['Output_Voltage_D'] * pdata['Output_Current_D'],
                    'solar_mppt_A_Output_Current': pdata['Output_Current_A'],
                    'solar_mppt_B_Output_Current': pdata['Output_Current_B'],
                    'solar_mppt_C_Output_Current': pdata['Output_Current_C'],
                    'solar_mppt_D_Output_Current': pdata['Output_Current_D'],

                    'Distance': metric['distance_travelled'],
                    'Altitude': pdata['Altitude'],
                    'Acceleration': math.sqrt(sum(pdata[f'acc_{i}']**2 for i in ('X', 'Y'))),
                    'Latitudes': pdata['Latitude'],
                    'Longitudes': pdata['Longitude'],
                }

                for k in current_data['historic']:
                    if k in historic:
                        current_data['historic'][k].append(historic[k])

                update_packet['historic'] = historic
            
            if ptype == 'B':
                for k in PACKET_B_DIRECT_KEYS:
                    metric[k] = pdata[k]
                
                mppts = []
                for i, old in zip(MPPT_NAMES, current_data['metric']['mppts']):
                    mppts.append({
                        **old,
                        'Mosfet_Temperature': pdata[f'Mosfet_Temp_{i}'],
                        'MPPT_Temperature': min(pdata[f'Controller_Temp_{i}'], 100),
                    })
                metric['mppts'] = mppts
                
                metric['cmus'] = []
                output_power = 0
                
                minTemp, maxTemp = 100, -100
                minVolt, maxVolt = float('inf'), float('-inf')

                for i in range(1, 5):
                    d = {
                        "temperature": pdata[f"CMU{i}_Temp"],
                        "cell_temperature": pdata[f"Cell{i}_Temp"],
                    }
                    minTemp = min(minTemp, d['temperature'], d['cell_temperature'])
                    maxTemp = max(maxTemp, d['temperature'], d['cell_temperature'])

                    cell_vs = []
                    for j in range(8):
                        v = cell_voltage_mv_to_v(pdata.get(f"CMU{i}_Cell{j}_Voltage"))
                        cell_vs.append(v)
                        if v is not None:
                            minVolt = min(minVolt, v)
                            maxVolt = max(maxVolt, v)
                    d['cell_voltages'] = cell_vs

                    metric['cmus'].append(d)

                if minVolt == float('inf'):
                    minVolt = 0
                if maxVolt == float('-inf'):
                    maxVolt = 0

                metric['battery_ranges'] = {
                    'min_temp': minTemp,
                    'max_temp': maxTemp,
                    'min_volt': minVolt,
                    'max_volt': maxVolt,
                }

                metric['CabinSensors'] = {
                    key: pdata[key]
                    for key in CABIN_FLAG_NAMES       
                }
            if ptype == "C":
                current_data['profile']['Altitude']=pdata['Altitude']
                current_data['profile']['Distance']=pdata['Distance']
                current_data['profile']['Gradient']=pdata['Gradient']   
            current_data['metric'] = metric
            update_packet['metric'] = {**metric}
            update_packet['profile']={**current_data['profile']}

            print("broadcasting")
            await manager.broadcast(json.dumps(update_packet, default=lambda o: "Infinity" if o == float('inf') 
                  else "-Infinity" if o == float('-inf') 
                  else "NaN" if o != o  # NaN check
                  else None))

        except Exception as e:
            print(f"[update_processor] Bad packet skipped: {e}")
            traceback.print_exc()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize data and start background tasks"""
    
    app.state.queue = asyncio.Queue()
    queue = app.state.queue

    loop = asyncio.get_event_loop()
    t1 = asyncio.create_task(update_processor(queue))

    thread =  threading.Thread(
        target=run_downlink,
        args=(queue, loop),
        daemon=True
    )
    thread.start()

    yield

    t1.cancel()

    return

app = FastAPI(title="Telemetry Dashboard API", lifespan=lifespan)

frontend_dir = os.path.join(os.path.dirname(__file__), "prodbuild")

app.mount("/_app", StaticFiles(directory=os.path.join(frontend_dir, "_app")), name="_app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://192.168.1.232:5173"],  # Add your Svelte dev server ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse, HTMLResponse
import time


@app.get("/api/data/historical")
async def get_historical_data():
    """Get all cached historical data for initial dashboard load"""
    return {
        'metric': current_data["metric"],
        'historic': current_data["historic"]
    }

@app.get("/api/data/clear")
async def clear_historical_data():
    """Clear all cached historical data for initial dashboard load"""
    global current_data,tracker_state,current_data_default
    print("Got clear request")
    profile=current_data['profile']
    current_data = copy.deepcopy(current_data_default)
    current_data['profile']={**profile}
    tracker_state["count"] = 0
    tracker_state["current_soc_percentage"] = None
    tracker_state["initial_SOC_Ah"] = None
    tracker_state["last_update_time"] = None
    reload_signal = {
        "type": "reload" 
    }
    await manager.broadcast(json.dumps(reload_signal))
    print("🧹 State fully cleared and broadcasted to dashboard UI.")
    return {"status":"success"}
@app.get("/api/get-session-options/{session_id}")
async def get_session_options(session_id: str):
    kml_bytes = TRACK_SESSIONS.get(session_id)
    if not kml_bytes:
        # Session expired on backend server (e.g., server restarted)
        raise HTTPException(status_code=404, detail="Session expired")
    try:
        _, structural_options = analyze_kml_structure(kml_bytes)
        return {"options": structural_options}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.get("/")
async def root():
    index_path = os.path.join(frontend_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    headers = {"Clear-Site-Data": '"cache"'}
    return HTMLResponse(content=content, headers=headers)

@app.get("/{full_path:path}")
async def spa_catch_all(full_path: str):
    file_path = os.path.join(frontend_dir, full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.post("/api/simulate")
async def simulate_endpoint(request: Request):
    """Endpoint for the simulator script to push telemetry data"""
    try:
        data = await request.json()
        ptype = data.get("type")
        if ptype is None:
            ptype= "A" if "SOC_Ah" in data else ("B" if "Motor_Temp" in data else "C")
        await app.state.queue.put((ptype, data))
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/upload-kml")
async def upload_kml(file: UploadFile = File(...)):
    try:
        kml_bytes = await file.read()
        # Parse once to ensure structure is clean and gather selectable parameters
        _, structural_options = analyze_kml_structure(kml_bytes)
        
        # Issue a temporary session identifier
        session_id = str(uuid.uuid4())
        TRACK_SESSIONS[session_id] = kml_bytes
        
        return {"session_id": session_id, "options": structural_options}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to inspect KML: {str(e)}"})


class SelectionPayload(BaseModel):
    session_id: str
    folder_index: int
    placemark_index: int

@app.post("/api/render-selected-track")
async def render_selected_track(payload: SelectionPayload):
    global current_data
    kml_bytes = TRACK_SESSIONS.get(payload.session_id)
    if not kml_bytes:
        raise HTTPException(status_code=410, detail="Session expired or file context lost. Please re-upload.")
        
    try:
        root, _ = analyze_kml_structure(kml_bytes)
        
        # Determine depth dynamically
        depth = 1
        folder_active = True
        while folder_active:
            try:
                tag = root.find('Document/' + ('Folder/' * depth) * 1 + 'name')
                if tag is not None and len(tag.text):
                    depth += 1
                else:
                    depth -= 1
                    folder_active = False
            except:
                depth -= 1
                folder_active = False
                
        # Resolve user choice using the explicit indexes sent back from the web page
        folder = root.findall('Document' + ('/Folder') * depth)[payload.folder_index]
        path_element = folder.findall("Placemark")[payload.placemark_index]
        route_name=path_element.find("name").text
        coords_text = path_element.find("LineString/coordinates").text if path_element.find("LineString/coordinates") is not None else path_element.find("Polygon/outerBoundaryIs/LinearRing/coordinates").text
        coords_split = coords_text.split()
        
        coordinates = []
        for pair in coords_split:
            lon, lat, *_ = pair.split(",")
            coordinates.append((float(lat), float(lon)))
            
        # Pass the cleanly parsed layout array directly into your pipeline execution framework
        # maps_main(coordinates) -> modify maps_main to build your folium object and return HTML
        map_html,altitude_profile,distance_profile = maps_main(route_name,coordinates)
        total_distance = distance_profile[-1]*1000  # Total length of the loop in meters
        # 1. DYNAMICALLY CALCULATE WINDOW SIZE
        # Goal: We want the smoothing window to span roughly 45 meters of trail.
        DESIRED_SMOOTHING_DISTANCE = 100  # in meters
        # Calculate how many meters are between each of your 999 points
        meters_per_point = total_distance / len(altitude_profile)
        # Determine how many points are needed to cover that distance
        calculated_window = int(DESIRED_SMOOTHING_DISTANCE / meters_per_point)

        # The window size MUST be at least 3, and it MUST be an odd number for symmetrical padding
        window_size = max(3, calculated_window)
        if window_size % 2 == 0:
            window_size += 1
        # 2. PRE-SMOOTH ALTITUDE WITH THE DYNAMIC WINDOW
        window = np.ones(window_size) / window_size
        padded_altitude = np.pad(altitude_profile, window_size // 2, mode='edge')
        smoothed_altitude = np.convolve(padded_altitude, window, mode='valid').tolist()
        m_per_point = total_distance / len(altitude_profile)
        points_in_window = int(DESIRED_SMOOTHING_DISTANCE / m_per_point)

        # Ensure window size is at least 3 and odd
        if points_in_window < 3: points_in_window = 3
        if points_in_window % 2 == 0: points_in_window += 1

        half_win = points_in_window // 2
        gradient_profile = []

        # 2. Compute slope using Linear Regression (Polyfit)
        for i in range(len(smoothed_altitude)):
            # Determine window boundaries around point i
            start = max(0, i - half_win)
            end = min(len(smoothed_altitude), i + half_win + 1)
            
            window_dist_km = distance_profile[start:end]
            window_alt_m = smoothed_altitude[start:end]
            
            # Convert distances from km to meters so units match altitude
            window_dist_meters = [d * 1000 for d in window_dist_km]
            
            if len(window_dist_meters) >= 2:
                # np.polyfit(x, y, 1) fits a straight line (degree 1: y = mx + c)
                # It returns [slope, intercept]. We just want the slope (index 0).
                slope, intercept = np.polyfit(window_dist_meters, window_alt_m, 1)
                
                # Convert slope to a percentage
                gradient = slope * 100
            else:
                gradient = 0.0
                
            gradient_profile.append(gradient)
        '''gradient_profile=[]
        for i in range(len(altitude_profile)-1):
            gradient_profile.append(((smoothed_altitude[i+1]-smoothed_altitude[i])/(distance_profile[i+1]-distance_profile[i]))*0.1)
        gradient_profile=[0]+gradient_profile
        raw_gradients_arr = np.array(gradient_profile)
        # Define your window size and weights
        window_size = 3
        window = np.ones(window_size) / window_size

        # Apply the moving average
        # 'edge' padding prevents the ends of your data from dropping off to zero
        padded_gradients = np.pad(raw_gradients_arr, window_size // 2, mode='edge')
        smoothed_gradient = np.convolve(padded_gradients, window, mode='valid').tolist()'''
        distance_profile=[round(i,2) for i in distance_profile]
        packet_c = {
            "Altitude": smoothed_altitude,
            "Distance": distance_profile,
            "Gradient": gradient_profile
        }
        await app.state.queue.put(("C", packet_c)) 
        return {"map_html": map_html}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Execution processing error: {str(e)}"})
    
@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        data = {
            'type': 'data',
            'metric': current_data["metric"],
            "historic": current_data["historic"],
            "profile": current_data['profile']
        }
        await websocket.send_text(json.dumps(data))

        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)