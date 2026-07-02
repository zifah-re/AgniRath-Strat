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
from constants import SOC_CURVE,BATTERY_CAPACITY_AH,BATTERY_CAPACITY_WH,INVALID_CELL_MV,MAX_SPEED
import uuid
import xml.etree.ElementTree as ET
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from Google_Earth import main as maps_main
from geopy.distance import geodesic
from geopy.point import Point


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
        'SOC_Ah': 100,
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
        },
        'Latitude': 0.0,
        'Longitude': 0.0,
        'Altitude': 0.0,
        'Gradient': 0.0,
        'Heading': 0.0,
        'ETA': 0
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
        'Latitudes': [],
        'Longitudes': [],
    },
    "profile":{
        "Altitude": [],
        "Gradient": [],
        "Coordinates": [],
        "Distance": [],
        "SpeedLimit": [],
        "SpeedProfile":[],       # Speeds of traffic at that particular distance, not to be confused with target velocity profile
        "Headings":[],
        "TargetProfile": []
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
def get_icon_url(root,point):
    icon_style=point.find("styleUrl").text.lstrip("#")
    icon_tag=root.find(f"Document/StyleMap[@id='{icon_style}']")
    icon_super=icon_tag.findall("Pair")[0].find("styleUrl").text.lstrip("#")
    icon_tag=root.find(f"Document/Style[@id='{icon_super}']")
    icon_url=icon_tag.find("IconStyle/Icon/href").text
    icon_anchor=icon_tag.find("IconStyle/hotSpot")
    icon_anchor=(icon_anchor.attrib['x'],icon_anchor.attrib['y'])
    return icon_url,icon_anchor
def get_line_colour(root,point):
    try:
        icon_style=point.find("styleUrl").text.lstrip("#")
        icon_tag=root.find(f"Document/StyleMap[@id='{icon_style}']")
        icon_super=icon_tag.findall("Pair")[0].find("styleUrl").text.lstrip("#")
        icon_tag=root.find(f"Document/Style[@id='{icon_super}']")
        colour=icon_tag.find("LineStyle/color").text
        opacity=colour[:2]
        opacity=int(opacity,16)/255
        b=colour[2:4]
        g=colour[4:6]
        r=colour[6:8]
        colour=r+g+b
        width=int(icon_tag.find("LineStyle/width").text)
    except:
        return None,None,None
    return colour,width,opacity
    
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
                if len(current_data['profile']['Coordinates'])>0:
                    profile_distance=current_data['profile']['Distance'][-1]
                    distance=metric['distance_travelled']%profile_distance
                    i=-1
                    while i<len(current_data['profile']['Distance'])-1 and distance>=current_data['profile']['Distance'][i+1]:
                        i+=1
                    if len(current_data['profile']['SpeedLimit'])>0:
                        seg_dist,eta=current_data['profile']['Distance'][i+1]-distance,0
                        for j in range(i,len(current_data['profile']['Distance'])-1):
                            if j!=i:
                                seg_dist=current_data['profile']['Distance'][j+1]-current_data['profile']['Distance'][j]
                            if current_data['profile']['SpeedLimit'][j]!=0 and current_data['profile']['SpeedProfile'][j]!=0:
                                eta+=(seg_dist*1000)/(min((MAX_SPEED,current_data['profile']['SpeedLimit'][j],current_data['profile']['SpeedProfile'][j]))*(5/18))
                            elif (current_data['profile']['SpeedLimit'][j]!=0) ^ (current_data['profile']['SpeedProfile'][j]!=0):
                                eta+=(seg_dist*1000)/(min(max(current_data['profile']['SpeedProfile'][j],current_data['profile']['SpeedLimit'][j]),MAX_SPEED)*(5/18))
                            else:
                                eta+=(seg_dist*1000)/(MAX_SPEED*(5/18))
                        metric["ETA"]=eta

                    f=(distance-current_data['profile']['Distance'][i])/(current_data['profile']['Distance'][i+1]-current_data['profile']['Distance'][i])
                    lat1,lon1=current_data['profile']['Coordinates'][i]
                    lat2,lon2=current_data['profile']['Coordinates'][i+1]
                    alt1=current_data['profile']['Altitude'][i]
                    alt2=current_data['profile']['Altitude'][i+1]
                    grad1=current_data['profile']['Gradient'][i]
                    grad2=current_data['profile']['Gradient'][i+1]
                    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
                    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
                    cos_delta = math.sin(lat1_rad) * math.sin(lat2_rad) + math.cos(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)
                    cos_delta = max(-1.0, min(1.0, cos_delta))
                    delta = math.acos(cos_delta)
                    a = math.sin((1 - f) * delta) / math.sin(delta)
                    b = math.sin(f * delta) / math.sin(delta)
                    x = a * math.cos(lat1_rad) * math.cos(lon1_rad) + b * math.cos(lat2_rad) * math.cos(lon2_rad)
                    y = a * math.cos(lat1_rad) * math.sin(lon1_rad) + b * math.cos(lat2_rad) * math.sin(lon2_rad)
                    z = a * math.sin(lat1_rad) + b * math.sin(lat2_rad)
                    interp_lat = math.atan2(z, math.sqrt(x**2 + y**2))
                    interp_lon = math.atan2(y, x)
                    metric['Latitude'] = math.degrees(interp_lat)
                    metric['Longitude'] = math.degrees(interp_lon)
                    metric['Altitude']=alt1 +f*(alt2-alt1)
                    metric['Gradient']=grad1 + f*(grad2-grad1)
                    metric['Heading']=current_data['profile']['Headings'][i]
                    if current_data['profile']['TargetProfile']:
                        v1,v2=current_data['profile']['TargetProfile'][i],current_data['profile']['TargetProfile'][i+1]
                        metric['predicted']=v1 + f*(v2-v1)
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
                    'Altitude': metric['Altitude'] if metric['Altitude'] else pdata['Altitude'],
                    'Acceleration': math.sqrt(sum(pdata[f'acc_{i}']**2 for i in ('X', 'Y'))),
                    'Latitudes': metric['Latitude'] if metric['Latitude'] else pdata['Latitude'],
                    'Longitudes': metric['Longitude'] if metric['Longitude'] else pdata['Longitude'],
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
                for key in pdata:
                    current_data['profile'][key]=pdata[key]  
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
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://192.168.1.232:5173","http://localhost:8000","http://127.0.0.1:8000"],  # Add your Svelte dev server ports
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
@app.get("/api/data/profile")
async def get_profile_data():
    """Get all cached profile data for solvers"""
    return {
        'profile':current_data['profile']
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
@app.get("/api/live-car-gps")
def get_live_car_gps():
    # Format the live data exactly how Folium's Realtime plugin expects it
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [current_data['metric']["Longitude"], current_data['metric']["Latitude"]] # GeoJSON expects [Lon, Lat]
                },
                "properties": {
                    "id": "live-car-pin",
                    "bearing": current_data['metric']['Heading']
                }
            }
        ]
    }
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
        files=folder.findall("Placemark")
        points_list=[]
        for file in files:
            if file.find("Point") is not None:
                points_list.append(file)
        path_element = files[payload.placemark_index]
        route_name=path_element.find("name").text+":\n"+path_element.find("description").text if path_element.find("description") is not None else path_element.find("name").text
        coords_text = path_element.find("LineString/coordinates").text if path_element.find("LineString/coordinates") is not None else path_element.find("Polygon/outerBoundaryIs/LinearRing/coordinates").text
        coords_split = coords_text.split()
        colour,width,opacity=get_line_colour(root,path_element)
        colour="#"+colour if colour is not None else "#00ff66"
        width=5 if width is None else width
        opacity=0.9 if opacity is None else opacity
        route_info={"name":route_name,"colour":colour,"width":width,"opacity":opacity}
        coordinates = []
        relevant_points=[]
        for pair in coords_split:
            lon, lat, *_ = pair.split(",")
            lat,lon=float(lat),float(lon)
            coordinates.append((lat,lon))
            for point in points_list:
                p_lon,p_lat,_=point.find("Point/coordinates").text.split(",")
                p_lat,p_lon=float(p_lat),float(p_lon)
                icon_url,icon_anchor=get_icon_url(root,point)
                if geodesic((lat,lon),(p_lat,p_lon)).kilometers<1.5 and not {"name":point.find("name").text,"description":point.find("description").text if point.find("description") is not None else None,"coordinates":(p_lat,p_lon),"url":icon_url,"anchor":icon_anchor} in relevant_points:
                    relevant_points.append({"name":point.find("name").text,"description":point.find("description").text if point.find("description") is not None else None,"coordinates":(p_lat,p_lon),"url":icon_url,"anchor":icon_anchor})
        
        results = maps_main(route_info,coordinates,relevant_points)
        map_html,smoothed_altitude,distance_profile,coordinates,speed_limit,eta=results["Map"],results["Altitude"],results["Distances"],results["Coordinates"],results["SpeedLimit"],results["ETA"]
        speed_profile=results["SpeedProfile"]
        gradient_profile = []
        for i in range(1,len(smoothed_altitude)):
            rise=(smoothed_altitude[i]-smoothed_altitude[i-1])
            run=(distance_profile[i]-distance_profile[i-1])
            gradient=(rise/run)*0.1 if run!=0 else 0
            gradient_profile.append(gradient)
        gradient_profile=savgol_filter(gradient_profile,window_length=11,polyorder=3)
        current_data['metric']["ETA"]=eta
        packet_c = {
            "Altitude": smoothed_altitude,
            "Gradient": np.clip(gradient_profile,min=-7.5,max=7.5).tolist(),
            "Distance": distance_profile,
            "Coordinates": coordinates,
            "SpeedLimit": speed_limit,
            "SpeedProfile": speed_profile,
            "Headings":results["Headings"]
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