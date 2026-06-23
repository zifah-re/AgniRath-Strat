import json
import math
import random
import time
import urllib.request
from datetime import datetime
from typing import Optional, Tuple

URL = "http://127.0.0.1:8000/api/simulate"
VALUE_INTERVAL = 2.0
SEND_HZ = 10.0

MPPT_NAMES = ("A", "B", "C", "D")

PACKET_A_BOOL_PREFIXES = (
    ("Precharge_Contactor_Flag", 8),
    ("Precharge_State_Flag", 5),
    ("BMS_Flag", 13),
    ("MC_Limit_Flag", 7),
    ("MC_Error_Flag", 9),
)
PACKET_A_MPPT_FLAGS = [f"MPPT_{m}_Flag{i}" for m in MPPT_NAMES for i in range(1, 9)]

_cached_a: Optional[dict] = None
_cached_b: Optional[dict] = None
_last_refresh = 0.0
_phase = 0

_sim_battery_wh = 588.0
_current_soc = 100.0
_last_sim_time = None

_control_points = [
    (100.0, 4.12),
    (98.0, 4.05),
    (96.0, 4.00),
    (94.0, 3.96),
    (92.0, 3.94),
    (90.0, 3.92),
    (88.0, 3.90),
    (86.0, 3.88),
    (84.0, 3.86),
    (82.0, 3.84),
    (80.0, 3.82),
    (76.0, 3.79),
    (72.0, 3.76),
    (68.0, 3.73),
    (64.0, 3.69),
    (60.0, 3.65),
    (56.0, 3.61),
    (52.0, 3.57),
    (48.0, 3.52),
    (44.0, 3.47),
    (40.0, 3.42),
    (36.0, 3.37),
    (32.0, 3.31),
    (28.0, 3.25),
    (24.0, 3.18),
    (20.0, 3.11),
    (16.0, 3.03),
    (12.0, 2.93),
    (8.0, 2.81),
    (6.0, 2.73),
    (4.0, 2.62),
    (2.0, 2.48),
    (0.0, 2.45)
]

def get_pack_voltage_from_soc(soc: float) -> float:
    if soc >= 100.0:
        return 4.12 * 28.0
    if soc <= 0.0:
        return 2.45 * 28.0
    for i in range(len(_control_points) - 1):
        soc_upper, v_upper = _control_points[i]
        soc_lower, v_lower = _control_points[i+1]
        if soc_lower <= soc <= soc_upper:
            ratio = (soc - soc_lower) / (soc_upper - soc_lower)
            cell_v = v_lower + ratio * (v_upper - v_lower)
            return cell_v * 28.0
    return 2.45 * 28.0

def _rand_bool(p_true: float = 0.35) -> bool:
    return random.random() < p_true


def _refresh_packet_a() -> dict:
    global _phase, _sim_battery_wh, _current_soc, _last_sim_time
    _phase += 1
    t = _phase * VALUE_INTERVAL

    now = time.time()
    if _last_sim_time is None:
        dt = VALUE_INTERVAL
    else:
        dt = now - _last_sim_time
    _last_sim_time = now

    vehicle_velocity = max(0.0, 6.0 + 8.0 * math.sin(t / 7.0) + random.uniform(-2, 2))
    motor_velocity = vehicle_velocity * random.uniform(25.0, 35.0) + random.uniform(0, 50.0)
    speed = vehicle_velocity * random.uniform(0.9, 1.1)

    pack_voltage = get_pack_voltage_from_soc(_current_soc)
    pack_voltage_mv = pack_voltage * 1000.0
    bus_voltage = pack_voltage

    bus_current = max(0.0, vehicle_velocity * random.uniform(4.0, 8.0) + random.uniform(0, 15.0))
    motor_power = bus_voltage * bus_current

    input_v_base = {"A": 58.7, "B": 52.5, "C": 44.7, "D": 39.8}

    solar_power = 0.0
    mppt_data = {}
    for m in MPPT_NAMES:
        in_v = input_v_base[m] + random.uniform(-3, 3)
        in_i = random.uniform(0.5, 1.4)
        out_v = bus_voltage + random.uniform(-0.1, 0.1) 
        out_i = random.uniform(0.35, 0.85)
        solar_power += out_v * out_i
        mppt_data[m] = (in_v, in_i, out_v, out_i)

    net_power = solar_power - motor_power
    energy_change_wh = net_power * (dt / 3600.0)
    
    _sim_battery_wh += energy_change_wh
    _sim_battery_wh = max(0.0, min(588.0, _sim_battery_wh))
    _current_soc = (_sim_battery_wh / 588.0) * 100.0

    pack_voltage = get_pack_voltage_from_soc(_current_soc)
    pack_voltage_mv = pack_voltage * 1000.0
    pack_current = bus_current - (solar_power / bus_voltage)

    packet: dict = {
        "Timestamp": 0.0,
        "SOC_Ah": round(_current_soc, 4),
        "Pack_Voltage": round(pack_voltage_mv, 1),
        "Pack_Current": round(pack_current, 4),
        "Bus_Voltage": round(bus_voltage, 4),
        "Bus_Current": round(bus_current, 4),
        "Motor_Velocity": round(motor_velocity, 4),
        "Vehicle_Velocity": round(vehicle_velocity, 4),
        "PhaseC_Current": round(bus_current * random.uniform(0.48, 0.52), 4),
        "PhaseB_Current": round(bus_current * random.uniform(0.48, 0.52), 4),
        "Latitude": round(-12.446822 + random.uniform(-1e-4, 1e-4), 6),
        "Longitude": round(130.907036 + random.uniform(-1e-4, 1e-4), 6),
        "Altitude": round(15.0 + random.uniform(-2, 2), 2),
        "Speed": round(speed, 4),
        "acc_X": round(random.uniform(-0.5, 0.5), 4),
        "acc_Y": round(random.uniform(-0.5, 0.5), 4),
        "acc_Z": round(random.uniform(-0.2, 0.2), 4),
        "Throttle_Perc": round(min(1.0, vehicle_velocity / 20.0), 4),
        "Brake_Status": int(vehicle_velocity < 0.8),
    }

    for m in MPPT_NAMES:
        in_v, in_i, out_v, out_i = mppt_data[m]
        packet[f"Input_Voltage_{m}"] = round(in_v, 4)
        packet[f"Input_Current_{m}"] = round(in_i, 4)
        packet[f"Output_Voltage_{m}"] = round(out_v, 4)
        packet[f"Output_Current_{m}"] = round(out_i, 4)

    for prefix, count in PACKET_A_BOOL_PREFIXES:
        for i in range(1, count + 1):
            packet[f"{prefix}{i}"] = _rand_bool(0.25 if prefix.startswith("MC_Error") else 0.4)

    for key in PACKET_A_MPPT_FLAGS:
        packet[key] = _rand_bool(0.2)

    return packet


def _refresh_packet_b() -> dict:
    global _phase, _current_soc
    heat = max(0.0, (_phase % 10) * 0.5)

    packet: dict = {
        "Timestamp": 0.0,
        "Motor_Temp": round(4.5 + heat + random.uniform(-1, 1), 4),
        "HeatSink_Temp": round(48.0 + heat * 2 + random.uniform(-2, 2), 4),
        "DSP_Board_Temp": round(54.0 + heat + random.uniform(-1.5, 1.5), 4),
        "Cabin_CO_Content": int(random.uniform(0, 8)),
        "Cabin_CH4_Content": int(random.uniform(0, 8)),
        "Cabin_NH3_Content": int(random.uniform(0, 8)),
        "Cabin_NO2_Content": int(random.uniform(0, 8)),
        "Cabin_O2_Content": int(random.uniform(19, 21)),
        "Cabin_Temperature": int(25 + random.uniform(-4, 8)),
        "Cabin_Pressure": int(1010 + random.uniform(-8, 8)),
        "Cabin_CO2_Content": int(380 + random.uniform(-50, 80)),
    }

    for m in MPPT_NAMES:
        packet[f"Mosfet_Temp_{m}"] = round(41.0 + heat + random.uniform(-2, 2), 4)
        packet[f"Controller_Temp_{m}"] = round(43.0 + heat + random.uniform(-2, 2), 4)

    for i in range(1, 6):
        if i <= 4:
            packet[f"CMU{i}_Temp"] = round(41.0 + i * 0.4 + random.uniform(-1.5, 1.5), 4)
            packet[f"Cell{i}_Temp"] = round(9.0 + random.uniform(-1, 2), 4)
        else:
            packet[f"CMU{i}_Temp"] = 0.0
            packet[f"Cell{i}_Temp"] = 0.0

    cell_v_mv = (get_pack_voltage_from_soc(_current_soc) / 28.0) * 1000.0

    for i in range(1, 6):
        for j in range(8):
            key = f"CMU{i}_Cell{j}_Voltage"
            if i > 4 or j >= 6:
                packet[key] = 0.0
            else:
                packet[key] = round(cell_v_mv + random.uniform(-10, 10), 1)

    return packet


def _maybe_refresh():
    global _cached_a, _cached_b, _last_refresh
    now = time.monotonic()
    if _cached_a is None or (now - _last_refresh) >= VALUE_INTERVAL:
        _cached_a = _refresh_packet_a()
        _cached_b = _refresh_packet_b()
        _last_refresh = now


def current_packets() -> Tuple[dict, dict]:
    _maybe_refresh()
    return _cached_b.copy(), _cached_a.copy()


def send_data(packet: dict) -> bool:
    req = urllib.request.Request(
        URL,
        data=json.dumps(packet, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


def main():
    print("=" * 50)
    print("Telemetry simulator (JSONL fields only)")
    print(f"Target: {URL}")
    print(f"Values refresh every {VALUE_INTERVAL}s | send {SEND_HZ} Hz")
    print("=" * 50)

    interval = 1.0 / SEND_HZ
    while True:
        try:
            pkt_b, pkt_a = current_packets()
            ok_b = send_data(pkt_b)
            ok_a = send_data(pkt_a)

            if ok_b and ok_a:
                if random.random() < 0.05:
                    solar_p = sum(
                        pkt_a[f"Output_Voltage_{m}"] * pkt_a[f"Output_Current_{m}"]
                        for m in MPPT_NAMES
                    )
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"solar {solar_p:.0f} W | bus "
                        f"{pkt_a['Bus_Voltage'] * pkt_a['Bus_Current']:.0f} W | "
                        f"SOC_Ah {pkt_a['SOC_Ah']:.2f}"
                    )
            else:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    "Connection refused — is main.py running?"
                )
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    main()
