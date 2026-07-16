import numpy as np
import math

# Vehicle constants
MASS = 330 # kg
CRR = 0.0048 # rolling resistance coefficient
CD = 0.0845 # drag coefficient
A_FRONTAL = 1.155 # m² frontal area
RHO = 1.05 # kg/m³ approx avg air density between Sasolburg to Zeerust
ETA_MOTOR= 0.90 # motor efficiency
ETA_REGEN = 0.70 # regen braking efficiency
P_AUX= 20        # W auxiliary loads for telemetry, etc

# Solar panel
PANEL_AREA  = 6.0 # m²
PANEL_EFF   = 0.2 # 20% efficiency

# Battery
E_MAX   = 3 * 1000 * 3600# 3 kWh in Joules
SOC_MIN     = 0.20

def calculate_irradiance(t):
    # Gaussian model centred at solar noon (t=14400s = 12:00 PM)
    if 0 < t< 32400:
        A = 1073
        mu = 14400 # Time of peak irradiance in seconds taking 8:00 am as t = 0
        sigma = 11600
        irradiance = A* np.exp(-0.5*((t-mu)/sigma)**2)
        return irradiance
    else:
        return 0

def calculate_power(v, t, slope, accel=0.0):
    P_solar = PANEL_EFF * PANEL_AREA * calculate_irradiance(t)
    P_drag = 0.5 * RHO *CD * A_FRONTAL * v**3 
    P_rolling = CRR * MASS * 9.81 * v *math.cos(math.radians(slope))
    P_grade = MASS * 9.81 * v * math.sin(math.radians(slope)) 
    P_accel = MASS * accel * v
    P_aux = 20 # Auxiliary power for lights, telemetry sensors, etc
    
    P_mech = P_drag + P_rolling + P_grade + P_accel
    
    if P_mech > 0:
        P_elec = P_mech/ETA_MOTOR + P_aux - P_solar
    else:
        P_elec = P_mech*ETA_REGEN + P_aux - P_solar
    
    return P_elec

def calculate_soc_update(SOC, P_total, dt):
    E_current = SOC * E_MAX
    E_new = E_current - P_total * dt
    SOC_new = E_new / E_MAX
    return np.clip(SOC_new, 0, 1)