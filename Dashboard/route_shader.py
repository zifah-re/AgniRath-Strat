# import json
# from pathlib import Path
# from constants import MASS,RHO, CDA,G,MOTOR_EFF,REGEN_EFF,CRR

# SCRIPT_DIR = Path(__file__).resolve().parent
# file_name=input("Enter file name: ")
# FILE_PATH=SCRIPT_DIR / "Saves" / file_name
# file=open(FILE_PATH,'r')
# data=file.read()
# data=json.loads(data)

# distance_profile=data['profile']['Distance']
# gradient_profile=data['profile']['Gradient']
# speed_limit=data['profile']['SpeedLimit']
# coords=data['profile']['Coordinates']
# speed=80/3.6
# safe=[]
# motor_power=[]
# speeds=[]
# last_downhill=0
# for i in range(0,len(gradient_profile)-1):
#     if gradient_profile[i]<0 and gradient_profile[i+1]>=0:
#         last_downhill=i+1
#         safe.append(True)
#         speed=80/3.6
#         motor_power.append(0)
#         speeds.append(speed*3.6)
#         continue
#     elif gradient_profile[i]<0:
#         safe.append(True)
#         motor_power.append(0)
#         speeds.append(speed*3.6)
#         continue
#     elif gradient_profile[i]>0 and safe[-1] if len(safe)>0 else True:
#         grad = gradient_profile[i] / 100
#         f_drag = 0.5 * RHO * CDA * (speed ** 2)
#         f_rolling = MASS * G * CRR * (1 - (grad**2)/2)
#         f_gravity = MASS * G * grad
#         dt = ((distance_profile[i+1]-distance_profile[i]) / speed) * 1000
#         P_MOTOR_MAX=4000*0.90
#         LOWEST_SPEED=60/3.6 if speed_limit[i]>=100 else 40/3.6
#         if LOWEST_SPEED <= (((P_MOTOR_MAX*MOTOR_EFF)/speed -(f_drag+f_rolling+f_gravity))*dt/MASS) +speed:
#             speed2= (((P_MOTOR_MAX*MOTOR_EFF)/speed -(f_drag+f_rolling+f_gravity))*dt/MASS) +speed
#             safe.append(True)
#         else:
#             safe.append(False)
#             safe[last_downhill:i+1]=[False]*(i+1-last_downhill)
#         f_acceleration = MASS * (speed2 - speed) / dt
#         f_total = f_drag + f_rolling + f_gravity + f_acceleration
#         p_mech = f_total * speed
#         p_electric = p_mech / MOTOR_EFF
#         speed=speed2
#         speeds.append(speed*3.6)
#         motor_power.append(p_electric)
#     elif gradient_profile[i]>0 and not safe[-1]:
#         safe.append(False)
#         speeds.append(speed*3.6)
#         continue
# if False in safe:
#     print(safe.count(False))
#     print(list(zip(safe,gradient_profile,motor_power)))
# else:
#     print("Safe!")
#     print(f"Max gradient {max(gradient_profile)}")
#     print(f"Max motor power {max(motor_power)}")
#     print(f"Max speed {max(speeds)}")
#     input()
#     print(list(zip(safe,gradient_profile,speeds)))

import json
from pathlib import Path
from constants import MASS, RHO, CDA, G, MOTOR_EFF, REGEN_EFF, CRR

SCRIPT_DIR = Path(__file__).resolve().parent
file_name = input("Enter file name: ")
FILE_PATH = SCRIPT_DIR / "Saves" / file_name

with open(FILE_PATH, 'r') as f:
    data = json.loads(f.read())

distance_profile = data['profile']['Distance']      # kilometres
gradient_profile = data['profile']['Gradient']       # percent
speed_limit = data['profile']['SpeedLimit']          # km/h
coords = data['profile']['Coordinates']

# =====================================================================
# Patch speed_limit: fill 0s by interpolating from nearest valid neighbours
# TomTom sometimes returns 0 where it has no data.
# =====================================================================
def patch_speed_limits(sl):
    """Replace 0s with linearly interpolated values from nearest non-zero neighbours."""
    patched = sl[:]
    n = len(patched)
    i = 0
    total_patched = 0
    while i < n:
        if patched[i] <= 0:
            j = i
            while j < n and patched[j] <= 0:
                j += 1
            left = patched[i - 1] if i > 0 else None
            right = patched[j] if j < n else None

            if left is not None and right is not None:
                span = j - i + 2
                for k in range(i, j):
                    t = (k - i + 1) / span
                    patched[k] = left + t * (right - left)
            elif left is not None:
                for k in range(i, j):
                    patched[k] = left
            elif right is not None:
                for k in range(i, j):
                    patched[k] = right
            else:
                for k in range(i, j):
                    patched[k] = 120

            total_patched += (j - i)
            i = j
        else:
            i += 1

    if total_patched > 0:
        print(f"WARNING: Patched {total_patched} points with speed_limit=0 (TomTom data gap)")
    return patched

speed_limit = patch_speed_limits(speed_limit)

# --- Constants ---
P_MOTOR_MAX = 4000.0    # 3600 W continuous (derated from 4kW peak)
CRUISE_SPEED = 90 / 3.6         # m/s — target cruising speed
LOWEST_SPEED_HIGHWAY = 60 / 3.6 # m/s — failure floor on highways (speed_limit >= 100)
LOWEST_SPEED_OTHER   = 40 / 3.6 # m/s — failure floor elsewhere
TRAILER_EXIT_SPEED   = 90 / 3.6 # m/s — assumed speed after being trailered
V_FLOOR = 5 / 3.6               # m/s — absolute minimum to prevent division by zero


def get_v_limit(i):
    """Speed limit at point i (already patched, but belt-and-suspenders)."""
    sl = speed_limit[i]
    if sl <= 0:
        return CRUISE_SPEED
    return min(CRUISE_SPEED, sl / 3.6)


def get_speed_floor(i):
    """Minimum acceptable speed at point i."""
    return LOWEST_SPEED_HIGHWAY if speed_limit[i] >= 100 else LOWEST_SPEED_OTHER


def simulate(safe_mask):
    """
    Run a single forward simulation through the entire route.
    Segments where safe_mask[i] is False are skipped (trailered).
    Returns (speeds, motor_power, failed_indices).
    """
    spd = [0.0] * n
    mpow = [0.0] * n
    failed = []

    v = CRUISE_SPEED

    for i in range(n):
        # Skip trailered segments
        if not safe_mask[i]:
            spd[i] = 0.0
            mpow[i] = 0.0
            if i + 1 < n and safe_mask[i + 1]:
                v = TRAILER_EXIT_SPEED
            continue

        v = max(v, V_FLOOR)

        grad = gradient_profile[i] / 100.0
        v_limit = get_v_limit(i)

        # --- Forces ---
        f_drag = 0.5 * RHO * CDA * (v ** 2)
        f_roll = MASS * G * CRR
        f_grav = MASS * G * grad

        # --- Motor strategy ---
        if grad > 0:
            f_motor = (P_MOTOR_MAX * MOTOR_EFF) / v
            p_elec = P_MOTOR_MAX
        else:
            f_resist = f_drag + f_roll + f_grav
            if f_resist > 0 and v <= v_limit:
                p_mech_needed = f_resist * v
                p_mech = min(p_mech_needed, P_MOTOR_MAX * MOTOR_EFF)
                f_motor = p_mech / v
                p_elec = p_mech / MOTOR_EFF
            else:
                f_motor = 0.0
                p_elec = 0.0

        f_net = f_motor - f_drag - f_roll - f_grav
        a = f_net / MASS

        spd[i] = v * 3.6
        mpow[i] = p_elec

        # Speed floor check — distinguish real hill failures from speed-limit
        # transitions. A car exiting a 60 km/h zone onto a mild uphill highway
        # is temporarily below 60 but will recover — that's not a hill problem.
        #
        # Test: can the motor sustain the floor speed on this gradient?
        # Evaluate f_net at v=floor. If negative, the hill genuinely defeats
        # the motor. If positive, the car can recover — it's just slow from
        # a speed limit zone upstream.
        floor = get_speed_floor(i)
        if grad > 0 and v < floor:
            f_drag_at_floor = 0.5 * RHO * CDA * (floor ** 2)
            f_motor_at_floor = (P_MOTOR_MAX * MOTOR_EFF) / floor
            f_net_at_floor = f_motor_at_floor - f_drag_at_floor - f_roll - f_grav
            if f_net_at_floor < 0:
                # Motor can't sustain floor speed on this gradient → real failure
                failed.append(i)

        # --- Advance ---
        if i + 1 < len(distance_profile):
            ds = (distance_profile[i + 1] - distance_profile[i]) * 1000
            if ds > 0:
                dt = ds / v
                v_new = v + a * dt
                v = max(v_new, V_FLOOR)
                v = min(v, v_limit)

    return spd, mpow, failed


# =====================================================================
# Find contiguous uphill stretches
# =====================================================================
n = len(gradient_profile)
stretches = []
i = 0
while i < n:
    if gradient_profile[i] > 0:
        start = i
        while i < n and gradient_profile[i] > 0:
            i += 1
        stretches.append((start, i - 1))
    else:
        i += 1

# =====================================================================
# Iterative simulation: first pass gets actual speeds, subsequent passes
# resolve trailer cascades.
# =====================================================================
safe = [True] * n
MAX_ITERATIONS = 10

# First pass — captures the real speeds before any trailering
first_pass_speeds, first_pass_power, failed = simulate(safe)

if failed:
    new_failures = set(failed)
    for (s, e) in stretches:
        if any(j in new_failures for j in range(s, e + 1)):
            for j in range(s, e + 1):
                safe[j] = False

    # Subsequent passes — resolve cascading failures from trailering
    for iteration in range(1, MAX_ITERATIONS):
        speeds, motor_power, failed = simulate(safe)
        if not failed:
            break
        new_failures = set(failed)
        for (s, e) in stretches:
            if any(j in new_failures for j in range(s, e + 1)):
                for j in range(s, e + 1):
                    safe[j] = False
    iteration_count = iteration + 1
else:
    speeds = first_pass_speeds
    motor_power = first_pass_power
    iteration_count = 1

# =====================================================================
# Report — use first_pass_speeds for min_speed so we show the actual
# speed the car had when it failed, not 0.0 from the trailered pass.
# =====================================================================
unsafe_count = safe.count(False)
if unsafe_count > 0:
    print(f"\n{unsafe_count} unsafe points (need trailer)\n")
    print("Failed uphill stretches:")
    for (s, e) in stretches:
        if not safe[s]:
            peak_grad = max(gradient_profile[s:e+1])
            dist_start = distance_profile[s]
            dist_end = distance_profile[e] if e < len(distance_profile) else distance_profile[-1]
            length_m = (dist_end - dist_start) * 1000
            min_speed = min(first_pass_speeds[s:e+1])
            entry_speed = first_pass_speeds[s]
            print(f"  idx {s:>5d}-{e:<5d}  |  dist {dist_start:>7.1f}-{dist_end:<7.1f} km  "
                  f"|  length {length_m:>6.0f} m  |  peak grad {peak_grad:>5.2f}%  "
                  f"|  entry {entry_speed:>5.1f} → min {min_speed:>5.1f} km/h")
    print(f"\nConverged in {iteration_count} iteration(s)")
else:
    print("\nAll clear — car can handle every hill on the route!")
    print(f"  Max gradient:    {max(gradient_profile):.2f}%")
    print(f"  Min speed seen:  {min(first_pass_speeds):.1f} km/h")
    print(f"  Max motor power: {max(first_pass_power):.0f} W")
    print(f"  Uphill stretches checked: {len(stretches)}")