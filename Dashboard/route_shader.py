import json
from pathlib import Path
from constants import MASS,RHO, CDA,G,MOTOR_EFF,REGEN_EFF,CRR

SCRIPT_DIR = Path(__file__).resolve().parent
file_name=input("Enter file name: ")
FILE_PATH=SCRIPT_DIR / "Saves" / file_name
file=open(FILE_PATH,'r')
data=file.read()
data=json.loads(data)

distance_profile=data['profile']['Distance']
gradient_profile=data['profile']['Gradient']
speed_limit=data['profile']['SpeedLimit']
coords=data['profile']['Coordinates']
speed=80/3.6
safe=[]
motor_power=[]
speeds=[]
last_downhill=0
for i in range(0,len(gradient_profile)-1):
    if gradient_profile[i]<0 and gradient_profile[i+1]>=0:
        last_downhill=i+1
        safe.append(True)
        speed=80/3.6
        motor_power.append(0)
        speeds.append(speed*3.6)
        continue
    elif gradient_profile[i]<0:
        safe.append(True)
        motor_power.append(0)
        speeds.append(speed*3.6)
        continue
    elif gradient_profile[i]>0 and safe[-1] if len(safe)>0 else True:
        grad = gradient_profile[i] / 100
        f_drag = 0.5 * RHO * CDA * (speed ** 2)
        f_rolling = MASS * G * CRR * (1 - (grad**2)/2)
        f_gravity = MASS * G * grad
        dt = ((distance_profile[i+1]-distance_profile[i]) / speed) * 1000
        P_MOTOR_MAX=4000*0.90
        LOWEST_SPEED=60/3.6 if speed_limit[i]>=100 else 40/3.6
        if LOWEST_SPEED <= (((P_MOTOR_MAX*MOTOR_EFF)/speed -(f_drag+f_rolling+f_gravity))*dt/MASS) +speed:
            speed2= (((P_MOTOR_MAX*MOTOR_EFF)/speed -(f_drag+f_rolling+f_gravity))*dt/MASS) +speed
            safe.append(True)
        else:
            safe.append(False)
            safe[last_downhill:i+1]=[False]*(i+1-last_downhill)
        f_acceleration = MASS * (speed2 - speed) / dt
        f_total = f_drag + f_rolling + f_gravity + f_acceleration
        p_mech = f_total * speed
        p_electric = p_mech / MOTOR_EFF
        speed=speed2
        speeds.append(speed*3.6)
        motor_power.append(p_electric)
    elif gradient_profile[i]>0 and not safe[-1]:
        safe.append(False)
        speeds.append(speed*3.6)
        continue
if False in safe:
    print(safe.count(False))
    print(list(zip(safe,gradient_profile,motor_power)))
else:
    print("Safe!")
    print(f"Max gradient {max(gradient_profile)}")
    print(f"Max motor power {max(motor_power)}")
    print(f"Max speed {max(speeds)}")
    input()
    print(list(zip(safe,gradient_profile,speeds)))

