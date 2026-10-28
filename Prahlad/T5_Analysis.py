import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR.parent / "Dashboard" / "Logs"

file_names_input = input("Enter file names separated by commas: ")
file_names = [f.strip() for f in file_names_input.split(",") if f.strip()]

colours={"T5":"blue",
         "T6": "orange",
         "T7": "green",
         "T8": "red"}
plt.figure(figsize=(10, 6))

for file_name in file_names:
    file_path = LOGS_DIR / file_name
    if not file_path.exists():
        continue
        
    df = pd.read_json(file_path, lines=True, convert_dates=False)
    
    speed_kmh = df['Vehicle_Velocity'] * 3.6
    power_w = df['Bus_Voltage'] * df['Bus_Current']
    
    # Small size (s=6) and alpha (0.25) keep points legible when overlapping
    plt.scatter(speed_kmh, power_w, s=6,c=colours[re.search(r'T\d',file_name).group(0)], alpha=0.25, label=file_name, edgecolors='none')

plt.title("Power vs Velocity Comparison")
plt.xlabel("Vehicle Velocity (km/h)")
plt.ylabel("Power (W)")
plt.grid(True, linestyle="--", alpha=0.5)

# High alpha on legend markers so colors remain clear
leg = plt.legend(loc="upper left")
for lh in leg.legend_handles:
    lh.set_alpha(1.0)
    lh.set_sizes([30])

plt.tight_layout()
plt.show()