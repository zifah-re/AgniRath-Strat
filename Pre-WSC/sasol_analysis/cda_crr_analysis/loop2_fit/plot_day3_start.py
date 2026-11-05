"""Day 3, start of the day: Stage 1, the two loops and the stops after them.

Speed, motor bus power (Bus_Voltage x Bus_Current) and distance from integrated speed (v*dt, gaps > 30 s add nothing)
against clock time. Stops >= 2 min are shaded grey, telemetry gaps > 60 s are hatched red.
Distance landmarks are the route-file lengths: Stage 1 8.1 km, each loop 39.9 km.
"""
import json, os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, '..', '..', 'cleaned_logs', 'day3.jsonl')
T_END = '17:25'
BG, SURF, INK, INK2, GRID = '#141413', '#1F1E1D', '#F0EEE6', '#A8A69C', '#3A3936'
ORANGE, BLUE, GOLD, RED = '#D97757', '#3987E5', '#C98500', '#E5484D'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'grid.color': GRID, 'axes.grid': True,
                     'grid.alpha': 0.6, 'font.size': 10, 'legend.facecolor': SURF, 'legend.edgecolor': GRID})

t, v, p = [], [], []
with open(LOG, encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if 'Vehicle_Velocity' in r:
            t.append(datetime.fromisoformat(r['_rx_time']).timestamp()); v.append(abs(r['Vehicle_Velocity']))
            p.append(r.get('Bus_Voltage', np.nan) * r.get('Bus_Current', np.nan))
o = np.argsort(t); t, v, p = np.array(t)[o], np.array(v)[o], np.array(p)[o]
day0 = datetime.fromtimestamp(t[0]).replace(hour=0, minute=0, second=0).timestamp()
hh, mm = map(int, T_END.split(':')); keep = t <= day0 + hh * 3600 + mm * 60
t, v, p = t[keep], v[keep], p[keep]
dt = np.diff(t)
dist = np.r_[0, np.cumsum(np.where(dt <= 30, 0.5 * (v[1:] + v[:-1]) * dt, 0))] / 1000
T = [datetime.fromtimestamp(x) for x in t]

stops, i = [], 0
while i < len(t):
    if v[i] < 0.5:
        j = i
        while j + 1 < len(t) and v[j + 1] < 0.5:
            j += 1
        if t[j] - t[i] >= 120:
            stops.append((i, j))
        i = j + 1
    else:
        i += 1
gaps = np.where(dt > 60)[0]

fig, ax = plt.subplots(3, 1, figsize=(15, 9.5), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1.1, 1]))
for a in ax:
    for i, j in stops:
        a.axvspan(T[i], T[j], color=INK2, alpha=0.18, lw=0)
    for k in gaps:
        a.axvspan(T[k], T[k + 1], facecolor='none', edgecolor=RED, hatch='///', alpha=0.5, lw=0)
ax[0].plot(T, v * 3.6, color=ORANGE, lw=1.1)
ax[0].set_ylabel('speed [km/h]')
ax[1].plot(T, p, color=BLUE, lw=0.7)
ax[1].axhline(0, color=INK2, lw=0.6)
ax[1].set_ylabel('motor bus power [W]')
ax[2].plot(T, dist, color=GOLD, lw=1.6)
ax[2].set_ylabel('distance, v·dt [km]')
for km, lab in [(8.1, 'end of Stage 1 (8.1 km)'), (48.0, 'end of Loop 1 (48.0 km)'), (87.9, 'end of Loop 2 (87.9 km)')]:
    ax[2].axhline(km, color=INK2, ls='--', lw=0.8)
    ax[2].text(T[0], km + 2, lab, color=INK2, fontsize=8)
for i, j in stops:
    if (t[j] - t[i]) >= 240:
        ax[0].text(T[i], 92, f"{T[i]:%H:%M}\n{(t[j] - t[i]) / 60:.0f} min", fontsize=7.5, color=INK2, va='top')
ax[0].set_ylim(0, 95)
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax[2].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
ax[0].set_title('Day 3, start of day: Stage 1, two loops, then the stops (grey = stopped ≥ 2 min, red hatch = telemetry gap > 1 min)')
fig.tight_layout()
out = os.path.join(HERE, 'day3_start.png')
fig.savefig(out, dpi=140)
print('saved', out)
for i, j in stops:
    print(f'stop {T[i]:%H:%M}-{T[j]:%H:%M} {(t[j] - t[i]) / 60:5.1f} min at {dist[i]:6.1f} km')
for k in gaps:
    print(f'gap  {T[k]:%H:%M}-{T[k + 1]:%H:%M} {dt[k] / 60:5.1f} min at {dist[k]:6.1f} km, v {v[k] * 3.6:.0f} -> {v[k + 1] * 3.6:.0f} km/h')
