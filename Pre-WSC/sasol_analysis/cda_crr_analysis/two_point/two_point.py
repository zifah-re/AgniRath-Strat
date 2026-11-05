"""Lead's method: pick two steady points (speed, power), solve 2 equations for Crr and CdA.
Steady point = 60 s with speed range < 2 km/h, motor drawing power, no gap > 10 s.
Done for every pair of points from the same continuous stretch with a speed difference >= 12 km/h
(same stretch = same day/weather), to show how much the answer depends on which two points you pick."""
import json, os
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); LOGS = os.path.join(HERE, '..', '..', 'cleaned_logs')
M, G, ETA, RHO, PAUX = 336.5, 9.81, 0.95, 1.07, 60.0

def load(day):
    t, v, p = [], [], []
    for line in open(os.path.join(LOGS, f'day{day}.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        if 'Vehicle_Velocity' in r and 'Bus_Current' in r:
            t.append(datetime.fromisoformat(r['_rx_time']).timestamp()); v.append(abs(r['Vehicle_Velocity'])); p.append(r['Bus_Voltage'] * r['Bus_Current'])
    o = np.argsort(t); return np.array(t)[o], np.array(v)[o], np.array(p)[o]

allsol = []
for day in [1, 2, 3, 5, 6, 7, 8]:
    t, v, p = load(day); pts = []; i = 0; seg = 0
    while i < len(t) - 5:
        j = int(np.searchsorted(t, t[i] + 60))
        if j >= len(t): break
        if t[i] - t[i - 1] > 30 if i else False: seg += 1
        sl = slice(i, j + 1); kmh = v[sl] * 3.6
        if np.diff(t[sl]).max() <= 10 and kmh.min() > 20 and np.ptp(kmh) < 2 and p[sl].min() > 0:
            T = t[j] - t[i]; tr = lambda f: np.trapezoid(f, t[sl]) / T
            pts.append((seg, tr(v[sl]), tr(v[sl] ** 3), tr(p[sl]))); i = j
        else:
            if np.any(v[sl] < 0.5): seg += 1
            i += 3
    pts = np.array(pts); sols = []
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            if pts[a, 0] != pts[b, 0] or abs(pts[a, 1] - pts[b, 1]) * 3.6 < 12: continue
            X = np.array([[M * G * pts[k, 1] / ETA, 0.5 * RHO * pts[k, 2] / ETA] for k in (a, b)])
            c = np.linalg.solve(X, [pts[a, 3] - PAUX, pts[b, 3] - PAUX]); sols.append(c)
    sols = np.array(sols)
    if len(sols):
        phys = (sols[:, 0] > 0.002) & (sols[:, 0] < 0.03) & (sols[:, 1] > 0.03) & (sols[:, 1] < 0.4)
        print(f'Day {day}: {len(pts)} steady points, {len(sols)} pairs >=12 km/h apart, physical {phys.sum()} ({100 * phys.mean():.0f}%)'
              + (f' -> median Crr {np.median(sols[phys, 0]):.4f} [IQR {np.percentile(sols[phys, 0], 25):.4f}-{np.percentile(sols[phys, 0], 75):.4f}], CdA {np.median(sols[phys, 1]):.3f} [IQR {np.percentile(sols[phys, 1], 25):.3f}-{np.percentile(sols[phys, 1], 75):.3f}]' if phys.any() else ''))
        print(f'        all pairs: Crr 10-90% {np.percentile(sols[:, 0], 10):.4f} .. {np.percentile(sols[:, 0], 90):.4f}, CdA 10-90% {np.percentile(sols[:, 1], 10):.3f} .. {np.percentile(sols[:, 1], 90):.3f}')
        allsol.append(sols)
    else:
        print(f'Day {day}: {len(pts)} steady points, no usable pair')
S = np.vstack(allsol); ph = (S[:, 0] > 0.002) & (S[:, 0] < 0.03) & (S[:, 1] > 0.03) & (S[:, 1] < 0.4)
print(f'\nALL: {len(S)} pairs, physical {ph.sum()} ({100 * ph.mean():.0f}%), median of physical: Crr {np.median(S[ph, 0]):.4f}, CdA {np.median(S[ph, 1]):.3f}')
print(f'     median of all pairs: Crr {np.median(S[:, 0]):.4f}, CdA {np.median(S[:, 1]):.3f}')
