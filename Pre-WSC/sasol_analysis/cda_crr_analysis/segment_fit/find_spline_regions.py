"""Scan every day for stretches where a P(v) spline is possible.

A stretch = continuous driving (no telemetry gap > 30 s, no stop > 60 s), at least 20 min long.
Per stretch: steady motor-on samples (speed change < 1.5 km/h over +-10 s, v > 20 km/h, P > 0),
5 km/h bins with >= 1 min each. A spline needs >= 4 bins; we also want the bin means to rise with speed.
"""
import json, os
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, '..', '..', 'cleaned_logs')

for day in range(1, 9):
    fn = os.path.join(LOGS, f'day{day}.jsonl')
    if not os.path.exists(fn):
        continue
    t, v, p = [], [], []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if 'Vehicle_Velocity' in r and 'Bus_Current' in r:
                t.append(datetime.fromisoformat(r['_rx_time']).timestamp()); v.append(abs(r['Vehicle_Velocity']))
                p.append(r['Bus_Voltage'] * r['Bus_Current'])
    if not t:
        continue
    o = np.argsort(t); t, v, p = np.array(t)[o], np.array(v)[o], np.array(p)[o]
    dt = np.r_[np.diff(t), 0]
    # break points: gaps > 30 s or stops > 60 s
    brk = set(np.where(dt > 30)[0])
    i = 0
    while i < len(t):
        if v[i] < 0.5:
            j = i
            while j + 1 < len(t) and v[j + 1] < 0.5:
                j += 1
            if t[j] - t[i] > 60:
                brk.add(i)
            i = j + 1
        else:
            i += 1
    cuts = [-1] + sorted(brk) + [len(t) - 1]
    vm, vp = np.interp(t - 10, t, v), np.interp(t + 10, t, v)
    steady = (np.abs(vp - vm) * 3.6 < 1.5) & (v * 3.6 > 20) & (p > 0) & (dt > 0) & (dt <= 30)
    for a, b in zip(cuts[:-1], cuts[1:]):
        s = slice(a + 1, b + 1)
        T = t[b] - t[a + 1]
        if T < 1200:
            continue
        kmh, w, pw, st = v[s] * 3.6, dt[s], p[s], steady[s]
        bins = []
        for lo in range(20, 95, 5):
            k = st & (kmh >= lo) & (kmh < lo + 5)
            if w[k].sum() >= 60:
                bins.append((lo + 2.5, np.sum(w[k] * pw[k]) / w[k].sum(), w[k].sum() / 60))
        mono = all(b2[1] > b1[1] for b1, b2 in zip(bins, bins[1:])) if len(bins) > 1 else False
        tag = 'SPLINE OK' if len(bins) >= 4 else ''
        print(f'day {day} {datetime.fromtimestamp(t[a + 1]):%H:%M}-{datetime.fromtimestamp(t[b]):%H:%M} '
              f'{T / 60:5.0f} min {np.sum(v[s] * w) / 1000:5.1f} km steady {w[st].sum() / 60:4.0f} min, '
              f'{len(bins)} bins {[int(x[0]) for x in bins]} rising={mono} {tag}')
        if len(bins) >= 4:
            print('      ' + '  '.join(f'{x[0]:.0f}:{x[1]:.0f}W/{x[2]:.0f}m' for x in bins))
