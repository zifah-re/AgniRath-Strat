"""Cluster averaging on ONE continuous stretch of the race (no stops, no telemetry gaps > 30 s).

Same method as cluster_fit.py: steady-speed samples (speed change < 1.5 km/h over +-10 s), motor drawing power,
5 km/h speed bins (>= 2 min each), time-weighted fit of P = (Crr*m*g*v + 0.5*rho*CdA*v^3)/eta + P_aux.
Gradient assumed to average out over the stretch.
"""
import json, os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, '..', '..', 'cleaned_logs')
SEGMENTS = [(5, '18:35', '20:38'), (2, '18:26', '19:04')]
M, G, ETA = 336.5, 9.81, 0.95
RHO = {5: 1.08, 2: 1.07}
P_AUX = [20, 40, 60, 80, 100]
BG, SURF, INK, INK2, GRID = '#141413', '#1F1E1D', '#F0EEE6', '#A8A69C', '#3A3936'
ORANGE, BLUE, GOLD = '#D97757', '#3987E5', '#C98500'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'grid.color': GRID, 'axes.grid': True,
                     'grid.alpha': 0.6, 'font.size': 10, 'legend.facecolor': SURF, 'legend.edgecolor': GRID})


def load(day, t0, t1):
    t, v, p = [], [], []
    with open(os.path.join(LOGS, f'day{day}.jsonl'), encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if 'Vehicle_Velocity' in r and 'Bus_Current' in r:
                t.append(datetime.fromisoformat(r['_rx_time']).timestamp()); v.append(abs(r['Vehicle_Velocity']))
                p.append(r['Bus_Voltage'] * r['Bus_Current'])
    o = np.argsort(t); t, v, p = np.array(t)[o], np.array(v)[o], np.array(p)[o]
    d0 = datetime.fromtimestamp(t[0]).replace(hour=0, minute=0, second=0).timestamp()
    sec = lambda s: d0 + int(s[:2]) * 3600 + int(s[3:]) * 60
    k = (t >= sec(t0)) & (t <= sec(t1) + 59)
    return t[k], v[k], p[k]


fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(13, 5.2))
for ax, (day, t0, t1) in zip(axes, SEGMENTS):
    t, v, p = load(day, t0, t1)
    dt = np.r_[np.diff(t), 0]
    vm, vp = np.interp(t - 10, t, v), np.interp(t + 10, t, v)
    ok = (np.abs(vp - vm) * 3.6 < 1.5) & (v * 3.6 > 20) & (p > 0) & (dt > 0) & (dt <= 30)
    kmh, w, pw, vs = v[ok] * 3.6, dt[ok], p[ok], v[ok]
    B = []
    for lo in range(20, 95, 5):
        s = (kmh >= lo) & (kmh < lo + 5)
        if w[s].sum() >= 120:
            B.append((np.sum(w[s] * kmh[s]) / w[s].sum(), np.sum(w[s] * vs[s]) / w[s].sum(),
                      np.sum(w[s] * vs[s] ** 3) / w[s].sum(), np.sum(w[s] * pw[s]) / w[s].sum(), w[s].sum() / 60))
    B = np.array(B)
    print(f'\nDay {day} {t0}-{t1}: {ok.sum()} steady samples, {w.sum() / 60:.0f} min')
    for b in B:
        print(f'  {b[0]:5.1f} km/h  {b[3]:6.0f} W  {b[4]:4.0f} min')
    sw = np.sqrt(B[:, 4])
    X = np.c_[M * G * B[:, 1] / ETA, 0.5 * RHO[day] * B[:, 2] / ETA] * sw[:, None]
    fits = {}
    for pa in P_AUX:
        c = np.linalg.lstsq(X, (B[:, 3] - pa) * sw, rcond=None)[0]; fits[pa] = c
        print(f'  P_aux {pa:3d} W -> Crr {c[0]:.4f}  CdA {c[1]:.3f}')
    ax.scatter(kmh, pw, s=3, color=BLUE, alpha=0.25, label='steady samples')
    ax.scatter(B[:, 0], B[:, 3], s=30 + 3 * B[:, 4], color=ORANGE, edgecolors=INK, zorder=3, label='bin means (size = minutes)')
    vv = np.linspace(max(20, B[0, 0] - 10), B[-1, 0] + 10, 60); c = fits[60]
    ax.plot(vv, (c[0] * M * G * vv / 3.6 + 0.5 * RHO[day] * c[1] * (vv / 3.6) ** 3) / ETA + 60, color=ORANGE, lw=2.2,
            label=f'fit (P_aux 60 W): Crr {c[0]:.4f}, CdA {c[1]:.3f}')
    ax.plot(vv, (0.007 * M * G * vv / 3.6 + 0.5 * RHO[day] * 0.16 * (vv / 3.6) ** 3) / ETA + 60, color=INK2, ls='--', label='team model (0.007, 0.16)')
    ax.set_ylim(0, 3000); ax.set_xlabel('speed [km/h]'); ax.set_ylabel('motor bus power [W]')
    ax.set_title(f'Day {day}, {t0}-{t1} (one continuous stretch)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'segment_fit.png'), dpi=140)
