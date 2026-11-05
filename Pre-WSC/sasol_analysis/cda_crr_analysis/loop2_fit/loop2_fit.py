"""Day 3 Loop 2 (the one loop with telemetry, 17:22-17:51): cluster averaging + spline on P vs v.

Closed loop -> net height change 0, wind assumed 0.
1) Cluster: driving samples (v > 20 km/h, P > 0), 5 km/h bins (>= 1 min), time-weighted mean P, fit
   P = (Crr*m*g*v + 0.5*rho*CdA*v^3)/eta + P_aux for P_aux 20..100 W. Also with a steady-speed filter.
2) Spline: smoothing spline of P(v) through the bin means, then fit the physics curve to the spline.
3) Whole-loop energy balance (one equation): E_bus = sum[(Crr*m*g*v + 0.5*rho*CdA*v^3)/eta + P_aux]dt + dKE.
"""
import json, os
from datetime import datetime
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, '..', '..', 'cleaned_logs', 'day3.jsonl')
T0, T1 = '17:22', '17:51'
M, G, ETA, RHO = 336.5, 9.81, 0.95, 1.08
P_AUX = [20, 40, 60, 80, 100]
BG, SURF, INK, INK2, GRID = '#141413', '#1F1E1D', '#F0EEE6', '#A8A69C', '#3A3936'
ORANGE, BLUE, GREEN = '#D97757', '#3987E5', '#199E70'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'grid.color': GRID, 'axes.grid': True,
                     'grid.alpha': 0.6, 'font.size': 10, 'legend.facecolor': SURF, 'legend.edgecolor': GRID})

t, v, p = [], [], []
with open(LOG, encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if 'Vehicle_Velocity' in r and 'Bus_Current' in r:
            t.append(datetime.fromisoformat(r['_rx_time']).timestamp()); v.append(abs(r['Vehicle_Velocity']))
            p.append(r['Bus_Voltage'] * r['Bus_Current'])
o = np.argsort(t); t, v, p = np.array(t)[o], np.array(v)[o], np.array(p)[o]
d0 = datetime.fromtimestamp(t[0]).replace(hour=0, minute=0, second=0).timestamp()
sec = lambda s: d0 + int(s[:2]) * 3600 + int(s[3:]) * 60
k = (t >= sec(T0)) & (t <= sec(T1) + 59); t, v, p = t[k], v[k], p[k]
dt = np.r_[np.diff(t), 0]; dt[dt > 30] = 0
print(f'loop window {T0}-{T1}: {len(t)} records, {np.sum(v * dt) / 1000:.1f} km, max gap {np.diff(t).max():.0f} s, '
      f'moving {np.sum(dt[v > 0.5]) / 60:.1f} min, mean moving speed {np.sum(v * dt) / np.sum(dt[v > 0.5]) * 3.6:.1f} km/h')

vm, vp = np.interp(t - 10, t, v), np.interp(t + 10, t, v)
steady = np.abs(vp - vm) * 3.6 < 1.5
base = (v * 3.6 > 20) & (p > 0) & (dt > 0)


def bins(ok, minsec=60):
    kmh = v * 3.6; out = []
    for lo in range(20, 95, 5):
        s = ok & (kmh >= lo) & (kmh < lo + 5); w = dt[s]
        if w.sum() >= minsec:
            out.append((np.sum(w * kmh[s]) / w.sum(), np.sum(w * v[s]) / w.sum(), np.sum(w * v[s] ** 3) / w.sum(),
                        np.sum(w * p[s]) / w.sum(), w.sum() / 60))
    return np.array(out)


def fit(vs, v3s, P, w, pa):
    X = np.c_[M * G * vs / ETA, 0.5 * RHO * v3s / ETA] * np.sqrt(w)[:, None]
    return np.linalg.lstsq(X, (P - pa) * np.sqrt(w), rcond=None)[0]


res = {}
for name, ok in [('all driving', base), ('steady only', base & steady)]:
    B = bins(ok)
    print(f'\n{name}: {ok.sum()} samples, {dt[ok].sum() / 60:.1f} min, {len(B)} bins')
    for b in B:
        print(f'  {b[0]:5.1f} km/h  {b[3]:6.0f} W  {b[4]:4.1f} min')
    if len(B) < 2:
        print('  fewer than 2 bins -> cannot split Crr/CdA'); continue
    res[name] = {'bins': B.tolist()}
    for pa in P_AUX:
        c = fit(B[:, 1], B[:, 2], B[:, 3], B[:, 4], pa); res[name][pa] = c.tolist()
        print(f'  P_aux {pa:3d} W -> Crr {c[0]:.4f}  CdA {c[1]:.3f}')
    if len(B) >= 4:
        sp = UnivariateSpline(B[:, 0], B[:, 3], w=np.sqrt(B[:, 4]), k=min(3, len(B) - 1))
        vv = np.linspace(B[0, 0], B[-1, 0], 40)
        for pa in P_AUX:
            c = fit(vv / 3.6, (vv / 3.6) ** 3, sp(vv), np.ones_like(vv), pa)
            res[name][f'spline_{pa}'] = c.tolist()
            print(f'  spline, P_aux {pa:3d} W -> Crr {c[0]:.4f}  CdA {c[1]:.3f}')

# whole-loop energy balance: report what CdA each Crr implies (one equation, two unknowns)
mv = (v > 0.5) & (dt > 0)
E_bus = np.sum(np.where(p > 0, p, p) * dt)              # net bus energy incl. regen
dKE = 0.5 * M * (v[-1] ** 2 - v[0] ** 2)
Sv, Sv3, T = np.sum(v * dt), np.sum(v ** 3 * dt), np.sum(dt)
print(f'\nloop energy: bus {E_bus / 3600:.0f} Wh over {T / 60:.0f} min ({Sv / 1000:.1f} km), avg {E_bus / T:.0f} W, {E_bus / 3600 / (Sv / 1000):.1f} Wh/km')
print('one equation -> Crr/CdA trade-off (P_aux 60 W):')
for crr in [0.006, 0.008, 0.010, 0.012, 0.015]:
    cda = ((E_bus - dKE) * ETA - ETA * 60 * T - crr * M * G * Sv) / (0.5 * RHO * Sv3)
    print(f'  Crr {crr:.3f} -> CdA {cda:.3f}')
json.dump(res, open(os.path.join(HERE, 'loop2_fit.json'), 'w'), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
T_ = [datetime.fromtimestamp(x) for x in t]
ax[0].plot(T_, v * 3.6, color=ORANGE, lw=1); ax[0].set_ylabel('speed [km/h]', color=ORANGE)
a2 = ax[0].twinx(); a2.plot(T_, p, color=BLUE, lw=0.6, alpha=0.7); a2.set_ylabel('bus power [W]', color=BLUE)
ax[0].set_title(f'Day 3 Loop 2, {T0}-{T1}')
import matplotlib.dates as mdates; ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax[1].scatter(v[base] * 3.6, p[base], s=4, color=BLUE, alpha=0.3, label='driving samples')
if 'all driving' in res:
    B = np.array(res['all driving']['bins']); c = res['all driving'][60]
    ax[1].scatter(B[:, 0], B[:, 3], s=30 + 10 * B[:, 4], color=ORANGE, edgecolors=INK, zorder=3, label='bin means')
    vv = np.linspace(20, 90, 60)
    ax[1].plot(vv, (c[0] * M * G * vv / 3.6 + 0.5 * RHO * c[1] * (vv / 3.6) ** 3) / ETA + 60, color=ORANGE, lw=2,
               label=f'fit P_aux 60: Crr {c[0]:.4f}, CdA {c[1]:.3f}')
    ax[1].plot(vv, (0.007 * M * G * vv / 3.6 + 0.5 * RHO * 0.16 * (vv / 3.6) ** 3) / ETA + 60, color=INK2, ls='--', label='team (0.007, 0.16)')
ax[1].set_ylim(0, 4000); ax[1].set_xlabel('speed [km/h]'); ax[1].set_ylabel('bus power [W]'); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'loop2_fit.png'), dpi=140)
