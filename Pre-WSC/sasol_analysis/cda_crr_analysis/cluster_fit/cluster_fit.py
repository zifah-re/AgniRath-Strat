"""Cluster averaging: Crr / CdA from speed-binned mean motor power, Days 1-3, gradient assumed to average to 0.

Samples: moving > 20 km/h, motor drawing power (no regen), no telemetry gap > 10 s around the sample.
Bins: 5 km/h, time-weighted mean power, kept if >= 3 min of driving.
Fit:  P_bin = (Crr*m*g*v + 0.5*rho*CdA*v^3)/eta + P_aux, for P_aux = 20..100 W (slider), weighted by bin time.
"""
import json, os
from datetime import datetime
import numpy as np
from scipy.optimize import lsq_linear
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, '..', '..', 'cleaned_logs')
DAYS = [1, 2, 3]
M, G, ETA = 336.5, 9.81, 0.95
RHO = 1.07                        # ~1200-1400 m altitude, 22 C
BINS = np.arange(20, 95, 5)
P_AUX = np.arange(20, 101, 10)
STEADY = True                     # False = plain cluster averaging of every driving sample
BG, SURF, INK, INK2, GRID = '#141413', '#1F1E1D', '#F0EEE6', '#A8A69C', '#3A3936'
ORANGE, BLUE = '#D97757', '#3987E5'
DAYCOL = {1: '#3987E5', 2: '#C98500', 3: '#199E70'}
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'grid.color': GRID, 'axes.grid': True,
                     'grid.alpha': 0.6, 'font.size': 10, 'legend.facecolor': SURF, 'legend.edgecolor': GRID})


def load(day):
    t, v, p = [], [], []
    with open(os.path.join(LOGS, f'day{day}.jsonl'), encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if 'Vehicle_Velocity' in r and 'Bus_Current' in r:
                t.append(datetime.fromisoformat(r['_rx_time']).timestamp()); v.append(abs(r['Vehicle_Velocity']))
                p.append(r['Bus_Voltage'] * r['Bus_Current'])
    o = np.argsort(t)
    return np.array(t)[o], np.array(v)[o], np.array(p)[o]


samples = []
for d in DAYS:
    t, v, p = load(d)
    dt = np.r_[np.diff(t), 0]
    # steady speed: change over +-10 s below 1.5 km/h (drops launches and braking)
    vm = np.interp(t - 10, t, v); vp = np.interp(t + 10, t, v)
    steady = np.abs(vp - vm) * 3.6 < 1.5
    ok = (v * 3.6 > 20) & (p > 0) & (dt > 0) & (dt <= 10) & (steady if STEADY else True)
    samples.append(np.c_[np.full(ok.sum(), d), v[ok] * 3.6, p[ok], dt[ok]])
S = np.vstack(samples)


def bins(S):
    out = []
    for lo in BINS[:-1]:
        s = (S[:, 1] >= lo) & (S[:, 1] < lo + 5)
        w = S[s, 3]
        if w.sum() < 180:
            continue
        vk = S[s, 1] / 3.6
        out.append((np.sum(w * S[s, 1]) / w.sum(), np.sum(w * vk) / w.sum(), np.sum(w * vk ** 3) / w.sum(),
                    np.sum(w * S[s, 2]) / w.sum(), w.sum() / 60))
    return np.array(out)


def fit(B, paux, bounded=True):
    sw = np.sqrt(B[:, 4])
    X = np.c_[M * G * B[:, 1] / ETA, 0.5 * RHO * B[:, 2] / ETA] * sw[:, None]
    y = (B[:, 3] - paux) * sw
    if bounded:
        return lsq_linear(X, y, bounds=([0.003, 0.06], [0.02, 0.25])).x
    return np.linalg.lstsq(X, y, rcond=None)[0]


B = bins(S)
print(f'{len(S)} samples, {S[:, 3].sum() / 3600:.1f} h of driving on days {DAYS}')
print('bins (km/h, mean power W, minutes):')
for b in B:
    print(f'  {b[0]:5.1f}  {b[3]:6.0f} W  {b[4]:5.0f} min')
print('\nP_aux   Crr      CdA     (bounded 0.003-0.02 / 0.06-0.25)   |  unbounded Crr  CdA')
res = {}
for pa in P_AUX:
    c = fit(B, pa); u = fit(B, pa, False)
    res[int(pa)] = dict(Crr=float(c[0]), CdA=float(c[1]), Crr_unbounded=float(u[0]), CdA_unbounded=float(u[1]))
    print(f'{pa:4d} W  {c[0]:.4f}  {c[1]:.3f}                                  |  {u[0]:.4f}  {u[1]:.3f}')
for d in DAYS:
    Bd = bins(S[S[:, 0] == d]); c = fit(Bd, 60); u = fit(Bd, 60, False)
    print(f'Day {d} alone (P_aux 60 W): bounded Crr {c[0]:.4f} CdA {c[1]:.3f} | unbounded Crr {u[0]:.4f} CdA {u[1]:.3f} | bins {Bd[0, 0]:.0f}-{Bd[-1, 0]:.0f} km/h')
json.dump(dict(bins=B.tolist(), fits=res, rho=RHO, mass=M, eta=ETA), open(os.path.join(HERE, 'cluster_fit.json'), 'w'), indent=1)

fig, ax = plt.subplots(figsize=(10, 5.8))
for d in DAYS:
    Bd = bins(S[S[:, 0] == d])
    ax.scatter(Bd[:, 0], Bd[:, 3], s=8 + Bd[:, 4], color=DAYCOL[d], alpha=0.75, label=f'Day {d} bins (size = minutes)')
ax.scatter(B[:, 0], B[:, 3], s=60, color=ORANGE, edgecolors=INK, zorder=3, label='Days 1-3 combined')
vv = np.linspace(20, 90, 80)
curve = lambda c, pa: (c[0] * M * G * vv / 3.6 + 0.5 * RHO * c[1] * (vv / 3.6) ** 3) / ETA + pa
c60 = fit(B, 60)
ax.plot(vv, curve(c60, 60), color=ORANGE, lw=2.4, label=f'fit, P_aux 60 W: Crr {c60[0]:.4f}, CdA {c60[1]:.3f}')
ax.plot(vv, curve([0.007, 0.16], 60), color=INK2, ls='--', label='team model (0.007, 0.16)')
ax.set_xlabel('speed [km/h]'); ax.set_ylabel('mean motor bus power [W]')
ax.set_title('Cluster averaging, Sasol Days 1-3 (gradient assumed to average out)')
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'cluster_fit.png'), dpi=140)
