"""Per day: velocity (top) and route altitude (bottom) against clock time.

Position along the route comes from integrating logged speed (v*dt; gaps > GAP_S add nothing), starting at 0
(Days 5 and 8 start at the end of Stage 1). Altitude is the route-file altitude at that position, using the FINAL route
files in Dashboard/Saves: Day 2 loop = "SSC ROUTE FINAL_Day 2 Half Blind_Day 2 Loop", Day 3 = the "Day3_" files.
Loops driven per day (team): Days 1, 3, 5 two each, others none. Day 2 has no loop driven, so its final loop file is unused.
Colours/style are those of ../plotter.py (velocity orange, race window dashed, outside-window shading).
"""
import json, os, sys
import numpy as np
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import plotter as pl
from plotter import COLORS, plt, RACE_DAYS, SAST, GAP_S, MAIN_MARKER, REPO_ROOT

SAVES = os.path.join(str(REPO_ROOT), 'Dashboard', 'Saves')
R = '2026 Sasol Solar Challenge Route (Publish)_'
ROUTES = {  # stage 1, loop, stage 2
    1: (R + 'Day 1 _10 Sept Stage 1 Boiketlong to Rustenburg', R + 'Day 1 _Rustenburg Loop', R + 'Day 1 _10 Sept Stage 2 Rustenburg to Swartruggens'),
    2: (R + 'Day 2 Half Blind_11 Sept Stage 1 Swart Ruggens to Zeerust', 'SSC ROUTE FINAL_Day 2 Half Blind_Day 2 Loop', R + 'Day 2 Half Blind_11 Sept Stage 2 Zeerust to Vryburg'),
    3: ('Day3_Stage 1_Stage 1', 'Day3_Loop_Jan Kempdorp Loop', 'Day3_Stage 2_Stage 2'),
    5: (R + 'Day 5 _14 Sept Stage 1 Olifantshoek to Upington', R + 'Day 5 _Upington Loop', R + 'Day 5 _14 Sept Stage 2 Upington to Augrabies'),
    6: (R + 'Day 6 _15 Sept Stage 1 Augrabies to Springbok', R + 'Day 6 _Springbok Loop', None),
    7: (R + 'Day 7_16 Sept Stage 1 Springbok to Van Rhynsdorp', R + 'Day 7_Van Rhynsdorp Loop', R + 'Day 7_16 Sept Stage 2 Van Rhynsdorp to Clanwilliam'),
    8: (R + 'Day 8_17 Sept Stage 1 Clanwilliam to Ceres', R + 'Day 8_Ceres Loop', R + 'Day 8_17 Sept Stage 2 Ceres to Paarl'),
}
START_AFTER_STAGE1 = {5, 8}
DRIVEN_LOOPS = {1: 2, 2: 0, 3: 2, 5: 2, 6: 0, 7: 0, 8: 0}   # loops actually driven (from the team)
GIVEN_KM = {1: 214.6, 2: 302.5, 3: 255.0, 5: 182.0, 6: 57.0, 7: 100.5, 8: 80.0}


def prof(name):
    p = json.load(open(os.path.join(SAVES, name + '.kml.save'), encoding='utf-8'))['profile']
    return np.array(p['Distance']) * 1000.0, np.array(p['Altitude'])


def route(day):
    s1, lp, s2 = (prof(n) if n else None for n in ROUTES[day])
    L1, LL, L2 = s1[0][-1], lp[0][-1], (s2[0][-1] if s2 else 0.0)
    skipped = L1 if day in START_AFTER_STAGE1 else 0.0
    n = DRIVEN_LOOPS[day]
    parts = [('Stage 1', s1)] + [(f'Loop {k + 1}', lp) for k in range(n)] + ([('Stage 2', s2)] if s2 else [])
    D, A, bounds, off = [], [], [], 0.0
    for lab, (d, a) in parts:
        D.append(d + off); A.append(a); bounds.append((lab, off, off + d[-1])); off += d[-1]
    D, A = np.concatenate(D), np.concatenate(A)
    keep = np.r_[True, np.diff(D) > 0]
    print(f'Day {day}: stage 1 {L1 / 1000:.1f} km, loop {LL / 1000:.1f} km, stage 2 {L2 / 1000:.1f} km -> {n} loop(s), route {off / 1000:.1f} km, '
          f'driven {GIVEN_KM[day]:.1f} km{" (starts after stage 1)" if skipped else ""}')
    return D[keep], A[keep], bounds, skipped, n


def draw(day):
    date, h0, h1 = RACE_DAYS[day]
    recs = [r for r in pl.load_day(day) if MAIN_MARKER in r and 'Vehicle_Velocity' in r]
    t = np.array([r['t'].timestamp() for r in recs]); v = np.abs(np.array([r['Vehicle_Velocity'] for r in recs]))
    o = np.argsort(t); t, v = t[o], v[o]
    D, A, bounds, start, n = route(day)
    dt = np.r_[0, np.diff(t)]
    pos = start + np.cumsum(np.where(dt <= GAP_S, np.r_[0, 0.5 * (v[1:] + v[:-1])][:len(t)] * dt, 0.0))
    alt = np.interp(pos, D, A)
    T = np.array([np.datetime64(int(x * 1e6), 'us') for x in t])
    from datetime import datetime
    TT = [datetime.fromtimestamp(x, SAST) for x in t]
    brk = np.r_[False, dt[1:] > GAP_S]            # break the lines at telemetry gaps
    vv, aa = (v * 3.6).copy(), alt.copy(); vv[brk] = np.nan; aa[brk] = np.nan

    fig, ax = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, height_ratios=[1, 1])
    ax[0].plot(TT, vv, color=COLORS['orange'], linewidth=1.0); ax[0].set_ylabel('Vehicle velocity (km/h)')
    ax[1].plot(TT, aa, color=COLORS['mppt_A'], linewidth=1.2); ax[1].set_ylabel('Route altitude at v·dt position (m)')
    w0 = datetime.fromisoformat(f'{date}T{h0:02d}:00:00+02:00'); w1 = datetime.fromisoformat(f'{date}T{h1:02d}:00:00+02:00')
    x0, x1 = min(w0, TT[0]), max(w1, TT[-1])
    for a in ax:
        a.set_xlim(x0, x1)
        for lo, hi in ((x0, w0), (w1, x1)):
            if lo < hi:
                a.axvspan(lo, hi, color=COLORS['outside_window'], zorder=0)
        for e in (w0, w1):
            a.axvline(e, color=COLORS['orange_light'], linestyle='--', linewidth=0.8, alpha=0.7)
        for s in a.spines.values():
            s.set_visible(False)
    # time at which the integrated position crosses each route boundary
    for lab, lo, hi in bounds[:-1]:
        k = np.searchsorted(pos, hi)
        if k < len(t) and hi > start:
            for a in ax:
                a.axvline(TT[k], color=COLORS['muted'], linestyle=':', linewidth=0.9)
            ax[1].text(TT[k], ax[1].get_ylim()[1], f' end {lab}', color=COLORS['muted'], fontsize=8, va='top')
    ax[1].xaxis.set_major_locator(mdates.HourLocator(tz=SAST)); ax[1].xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45], tz=SAST))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=SAST)); ax[1].set_xlabel('Time (SAST)')
    ax[0].set_title(f'Day {day} ({date}): velocity and route altitude vs time (position from v·dt, {n} loop(s), final route files)',
                    loc='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    pl.PLOT_DIR.mkdir(exist_ok=True)
    out = str(pl.PLOT_DIR / f'day{day}_altitude_velocity.png'); fig.savefig(out, dpi=150); plt.close(fig)
    print(f'   v·dt travelled {(pos[-1] - start) / 1000:.1f} km vs driven {GIVEN_KM[day]:.1f} km (route {(bounds[-1][2] - start) / 1000:.1f} km) -> {out}')


for d in ROUTES:
    draw(d)
