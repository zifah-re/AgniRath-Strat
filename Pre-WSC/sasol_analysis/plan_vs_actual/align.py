"""
Route alignment (map matching without GPS).

The logger has no GPS / altitude / accelerometer data (all logged as 0), so route position comes from
integrating speed. Gaps, wheel-speed scale error and trailering make that position drift, and the
route gradient read at a drifted position is wrong. In cruise the bus power follows m g v sin(theta)
closely (a 3 % grade at 60 km/h is ~1.6 kW), so the grade pattern itself can be used to re-align:

  * the telemetry is cut into 1 km pieces (by integrated distance);
  * for every piece and every offset d in [-OFF, +OFF] km, score = correlation between the speed-detrended
    bus power and v * grade(x + d) over the piece and its neighbours (cruise packets only);
  * a Viterbi pass picks the offset path that maximises total score minus a penalty on offset changes;
    the penalty is waived where the logger was silent for > 5 min (position genuinely unknown there).

Returns aligned position per packet; align_report() writes align_report.csv (correlation before/after).
"""
import numpy as np
import pandas as pd

OFF_KM = 8.0
STEP_KM = 0.1
PIECE_KM = 1.0
JUMP_PEN = 0.05      # score penalty per 0.1 km of offset change between neighbouring pieces
MIN_PKTS = 20


def _detrend(v, p):
    if len(v) < 10:
        return p - p.mean()
    c = np.polyfit(v, p, 2)
    return p - np.polyval(c, v)


def align(x_km, v, p, cruise, gap_before, dist_km, grad_pct, loop_len=None):
    """x_km, v (m/s), p (W), cruise (bool), gap_before (bool: long silent gap before this packet).
    Returns aligned x (km)."""
    offs = np.arange(-OFF_KM, OFF_KM + 1e-9, STEP_KM)
    L = float(dist_km[-1])
    piece = np.floor(x_km / PIECE_KM).astype(int)
    pieces = np.unique(piece)
    sig = np.full(len(p), np.nan)
    sig[cruise] = _detrend(v[cruise], p[cruise])
    score = np.zeros((len(pieces), len(offs)))
    for j, d in enumerate(offs):
        xx = x_km + d
        xx = np.mod(xx, L) if loop_len else np.clip(xx, 0, L)
        gv = np.interp(xx, dist_km, grad_pct) / 100.0 * v
        for i, k in enumerate(pieces):
            m = cruise & (piece >= k - 1) & (piece <= k + 1)
            if m.sum() < MIN_PKTS:
                continue
            a, b = sig[m], gv[m]
            if a.std() < 1e-6 or b.std() < 1e-6:
                continue
            score[i, j] = np.corrcoef(a, b)[0, 1]
    # piece starts after a long gap -> free jump
    free = np.array([gap_before[piece == k].any() for k in pieces])
    pen = JUMP_PEN * np.abs(offs[:, None] - offs[None, :]) / STEP_KM
    acc = score[0].copy()
    back = np.zeros((len(pieces), len(offs)), int)
    for i in range(1, len(pieces)):
        tot = acc[:, None] - (0.0 * pen if free[i] else pen)    # [prev, cur]
        back[i] = np.argmax(tot, axis=0)
        acc = tot[back[i], np.arange(len(offs))] + score[i]
    path = np.zeros(len(pieces), int)
    path[-1] = int(np.argmax(acc))
    for i in range(len(pieces) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    d_piece = dict(zip(pieces, offs[path]))
    xa = x_km + np.array([d_piece[k] for k in piece])
    return np.mod(xa, L) if loop_len else np.clip(xa, 0, L)


def corr_with_grade(x_km, v, p, cruise, dist_km, grad_pct):
    sig = _detrend(v[cruise], p[cruise])
    gv = np.interp(x_km[cruise], dist_km, grad_pct) / 100.0 * v[cruise]
    return float(np.corrcoef(sig, gv)[0, 1])
