"""
turn_radius.py - find the sharpest turns on a route and histogram their radii.

Handles the dashboard's *.kml.save / *.save JSON files (profile.Coordinates +
profile.Distance in km). Also accepts a raw .kml with a LineString.

Method (density-aware, as discussed):
  1. Project lat/lon to local metres; segment lengths come from the file's
     pre-computed Distance column (km -> m).
  2. A segment is "dense" if it is much shorter than its local median spacing
     AND shorter than an absolute cap. Dense runs (bridging 1-point gaps) are
     the candidate turns, because the digitizer places vertices closely where
     the road bends.
  3. Each cluster is padded by a few points (entry/exit of the turn) and split
     where the turn direction reverses (S-bends).
  4. Radius = arc length / |total heading change|  (robust), cross-checked by a
     least-squares circle fit (Kasa).
  5. Clusters with too little heading change, or R above a cap, are rejected
     as false positives. The top-N of what survives is reported.

Usage:
  python turn_radius.py Day1_WSC_Day_1_kml.save
  python turn_radius.py file.save --top 10 --rank sharpest      (default)
  python turn_radius.py file.save --top 10 --rank largest
  python turn_radius.py a.save b.save c.kml --outdir results
"""
import argparse
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_EARTH = 6371008.8

# ---------------------------------------------------------------- tunables --
DENSE_ABS_M = 60.0        # segment must be shorter than this to be "dense"
DENSE_REL = 0.6           # ... and shorter than this x local median spacing
MEDIAN_HALF_WIN = 25      # local median taken over +/- this many segments
BRIDGE_GAP = 1            # merge dense runs separated by <= this many segments
PAD_PTS = 2               # points added each side of a cluster
SPLIT_MIN_DEG = 3.0       # turn-direction flips only count above this angle
MIN_HEADING_CHANGE_DEG = 25.0
MAX_RADIUS_M = 500.0      # beyond this a "turn" is effectively straight
MIN_ARC_M = 8.0
# Radius tiers for the histogram (metres). Fine at the tight end where it
# matters for speed limits, coarse for gentle curves.
TIER_EDGES = [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]


# ----------------------------------------------------------------- loading --
def load_route(path):
    """Return (lat, lon, dist_m) arrays. dist_m is cumulative metres."""
    if path.lower().endswith(".kml"):
        return _load_kml(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prof = data["profile"]
    coords = np.asarray(prof["Coordinates"], dtype=float)
    dist_m = np.asarray(prof["Distance"], dtype=float) * 1000.0  # km -> m
    n = min(len(coords), len(dist_m))
    return coords[:n, 0], coords[:n, 1], dist_m[:n]


def _load_kml(path):
    root = ET.parse(path).getroot()
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    txt = root.find(".//LineString/coordinates").text.split()
    pts = np.array([[float(v) for v in t.split(",")[:2]] for t in txt])
    lat, lon = pts[:, 1], pts[:, 0]
    return lat, lon, _haversine_cumulative(lat, lon)


def _haversine_cumulative(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    a = (np.sin(np.diff(la) / 2) ** 2
         + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2)
    seg = 2 * R_EARTH * np.arcsin(np.sqrt(a))
    return np.concatenate([[0.0], np.cumsum(seg)])


def project_xy(lat, lon):
    """Local equirectangular projection, segment-wise (stays accurate over
    hundreds of km because cos(lat) is re-evaluated per segment)."""
    la, lo = np.radians(lat), np.radians(lon)
    mid = 0.5 * (la[:-1] + la[1:])
    dx = R_EARTH * np.cos(mid) * np.diff(lo)
    dy = R_EARTH * np.diff(la)
    return np.concatenate([[0.0], np.cumsum(dx)]), np.concatenate([[0.0], np.cumsum(dy)])


def drop_duplicates(lat, lon, dist_m, min_step=0.05):
    keep = np.concatenate([[True], np.diff(dist_m) > min_step])
    return lat[keep], lon[keep], dist_m[keep]


# --------------------------------------------------------------- geometry ---
def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def segment_bearings(x, y):
    return np.degrees(np.arctan2(np.diff(x), np.diff(y)))  # clockwise from north


def rolling_median(v, half):
    out = np.empty_like(v)
    for i in range(len(v)):
        out[i] = np.median(v[max(0, i - half): i + half + 1])
    return out


def kasa_circle(x, y):
    """Algebraic least-squares circle fit. Returns (R, rms_residual) or (nan, nan)."""
    if len(x) < 4:
        return np.nan, np.nan
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    A = np.column_stack([u, v, np.ones_like(u)])
    b = u ** 2 + v ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    uc, vc = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + uc ** 2 + vc ** 2
    if r2 <= 0:
        return np.nan, np.nan
    R = np.sqrt(r2)
    rms = np.sqrt(np.mean((np.hypot(u - uc, v - vc) - R) ** 2))
    return R, rms


# --------------------------------------------------------------- clustering --
def dense_clusters(seg_len):
    """Return list of (first_seg, last_seg) index pairs of dense runs."""
    local_med = rolling_median(seg_len, MEDIAN_HALF_WIN)
    dense = (seg_len < DENSE_ABS_M) & (seg_len < DENSE_REL * local_med)
    runs, i, n = [], 0, len(dense)
    while i < n:
        if dense[i]:
            j = i
            while j + 1 < n and dense[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    # bridge small gaps
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= BRIDGE_GAP:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def split_on_reversal(turn_deg, lo, hi):
    """Split vertex range [lo, hi] (inclusive, indices into turn_deg) where the
    turn direction flips (S-bends). Returns list of (lo, hi)."""
    parts, start, last_sign = [], lo, 0
    for k in range(lo, hi + 1):
        if abs(turn_deg[k]) < SPLIT_MIN_DEG:
            continue
        s = int(np.sign(turn_deg[k]))
        if last_sign and s != last_sign:
            parts.append((start, k - 1))
            start = k
        last_sign = s
    parts.append((start, hi))
    return parts


# ---------------------------------------------------------------- analysis --
def analyze(path):
    lat, lon, dist = load_route(path)
    lat, lon, dist = drop_duplicates(lat, lon, dist)
    x, y = project_xy(lat, lon)
    seg_len = np.diff(dist)
    brg = segment_bearings(x, y)
    # turn_deg[k] = heading change at vertex k+1 (between segment k and k+1)
    turn_deg = wrap_deg(np.diff(brg))
    n_pts = len(lat)

    turns = []
    for s0, s1 in dense_clusters(seg_len):
        # vertices strictly inside the dense run: s0+1 .. s1  -> turn idx s0 .. s1-1
        # pad: widen by PAD_PTS segments each side
        a = max(0, s0 - PAD_PTS)
        b = min(len(turn_deg) - 1, s1 - 1 + PAD_PTS)
        if b < a:
            continue
        for lo, hi in split_on_reversal(turn_deg, a, b):
            p_lo, p_hi = lo, hi + 2          # point indices spanned by turns lo..hi
            p_hi = min(p_hi, n_pts - 1)
            total = float(np.sum(turn_deg[lo:hi + 1]))
            arc = float(dist[p_hi] - dist[p_lo])
            if abs(total) < MIN_HEADING_CHANGE_DEG or arc < MIN_ARC_M:
                continue
            r_arc = arc / np.radians(abs(total))
            if r_arc > MAX_RADIUS_M:
                continue
            r_fit, rms = kasa_circle(x[p_lo:p_hi + 1], y[p_lo:p_hi + 1])
            k_apex = lo + int(np.argmax(np.abs(turn_deg[lo:hi + 1])))
            turns.append(dict(
                km=float(dist[k_apex + 1] / 1000.0),
                lat=float(lat[k_apex + 1]), lon=float(lon[k_apex + 1]),
                radius_m=float(r_arc),
                fit_radius_m=float(r_fit) if np.isfinite(r_fit) else None,
                fit_rms_m=float(rms) if np.isfinite(rms) else None,
                heading_change_deg=abs(total),
                direction="right" if total > 0 else "left",
                arc_m=arc, n_points=p_hi - p_lo + 1,
                p_lo=p_lo, p_hi=p_hi,
            ))
    turns = dedupe_overlaps(turns)
    return dict(lat=lat, lon=lon, dist=dist, x=x, y=y, turns=turns)


def dedupe_overlaps(turns):
    """Padding can make neighbouring clusters claim the same points (same
    physical turn counted twice). Where two turns share points, keep the
    sharper (smaller radius) one."""
    kept = []
    for t in sorted(turns, key=lambda t: t["p_lo"]):
        if kept and t["p_lo"] <= kept[-1]["p_hi"]:
            if t["radius_m"] < kept[-1]["radius_m"]:
                kept[-1] = t
        else:
            kept.append(t)
    return kept


# ------------------------------------------------------------------ output --
def tier_labels(edges):
    return [f"{edges[i]}-{edges[i+1]} m" for i in range(len(edges) - 1)]


def histogram_counts(radii):
    counts, _ = np.histogram(radii, bins=TIER_EDGES)
    return counts


def plot_histogram(radii, title, out_png):
    counts = histogram_counts(radii)
    labels = tier_labels(TIER_EDGES)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts, color="#1976d2", edgecolor="white")
    for b, c in zip(bars, counts):
        if c:
            ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom")
    ax.set_xlabel("Turn radius tier")
    ax.set_ylabel("Number of turns")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_route(res, top, out_png, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(res["x"] / 1000, res["y"] / 1000, color="#999", lw=1)
    for rank, t in enumerate(top, 1):
        sl = slice(t["p_lo"], t["p_hi"] + 1)
        ax.plot(res["x"][sl] / 1000, res["y"][sl] / 1000, color="red", lw=3)
        ax.annotate(str(rank), (res["x"][t["p_lo"]] / 1000, res["y"][t["p_lo"]] / 1000),
                    color="red", fontsize=9, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def report(path, top_n, rank, outdir):
    res = analyze(path)
    turns = res["turns"]
    stem = re.sub(r"[^\w\-]+", "_", os.path.splitext(os.path.basename(path))[0])
    os.makedirs(outdir, exist_ok=True)

    ordered = sorted(turns, key=lambda t: t["radius_m"], reverse=(rank == "largest"))
    top = ordered[:top_n]

    print("=" * 78)
    print(f"{os.path.basename(path)}: {res['dist'][-1]/1000:.1f} km, "
          f"{len(res['lat'])} points, {len(turns)} valid turns detected")
    print(f"Top {len(top)} turns ({'smallest' if rank == 'sharpest' else 'largest'} radius first)")
    print("-" * 78)
    print(f"{'#':>2} {'km':>8} {'R (m)':>8} {'fit R':>8} {'rms':>6} "
          f"{'turn deg':>8} {'dir':>5} {'pts':>4}   lat, lon")
    for i, t in enumerate(top, 1):
        fr = f"{t['fit_radius_m']:.1f}" if t["fit_radius_m"] else "-"
        rm = f"{t['fit_rms_m']:.1f}" if t["fit_rms_m"] is not None else "-"
        print(f"{i:>2} {t['km']:>8.2f} {t['radius_m']:>8.1f} {fr:>8} {rm:>6} "
              f"{t['heading_change_deg']:>8.0f} {t['direction']:>5} {t['n_points']:>4}   "
              f"{t['lat']:.5f}, {t['lon']:.5f}")

    radii = np.array([t["radius_m"] for t in turns])
    counts = histogram_counts(radii) if len(radii) else np.zeros(len(TIER_EDGES) - 1, int)
    print("\nTurns per radius tier (all valid turns, not just the top list):")
    for lab, c in zip(tier_labels(TIER_EDGES), counts):
        print(f"  {lab:>10}: {c:>4} {'#' * min(int(c), 60)}")

    # files
    csv_path = os.path.join(outdir, f"{stem}_top_turns.csv")
    with open(csv_path, "w") as f:
        f.write("rank,km,radius_m,fit_radius_m,fit_rms_m,heading_change_deg,direction,n_points,lat,lon\n")
        for i, t in enumerate(top, 1):
            f.write(f"{i},{t['km']:.3f},{t['radius_m']:.2f},{t['fit_radius_m'] or ''},"
                    f"{t['fit_rms_m'] if t['fit_rms_m'] is not None else ''},"
                    f"{t['heading_change_deg']:.1f},{t['direction']},{t['n_points']},"
                    f"{t['lat']:.6f},{t['lon']:.6f}\n")
    hist_png = os.path.join(outdir, f"{stem}_radius_histogram.png")
    plot_histogram(radii, f"{os.path.basename(path)} - turns per radius tier "
                          f"(n={len(radii)})", hist_png)
    route_png = os.path.join(outdir, f"{stem}_route_top_turns.png")
    plot_route(res, top, route_png, f"{os.path.basename(path)} - top {len(top)} turns")
    print(f"\nSaved: {csv_path}\n       {hist_png}\n       {route_png}")
    return top, counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--rank", choices=["sharpest", "largest"], default="sharpest",
                    help="sharpest = smallest radius first (default); largest = biggest radius first")
    ap.add_argument("--outdir", default="turn_results")
    args = ap.parse_args()
    for fp in args.files:
        report(fp, args.top, args.rank, args.outdir)