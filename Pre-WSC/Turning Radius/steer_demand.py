#!/usr/bin/env python3
"""
steer_demand.py - how much steering does the route actually demand?

Purpose: size the wheel-fairing opening. If the wheel can turn up to some
angle without opening the fairing, what fraction of the route (by distance)
stays inside that limit?

Method:
  * every set of 3 consecutive route points -> circumscribed-circle radius
    (Menger curvature k = 1/R, computed in local metres)
  * each triplet is weighted by the road length it represents
    ((d_prev + d_next) / 2), so results are "% of route distance", not just
    "% of points" (points are dense in turns, which would over-count them)
  * steering angle from a bicycle/Ackermann model: delta = atan(wheelbase * k)

Outputs (in --outdir):
  steer_histogram.png    triplet count per radius tier | % route distance per
                         radius tier | coverage curve (% route within X deg)
  steer_tiers.csv        tier table (count, km, %, cumulative %)
  steer_coverage.csv     required steering angle for 50 .. 99.9 % coverage
  steer_per_file.csv     per-file summary

Usage (PowerShell, quote names with spaces):
  python steer_demand.py "*.kml.save" --wheelbase 2.6
  python steer_demand.py "..\\..\\Fallback Model\\Saves\\*.kml.save" --skip-loops --skip-days 3
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import turn_radius as tr   # reuse the loader / projection (same folder)

# Radius tiers in metres. Last tier is "effectively straight".
RADIUS_EDGES = [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, np.inf]
COVERAGE_LEVELS = [50, 75, 90, 95, 97.5, 99, 99.5, 99.9]
MAX_R_CAP = 1e6   # collinear points -> treat as straight, avoids inf


def radius_label(lo, hi):
    if np.isinf(hi):
        return f"{int(lo)}+ (straight)"
    return f"{int(lo)}-{int(hi)}"


def triplet_curvature(x, y):
    """Menger curvature for each interior point i (points i-1, i, i+1)."""
    ax, ay = x[:-2], y[:-2]
    bx, by = x[1:-1], y[1:-1]
    cx, cy = x[2:], y[2:]
    ab = np.hypot(bx - ax, by - ay)
    bc = np.hypot(cx - bx, cy - by)
    ac = np.hypot(cx - ax, cy - ay)
    cross = np.abs((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    denom = ab * bc * ac
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(denom > 0, 2.0 * cross / denom, 0.0)
    return k


def thin_points(lat, lon, dist, min_seg):
    """Optionally drop points closer than min_seg metres to the last kept one
    (digitising jitter at tiny spacing blows up 3-point curvature)."""
    if min_seg <= 0:
        return lat, lon, dist
    keep = [0]
    for i in range(1, len(dist)):
        if dist[i] - dist[keep[-1]] >= min_seg:
            keep.append(i)
    if keep[-1] != len(dist) - 1:
        keep.append(len(dist) - 1)
    keep = np.array(keep)
    return lat[keep], lon[keep], dist[keep]


def analyze_file(path, min_seg):
    lat, lon, dist = tr.load_route(path)
    lat, lon, dist = tr.drop_duplicates(lat, lon, dist)
    lat, lon, dist = thin_points(lat, lon, dist, min_seg)
    x, y = tr.project_xy(lat, lon)
    k = triplet_curvature(x, y)
    d = np.diff(dist)
    w = 0.5 * (d[:-1] + d[1:])             # metres of road each triplet represents
    return dict(k=k, w=w, length_m=float(dist[-1]))


def day_of(name):
    m = re.search(r"Day\s*(\d+)(?!\d)", name, re.I)
    return int(m.group(1)) if m else None


def collect_files(patterns, skip_loops, skip_days):
    files, skipped = [], []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits:
            raise SystemExit(f"No file matches {pat!r}. Quote names with spaces and "
                             f"run from the right folder.")
        for fp in hits:
            name = os.path.basename(fp)
            if skip_loops and "loop" in name.lower():
                skipped.append((name, "loop"))
            elif day_of(name) in skip_days:
                skipped.append((name, f"day {day_of(name)}"))
            else:
                files.append(fp)
    return sorted(set(files)), skipped


def weighted_quantile_curve(delta_deg, w):
    order = np.argsort(delta_deg)
    d_sorted, w_sorted = delta_deg[order], w[order]
    cum = np.cumsum(w_sorted) / w_sorted.sum() * 100.0
    return d_sorted, cum


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="file(s) or wildcards (.kml.save / .kml)")
    ap.add_argument("--wheelbase", type=float, default=2.6,
                    help="wheelbase in metres for radius->steer angle (default 2.6 is an ASSUMPTION; set the real value)")
    ap.add_argument("--min-seg", type=float, default=0.0,
                    help="drop points closer than this many metres (default 0 = use every consecutive triplet)")
    ap.add_argument("--skip-loops", action="store_true", help="ignore files with 'loop' in the name")
    ap.add_argument("--skip-days", type=int, nargs="*", default=[], help="day numbers to ignore, e.g. 3")
    ap.add_argument("--outdir", default="steer_results")
    args = ap.parse_args()

    files, skipped = collect_files(args.files, args.skip_loops, set(args.skip_days))
    if not files:
        raise SystemExit("Nothing left to analyse after filters.")
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Analysing {len(files)} files, skipping {len(skipped)}")
    for n, why in skipped:
        print(f"    skip ({why}): {n}")
    print(f"Wheelbase used for steer angle: {args.wheelbase} m"
          f"{'  (default assumption!)' if args.wheelbase == 2.6 else ''}")

    all_k, all_w, rows = [], [], []
    for fp in files:
        r = analyze_file(fp, args.min_seg)
        all_k.append(r["k"]); all_w.append(r["w"])
        rows.append((os.path.basename(fp), r))
    k = np.concatenate(all_k)
    w = np.concatenate(all_w)
    total_km = w.sum() / 1000.0

    # ---- radius tiers -------------------------------------------------
    with np.errstate(divide="ignore"):
        R = np.where(k > 0, np.minimum(1.0 / k, MAX_R_CAP), MAX_R_CAP)
    edges = RADIUS_EDGES
    labels = [radius_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    idx = np.clip(np.digitize(R, edges) - 1, 0, len(labels) - 1)
    counts = np.bincount(idx, minlength=len(labels))
    km_tier = np.bincount(idx, weights=w, minlength=len(labels)) / 1000.0
    pct_tier = km_tier / total_km * 100.0
    # cumulative from the straight end: "% of route with R >= this tier's lower edge"
    cum_gentler = np.cumsum(pct_tier[::-1])[::-1]

    print("\n" + "=" * 78)
    print(f"{len(k)} triplets over {total_km:.1f} km")
    print(f"{'R tier (m)':>18} {'triplets':>9} {'km':>9} {'% route':>8} {'% route with R >= lower edge':>30}")
    for lab, c, km, p, cg in zip(labels, counts, km_tier, pct_tier, cum_gentler):
        print(f"{lab:>18} {c:>9} {km:>9.1f} {p:>7.2f}% {cg:>29.2f}%")

    # ---- steering angle coverage --------------------------------------
    delta = np.degrees(np.arctan(args.wheelbase * k))      # bicycle-model steer angle
    d_sorted, cum = weighted_quantile_curve(delta, w)
    print("\nSteering needed to cover X% of the route distance "
          f"(wheelbase {args.wheelbase} m):")
    print(f"{'coverage':>10} {'steer deg':>10} {'R (m)':>10}")
    cov_rows = []
    for lvl in COVERAGE_LEVELS:
        i = min(np.searchsorted(cum, lvl), len(d_sorted) - 1)
        dv = float(d_sorted[i])
        rv = args.wheelbase / np.tan(np.radians(dv)) if dv > 1e-9 else float("inf")
        cov_rows.append((lvl, dv, rv))
        print(f"{lvl:>9}% {dv:>10.2f} {rv:>10.0f}")

    # ---- CSVs ---------------------------------------------------------
    with open(os.path.join(args.outdir, "steer_tiers.csv"), "w") as f:
        f.write("radius_tier_m,triplets,km,pct_route,pct_route_with_R_ge_lower_edge\n")
        for lab, c, km_, p, cg in zip(labels, counts, km_tier, pct_tier, cum_gentler):
            f.write(f"{lab},{c},{km_:.3f},{p:.3f},{cg:.3f}\n")
    with open(os.path.join(args.outdir, "steer_coverage.csv"), "w") as f:
        f.write(f"coverage_pct,steer_deg_wheelbase_{args.wheelbase}m,radius_m\n")
        for lvl, dv, rv in cov_rows:
            f.write(f"{lvl},{dv:.3f},{rv:.1f}\n")
    with open(os.path.join(args.outdir, "steer_per_file.csv"), "w") as f:
        f.write("file,km,triplets,pct_within_1deg,pct_within_2deg,pct_within_5deg,pct_within_10deg\n")
        for name, r in rows:
            dl = np.degrees(np.arctan(args.wheelbase * r["k"]))
            ww = r["w"]
            pcts = [ww[dl <= a].sum() / ww.sum() * 100 for a in (1, 2, 5, 10)]
            f.write(f"{name},{r['length_m']/1000:.2f},{len(r['k'])},"
                    + ",".join(f"{p:.2f}" for p in pcts) + "\n")

    # ---- plots --------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    xs = np.arange(len(labels))
    axs[0].bar(xs, counts, color="#1976d2", edgecolor="white")
    axs[0].set_title(f"Triplets per radius tier (n={len(k)})")
    axs[0].set_ylabel("Number of 3-point sets")
    axs[1].bar(xs, pct_tier, color="#2e7d32", edgecolor="white")
    axs[1].set_title(f"% of route distance per radius tier ({total_km:.0f} km)")
    axs[1].set_ylabel("% of route distance")
    for ax, vals in ((axs[0], counts), (axs[1], pct_tier)):
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_xlabel("Radius tier (m)")
        ax.grid(axis="y", alpha=0.3)
        for xi, v in zip(xs, vals):
            if v:
                ax.text(xi, v, f"{v:.1f}" if vals is pct_tier else f"{int(v)}",
                        ha="center", va="bottom", fontsize=8)
    axs[2].plot(d_sorted, cum, color="#d32f2f", lw=2)
    for lvl, dv, _ in cov_rows:
        if lvl in (90, 95, 99, 99.9):
            axs[2].axhline(lvl, color="grey", ls=":", lw=0.8)
            axs[2].plot([dv], [lvl], "ko", ms=4)
            axs[2].annotate(f"{lvl}% -> {dv:.1f} deg", (dv, lvl),
                            textcoords="offset points", xytext=(6, -12), fontsize=8)
    axs[2].set_xscale("symlog", linthresh=1.0)
    axs[2].set_xlabel(f"Wheel steer angle (deg, wheelbase {args.wheelbase} m)")
    axs[2].set_ylabel("% of route distance within this angle")
    axs[2].set_title("Coverage: steering limit vs route covered")
    axs[2].set_ylim(0, 101); axs[2].grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(args.outdir, "steer_histogram.png")
    fig.savefig(png, dpi=150); plt.close(fig)
    print(f"\nSaved to {os.path.abspath(args.outdir)}: steer_histogram.png, steer_tiers.csv, "
          f"steer_coverage.csv, steer_per_file.csv")


if __name__ == "__main__":
    main()