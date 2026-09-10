"""
day1_gradient_analysis.py

Extracts a gradient profile for the STITCHED Day 1 route (Stage 1 -> Loop ->
Stage 2) using the exact same calculation Dashboard/main.py's
`render_selected_track` endpoint uses to build the live "Gradient" profile.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DAY1_FILES = {
    "Stage 1": "2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 1 Boiketlong to Rustenburg.kml.save",
    "Loop":    "2026 Sasol Solar Challenge Route (Publish)_Day 1 _Rustenburg Loop.kml.save",
    "Stage 2": "2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 2 Rustenburg to Swartruggens.kml.save",
}

# Savitzky-Golay params -- IDENTICAL to Dashboard/main.py's render_selected_track.
SG_WINDOW = 15
SG_POLYORDER = 3
SG_DELTA_M = 100.0        # matches main.py's `delta=100`
GRID_SPACING_M = 100.0    # main.py builds target_points as (total_km*1000)//100

# --- Sustained climb / descent detection ---
MIN_ABS_GRADIENT_PCT = 0.5   # gradient magnitude below this counts as "flat"
MIN_RUN_LENGTH_M = 500.0     # region minimum span distance
SIGN_SMOOTH_WINDOW = 5       # median-filters labels to drop noise
MERGE_GAP_M = 150.0          # merge same-direction regions separated by gaps < this


# ---------------------------------------------------------------------------
# Loading + stitching
# ---------------------------------------------------------------------------

def load_stage(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    profile = data["profile"]
    dist_km = np.asarray(profile["Distance"], dtype=float)
    alt_m = np.asarray(profile["Altitude"], dtype=float)
    n = min(len(dist_km), len(alt_m))
    return {
        "distance_km": dist_km[:n],
        "altitude_m": alt_m[:n],
        "label": data.get("placemark_name", path.stem),
    }


def stitch_day(saves_dir: Path, files: dict[str, str]) -> dict:
    cum_offset_m = 0.0
    dist_chunks, alt_chunks, boundaries = [], [], []

    for label, fname in files.items():
        path = saves_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find '{fname}' in {saves_dir}. "
                f"Pass --saves-dir to point at the folder containing .kml.save files."
            )
        stage = load_stage(path)
        dist_m = stage["distance_km"] * 1000.0 + cum_offset_m
        dist_chunks.append(dist_m)
        alt_chunks.append(stage["altitude_m"])
        boundaries.append({"label": label, "start_m": cum_offset_m, "end_m": float(dist_m[-1])})
        cum_offset_m = float(dist_m[-1])

    distance_m = np.concatenate(dist_chunks)
    altitude_m = np.concatenate(alt_chunks)

    distance_m, unique_idx = np.unique(distance_m, return_index=True)
    altitude_m = altitude_m[unique_idx]

    return {"distance_m": distance_m, "altitude_m": altitude_m, "boundaries": boundaries}


# ---------------------------------------------------------------------------
# Gradient calculation
# ---------------------------------------------------------------------------

def compute_gradient(distance_m: np.ndarray, altitude_m: np.ndarray):
    total_m = float(distance_m[-1])
    n_target = int(total_m // GRID_SPACING_M)
    n_target = max(n_target, SG_WINDOW + 1)
    target_grid_m = np.linspace(0.0, total_m, n_target)

    interp_alt_m = np.interp(target_grid_m, distance_m, altitude_m)
    slope = savgol_filter(
        interp_alt_m, window_length=SG_WINDOW, polyorder=SG_POLYORDER,
        deriv=1, delta=SG_DELTA_M,
    )
    gradient_pct_grid = slope * 100.0
    gradient_pct_full = np.interp(distance_m, target_grid_m, gradient_pct_grid)

    return target_grid_m, gradient_pct_grid, gradient_pct_full


# ---------------------------------------------------------------------------
# Sustained region detection
# ---------------------------------------------------------------------------

@dataclass
class Region:
    kind: str          # "climb" or "descent"
    start_m: float
    end_m: float
    length_m: float
    avg_grad_pct: float
    max_abs_grad_pct: float
    elevation_change_m: float


def _median_filter_labels(labels: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or window % 2 == 0:
        return labels.copy()
    half = window // 2
    out = labels.copy()
    n = len(labels)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window_vals = labels[lo:hi]
        vals, counts = np.unique(window_vals, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def find_sustained_regions(distance_m: np.ndarray, gradient_pct: np.ndarray,
                           altitude_m: np.ndarray | None = None) -> list[Region]:
    raw_labels = np.where(gradient_pct > MIN_ABS_GRADIENT_PCT, 1,
                          np.where(gradient_pct < -MIN_ABS_GRADIENT_PCT, -1, 0))
    labels = _median_filter_labels(raw_labels, SIGN_SMOOTH_WINDOW)

    alt_for_dist = altitude_m if altitude_m is not None and len(altitude_m) == len(distance_m) else None

    raw_regions: list[Region] = []
    n = len(labels)
    i = 0
    while i < n:
        lab = labels[i]
        j = i
        while j + 1 < n and labels[j + 1] == lab:
            j += 1
        if lab != 0:
            start_m, end_m = float(distance_m[i]), float(distance_m[j])
            seg = gradient_pct[i:j + 1]
            length_m = end_m - start_m
            elev_change = (float(alt_for_dist[j] - alt_for_dist[i])
                           if alt_for_dist is not None else float("nan"))
            raw_regions.append(Region(
                kind="climb" if lab == 1 else "descent",
                start_m=start_m, end_m=end_m, length_m=length_m,
                avg_grad_pct=float(np.mean(seg)),
                max_abs_grad_pct=float(np.max(np.abs(seg))),
                elevation_change_m=elev_change,
            ))
        i = j + 1

    merged: list[Region] = []
    for reg in raw_regions:
        if merged and merged[-1].kind == reg.kind and (reg.start_m - merged[-1].end_m) <= MERGE_GAP_M:
            prev = merged[-1]
            new_len = reg.end_m - prev.start_m
            w_prev = prev.length_m / new_len if new_len > 0 else 0.5
            w_reg = 1.0 - w_prev
            merged[-1] = Region(
                kind=prev.kind,
                start_m=prev.start_m,
                end_m=reg.end_m,
                length_m=new_len,
                avg_grad_pct=prev.avg_grad_pct * w_prev + reg.avg_grad_pct * w_reg,
                max_abs_grad_pct=max(prev.max_abs_grad_pct, reg.max_abs_grad_pct),
                elevation_change_m=prev.elevation_change_m + reg.elevation_change_m,
            )
        else:
            merged.append(reg)

    return [r for r in merged if r.length_m >= MIN_RUN_LENGTH_M]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_analysis(distance_m: np.ndarray, altitude_m: np.ndarray,
                  gradient_pct: np.ndarray, regions: list[Region],
                  boundaries: list[dict], out_path: Path) -> None:
    dist_km = distance_m / 1000.0
    fig, (ax_alt, ax_grad) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    CLIMB_COLOR = "#d62728"
    DESCENT_COLOR = "#1f77b4"

    # --- altitude panel ---
    ax_alt.plot(dist_km, altitude_m, color="black", linewidth=1.2, label="Altitude")
    for r in regions:
        color = CLIMB_COLOR if r.kind == "climb" else DESCENT_COLOR
        ax_alt.axvspan(r.start_m / 1000.0, r.end_m / 1000.0, color=color, alpha=0.18)
    for b in boundaries:
        ax_alt.axvline(b["start_m"] / 1000.0, color="gray", linestyle=":", linewidth=0.8)
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title("Day 1 Stitched Route: Altitude & Gradient", fontweight="bold")
    ax_alt.grid(True, linestyle="--", alpha=0.4)

    ymax = ax_alt.get_ylim()[1]
    for b in boundaries:
        mid_km = (b["start_m"] + b["end_m"]) / 2.0 / 1000.0
        ax_alt.text(mid_km, ymax, b["label"], ha="center", va="bottom", fontsize=9, color="dimgray")

    # --- gradient panel ---
    ax_grad.plot(dist_km, gradient_pct, color="darkorange", linewidth=0.9, label="Gradient (%)")
    ax_grad.axhline(0, color="black", linewidth=0.8)
    ax_grad.axhline(MIN_ABS_GRADIENT_PCT, color="gray", linestyle="--", linewidth=0.7)
    ax_grad.axhline(-MIN_ABS_GRADIENT_PCT, color="gray", linestyle="--", linewidth=0.7)
    for r in regions:
        color = CLIMB_COLOR if r.kind == "climb" else DESCENT_COLOR
        ax_grad.axvspan(r.start_m / 1000.0, r.end_m / 1000.0, color=color, alpha=0.18)
    ax_grad.set_ylabel("Gradient (%)")
    ax_grad.set_xlabel("Cumulative Distance (km)")
    ax_grad.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_last_60km(distance_m: np.ndarray, altitude_m: np.ndarray,
                   gradient_pct: np.ndarray, regions: list[Region],
                   out_path: Path) -> None:
    total_km = distance_m[-1] / 1000.0
    start_km = max(0.0, total_km - 60.0)
    breakdown_km = total_km - 47.0

    mask = distance_m / 1000.0 >= start_km
    sub_dist_km = distance_m[mask] / 1000.0
    sub_alt = altitude_m[mask]
    sub_grad = gradient_pct[mask]

    fig, (ax_alt, ax_grad) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    CLIMB_COLOR = "#d62728"
    DESCENT_COLOR = "#1f77b4"

    # --- Altitude Plot ---
    ax_alt.plot(sub_dist_km, sub_alt, color="black", linewidth=1.5, label="Altitude")
    for r in regions:
        r_start_km, r_end_km = r.start_m / 1000.0, r.end_m / 1000.0
        if r_end_km >= start_km:
            color = CLIMB_COLOR if r.kind == "climb" else DESCENT_COLOR
            ax_alt.axvspan(r_start_km, r_end_km, color=color, alpha=0.2)

    # Point label for breakdown at 47 km from end
    breakdown_alt = np.interp(breakdown_km, sub_dist_km, sub_alt)
    ax_alt.plot(breakdown_km, breakdown_alt, "ro", markersize=8, zorder=5)
    ax_alt.annotate(
        f"Breakdown Point\n({breakdown_km:.2f} km / 47 km to end)",
        xy=(breakdown_km, breakdown_alt),
        xytext=(breakdown_km - 8, breakdown_alt + (np.ptp(sub_alt) * 0.15)),
        arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=8),
        fontweight="bold",
        color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8),
    )

    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title("Zoomed Profile: Final 60 km of Race", fontweight="bold")
    ax_alt.grid(True, linestyle="--", alpha=0.5)

    # --- Gradient Plot ---
    ax_grad.plot(sub_dist_km, sub_grad, color="darkorange", linewidth=1.2)
    ax_grad.axhline(0, color="black", linewidth=0.8)
    for r in regions:
        r_start_km, r_end_km = r.start_m / 1000.0, r.end_m / 1000.0
        if r_end_km >= start_km:
            color = CLIMB_COLOR if r.kind == "climb" else DESCENT_COLOR
            ax_grad.axvspan(r_start_km, r_end_km, color=color, alpha=0.2)

    breakdown_grad = np.interp(breakdown_km, sub_dist_km, sub_grad)
    ax_grad.plot(breakdown_km, breakdown_grad, "ro", markersize=6, zorder=5)

    ax_grad.set_ylabel("Gradient (%)")
    ax_grad.set_xlabel("Cumulative Distance (km)")
    ax_grad.set_xlim(start_km, total_km)
    ax_grad.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_regions_csv(regions: list[Region], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "start_km", "end_km", "length_m",
                         "avg_gradient_pct", "max_abs_gradient_pct", "elevation_change_m"])
        for r in sorted(regions, key=lambda x: x.start_m):
            writer.writerow([
                r.kind, round(r.start_m / 1000.0, 3), round(r.end_m / 1000.0, 3),
                round(r.length_m, 1), round(r.avg_grad_pct, 2),
                round(r.max_abs_grad_pct, 2), round(r.elevation_change_m, 1),
            ])


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_20km_summary(distance_m: np.ndarray, altitude_m: np.ndarray, regions: list[Region]) -> None:
    total_km = distance_m[-1] / 1000.0
    
    # --- Print Highest Uphill Region ---
    climbs = [r for r in regions if r.kind == "climb"]
    if climbs:
        highest_uphill = max(climbs, key=lambda x: x.elevation_change_m)
        print("\n==================================================")
        print("REGION OF HIGHEST UPHILL (MAX ELEVATION GAIN)")
        print("==================================================")
        print(f"  Distance Span  : km {highest_uphill.start_m/1000:.2f} -> km {highest_uphill.end_m/1000:.2f}")
        print(f"  Length         : {highest_uphill.length_m:.0f} m")
        print(f"  Elevation Gain : +{highest_uphill.elevation_change_m:.1f} m")
        print(f"  Avg Gradient   : {highest_uphill.avg_grad_pct:.2f}%")
        print(f"  Max Gradient   : {highest_uphill.max_abs_grad_pct:.2f}%")

    # --- Print 20 km Interval Summary ---
    print("\n==================================================")
    print("NET ALTITUDE CHANGE EVERY 20 KM")
    print("==================================================")
    print(f"{'Interval (km)':<18}{'Start Alt (m)':>15}{'End Alt (m)':>15}{'Net ΔAlt (m)':>15}")
    print("-" * 63)

    interval_km = 20.0
    curr_km = 0.0
    while curr_km < total_km:
        next_km = min(curr_km + interval_km, total_km)
        alt_start = np.interp(curr_km * 1000.0, distance_m, altitude_m)
        alt_end = np.interp(next_km * 1000.0, distance_m, altitude_m)
        net_change = alt_end - alt_start
        
        label = f"{curr_km:.0f} - {next_km:.1f}"
        print(f"{label:<18}{alt_start:>15.1f}{alt_end:>15.1f}{net_change:>15.1f}")
        curr_km += interval_km
    print("==================================================\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saves-dir", type=Path, default=Path("Dashboard/Saves"),
                        help="Folder containing the Day 1 .kml.save files")
    parser.add_argument("--out-dir", type=Path, default=Path("."),
                        help="Where to write output files")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Day 1 stages from {args.saves_dir} ...")
    stitched = stitch_day(args.saves_dir, DAY1_FILES)
    distance_m, altitude_m = stitched["distance_m"], stitched["altitude_m"]

    print("Computing gradient (Savitzky-Golay, 100 m grid) ...")
    _, _, gradient_pct = compute_gradient(distance_m, altitude_m)

    print("Detecting sustained climb/descent regions ...")
    regions = find_sustained_regions(distance_m, gradient_pct, altitude_m)

    # Cleaned up summary print output
    print_20km_summary(distance_m, altitude_m, regions)

    plot_path = args.out_dir / "day1_gradient_profile.png"
    last60_plot_path = args.out_dir / "day1_last60km_profile.png"
    csv_path = args.out_dir / "day1_gradient_regions.csv"

    plot_analysis(distance_m, altitude_m, gradient_pct, regions, stitched["boundaries"], plot_path)
    plot_last_60km(distance_m, altitude_m, gradient_pct, regions, last60_plot_path)
    write_regions_csv(regions, csv_path)

    print(f"Saved full plot -> {plot_path}")
    print(f"Saved last 60km plot -> {last60_plot_path}")
    print(f"Saved region table -> {csv_path}")


if __name__ == "__main__":
    main()