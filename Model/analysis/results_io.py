"""
analysis/results_io.py — per-stage/loop CSV output and folder structure.

This module bridges the optimizer output (one velocity profile per day) to
the dashboard's input (one CSV per physical stage or loop). It:

  1. Records segment boundaries during route loading (SegmentBoundary),
  2. Splits a day-level solve output into per-stage/loop chunks,
  3. Calls build_run_csv() on each chunk with a sliced sub-Route,
  4. Saves everything to data/results/{variant}/day{N}/{slug}.csv,
  5. Writes a summary.json per variant tree.

Usage from trust_region.py __main__:

    from analysis.results_io import (
        SegmentBoundary, record_segment_boundary,
        split_and_save_day, write_summary_json,
    )

Architecture notes (13 Aug 2026):
  - Each CSV = one build_run_csv() call scoped to a stage boundary.
  - Day 3 has multiple route variants; each gets its own results subtree.
  - Days 1-2, 4-8 are shared across variants but are written into each
    variant's folder for self-containment (no symlinks — simpler).
  - The dashboard can load a single CSV, a day folder, or a full tree.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from configs.car_config import CarState
from core.route import Route

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Segment boundary tracking
# ===========================================================================

@dataclass
class SegmentBoundary:
    """Records where one .save file's data lives in the concatenated day route."""
    filename: str          # original .save basename
    slug: str              # cleaned name for the CSV filename
    seg_type: str          # 'stage1', 'loop', 'stage2'
    start_m: float         # cumulative distance at segment start
    end_m: float           # cumulative distance at segment end
    source_path: str       # full path to the .save file


def _slugify(filename: str) -> str:
    """Convert a .save filename to a clean CSV-friendly slug.

    'Day 1 _10 Sept Stage 1 Boiketlong to Rustenburg.kml.save'
    -> 'stage1_boiketlong_to_rustenburg'
    """
    name = filename.lower()
    # Remove prefix junk
    name = re.sub(r'^2026 sasol solar challenge route \(publish\)_', '', name)
    name = re.sub(r'^day \d+ probables_probable \w+ (?:route_|day \d+_)?', '', name)
    name = re.sub(r'^day \d+[_ ]*(half blind_)?', '', name)
    # Remove date patterns like '10 sept ', '13 sept '
    name = re.sub(r'\d+ sept ', '', name)
    # Remove file extensions
    name = re.sub(r'\.kml\.save$', '', name)
    name = re.sub(r'\.kml$', '', name)
    # Normalize spaces and special chars
    name = re.sub(r'[^a-z0-9]+', '_', name).strip('_')
    # Collapse repeated underscores
    name = re.sub(r'_+', '_', name)
    return name


def classify_segment(filename: str) -> str:
    """Determine seg_type from the filename."""
    lower = filename.lower()
    if 'loop' in lower:
        return 'loop'
    if 'stage 2' in lower or 'stage2' in lower:
        return 'stage2'
    return 'stage1'


def record_segment_boundary(
    filepath: str, start_m: float, end_m: float
) -> SegmentBoundary:
    """Create a SegmentBoundary from a loaded .save file."""
    basename = os.path.basename(filepath)
    return SegmentBoundary(
        filename=basename,
        slug=_slugify(basename),
        seg_type=classify_segment(basename),
        start_m=start_m,
        end_m=end_m,
        source_path=filepath,
    )


# ===========================================================================
# 2. Splitting a day's solve output into per-stage chunks
# ===========================================================================

def _slice_solve_output(
    solve_output: dict,
    boundary: SegmentBoundary,
) -> dict | None:
    """Extract the portion of a solve_output that falls within a boundary.

    Returns a new dict with the same keys as solve_output but sliced arrays,
    or None if the boundary has no overlap with the velocity profile.
    """
    seg_start_m = np.asarray(solve_output["seg_start_m"], dtype=float)
    v_kmh = np.asarray(solve_output["v_kmh"], dtype=float)

    # Find segments whose start falls within [boundary.start_m, boundary.end_m)
    mask = (seg_start_m >= boundary.start_m - 0.5) & (seg_start_m < boundary.end_m - 0.5)
    if not np.any(mask):
        return None

    indices = np.where(mask)[0]
    return dict(
        v_kmh=v_kmh[indices],
        seg_start_m=seg_start_m[indices] - boundary.start_m,  # re-zero
    )


def _make_sub_route(full_route: Route, boundary: SegmentBoundary) -> Route:
    """Slice the full day route to just the rows in this segment's range."""
    df = full_route.df
    dist = df["distance_m"].to_numpy()

    mask = (dist >= boundary.start_m - 0.5) & (dist <= boundary.end_m + 0.5)
    sub_df = df.loc[mask].copy()
    sub_df["distance_m"] = sub_df["distance_m"] - boundary.start_m
    sub_df = sub_df.reset_index(drop=True)

    if len(sub_df) == 0:
        logger.warning(f"Empty sub-route for {boundary.slug}")
        return None

    return Route(sub_df)


# ===========================================================================
# 3. Save per-stage CSVs for one day
# ===========================================================================

def split_and_save_day(
    solve_output: dict,
    full_route: Route,
    car: CarState,
    solar_provider,
    wind_provider,
    day_index: int,
    start_soc_pct: float,
    boundaries: list[SegmentBoundary],
    out_dir: str | pathlib.Path,
) -> list[dict]:
    """Split one day's solve output into per-stage CSVs and save them.

    Returns a list of dicts with metadata about each saved CSV:
        [{"slug": ..., "seg_type": ..., "csv_path": ..., "start_soc": ..., "end_soc": ...}, ...]
    """
    from analysis.offline_dashboard import build_run_csv

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for boundary in boundaries:
        chunk = _slice_solve_output(solve_output, boundary)
        if chunk is None:
            logger.warning(f"  No velocity data for segment {boundary.slug}, skipping")
            continue

        sub_route = _make_sub_route(full_route, boundary)
        if sub_route is None:
            continue

        csv_name = f"{boundary.slug}.csv"
        csv_path = out_dir / csv_name

        # Estimate start SOC for this chunk by running through prior segments
        # For the first segment, use the day's start SOC.
        # For subsequent segments, use the end SOC of the previous chunk.
        chunk_start_soc = start_soc_pct
        if saved:
            chunk_start_soc = saved[-1].get("end_soc", start_soc_pct)

        try:
            df = build_run_csv(
                solve_output=chunk,
                route=sub_route,
                car=car,
                solar_provider=solar_provider,
                wind_provider=wind_provider,
                day_index=day_index,
                start_soc_pct=chunk_start_soc,
                out_path=str(csv_path),
            )

            end_soc = float(df["Battery"].iloc[-1]) if len(df) > 0 else chunk_start_soc

            saved.append({
                "slug": boundary.slug,
                "seg_type": boundary.seg_type,
                "csv_path": str(csv_path),
                "csv_name": csv_name,
                "start_soc": chunk_start_soc,
                "end_soc": end_soc,
                "distance_km": float(sub_route.total_m / 1000.0),
                "source_file": boundary.filename,
            })
            logger.info(f"  Saved {csv_path.name}  SOC {chunk_start_soc:.1f}% -> {end_soc:.1f}%")

        except Exception as e:
            logger.error(f"  Failed to build CSV for {boundary.slug}: {e}")

    return saved


# ===========================================================================
# 4. Summary JSON
# ===========================================================================

def write_summary_json(
    variant_name: str,
    out_dir: str | pathlib.Path,
    soc_trajectory: list[float],
    loop_plan: dict,
    per_day_meta: dict[int, list[dict]],
    converged: bool,
    iterations: int,
    total_distance_km: float,
) -> str:
    """Write a summary.json for one variant tree.

    Returns the path to the written file.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    summary = {
        "variant": variant_name,
        "converged": converged,
        "iterations": iterations,
        "total_distance_km": round(total_distance_km, 2),
        "soc_trajectory_pct": [round(s, 2) for s in soc_trajectory],
        "loop_plan": {str(k): v for k, v in loop_plan.items()},
        "days": {},
    }

    for d, stages in per_day_meta.items():
        day_key = f"day{d + 1}"
        summary["days"][day_key] = {
            "stages": stages,
            "start_soc": stages[0]["start_soc"] if stages else None,
            "end_soc": stages[-1]["end_soc"] if stages else None,
            "total_distance_km": round(sum(s["distance_km"] for s in stages), 2),
        }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary written to {summary_path}")
    return str(summary_path)


# ===========================================================================
# 5. Top-level: process entire optimizer result for one variant
# ===========================================================================

def save_variant_results(
    variant_name: str,
    results_base_dir: str | pathlib.Path,
    routes: dict,
    car: CarState,
    solar_providers: dict,
    wind_providers: dict,
    optimize_result: dict,
    day_boundaries: dict[int, list[SegmentBoundary]],
    solve_outputs: dict[int, dict],
) -> str:
    """Save all per-stage CSVs and summary.json for one variant.

    Parameters
    ----------
    variant_name : e.g. 'prahlad', 'aryaman'
    results_base_dir : e.g. 'data/results'
    routes : {day_index: Route}
    solve_outputs : {day_index: dict from singleday.solve()}
    day_boundaries : {day_index: [SegmentBoundary, ...]}

    Returns the variant output directory path.
    """
    variant_dir = pathlib.Path(results_base_dir) / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    soc_traj = optimize_result.get("s_start_pct", [])
    if hasattr(soc_traj, 'tolist'):
        soc_traj = soc_traj.tolist()

    per_day_meta = {}
    start_day = optimize_result.get("start_day_index", 0)

    for d in range(start_day, len(routes)):
        if d not in solve_outputs:
            continue

        day_dir = variant_dir / f"day{d + 1}"

        boundaries = day_boundaries.get(d, [])
        if not boundaries:
            logger.warning(f"Day {d + 1}: no segment boundaries recorded, skipping CSV split")
            continue

        route = routes.get(d)
        if route is None:
            continue

        start_soc = float(soc_traj[d]) if d < len(soc_traj) else 100.0

        per_day_meta[d] = split_and_save_day(
            solve_output=solve_outputs[d],
            full_route=route,
            car=car,
            solar_provider=solar_providers.get(d),
            wind_provider=wind_providers.get(d),
            day_index=d,
            start_soc_pct=start_soc,
            boundaries=boundaries,
            out_dir=day_dir,
        )

    write_summary_json(
        variant_name=variant_name,
        out_dir=variant_dir,
        soc_trajectory=soc_traj,
        loop_plan=optimize_result.get("loop_plan", {}),
        per_day_meta=per_day_meta,
        converged=optimize_result.get("converged", False),
        iterations=optimize_result.get("iterations", 0),
        total_distance_km=optimize_result.get("total_distance_km", 0.0),
    )

    return str(variant_dir)