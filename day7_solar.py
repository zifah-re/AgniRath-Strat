"""
run_day7_solar_2.py -- orchestrates the full solar pipeline for the three
Day 7 route files and writes the final mean_*.jsonl output into
Dashboard/Solar with a "_2" suffix (matching the Day 5 re-run convention).

This exists because the pipeline has three folders that don't line up:
  solar.py            reads   Fallback Model/Saves/{filename}   (hardcoded,
                               strips any folder you pass -- only the
                               filename is used)
  solar.py            writes  Model/Solar/{filename}.jsonl       (hardcoded,
                               --output-dir is accepted but never actually
                               used for the save path)
  solarProcessing.py  reads   Fallback Model/Solar/*.jsonl       (globs
                               EVERYTHING there, not just new files)
  solarProcessing.py  writes  Fallback Model/Solar_Processed/mean_*.jsonl

This script copies files across those gaps for you, in order, and finally
copies just the 3 new mean_*.jsonl into Dashboard/Solar as *_2.jsonl.

Run from the repo root (AgniRath-Strat), with your Solcast key set first:
    $env:SOLCAST_API_KEY = "your_key_here"
    python run_day7_solar_2.py

If anything fails partway through, rerun it -- every step is safe to
repeat (copies overwrite, solar.py/solarProcessing.py just regenerate).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# 0. Paths (relative to repo root -- run this script from AgniRath-Strat/)
# ---------------------------------------------------------------------
REPO_ROOT = Path(".")
SAVES_SRC = REPO_ROOT / "Dashboard" / "Saves"
SAVES_DST = REPO_ROOT / "Fallback Model" / "Saves"
MODEL_SOLAR = REPO_ROOT / "Model" / "Solar"
FALLBACK_SOLAR = REPO_ROOT / "Fallback Model" / "Solar"
FALLBACK_SOLAR_PROCESSED = REPO_ROOT / "Fallback Model" / "Solar_Processed"
DASHBOARD_SOLAR = REPO_ROOT / "Dashboard" / "Solar"

SOLAR_PY = REPO_ROOT / "Model" / "solar.py"
SOLAR_PROCESSING_PY = REPO_ROOT / "Model" / "solarProcessing.py"

DAY7_FILES = [
    "2026 Sasol Solar Challenge Route (Publish)_Day 7_16 Sept Stage 1 Springbok to Van Rhynsdorp.kml.save",
    "2026 Sasol Solar Challenge Route (Publish)_Day 7_16 Sept Stage 2 Van Rhynsdorp to Clanwilliam.kml.save",
    "2026 Sasol Solar Challenge Route (Publish)_Day 7_Van Rhynsdorp Loop.kml.save",
]

SUFFIX = "_2"   # what to append before ".jsonl" in the final Dashboard/Solar output


def raw_jsonl_name(kml_save_name: str) -> str:
    """Matches solar.py's own naming: fname[:-9] + '.jsonl' (".kml.save" is 9 chars)."""
    assert kml_save_name.endswith(".kml.save"), kml_save_name
    return kml_save_name[: -len(".kml.save")] + ".jsonl"


def step1_copy_kml_to_fallback() -> None:
    print("=== Step 1: copy Day 7 KML files into 'Fallback Model/Saves' ===")
    SAVES_DST.mkdir(parents=True, exist_ok=True)
    for fname in DAY7_FILES:
        src = SAVES_SRC / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing source KML: {src}")
        dst = SAVES_DST / fname
        shutil.copy2(src, dst)
        print(f"  copied -> {dst}")


def step2_run_solar_py() -> None:
    print("\n=== Step 2: run solar.py (Solcast query, one call per ~10km point) ===")
    if not os.environ.get("SOLCAST_API_KEY"):
        raise RuntimeError(
            "SOLCAST_API_KEY is not set in the environment. Set it first, e.g.:\n"
            "  PowerShell:  $env:SOLCAST_API_KEY = \"your_key_here\"\n"
            "  cmd.exe:     set SOLCAST_API_KEY=your_key_here"
        )
    # solar.py saves to Model/Solar/{name}.jsonl but never creates that folder
    # itself -- if it doesn't exist yet, open(file_path, 'w') fails with
    # FileNotFoundError after the (successful) Solcast query, which looks
    # like a network failure but isn't. Create it up front so that can't happen.
    MODEL_SOLAR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SOLAR_PY), *DAY7_FILES]
    print("  running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def step3_copy_raw_jsonl_to_fallback() -> list[str]:
    print("\n=== Step 3: copy solar.py's raw output into 'Fallback Model/Solar' ===")
    FALLBACK_SOLAR.mkdir(parents=True, exist_ok=True)
    raw_names = []
    for fname in DAY7_FILES:
        raw_name = raw_jsonl_name(fname)
        src = MODEL_SOLAR / raw_name
        if not src.exists():
            raise FileNotFoundError(f"Expected solar.py output missing: {src}")
        dst = FALLBACK_SOLAR / raw_name
        shutil.copy2(src, dst)
        print(f"  copied -> {dst}")
        raw_names.append(raw_name)
    return raw_names


def step4_run_solar_processing() -> None:
    print("\n=== Step 4: run solarProcessing.py (trims to 06:00-17:00 SAST, "
          "builds diurnal mean) ===")
    print("  NOTE: this reprocesses EVERY .jsonl currently in 'Fallback Model/Solar', "
          "not just the 3 new ones -- harmless, just regenerates old output too.")
    cmd = [sys.executable, str(SOLAR_PROCESSING_PY)]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def step5_copy_mean_outputs_with_suffix(raw_names: list[str]) -> None:
    print(f"\n=== Step 5: copy the 3 new mean_*.jsonl into 'Dashboard/Solar' as *{SUFFIX}.jsonl ===")
    DASHBOARD_SOLAR.mkdir(parents=True, exist_ok=True)
    for raw_name in raw_names:
        mean_name = f"mean_{raw_name}"
        src = FALLBACK_SOLAR_PROCESSED / mean_name
        if not src.exists():
            raise FileNotFoundError(f"Expected processed output missing: {src}")
        dst_name = mean_name[: -len(".jsonl")] + f"{SUFFIX}.jsonl"
        dst = DASHBOARD_SOLAR / dst_name
        shutil.copy2(src, dst)
        print(f"  copied -> {dst}")


def main() -> None:
    step1_copy_kml_to_fallback()
    step2_run_solar_py()
    raw_names = step3_copy_raw_jsonl_to_fallback()
    step4_run_solar_processing()
    step5_copy_mean_outputs_with_suffix(raw_names)
    print("\nDone. New files in Dashboard/Solar:")
    for raw_name in raw_names:
        print(f"  mean_{raw_name[:-len('.jsonl')]}{SUFFIX}.jsonl")


if __name__ == "__main__":
    main()