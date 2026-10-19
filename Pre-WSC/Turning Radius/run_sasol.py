"""
run_sasol.py - run the turn-radius analysis over the Sasol Solar Challenge
(South Africa) route saves.

Put this file next to turn_radius.py (in the 'Turning Radius' folder) and run:
    python run_sasol.py

Reads : Agnirath-Strat/Fallback Model/Saves/*.kml.save
Writes: Agnirath-Strat/Pre-WSC/Turning Radius/sasol_results/
          per_file/            one CSV + histogram + route plot per stage file
          ALL_DAYS_*.png/csv   combined histogram (stages merged per day)

Skipped: any file with "loop" in its name, and every Day 3 file.
Optional overrides:  --saves "<folder>"  --outdir "<folder>"  --top 10
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import turn_radius as tr

HERE = os.path.dirname(os.path.abspath(__file__))              # .../Pre-WSC/Turning Radius
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))         # .../Agnirath-Strat
DEFAULT_SAVES = os.path.join(REPO, "Fallback Model", "Saves")
DEFAULT_OUT = os.path.join(HERE, "sasol_results")

SKIP_DAYS = {3}


def day_of(name):
    m = re.search(r"Day\s*(\d+)(?!\d)", name, re.I)
    return int(m.group(1)) if m else None


def select_files(saves_dir):
    keep, skipped = [], []
    for fp in sorted(glob.glob(os.path.join(saves_dir, "*.kml.save"))):
        name = os.path.basename(fp)
        d = day_of(name)
        if "loop" in name.lower():
            skipped.append((name, "loop"))
        elif d in SKIP_DAYS:
            skipped.append((name, f"day {d}"))
        elif d is None:
            skipped.append((name, "no day number in name"))
        else:
            keep.append((d, fp))
    return keep, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--saves", default=DEFAULT_SAVES)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--rank", choices=["sharpest", "largest"], default="sharpest")
    args = ap.parse_args()

    if not os.path.isdir(args.saves):
        raise SystemExit(f"Saves folder not found: {args.saves}\n"
                         f"Use --saves \"<path>\" to point at it.")
    files, skipped = select_files(args.saves)
    if not files:
        raise SystemExit(f"No usable .kml.save files in {args.saves}")

    print(f"Saves folder : {args.saves}")
    print(f"Output folder: {args.outdir}")
    print(f"Running on {len(files)} files, skipping {len(skipped)}:")
    for name, why in skipped:
        print(f"   skip ({why}): {name}")
    print()

    per_file_dir = os.path.join(args.outdir, "per_file")
    by_day = defaultdict(list)
    for d, fp in files:
        _, _, turns, _ = tr.report(fp, args.top, args.rank, per_file_dir)
        by_day[d].extend(turns)          # merge Stage 1 + Stage 2 of the same day

    per_day = [(f"Day {d}", by_day[d]) for d in sorted(by_day)]
    tr.cumulative_report(per_day, args.outdir)