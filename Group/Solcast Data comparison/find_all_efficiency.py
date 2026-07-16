import json
import sys
import glob
import argparse
from pathlib import Path
from solar_ratio import main as main_solar
from statistics import mean, median, mode


def main():
    parser = argparse.ArgumentParser(description="Plot solar JSONL files vs IST time of day")
    parser.add_argument("files", nargs="*", help="JSONL paths (default: auto-discover)")
    parser.add_argument("--output-dir", default=".", help="Directory to save plots (default: current dir)")
    args = parser.parse_args()

    paths = args.files or sorted(glob.glob("Logs\\*.jsonl"))
    if not paths:
        print("No JSONL files found. Pass file paths or run from the folder containing them.")
        sys.exit(1)

    print(f"Loading {len(paths)} file(s): {[Path(p).name for p in paths]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    efficiency=[]
    count=0
    for p in paths:
        _,_,mean_ratio=main_solar(Path(p).name)
        if mean_ratio<0:
            count+=1
            continue
        efficiency.append(float(mean_ratio*24))
    print(f"Skipped {count} files because mean ratio was negative")
    print(f"Mean: {mean(efficiency)}\nMedian: {median(efficiency)}\nMode: {mode(efficiency)}")
    print(efficiency)
main()