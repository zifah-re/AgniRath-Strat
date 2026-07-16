import pandas as pd
import json
import sys
import glob
import argparse
from pathlib import Path
from datetime import timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)
def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Add ist_seconds (seconds since midnight IST) and ist_date columns."""
    # period_end already has +00:00 offset; parse directly
    dt_utc = pd.to_datetime(df["period_end"], utc=True)
    dt_ist = dt_utc.dt.tz_convert(IST)

    midnight_ist = dt_ist.dt.normalize()  # 00:00:00 IST of that day
    df["period_end"]=dt_ist #changing utc to ist
    df["ist_seconds"] = (dt_ist - midnight_ist).dt.total_seconds()
    df["ist_date"] = dt_ist.dt.strftime("%d %b %Y")
    df["ist_time_label"] = dt_ist.dt.strftime("%H:%M")
    return df

def main():
    parser = argparse.ArgumentParser(description="Plot solar JSONL files vs IST time of day")
    parser.add_argument("files", nargs="*", help="JSONL paths (default: auto-discover)")
    parser.add_argument("--output-dir", default=".", help="Directory to save plots (default: current dir)")
    args = parser.parse_args()

    paths = args.files or sorted(glob.glob("solar_input_*.jsonl"))
    if not paths:
        print("No JSONL files found. Pass file paths or run from the folder containing them.")
        sys.exit(1)

    print(f"Loading {len(paths)} file(s): {[Path(p).name for p in paths]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        df = load_jsonl(p)
        df = parse_timestamps(df)
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.set_index("period_end")
        df_resampled = df.resample("2s").asfreq()
        columns_to_interpolate = ["dni", "ghi", "air_temp"]
        df_resampled[columns_to_interpolate] = df_resampled[columns_to_interpolate].interpolate(method="linear")
        df_final=df_resampled.drop(columns=['ist_seconds','ist_date','ist_time_label'])
        df_final=df_final.reset_index()
        df_final["period_end"] = df_final["period_end"].dt.strftime("%Y-%m-%dT%H:%M:%S%:z")
        df_final=df_final[['dni','ghi','air_temp','period_end']]
        df_final.to_json(p, orient="records", lines=True)

main()