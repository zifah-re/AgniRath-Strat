"""
Stitch the Sasol race telemetry fragments into one JSONL file per race day.

Reads   Dashboard/Logs/Sasol/Day N/*.jsonl            (raw logger fragments, never modified)
Writes  Pre-WSC/sasol_analysis/cleaned_logs/dayN.jsonl (telemetry only, original schema)
        Pre-WSC/sasol_analysis/cleaned_logs/summary.json

What it does:
  1. Drops event lines ({"error": "crc_fail"}, {"info": "LoRa ready"}), repairs keys with
     stray backticks, drops records whose key set is not exactly the MAIN or TEMP packet.
  2. Normalises every _rx_time to SAST (+02:00). Files logged with another offset
     (day1stage1_3_2 is tagged +05:30) are split at the first >1 h forward clock jump:
     before it the face value is UTC, after it the stated offset is correct. Within a
     fragment, line order is packet order: if the clock steps backwards, the records before
     the step are shifted earlier so the fragment stays in order.
  3. Where two fragments of the same day overlap in time, the earlier fragment's tail was
     received late (its last readings continue into the later fragment's first ones), so
     the earlier fragment is moved back to end just before the later one starts. No
     records are dropped. See OVERLAP_FIXES.
  4. Sorts by _rx_time and writes the day file. MAIN (~2 s) and TEMP (~5 s) packets stay
     interleaved at their native receive times; nothing is resampled or annotated.
"""

import json
import statistics as st
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "Dashboard" / "Logs" / "Sasol"
OUT_DIR = Path(__file__).resolve().parent / "cleaned_logs"

SAST = timezone(timedelta(hours=2))

# Day number -> (date, window start hour, window end hour), SAST
RACE_DAYS = {
    1: ("2026-09-10", 9, 17),
    2: ("2026-09-11", 8, 17),
    3: ("2026-09-12", 8, 17),
    4: ("2026-09-13", 8, 17),
    5: ("2026-09-14", 8, 17),
    6: ("2026-09-15", 8, 17),
    7: ("2026-09-16", 8, 17),
    8: ("2026-09-17", 8, 15),
}

MAIN_MARKER = "Vehicle_Velocity"
TEMP_MARKER = "Motor_Temp"
CLOCK_JUMP_S = 3600       # forward jump that marks a laptop clock/timezone switch
GAP_S = 30               # silences longer than this are listed in the summary


# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------
def read_fragment(path):
    """Returns (records, stats). Each record is the parsed dict, keys repaired."""
    records, stats = [], Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                stats["unparseable"] += 1
                continue
            if "info" in r or "error" in r:
                stats["events_dropped"] += 1
                continue
            if any("`" in k for k in r):
                r = {k.replace("`", ""): v for k, v in r.items()}
                stats["keys_repaired"] += 1
            records.append(r)
    return records, stats


def canonical_schemas(all_records):
    """Most common key set among records carrying each packet's marker key."""
    main, temp = Counter(), Counter()
    for r in all_records:
        if MAIN_MARKER in r:
            main[frozenset(r)] += 1
        elif TEMP_MARKER in r:
            temp[frozenset(r)] += 1
    return main.most_common(1)[0][0], temp.most_common(1)[0][0]


# ---------------------------------------------------------
# Time normalisation
# ---------------------------------------------------------
def to_sast(records, fname):
    """Attaches a SAST datetime to each record (as r['__t']) and rewrites _rx_time if needed."""
    raw = [datetime.fromisoformat(r["_rx_time"]) for r in records]
    if all(t.utcoffset() == timedelta(hours=2) for t in raw):
        for r, t in zip(records, raw):
            r["__t"] = t
        return None

    # Wrong-offset file: face value is UTC until the clock jumps forward, then the offset is right
    jump = next((i for i in range(1, len(raw))
                 if (raw[i] - raw[i - 1]).total_seconds() > CLOCK_JUMP_S), len(raw))
    for i, (r, t) in enumerate(zip(records, raw)):
        if i < jump:
            t = t.replace(tzinfo=timezone.utc)
        t = t.astimezone(SAST)
        r["__t"] = t
        r["_rx_time"] = t.isoformat(timespec="microseconds")
    return {"file": fname, "offset": raw[0].isoformat()[-6:], "switch_index": jump,
            "sast_range": [records[0]["__t"].isoformat(), records[-1]["__t"].isoformat()]}


def fix_backward_steps(records, fname):
    """
    Line order within a fragment is the true packet order. If the clock steps backwards,
    trust the later clock and shift everything before the step earlier so it ends one
    packet interval before the step. Returns a list of the corrections applied.
    """
    dts = [(b["__t"] - a["__t"]).total_seconds() for a, b in zip(records, records[1:])]
    step = timedelta(seconds=st.median([d for d in dts if d > 0])) if dts else timedelta(0)
    fixes = []
    for i in range(len(records) - 1, 0, -1):
        if records[i]["__t"] >= records[i - 1]["__t"]:
            continue
        shift = records[i - 1]["__t"] - records[i]["__t"] + step
        for r in records[:i]:
            r["__t"] -= shift
            r["_rx_time"] = r["__t"].isoformat(timespec="microseconds")
        fixes.append({"file": fname, "at_index": i, "shifted_records": i,
                      "shift_s": -round(shift.total_seconds(), 3)})
    return fixes


# ---------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------
# The overlapping fragments are not two simultaneous streams: temperatures, speed and
# current show the later file starts where the earlier one ends, so the earlier file's
# tail was received late. Each known overlap names how the earlier file is moved back:
#   "whole"      - shift the entire file (the gap before it absorbs the shift)
#   "after_stop" - compress its last standstill and shift everything after it
OVERLAP_FIXES = {
    "day1stage1_1.jsonl": "whole",
    "day3stage1_2.jsonl": "whole",
    "day7stage1_2.jsonl": "after_stop",
}


def set_time(r, t):
    r["__t"] = t
    r["_rx_time"] = t.isoformat(timespec="microseconds")


def packet_step(records):
    dts = [(b["__t"] - a["__t"]).total_seconds() for a, b in zip(records, records[1:])]
    return timedelta(seconds=st.median([d for d in dts if d > 0]))


def last_standstill(records, min_len):
    """Index range (i0, i1) of the last run of MAIN packets with zero speed and current."""
    best, start = None, None
    for i, r in enumerate(records):
        if MAIN_MARKER not in r:
            continue
        still = abs(r["Vehicle_Velocity"]) < 0.05 and abs(r["Bus_Current"]) < 0.05
        if still and start is None:
            start = i
        if not still and start is not None:
            if records[last_main]["__t"] - records[start]["__t"] >= min_len:
                best = (start, last_main)
            start = None
        if still:
            last_main = i
    if start is not None and records[last_main]["__t"] - records[start]["__t"] >= min_len:
        best = (start, last_main)
    return best


def shift_overlaps(fragments):
    """fragments: list of (fname, records) for one day. Moves earlier files back in place."""
    decisions = []
    frags = sorted((f for f in fragments if f[1]), key=lambda f: f[1][0]["__t"])
    for (fa, ra), (fb, rb) in zip(frags, frags[1:]):
        if ra[-1]["__t"] < rb[0]["__t"]:
            continue
        overlap = ra[-1]["__t"] - rb[0]["__t"]
        shift = overlap + packet_step(ra)
        mode = OVERLAP_FIXES.get(fa)
        assert mode, f"Unhandled overlap: {fa} runs {overlap.total_seconds():.0f} s into {fb}"

        if mode == "whole":
            first_moved = 0
            for r in ra:
                set_time(r, r["__t"] - shift)
            note = "whole file shifted"
        else:
            i0, i1 = last_standstill(ra, 2 * shift)
            t0, length = ra[i0]["__t"], ra[i1]["__t"] - ra[i0]["__t"]
            scale = (length - shift) / length
            for r in ra[i0:i1 + 1]:
                set_time(r, t0 + (r["__t"] - t0) * scale)
            for r in ra[i1 + 1:]:
                set_time(r, r["__t"] - shift)
            first_moved = i0
            note = (f"standstill {ra[i0]['__t'].time()}-{ra[i1]['__t'].time()} compressed from "
                    f"{length.total_seconds():.0f} s to {(length - shift).total_seconds():.0f} s, "
                    f"everything after shifted")
        decisions.append({
            "earlier_file": fa, "later_file": fb,
            "overlap_s": round(overlap.total_seconds(), 1), "shift_s": -round(shift.total_seconds(), 1),
            "mode": mode, "note": note, "records_moved": len(ra) - first_moved,
            "earlier_file_now_ends": ra[-1]["_rx_time"], "later_file_starts": rb[0]["_rx_time"],
        })

    # Shifting must not have created a new overlap with the file before
    frags = sorted((f for f in fragments if f[1]), key=lambda f: f[1][0]["__t"])
    for (fa, ra), (fb, rb) in zip(frags, frags[1:]):
        assert ra[-1]["__t"] < rb[0]["__t"], f"{fa} still overlaps {fb} after shifting"
    return decisions


# ---------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------
def gaps_and_coverage(records, date, h0, h1):
    w0 = datetime.fromisoformat(f"{date}T{h0:02d}:00:00+02:00")
    w1 = datetime.fromisoformat(f"{date}T{h1:02d}:00:00+02:00")
    times = [r["__t"] for r in records]
    gaps = []
    for a, b in zip(times, times[1:]):
        if (b - a).total_seconds() > GAP_S:
            gaps.append({"start": a.isoformat(), "end": b.isoformat(),
                         "duration_s": round((b - a).total_seconds(), 1)})

    # Window time covered = window minus silences (>GAP_S) and minus time before first/after last record
    edges = [w0] + [t for t in times if w0 <= t <= w1] + [w1]
    uncovered = sum((b - a).total_seconds() for a, b in zip(edges, edges[1:])
                    if (b - a).total_seconds() > GAP_S)
    total = (w1 - w0).total_seconds()
    return gaps, round(100 * (total - uncovered) / total, 1)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded = {}  # day -> list of (fname, records)
    stats = {}
    for day in RACE_DAYS:
        loaded[day], stats[day] = [], Counter()
        for path in sorted((RAW_DIR / f"Day {day}").glob("*.jsonl")):
            recs, s = read_fragment(path)
            stats[day].update(s)
            loaded[day].append((path.name, recs))

    main_keys, temp_keys = canonical_schemas(r for d in loaded.values() for _, rs in d for r in rs)

    summary = {}
    for day, (date, h0, h1) in RACE_DAYS.items():
        s = stats[day]
        tz_fixes, step_fixes = [], []
        for fname, recs in loaded[day]:
            kept = [r for r in recs if frozenset(r) in (main_keys, temp_keys)]
            s["bad_schema_dropped"] += len(recs) - len(kept)
            recs[:] = kept
            fix = to_sast(recs, fname)
            step_fixes.extend(fix_backward_steps(recs, fname))
            if fix:
                fix["sast_range"] = [recs[0]["_rx_time"], recs[-1]["_rx_time"]]
                tz_fixes.append(fix)
            wrong_date = [r for r in recs if r["__t"].date().isoformat() != date]
            assert not wrong_date, f"Day {day}/{fname}: {len(wrong_date)} records not on {date}"
        overlaps = shift_overlaps(loaded[day])

        merged = sorted((r for _, recs in loaded[day] for r in recs), key=lambda r: r["__t"])
        out_path = OUT_DIR / f"day{day}.jsonl"
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            for r in merged:
                f.write(json.dumps({k: v for k, v in r.items() if k != "__t"},
                                   separators=(",", ":")) + "\n")

        gaps, coverage = gaps_and_coverage(merged, date, h0, h1)
        summary[f"day{day}"] = {
            "date": date,
            "race_window_sast": [f"{h0:02d}:00", f"{h1:02d}:00"],
            "files": [name for name, _ in loaded[day]],
            "first_record": merged[0]["_rx_time"] if merged else None,
            "last_record": merged[-1]["_rx_time"] if merged else None,
            "records_written": len(merged),
            "main_packets": sum(1 for r in merged if MAIN_MARKER in r),
            "temp_packets": sum(1 for r in merged if TEMP_MARKER in r),
            "events_dropped": s["events_dropped"],
            "bad_schema_dropped": s["bad_schema_dropped"],
            "unparseable_lines": s["unparseable"],
            "keys_repaired": s["keys_repaired"],
            "timezone_fixes": tz_fixes,
            "clock_step_fixes": step_fixes,
            "overlap_shifts": overlaps,
            "window_coverage_pct": coverage,
            "gaps_over_30s": gaps,
        }
        print(f"Day {day} ({date}): {len(merged):6d} records  "
              f"{summary[f'day{day}']['first_record'][11:19]} -> {summary[f'day{day}']['last_record'][11:19]}  "
              f"coverage {coverage:5.1f}%  gaps>{GAP_S}s: {len(gaps)}")
        for fx in tz_fixes:
            print(f"    tz fix {fx['file']} ({fx['offset']}): now {fx['sast_range'][0][11:19]} -> {fx['sast_range'][1][11:19]} SAST")
        for fx in step_fixes:
            print(f"    clock step {fx['file']} at line {fx['at_index']}: first {fx['shifted_records']} records shifted {fx['shift_s']} s")
        for o in overlaps:
            print(f"    overlap {o['earlier_file']} ran {o['overlap_s']} s into {o['later_file']}: "
                  f"shifted {o['shift_s']} s ({o['note']}); now ends {o['earlier_file_now_ends'][11:19]}, "
                  f"next starts {o['later_file_starts'][11:19]}")

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {len(RACE_DAYS)} day files and summary.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
