"""
pipeline/blind_stage.py — STUB (block 2.3, owner: Junior A).

Golden-Envelope overnight rebuild (Plan v3 §2.2): ingest the evening KML,
re-run build_route + red-flag pass for the blind day, regenerate placeholder
loops (race_config.BLIND_LOOP_PLACEHOLDER_KM) where loop info is absent.
Must be ONE command; drilled before race (workplan 9.1).
2024-corridor prior for Day 3 lives in data/raw/ (2024 KML already in the
AgniRath-Strat Dashboard folder).
"""

def rebuild_from_envelope(kml_path: str, day_index: int) -> None:
    raise NotImplementedError("block 2.3 — Junior A")
