"""
pipeline/build_route.py — STUB (block 2.1, owner: Junior A).

KMZ -> elevation -> smoothing -> TomTom v_max -> curvature/turn caps ->
equal-area circle assignment -> red flags -> data/processed/route_day{d}.parquet
(schema contract: core/route.py REQUIRED_COLUMNS).

Reuse: AgniRath-Strat/Dashboard/Google_Earth.py (elevation),
Dashboard/traffic.py (TomTom), Aryaman/GradientMapping (validation),
race_completion/process_route_data.py (pattern).
"""

def build(kmz_path: str, out_dir: str = "data/processed") -> None:
    raise NotImplementedError("block 2.1 — Junior A")
