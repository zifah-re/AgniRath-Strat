"""
pipeline/update_pmf.py — STUB (block 4.3, owner: Junior C).

Nightly: join data/raw/solcast_forecast.csv x solcast_actuals.csv on
(circle_id, period_end) -> actual/forecast ratio samples -> per-category PMFs
(Paper 1 Table 5 categories, core/solar.py PMF_TABLE) -> write
data/processed/pmf.json + P10/P50/P90 day curves for the DP.
Until enough pairs exist, core/solar.pmf_correction_factor serves the
Paper 1 seed values.
"""

def update(raw_dir: str = "data/raw", out: str = "data/processed/pmf.json") -> None:
    raise NotImplementedError("block 4.3 — Junior C")
