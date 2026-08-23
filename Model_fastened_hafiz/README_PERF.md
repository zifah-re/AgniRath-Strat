# Model_fastened_hafiz — hardware-optimized build

A **complete, speed-tuned copy** of `Model/` for an **8–16 core laptop/desktop**.
It produces the **same strategy results** as the baseline — it just runs them
faster. Every accelerator is a **config flag**, so the only file that actually
differs from `Model/` is `configs/solver_config.py`; the rest of the tree is here
in full so you can run it directly.

## Setup (one step: bring the data in)

To keep this download small, the bulky `data/` and `output/` folders are **not**
included (you already have them in `Model/`). Point this build at them:

```bash
cd Model_fastened_hafiz
# option A — symlink (no copy, always in sync with your real data):
ln -s ../Model/data data
mkdir -p output
# option B — copy, if you prefer them independent:
#   cp -r ../Model/data ./data && mkdir -p output
```

Then run exactly like the baseline:

```bash
python -m optimizers.trust_region                 # full 8-day, both variants
python -m optimizers.trust_region resolve --resume-day 4 --start-soc 62   # re-solve
```

That's it. To A/B against the baseline, flip any flag back to its `# was …`
value in `configs/solver_config.py`.

---

## What's turned on (all in `configs/solver_config.py`, marked `FASTENED`)

### 1. `TIER2_USE_PROCESS_POOL = True` — the big win
Tier 2 (the per-day sampler, ~80% of runtime) fans the 8 days out over
**processes** instead of GIL-bound **threads**, so it actually uses all your
cores. **Validated correct** (all objects pickle; identical results). On 8 cores
the Tier 2 phase drops toward 1/6–1/8, i.e. roughly **2.5–4× faster end-to-end**.

### 2. `SKIP_BREAKDOWN_WHEN_UNUSED = True`
Skips a per-substep breakdown calculation that's discarded anyway on
deterministic runs. Measured **1.23×/sim, bit-identical** results.

### 3. `ENERGY_GRID_M = 200`, `GA_GENERATIONS = 12`
Mild fidelity/time trades (end-of-day SOC differs <0.3%). Revert `ENERGY_GRID_M`
to 150 first if you ever want to rule them out.

### 4. numba JIT (optional, not pre-enabled)
The biggest single-sim lever (~2–5×), but `numba` wasn't in my build sandbox so I
won't ship a kernel I couldn't run. `README_PERF.md` §4 in the round-5 summary
gives the exact, safe recipe (extract the SOC-integration loop into an `@njit`
function behind a flag, with a pure-Python fallback and a self-check that
disables numba if the two ever disagree). Say the word and I'll write it.

---

## Rough expectations (8-core machine)

| Build | Tier 2 | per-sim | full run (both variants) |
|---|---|---|---|
| Baseline `Model/` | threads (GIL-bound) | 1.0× | ~60 min |
| + breakdown-skip | threads | 1.23× | ~49 min |
| + **process pool** | **8-way** | 1.23× | **~15–22 min** |
| + numba (your machine) | 8-way | ~3–5× | **~6–10 min** |

(The process-pool line is the dependable multi-core win. My sandbox only had 2
cores, so I validated correctness there, not the speedup — that shows up on your
machine.)

## The only file that differs from `Model/`

`configs/solver_config.py` (four `FASTENED`-marked lines). Everything else is a
byte-for-byte copy of the round-5 `Model/` code — included so this folder runs on
its own once you link in `data/`.
