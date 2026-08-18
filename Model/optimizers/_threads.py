"""
optimizers/_threads.py — single source of truth for worker-count normalization.

The Tier 2 parallel sampling loop (and anything else that spawns worker
processes) must never request more workers than the machine can actually run
— a race day with 8 days is not a 32-core machine. Every parallel entry point
that accepts an explicit n_workers should funnel it through worker_cap() so a
user-supplied n_workers is clamped to os.cpu_count() exactly once, in one
place, instead of each caller doing its own os.cpu_count() math (and each
getting a slightly different answer for the same logical request).
"""

from __future__ import annotations

import os


def worker_cap(n_workers: int | None) -> int | None:
    """Return the effective worker count for process-pool parallelism.

    - None        -> None (let the pool default to os.cpu_count()).
    - 1           -> 1 (sequential path).
    - > 1         -> min(n_workers, os.cpu_count()) — never overcommit.
    - <= 0        -> None (treat non-positive as "let the pool decide").
    """
    if n_workers is None or n_workers <= 0:
        return None
    return min(int(n_workers), os.cpu_count() or 1)
