r"""Bucket lookup cost per street, and how noisy Monte Carlo E[HS] buckets would be.

Run:
    .venv\Scripts\python examples\bucket_noise.py [n_deals]

The default ``ehs_method="exact"`` has no bucket noise at all: E[HS] is
enumerated, so a hand's bucket is a fixed function of (hand, board). This
script shows what the Monte Carlo alternative (:func:`ehs_mc`) would cost in
noise: for random (hand, board) pairs per street it compares the MC bucket
(several sample counts, two seeds) with the exact bucket.

Reported per street and sample count:
- ``!=exact``: share of deals whose MC bucket differs from the exact bucket;
- ``mean|d|``: mean absolute bucket difference;
- ``seed!=``: share of deals whose bucket changes between two MC seeds;
- ``ms``: time per MC lookup.
And the exact lookup cost: cold (new table) and warm (cached table).
Wall times on this laptop vary by up to ~5x (thermal throttling): best of 3.
"""

from __future__ import annotations

import sys
import time
from bisect import bisect_right

import numpy as np

from nutsfinder.abstraction import strength
from nutsfinder.abstraction.bucket_data import EDGES
from nutsfinder.abstraction.cards import EHSTables, canonical_board, ehs_mc, sample_deals, turn_table_exact

SAMPLE_COUNTS = (200, 1000, 5000)


def best_of(fn, reps: int = 3) -> float:
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def exact_costs(tables: EHSTables) -> None:
    deals = {s: sample_deals(s, 2000, 77) for s in (1, 2, 3)}
    for s in (1, 2, 3):
        if s == 2:
            cold = best_of(lambda: [turn_table_exact(canonical_board(b)[0]) for _, b in deals[2][:5]], 2) / 5
            print(f"turn  cold (new turn table): {cold * 1e3:.1f} ms")
        if s == 1:
            f = EHSTables(flop_file=None)
            cold = best_of(lambda: f.flop_table(canonical_board(deals[1][0][1])[0]), 1)
            print(f"flop  cold (no file, 49 turn tables): {cold:.2f} s")
        for h, b in deals[s]:
            tables.ehs(h, b)  # warm
        warm = best_of(lambda: [tables.ehs(h, b) for h, b in deals[s]]) / len(deals[s])
        print(f"street {s} warm lookup: {warm * 1e6:.1f} us")
    river = best_of(lambda: [strength.river_hs(b) for _, b in deals[3][:200]]) / 200
    print(f"river table (new board, all 1,081 hands): {river * 1e3:.2f} ms")


def noise(tables: EHSTables, n: int) -> None:
    print(f"\nMC noise vs exact buckets, {n} deals per street")
    for s in (1, 2, 3):
        deals = sample_deals(s, n, 4242)
        exact = np.array([bisect_right(EDGES[s], tables.ehs(h, b)) for h, b in deals])
        for k in SAMPLE_COUNTS:
            t = time.perf_counter()
            b0 = np.array([bisect_right(EDGES[s], ehs_mc(h, b, k, salt=0)) for h, b in deals])
            ms = (time.perf_counter() - t) / n * 1e3
            b1 = np.array([bisect_right(EDGES[s], ehs_mc(h, b, k, salt=1)) for h, b in deals])
            print(f"street {s} samples {k:5d}: !=exact {np.mean(b0 != exact):.2f}  "
                  f"mean|d| {np.mean(np.abs(b0 - exact)):.2f}  max|d| {np.max(np.abs(b0 - exact))}  "
                  f"seed!= {np.mean(b0 != b1):.2f}  {ms:.2f} ms")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    tables = EHSTables()
    strength.strength_tables()
    exact_costs(tables)
    noise(tables, n)


if __name__ == "__main__":
    main()
