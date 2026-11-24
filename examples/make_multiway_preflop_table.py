r"""Build the multiway preflop table: each of the 169 classes vs k = 1..7 random hands.

Run:
    .venv\Scripts\python examples\make_multiway_preflop_table.py [samples] [processes]

Writes ``src/nutsfinder/bots/multiway/data/preflop_vs_random.npz`` (arrays
``classes``, ``equity``, ``se``, ``samples``, ``seed``). Default 100,000
samples per entry (SE at most 0.16 points); each entry has its own seed, so
the result does not depend on the number of processes. Took about 10 minutes
with 22 processes on the development machine (shared with other jobs).
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from nutsfinder.bots.multiway.equity import PREFLOP_TABLE_FILE, build_preflop_table
from nutsfinder.engine.preflop import ALL_CLASSES

SEED = 20261124


def _row(args: tuple[str, int]) -> tuple[np.ndarray, np.ndarray]:
    cls, samples = args
    eq, se = build_preflop_table(samples, SEED, classes=[cls])
    return eq[0], se[0]


def main() -> None:
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    start = time.perf_counter()
    with ProcessPoolExecutor(processes) as pool:
        rows = list(pool.map(_row, [(cls, samples) for cls in ALL_CLASSES]))
    eq = np.array([r[0] for r in rows])
    se = np.array([r[1] for r in rows])
    PREFLOP_TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREFLOP_TABLE_FILE, classes=np.array(ALL_CLASSES), equity=eq, se=se,
                        samples=samples, seed=SEED)
    print(f"wrote {PREFLOP_TABLE_FILE} ({len(ALL_CLASSES)} x 7, {samples:,} samples each, "
          f"max SE {se.max():.4f}) in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
