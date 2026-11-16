"""Preflop hand classes, their equity against a random hand, and ranges.

The 169 hand classes
--------------------
There are C(52, 2) = 1,326 starting hands, but before the flop the suits
only matter through "same suit or not". So the hands fall into 169 classes:

- 13 pairs (``"AA"``, ..., ``"22"``), 6 combos each: C(4, 2) = 6 ways to pick
  two suits;
- 78 suited hands (``"AKs"``), 4 combos each: one per suit;
- 78 offsuit hands (``"AKo"``), 12 combos each: 4 x 3 ordered suit pairs.

Check: 13 * 6 + 78 * 4 + 78 * 12 = 78 + 312 + 936 = 1,326. (78 = C(13, 2),
the ways to pick two different ranks.)

The equity table
----------------
``PREFLOP_EQUITY[cls]`` is the class's all-in equity against a uniformly
random opponent hand, precomputed by :func:`compute_preflop_table` (see
``examples/make_preflop_table.py``) and stored in ``preflop_data.py``. Every
combo of a class has the same equity against a *random* hand (relabelling
the suits changes nothing), so one representative combo per class is enough.

How many samples? A sample's payoff X is in [0, 1], so Var(X) <= 1/4 and
SE <= 0.5 / sqrt(n). For SE < 0.2 percentage points we need
n > (0.5 / 0.002)^2 = 62,500. The table uses 100,000 per class, so every SE
is at most 0.5 / sqrt(100,000) = 0.16 points (the stored SEs are a bit
smaller, since Var(X) < 1/4 away from 50%).

Ranges from the ranking
-----------------------
:data:`RANKING` sorts the classes by that equity, best first. "The top x%
of hands", :func:`top_range`, takes classes from the top of the ranking until
they hold at least x% of the 1,326 combos. This is a crude but simple
definition of a "strong range".

Bots only ever use the widths in :data:`RANGE_GRID` (1%, 2%, ..., 100%), so
the equity of every class against every grid range is precomputed too
(:data:`PREFLOP_VS_TOP`, ``preflop_range_data.py``, 5,000 samples per entry)
and preflop decisions cost a table lookup. Known weakness of the ranking: equity against a *random*
hand overrates big offsuit cards (e.g. K9o) and underrates small pairs and
suited connectors, which play better against strong ranges than the ranking
suggests.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from nutsfinder.engine.cards import Card, Rank, Suit
from nutsfinder.engine.equity import RANDOM_RANGE, Combo, equity_vs_range
from nutsfinder.engine.preflop_data import PREFLOP_TABLE
from nutsfinder.engine.preflop_range_data import PREFLOP_VS_TOP

__all__ = [
    "ALL_CLASSES",
    "PREFLOP_EQUITY",
    "PREFLOP_SE",
    "RANKING",
    "hand_class",
    "class_combos",
    "representative",
    "compute_preflop_table",
    "top_classes",
    "top_range",
    "n_top_classes",
    "range_of_top",
    "RANGE_GRID",
    "PREFLOP_VS_TOP",
    "snap_to_grid",
    "compute_range_table",
    "equity_vs_top",
]

_RANK_CHARS = "23456789TJQKA"  # index + 2 = rank value


def _all_classes() -> tuple[str, ...]:
    """AA, AKs, AKo, AQs, ..., 22: high rank first, then the second rank."""
    classes = []
    for hi in reversed(_RANK_CHARS):
        for lo in reversed(_RANK_CHARS[: _RANK_CHARS.index(hi) + 1]):
            classes += [hi + lo] if hi == lo else [hi + lo + "s", hi + lo + "o"]
    return tuple(classes)


ALL_CLASSES: tuple[str, ...] = _all_classes()
"""All 169 classes, e.g. ``"AA"``, ``"AKs"``, ``"AKo"``."""


def hand_class(cards: Sequence[Card]) -> str:
    """The class of two hole cards: ``As Kd -> "AKo"``, ``7h 7c -> "77"``."""
    a, b = sorted(cards, reverse=True)  # higher rank first
    name = a.rank.char + b.rank.char
    if a.rank == b.rank:
        return name
    return name + ("s" if a.suit == b.suit else "o")


@lru_cache(maxsize=None)
def class_combos(cls: str) -> tuple[Combo, ...]:
    """Every combo of a class: 6 for a pair, 4 suited, 12 offsuit."""
    hi, lo = Rank.from_char(cls[0]), Rank.from_char(cls[1])
    combos = []
    for s1 in Suit:
        for s2 in Suit:
            if hi == lo and s1 >= s2:
                continue  # a pair: each unordered pair of suits once
            if len(cls) == 3 and (s1 == s2) != (cls[2] == "s"):
                continue  # suited needs equal suits, offsuit different ones
            combos.append((Card(hi, s1), Card(lo, s2)))
    return tuple(combos)


def representative(cls: str) -> Combo:
    """One fixed combo of the class (the first in :func:`class_combos`)."""
    return class_combos(cls)[0]


def compute_preflop_table(samples: int = 100_000, seed: int = 0) -> dict[str, tuple[float, float]]:
    """``{class: (equity, std_error)}`` against a random hand, by Monte Carlo.

    Class number i uses seed ``seed + i``, so the table is reproducible.
    Takes about 169 * samples * 11 us (about 3 minutes for the default).

    Note on seed overlap (kept on purpose, see :func:`compute_range_table`):
    the streams ``seed + i`` here and ``seed + 100 * i + j`` there are not
    disjoint, so some entries of the two tables start from the same seed.
    """
    table = {}
    for i, cls in enumerate(ALL_CLASSES):
        r = equity_vs_range(representative(cls), RANDOM_RANGE, samples=samples, seed=seed + i)
        table[cls] = (r.equity, r.std_error)
    return table


PREFLOP_EQUITY: dict[str, float] = {cls: eq for cls, (eq, _) in PREFLOP_TABLE.items()}
"""Equity of each class against a uniformly random hand."""

PREFLOP_SE: dict[str, float] = {cls: se for cls, (_, se) in PREFLOP_TABLE.items()}
"""Monte Carlo standard error of each entry of :data:`PREFLOP_EQUITY`."""

RANKING: tuple[str, ...] = tuple(sorted(PREFLOP_EQUITY, key=lambda c: (-PREFLOP_EQUITY[c], c)))
"""The classes, best first (by equity against a random hand)."""


def top_classes(fraction: float) -> tuple[str, ...]:
    """The best classes that together hold at least ``fraction`` of all 1,326 combos.

    Always at least one class (AA). ``fraction >= 1`` gives all 169.
    Example: ``top_classes(0.01)`` is ``("AA", "KK", "QQ")``: AA and KK are
    12 combos (0.9%), not yet 1%, so QQ is added (18 combos, 1.4%).
    """
    return RANKING[: n_top_classes(fraction)]


def top_range(fraction: float) -> tuple[Combo, ...]:
    """Every combo of :func:`top_classes`, best class first."""
    return range_of_top(n_top_classes(fraction))


def n_top_classes(fraction: float) -> int:
    """How many classes :func:`top_classes` takes for ``fraction`` (1 to 169)."""
    target = fraction * len(RANDOM_RANGE)
    total = 0
    for n, cls in enumerate(RANKING, start=1):
        total += len(class_combos(cls))
        if total >= target:
            return n
    return len(RANKING)


@lru_cache(maxsize=None)
def range_of_top(n_classes: int) -> tuple[Combo, ...]:
    """Every combo of the best ``n_classes`` classes, best class first."""
    return tuple(combo for cls in RANKING[:n_classes] for combo in class_combos(cls))


RANGE_GRID: tuple[float, ...] = (
    0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.0,
)
"""The range widths ("top x%") that bots may assume. A fixed, small set, so the
preflop equity against every one of them can be precomputed (see
:data:`PREFLOP_VS_TOP`) and postflop simulations reuse a few ranges."""


def snap_to_grid(fraction: float) -> float:
    """The largest :data:`RANGE_GRID` width ``<= fraction`` (at least the smallest, 1%).

    Rounding *down* means a bot assumes a slightly stronger range than its
    model says, which errs on the side of caution.
    """
    return max([w for w in RANGE_GRID if w <= fraction], default=RANGE_GRID[0])


def compute_range_table(samples: int = 5_000, seed: int = 0) -> dict[str, tuple[tuple[float, float], ...]]:
    """``{class: ((equity, se) vs top RANGE_GRID[j] for each j except 100%)}``, by Monte Carlo.

    Uses the class's :func:`representative` combo. That gives the right
    answer (up to noise) for every combo of the class, because a top-x%
    range contains whole classes, so it looks the same under any relabelling
    of the suits. Class i, grid point j uses seed ``seed + 100 * i + j``.
    With 5,000 samples each SE is at most 0.5 / sqrt(5,000) = 0.71 points.
    Takes about 169 * 15 * samples * 11 us (about 2.5 minutes).

    Known flaw: overlapping seeds (documented, deliberately not fixed).
    Within this table the seeds ``seed + 100 * i + j`` are distinct (only
    because j < 15 < 100), but they overlap :func:`compute_preflop_table`'s
    ``seed + i``: e.g. class 1, grid point 0 here uses ``seed + 100``, the
    same seed as class 100 of the random-hand table, and class 0, grid
    point 0 uses ``seed`` itself, the same as class 0 there. The committed tables were
    made with ``seed = 20261116`` (``examples/make_preflop_table.py``), and
    that is also the Phase 4 master seed (``examples/run_phase4.py``).
    Changing any of this would regenerate the tables and so change every
    preflop decision and the official Phase 4 results, so it stays. Why it
    does no measurable harm:

    - The tables are *constants*. Every bot and every match uses the same
      numbers, so an error in an entry is a fixed (tiny, SE <= 0.71 points)
      offset, not a random effect that could correlate with the cards dealt.
    - Two entries sharing a seed estimate *different* quantities (different
      hands, different villain ranges), so their samples are consumed
      differently from the first draw on: the same starting state does not
      make the two estimates equal, only (very weakly) dependent.
    - The match derives its deals and bot seeds from
      ``random.Random(master_seed).getrandbits(64)``, a different use of the
      stream again; the deals it produces are ordinary pseudo-random deals.

    For future generators: derive each stream's seed by hashing its purpose
    and indices, e.g. ``int.from_bytes(hashlib.sha256(f"vs_top/{seed}/{i}/{j}"
    .encode()).digest()[:8], "big")``, so streams can never collide; and
    choose match master seeds that differ from every table seed.
    """
    table = {}
    for i, cls in enumerate(ALL_CLASSES):
        row = []
        for j, width in enumerate(RANGE_GRID[:-1]):
            r = equity_vs_range(representative(cls), top_range(width), samples=samples, seed=seed + 100 * i + j)
            row.append((r.equity, r.std_error))
        table[cls] = tuple(row)
    return table


def equity_vs_top(cls: str, width: float) -> tuple[float, float]:
    """``(equity, std_error)`` of class ``cls`` against the top ``width`` of hands, preflop.

    ``width`` must be a :data:`RANGE_GRID` value. A table lookup: 100% is
    :data:`PREFLOP_EQUITY`, the rest is :data:`PREFLOP_VS_TOP`.
    """
    j = RANGE_GRID.index(width)  # ValueError if width is not on the grid
    if width == 1.0:
        return PREFLOP_EQUITY[cls], PREFLOP_SE[cls]
    return PREFLOP_VS_TOP[cls][j]


def _check_tables() -> None:
    # A cheap guard against a corrupted or half-written data file.
    if PREFLOP_TABLE and set(PREFLOP_TABLE) != set(ALL_CLASSES):
        raise RuntimeError("preflop_data.PREFLOP_TABLE must have all 169 classes")
    if PREFLOP_VS_TOP and (
        set(PREFLOP_VS_TOP) != set(ALL_CLASSES)
        or any(len(row) != len(RANGE_GRID) - 1 for row in PREFLOP_VS_TOP.values())
    ):
        raise RuntimeError("preflop_range_data.PREFLOP_VS_TOP must have 169 rows of 15 entries")


_check_tables()
