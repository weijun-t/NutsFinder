"""Multiway Monte Carlo equity: my hand against k opponents, each with its own range.

What is estimated
-----------------
Each opponent ``i`` holds a hand from a *range*: a weight per combo,
``w_i[c] >= 0`` over the 1,326 combos (a uniformly random hand is all ones;
"the top 30%" is 1 on those combos and 0 elsewhere). The hands are dealt
jointly, so they must not share cards with each other, with my hand or
with the board::

    P(h_1, ..., h_k) ∝ w_1[h_1] ... w_k[h_k] × 1[all 2k + 2 + |board| cards distinct]

That is "independent ranges, conditioned on a legal deal". Then the rest
of the board is dealt uniformly from the cards nobody holds. My payoff in
one deal is my **pot share**: 1 if I hold the single best hand, ``1/t`` if
I tie for best with ``t - 1`` opponents, 0 otherwise. Equity is its mean,
and the standard error is the sample standard deviation over sqrt(n),
exactly as in :mod:`nutsfinder.engine.equity` (a split counts as a partial
win, so ties reduce the variance below the binomial ``p(1 - p)``).

How it is sampled (vectorised)
------------------------------
Samples are drawn in chunks of ``CHUNK`` rows with numpy:

1. **Opponent hands.** For every opponent, draw a combo per row from its
   range (inverse-CDF on the cumulative weights, with the combos that touch
   my cards or the board given weight 0). Rows in which two opponents
   share a card are **redrawn whole** (all opponents), until no row
   conflicts. Redrawing the whole row is plain rejection sampling, so the
   accepted rows follow the joint law above exactly. (Redrawing only the
   later opponent would be "sequential" sampling, which is exact for
   uniform ranges but slightly biased for narrow ones.) The redraw rounds
   continue as long as they make progress: they stop when no row is left,
   or after ``STALL_ROUNDS`` (25) consecutive rounds in which no row was
   accepted, and after at most ``MAX_REDRAWS`` (200) rounds. Each round redraws only the rows still
   in conflict, so this is cheap: eight random hands collide in about 81% of
   rows, 0.81^25 is about 0.5%, so a stall of 25 rounds with rows left
   essentially only happens when the acceptance rate is tiny. Rows still
   in conflict then (only when the ranges are tiny and overlap, e.g. three
   players all on {AA, KK} while I hold an ace) are dealt **sequentially**
   instead: each opponent from its range
   minus the cards already dealt in that row, or, if that leaves nothing,
   from any hand that is still possible. Those few rows are slightly
   biased (sequential sampling, plus widening an impossible range), but a
   decision never fails: with overlapping narrow ranges *some* opponent's
   assumed range must be wrong anyway.
2. **The run-out.** Each row gets 52 uniform keys; cards already held
   (mine, the board, the row's opponent cards) get key 2. The
   ``5 - |board|`` smallest keys are a uniform draw without replacement
   from the remaining deck (``argpartition``).
3. **Strengths.** Every 7-card hand is scored with the vectorised
   :func:`nutsfinder.abstraction.strength.strength7` (an integer whose
   order is the evaluator's order, equal means split). One call scores all
   ``(k + 1) × rows`` hands.

The result keeps the whole ``(n, k + 1)`` strength matrix (column 0 is me),
so any *subset* of the opponents can be scored afterwards without
resampling: :meth:`MultiwayDeals.share` gives my pot share against the
opponents in a subset. That is what side pots need: a side pot is
contested only by the players who covered it.

Speed: about 1.3 ms for 6,000 seven-card hands (1,000 samples 6-handed),
plus the sampling, so a 1,000-sample six-way estimate takes about 3 ms.

Exact enumeration (:func:`exact_multiway_equity`) does the same with known
opponent hands and every run-out, with the engine's scalar evaluator; it is
the oracle for the tests.

Preflop against k random hands
------------------------------
:func:`preflop_vs_random` reads a 169 × 7 table (k = 1..7 random
opponents), built by :func:`build_preflop_table` (100,000 samples per
entry by default, SE at most 0.16 points) and stored in
``data/preflop_vs_random.npz``. If the file is missing, entries are
computed lazily (``LAZY_SAMPLES`` each, with a seed derived from the class
and k, so the answer does not depend on the order of the calls).
Reference values [memory, approximate]: AA wins about 85% against one
random hand, 73% against two, 64% against three, 56% against four and 49%
against five; ``tests/test_multiway_bots.py`` checks the table and the
sampler against these.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from nutsfinder.abstraction.strength import COMBO_CARDS, N_COMBOS, combo_index, strength7, strength_tables
from nutsfinder.engine.cards import FULL_DECK, Card
from nutsfinder.engine.equity import Combo
from nutsfinder.engine.evaluator import evaluate
from nutsfinder.engine.preflop import ALL_CLASSES, hand_class, representative, top_range

__all__ = [
    "CHUNK",
    "MAX_REDRAWS",
    "STALL_ROUNDS",
    "RANDOM_WEIGHTS",
    "MultiwayDeals",
    "MultiwayEquity",
    "range_weights",
    "top_weights",
    "sample_deals",
    "multiway_equity",
    "exact_multiway_equity",
    "preflop_vs_random",
    "build_preflop_table",
    "PREFLOP_TABLE_FILE",
    "MAX_OPPONENTS",
    "warm_up",
]

CHUNK = 256
"""Rows per sampling chunk (the time budget is checked between chunks)."""

MAX_REDRAWS = 200
"""Upper bound on whole-row redraw rounds per chunk. Eight random hands: 0.81^200 is about 5e-19, so the
fallback never happens for wide ranges; for nearly infeasible ranges it bounds the work (a higher cap
let a decision take seconds when a row was accepted only every few rounds, so the stall rule never fired)."""

STALL_ROUNDS = 25
"""Stop redrawing after this many consecutive rounds that accept no row (see the module doc)."""

SEQUENTIAL_REDRAWS = 30
"""Redraw rounds for rows whose opponent hands collide (see the module doc)."""

MAX_OPPONENTS = 7

RANDOM_WEIGHTS: np.ndarray = np.ones(N_COMBOS)
RANDOM_WEIGHTS.flags.writeable = False
"""Every combo equally likely: a uniformly random hand."""

PREFLOP_TABLE_FILE = Path(__file__).resolve().parent / "data" / "preflop_vs_random.npz"
LAZY_SAMPLES = 20_000

_CLASS_INDEX = {cls: i for i, cls in enumerate(ALL_CLASSES)}


def _ids(cards: Sequence[Card]) -> list[int]:
    return [c.id for c in cards]


def range_weights(combos: Sequence[Combo]) -> np.ndarray:
    """Weights (0/1) over the 1,326 combos for a list of two-card combos."""
    w = np.zeros(N_COMBOS)
    for a, b in combos:
        w[combo_index(a.id, b.id)] = 1.0
    w.flags.writeable = False
    return w


@lru_cache(maxsize=None)
def top_weights(width: float) -> np.ndarray:
    """Weights of the top ``width`` of hands (preflop ranking, as the HU bots use).

    ``width = 1`` is :data:`RANDOM_WEIGHTS`. Cached per width (use grid widths).
    """
    if width >= 1.0:
        return RANDOM_WEIGHTS
    return range_weights(top_range(width))


@dataclass(frozen=True)
class MultiwayEquity:
    """``equity`` (mean pot share), ``std_error``, ``samples``, and P(win alone) / P(split)."""

    equity: float
    std_error: float
    samples: int
    win: float
    tie: float


class MultiwayDeals:
    """Sampled deals: strengths of me (column 0) and each opponent (columns 1..k).

    Attributes:
        strengths: ``(n, k + 1)`` int array.
        combos: ``(n, k)`` combo index of each opponent's hand.
    """

    def __init__(self, strengths: np.ndarray, combos: np.ndarray, fallback_rows: int = 0,
                 widened: tuple[int, ...] = ()) -> None:
        self.strengths = strengths
        self.combos = combos
        self.fallback_rows = fallback_rows
        """Rows dealt by the sequential fallback (the ranges could not be dealt jointly by rejection)."""
        self.widened = widened
        """Opponents (0-based) whose range was empty after card removal, so a random hand was used."""
        self._share_cache: dict[tuple[int, ...], np.ndarray] = {}

    @property
    def n(self) -> int:
        return int(self.strengths.shape[0])

    @property
    def k(self) -> int:
        return int(self.strengths.shape[1]) - 1

    def share(self, opponents: Sequence[int] | None = None) -> np.ndarray:
        """My pot share per sample against the opponents (0-based indices) in ``opponents``.

        ``None`` means all of them; an empty subset means I win the whole pot (1).
        """
        key = tuple(range(self.k)) if opponents is None else tuple(sorted(opponents))
        out = self._share_cache.get(key)
        if out is None:
            me = self.strengths[:, 0]
            if not key:
                out = np.ones(self.n)
            else:
                others = self.strengths[:, [i + 1 for i in key]]
                best = others.max(axis=1)
                tied = (others == me[:, None]).sum(axis=1)
                out = np.where(me > best, 1.0, np.where(me == best, 1.0 / (1 + tied), 0.0))
            self._share_cache[key] = out
        return out

    def share_mean(self, opponents: Sequence[int] | None = None) -> float:
        return float(self.share(opponents).mean()) if self.n else 0.0

    def equity(self, opponents: Sequence[int] | None = None) -> MultiwayEquity:
        """Mean share, its standard error, and P(win alone), P(split) against ``opponents``."""
        s = self.share(opponents)
        n = len(s)
        if n == 0:
            raise ValueError("no samples")
        se = float(s.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
        return MultiwayEquity(float(s.mean()), se, n, float((s == 1.0).mean()), float(((s > 0) & (s < 1)).mean()))


def _validate(hero: Sequence[Card], board: Sequence[Card]) -> None:
    if len(hero) != 2:
        raise ValueError(f"hero needs exactly 2 hole cards, got {len(hero)}")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"board must hold 0, 3, 4 or 5 cards, got {len(board)}")
    known = [*hero, *board]
    if len(set(known)) != len(known):
        raise ValueError(f"cards must all be distinct: {[str(c) for c in known]}")


def _cdfs(ranges: Sequence[np.ndarray | None], dead: list[int]) -> tuple[list[np.ndarray], tuple[int, ...]]:
    """Cumulative weights per opponent with the combos that touch dead cards removed.

    A range that my cards and the board empty completely (e.g. "top 1%" =
    AA-QQ on a board holding the other aces and kings) is replaced by a
    random hand; such opponents are returned as the second value.
    """
    touches = np.zeros(N_COMBOS, dtype=bool)
    if dead:
        d = np.zeros(52, dtype=bool)
        d[dead] = True
        touches = d[COMBO_CARDS[:, 0]] | d[COMBO_CARDS[:, 1]]
    out = []
    widened = []
    for i, w in enumerate(ranges):
        w = RANDOM_WEIGHTS if w is None else np.asarray(w, dtype=float)
        if w.shape != (N_COMBOS,) or (w < 0).any():
            raise ValueError(f"range {i} must be {N_COMBOS} non-negative weights")
        cdf = np.cumsum(np.where(touches, 0.0, w))
        if cdf[-1] <= 0:
            cdf = np.cumsum(np.where(touches, 0.0, 1.0))
            widened.append(i)
        out.append(cdf)
    return out, tuple(widened)


def _draw(cdfs: list[np.ndarray], m: int, rng: np.random.Generator) -> np.ndarray:
    """``(m, k)`` combo indices, one inverse-CDF draw per opponent."""
    out = np.empty((m, len(cdfs)), dtype=np.int64)
    for i, cdf in enumerate(cdfs):
        u = rng.random(m) * cdf[-1]
        out[:, i] = np.minimum(np.searchsorted(cdf, u, side="right"), N_COMBOS - 1)
    return out


def _conflicts(idx: np.ndarray) -> np.ndarray:
    """Rows in which two opponents share a card."""
    if idx.shape[1] < 2:
        return np.zeros(len(idx), dtype=bool)
    cards = np.sort(COMBO_CARDS[idx].reshape(len(idx), -1), axis=1)
    return (cards[:, 1:] == cards[:, :-1]).any(axis=1)


_CARD_IN_COMBO = np.zeros((52, N_COMBOS), dtype=bool)
_CARD_IN_COMBO[COMBO_CARDS[:, 0], np.arange(N_COMBOS)] = True
_CARD_IN_COMBO[COMBO_CARDS[:, 1], np.arange(N_COMBOS)] = True


def _sequential(cdfs: list[np.ndarray], known: list[int], m: int, rng: np.random.Generator) -> np.ndarray:
    """Fallback for rows rejection could not fill: deal the opponents one after another.

    Opponent i is drawn from its range restricted to combos that avoid the
    cards of opponents 0..i-1 in that row; if nothing is left, from every
    combo that avoids all cards held so far (a random hand). Slightly biased
    for narrow ranges (module doc); only used when the whole-row redraws stall.
    """
    weights = [np.diff(c, prepend=0.0) for c in cdfs]
    blocked = np.zeros(N_COMBOS, dtype=bool)
    for x in known:
        blocked |= _CARD_IN_COMBO[x]
    used = np.broadcast_to(blocked, (m, N_COMBOS)).copy()      # (m, 1326) per-row blocked combos
    out = np.empty((m, len(cdfs)), dtype=np.int64)
    rows = np.arange(m)
    for i, (cdf, w) in enumerate(zip(cdfs, weights)):
        # Cheap first: redraw opponent i alone where it hits a card already dealt in its row.
        c = _draw([cdf], m, rng)[:, 0]
        bad = used[rows, c]
        for _ in range(SEQUENTIAL_REDRAWS):
            if not bad.any():
                break
            r = np.flatnonzero(bad)
            c[r] = _draw([cdf], len(r), rng)[:, 0]
            bad[r] = used[r, c[r]]
        if bad.any():  # exact masked draw for the rest (widened to any possible hand if empty)
            r = np.flatnonzero(bad)
            ww = np.where(used[r], 0.0, w)
            empty = ww.sum(axis=1) <= 0
            if empty.any():
                ww[empty] = np.where(used[r][empty], 0.0, 1.0)
            cum = np.cumsum(ww, axis=1)
            u = rng.random(len(r)) * cum[:, -1]
            c[r] = np.minimum((cum <= u[:, None]).sum(axis=1), N_COMBOS - 1)
        out[:, i] = c
        used |= _CARD_IN_COMBO[COMBO_CARDS[c, 0]] | _CARD_IN_COMBO[COMBO_CARDS[c, 1]]
    return out


def _sample_chunk(hero: list[int], board: list[int], cdfs: list[np.ndarray], m: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    idx = _draw(cdfs, m, rng)
    bad = _conflicts(idx)
    stalled = 0
    for _ in range(MAX_REDRAWS):
        n_bad = int(bad.sum())
        if not n_bad:
            break
        rows = np.flatnonzero(bad)
        idx[rows] = _draw(cdfs, len(rows), rng)
        bad[rows] = _conflicts(idx[rows])
        stalled = stalled + 1 if int(bad.sum()) == n_bad else 0
        if stalled >= STALL_ROUNDS:
            break  # nothing accepted for a while: the ranges (almost) cannot be dealt together
    n_fallback = int(bad.sum())
    if n_fallback:
        rows = np.flatnonzero(bad)
        idx[rows] = _sequential(cdfs, hero + board, len(rows), rng)
    m = len(idx)
    k = len(cdfs)
    opp_cards = COMBO_CARDS[idx].reshape(m, 2 * k)                    # (m, 2k)
    need = 5 - len(board)
    if need:
        keys = rng.random((m, 52))
        keys[:, hero + board] = 2.0
        np.put_along_axis(keys, opp_cards, 2.0, axis=1)
        runout = np.argpartition(keys, need - 1, axis=1)[:, :need]    # (m, need)
    else:
        runout = np.empty((m, 0), dtype=np.int64)
    full_board = np.hstack([np.broadcast_to(np.array(board, dtype=np.int64), (m, len(board))), runout])
    hands = np.empty((m, k + 1, 7), dtype=np.int64)
    hands[:, 0, :2] = hero
    hands[:, 1:, :2] = opp_cards.reshape(m, k, 2)
    hands[:, :, 2:] = full_board[:, None, :]
    s = strength7(hands.reshape(-1, 7)).reshape(m, k + 1)
    return s, idx, n_fallback


def sample_deals(
    hero: Sequence[Card],
    board: Sequence[Card],
    ranges: Sequence[np.ndarray | None],
    samples: int,
    rng: np.random.Generator,
    deadline: float | None = None,
) -> MultiwayDeals:
    """Sample ``samples`` deals against one range per opponent (``None`` = random hand).

    ``deadline`` (a ``time.perf_counter()`` value) stops sampling between
    chunks once passed; at least one chunk is always drawn. With a deadline
    the result depends on machine speed, so leave it ``None`` for experiments.
    """
    hero, board = list(hero), list(board)
    _validate(hero, board)
    if not 1 <= len(ranges) <= MAX_OPPONENTS:
        raise ValueError(f"need 1..{MAX_OPPONENTS} opponent ranges, got {len(ranges)}")
    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")
    h, b = _ids(hero), _ids(board)
    cdfs, widened = _cdfs(ranges, h + b)
    strengths, combos = [], []
    fallback = 0
    left = samples
    while left > 0:
        m = min(CHUNK, left)
        s, idx, nf = _sample_chunk(h, b, cdfs, m, rng)
        fallback += nf
        strengths.append(s)
        combos.append(idx)
        left -= m
        if deadline is not None and time.perf_counter() > deadline:
            break
    return MultiwayDeals(np.vstack(strengths), np.vstack(combos), fallback, widened)


def multiway_equity(
    hero: Sequence[Card],
    board: Sequence[Card] = (),
    ranges: Sequence[np.ndarray | None] | int = 1,
    samples: int = 10_000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> MultiwayEquity:
    """Equity against ``ranges`` (or an int k: k random hands); see the module doc."""
    if isinstance(ranges, int):
        ranges = [None] * ranges
    rng = rng if rng is not None else np.random.Generator(np.random.PCG64(seed))
    return sample_deals(hero, board, ranges, samples, rng).equity()


def exact_multiway_equity(hero: Sequence[Card], opponents: Sequence[Sequence[Card]],
                          board: Sequence[Card]) -> float:
    """My exact pot share against *known* opponent hands, over every run-out (scalar evaluator).

    Needs at least a flop (preflop has C(48 - 2k, 5) run-outs: too many for a test oracle).
    """
    hero, board = list(hero), list(board)
    if len(board) < 3:
        raise ValueError("exact enumeration needs at least the flop")
    known = [*hero, *board, *(c for h in opponents for c in h)]
    if len(set(known)) != len(known):
        raise ValueError("cards must all be distinct")
    stub = [c for c in FULL_DECK if c not in set(known)]
    total, count = 0.0, 0
    for extra in itertools.combinations(stub, 5 - len(board)):
        full = board + list(extra)
        me = evaluate(hero + full)
        ranks = [evaluate(list(h) + full) for h in opponents]
        best = max(ranks)
        if me > best:
            total += 1.0
        elif me == best:
            total += 1.0 / (1 + ranks.count(me))
        count += 1
    return total / count


# --- Preflop table vs k random hands -----------------------------------------------


def _class_seed(cls: str, k: int, seed: int) -> int:
    return seed * 1_000_003 + _CLASS_INDEX[cls] * 11 + k


def build_preflop_table(samples: int = 100_000, seed: int = 20261124,
                        classes: Sequence[str] = ALL_CLASSES) -> tuple[np.ndarray, np.ndarray]:
    """``(equity, se)``, each ``(len(classes), 7)``: class vs k = 1..7 random hands.

    Uses each class's representative combo (suits cannot matter against
    random hands). Each entry has its own seed (from the class, k and
    ``seed``), so entries can be built in any order or in parallel.
    """
    eq = np.zeros((len(classes), MAX_OPPONENTS))
    se = np.zeros((len(classes), MAX_OPPONENTS))
    for i, cls in enumerate(classes):
        hero = representative(cls)
        for k in range(1, MAX_OPPONENTS + 1):
            rng = np.random.Generator(np.random.PCG64(_class_seed(cls, k, seed)))
            r = sample_deals(hero, (), [None] * k, samples, rng).equity()
            eq[i, k - 1], se[i, k - 1] = r.equity, r.std_error
    return eq, se


@lru_cache(maxsize=1)
def _preflop_file() -> dict[str, tuple[float, float]] | None:
    if not PREFLOP_TABLE_FILE.is_file():
        return None
    with np.load(PREFLOP_TABLE_FILE) as z:
        classes = [str(c) for c in z["classes"]]
        eq, se = z["equity"], z["se"]
    return {f"{cls}/{k}": (float(eq[i, k - 1]), float(se[i, k - 1]))
            for i, cls in enumerate(classes) for k in range(1, MAX_OPPONENTS + 1)}


_LAZY: dict[str, tuple[float, float]] = {}


def preflop_vs_random(cards_or_class: Sequence[Card] | str, k: int) -> tuple[float, float]:
    """``(equity, se)`` of a hand (or class) against ``k`` random hands, preflop (table lookup)."""
    if not 1 <= k <= MAX_OPPONENTS:
        raise ValueError(f"k must be 1..{MAX_OPPONENTS}, got {k}")
    cls = cards_or_class if isinstance(cards_or_class, str) else hand_class(cards_or_class)
    key = f"{cls}/{k}"
    table = _preflop_file()
    if table is not None:
        return table[key]
    if key not in _LAZY:
        rng = np.random.Generator(np.random.PCG64(_class_seed(cls, k, 7)))
        r = sample_deals(representative(cls), (), [None] * k, LAZY_SAMPLES, rng).equity()
        _LAZY[key] = (r.equity, r.std_error)
    return _LAZY[key]


def warm_up() -> None:
    """Build the evaluator tables now (about 0.6 s, once per process) instead of in the first decision."""
    strength_tables()

