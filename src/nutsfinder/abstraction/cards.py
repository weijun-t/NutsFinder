"""Card abstraction: hole cards + board -> bucket.

Preflop: the 169 hand classes
-----------------------------
Before the flop only "same suit or not" matters, so the bucket is the index
of :func:`~nutsfinder.engine.preflop.hand_class` in
:data:`~nutsfinder.engine.preflop.ALL_CLASSES` (AA = 0, AKs = 1, AKo = 2, ...,
22 = 168). Lossless: two hands with the same class play identically.

Postflop: equal-frequency E[HS] buckets, one street at a time
-------------------------------------------------------------
E[HS](h, board) is the equity of h against a uniformly random opponent hand
(card removal applied), averaged over the run-outs: on the river it is the
plain hand strength HS of :mod:`~nutsfinder.abstraction.strength`. The
bucket on a street is the percentile group of E[HS] on that street: the
edges (in :mod:`~nutsfinder.abstraction.bucket_data`) cut E[HS] of ~1e5
random (hole, board) pairs into B groups of equal size, so each bucket holds
about 1/B of all deals. Imperfect recall: a flop bucket says nothing about
the preflop class, a turn bucket nothing about the flop bucket.

Exact E[HS], by enumeration (``ehs_method="exact"``, the default)
-----------------------------------------------------------------
Monte Carlo E[HS] is noisy: with 1,000 samples the standard error is about
0.014, while 50 flop buckets are on average 0.017 wide. Measured with
``examples/bucket_noise.py`` (300 random flops): at 1,000 samples 59% of
hands get a different bucket from the exact one (and 67% change bucket
between two seeds); even at 5,000 samples (~40 ms per lookup) it is 36%.
So every E[HS] here is computed *exactly*, for every hand on a board at once:

- **River** (5 cards): :func:`~nutsfinder.abstraction.strength.river_hs`,
  all 1,081 hands, ~0.15 ms (this laptop throttles to ~1 ms under sustained load).
- **Turn** (4 cards): average the river table over the 48 river cards.
  Hand h is valid for the 46 rivers that avoid it, and every (opponent,
  river) pair is equally likely, so E[HS](h) = mean of HS(h | board + r)
  over those 46 rivers. ~7 ms per board (~50 ms throttled).
- **Flop** (3 cards): likewise the mean of the turn tables over the 47 turn
  cards that avoid h (the turn tables are themselves uniform averages over
  rivers, so this is the uniform average over (turn, river) pairs).

Why the averaging is right: the uniform distribution over valid (opponent,
run-out) pairs can be generated "run-out first, then opponent", because
every run-out leaves the same number C(45, 2) = 990 of opponent hands. So
E[HS] = E_runout[HS(h | full board)].

Suit isomorphism and the caches
-------------------------------
Relabelling suits changes nothing: E[HS](sigma h, sigma B) = E[HS](h, B) for
any of the 24 suit permutations sigma. So a table is stored only for the
**canonical** board: the lexicographically smallest sorted tuple of card ids
among the 24 relabellings. There are 1,755 canonical flops and 16,432
canonical turn boards (out of 22,100 and 270,725). For a real board B with
canonical C = sigma(B), E[HS](h, B) = table_C[sigma(h)].

Worked example: Kc 7d 2h (ids 44, 21, 2). Relabel hearts -> clubs and
clubs -> hearts: 2c 7d Kh, ids (0, 21, 46). No flop can start below 2c (id
0), and with clubs taken by the deuce the seven's smallest suit is diamonds,
so this is the minimum of the 24 relabellings. A hand is mapped with the
same relabelling: Ah Ac on that flop is looked up as Ac Ah in the table.

Cost: a turn table is computed once per canonical turn board (~8 ms, then a
dict hit), a flop table once per canonical flop (~0.4 s from cold turn
tables, or loaded from ``data/flop_ehs.npz``, see
``examples/make_bucket_edges.py``). Memory when every turn table is cached:
16,432 x 1,326 uint16 = 43 MB; the flop file is 1,755 x 1,326 uint16.

Monte Carlo E[HS] (``ehs_method="mc"``, for comparison)
-------------------------------------------------------
:func:`ehs_mc` samples (opponent, run-out) pairs with a numpy generator
seeded by hashing the *suit-canonical* (hand, board) (the recipe of the
note in :mod:`nutsfinder.engine.preflop`), so a given hand always gets the
same estimate and isomorphic hands get identical ones.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable, Sequence
from itertools import permutations
from pathlib import Path

import numpy as np

from nutsfinder.abstraction import strength
from nutsfinder.abstraction.config import DEFAULT_CONFIG, AbstractionConfig
from nutsfinder.abstraction.strength import COMBO_CARDS, N_COMBOS, combo_index
from nutsfinder.engine.cards import Card
from nutsfinder.engine.preflop import ALL_CLASSES

__all__ = [
    "SUIT_PERMS", "stream_seed", "canonical_board", "canonical_hand_board",
    "preflop_bucket", "EHSTables", "ehs_mc", "equal_frequency_edges",
    "sample_deals", "CardAbstraction", "FLOP_FILE", "EHS_SCALE", "quantize",
    "dequantize", "turn_table_exact", "ehs_all",
]

SUIT_PERMS: tuple[tuple[int, int, int, int], ...] = tuple(permutations(range(4)))
"""The 24 suit relabellings; ``perm[s]`` is the new suit of suit s."""

_PERM_CARD: list[list[int]] = [[(c & ~3) | p[c & 3] for c in range(52)] for p in SUIT_PERMS]
_PERM_COMBO = np.array([[combo_index(pc[a], pc[b]) for a, b in COMBO_CARDS.tolist()]
                        for pc in _PERM_CARD], dtype=np.int64)
"""``_PERM_COMBO[k][h]``: index of combo h after relabelling suits by ``SUIT_PERMS[k]``."""

FLOP_FILE = Path(__file__).resolve().parent / "data" / "flop_ehs.npz"
"""Optional precomputed exact flop E[HS] for every canonical flop."""


def stream_seed(*parts: object) -> int:
    """A 64-bit seed from the purpose and indices of a random stream.

    ``sha256("/".join(parts))``, first 8 bytes: distinct purposes or indices
    can never share a stream (the overlap problem noted in
    :func:`nutsfinder.engine.preflop.compute_range_table`).

    >>> stream_seed("ehs_mc", 0, "1-2/3-4-5") == stream_seed("ehs_mc", 0, "1-2/3-4-5")
    True
    """
    text = "/".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


# --- Suit canonicalisation --------------------------------------------------

_CANON_CACHE: dict[tuple[int, ...], tuple[tuple[int, ...], int]] = {}
_CANON_CACHE_MAX = 1 << 19


def canonical_board(board: Sequence[int]) -> tuple[tuple[int, ...], int]:
    """``(canonical board, k)``: the smallest relabelled sorted board and a perm achieving it.

    ``SUIT_PERMS[k]`` maps the real board onto the canonical one; map the
    hand with the same k. Cached (boards are few on flop and turn).

    >>> canonical_board([44, 21, 2])[0]     # Kc 7d 2h -> 2c 7d Kh
    (0, 21, 46)
    """
    key = tuple(sorted(board))
    hit = _CANON_CACHE.get(key)
    if hit is not None:
        return hit
    best, best_k = None, 0
    for k, pc in enumerate(_PERM_CARD):
        cand = tuple(sorted([pc[c] for c in key]))
        if best is None or cand < best:
            best, best_k = cand, k
    if len(_CANON_CACHE) >= _CANON_CACHE_MAX:
        _CANON_CACHE.clear()  # a crude bound; the river is the only street that can fill it
    _CANON_CACHE[key] = (best, best_k)
    return best, best_k


def canonical_hand_board(hole: Sequence[int], board: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The smallest relabelling of the *pair* (sorted board, sorted hole): an isomorphism-class name."""
    best = None
    for pc in _PERM_CARD:
        cand = (tuple(sorted([pc[c] for c in board])), tuple(sorted([pc[c] for c in hole])))
        if best is None or cand < best:
            best = cand
    assert best is not None
    return best[1], best[0]


# --- Preflop ------------------------------------------------------------------

_CLASS_INDEX = {cls: i for i, cls in enumerate(ALL_CLASSES)}
_RANK_CHARS = "23456789TJQKA"


def _class_of_ids(a: int, b: int) -> str:
    ra, rb = a >> 2, b >> 2
    if ra < rb:
        ra, rb = rb, ra
    name = _RANK_CHARS[ra] + _RANK_CHARS[rb]
    if ra == rb:
        return name
    return name + ("s" if (a & 3) == (b & 3) else "o")


_PREFLOP_BUCKET: list[list[int]] = [[_CLASS_INDEX[_class_of_ids(a, b)] if a != b else -1
                                     for b in range(52)] for a in range(52)]


def preflop_bucket(hole: Sequence[int]) -> int:
    """Index of the hand class in ``ALL_CLASSES``: As Ad -> 0 ("AA"), 2c 2d -> 168 ("22")."""
    return _PREFLOP_BUCKET[hole[0]][hole[1]]


# --- Exact E[HS] tables -------------------------------------------------------

EHS_SCALE = 65534
"""Tables store round(E[HS] * 65534) as uint16 (step 1.5e-5, far below any
bucket width); 65535 marks a combo that touches the board. Every path
(computed in memory or loaded from the file) goes through the same
quantisation, so the bucket of a hand never depends on where its table came from."""
_INVALID = 65535


def quantize(ehs: np.ndarray) -> np.ndarray:
    """float E[HS] (NaN = invalid) -> uint16 table."""
    q = np.full(ehs.shape, _INVALID, dtype=np.uint16)
    ok = ~np.isnan(ehs)
    q[ok] = np.rint(ehs[ok] * EHS_SCALE).astype(np.uint16)
    return q


def dequantize(q: np.ndarray) -> np.ndarray:
    """uint16 table -> float E[HS] (NaN = invalid)."""
    out = q.astype(np.float64) / EHS_SCALE
    out[q == _INVALID] = np.nan
    return out


def turn_table_exact(canon: tuple[int, ...]) -> np.ndarray:
    """Quantised exact turn E[HS] of every combo on a (canonical) 4-card board."""
    return quantize(_average_next_street(canon, strength.river_hs))


class EHSTables:
    """Exact E[HS] for every hand on a board, cached per canonical board (see module doc).

    One instance per process is enough (the tables are pure functions of the
    board); the caches are instance attributes, not hidden globals.
    ``flop`` and ``turn`` map a canonical board to a uint16 table over the
    1,326 combos (in canonical suits); the river is computed per real board
    and kept in a small LRU (the two players of a deal share the board).
    """

    def __init__(self, flop_file: Path | None = FLOP_FILE, river_cache: int = 256) -> None:
        self.flop: dict[tuple[int, ...], np.ndarray] = {}
        self.turn: dict[tuple[int, ...], np.ndarray] = {}
        self._river: OrderedDict[tuple[int, ...], np.ndarray] = OrderedDict()
        self._river_max = river_cache
        if flop_file is not None and Path(flop_file).exists():
            self.load_flops(Path(flop_file))

    # -- tables per canonical board
    def turn_table(self, canon: tuple[int, ...]) -> np.ndarray:
        """Quantised E[HS] of all combos on canonical turn board ``canon``."""
        t = self.turn.get(canon)
        if t is None:
            t = turn_table_exact(canon)
            self.turn[canon] = t
        return t

    def flop_table(self, canon: tuple[int, ...]) -> np.ndarray:
        """Quantised E[HS] of all combos on canonical flop ``canon`` (mean of 49 turn tables)."""
        t = self.flop.get(canon)
        if t is None:
            t = quantize(_average_next_street(canon, self._turn_real))
            self.flop[canon] = t
        return t

    def river_table(self, board: Sequence[int]) -> np.ndarray:
        """Quantised HS of all combos on a real 5-card board."""
        key = tuple(sorted(board))
        t = self._river.get(key)
        if t is None:
            t = quantize(strength.river_hs(key))
            self._river[key] = t
            if len(self._river) > self._river_max:
                self._river.popitem(last=False)
        return t

    def _turn_real(self, board: Sequence[int]) -> np.ndarray:
        """Float turn table of a real board (indexed by real combos)."""
        canon, k = canonical_board(board)
        return dequantize(self.turn_table(canon)[_PERM_COMBO[k]])

    # -- the lookup
    def ehs(self, hole: Sequence[int], board: Sequence[int]) -> float:
        """Exact E[HS] of ``hole`` (2 ids) on ``board`` (3, 4 or 5 ids), on the 1/65534 grid."""
        n = len(board)
        if n == 5:
            q = self.river_table(board)[combo_index(hole[0], hole[1])]
        elif n in (3, 4):
            canon, k = canonical_board(board)
            pc = _PERM_CARD[k]
            table = self.flop_table(canon) if n == 3 else self.turn_table(canon)
            q = table[combo_index(pc[hole[0]], pc[hole[1]])]
        else:
            raise ValueError(f"board must have 3, 4 or 5 cards, got {n}")
        if q == _INVALID:
            raise ValueError("hole cards overlap the board")
        return int(q) / EHS_SCALE

    # -- bulk filling and persistence of the flop tables
    @staticmethod
    def all_canonical_flops() -> list[tuple[int, ...]]:
        """The 1,755 canonical flops, sorted."""
        return sorted({canonical_board(f)[0] for f in _all_boards(3)})

    @staticmethod
    def all_canonical_turns() -> list[tuple[int, ...]]:
        """The 16,432 canonical turn boards, sorted (~2 s)."""
        return sorted({canonical_board(b)[0] for b in _all_boards(4)})

    def compute_all_flops(self, progress: Callable[[int, int], None] | None = None) -> None:
        """Fill :attr:`flop` for every canonical flop (needs every turn table: ~2.5 min cold)."""
        flops = self.all_canonical_flops()
        for i, f in enumerate(flops):
            self.flop_table(f)
            if progress is not None:
                progress(i + 1, len(flops))

    def save_flops(self, path: Path) -> None:
        """Write the flop tables to an ``.npz`` file (boards int8, tables uint16)."""
        _save(self.flop, path)

    def load_flops(self, path: Path) -> None:
        """Read flop tables written by :meth:`save_flops`."""
        self.flop.update(_load(path))

    def save_turns(self, path: Path) -> None:
        """Write the cached turn tables (all 16,432 = 43 MB raw) for reuse by other processes."""
        _save(self.turn, path)

    def load_turns(self, path: Path) -> None:
        """Read turn tables written by :meth:`save_turns`."""
        self.turn.update(_load(path))


def _save(tables: dict[tuple[int, ...], np.ndarray], path: Path) -> None:
    boards = sorted(tables)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, boards=np.array(boards, dtype=np.int8),
                        ehs=np.stack([tables[b] for b in boards]))


def _load(path: Path) -> dict[tuple[int, ...], np.ndarray]:
    with np.load(path) as data:
        return {tuple(b): row for b, row in zip(data["boards"].tolist(), data["ehs"])}


def _average_next_street(board: tuple[int, ...], table_of: Callable[[Sequence[int]], np.ndarray]) -> np.ndarray:
    """Mean over every next card c of ``table_of(board + c)``, per combo (NaN-aware).

    A combo that avoids the board is valid for every c it does not hold
    (52 - len(board) - 2 of them), and invalid (NaN) for the 2 it holds.
    """
    acc = np.zeros(N_COMBOS)
    for c in range(52):
        if c in board:
            continue
        acc += np.nan_to_num(table_of((*board, c)), nan=0.0)
    out = acc / (52 - len(board) - 2)
    out[np.isin(COMBO_CARDS, board).any(axis=1)] = np.nan
    return out


def _all_boards(n: int) -> list[tuple[int, ...]]:
    from itertools import combinations
    return list(combinations(range(52), n))


# --- Monte Carlo E[HS] --------------------------------------------------------


def ehs_mc(hole: Sequence[int], board: Sequence[int], samples: int, salt: object = 0) -> float:
    """Monte Carlo E[HS]: ``samples`` random (opponent hand, run-out) pairs, vectorised.

    The generator is seeded with :func:`stream_seed` of the suit-canonical
    (hand, board) and ``salt``, so the same hand on the same board (in any
    suit relabelling) always gives the same number. On the river the
    opponent hand is the only random part.
    """
    canon_hole, canon_board = canonical_hand_board(hole, board)
    rng = np.random.Generator(np.random.PCG64(stream_seed("ehs_mc", salt, canon_hole, canon_board)))
    known = set(canon_hole) | set(canon_board)
    stub = np.array([c for c in range(52) if c not in known], dtype=np.int64)
    need = 5 - len(canon_board)
    # The first 2 + need entries of a random permutation of the stub: the
    # opponent's two cards, then the run-out (a uniform draw of distinct cards).
    picks = stub[np.argsort(rng.random((samples, len(stub))), axis=1)[:, :2 + need]]
    fixed = np.broadcast_to(np.array(canon_board, dtype=np.int64), (samples, len(canon_board)))
    full = np.concatenate([fixed, picks[:, 2:]], axis=1)
    hero = np.concatenate([full, np.broadcast_to(np.array(canon_hole), (samples, 2))], axis=1)
    opp = np.concatenate([full, picks[:, :2]], axis=1)
    sh, so = strength.strength7(hero), strength.strength7(opp)
    return float(((sh > so) + 0.5 * (sh == so)).mean())


# --- Edges and the bucketer ---------------------------------------------------


def equal_frequency_edges(values: Sequence[float], n_buckets: int) -> tuple[float, ...]:
    """``n_buckets - 1`` cut points splitting the sorted ``values`` into (nearly) equal-count groups.

    The ideal cut j sits at position c = j * n / B of the sorted values. E[HS]
    has *atoms* (many deals with exactly the same value, e.g. "the board
    plays" on the river, where HS is a multiple of 1/1980), and a bucket is a
    function of the value, so a run of equal values cannot be split. If c
    falls inside a run, the cut moves to the nearer end of the run. The edge
    is the midpoint between the two values on either side of the cut, and
    the bucket of x is ``bisect_right(edges, x)``, monotone in x.

    >>> equal_frequency_edges([0.1, 0.2, 0.3, 0.4], 2)
    (0.25,)
    >>> equal_frequency_edges([0.1, 0.5, 0.5, 0.5, 0.9, 0.9], 2)   # cut 3 is inside the 0.5 run
    (0.7,)
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    edges: list[float] = []
    last = 0
    for j in range(1, n_buckets):
        c = (j * n) // n_buckets
        if v[c - 1] == v[c]:  # inside a run [a, b) of equal values
            a = int(np.searchsorted(v, v[c], "left"))
            b = int(np.searchsorted(v, v[c], "right"))
            c = a if (c - a <= b - c and a > last) or b >= n else b
        if not last < c < n:
            raise ValueError(f"cannot place {n_buckets - 1} distinct cuts: too few distinct values")
        edges.append(float((v[c - 1] + v[c]) / 2.0))
        last = c
    return tuple(edges)


def sample_deals(street: int, n: int, seed: int) -> list[tuple[tuple[int, int], tuple[int, ...]]]:
    """``n`` uniformly random (hole ids, board ids) for ``street`` (1-3), from a hashed seed."""
    rng = np.random.Generator(np.random.PCG64(stream_seed("sample_deals", street, seed)))
    size = (0, 3, 4, 5)[street]
    cards = np.argsort(rng.random((n, 52)), axis=1)[:, :2 + size].tolist()
    return [((c[0], c[1]), tuple(c[2:])) for c in cards]


def _default_edges() -> dict[int, tuple[float, ...]]:
    from nutsfinder.abstraction.bucket_data import EDGES
    return EDGES


def ehs_all(tables: EHSTables, board: Sequence[int]) -> np.ndarray:
    """:meth:`EHSTables.ehs` of *every* combo on ``board`` at once (float64, NaN if it touches the board).

    Indexed by real combo (:func:`~nutsfinder.abstraction.strength.combo_index`); entry h
    equals ``tables.ehs(COMBO_CARDS[h], board)`` exactly (same table, same quantisation).
    """
    n = len(board)
    if n == 5:
        q = tables.river_table(board)
    elif n in (3, 4):
        canon, k = canonical_board(board)
        table = tables.flop_table(canon) if n == 3 else tables.turn_table(canon)
        q = table[_PERM_COMBO[k]]
    else:
        raise ValueError(f"board must have 3, 4 or 5 cards, got {n}")
    out = q.astype(np.float64) / EHS_SCALE
    out[q == _INVALID] = np.nan
    return out


class CardAbstraction:
    """Bucket lookup for one :class:`AbstractionConfig`.

    ``bucket(hole, board)`` with card ids: the preflop class index for an
    empty board, else the E[HS] percentile bucket of the street.
    ``edges`` defaults to the generated :mod:`bucket_data` (built for 50
    buckets per street); pass your own for another bucket count.
    """

    def __init__(self, config: AbstractionConfig = DEFAULT_CONFIG,
                 edges: dict[int, tuple[float, ...]] | None = None,
                 tables: EHSTables | None = None) -> None:
        self.config = config
        self.edges = edges if edges is not None else _default_edges()
        for street in (1, 2, 3):
            if len(self.edges[street]) != config.buckets(street) - 1:
                raise ValueError(f"street {street}: {len(self.edges[street])} edges for "
                                 f"{config.buckets(street)} buckets")
        self.tables = tables if tables is not None else EHSTables()
        self._edge_lists = {s: list(self.edges[s]) for s in (1, 2, 3)}

    def ehs(self, hole: Sequence[int], board: Sequence[int]) -> float:
        """E[HS] by the configured method."""
        if self.config.ehs_method == "mc":
            return ehs_mc(hole, board, self.config.mc_samples)
        return self.tables.ehs(hole, board)

    def bucket(self, hole: Sequence[int], board: Sequence[int]) -> int:
        """Bucket of ``hole`` on ``board`` (ids); 0..168 preflop, 0..B-1 postflop."""
        if not board:
            return preflop_bucket(hole)
        street = (0, 0, 0, 1, 2, 3)[len(board)]
        return bisect_right(self._edge_lists[street], self.ehs(hole, board))

    def bucket_cards(self, hole: Sequence[Card], board: Sequence[Card]) -> int:
        """:meth:`bucket` for :class:`Card` objects."""
        return self.bucket(strength.card_ids(hole), strength.card_ids(board))
