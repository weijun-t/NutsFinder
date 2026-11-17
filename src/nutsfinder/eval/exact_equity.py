"""Exact (enumerated) equities for *every* hole-card pair at once, vectorised with numpy.

Two questions come up in Local Best Response (:mod:`nutsfinder.eval.lbr`):

1. **LBR's equity against each opponent holding.** LBR holds ``hero`` and
   wants, for each of the 1,326 combos ``c`` the opponent might hold, the
   share of the pot ``hero`` wins against ``c`` if the hand is checked down
   from here: :func:`hero_vs_each`.
2. **The opponent's own equity estimate, for every holding.** EquityBot,
   BayesBot and PotOddsBot decide by their equity against an assumed range
   (top ``w`` of hands, or a random hand). To ask "what would you do with
   holding ``h``?" for all ``h`` without 1,326 Monte Carlo runs, we compute
   every holding's *exact* equity against a weighted range in one pass:
   :func:`holdings_vs_ranges`.

Both enumerate every run-out of the board and use
:func:`nutsfinder.abstraction.strength.river_strengths` (the strength of
all 1,326 combos on one complete board in ~40 us).

Why enumerating all run-outs is the uniform average
---------------------------------------------------
Equity of ``h`` against a range is the average over (opponent combo ``c``,
run-out ``r``) drawn uniformly among the pairs that share no card with
``h``, the board or each other, ``c`` weighted by the range. We enumerate
*every* set ``r`` of missing board cards (from the cards not on the board)
and count a pair only when ``r`` avoids both ``h`` and ``c``; each disjoint
``(h, c)`` gets exactly ``C(52 - |board| - 4, 5 - |board|)`` run-outs
(990 on the flop, 44 on the turn, 1 on the river), the same number for
every pair. So the plain sum over run-outs, divided by the total weight
counted, is the uniform average.

Card removal by inclusion-exclusion (as in ``strength.river_hs``)
-----------------------------------------------------------------
On one complete board with strengths ``s`` and combo weights ``w`` (zero
for combos that touch the board), for a holding ``h = {a, b}``::

    wins(h)  = W_below(s_h) - W_below_a(s_h) - W_below_b(s_h)
    ties(h)  = W_equal(s_h) - W_equal_a(s_h) - W_equal_b(s_h) + w_h
    total(h) = W - W_a - W_b + w_h

where ``W_*`` sums over all live combos and ``W_*_x`` over the 51 combos
that contain card ``x``. The only combo containing both ``a`` and ``b`` is
``h`` itself, which is never below itself but is equal to itself and is in
both card rows, hence the ``+ w_h`` corrections. The per-card sums come
from one sorted array of the 52 card rows (offset per card) and
``searchsorted``, as in ``strength.river_hs``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Sequence
from collections import OrderedDict
from itertools import combinations
from pathlib import Path

import numpy as np

from nutsfinder.abstraction.cards import _PERM_COMBO, preflop_bucket, stream_seed
from nutsfinder.abstraction.strength import COMBO_CARDS, N_COMBOS, combo_index, river_strengths, strength_tables

log = logging.getLogger(__name__)

__all__ = ["hero_vs_each", "holdings_vs_ranges", "combo_weights", "CARD_COMBOS", "preflop_vs_each",
           "build_preflop_table", "save_preflop_table", "load_preflop_table", "ensure_preflop_table",
           "preflop_table_info", "PREFLOP_TABLE_FILE", "PREFLOP_TABLE_BOARDS"]

_SENTINEL = 8000  # strength.river_strengths marks combos that touch the board with this

CARD_COMBOS = np.array([[k for k in range(N_COMBOS) if x in COMBO_CARDS[k]] for x in range(52)], dtype=np.int64)
"""Row x: the 51 combo indices that contain card x (increasing)."""

_A, _B = COMBO_CARDS[:, 0], COMBO_CARDS[:, 1]
_ROWS_FLAT = CARD_COMBOS.ravel()
_AB = np.stack([_A, _B])                           # (2, N): each combo's low and high card
_START = np.concatenate([51 * _A, 51 * _B])        # where rows a and b start in the flat array
_ROW_OFFSET = 8192  # > every strength and the sentinel: keeps groups apart in one sorted array


def _runouts(board: Sequence[int], exclude: Sequence[int] = ()) -> list[tuple[int, ...]]:
    known = set(board) | set(exclude)
    rest = [c for c in range(52) if c not in known]
    return list(combinations(rest, 5 - len(board)))


def hero_vs_each(hero: Sequence[int], board: Sequence[int], preflop_boards: int = 2000) -> np.ndarray:
    """Equity of ``hero`` (2 card ids) against each combo, checked down from ``board``.

    Returns a float array over the 1,326 combos (combo order of
    :data:`~nutsfinder.abstraction.strength.COMBO_CARDS`): win 1, split 1/2,
    averaged over every run-out that avoids both hands. NaN for combos that
    touch ``hero`` or ``board``. Exact on the flop, turn and river (one
    ``river_strengths`` per run-out: 1,081 on the flop, ~45 ms; 46 on the
    turn; 1 on the river). Preflop (empty board) it averages over
    ``preflop_boards`` random 5-card boards (~80 ms for 2,000; per-combo SE
    about 1 point), drawn from a generator seeded by the hero's cards, so the
    answer is deterministic.
    """
    hero, board = list(hero), list(board)
    if len(hero) != 2 or len(board) not in (0, 3, 4, 5) or len(set(hero + board)) != len(hero) + len(board):
        raise ValueError(f"bad hero {hero} / board {board}")
    strength_tables()
    h = combo_index(hero[0], hero[1])
    total = np.zeros(N_COMBOS)
    count = np.zeros(N_COMBOS)
    if board:
        runouts = _runouts(board, hero)
    else:
        rest = np.array([c for c in range(52) if c not in hero])
        gen = np.random.default_rng([h, 1326])
        runouts = [tuple(gen.choice(rest, 5, replace=False).tolist()) for _ in range(preflop_boards)]
    for r in runouts:
        s = river_strengths(board + list(r))
        sh = s[h]
        live = s != _SENTINEL
        total += np.where(live, (sh > s) + 0.5 * (sh == s), 0.0)
        count += live
    dead = np.zeros(N_COMBOS, dtype=bool)
    dead[CARD_COMBOS[hero].ravel()] = True
    dead[CARD_COMBOS[board].ravel()] = True
    out = np.full(N_COMBOS, np.nan)
    ok = ~dead
    out[ok] = total[ok] / count[ok]
    return out


_PREP: OrderedDict[tuple[int, ...], tuple] = OrderedDict()
_PREP_MAX = 600  # run-outs kept (~45 KB each, int32): a sampled flop's 200 plus a few turns' 46


def _prepared(full_board: tuple[int, ...]) -> tuple:
    """The weight-independent part of one complete board, cached (LRU): strengths order and ranks.

    Returns ``(live, order, lo, hi, rows_order, lo_r, hi_r)``: ``order`` sorts the
    strengths, ``lo``/``hi`` are each combo's first / one-past-last position among
    equal strengths. For the per-card rows (the 51 combos holding card x), the
    rows are laid end to end with an offset per card (as in ``strength.river_hs``),
    so one argsort serves all 52 rows (row x starts at position 51 x);
    ``rows_order`` maps sorted positions to combo indices and ``lo_r``/``hi_r`` are
    the positions of each combo in the row of its low card, then of its high card.
    """
    hit = _PREP.get(full_board)
    if hit is not None:
        return hit
    s = river_strengths(list(full_board))
    live = s != _SENTINEL
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    lo = np.searchsorted(s_sorted, s, side="left")
    hi = np.searchsorted(s_sorted, s, side="right")
    flat = (s[CARD_COMBOS] + _ROW_OFFSET * np.arange(52)[:, None]).ravel()
    order_r = np.argsort(flat, kind="stable")
    flat_sorted = flat[order_r]
    q = (s[None, :] + _ROW_OFFSET * _AB).ravel()  # (2, N): queries in row a and in row b
    i32 = np.int32
    out = (live, order.astype(i32), lo.astype(i32), hi.astype(i32), _ROWS_FLAT[order_r].astype(i32),
           np.searchsorted(flat_sorted, q, side="left").astype(i32),
           np.searchsorted(flat_sorted, q, side="right").astype(i32))
    _PREP[full_board] = out
    if len(_PREP) > _PREP_MAX:
        _PREP.popitem(last=False)
    return out


def holdings_vs_ranges(board: Sequence[int], weights: np.ndarray, max_runouts: int | None = None) -> np.ndarray:
    """Exact equity of *every* holding against each weighted range, checked down from ``board``.

    Args:
        board: 3, 4 or 5 card ids.
        weights: ``(K, 1326)`` non-negative combo weights (one range per
            row; e.g. 1 for the combos of "top 30%", 0 elsewhere). Combos
            touching the board are ignored.
        max_runouts: if set and the board has more run-outs (the flop has
            1,176), use this many, evenly spaced in lexicographic order (a
            deterministic function of the board). ``None``: exact.

    Returns ``(K, 1326)``: entry ``[k, h]`` is the equity of holding ``h``
    against range ``k`` with card removal (the range's combos that share a
    card with ``h`` are dropped, as in
    :func:`nutsfinder.engine.equity.equity_vs_range`). NaN where ``h``
    touches the board or no range combo is compatible with ``h``.
    Cost ~0.7 ms of CPU per run-out for K = 2 (so ~0.8 s for an exact flop, 30 ms on the turn).
    """
    board = list(board)
    if len(board) not in (3, 4, 5) or len(set(board)) != len(board):
        raise ValueError(f"bad board {board}")
    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    if w.shape[1] != N_COMBOS:
        raise ValueError(f"weights must have {N_COMBOS} columns, got {w.shape}")
    strength_tables()
    k_ranges = w.shape[0]
    win = np.zeros((k_ranges, N_COMBOS))
    tie = np.zeros((k_ranges, N_COMBOS))
    tot = np.zeros((k_ranges, N_COMBOS))
    zero = np.zeros((k_ranges, 1))
    n = N_COMBOS
    runouts = _runouts(board)
    if max_runouts is not None and len(runouts) > max_runouts:
        # Deterministic, evenly spaced subset of the lexicographically ordered run-outs (so roughly
        # stratified by the turn card); depends only on the board, i.e. public information.
        idx = np.linspace(0, len(runouts) - 1, max_runouts).round().astype(int)
        runouts = [runouts[i] for i in sorted(set(idx.tolist()))]
    for r in runouts:
        live, order, lo, hi, rows_order, lo_r, hi_r = _prepared(tuple(board) + tuple(r))
        wl = w * live  # (K, 1326)
        # Global weighted counts below / equal to each strength.
        cum = np.concatenate([zero, np.cumsum(wl[:, order], axis=1)], axis=1)
        below = cum[:, lo]
        equal = cum[:, hi] - below
        total = wl.sum(axis=1, keepdims=True)
        # Per-card rows (see _prepared).
        cum_r = np.concatenate([zero, np.cumsum(wl[:, rows_order], axis=1)], axis=1)
        below_r = cum_r[:, lo_r] - cum_r[:, _START]
        equal_r = cum_r[:, hi_r] - cum_r[:, lo_r]
        tot_r = cum_r[:, _START + 51] - cum_r[:, _START]
        win += np.where(live, below - below_r[:, :n] - below_r[:, n:], 0.0)
        tie += np.where(live, equal - equal_r[:, :n] - equal_r[:, n:] + wl, 0.0)
        tot += np.where(live, total - tot_r[:, :n] - tot_r[:, n:] + wl, 0.0)
    out = np.full((k_ranges, N_COMBOS), np.nan)
    on_board = np.zeros(N_COMBOS, dtype=bool)
    on_board[CARD_COMBOS[board].ravel()] = True
    ok = (tot > 1e-12) & ~on_board[None, :]
    out[ok] = (win[ok] + 0.5 * tie[ok]) / tot[ok]
    return out


def combo_weights(combos: Sequence[tuple]) -> np.ndarray:
    """A 0/1 weight vector over the 1,326 combos for a list of :class:`Card` pairs (repeats count once)."""
    w = np.zeros(N_COMBOS)
    for a, b in combos:
        w[combo_index(a.id, b.id)] = 1.0
    return w


# --- Preflop: a precomputed hero-class x opponent-combo equity table ---------------------------
#
# Preflop, LBR needs its equity against *each* of the opponent's 1,326 holdings (to weight by
# the belief). Enumerating all C(48, 5) = 1.7 million boards per hand is far too slow, and
# :func:`hero_vs_each` with 2,000 sampled boards has a *correlated* error (the same boards for
# every opponent combo): measured up to 1.7 points against a belief-weighted range, too much
# for EV comparisons whose margins are often a point or two. So we precompute once:
#
# - for each of the 169 hand classes, a canonical hero combo (the lowest combo index of the
#   class) and its equity against each of the 1,326 combos (NaN where they share a card),
#   averaged over PREFLOP_TABLE_BOARDS random boards (seeded per class with ``stream_seed``,
#   deterministic), then symmetrised over the suit permutations that fix the hero (2 to 6
#   of them), which averages equivalent opponent combos (equity(h, j) = equity(pi h, pi j));
# - for any hero h of the class, a suit permutation pi with pi(h) = canonical, so
#   equity(h vs j) = table[class, pi(j)] (exact relabelling, no approximation).
#
# Per-entry SE before symmetrising <= 0.5/sqrt(boards) (0.16 points at 100,000); the
# belief-weighted error is at most that. Build cost about 169 x boards x 60 us (~17 CPU
# minutes at 50,000); stored as float32 (0.9 MB) in models/ (gitignored, like the turn tables).

PREFLOP_TABLE_BOARDS = 100_000
PREFLOP_TABLE_FILE = Path(__file__).resolve().parents[3] / "models" / "preflop_vs_combo.npz"

_CLASS_CANON = np.full(169, -1, dtype=np.int64)
for _k in range(N_COMBOS):
    _c = preflop_bucket(COMBO_CARDS[_k].tolist())
    if _CLASS_CANON[_c] < 0:
        _CLASS_CANON[_c] = _k
_HERO_CLASS = np.array([preflop_bucket(COMBO_CARDS[k].tolist()) for k in range(N_COMBOS)], dtype=np.int64)
_HERO_PERM = np.full(N_COMBOS, -1, dtype=np.int64)
for _p in range(len(_PERM_COMBO)):
    _hit = (_PERM_COMBO[_p] == _CLASS_CANON[_HERO_CLASS]) & (_HERO_PERM < 0)
    _HERO_PERM[_hit] = _p
assert (_HERO_PERM >= 0).all()
del _k, _c, _p, _hit

_PF_TABLE: np.ndarray | None = None
_PF_INFO: dict = {}


def _preflop_row(cls: int, boards: int) -> np.ndarray:
    """Equity of class ``cls``'s canonical hero against each combo (sampled boards, symmetrised)."""
    strength_tables()
    h = int(_CLASS_CANON[cls])
    hero = COMBO_CARDS[h].tolist()
    rest = np.array([c for c in range(52) if c not in hero])
    gen = np.random.default_rng(stream_seed("lbr_preflop_table", cls, boards))
    total = np.zeros(N_COMBOS)
    count = np.zeros(N_COMBOS)
    for _ in range(boards):
        s = river_strengths(gen.choice(rest, 5, replace=False).tolist())
        sh = s[h]
        live = s != _SENTINEL
        total += np.where(live, (sh > s) + 0.5 * (sh == s), 0.0)
        count += live
    dead = np.zeros(N_COMBOS, dtype=bool)
    dead[CARD_COMBOS[hero].ravel()] = True
    row = np.where(dead, np.nan, total / np.maximum(count, 1))
    stab = [p for p in range(len(_PERM_COMBO)) if _PERM_COMBO[p][h] == h]
    return np.mean([row[_PERM_COMBO[p]] for p in stab], axis=0)


def build_preflop_table(boards: int = PREFLOP_TABLE_BOARDS, processes: int = 1) -> np.ndarray:
    """The (169, 1326) float32 table (see the comment above); ``processes > 1`` uses a spawn pool."""
    t0 = time.perf_counter()
    if processes > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes) as pool:
            rows = pool.starmap(_preflop_row, [(c, boards) for c in range(169)])
    else:
        rows = [_preflop_row(c, boards) for c in range(169)]
    log.info("preflop equity table: %d boards per class in %.0f s", boards, time.perf_counter() - t0)
    return np.stack(rows).astype(np.float32)


def save_preflop_table(table: np.ndarray, boards: int, path: Path = PREFLOP_TABLE_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, table=table, boards=np.int64(boards))
    os.replace(tmp, path)


def load_preflop_table(path: Path = PREFLOP_TABLE_FILE) -> dict:
    """Load the table into this process; returns :func:`preflop_table_info`."""
    global _PF_TABLE, _PF_INFO
    with np.load(path) as z:
        table = np.asarray(z["table"], dtype=np.float32)
        boards = int(z["boards"])
    if table.shape != (169, N_COMBOS):
        raise ValueError(f"bad preflop table shape {table.shape}")
    _PF_TABLE = table
    _PF_INFO = {"path": str(path), "boards": boards,
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    return dict(_PF_INFO)


def ensure_preflop_table(path: Path = PREFLOP_TABLE_FILE, boards: int = PREFLOP_TABLE_BOARDS,
                         processes: int = 1) -> dict:
    """Load the table, building and saving it first if the file is missing."""
    path = Path(path)
    if not path.exists():
        log.warning("building the preflop equity table (%d boards per class, %d processes) -> %s",
                    boards, processes, path)
        save_preflop_table(build_preflop_table(boards, processes), boards, path)
    return load_preflop_table(path)


def preflop_table_info() -> dict:
    """Path, boards and SHA-256 of the loaded table (empty if none is loaded)."""
    return dict(_PF_INFO)


def preflop_vs_each(hero: Sequence[int]) -> np.ndarray:
    """Preflop equity of ``hero`` (2 card ids) against each combo, from the precomputed table.

    Same convention as :func:`hero_vs_each` (float64 over the 1,326 combos, NaN for combos
    that share a card with ``hero``). Loads (or builds) :data:`PREFLOP_TABLE_FILE` on first use.
    """
    if _PF_TABLE is None:
        ensure_preflop_table()
    assert _PF_TABLE is not None
    h = combo_index(hero[0], hero[1])
    return _PF_TABLE[_HERO_CLASS[h]][_PERM_COMBO[_HERO_PERM[h]]].astype(np.float64)
