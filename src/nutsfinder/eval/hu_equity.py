"""Exact heads-up equity of two KNOWN hands, by full enumeration of the run-outs.

Variance reduction (:mod:`nutsfinder.eval.variance_reduction`) needs, for
every logged hand, seat 0's exact share of the pot given both hole cards and
the board so far:

    e = P(seat 0 wins) + P(tie) / 2,   over all run-outs of the missing board cards,

uniformly (the deck is shuffled uniformly, so given everything dealt so far
each set of missing board cards is equally likely). This module computes it
exactly:

- river (5 board cards): one comparison;
- turn (4): 44 run-outs; flop (3): C(45, 2) = 990 run-outs. Both use the
  vectorised 7-card strength :func:`nutsfinder.abstraction.strength.strength7`
  on all run-outs at once (well under a millisecond).
- preflop (0): C(48, 5) = 1,712,304 run-outs. Calling ``strength7`` on 3.4
  million 7-card hands per matchup would take seconds, so we precompute,
  once per process, a description of all C(52, 5) = 2,598,960 boards:

    * the board's rank multiset (index into the 6,175 multisets of 5 ranks
      with at most 4 of a kind) and a table ``NF[multiset, hole rank pair]``
      of the non-flush 7-card strength (the strength of a 7-card hand
      without a flush depends only on its rank multiset);
    * per suit, the 13-bit mask of the board's ranks in that suit (a flush
      among 7 cards excludes quads and full houses, so when 5+ cards share a
      suit the flush strength from the mask *is* the hand's strength, see
      :mod:`nutsfinder.abstraction.strength`);
    * a 52-bit card mask per board, to drop the boards that touch a hole card.

  A preflop equity is then a few gathers over 1.7M boards (~0.1 s). Results
  are cached per suit-isomorphism class (relabelling suits, or swapping the
  two cards inside a hand, cannot change the equity), so repeated matchups
  (every duplicate pair deals the same two hands twice) cost nothing.

The tables cost ~60 MB and ~3 s to build, once per process. Everything is
cross-checked against :func:`strength7` and the engine's evaluator in
``tests/test_variance_reduction.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, permutations

import numpy as np

from nutsfinder.abstraction.strength import strength7, strength_tables
from nutsfinder.engine.cards import Card

__all__ = ["equity_hu", "preflop_equity_hu", "canonical_matchup", "card_id"]

_RANK = np.arange(52) // 4
_SUIT = np.arange(52) % 4
_POW5 = 5 ** _RANK
_RBIT = (1 << _RANK).astype(np.int64)


def card_id(c: Card | int) -> int:
    """Card id 0..51, ``(rank - 2) * 4 + suit`` (ints pass through)."""
    if isinstance(c, (int, np.integer)):
        return int(c)
    return (int(c.rank) - 2) * 4 + int(c.suit)


def _ids(cards: Sequence[Card | int]) -> list[int]:
    return [card_id(c) for c in cards]


# --- Suit isomorphism -------------------------------------------------------------

_SUIT_PERMS = list(permutations(range(4)))


def canonical_matchup(h0: Sequence[int], h1: Sequence[int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """The lexicographically smallest relabelling of ``(h0, h1)`` under the 24 suit permutations.

    Each hand is sorted, so card order inside a hand does not matter either.
    Seat order is kept (equity is seat 0's), so ``(A, B)`` and ``(B, A)``
    are different keys (their equities sum to 1).
    """
    best = None
    for p in _SUIT_PERMS:
        a = tuple(sorted(4 * (x // 4) + p[x % 4] for x in h0))
        b = tuple(sorted(4 * (x // 4) + p[x % 4] for x in h1))
        key = (a, b)
        if best is None or key < best:
            best = key
    return best  # type: ignore[return-value]


# --- Preflop board tables ---------------------------------------------------------


@lru_cache(maxsize=1)
def _board_tables() -> dict[str, np.ndarray]:
    """Per-board descriptions of all C(52, 5) boards (see the module docstring)."""
    keys, nf, _, _ = strength_tables()
    boards = np.array(list(combinations(range(52), 5)), dtype=np.int8)  # 2,598,960 x 5
    b = boards.astype(np.int64)
    # Rank multiset of the board -> index among the 5-rank multisets.
    ms_list = [m for m in combinations_with_replacement(range(13), 5) if max(m.count(r) for r in set(m)) <= 4]
    ms_keys = np.array(sorted(sum(5 ** r for r in m) for m in ms_list), dtype=np.int64)
    bkey = _POW5[b].sum(axis=1)
    ms_idx = np.searchsorted(ms_keys, bkey).astype(np.int16)
    # NF[multiset, hole rank pair (r1 <= r2)]: non-flush strength of the 7 cards.
    pair_keys = np.array([[5 ** r1 + 5 ** r2 for r2 in range(13)] for r1 in range(13)], dtype=np.int64)
    tot = ms_keys[:, None, None] + pair_keys[None, :, :]                      # (6175, 13, 13)
    row = np.searchsorted(keys, tot.ravel())
    row = np.minimum(row, len(keys) - 1)
    found = keys[row] == tot.ravel()  # impossible multisets (5 of a rank) are never used
    nf_tab = np.where(found, nf[row], -1).reshape(tot.shape).astype(np.int16)
    # Per suit 13-bit rank masks and card counts.
    suit = _SUIT[b]
    masks = np.zeros((4, len(b)), dtype=np.int16)
    counts = np.zeros((4, len(b)), dtype=np.int8)
    for s in range(4):
        ins = suit == s
        masks[s] = np.where(ins, _RBIT[b], 0).sum(axis=1)
        counts[s] = ins.sum(axis=1)
    cardmask = (np.int64(1) << b).sum(axis=1)  # distinct cards: sum == OR
    return {"ms_idx": ms_idx, "nf_tab": nf_tab, "masks": masks, "counts": counts, "cardmask": cardmask}


def _strength_preflop(hole: Sequence[int], sel: np.ndarray, t: dict[str, np.ndarray]) -> np.ndarray:
    """7-card strengths of ``hole`` with every selected board (``sel`` = indices into the board table)."""
    _, _, fl, _ = strength_tables()
    r1, r2 = sorted((int(_RANK[hole[0]]), int(_RANK[hole[1]])))
    out = t["nf_tab"][t["ms_idx"][sel], r1, r2].astype(np.int64)
    for s in range(4):
        hc = sum(1 for x in hole if _SUIT[x] == s)
        hm = sum(int(_RBIT[x]) for x in hole if _SUIT[x] == s)
        cnt = t["counts"][s][sel]
        has = cnt + hc >= 5
        if has.any():
            m = t["masks"][s][sel][has].astype(np.int64) | hm
            out[has] = fl[m]
    return out


_PREFLOP_CACHE: dict[tuple, float] = {}


def preflop_equity_hu(h0: Sequence[Card | int], h1: Sequence[Card | int]) -> float:
    """Seat 0's exact preflop equity (win + tie/2) over all 1,712,304 boards. Cached by suit isomorphism."""
    a, b = _ids(h0), _ids(h1)
    if len(set(a + b)) != 4:
        raise ValueError("need four distinct hole cards")
    key = canonical_matchup(a, b)
    hit = _PREFLOP_CACHE.get(key)
    if hit is not None:
        return hit
    t = _board_tables()
    dead = sum(1 << x for x in key[0] + key[1])  # the CANONICAL cards: the boards must avoid them
    sel = np.flatnonzero((t["cardmask"] & dead) == 0)
    s0 = _strength_preflop(key[0], sel, t)
    s1 = _strength_preflop(key[1], sel, t)
    e = (np.count_nonzero(s0 > s1) + 0.5 * np.count_nonzero(s0 == s1)) / len(sel)
    _PREFLOP_CACHE[key] = float(e)
    return float(e)


# --- Postflop: direct enumeration -------------------------------------------------


def _postflop_equity(a: list[int], b: list[int], board: list[int]) -> float:
    missing = 5 - len(board)
    if missing == 0:
        s = strength7(np.array([a + board, b + board], dtype=np.int64))
        return 1.0 if s[0] > s[1] else 0.0 if s[0] < s[1] else 0.5
    live = [x for x in range(52) if x not in set(a + b + board)]
    runs = np.array(list(combinations(live, missing)), dtype=np.int64)
    n = len(runs)
    full = np.concatenate([np.tile(np.array(board, dtype=np.int64), (n, 1)), runs], axis=1)
    s0 = strength7(np.concatenate([np.tile(np.array(a, dtype=np.int64), (n, 1)), full], axis=1))
    s1 = strength7(np.concatenate([np.tile(np.array(b, dtype=np.int64), (n, 1)), full], axis=1))
    return float((np.count_nonzero(s0 > s1) + 0.5 * np.count_nonzero(s0 == s1)) / n)


def equity_hu(h0: Sequence[Card | int], h1: Sequence[Card | int], board: Sequence[Card | int] = ()) -> float:
    """Seat 0's exact equity (win + tie/2) given both hands and a board of 0, 3, 4 or 5 cards."""
    a, b, bd = _ids(h0), _ids(h1), _ids(board)
    if len(a) != 2 or len(b) != 2 or len(bd) not in (0, 3, 4, 5):
        raise ValueError("two hole cards each and a board of 0, 3, 4 or 5 cards")
    if len(set(a + b + bd)) != 4 + len(bd):
        raise ValueError("cards must be distinct")
    if not bd:
        return preflop_equity_hu(a, b)
    return _postflop_equity(a, b, bd)
