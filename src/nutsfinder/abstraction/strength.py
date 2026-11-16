"""Vectorised 7-card hand strength and exact river hand strength (HS).

Why a second evaluator
----------------------
:func:`nutsfinder.engine.evaluator.evaluate` ranks one hand in about 10 us.
Bucketing needs the river hand strength of *every* hole-card pair on a
board (1,081 pairs), and the turn needs that for 48 river cards, so we
need numpy: one array operation over all 1,081 hands at once.

The trick: an integer strength from two lookup tables
-----------------------------------------------------
Every 7-card hand gets a **strength** in ``0..4823``: the index of its
:func:`evaluate` tuple among the 4,824 distinct values a *7-card* hand can
have (fewer than the 7,462 five-card values: e.g. 7-5-4-3-2 high is never
the best five of seven cards, whose worst high card is 9-7-5-4-3), so
``strength(a) < strength(b)`` exactly when ``evaluate(a) < evaluate(b)``
and equal strengths are split pots. Two facts make a table lookup possible:

1. **No flush:** the hand value depends only on the *multiset of ranks*.
   Encode it as ``key = sum over the 7 cards of 5 ** (rank - 2)``: a
   base-5 number whose digit r counts the cards of rank r (at most 4 < 5,
   so no carries and the key is unique). There are 49,205 such multisets;
   ``np.searchsorted`` on the sorted keys finds the row.
2. **A flush (5+ cards of one suit):** with only 7 cards, the other 2 cards
   cannot make quads or a full house (that needs two cards of a rank twice
   more, or 4 of a rank, but five suited cards have five distinct ranks), so
   the flush or straight flush *is* the hand. Its value depends only on the
   13-bit mask of the suited ranks: a table of 8,192 entries.

Worked example: ``As Ad Kc Kd 7h 2s 3c`` has rank digits A:2, K:2, 7:1, 3:1,
2:1, key ``2*5**12 + 2*5**11 + 5**5 + 5 + 1``, and no suit holds 5 cards, so
its strength is the row of that key: two pair, aces and kings, 7 kicker.

Exact river hand strength
-------------------------
HS(h | board) = P(h beats a uniformly random opponent hand) + P(tie) / 2,
over the C(45, 2) = 990 opponent hands that avoid h and the board. For a
board we compute the strength s[c] of all 1,081 hole pairs c at once, sort
them, and for each hero h count with ``searchsorted``

    below(h) = #{c : s[c] < s[h]},   equal(h) = #{c : s[c] == s[h]}.

Card removal: those counts include opponent hands that share a card with
h, which cannot exist. For a card x let below_x(h) count the 46 pairs that
contain x. The only pair containing both of h's cards x, y is h itself, so
by inclusion-exclusion

    wins(h) = below(h) - below_x(h) - below_y(h)           (h is not below itself)
    ties(h) = equal(h) - equal_x(h) - equal_y(h) + 1       (h is equal to itself)

and HS(h) = (wins + ties / 2) / 990. The per-card counts come from one
``searchsorted`` on the per-card sorted strengths, laid end to end with an
offset of 8,192 per card so a single flat search serves all 52 cards.
``tests/test_abstraction_cards.py`` checks this against
:func:`~nutsfinder.engine.equity.equity_vs_range` (exact on the river).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from itertools import combinations, combinations_with_replacement

import numpy as np

from nutsfinder.engine.cards import Card
from nutsfinder.engine.evaluator import HandRank, evaluate

__all__ = [
    "N_COMBOS", "COMBO_CARDS", "combo_index", "card_ids",
    "strength_tables", "strength7", "river_strengths", "river_hs",
]

# --- Combo indexing ---------------------------------------------------------

N_COMBOS = 1326
"""C(52, 2): hole-card pairs, indexed in ``itertools.combinations`` order of card ids."""

COMBO_CARDS: np.ndarray = np.array(list(combinations(range(52), 2)), dtype=np.int64)
"""``COMBO_CARDS[k] = (low id, high id)`` of combo k."""

_INDEX = np.full((52, 52), -1, dtype=np.int64)
_INDEX[COMBO_CARDS[:, 0], COMBO_CARDS[:, 1]] = np.arange(N_COMBOS)
_INDEX[COMBO_CARDS[:, 1], COMBO_CARDS[:, 0]] = np.arange(N_COMBOS)
_COMBO_INDEX: list[list[int]] = _INDEX.tolist()

_CARD_COMBOS = np.array([[k for k in range(N_COMBOS) if x in COMBO_CARDS[k]] for x in range(52)])
"""Row x: the 51 combos that contain card x."""

_RANK = np.arange(52) // 4          # 0..12 (two .. ace)
_SUIT = np.arange(52) % 4
_POW5 = 5 ** _RANK                  # base-5 rank-count digit of each card
_RBIT = 1 << _RANK                  # rank bit of each card

_C1, _C2 = COMBO_CARDS[:, 0], COMBO_CARDS[:, 1]
_COMBO_KEY = _POW5[_C1] + _POW5[_C2]
_COMBO_HAS = np.zeros((N_COMBOS, 52), dtype=bool)
_COMBO_HAS[np.arange(N_COMBOS), _C1] = True
_COMBO_HAS[np.arange(N_COMBOS), _C2] = True

_SENTINEL = 8000     # above every strength: marks pairs that touch the board
_ROW_OFFSET = 8192   # per-card offset in the flat search array (> _SENTINEL)


def combo_index(a: int, b: int) -> int:
    """Index of the pair of card ids ``{a, b}`` (any order)."""
    return _COMBO_INDEX[a][b]


def card_ids(cards: Sequence[Card]) -> list[int]:
    """Card ids of ``cards`` (``Card.id`` without the enum arithmetic)."""
    return [(int(c.rank) - 2) * 4 + int(c.suit) for c in cards]


# --- The two lookup tables --------------------------------------------------


def _cards_for(ranks: Sequence[int], suits: Sequence[int]) -> list[Card]:
    return [Card(r + 2, s) for r, s in zip(ranks, suits)]


@lru_cache(maxsize=1)
def strength_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[HandRank, ...]]:
    """``(keys, nonflush_strength, flush_strength, values)``, built once (~0.6 s).

    ``keys`` (sorted) and ``nonflush_strength`` are aligned: the strength of
    every 7-card rank multiset. ``flush_strength[mask]`` is the strength of
    the best flush in a 13-bit suited-rank mask (-1 below 5 bits).
    ``values[s]`` is the :func:`evaluate` tuple of strength s, so tests can
    check ``values[strength7(h)] == evaluate(h)``.
    """
    nonflush: dict[int, HandRank] = {}
    for ranks in combinations_with_replacement(range(13), 7):
        if max(ranks.count(r) for r in set(ranks)) > 4:
            continue
        # Suits cycle 0,1,2,3,0,...: copies of a rank are adjacent, so they get
        # distinct suits, and 7 cards over 4 suits never make a flush.
        cards = _cards_for(ranks, [j % 4 for j in range(7)])
        nonflush[int(sum(5 ** r for r in ranks))] = evaluate(cards)
    flush: dict[int, HandRank] = {}
    for n in (5, 6, 7):
        for ranks in combinations(range(13), n):
            flush[sum(1 << r for r in ranks)] = evaluate(_cards_for(ranks, [0] * n))
    values = tuple(sorted(set(nonflush.values()) | set(flush.values())))
    if len(values) != 4824:
        raise AssertionError(f"expected 4824 distinct 7-card hand values, got {len(values)}")
    index = {v: i for i, v in enumerate(values)}
    keys = np.array(sorted(nonflush), dtype=np.int64)
    nf = np.array([index[nonflush[k]] for k in keys.tolist()], dtype=np.int64)
    fl = np.full(8192, -1, dtype=np.int64)
    for mask, v in flush.items():
        fl[mask] = index[v]
    return keys, nf, fl, values


def strength7(hands: np.ndarray) -> np.ndarray:
    """Strengths of an ``(n, 7)`` int array of card ids (one hand per row)."""
    keys, nf, fl, _ = strength_tables()
    hands = np.asarray(hands, dtype=np.int64)
    out = nf[np.searchsorted(keys, _POW5[hands].sum(axis=1))]
    suits = _SUIT[hands]
    for s in range(4):
        in_suit = suits == s
        has_flush = in_suit.sum(axis=1) >= 5
        if has_flush.any():
            masks = np.where(in_suit, _RBIT[hands], 0).sum(axis=1)  # distinct ranks: sum == OR
            out = np.where(has_flush, fl[masks], out)
    return out


# --- River: strengths and HS for every hand on one board --------------------


def river_strengths(board: Sequence[int]) -> np.ndarray:
    """Strength of every combo with the 5-card ``board`` (ids); ``_SENTINEL`` if it touches it."""
    keys, nf, fl, _ = strength_tables()
    b = np.asarray(board, dtype=np.int64)
    # Pairs that touch the board can form impossible multisets (5 of a rank);
    # clip their row, they are overwritten with the sentinel below.
    row = np.searchsorted(keys, _COMBO_KEY + int(_POW5[b].sum()))
    s = nf[np.minimum(row, len(keys) - 1)]
    b_suits = _SUIT[b]
    for suit in range(4):
        n_board = int((b_suits == suit).sum())
        if n_board < 3:
            continue  # at most one suit can have 3+ of the 5 board cards
        m1, m2 = _SUIT[_C1] == suit, _SUIT[_C2] == suit
        has_flush = n_board + m1 + m2 >= 5
        mask = int(_RBIT[b[b_suits == suit]].sum()) | np.where(m1, _RBIT[_C1], 0) | np.where(m2, _RBIT[_C2], 0)
        s = np.where(has_flush, fl[mask], s)
    touches = _COMBO_HAS[:, b].any(axis=1)
    return np.where(touches, _SENTINEL, s)


def river_hs(board: Sequence[int]) -> np.ndarray:
    """Exact HS of every combo on the 5-card ``board`` (float64, NaN if it touches the board).

    See the module docstring for the counting argument. Strengths are small
    integers, so the global counts come from a histogram (``below`` = its
    cumulative sum), and "right" searches are "left" searches for s + 1: one
    ``searchsorted`` call does all four per-card counts. About 0.1 ms.
    """
    s = river_strengths(board)
    live = s != _SENTINEL
    hist = np.bincount(s[live], minlength=_SENTINEL + 2)
    below = (np.cumsum(hist) - hist)[s]
    equal = hist[s]
    per_card = np.sort(s[_CARD_COMBOS], axis=1)  # (52, 51); board-touching pairs sort last
    flat = (per_card + _ROW_OFFSET * np.arange(52)[:, None]).ravel()
    q = np.concatenate([s + _ROW_OFFSET * _C1, s + _ROW_OFFSET * _C2])
    lo1, lo2, hi1, hi2 = np.searchsorted(flat, np.concatenate([q, q + 1])).reshape(4, N_COMBOS)
    wins = below - (lo1 - 51 * _C1) - (lo2 - 51 * _C2)
    ties = equal - (hi1 - lo1) - (hi2 - lo2) + 1
    out = (wins + 0.5 * ties) / 990.0
    out[~live] = np.nan
    return out
