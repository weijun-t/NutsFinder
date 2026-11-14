"""Hand evaluator: ranks the best 5-card poker hand from 5, 6 or 7 cards.

Phase 1, PLAN.md rule 4: the ranking logic is the project owner's to write.
This module only fixes the interface and the contract that
``tests/test_evaluator.py`` checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

from nutsfinder.engine.cards import Card

__all__ = ["HandCategory", "HandRank", "evaluate", "category"]


class HandCategory(IntEnum):
    """The nine hand categories, weakest first."""

    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


HandRank = tuple[int, ...]
"""A totally ordered hand strength: ``(category, tiebreak_1, tiebreak_2, ...)``.

``rank[0]`` is a :class:`HandCategory` value; the remaining entries are card
ranks (2..14) in the order they are compared. Compare two ``HandRank`` values
with ``<``, ``==``, ``>``: higher is better, equal means a split pot.
"""


def evaluate(cards: Sequence[Card]) -> HandRank:
    """Return the strength of the best 5-card hand that can be made from ``cards``.

    Contract (checked by ``tests/test_evaluator.py``):

    - ``cards`` holds 5, 6 or 7 distinct cards. Any other count raises
      ``ValueError``. With 6 or 7 cards, the result is the best of all 5-card
      subsets.
    - The result is a :data:`HandRank` tuple whose first element is the
      :class:`HandCategory` value. Tuples compare lexicographically, so a higher
      category always wins, and within a category the tiebreak ranks decide.
    - Equal results mean a split: two hands that use the same ranks in the same
      roles compare equal. **Suits never break ties** (an ace-high flush in
      spades equals one in hearts).
    - Every card that is part of the 5-card hand matters. For example a pair
      compares the pair, then all three kickers in descending order; a flush
      and a high-card hand compare all five cards in descending order; a full
      house compares the trips first, then the pair; two pair compares the
      higher pair, then the lower pair, then the kicker. Cards outside the best
      five never matter.
    - Aces are high, except in the straight A-2-3-4-5 (the "wheel"), which is
      the *lowest* straight (five-high). Straights do not wrap around:
      Q-K-A-2-3 is not a straight. The same holds for straight flushes: the
      wheel straight flush is the lowest one, and A-K-Q-J-T suited (a royal
      flush) is the highest.
    - The function is pure: same input, same output, no randomness.
    """
    raise NotImplementedError("Phase 1: yours to write (PLAN.md rule 4)")


def category(rank: HandRank) -> HandCategory:
    """The :class:`HandCategory` of an :func:`evaluate` result."""
    return HandCategory(rank[0])
