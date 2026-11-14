"""Hand evaluator: ranks the best 5-card poker hand from 5, 6 or 7 cards.

How it works
------------
The obvious method is "try all 5-card subsets and keep the best": 21 subsets
for 7 cards. That is correct but slow. Instead we look at the 5-7 cards *once*
and read the answer off two summaries:

1. **How many cards of each rank** there are. Pairs, trips, quads and full
   houses are all statements about these counts. Sorting the distinct ranks
   by ``(count, rank)``, largest first, puts the cards that matter at the
   front: for ``K K K 7 7 2`` it gives ``(3, K), (2, 7), (1, 2)``.
2. **Which ranks each suit holds.** A flush needs 5+ cards of one suit, and a
   straight flush is a straight *inside* that suit's ranks.

Straights are found with a *rank bitmask*: an integer with bit ``r`` set for
every rank ``r`` that is present. Five consecutive ranks ending at ``high``
are the five bits ``0b11111 << (high - 4)``, so "is there a straight to
``high``?" is a single AND. The ace is also stored as bit 1, so the wheel
A-2-3-4-5 is bits 1..5, a straight whose high card is 5 (the lowest one).
Because no rank sits above the ace, Q-K-A-2-3 can never look consecutive.

The result is a :data:`HandRank` tuple that holds the category and then
*only* the ranks that decide ties within that category. Holding nothing else
makes the tuple canonical: two hands of equal strength give the identical
tuple, so ``==`` means "split pot" and there are exactly 7,462 distinct
values over all 2,598,960 five-card hands.
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

# Plain-int copies of the category values. The returned tuples hold plain ints
# (not enum members) so they print simply as e.g. (6, 13, 7).
_HIGH_CARD = int(HandCategory.HIGH_CARD)
_PAIR = int(HandCategory.PAIR)
_TWO_PAIR = int(HandCategory.TWO_PAIR)
_TRIPS = int(HandCategory.THREE_OF_A_KIND)
_STRAIGHT = int(HandCategory.STRAIGHT)
_FLUSH = int(HandCategory.FLUSH)
_FULL_HOUSE = int(HandCategory.FULL_HOUSE)
_QUADS = int(HandCategory.FOUR_OF_A_KIND)
_STRAIGHT_FLUSH = int(HandCategory.STRAIGHT_FLUSH)

_ACE = 14
_WHEEL_ACE_BIT = 1 << 1  # the ace, counted low (as a "1") for the wheel

# (high card, bitmask of its five ranks), best straight first: A-high (T..A)
# down to 5-high (the wheel, A..5 with the ace on bit 1).
_STRAIGHT_WINDOWS: tuple[tuple[int, int], ...] = tuple(
    (high, 0b11111 << (high - 4)) for high in range(_ACE, 4, -1)
)

# A card's rank and suit as plain ints, keyed by card id (0..51). Reading
# these is cheaper than converting the Rank/Suit enum members on every call.
_RANK_OF_ID: tuple[int, ...] = tuple(card_id // 4 + 2 for card_id in range(52))
_SUIT_OF_ID: tuple[int, ...] = tuple(card_id % 4 for card_id in range(52))


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
    rank_counts, suit_ranks = _tally(cards)

    # Categories are tested from the strongest down; the first match is the
    # answer, because any 5-card hand of a higher category beats every hand
    # of a lower one.
    flush_ranks = _flush_ranks(suit_ranks)
    if flush_ranks is not None:
        high = _straight_high(_rank_mask(flush_ranks))
        if high:
            return (_STRAIGHT_FLUSH, high)

    groups = _rank_groups(rank_counts)
    (top_count, top_rank), (second_count, second_rank) = groups[0], groups[1]

    if top_count == 4:
        # The kicker is the best other rank, whatever its count.
        return (_QUADS, top_rank, max(rank for _, rank in groups[1:]))
    if top_count == 3 and second_count >= 2:
        # groups[1] is the best rank with 2+ cards besides the trips; with two
        # sets of trips it is the lower set, used as the pair.
        return (_FULL_HOUSE, top_rank, second_rank)
    if flush_ranks is not None:
        return (_FLUSH, *flush_ranks[:5])

    high = _straight_high(_rank_mask(rank_counts))
    if high:
        return (_STRAIGHT, high)

    if top_count == 3:
        # No pair besides the trips (else it was a full house), so the next
        # two groups are the two best single kickers.
        return (_TRIPS, top_rank, second_rank, groups[2][1])
    if top_count == 2 and second_count == 2:
        # With three pairs the lowest pair can still supply the kicker, so
        # take the best rank among *all* the remaining groups.
        return (_TWO_PAIR, top_rank, second_rank, max(rank for _, rank in groups[2:]))
    if top_count == 2:
        return (_PAIR, top_rank, second_rank, groups[2][1], groups[3][1])
    return (_HIGH_CARD, *(rank for _, rank in groups[:5]))


def category(rank: HandRank) -> HandCategory:
    """The :class:`HandCategory` of an :func:`evaluate` result."""
    return HandCategory(rank[0])


# --- Helpers ----------------------------------------------------------------


def _tally(cards: Sequence[Card]) -> tuple[dict[int, int], list[list[int]]]:
    """Count cards per rank and list the ranks held in each suit.

    Also validates the input: 5..7 cards, no card repeated. Duplicates are
    caught with a 52-bit "seen" mask, one bit per card id.
    """
    if not 5 <= len(cards) <= 7:
        raise ValueError(f"evaluate needs 5, 6 or 7 cards, got {len(cards)}")
    rank_counts: dict[int, int] = {}
    suit_ranks: list[list[int]] = [[], [], [], []]
    seen = 0
    for card in cards:
        card_id = card.id
        bit = 1 << card_id
        if seen & bit:
            raise ValueError(f"duplicate card: {card}")
        seen |= bit
        rank = _RANK_OF_ID[card_id]
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
        suit_ranks[_SUIT_OF_ID[card_id]].append(rank)
    return rank_counts, suit_ranks


def _flush_ranks(suit_ranks: list[list[int]]) -> list[int] | None:
    """Ranks of the flush suit, highest first, or ``None`` if there is no flush.

    With at most 7 cards only one suit can hold 5 or more of them (two would
    need 10 cards), so the first such suit is the only one.
    """
    for ranks in suit_ranks:
        if len(ranks) >= 5:
            return sorted(ranks, reverse=True)
    return None


def _rank_mask(ranks: dict[int, int] | list[int]) -> int:
    """Bitmask with bit ``r`` set for each rank present (the ace also on bit 1)."""
    mask = 0
    for rank in ranks:
        mask |= 1 << rank
    if mask >> _ACE & 1:
        mask |= _WHEEL_ACE_BIT
    return mask


def _straight_high(mask: int) -> int:
    """High card of the best straight in a rank bitmask, or 0 if there is none.

    Windows are tried from A-high down, so with six or seven ranks in a row
    the top five are used (e.g. 4-5-6-7-8-9 is a nine-high straight).
    """
    for high, window in _STRAIGHT_WINDOWS:
        if mask & window == window:
            return high
    return 0


def _rank_groups(rank_counts: dict[int, int]) -> list[tuple[int, int]]:
    """Distinct ranks as ``(count, rank)`` pairs, sorted best first.

    Sorting by count first, then rank, lines the ranks up in exactly the
    order a hand compares them: quads before kickers, trips before the pair,
    the higher pair before the lower one, and kickers high to low.
    """
    return sorted(((count, rank) for rank, count in rank_counts.items()), reverse=True)
