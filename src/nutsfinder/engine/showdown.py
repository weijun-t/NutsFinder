"""Settling a finished hand: who wins the pot, and everyone's net result.

Two ways a hand can end
-----------------------
- **Fold.** The player who did not fold wins the whole pot. No cards are
  compared (and none need to be known).
- **Showdown.** Each player's best five-card hand from their two hole cards
  plus the five board cards is ranked with :func:`evaluator.evaluate`. The
  higher :data:`HandRank` wins; equal ranks split the pot.

The odd chip
------------
When a pot is split, it can only be split evenly if it is even. Heads-up,
by the time of a showdown any uncalled chips have been returned, so both
players have put in exactly the same total and the pot ``2 * x`` is always
even. :func:`split_pot` still handles an odd pot, because "chips can never
vanish" should not rest on an argument made somewhere else: the odd chip
goes to the big blind, the first seat to the left of the button. That is the
usual casino rule ("odd chip to the first player left of the button"), and
heads-up the big blind is also the player out of position.

Net result
----------
``net[seat] = final_stack - starting_stack``, which is the same as
"chips won from the pot minus chips put into it". Over both seats the nets
always add up to 0: poker (without a rake) moves chips, it never makes them.
"""

from __future__ import annotations

from dataclasses import dataclass

from nutsfinder.engine.betting import HandState
from nutsfinder.engine.evaluator import HandRank, evaluate

__all__ = ["Settlement", "settle", "split_pot"]


@dataclass(frozen=True, slots=True)
class Settlement:
    """How a finished hand paid out. Per-seat values are indexed by seat.

    Attributes:
        winners: the seats that won chips: one seat, or ``(0, 1)`` for a split.
        ranks: each seat's :data:`HandRank` at showdown; ``(None, None)`` if
            the hand ended with a fold.
        winnings: chips each seat took from the pot.
        final_stacks: each seat's stack after collecting.
        net: ``final_stacks - starting_stacks`` for each seat; sums to 0.
    """

    winners: tuple[int, ...]
    ranks: tuple[HandRank | None, HandRank | None]
    winnings: tuple[int, int]
    final_stacks: tuple[int, int]
    net: tuple[int, int]


def split_pot(pot: int, button: int) -> tuple[int, int]:
    """Split ``pot`` between the two seats; any odd chip goes to the big blind.

    Contract: returns ``(a, b)`` with ``a + b == pot``, ``|a - b| <= 1``, and
    if they differ the bigger share belongs to seat ``1 - button``.
    """
    half, odd_chip = divmod(pot, 2)
    shares = [half, half]
    shares[1 - button] += odd_chip
    return shares[0], shares[1]


def settle(state: HandState) -> Settlement:
    """Pay out the pot of a finished hand.

    Contract (checked by ``tests/test_showdown.py``):

    - ``state.is_hand_over`` must be true, and for a showdown both seats'
      hole cards must be in the state; otherwise ``ValueError``.
    - Every chip is accounted for: ``sum(final_stacks) == sum(starting_stacks)``
      and ``sum(net) == 0`` (asserted here, not just tested).
    """
    if not state.is_hand_over:
        raise ValueError("the hand is not over yet")
    if state.folded is not None:
        winners: tuple[int, ...] = (1 - state.folded,)
        ranks: tuple[HandRank | None, HandRank | None] = (None, None)
        winnings = _award(state.pot, winners, state.button)
    else:
        ranks = _showdown_ranks(state)
        winners = _best(ranks)
        winnings = _award(state.pot, winners, state.button)

    final = (state.stacks[0] + winnings[0], state.stacks[1] + winnings[1])
    net = (final[0] - state.starting_stacks[0], final[1] - state.starting_stacks[1])
    # Chips can never appear or vanish. Cheap enough to check on every hand.
    assert sum(final) == sum(state.starting_stacks), (final, state.starting_stacks)
    assert net[0] + net[1] == 0, net
    return Settlement(winners, ranks, winnings, final, net)


def _showdown_ranks(state: HandState) -> tuple[HandRank, HandRank]:
    """Evaluate both hands: two hole cards + the five board cards each."""
    if len(state.hole_cards) != 2:
        raise ValueError("a showdown needs both players' hole cards in the state")
    rank0 = evaluate([*state.hole_cards[0], *state.board])
    rank1 = evaluate([*state.hole_cards[1], *state.board])
    return rank0, rank1


def _best(ranks: tuple[HandRank, HandRank]) -> tuple[int, ...]:
    """The seat(s) holding the best hand: one seat, or both on a tie."""
    if ranks[0] > ranks[1]:
        return (0,)
    if ranks[1] > ranks[0]:
        return (1,)
    return (0, 1)


def _award(pot: int, winners: tuple[int, ...], button: int) -> tuple[int, int]:
    """Chips each seat takes: the whole pot to one winner, or a split."""
    if len(winners) == 2:
        return split_pot(pot, button)
    return (pot, 0) if winners[0] == 0 else (0, pot)
