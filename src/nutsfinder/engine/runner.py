"""Playing hands between bots: the only place that holds the deck.

:func:`play_hand` runs one hand from start to finish:

1. shuffle a :class:`Deck` with the hand's seed;
2. deal two hole cards to seat 0, then two to seat 1;
3. loop: if the engine needs board cards, deal them from the top of the
   deck; otherwise show the seat to act its :class:`PublicState` and its own
   hole cards, and apply the action it returns;
4. settle the pot, tell both bots how the hand ended, return the
   :class:`HandLog`.

There are no burn cards, and cards always come off the top in the same
order, so for a given seed the hole cards *and* the board (5 cards after the
4 hole cards) are fixed, whatever the bots do. Phase 3's duplicate matches
rely on that: replay the same seed with the seats swapped and each bot gets
the other's cards against the same board.

Reproducibility: the deck's shuffle depends only on the seed, and the bots
must keep their own seeded RNGs (as :class:`RandomBot` does). Then the same
seed and freshly built, identically seeded bots always give an identical
log. Nothing here uses the global :mod:`random` state.

:func:`play_match` plays many hands: the button alternates (seat 0 has it on
even hands), stacks reset to 10,000 every hand, and the per-hand seeds are
drawn from one ``random.Random(master_seed)`` so a whole match is
reproducible from a single number.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from nutsfinder.engine.betting import (
    BIG_BLIND,
    SMALL_BLIND,
    STARTING_STACK,
    HandState,
    apply_action,
    deal_board,
    new_hand,
    public_view,
)
from nutsfinder.engine.bot import Bot, HandSummary
from nutsfinder.engine.cards import Card, Deck
from nutsfinder.engine.hand_log import HandLog, make_log

__all__ = ["play_hand", "play_match", "hand_seeds"]


def play_hand(
    bots: Sequence[Bot],
    seed: int,
    button: int = 0,
    stacks: tuple[int, int] = (STARTING_STACK, STARTING_STACK),
    small_blind: int = SMALL_BLIND,
    big_blind: int = BIG_BLIND,
) -> HandLog:
    """Play one hand between ``bots[0]`` (seat 0) and ``bots[1]`` (seat 1).

    Contract:

    - Deterministic: the same seed and identically seeded bots give an equal
      :class:`HandLog`.
    - A bot that returns an illegal action raises ``ValueError`` naming the
      seat and bot (the hand is abandoned; nothing is half-applied).
    - Each bot sees only its own seat's :func:`public_view` and hole cards.
    """
    if len(bots) != 2:
        raise ValueError(f"heads-up needs exactly 2 bots, got {len(bots)}")
    deck = Deck(seed)
    hole = (tuple(deck.deal(2)), tuple(deck.deal(2)))
    state = new_hand(button, stacks, small_blind, big_blind, hole_cards=hole)
    while not state.is_hand_over:
        if state.cards_needed:
            state = deal_board(state, deck.deal(state.cards_needed))
        else:
            state = _ask_bot(bots, state)
    log = make_log(state, seed)
    _tell_bots(bots, state, log)
    return log


def _ask_bot(bots: Sequence[Bot], state: HandState) -> HandState:
    """Ask the seat to act for its action and apply it."""
    seat = state.to_act
    assert seat is not None
    bot = bots[seat]
    action = bot.act(public_view(state, seat), state.hole_cards[seat])
    try:
        return apply_action(state, seat, action)
    except ValueError as err:
        raise ValueError(f"seat {seat} ({bot!r}) chose an illegal action: {err}") from err


def _tell_bots(bots: Sequence[Bot], state: HandState, log: HandLog) -> None:
    """Give each bot its end-of-hand summary (cards shown only at showdown)."""
    shown: tuple[tuple[Card, ...] | None, tuple[Card, ...] | None]
    if state.is_showdown:
        shown = (state.hole_cards[0], state.hole_cards[1])
    else:
        shown = (None, None)
    for seat in (0, 1):
        summary = HandSummary(seat, public_view(state, seat), shown, log.result)
        bots[seat].end_hand(summary)


def hand_seeds(master_seed: int, n_hands: int) -> list[int]:
    """The per-hand seeds of a match: ``n_hands`` 64-bit ints from one seed."""
    rng = random.Random(master_seed)
    return [rng.getrandbits(64) for _ in range(n_hands)]


def play_match(
    bots: Sequence[Bot],
    n_hands: int,
    master_seed: int,
    stacks: tuple[int, int] = (STARTING_STACK, STARTING_STACK),
) -> list[HandLog]:
    """Play ``n_hands`` hands; the button alternates, starting with seat 0.

    Stacks are reset to ``stacks`` before every hand, so each hand is an
    independent sample (which is what Phase 3's statistics assume).
    """
    return [
        play_hand(bots, seed, button=i % 2, stacks=stacks)
        for i, seed in enumerate(hand_seeds(master_seed, n_hands))
    ]
