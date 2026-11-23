"""Playing one N-player hand between multiway bots; the only holder of the deck.

Deal order (deterministic given the seed)
-----------------------------------------
1. ``Deck(seed)`` shuffles once (the same deck class as the heads-up runner).
2. Hole cards: two consecutive cards to seat 0, then two to seat 1, ...,
   then seat n-1 (by seat index, *not* starting left of the button). So for
   a given seed and n each seat's cards are fixed wherever the button is.
3. Board: the next 3 cards are the flop, then 1 turn, then 1 river, straight
   off the top (no burn cards), whatever the bots do.

At two players this is exactly the heads-up runner's deal, so
``play_hand_multi`` and ``runner.play_hand`` see the same cards.

Chips conservation (``sum(stacks) + pot == sum(starting_stacks)``) is
checked after every transition and settlement checks that nets sum to 0;
a violation raises ``RuntimeError`` (not ``assert``, so ``python -O`` keeps it).

Showdown reveal: at a showdown *every* live hand is revealed to all bots in
``MultiHandSummary.shown_cards`` (nobody mucks), exactly as the heads-up
runner reveals both hands; folded hands and fold-won hands are never shown.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from nutsfinder.engine.betting import BIG_BLIND, SMALL_BLIND, STARTING_STACK
from nutsfinder.engine.cards import Card, Deck
from nutsfinder.engine.multiway.bot import MultiBot, MultiHandSummary
from nutsfinder.engine.multiway.hand_log import MultiHandLog, make_log
from nutsfinder.engine.multiway.state import (
    MultiHandState,
    apply_action,
    deal_board,
    new_hand,
    public_view,
)

__all__ = ["play_hand_multi", "play_match_multi", "deal_hole_cards", "shown_cards"]


def deal_hole_cards(deck: Deck, n: int) -> tuple[tuple[Card, ...], ...]:
    """Two cards per seat, seat 0 first (see the module doc)."""
    return tuple(tuple(deck.deal(2)) for _ in range(n))


def play_hand_multi(
    bots: Sequence[MultiBot],
    seed: int,
    button: int = 0,
    stacks: Sequence[int] | None = None,
    small_blind: int = SMALL_BLIND,
    big_blind: int = BIG_BLIND,
) -> MultiHandLog:
    """Play one hand; ``bots[i]`` sits in seat ``i``.

    ``stacks`` defaults to 10,000 for every seat. The same seed and
    identically seeded bots give an equal log. An illegal action raises
    ``ValueError`` naming the seat and bot.
    """
    n = len(bots)
    stacks = tuple(stacks) if stacks is not None else (STARTING_STACK,) * n
    if len(stacks) != n:
        raise ValueError(f"{n} bots but {len(stacks)} stacks")
    deck = Deck(seed)
    hole = deal_hole_cards(deck, n)
    state = new_hand(stacks, button, small_blind, big_blind, hole_cards=hole)
    total = sum(stacks)
    while not state.is_hand_over:
        if state.cards_needed:
            state = deal_board(state, deck.deal(state.cards_needed))
        else:
            state = _ask_bot(bots, state)
        if sum(state.stacks) + state.pot != total:
            raise RuntimeError(f"chips not conserved: {state.stacks} + pot {state.pot} != {total}")
    log = make_log(state, seed)
    _tell_bots(bots, state, log)
    return log


def _ask_bot(bots: Sequence[MultiBot], state: MultiHandState) -> MultiHandState:
    seat = state.to_act
    assert seat is not None
    bot = bots[seat]
    action = bot.act(public_view(state, seat), state.hole_cards[seat])
    try:
        return apply_action(state, seat, action)
    except ValueError as err:
        raise ValueError(f"seat {seat} ({bot!r}) chose an illegal action: {err}") from err


def shown_cards(state: MultiHandState) -> tuple[tuple[Card, ...] | None, ...]:
    """Hole cards of the players who reached a showdown; ``None`` for the rest."""
    if not state.is_showdown:
        return (None,) * state.n
    return tuple(None if state.folded[s] else state.hole_cards[s] for s in range(state.n))


def _tell_bots(bots: Sequence[MultiBot], state: MultiHandState, log: MultiHandLog) -> None:
    shown = shown_cards(state)
    for seat, bot in enumerate(bots):
        bot.end_hand(MultiHandSummary(seat, public_view(state, seat), shown, log.result))


def play_match_multi(
    bots: Sequence[MultiBot],
    n_hands: int,
    master_seed: int,
    stacks: Sequence[int] | None = None,
) -> list[MultiHandLog]:
    """``n_hands`` hands, stacks reset every hand, button moving one seat
    clockwise per hand from seat 0. Per-hand seeds come from one
    ``random.Random(master_seed)``, the same scheme as ``runner.hand_seeds``."""
    rng = random.Random(master_seed)
    seeds = [rng.getrandbits(64) for _ in range(n_hands)]
    return [
        play_hand_multi(bots, seed, button=i % len(bots), stacks=stacks)
        for i, seed in enumerate(seeds)
    ]
