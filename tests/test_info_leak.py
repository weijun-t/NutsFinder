"""Bots must never see the opponent's hole cards or the undealt deck.

A ``SpyBot`` plays randomly but keeps everything it is ever handed: every
``act`` call's arguments and every end-of-hand summary. After each hand we
walk through *every object reachable* from what it received (dataclass
fields, tuples, lists, dicts, ...) and check:

- no ``Deck`` or ``HandState`` is reachable (those hold hidden cards);
- during the hand, every card it saw is either its own hole card or a board
  card that had already been dealt at that moment;
- so the opponent's hole cards and the undealt cards never appear;
- after the hand, the opponent's cards are revealed only after a showdown.

Walking the object graph, rather than checking named fields, means a new
field added to ``PublicState`` that leaks cards would be caught too.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest

from nutsfinder.bots.random_bot import RandomBot
from nutsfinder.engine.actions import Action
from nutsfinder.engine.betting import HandState, PublicState
from nutsfinder.engine.bot import HandSummary
from nutsfinder.engine.cards import FULL_DECK, Card, Deck
from nutsfinder.engine.hand_log import HandLog
from nutsfinder.engine.runner import hand_seeds, play_hand


class SpyBot(RandomBot):
    """A RandomBot that records everything it is given."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.seen_acts: list[tuple[PublicState, tuple[Card, ...]]] = []
        self.seen_ends: list[HandSummary] = []

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        self.seen_acts.append((state, hole_cards))
        return super().act(state, hole_cards)

    def end_hand(self, summary: HandSummary) -> None:
        self.seen_ends.append(summary)


def reachable(obj: object) -> Iterator[object]:
    """Every object reachable from ``obj`` through containers and dataclasses."""
    stack, seen = [obj], set()
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        yield x
        if isinstance(x, Card | str | int | float | bool | type(None)):
            continue
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            stack.extend(getattr(x, f.name) for f in dataclasses.fields(x))
        elif isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())
        elif isinstance(x, tuple | list | set | frozenset):
            stack.extend(x)
        else:
            stack.extend(vars(x).values() if hasattr(x, "__dict__") else ())


def cards_in(obj: object) -> set[Card]:
    found = set()
    for x in reachable(obj):
        assert not isinstance(x, Deck | HandState), f"bot was handed a {type(x).__name__}"
        if isinstance(x, Card):
            found.add(x)
    return found


def check_spy(spy: SpyBot, seat: int, log: HandLog, acts: list, end: HandSummary) -> None:
    own = set(log.hole_cards[seat])
    opponent = set(log.hole_cards[1 - seat])
    undealt = set(FULL_DECK) - own - opponent - set(log.board)
    for view, hole in acts:
        assert view.seat == seat and set(hole) == own
        # The board shown is exactly the cards dealt so far, in order.
        assert view.board == log.board[: len(view.board)]
        seen = cards_in((view, hole))
        assert seen <= own | set(view.board), seen - own - set(view.board)
        assert not seen & opponent and not seen & undealt
    seen_end = cards_in(end)
    assert end.seat == seat
    assert not seen_end & undealt
    if log.folded is None:
        assert end.shown_cards == log.hole_cards  # showdown: both hands shown
    else:
        assert end.shown_cards == (None, None)
        assert not seen_end & opponent


def test_spy_bots_never_see_hidden_cards():
    spies = [SpyBot(1), SpyBot(2)]
    showdowns = folds = 0
    for i, seed in enumerate(hand_seeds(11, 500)):
        log = play_hand(spies, seed, button=i % 2)
        for seat, spy in enumerate(spies):
            assert len(spy.seen_ends) == 1  # exactly one summary per hand
            check_spy(spy, seat, log, spy.seen_acts, spy.seen_ends[0])
            spy.seen_acts.clear()
            spy.seen_ends.clear()
        showdowns += log.folded is None
        folds += log.folded is not None
    assert showdowns > 0 and folds > 0  # both endings were exercised


def test_reachable_finds_nested_cards():
    # Sanity check of the checker itself: it must find a card hidden deep down.
    card = FULL_DECK[0]
    assert card in cards_in({"a": [(1, {"b": (card,)})]})


def test_checker_catches_a_leak():
    # Mutation check: hand a spy a view that smuggles in the opponent's card
    # (tucked into the history, where no field-name check would look) and the
    # object-graph walker must find it.
    spies = [SpyBot(1), SpyBot(2)]
    log = play_hand(spies, seed=3)
    view, hole = spies[0].seen_acts[0]
    leaked = dataclasses.replace(view, history=(*view.history, log.hole_cards[1][0]))
    with pytest.raises(AssertionError):
        check_spy(spies[0], 0, log, [(leaked, hole)], spies[0].seen_ends[0])
