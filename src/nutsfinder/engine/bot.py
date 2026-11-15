"""The interface every bot implements, and what a bot is allowed to know.

A bot is asked for one decision at a time::

    action = bot.act(view, hole_cards)

- ``view`` is a :class:`PublicState`: the table as everyone can see it
  (stacks, bets, board, the full action history, the legal actions), plus
  ``view.seat``, the bot's own seat.
- ``hole_cards`` are the bot's own two cards.

That is *all* it gets. The opponent's cards and the deck stay inside the
hand runner, and everything handed over is immutable (frozen dataclasses
and tuples), so a bot cannot peek or cheat by editing what it was given.
``tests/test_info_leak.py`` checks this with a spy bot.

After every hand, the runner also calls ``bot.end_hand(summary)``. Bots that
learn about their opponent (Phase 5) use it; others ignore it (the default
does nothing). The summary only reveals the opponent's cards if they were
shown at a showdown, as at a real table.

Why an abstract base class rather than a ``Protocol``: a base class can
provide the do-nothing ``end_hand`` for free, so the simplest bot only has
to write ``act``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from nutsfinder.engine.actions import Action
from nutsfinder.engine.betting import PublicState
from nutsfinder.engine.cards import Card
from nutsfinder.engine.showdown import Settlement

__all__ = ["Bot", "HandSummary"]


@dataclass(frozen=True, slots=True)
class HandSummary:
    """What one seat learns when a hand ends.

    Attributes:
        seat: the seat this summary is for.
        final_view: the public view of the finished hand (full history and
            board), from this seat.
        shown_cards: each seat's hole cards if they were shown, else ``None``.
            Both hands are shown at a showdown; nothing is shown after a fold.
        result: the payout (winners, showdown ranks, net chips per seat).
    """

    seat: int
    final_view: PublicState
    shown_cards: tuple[tuple[Card, ...] | None, tuple[Card, ...] | None]
    result: Settlement


class Bot(ABC):
    """Base class for all bots. Subclasses must implement :meth:`act`."""

    @abstractmethod
    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        """Choose an action for ``state.seat``, which is the seat to act.

        Contract: return one of the actions allowed by ``state.legal``. An
        illegal action makes the hand runner raise ``ValueError``.
        """

    def end_hand(self, summary: HandSummary) -> None:
        """Called once after every hand. Does nothing unless overridden."""
