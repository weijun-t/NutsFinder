"""The multiway bot interface, and what a bot may know.

A bot is asked for one decision at a time::

    action = bot.act(view, hole_cards)

``view`` is a :class:`MultiPublicState` (the table as everyone sees it,
plus ``view.seat`` and the legal actions); ``hole_cards`` are the bot's own
two cards. Opponents' cards and the deck stay in the runner, and everything
handed over is immutable.

After every hand the runner calls ``bot.end_hand(summary)``. The summary
reveals the hole cards of exactly the players who reached a showdown: at a
showdown **every live hand is revealed to every bot** (no mucking, even for
a losing hand), the same as the heads-up engine, which shows both hands.
Folded hands, and the winner's hand when everyone else folded, are never
shown.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from nutsfinder.engine.actions import Action
from nutsfinder.engine.cards import Card
from nutsfinder.engine.multiway.showdown import MultiSettlement
from nutsfinder.engine.multiway.state import MultiPublicState

__all__ = ["MultiBot", "MultiHandSummary"]


@dataclass(frozen=True, slots=True)
class MultiHandSummary:
    """What one seat learns when a hand ends.

    Attributes:
        seat: the seat this summary is for.
        final_view: the public view of the finished hand from this seat.
        shown_cards: per seat, hole cards if shown at a showdown, else ``None``.
        result: the payout (pots, winners, ranks, nets).
    """

    seat: int
    final_view: MultiPublicState
    shown_cards: tuple[tuple[Card, ...] | None, ...]
    result: MultiSettlement


class MultiBot(ABC):
    """Base class for multiway bots. Subclasses must implement :meth:`act`."""

    @abstractmethod
    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:
        """Choose an action for ``view.seat`` (which is ``view.to_act``).

        Must return an action allowed by ``view.legal``; an illegal one makes
        the runner raise ``ValueError``.
        """

    def end_hand(self, summary: MultiHandSummary) -> None:
        """Called once after every hand. Does nothing unless overridden."""

    def reseed(self, seed: int) -> None:
        """Restart the bot's RNG from ``seed``; bots without randomness ignore it."""
