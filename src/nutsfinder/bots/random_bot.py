"""RandomBot: plays a uniformly random legal action. The simplest baseline.

How it chooses, in two steps:

1. Pick an action *kind* uniformly from the legal ones. There are at most
   three: fold, check-or-call (only one of the two is ever legal), and
   bet-or-raise (only one of the two, and only if allowed). So with all three
   available each has probability 1/3. Fold is always included, even when
   checking is free, because the rules allow it; that makes RandomBot a
   genuinely bad player, which is what a baseline should be.
2. For a bet or raise, pick the "to" amount uniformly from every whole number
   in ``[min_to, max_to]``.

It has its own seeded random number generator, so given the same seed it
makes the same choices in the same situations, and it never touches the
global :mod:`random` state.
"""

from __future__ import annotations

import random

from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import PublicState
from nutsfinder.engine.bot import Bot
from nutsfinder.engine.cards import Card

__all__ = ["RandomBot"]


class RandomBot(Bot):
    """Uniform over legal action kinds, then uniform over legal sizes."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        """Restart the RNG from ``seed`` (see :meth:`Bot.reseed`)."""
        self._rng = random.Random(seed)

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        legal = state.legal
        if legal is None or legal.player != state.seat:
            raise ValueError(f"seat {state.seat} was asked to act out of turn")
        kinds = ["fold", "check" if legal.can_check else "call"]
        if legal.can_bet or legal.can_raise:
            kinds.append("bet" if legal.can_bet else "raise")
        kind = self._rng.choice(kinds)
        if kind in ("bet", "raise"):
            size = self._rng.randint(legal.min_to, legal.max_to)  # inclusive
            return Bet(size) if kind == "bet" else Raise(size)
        return {"fold": Fold(), "check": Check(), "call": Call()}[kind]

    def __repr__(self) -> str:
        return "RandomBot()"
