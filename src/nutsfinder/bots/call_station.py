"""CallStationBot: checks when it can, otherwise calls. A passive baseline.

It never folds, never bets and never raises, whatever its cards. So it
always goes to showdown unless the opponent folds, and it never wins a pot
the opponent would not have handed it. Beating it is easy for any bot that
bets its good hands and checks its bad ones ("value betting"), which is
exactly why it is a useful first opponent.

It has no randomness, so it takes no seed: the same situation always gets
the same answer.
"""

from __future__ import annotations

from nutsfinder.engine.actions import Action, Call, Check
from nutsfinder.engine.betting import PublicState
from nutsfinder.engine.bot import Bot
from nutsfinder.engine.cards import Card

__all__ = ["CallStationBot"]


class CallStationBot(Bot):
    """Check if checking is legal, otherwise call."""

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        legal = state.legal
        if legal is None or legal.player != state.seat:
            raise ValueError(f"seat {state.seat} was asked to act out of turn")
        # Exactly one of can_check / can_call is true (see legal_actions).
        return Check() if legal.can_check else Call()

    def __repr__(self) -> str:
        return "CallStationBot()"
