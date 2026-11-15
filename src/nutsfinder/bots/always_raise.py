"""AlwaysRaiseBot: bets or raises the size of the pot every time it can.

A maximally aggressive baseline. Whenever a bet or raise is legal it makes a
*pot-sized* one; when it is not (for example the opponent is already
all-in, or it cannot afford more than a call) it calls, or checks if there
is nothing to call. It never folds.

What "pot-sized" means here
---------------------------
``state.pot`` is every chip put in so far, *including* the bets already made
on this street (it is ``total_bets[0] + total_bets[1]``).

- **Bet** (nobody has bet this street, so nothing is owed)::

      bet amount = pot

- **Raise** (facing a bet). The standard pot-sized raise is "first call,
  then raise by the size of the pot that is then in the middle"::

      pot after calling = pot + call
      raise to          = current_bet + pot after calling
                        = current_bet + pot + call

  where ``call = current_bet - my street bet`` (``legal.call_amount``).

Then the result is clamped to the legal range: at least ``legal.min_to``
(the minimum raise) and at most ``legal.max_to`` (all-in). So a short stack
whose pot-sized raise would cost more than it has simply goes all-in.

Worked examples (blinds 50/100, stacks 10,000):

- Preflop, small blind to act: SB has 50 in, BB has 100 in. pot = 150,
  current bet = 100, call = 50. Pot after calling = 200, so raise to
  100 + 200 = **300**.
- Flop, pot 1,000, nobody has bet: bet **1,000**.
- Flop, pot was 1,000 and the opponent bets 500: now pot = 1,500,
  current bet = 500, call = 500. Pot after calling = 2,000, so raise to
  500 + 2,000 = **2,500**.

It has no randomness, so it takes no seed.
"""

from __future__ import annotations

from nutsfinder.engine.actions import Action, Bet, Call, Check, Raise
from nutsfinder.engine.betting import PublicState
from nutsfinder.engine.bot import Bot
from nutsfinder.engine.cards import Card

__all__ = ["AlwaysRaiseBot", "pot_sized_to"]


def pot_sized_to(state: PublicState) -> int:
    """The unclamped pot-sized bet amount / raise-to total for the seat to act.

    Bet: ``pot``. Raise: ``current_bet + pot + call``. (When not facing a bet
    the call is 0 and the current bet is 0, so both cases are the same
    formula.) See the module docstring for worked examples.
    """
    assert state.legal is not None
    call = state.legal.call_amount
    return state.current_bet + state.pot + call


class AlwaysRaiseBot(Bot):
    """Pot-sized bet or raise when legal (clamped to [min_to, max_to]), else call/check."""

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        legal = state.legal
        if legal is None or legal.player != state.seat:
            raise ValueError(f"seat {state.seat} was asked to act out of turn")
        if legal.can_bet or legal.can_raise:
            size = min(max(pot_sized_to(state), legal.min_to), legal.max_to)
            return Bet(size) if legal.can_bet else Raise(size)
        return Check() if legal.can_check else Call()

    def __repr__(self) -> str:
        return "AlwaysRaiseBot()"
