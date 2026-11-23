"""RaiseOrFoldBot: bets or raises the pot whenever the opponent has not, and folds to any bet or raise.

A baseline built to *bait* an opponent model (Phase 7, "can the exploiter be
exploited back"). It ignores its cards. On every street:

- If the opponent has bet or raised on this street (a Bet or Raise by the
  other seat in this street's history; the blinds are not history entries),
  it **folds**.
- Otherwise it makes a **pot-sized** bet or raise when one is legal (exactly
  :class:`~nutsfinder.bots.always_raise.AlwaysRaiseBot`'s sizing, clamped to
  ``[min_to, max_to]``), else checks (or calls, when a call is all that is
  legal and nobody raised: e.g. the small blind with no raise possible).

So preflop it raises the small blind, raises a limp, and folds the big blind
to a raise; postflop it bets when checked to and folds to any bet or raise.

What a Beta-Binomial model (BayesBot) learns from it
----------------------------------------------------
- ``fold_pre``, ``fold_post``, ``fold_*_big``: every time BayesBot bets or
  raises, it folds, so these go towards 1.
- ``bluff``: the hands it shows at a showdown (only when BayesBot calls it
  down) are random, so on about half of its betting streets the shown hand is
  below the median: ``bluff`` -> ~0.5 and BayesBot's ``m`` ("its actions are
  random") -> ~1. That is the point: BayesBot believes a fold rate above its
  prior only "as far as m" (``trusted_fold_rate``), so an *honest* folder
  (a nit, m ~ 0) would not make it bluff more; a random-looking folder does.
- ``agg_*`` high, ``vpip`` about 0.5.

The intended punishment is a switch to :class:`CallStationBot`: the model then
says "bet anything, they fold", and the station calls every bet.

It has no randomness, so it takes no seed.
"""

from __future__ import annotations

from nutsfinder.bots.always_raise import pot_sized_to
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import PublicState
from nutsfinder.engine.bot import Bot
from nutsfinder.engine.cards import Card

__all__ = ["RaiseOrFoldBot"]


class RaiseOrFoldBot(Bot):
    """Fold if the opponent bet/raised this street; else pot-sized bet/raise if legal; else check/call."""

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        legal = state.legal
        if legal is None or legal.player != state.seat:
            raise ValueError(f"seat {state.seat} was asked to act out of turn")
        opponent_aggressed = any(
            r.street == state.street and r.player != state.seat and isinstance(r.action, (Bet, Raise))
            for r in state.history
        )
        if opponent_aggressed:
            return Fold()
        if legal.can_bet or legal.can_raise:
            size = min(max(pot_sized_to(state), legal.min_to), legal.max_to)
            return Bet(size) if legal.can_bet else Raise(size)
        return Check() if legal.can_check else Call()

    def __repr__(self) -> str:
        return "RaiseOrFoldBot()"
