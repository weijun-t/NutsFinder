"""PotOddsBot (tier 1): calls when its equity beats the pot odds, bets its strong hands.

It estimates one number, its equity ``e`` against a *uniformly random*
opponent hand, and follows three rules (notation as in
:mod:`nutsfinder.bots.common`: ``P`` is the pot including any bet we face,
``c`` the cost of calling):

1. **Facing a bet.** If ``e > raise_threshold`` (default 0.70) and raising is
   legal, raise pot-sized: to ``current_bet + (P + c)``. Otherwise call if
   ``e > c / (P + c)`` (the pot odds: exactly when ``EV(call) = e (P + c) - c``
   is positive), else fold. Worked example: the pot was 1,000 and the
   opponent bets 500, so ``P = 1,500``, ``c = 500`` and the pot odds are
   500 / 2,000 = 25%. With 30% equity: call (EV = 0.30 * 2,000 - 500 = +100).
   With 20%: fold (EV of calling would be -100). Exactly 25%: fold (EV 0;
   ties go to the cheaper option).
2. **Not facing a bet.** If ``e > bet_threshold`` (default 0.60) and betting
   (or, for the big blind after a limp, raising) is legal, bet 2/3 of the
   pot (``bet_fraction``): to ``current_bet + 2/3 P``. Otherwise check.
3. It never folds when it can check.

Why these thresholds: against a random hand, 50% equity is "average". A
bot that bets above 60% is betting hands that are clearly ahead of what an
opponent who calls everything holds, which is the textbook value bet. A
raise puts more money in when the opponent has *already* shown interest, so
the bar is higher (70%).

Equity estimates
----------------
- Preflop: the precomputed table (:data:`nutsfinder.engine.preflop.PREFLOP_EQUITY`),
  SE below 0.16 percentage points. Free.
- Flop and turn: Monte Carlo against a random hand with ``samples`` draws
  (default 300) from the bot's own seeded RNG. The SE is at most
  0.5 / sqrt(300) = 2.9 points (at 50% equity; smaller near 0 or 1).
  About 3.3 ms per decision (300 samples at about 11 us each).
- River: nothing left to deal, so each sample is one random opponent hand
  (same SE bound, about 1.5 ms). With ``samples >= 990`` it is exact instead.

What it ignores, on purpose (this is tier 1): the opponent's actions (a
raise from a tight player means something; here it does not), position,
implied odds and future streets, and bluffing (it never bets a weak hand).
So it is easy to exploit: bet big and it folds every hand below the pot odds;
check behind and it never bluffs you off a hand.

The "EV" numbers in its decision log are the one-street formulas of
:mod:`nutsfinder.bots.common` with the opponent assumed to always call
("EV if called"). PotOddsBot does not choose by them (it uses the threshold
rules above); they are there so every decision can be explained.
"""

from __future__ import annotations

import logging
import random
import time

from nutsfinder.bots.common import (
    Decision,
    aggressive_action,
    cards_text,
    check_turn,
    chips_added,
    chips_to_call_raise,
    ev_bet,
    ev_call,
    ev_check,
    label,
    pot_odds,
    raise_to_for_fraction,
)
from nutsfinder.engine.actions import Action, Call, Check, Fold
from nutsfinder.engine.betting import PublicState, Street
from nutsfinder.engine.bot import Bot, HandSummary
from nutsfinder.engine.cards import Card
from nutsfinder.engine.equity import RANDOM_RANGE, equity_vs_range
from nutsfinder.engine.preflop import PREFLOP_EQUITY, PREFLOP_SE, hand_class

__all__ = ["PotOddsBot"]

log = logging.getLogger(__name__)


class PotOddsBot(Bot):
    """Equity vs a random hand; call above the pot odds, bet/raise above thresholds.

    Args:
        seed: seeds the bot's RNG (used only by the Monte Carlo equity).
        bet_threshold: bet when not facing a bet and ``e`` is above this.
        raise_threshold: raise when facing a bet and ``e`` is above this.
        bet_fraction: bet size as a fraction of the pot (default 2/3).
        raise_fraction: raise size as a fraction of the pot after calling
            (default 1, pot-sized).
        samples: Monte Carlo samples per flop/turn decision.
        time_budget: optional seconds per decision. If set, the Monte Carlo
            stops early when time runs out. That makes results depend on the
            machine's speed, so leave it ``None`` for reproducible matches.
        debug: record each decision in :attr:`last_decision` and log it at
            DEBUG level (``logging.getLogger("nutsfinder.bots.pot_odds")``).
    """

    def __init__(
        self,
        seed: int | None = None,
        bet_threshold: float = 0.60,
        raise_threshold: float = 0.70,
        bet_fraction: float = 2 / 3,
        raise_fraction: float = 1.0,
        samples: int = 300,
        time_budget: float | None = None,
        debug: bool = False,
    ) -> None:
        self._rng = random.Random(seed)
        self.bet_threshold = bet_threshold
        self.raise_threshold = raise_threshold
        self.bet_fraction = bet_fraction
        self.raise_fraction = raise_fraction
        self.samples = samples
        self.time_budget = time_budget
        self.debug = debug
        self.last_decision: Decision | None = None
        self._cache: dict[tuple, tuple[float, float]] = {}

    def reseed(self, seed: int) -> None:
        """Restart the RNG from ``seed`` (see :meth:`Bot.reseed`)."""
        self._rng = random.Random(seed)
        self._cache.clear()

    def end_hand(self, summary: HandSummary) -> None:
        """Forget this hand's cached equities."""
        self._cache.clear()

    def estimate_equity(self, state: PublicState, hole_cards: tuple[Card, ...]) -> tuple[float, float]:
        """``(equity, std_error)`` against a uniformly random hand (see the module docstring).

        The answer depends only on our cards and the board, so within a hand
        it is computed once per street and reused (e.g. when we check and then
        face a bet). The cache is emptied by :meth:`reseed` and :meth:`end_hand`,
        so it never carries over from one hand to the next.
        """
        if state.street == Street.PREFLOP:
            cls = hand_class(hole_cards)
            return PREFLOP_EQUITY[cls], PREFLOP_SE[cls]
        key = (tuple(hole_cards), state.board)
        if key not in self._cache:
            deadline = None if self.time_budget is None else time.perf_counter() + self.time_budget
            r = equity_vs_range(
                hole_cards, RANDOM_RANGE, state.board, self.samples, rng=self._rng, deadline=deadline
            )
            self._cache[key] = (r.equity, r.std_error)
        return self._cache[key]

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        legal = check_turn(state)
        equity, se = self.estimate_equity(state, hole_cards)
        pot, call = state.pot, legal.call_amount
        can_aggress = legal.can_bet or legal.can_raise

        if legal.can_call:
            odds = pot_odds(call, pot)
            if can_aggress and equity > self.raise_threshold:
                action = aggressive_action(legal, raise_to_for_fraction(state, self.raise_fraction))
            elif equity > odds:
                action = Call()
            else:
                action = Fold()
        else:
            odds = None
            if can_aggress and equity > self.bet_threshold:
                action = aggressive_action(legal, raise_to_for_fraction(state, self.bet_fraction))
            else:
                action = Check()

        if self.debug:
            self._record(state, hole_cards, equity, se, odds, action)
        return action

    def _record(self, state, hole_cards, equity, se, odds, action) -> None:
        """Store (and log) the decision with the "EV if called" of each option."""
        legal = state.legal
        pot, call = state.pot, legal.call_amount
        evs: list[tuple[str, float]] = []
        if legal.can_call:
            evs += [("fold", 0.0), ("call", ev_call(equity, pot, call))]
        else:
            evs += [("check", ev_check(equity, pot))]
        if legal.can_bet or legal.can_raise:
            for fraction in sorted({self.bet_fraction, self.raise_fraction}):
                to = raise_to_for_fraction(state, fraction)
                x, r = chips_added(state, to), chips_to_call_raise(state, to)
                name = label(aggressive_action(legal, to))
                evs.append((name, ev_bet(pot, x, r, fold_prob=0.0, equity_called=equity)))
        self.last_decision = Decision(
            street=state.street.name.lower(),
            hole_cards=cards_text(hole_cards),
            board=cards_text(state.board),
            villain_range=1.0,
            equity=equity,
            std_error=se,
            pot=pot,
            to_call=call,
            pot_odds=odds,
            evs=tuple(evs),
            action=action,
            notes=("bet/raise EVs assume the opponent always calls",),
        )
        log.debug("%s", self.last_decision)

    def __repr__(self) -> str:
        return (
            f"PotOddsBot(bet_threshold={self.bet_threshold}, raise_threshold={self.raise_threshold}, "
            f"samples={self.samples})"
        )
