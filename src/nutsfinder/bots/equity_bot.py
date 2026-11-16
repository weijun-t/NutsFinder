"""EquityBot (tier 2): equity against an *assumed range*, then the max-EV action.

PotOddsBot treats every opponent hand as random. EquityBot makes three
improvements, each kept deliberately simple:

1. **The opponent's range depends on what they did.** Count the opponent's
   bets and raises so far this hand (calls and checks do not count; the
   blinds are forced, so they do not count either). Assume they hold the
   top ``w`` of hands by the preflop ranking (:mod:`nutsfinder.engine.preflop`)::

       bets/raises so far:   0      1     2     3 or more
       assumed range w:     100%   50%   30%   15%

   (``range_widths``.) So a raise makes us assume the top half, a re-raise
   the top 30%, and so on.
2. **Several bet sizes.** Besides fold and check/call it considers bets or
   raises of 1/2 pot and pot (``bet_fractions``) and all-in, and computes a
   one-street EV for each (formulas in :mod:`nutsfinder.bots.common`).
3. **A fold model, so bets can win the pot right away.** See below.

It then plays the action with the highest EV (ties go to the more passive
action: fold, then check/call, then the smaller bet).

The EVs (notation of :mod:`nutsfinder.bots.common`)
---------------------------------------------------
With ``e`` the equity against the assumed range R::

    EV(fold)  = 0
    EV(check) = e * P
    EV(call)  = e * (P + c) - c
    EV(bet)   = f * P + (1 - f) * (e_called * (P + x + r) - x)

With unequal stacks these use the effective numbers (see "The effective
stack" in :mod:`nutsfinder.bots.common`): every size, all-in included, is
capped at what the opponent can call, ``x`` counts only the chips that are
really at risk, and a call that is all-in for less uses the effective pot
``P_eff`` (the opponent's unmatched excess is refunded). The fold model
below then uses that same ``x``. With equal stacks nothing changes.

The fold model
--------------
Facing a bet that risks ``x`` to win ``P``, a pure bluff breaks even if
the opponent folds ``x / (P + x)`` of the time. To stop that, a balanced
defender continues with the *minimum defence frequency*
``MDF = P / (P + x)`` of their range. Example: a pot-sized bet (``x = P``)
gives MDF = 1/2; a half-pot bet gives MDF = 2/3. We use it twice:

- **Who calls.** The calling range is the strongest ``MDF`` share of R, i.e.
  the top ``w * MDF`` hands, rounded down to the nearest width in
  :data:`nutsfinder.engine.preflop.RANGE_GRID` (1%, 2%, 3%, 5%, 7.5%, 10%,
  15%, ..., 100%); ``e_called`` is our equity against that.
- **How often they fold.** A balanced defender folds ``1 - MDF``, but weak
  opponents fold far less (a calling station never folds), and assuming
  ``1 - MDF`` would make a huge overbet look like a free win (``1 - MDF``
  tends to 1 as ``x`` grows). So we cap it::

      f = min(max_fold, x / (P + x))        (max_fold = 0.1 by default)

Why so low a cap: none of the opponents it will meet folds like a balanced
player. CallStationBot and AlwaysRaiseBot never fold, RandomBot folds 1/3
of the time whatever the size, and PotOddsBot only folds hands with less
than pot-odds equity against a random hand. A quick scan (1,000 duplicate
pairs per setting, master seed 777, each opponent) of max_fold = 0, 0.1,
0.2, 0.3 and 0.5 showed that assuming more folds costs a lot: with 0.3 it
won +15 +/- 106 bb/100 against PotOddsBot and +65 +/- 220 against
AlwaysRaiseBot, with 0.1 it won +262 +/- 99 and +1,423 +/- 179. (Choosing
a default on the same data you measure with is a mild form of overfitting,
so the default was re-checked on a different seed, and the official
evaluation should use yet another one.)

Both choices are pessimistic: we assume the callers are the strong part of
the range, and that the opponent folds less than a balanced player would.
The model rewards betting when ``e_called`` is high (value) and only bets
a weak hand when the fold equity alone pays (``f * P`` beats checking).

Worked example (flop, pot P = 1,000, we may check, R = any hand, e = 0.70;
suppose e_called = 0.60 against the top 67% that call a half-pot bet)::

    EV(check)      = 0.70 * 1,000                                  = 700
    bet 500: f     = min(0.1, 500 / 1,500) = 0.1
    EV(bet 500)    = 0.1 * 1,000 + 0.9 * (0.60 * 2,000 - 500)      = 730

so it bets. ``tests/test_ev_bots.py`` checks numbers like these.

Equity estimates
----------------
- Preflop: a table lookup, :func:`nutsfinder.engine.preflop.equity_vs_top`
  (precomputed for every class and grid width; SE at most 0.71 points, 0.16
  against any hand). This is why ranges snap to a fixed grid.
- Flop/turn: one Monte Carlo run of ``samples`` (default 400) draws from R
  with the bot's own seeded RNG; the calling ranges are subsets of R, so the
  same run gives their equities too (rejection sampling, see
  :func:`nutsfinder.engine.equity.equity_vs_nested_ranges`). SE at most
  2.5 points against R; about ``1 / sqrt(MDF)`` times that for a calling
  range (3.1 points for a pot-sized bet). Tiny calling ranges (all-in into
  a small pot) are topped up to at least 100 samples (SE at most 5 points).
- River: the same, but each sample is just an opponent hand (nothing left
  to deal); exact if R has no more live combos than ``samples``.

Known weaknesses (it is tier 2, not the final bot)
--------------------------------------------------
- One street only: no implied odds, no plan for later streets, and checking
  is valued as if the hand were checked down (the opponent never bets).
- The opponent never raises in the model, so it can bet itself into
  re-raises without planning for them.
- Ranges come from the *preflop* ranking even on later streets: a flop bet
  narrows the range to "good preflop hands", not "hands that hit this board".
- The narrowing is the same for every opponent. Against a maniac who raises
  everything (AlwaysRaiseBot) it assumes a strong range and folds too much;
  against a calling station it still expects some folds. Fixing this needs
  opponent modelling (tier 3, BayesBot).
- No mixed strategy: the same spot always gets the same action (except for
  Monte Carlo noise), so a thinking opponent could read it.
- Because the EV of a bet is linear in its size once the fold cap binds, it
  tends to choose the smallest or the largest size (often all-in), which is
  high-variance.
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
    effective_max_to,
    effective_pot,
    label,
    pot_odds,
    raise_to_for_fraction,
)
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import PublicState, Street
from nutsfinder.engine.bot import Bot
from nutsfinder.engine.cards import Card
from nutsfinder.engine.equity import equity_vs_nested_ranges
from nutsfinder.engine.preflop import equity_vs_top, hand_class, snap_to_grid, top_range

__all__ = ["EquityBot", "fold_probability", "calling_share"]

log = logging.getLogger(__name__)

def calling_share(pot: int, added: int) -> float:
    """MDF = ``P / (P + x)``: the share of the range that continues against a bet of ``x``."""
    return pot / (pot + added)


def fold_probability(pot: int, added: int, max_fold: float) -> float:
    """``f = min(max_fold, x / (P + x))`` (see the module docstring)."""
    return min(max_fold, 1 - calling_share(pot, added))


class EquityBot(Bot):
    """Equity vs an action-dependent range; picks the max one-street-EV action.

    Args:
        seed: seeds the bot's RNG (used by the flop/turn Monte Carlo).
        range_widths: assumed range after 0, 1, 2, 3+ opponent bets/raises.
        bet_fractions: bet/raise sizes to consider, as fractions of the pot
            (after calling), as in :func:`raise_to_for_fraction`.
        consider_all_in: also consider moving all-in.
        max_fold: cap on the assumed fold probability.
        samples: Monte Carlo samples per flop/turn decision.
        time_budget: optional seconds per decision (stops the Monte Carlo
            early; makes results machine-dependent, so leave ``None`` for
            reproducible matches).
        debug: record each decision in :attr:`last_decision` and log it at
            DEBUG level (``logging.getLogger("nutsfinder.bots.equity_bot")``).
    """

    def __init__(
        self,
        seed: int | None = None,
        range_widths: tuple[float, ...] = (1.0, 0.5, 0.3, 0.15),
        bet_fractions: tuple[float, ...] = (0.5, 1.0),
        consider_all_in: bool = True,
        max_fold: float = 0.1,
        samples: int = 400,
        time_budget: float | None = None,
        debug: bool = False,
    ) -> None:
        self._rng = random.Random(seed)
        self.range_widths = range_widths
        self.bet_fractions = bet_fractions
        self.consider_all_in = consider_all_in
        self.max_fold = max_fold
        self.samples = samples
        self.time_budget = time_budget
        self.debug = debug
        self.last_decision: Decision | None = None

    def reseed(self, seed: int) -> None:
        """Restart the RNG from ``seed`` (see :meth:`Bot.reseed`)."""
        self._rng = random.Random(seed)

    # --- The model ---------------------------------------------------------

    def villain_range_width(self, state: PublicState) -> float:
        """The assumed opponent range ``w``, from their bets/raises this hand (snapped to the grid)."""
        aggressive = sum(
            1
            for rec in state.history
            if rec.player != state.seat and isinstance(rec.action, (Bet, Raise))
        )
        return snap_to_grid(self.range_widths[min(aggressive, len(self.range_widths) - 1)])

    def candidate_sizes(self, state: PublicState) -> list[int]:
        """Distinct legal bet amounts / raise-to totals to consider, smallest first.

        All are capped at :func:`effective_max_to` (what the opponent can
        call), so "all-in" means "put the opponent all-in" when they are the
        shorter stack, and sizes that collapse to the same amount appear once.
        """
        legal = state.legal
        if not (legal.can_bet or legal.can_raise):
            return []
        sizes = {raise_to_for_fraction(state, f) for f in self.bet_fractions}
        if self.consider_all_in:
            sizes.add(effective_max_to(state))
        return sorted(sizes)

    def range_equities(
        self, state: PublicState, hole_cards: tuple[Card, ...], widths: list[float]
    ) -> dict[float, tuple[float, float]]:
        """``{w: (equity, std_error)}`` against the top ``w`` of hands, for each grid width.

        ``widths[0]`` must be the largest (the assumed range R); the others
        are its calling sub-ranges.
        """
        if state.street == Street.PREFLOP:
            cls = hand_class(hole_cards)
            return {w: equity_vs_top(cls, w) for w in widths}
        deadline = None if self.time_budget is None else time.perf_counter() + self.time_budget
        results = equity_vs_nested_ranges(
            hole_cards,
            [top_range(w) for w in widths],
            state.board,
            self.samples,
            min_samples=100,
            rng=self._rng,
            deadline=deadline,
        )
        return {w: (r.equity, r.std_error) for w, r in zip(widths, results)}

    # --- Acting ------------------------------------------------------------

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        legal = check_turn(state)
        pot, call = state.pot, legal.call_amount
        # When our call is all-in for less, the opponent's unmatched excess is
        # refunded: price the call with the effective pot (== pot otherwise).
        pot_eff = effective_pot(state)
        width = self.villain_range_width(state)

        # For each bet size: chips we add (x), chips they need to call (r),
        # fold probability (f) and the size of the calling range.
        bets = []
        for to in self.candidate_sizes(state):
            x, r = chips_added(state, to), chips_to_call_raise(state, to)
            f = fold_probability(pot, x, self.max_fold)
            calling = snap_to_grid(width * calling_share(pot, x))
            bets.append((aggressive_action(legal, to), x, r, f, calling))

        needed = [width] + sorted({b[4] for b in bets} - {width}, reverse=True)
        equities = self.range_equities(state, hole_cards, needed)
        equity, se = equities[width]

        options: list[tuple[Action, float]] = []
        notes: list[str] = []
        if legal.can_call:
            options += [(Fold(), 0.0), (Call(), ev_call(equity, pot_eff, call))]
            if pot_eff != pot:
                notes.append(f"call is all-in for less: {pot - pot_eff} refundable, effective pot {pot_eff}")
        else:
            options += [(Check(), ev_check(equity, pot))]
        for action, x, r, f, calling in bets:
            e_called, se_called = equities[calling]
            options.append((action, ev_bet(pot, x, r, f, e_called)))
            notes.append(
                f"{label(action)}: fold {f:.0%}, calling range top {calling:.1%}, "
                f"equity when called {e_called:.1%} +/- {se_called:.1%}"
            )
        # max() keeps the first of equal values: the most passive action wins ties.
        action, _ = max(options, key=lambda option: option[1])

        if self.debug:
            self.last_decision = Decision(
                street=state.street.name.lower(),
                hole_cards=cards_text(hole_cards),
                board=cards_text(state.board),
                villain_range=width,
                equity=equity,
                std_error=se,
                pot=pot,
                to_call=call,
                pot_odds=pot_odds(call, pot_eff) if legal.can_call else None,
                evs=tuple((label(a), ev) for a, ev in options),
                action=action,
                notes=tuple(notes),
            )
            log.debug("%s", self.last_decision)
        return action

    def __repr__(self) -> str:
        return (
            f"EquityBot(range_widths={self.range_widths}, bet_fractions={self.bet_fractions}, "
            f"max_fold={self.max_fold}, samples={self.samples})"
        )
