"""MultiPotOddsBot: PotOddsBot (tier 1) for 2-8 players.

It estimates its pot share ``e`` against **k uniformly random hands**, one per
live opponent, and applies PotOddsBot's three rules:

1. **Facing a bet.** If ``e > raise_threshold_k`` and raising is legal,
   raise pot-sized. Otherwise call if ``EV(call) > 0``, else fold.
   ``EV(call)`` uses the side-pot-aware one-street model of
   :mod:`nutsfinder.bots.multiway.common` (opponents still to act who owe
   the bet are assumed to call it). With a single pot this is exactly
   ``e > c / (P_win + c)``: the pot odds, against the pot we can win.
2. **Not facing a bet.** Bet 2/3 pot if ``e > bet_threshold_k``, else check.
3. It never folds when it can check.

Thresholds scale with the number of opponents
---------------------------------------------
Against k random hands the "average" share is ``1 / (k + 1)``, not 1/2.
PotOddsBot's thresholds (0.60 / 0.70 heads-up) are 1.2× and 1.4× the fair
share there, and that is what is kept::

    threshold_k = threshold × 2 / (k + 1)        (k = 1: unchanged)

Why the fair share is the right scale: against opponents who all call (the
model of a random hand that never folds) a bet of ``x`` gains
``e (k + 1) x - x`` over checking, which is positive exactly when
``e > 1 / (k + 1)``. The 1.2× margin is PotOddsBot's. Example: 6-handed
(k = 5) it bets above 20% and raises above 23.3%; AA (49% vs 5) raises,
and 72o (about 9%) folds to any bet that needs more than 9%.

Equity estimates
----------------
- Preflop: the 169 × 7 table of :func:`nutsfinder.bots.multiway.equity.preflop_vs_random`
  (deterministic; no RNG used). Side pots need the share against *subsets*
  of the opponents: against j random hands that is the table's k = j entry.
- Postflop: one vectorised Monte Carlo run of ``samples`` deals (default
  600, SE at most 2 points) from the bot's own seeded generator; side pots
  read subsets of the same deals. Cached per board within a hand.

As HU, the bet/raise EVs in the debug record assume every opponent calls
("EV if called"); the bot does not choose by them.
"""

from __future__ import annotations

import logging
import time
from collections import Counter

import numpy as np

from nutsfinder.bots.multiway.common import (
    MultiDecision,
    aggressive_action,
    cards_text,
    check_turn,
    ev_aggressive,
    ev_call,
    ev_check,
    label,
    live_opponents,
    raise_to_for_fraction,
)
from nutsfinder.bots.multiway.equity import MultiwayDeals, preflop_vs_random, sample_deals
from nutsfinder.engine.actions import Action, Call, Check, Fold
from nutsfinder.engine.betting import Street
from nutsfinder.engine.cards import Card
from nutsfinder.engine.multiway.bot import MultiBot, MultiHandSummary
from nutsfinder.engine.multiway.state import MultiPublicState

__all__ = ["MultiPotOddsBot", "scaled_threshold", "table_share"]

log = logging.getLogger(__name__)


def scaled_threshold(threshold: float, k: int) -> float:
    """``threshold × 2 / (k + 1)``: the heads-up threshold as a multiple of the fair share."""
    return threshold * 2.0 / (k + 1)


def table_share(hole_cards: tuple[Card, ...]):  # type: ignore[no-untyped-def]
    """Preflop share function: against j random opponents, the table entry for k = j."""

    def share(opps: tuple[int, ...]) -> float:
        return preflop_vs_random(hole_cards, len(opps))[0] if opps else 1.0

    return share


def deals_share(deals: MultiwayDeals, columns: dict[int, int]):  # type: ignore[no-untyped-def]
    """Share function on sampled deals: opponent seats -> their columns in ``deals``."""

    def share(opps: tuple[int, ...]) -> float:
        return deals.share_mean([columns[s] for s in opps])

    return share


class MultiPotOddsBot(MultiBot):
    """Pot share vs k random hands; call when EV(call) > 0, bet/raise above scaled thresholds.

    Args:
        seed: seeds the numpy generator (postflop Monte Carlo only).
        bet_threshold, raise_threshold: heads-up values; scaled by ``2 / (k + 1)``.
        bet_fraction, raise_fraction: sizes as fractions of the pot after calling.
        samples: Monte Carlo deals per postflop decision.
        time_budget: optional seconds per decision (stops the Monte Carlo early;
            machine-dependent, leave ``None`` for reproducible matches).
        debug: record each decision in :attr:`last_decision` (a :class:`MultiDecision`).
    """

    mode = "pot-odds"

    def __init__(self, seed: int | None = None, bet_threshold: float = 0.60, raise_threshold: float = 0.70,
                 bet_fraction: float = 2 / 3, raise_fraction: float = 1.0, samples: int = 600,
                 time_budget: float | None = None, debug: bool = False) -> None:
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self.bet_threshold = bet_threshold
        self.raise_threshold = raise_threshold
        self.bet_fraction = bet_fraction
        self.raise_fraction = raise_fraction
        self.samples = samples
        self.time_budget = time_budget
        self.debug = debug
        self.last_decision: MultiDecision | None = None
        self.stats: Counter[str] = Counter()
        """Running totals: decisions, Monte Carlo runs (``deals``), ``rows``, ``fallback_rows``, ``widened``."""
        self._cache: dict[tuple, MultiwayDeals] = {}

    def reseed(self, seed: int) -> None:
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._cache.clear()

    def end_hand(self, summary: MultiHandSummary) -> None:
        self._cache.clear()

    def shares(self, view: MultiPublicState, hole_cards: tuple[Card, ...]):  # type: ignore[no-untyped-def]
        """``(share function, equity, se, samples)`` against random hands for the live opponents."""
        opps = live_opponents(view)
        if view.street == Street.PREFLOP:
            eq, se = preflop_vs_random(hole_cards, len(opps))
            return table_share(hole_cards), eq, se, 0
        key = (tuple(hole_cards), view.board, tuple(opps))
        deals = self._cache.get(key)
        if deals is None:
            deadline = None if self.time_budget is None else time.perf_counter() + self.time_budget
            deals = sample_deals(hole_cards, view.board, [None] * len(opps), self.samples, self._rng, deadline)
            self._cache[key] = deals
            self.stats.update(deals=1, rows=deals.n, fallback_rows=deals.fallback_rows, widened=len(deals.widened))
        r = deals.equity()
        return deals_share(deals, {s: i for i, s in enumerate(opps)}), r.equity, r.std_error, r.samples

    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:
        start = time.perf_counter()
        legal = check_turn(view)
        self.stats["decisions"] += 1
        k = len(live_opponents(view))
        share, equity, se, n = self.shares(view, hole_cards)
        can_aggress = legal.can_bet or legal.can_raise
        if legal.can_call:
            ev_c, winnable = ev_call(view, share)
            odds: float | None = legal.call_amount / winnable if winnable else 1.0
            if can_aggress and equity > scaled_threshold(self.raise_threshold, k):
                action: Action = aggressive_action(legal, raise_to_for_fraction(view, self.raise_fraction))
            elif ev_c > 0:
                action = Call()
            else:
                action = Fold()
        else:
            odds = None
            if can_aggress and equity > scaled_threshold(self.bet_threshold, k):
                action = aggressive_action(legal, raise_to_for_fraction(view, self.bet_fraction))
            else:
                action = Check()
        if self.debug:
            self._record(view, hole_cards, share, equity, se, n, odds, action, k, time.perf_counter() - start)
        return action

    def _record(self, view, hole_cards, share, equity, se, n, odds, action, k, seconds) -> None:  # type: ignore[no-untyped-def]
        legal = view.legal
        evs: list[tuple[str, float]] = []
        if legal.can_call:
            ev_c, winnable = ev_call(view, share)
            evs += [("fold", 0.0), ("call", ev_c)]
        else:
            winnable = view.pot
            evs.append(("check", ev_check(view, share)))
        if legal.can_bet or legal.can_raise:
            for fraction in sorted({self.bet_fraction, self.raise_fraction}):
                to = raise_to_for_fraction(view, fraction)
                ev, _ = ev_aggressive(view, to, {}, share)
                evs.append((label(aggressive_action(legal, to)), ev))
        notes = [
            "bet/raise EVs assume every opponent calls",
            f"thresholds for k={k}: bet > {scaled_threshold(self.bet_threshold, k):.1%}, "
            f"raise > {scaled_threshold(self.raise_threshold, k):.1%}",
        ]
        if len(view.pots) > 1:
            notes.append("pots: " + ", ".join(f"{p.amount} {list(p.eligible)}" for p in view.pots))
        self.last_decision = MultiDecision(
            street=view.street.name.lower(), hole_cards=cards_text(hole_cards), board=cards_text(view.board),
            mode=self.mode, n_opponents=k,
            opponents=tuple(f"seat {s} ({view.positions[s]}): random hand" for s in live_opponents(view)),
            equity=equity, std_error=se, samples=n, pot=view.pot, winnable_pot=winnable,
            to_call=legal.call_amount, pot_odds=odds, evs=tuple(evs), action=action, seconds=seconds,
            notes=tuple(notes),
        )
        log.debug("%s", self.last_decision)

    def __repr__(self) -> str:
        return (f"MultiPotOddsBot(bet_threshold={self.bet_threshold}, raise_threshold={self.raise_threshold}, "
                f"samples={self.samples})")
