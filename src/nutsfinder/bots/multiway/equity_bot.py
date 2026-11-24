"""MultiEquityBot: EquityBot (tier 2) for 2-8 players.

Same three ideas as :mod:`nutsfinder.bots.equity_bot`, generalised:

1. **One range per opponent**, from *that* opponent's bets and raises this
   hand (blinds, checks and calls do not count): the top ``w`` of hands by
   the preflop ranking, with ``w = 100%, 50%, 30%, 15%`` after 0, 1, 2, 3+
   bets/raises (``range_widths``, EquityBot's numbers).
2. **Several sizes:** 1/2 pot and pot (``bet_fractions``) and all-in, capped
   at what the deepest live opponent can match.
3. **A fold model per opponent:** facing our bet of ``x`` into ``P``, opponent
   ``j`` folds with ``f_j = min(max_fold, x / (P + x))`` and continues with
   the top ``MDF = P / (P + x)`` of its range (snapped down to the range
   grid): EquityBot's numbers, applied to each opponent separately.

The EVs are the side-pot-aware one-street formulas of
:mod:`nutsfinder.bots.multiway.common`. The central multiway point: a bet
**wins the pot at once only if every opponent who can act folds**,
``Π_j f_j`` (independence assumed), and every other combination of
folders and callers is valued explicitly: ``EV(bet) = Σ_F P(F) (W_F - x)``
over the 2^d subsets ``F`` of the d deciders. With the default cap of 10%
per opponent, a bet into 3 opponents takes the pot down 0.1% of the time,
so bluffs essentially disappear multiway (they are only worth it against
one or two opponents), which is the standard multiway advice.

It plays the action with the highest EV; ties go to the more passive
action (fold, check/call, then the smaller size).

Callers are never weaker than the range
---------------------------------------
The calling ranges are pessimistic on purpose (the top MDF of each
range), while the fold model lets each opponent continue up to 90% of the
time. Heads-up that mismatch only ever *lowers* our equity when called.
Multiway it can raise it: against five callers who each hold the top 1-2%
(AA-QQ), weak hands gain, because the overpairs block each other's sets
and our straights and flushes win against all of them at once. In the
review's spot UTG 8d2d had 13.9% against five random hands but "18.7%
when called", so the all-in looked +382 chips and it shoved (below its
fair share 1/6). So the share used in the called branch, for every pot
and every subset of callers, is capped at the share against the same
opponents' full assumed ranges: ``min(share vs calling ranges, share vs
ranges)``. That keeps the intended pessimism and changes nothing where
the calling ranges were already the worse case (all heads-up spots, in
practice). Regression test: ``test_no_shove_below_fair_share``.

The all-in gate
---------------
In the one-street model a called bet is worth ``e_called × (pot after
calls) - x``, which is *linear* in the size ``x``: once our equity against
the callers beats our fair share ``1 / (callers + 1)``, the biggest size
wins, i.e. all-in, however deep the stacks. The model has no
continuation value for the passive line (checking/calling keeps the later
streets, where we can still get the money in when we are ahead) and
values a 100-big-blind jam with an equity measured against hands that
would mostly fold or out-play us later. The review measured the damage
against five calling stations: MultiBayesBot, which learns that nobody
folds, shoved preflop 619 times (mean equity 22% against five random
hands, barely above the 1/6 fair share) against MultiEquityBot's 129, and
lost about 930 bb/100 to it; with the all-in removed from both it won
about +150 instead.

The fix is deliberately simple: the all-in is a candidate only when it is
at most ``all_in_max_spr`` (default 3) pots beyond a call, i.e.
``(eff_to - current_bet) <= 3 (P + c)``, a stack-to-pot ratio of about 3
or less. At those ratios a jam is not much bigger than the pot-sized bet,
so the linear-in-x error is small. Preflop at 100bb (ratio about 40) it
never shoves first in; it still can after the pot has grown. The 1/2-pot
and pot sizes are always considered. ``all_in_max_spr=None`` restores the
ungated behaviour. (The heads-up EquityBot is unchanged.)

Equity estimates (all vectorised Monte Carlo, :mod:`nutsfinder.bots.multiway.equity`)
--------------------------------------------------------------------------------------
- Check / call: one run of ``samples`` deals against the assumed ranges.
- Each bet size: one run in which every decider holds its *calling* range
  and all-in opponents keep their full range. Sizes with the same calling
  ranges share a run.
- Preflop too (no table: the ranges are arbitrary). The default 600
  deals gives an SE of at most 2 points; a decision costs 2-4 runs of about
  2-4 ms each 6-handed (see ``tests/test_multiway_bots.py`` for the timing check).

Known weaknesses (beyond EquityBot's)
-------------------------------------
- Players behind are assumed to call the current bet, and nobody ever
  raises in the model (no squeezes).
- Opponents are independent: three callers in a row are treated as three
  separate top-MDF ranges, not as a correlated "this pot is strong" signal.
- Calling preflop does not narrow a range (as HU); multiway a flat call
  from early position is usually much stronger than a random hand.
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
    deciders,
    effective_max_to,
    ev_aggressive,
    ev_call,
    ev_check,
    label,
    live_opponents,
    raise_to_for_fraction,
)
from nutsfinder.bots.multiway.equity import MultiwayDeals, sample_deals, top_weights
from nutsfinder.bots.multiway.pot_odds import deals_share
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise  # noqa: F401
from nutsfinder.engine.cards import Card
from nutsfinder.engine.multiway.bot import MultiBot, MultiHandSummary
from nutsfinder.engine.multiway.state import MultiPublicState
from nutsfinder.engine.preflop import snap_to_grid

__all__ = ["MultiEquityBot", "fold_probability", "calling_share", "aggression_count"]

log = logging.getLogger(__name__)


def calling_share(pot: int, added: int) -> float:
    """MDF = ``P / (P + x)``."""
    return pot / (pot + added) if pot + added > 0 else 1.0


def fold_probability(pot: int, added: int, max_fold: float) -> float:
    """``f = min(max_fold, x / (P + x))``."""
    return min(max_fold, 1.0 - calling_share(pot, added))


def aggression_count(view: MultiPublicState, seat: int) -> int:
    """Bets and raises by ``seat`` so far this hand."""
    return sum(1 for r in view.history if r.player == seat and isinstance(r.action, (Bet, Raise)))


def _no_better_than(called, full):  # type: ignore[no-untyped-def]
    """Share against callers, capped at the share against the full ranges (see "Callers are never weaker")."""

    def share(opps: tuple[int, ...]) -> float:
        return min(called(opps), full(opps))

    return share


class MultiEquityBot(MultiBot):
    """Per-opponent ranges, per-opponent fold model, side-pot-aware max-EV action.

    Args:
        seed: seeds the numpy generator (all Monte Carlo).
        range_widths: assumed range after 0, 1, 2, 3+ bets/raises of that opponent.
        bet_fractions: bet/raise sizes to consider (fractions of the pot after calling).
        consider_all_in: also consider the (effective) all-in.
        all_in_max_spr: consider the all-in only if it is at most this many pots
            (after calling) beyond a call; ``None`` = always (see "The all-in gate").
        max_fold: cap on each opponent's fold probability.
        samples: Monte Carlo deals per run.
        time_budget: optional seconds per decision, split over the runs (machine-dependent).
        debug: record each decision in :attr:`last_decision` (a :class:`MultiDecision`).
    """

    mode = "equity"

    def __init__(self, seed: int | None = None, range_widths: tuple[float, ...] = (1.0, 0.5, 0.3, 0.15),
                 bet_fractions: tuple[float, ...] = (0.5, 1.0), consider_all_in: bool = True,
                 all_in_max_spr: float | None = 3.0, max_fold: float = 0.1, samples: int = 600, time_budget: float | None = None,
                 debug: bool = False) -> None:
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self.range_widths = range_widths
        self.bet_fractions = bet_fractions
        self.consider_all_in = consider_all_in
        self.all_in_max_spr = all_in_max_spr
        self.max_fold = max_fold
        self.samples = samples
        self.time_budget = time_budget
        self.debug = debug
        self.last_decision: MultiDecision | None = None
        self.stats: Counter[str] = Counter()
        """Running totals (not reset by :meth:`reseed`): decisions, Monte Carlo runs (``deals``),
        sampled ``rows``, ``fallback_rows`` dealt sequentially, ``widened`` ranges, ``all_in_gated``."""

    def reseed(self, seed: int) -> None:
        self._rng = np.random.Generator(np.random.PCG64(seed))

    def end_hand(self, summary: MultiHandSummary) -> None:
        """Nothing is learned (MultiBayesBot overrides this)."""

    # --- model hooks (MultiBayesBot overrides these) -----------------------------------

    def opponent_width(self, view: MultiPublicState, seat: int) -> float:
        """Assumed range width of ``seat`` from its bets/raises this hand (grid width)."""
        n = aggression_count(view, seat)
        return snap_to_grid(self.range_widths[min(n, len(self.range_widths) - 1)])

    def fold_and_calling(self, view: MultiPublicState, seat: int, width: float, to: int,
                         added: int) -> tuple[float, float]:
        """``(f_j, calling range width)`` of ``seat`` against our bet/raise to ``to`` adding ``added``."""
        f = fold_probability(view.pot, added, self.max_fold)
        return f, snap_to_grid(width * calling_share(view.pot, added))

    def model_notes(self, view: MultiPublicState) -> list[str]:
        return []

    def opponent_note(self, view: MultiPublicState, seat: int, width: float) -> str:
        return f"seat {seat} ({view.positions[seat]}): top {width:.1%} after {aggression_count(view, seat)} bet(s)"

    # --- acting ---------------------------------------------------------------------

    def candidate_sizes(self, view: MultiPublicState) -> list[int]:
        legal = view.legal
        if not (legal.can_bet or legal.can_raise):
            return []
        sizes = {raise_to_for_fraction(view, f) for f in self.bet_fractions}
        if self.consider_all_in:
            jam = effective_max_to(view)
            if self.all_in_max_spr is None or (jam - view.current_bet) <= self.all_in_max_spr * (
                    view.pot + legal.call_amount):
                sizes.add(jam)
            else:
                self.stats["all_in_gated"] += 1
        return sorted(sizes)

    def _deals(self, hole_cards, view, opps, widths, cache, deadline) -> MultiwayDeals:  # type: ignore[no-untyped-def]
        key = tuple(widths[s] for s in opps)
        if key not in cache:
            d = sample_deals(hole_cards, view.board, [top_weights(w) for w in key], self.samples,
                             self._rng, deadline)
            cache[key] = d
            self.stats.update(deals=1, rows=d.n, fallback_rows=d.fallback_rows, widened=len(d.widened))
        return cache[key]

    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:
        start = time.perf_counter()
        legal = check_turn(view)
        self.stats["decisions"] += 1
        me = view.seat
        opps = live_opponents(view)
        cols = {s: i for i, s in enumerate(opps)}
        widths = {s: self.opponent_width(view, s) for s in opps}
        sizes = self.candidate_sizes(view)
        runs = 1 + len(sizes)
        deadline = None
        if self.time_budget is not None:
            deadline = start + self.time_budget / runs  # first run's share; later runs re-derive below
        cache: dict[tuple, MultiwayDeals] = {}
        base = self._deals(hole_cards, view, opps, widths, cache, deadline)
        share = deals_share(base, cols)
        eq = base.equity()

        options: list[tuple[Action, float]] = []
        notes: list[str] = []
        winnable = view.pot
        odds = None
        if legal.can_call:
            ev_c, winnable = ev_call(view, share)
            odds = legal.call_amount / winnable if winnable else 1.0
            options += [(Fold(), 0.0), (Call(), ev_c)]
        else:
            options.append((Check(), ev_check(view, share)))
        ds = deciders(view)
        for i, to in enumerate(sizes, start=2):
            if self.time_budget is not None:
                deadline = start + self.time_budget * i / runs
            x = to - view.street_bets[me]
            folds: dict[int, float] = {}
            call_widths = dict(widths)
            for j in ds:
                folds[j], call_widths[j] = self.fold_and_calling(view, j, widths[j], to, x)
            called = self._deals(hole_cards, view, opps, call_widths, cache, deadline)
            ev, p_all = ev_aggressive(view, to, folds, _no_better_than(deals_share(called, cols), share))
            action = aggressive_action(legal, to)
            options.append((action, ev))
            if self.debug:
                notes.append(
                    f"{label(action)}: P(all {len(ds)} fold) = {p_all:.2%}; "
                    + ", ".join(f"s{j} f={folds[j]:.0%} calls top {call_widths[j]:.1%}" for j in ds)
                    + f"; equity vs callers {called.equity().equity:.1%}"
                )
        action, _ = max(options, key=lambda o: o[1])  # first of equal EVs: the most passive
        if self.debug:
            for widths_key, deals in cache.items():
                if deals.widened:
                    notes.append(f"ranges {[f'{w:.1%}' for w in widths_key]}: opponent(s) "
                                 f"{[opps[i] for i in deals.widened]} have no combo left after card removal: "
                                 "random hand assumed")
                if deals.fallback_rows:
                    notes.append(
                        f"ranges {[f'{w:.1%}' for w in widths_key]} could not be dealt jointly in "
                        f"{deals.fallback_rows}/{deals.n} deals: dealt sequentially, widening an exhausted "
                        "range to any possible hand")
            if len(view.pots) > 1:
                notes.append("pots now: " + ", ".join(f"{p.amount} {list(p.eligible)}" for p in view.pots))
            notes += self.model_notes(view)
            self.last_decision = MultiDecision(
                street=view.street.name.lower(), hole_cards=cards_text(hole_cards), board=cards_text(view.board),
                mode=self.mode, n_opponents=len(opps),
                opponents=tuple(self.opponent_note(view, s, widths[s]) for s in opps),
                equity=eq.equity, std_error=eq.std_error, samples=eq.samples, pot=view.pot,
                winnable_pot=winnable, to_call=legal.call_amount, pot_odds=odds,
                evs=tuple((label(a), ev) for a, ev in options), action=action,
                seconds=time.perf_counter() - start, notes=tuple(notes),
            )
            log.debug("%s", self.last_decision)
        return action

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(range_widths={self.range_widths}, bet_fractions={self.bet_fractions}, "
                f"max_fold={self.max_fold}, samples={self.samples})")

