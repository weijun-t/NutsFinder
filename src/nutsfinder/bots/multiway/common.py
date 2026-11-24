"""Shared pieces of the multiway EV bots: pot layering, the EV formulas, sizing, the decision record.

Notation (chips), as in :mod:`nutsfinder.bots.common`: ``P`` = ``view.pot``
(every chip in the middle, this street's bets included), ``c`` = the cost
of calling (``legal.call_amount``, capped by our stack), and EV measured
from now on: chips already in are sunk, so folding is worth exactly 0.

The one-street showdown model
-----------------------------
Heads-up, the EV bots value an action by "the hand is then checked down".
With several opponents we also need to say what the others do with their
chips. The rule used everywhere here:

    Every live opponent who still has chips **matches the level** (the
    larger of the current bet and our new street total), as far as its
    stack allows, unless the fold model says it folds (bets only).

Heads-up this is exactly the HU model (the opponent has already matched
when we face a bet, and "called" means it matches our bet). Multiway it
also covers opponents *behind us* who still owe the current bet: they are
assumed to call it. That is optimistic about the pot and pessimistic about
our share (everybody stays in); the two errors pull in opposite directions,
and it is the natural model for "opponents with random hands who call".
Known gap (see docs/phase7_multiway_research.md, section 2): nobody ever
*raises* in the model, so a bet that faces several players behind is
valued as if no-one could squeeze.

Side pots
---------
Given the final contributions ``totals`` (one per seat, folded seats'
chips included as dead money) and the set of live seats, the unmatched top
contribution is returned to its owner and the rest is layered with
:func:`nutsfinder.engine.multiway.pots.side_pots`. Our expected winnings are::

    W = refund to us + Σ_pots [we are eligible] amount × E[our share among that pot's eligible seats]

where "share" counts splits (1/t for a t-way tie for best). The equity
estimates come with the full per-deal strength matrix
(:class:`nutsfinder.bots.multiway.equity.MultiwayDeals`), so each pot gets
our share against exactly its eligible opponents, from the same deals.
Then::

    EV(check)  = W(totals now)
    EV(call)   = W(totals after calling c) - c
    EV(bet x)  = Σ_{F ⊆ deciders} P(exactly F fold) × (W_F - x)

with the deciders the live opponents who still have chips, each folding
independently with probability ``f_j``, so ``P(F) = Π_{j∈F} f_j
Π_{j∉F} (1 - f_j)``, and ``W_F`` the winnings when the players in ``F``
fold (their chips stay in as dead money) and the rest match the bet. If
everybody folds, ``W_F - x = P``: we take the pot. With one opponent this
is ``f P + (1 - f)(e_called (P + x + r) - x)``, the HU formula. In
particular **a bet wins the pot outright only if all deciders fold**,
probability ``Π_j f_j`` (independent opponents; optimistic, because a
squeeze can come from behind).

"Pot odds" in the logs is ``c / (winnable pot after the call)``, where the
winnable pot is the sum of the pots we are eligible for. With a single pot
``EV(call) > 0`` exactly when ``equity > pot odds``; with side pots the
bots decide by the EV, which weighs each pot by our share against its own
contestants.

Sizing: :func:`raise_to_for_fraction` is the HU formula
``current_bet + fraction × (P + c)``, clamped to ``[min_to, eff_to]`` where
``eff_to`` is the most any live opponent can match (chips above it would be
refunded). With equal stacks ``eff_to = max_to``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from nutsfinder.bots.common import aggressive_action, cards_text, label
from nutsfinder.engine.actions import Action
from nutsfinder.engine.betting import LegalActions
from nutsfinder.engine.multiway.pots import side_pots, uncalled_bet
from nutsfinder.engine.multiway.state import MultiPublicState

__all__ = [
    "MultiDecision",
    "ShareFn",
    "check_turn",
    "live_opponents",
    "deciders",
    "effective_max_to",
    "raise_to_for_fraction",
    "showdown_totals",
    "expected_winnings",
    "winnable_pot",
    "ev_check",
    "ev_call",
    "ev_aggressive",
    "aggressive_action",
    "label",
    "cards_text",
    "MIN_BRANCH_PROB",
]

ShareFn = Callable[[tuple[int, ...]], float]
"""``share(opponent seats) -> E[our pot share against exactly those seats]`` (() -> 1.0)."""

MIN_BRANCH_PROB = 1e-7
"""Fold branches less likely than this are skipped (their total weight is at most 128 × this)."""


def check_turn(view: MultiPublicState) -> LegalActions:
    """The legal actions, after checking that it really is this seat's turn."""
    legal = view.legal
    if legal is None or legal.player != view.seat or view.to_act != view.seat:
        raise ValueError(f"seat {view.seat} was asked to act out of turn")
    return legal


def live_opponents(view: MultiPublicState) -> list[int]:
    """Seats still in the hand other than ours, in seat order."""
    return [s for s in range(view.n_players) if s != view.seat and view.in_hand[s]]


def deciders(view: MultiPublicState) -> list[int]:
    """Live opponents with chips behind: the ones who must answer a bet of ours."""
    return [s for s in live_opponents(view) if view.stacks[s] > 0]


def effective_max_to(view: MultiPublicState) -> int:
    """The largest bet/raise-to any live opponent can match (at least ``min_to``)."""
    legal = view.legal
    assert legal is not None and (legal.can_bet or legal.can_raise)
    cover = max((view.street_bets[s] + view.stacks[s] for s in deciders(view)), default=legal.min_to)
    return max(min(legal.max_to, cover), legal.min_to)


def raise_to_for_fraction(view: MultiPublicState, fraction: float) -> int:
    """``current_bet + fraction × (P + c)``, rounded, clamped to ``[min_to, effective_max_to]``."""
    legal = view.legal
    assert legal is not None and (legal.can_bet or legal.can_raise)
    to = round(view.current_bet + fraction * (view.pot + legal.call_amount))
    return min(max(to, legal.min_to), effective_max_to(view))


def showdown_totals(view: MultiPublicState, my_to: int, callers: Sequence[int] | None = None) -> list[int]:
    """Final contributions if we make our street total ``my_to`` and ``callers`` match the level.

    ``callers`` defaults to every live opponent with chips (see the module doc). Opponents not in
    ``callers`` keep what they have put in. The level is ``max(current_bet, my_to)``.
    """
    me = view.seat
    totals = list(view.total_bets)
    totals[me] += my_to - view.street_bets[me]
    level = max(view.current_bet, my_to)
    who = deciders(view) if callers is None else callers
    for j in who:
        owe = level - view.street_bets[j]
        if owe > 0:
            totals[j] += min(owe, view.stacks[j])
    return totals


def _layers(me: int, totals: Sequence[int], live: Sequence[bool]) -> tuple[int, list[tuple[int, tuple[int, ...]]]]:
    """``(refund to me, [(amount, eligible opponents) for each pot I am eligible for])``."""
    totals = list(totals)
    seat, amount = uncalled_bet(totals)
    refund = 0
    if seat is not None:
        totals[seat] -= amount
        refund = amount if seat == me else 0
    out = []
    for pot in side_pots(totals, live):
        if me in pot.eligible:
            out.append((pot.amount, tuple(s for s in pot.eligible if s != me)))
    return refund, out


def expected_winnings(share: ShareFn, me: int, totals: Sequence[int], live: Sequence[bool]) -> float:
    """``W``: refund to us plus each pot we are eligible for times our share in it."""
    refund, pots = _layers(me, totals, live)
    return refund + sum(amount * share(opps) for amount, opps in pots)


def winnable_pot(me: int, totals: Sequence[int], live: Sequence[bool]) -> int:
    """Chips in the pots we are eligible for (after any refund)."""
    return sum(amount for amount, _ in _layers(me, totals, live)[1])


def _live_mask(view: MultiPublicState, folded: Sequence[int] = ()) -> list[bool]:
    live = list(view.in_hand)
    for j in folded:
        live[j] = False
    return live


def ev_check(view: MultiPublicState, share: ShareFn) -> float:
    """``W`` with the contributions as they are (plus opponents matching the current bet)."""
    me = view.seat
    totals = showdown_totals(view, view.street_bets[me])
    return expected_winnings(share, me, totals, _live_mask(view))


def ev_call(view: MultiPublicState, share: ShareFn) -> tuple[float, int]:
    """``(EV(call), winnable pot after the call)``: ``W(after calling c) - c``."""
    legal = view.legal
    assert legal is not None
    me, c = view.seat, legal.call_amount
    totals = showdown_totals(view, view.street_bets[me] + c)
    live = _live_mask(view)
    return expected_winnings(share, me, totals, live) - c, winnable_pot(me, totals, live)


def ev_aggressive(view: MultiPublicState, to: int, fold_probs: dict[int, float],
                  share_called: ShareFn) -> tuple[float, float]:
    """``(EV, P(all deciders fold))`` of a bet/raise to ``to`` (module doc).

    ``fold_probs[j]`` is ``f_j`` for each decider (missing deciders never fold).
    ``share_called`` is evaluated on deals in which the deciders hold their
    *calling* ranges.
    """
    me = view.seat
    x = to - view.street_bets[me]
    ds = deciders(view)
    probs = [min(1.0, max(0.0, fold_probs.get(j, 0.0))) for j in ds]
    ev = 0.0
    p_all = 1.0
    for f in probs:
        p_all *= f
    for folds in itertools.product((False, True), repeat=len(ds)):
        p = 1.0
        for f, fold in zip(probs, folds):
            p *= f if fold else 1.0 - f
        if p < MIN_BRANCH_PROB:
            continue
        folded = [j for j, fold in zip(ds, folds) if fold]
        callers = [j for j, fold in zip(ds, folds) if not fold]
        totals = showdown_totals(view, to, callers)
        live = _live_mask(view, folded)
        if sum(live) == 1:  # only us: we take every chip, our own refund included
            w = float(sum(totals))
        else:
            w = expected_winnings(share_called, me, totals, live)
        ev += p * (w - x)
    return ev, p_all


@dataclass(frozen=True, slots=True)
class MultiDecision:
    """Why a multiway EV bot did what it did (recorded when ``debug=True``).

    Attributes:
        street, hole_cards, board: as text (``"AsKs"``, ``"Kh7h2s"``).
        mode: which logic decided (``"pot-odds"``, ``"equity"``, ``"bayes"``,
            ``"equity-fallback"`` for MultiCFRBot off the blueprint, ...).
        n_opponents: live opponents.
        opponents: one line per live opponent (seat, position, assumed range, fold model).
        equity: our pot share against all live opponents (ties split).
        std_error: its standard error (0 for a table lookup with negligible SE).
        samples: Monte Carlo deals behind ``equity`` (0 = table).
        pot: ``P`` (whole pot, including the bet faced).
        winnable_pot: the pots we would be eligible for after calling (or now, if not facing a bet).
        to_call: ``c``.
        pot_odds: ``c / winnable_pot``, ``None`` when not facing a bet.
        evs: ``(action label, EV in chips)`` for each action considered.
        action: the action chosen.
        seconds: wall time of the decision.
        notes: anything else (fold-branch numbers, side pots, model lines).
    """

    street: str
    hole_cards: str
    board: str
    mode: str
    n_opponents: int
    opponents: tuple[str, ...]
    equity: float
    std_error: float
    samples: int
    pot: int
    winnable_pot: int
    to_call: int
    pot_odds: float | None
    evs: tuple[tuple[str, float], ...]
    action: Action
    seconds: float
    notes: tuple[str, ...] = ()

    def ev(self, action_label: str) -> float:
        return dict(self.evs)[action_label]

    def __str__(self) -> str:
        odds = "-" if self.pot_odds is None else f"{self.pot_odds:.1%}"
        lines = [
            f"{self.street} {self.hole_cards} board [{self.board}] pot {self.pot} to call {self.to_call} "
            f"[{self.mode}]",
            f"  equity {self.equity:.1%} +/- {self.std_error:.1%} vs {self.n_opponents} opponent(s)"
            f" ({self.samples or 'table'} deals), pot odds {odds} (winnable pot {self.winnable_pot})",
            *(f"  opp {o}" for o in self.opponents),
            *(f"  EV({name}) = {ev:+.1f}" for name, ev in self.evs),
            *(f"  {note}" for note in self.notes),
            f"  -> {self.action} ({self.seconds * 1000:.1f} ms)",
        ]
        return "\n".join(lines)
