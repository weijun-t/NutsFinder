"""Shared pieces for the EV bots: bet sizing, the one-street EV formulas, and decision logs.

Notation used throughout (all in chips)
---------------------------------------
- ``P`` = ``state.pot``: every chip in the middle, *including* this street's
  bets, so including a bet we are facing.
- ``c`` = ``legal.call_amount``: what calling costs us.
- ``e`` = our equity: the share of the final pot we expect to win if the
  hand goes to showdown with no more betting.

EV is measured from now on: chips already in the pot are gone ("sunk"), so
folding is worth exactly 0 and every other action is compared with that.

Pot odds
--------
Calling ``c`` gives us a share ``e`` of a pot of ``P + c``::

    EV(call) = e * (P + c) - c

which is positive exactly when ``e > c / (P + c)``. That ratio is the *pot
odds*: the equity a call needs to break even. Worked example: the pot was
1,000 and the opponent bets 500. Then ``P = 1,500`` (it includes the bet)
and ``c = 500``, so the pot odds are 500 / 2,000 = 25%. With 30% equity,
EV(call) = 0.30 * 2,000 - 500 = +100; with 20%, it is 400 - 500 = -100.

Checking (one-street model: assume the hand is then checked down)::

    EV(check) = e * P

Betting or raising: we add ``x`` chips (raise-to minus what we already put
in this street) and the opponent needs ``r`` more to call. If they call,
the pot becomes ``P + x + r`` and we win our share of it with equity
``e_called`` against the hands that call. If they fold (probability ``f``)
we win ``P``::

    EV(bet) = f * P + (1 - f) * (e_called * (P + x + r) - x)

If the opponent always calls (``f = 0``) and ``e_called = e``, this is the
"EV if called" that PotOddsBot logs.

The effective stack (unequal stacks)
------------------------------------
Chips can only change hands up to the *effective stack*: whatever one
player puts in beyond what the other can match is handed back (the
engine's ``refunds``). The engine's ``legal.max_to`` is our own all-in and
is *not* capped by the opponent's stack (betting more than they have is
legal, just pointless), so the bots cap themselves. With equal starting
stacks none of this changes a single number: both players always have the
same "street total + stack", so the caps below never bind and the excess
is always 0.

Betting / raising. When we act, the opponent's street total is
``state.current_bet`` (they made the bet, or the bets are level), so the
most they can ever have in on this street is::

    eff_to = min(legal.max_to, current_bet + stacks[opponent])

(:func:`effective_max_to`). Every size we consider is capped there, all-in
included, and the EV uses ``x = min(to, eff_to) - our street total``, the
chips actually at risk; ``r`` (what they need to call) is already capped
by their stack. If even a minimum raise is more than the opponent can
match (``eff_to < legal.min_to``), the only size left is ``min_to``: it is
legal, it puts the opponent all-in, and the part above ``eff_to`` is
refunded, so its EV is computed as a raise to ``eff_to``. (Calling instead
is not quite the same: a raise makes them decide now, for all their chips.)

Worked example B: stacks 10,000 vs 1,000, the deep player limps, the big
blind checks, pot ``P = 200``; on the flop the short stack checks. The deep
player's all-in is 9,900 chips, but only ``eff_to = 0 + 900 = 900`` can be
called. With equity 0.8 (also when called) and ``f = 0.1``::

    EV(all-in) = 0.1 * 200 + 0.9 * (0.8 * (200 + 900 + 900) - 900) = +650

whereas using ``x = 9,900`` and ``r = 900`` gave
``0.1 * 200 + 0.9 * (0.8 * 11,000 - 9,900) = -970``.

Calling all-in for less. If the bet we face is more than our whole stack,
calling costs only ``c`` = our stack, and the opponent's chips beyond our
street total ``+ c`` come back to them. That excess was never really in
play, so we take it out of the pot (:func:`effective_pot`)::

    excess   = max(0, current_bet - (our street total + c))
    P_eff    = P - excess
    pot odds = c / (P_eff + c)
    EV(call) = e * (P_eff + c) - c

Worked example A: stacks 10,000 vs 600, limp and check (pot 200), the short
stack checks the flop and faces a bet of 3,000 with 500 behind. Then
``P = 3,200``, ``c = 500``, excess = 3,000 - 500 = 2,500 and ``P_eff = 700``:
pot odds 500 / 1,200 = 41.7% (the full pot would say 500 / 3,700 = 13.5%).
With 25% equity, EV(call) = 0.25 * 1,200 - 500 = -200: fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import LegalActions, PublicState
from nutsfinder.engine.cards import Card

__all__ = [
    "Decision",
    "check_turn",
    "effective_max_to",
    "effective_pot",
    "raise_to_for_fraction",
    "aggressive_action",
    "pot_odds",
    "ev_call",
    "ev_check",
    "ev_bet",
    "chips_added",
    "chips_to_call_raise",
    "label",
    "cards_text",
]


def check_turn(state: PublicState) -> LegalActions:
    """The legal actions, after checking that it really is this seat's turn."""
    legal = state.legal
    if legal is None or legal.player != state.seat:
        raise ValueError(f"seat {state.seat} was asked to act out of turn")
    return legal


def effective_max_to(state: PublicState) -> int:
    """The largest bet/raise-to worth making: what the opponent can still match.

    ``eff_to = min(legal.max_to, current_bet + stacks[opponent])``: the
    opponent's street total is ``current_bet`` and they cannot add more than
    their stack, so chips above ``eff_to`` would just be refunded. If that is
    below ``legal.min_to`` (they cannot cover a minimum raise), ``min_to`` is
    returned instead: the smallest legal raise, which puts them all-in (see
    the module docstring). Example B: stacks 10,000 vs 1,000 after a limp
    and a check, ``max_to = 9,900`` but this returns 900.
    """
    legal = state.legal
    assert legal is not None and (legal.can_bet or legal.can_raise)
    eff_to = min(legal.max_to, state.current_bet + state.stacks[1 - state.seat])
    return max(eff_to, legal.min_to)


def raise_to_for_fraction(state: PublicState, fraction: float) -> int:
    """The bet amount / raise-to total for a bet of ``fraction`` of the pot, clamped.

    Unclamped: ``current_bet + fraction * (P + c)``. "First call, then raise
    by ``fraction`` of the pot that is then in the middle". With
    ``fraction = 1`` this is the pot-sized raise of
    :func:`nutsfinder.bots.always_raise.pot_sized_to`; when not facing a bet
    (``c = 0``, current bet 0) it is simply ``fraction * P``. Example: pot
    1,000 on the flop, 2/3 pot bet = 667. The result is rounded to whole
    chips and clamped to ``[min_to, effective_max_to]``: never more than
    the opponent can call (:func:`effective_max_to`, which equals
    ``max_to`` when the stacks are equal).
    """
    legal = state.legal
    assert legal is not None and (legal.can_bet or legal.can_raise)
    to = round(state.current_bet + fraction * (state.pot + legal.call_amount))
    return min(max(to, legal.min_to), effective_max_to(state))


def aggressive_action(legal: LegalActions, to: int) -> Action:
    """``Bet(to)`` if nobody has bet this street, else ``Raise(to)``."""
    return Bet(to) if legal.can_bet else Raise(to)


def chips_added(state: PublicState, to: int) -> int:
    """``x``: chips we put *at risk* by making our street total ``to``.

    That is ``min(to, eff_to) - our street total``, with
    ``eff_to = current_bet + stacks[opponent]`` the most the opponent can
    match: anything above it is refunded, so it can be neither won nor lost.
    (Normally ``to <= eff_to`` already, see :func:`effective_max_to`; the cap
    matters when a minimum raise is more than the opponent can cover.)
    """
    eff_to = state.current_bet + state.stacks[1 - state.seat]
    return min(to, eff_to) - state.street_bets[state.seat]


def chips_to_call_raise(state: PublicState, to: int) -> int:
    """``r``: chips the opponent needs to call our bet/raise to ``to``.

    The opponent's street total is the current bet (heads-up, either they
    made it or the bets are level). They cannot put in more than they have.
    """
    return min(to - state.current_bet, state.stacks[1 - state.seat])


def effective_pot(state: PublicState) -> int:
    """``P_eff``: the pot minus the part of the opponent's bet we cannot match.

    ``excess = max(0, current_bet - (our street total + c))``, with ``c`` the
    (possibly all-in) call amount; ``P_eff = P - excess``. The excess goes
    back to the opponent whatever we do, so pot odds and EV(call) must use
    ``P_eff``. It is 0 unless our call is all-in for less than the bet (so
    always 0 with equal stacks). Example A: ``P = 3,200``, we have 500
    behind facing a 3,000 bet: excess = 3,000 - 500 = 2,500, ``P_eff = 700``.
    """
    legal = state.legal
    assert legal is not None
    excess = state.current_bet - (state.street_bets[state.seat] + legal.call_amount)
    return state.pot - max(0, excess)


def pot_odds(call: int, pot: int) -> float:
    """Break-even equity for a call: ``c / (P + c)``, ``P`` including the bet faced.

    Pass the effective pot (:func:`effective_pot`) so that a call that is
    all-in for less than the bet is priced correctly.
    """
    return call / (pot + call)


def ev_call(equity: float, pot: int, call: int) -> float:
    """``e * (P + c) - c``, with ``P`` the effective pot (:func:`effective_pot`)."""
    return equity * (pot + call) - call


def ev_check(equity: float, pot: int) -> float:
    """``e * P`` (the pot is checked down)."""
    return equity * pot


def ev_bet(pot: int, added: int, villain_call: int, fold_prob: float, equity_called: float) -> float:
    """``f * P + (1 - f) * (e_called * (P + x + r) - x)``."""
    called = equity_called * (pot + added + villain_call) - added
    return fold_prob * pot + (1 - fold_prob) * called


def label(action: Action) -> str:
    """A short name for an action, used as the key of an EV table."""
    match action:
        case Fold():
            return "fold"
        case Check():
            return "check"
        case Call():
            return "call"
        case Bet(amount=amount):
            return f"bet {amount}"
        case Raise(to=to):
            return f"raise to {to}"
    raise AssertionError(f"unknown action {action!r}")  # pragma: no cover


@dataclass(frozen=True, slots=True)
class Decision:
    """Why a bot did what it did, for one decision (recorded when ``debug=True``).

    Attributes:
        street: ``"preflop"``, ``"flop"``, ``"turn"`` or ``"river"``.
        hole_cards, board: as text, e.g. ``"AsKs"``, ``"Kh7h2s"``.
        villain_range: the fraction of all hands the opponent is assumed to
            hold (1.0 = any hand).
        equity: our equity against that range.
        std_error: its standard error (0 if exact, e.g. on the river).
        pot: ``P``, including any bet we face (the whole pot; when part of
            the bet would be refunded, the notes give ``P_eff``).
        to_call: ``c`` (0 if we may check).
        pot_odds: ``c / (P_eff + c)``, or ``None`` when not facing a bet.
        evs: ``(action label, EV in chips)`` for every action considered.
        action: the action chosen.
        notes: anything else worth showing (e.g. the fold model's numbers).
    """

    street: str
    hole_cards: str
    board: str
    villain_range: float
    equity: float
    std_error: float
    pot: int
    to_call: int
    pot_odds: float | None
    evs: tuple[tuple[str, float], ...]
    action: Action
    notes: tuple[str, ...] = ()

    def ev(self, action_label: str) -> float:
        """The EV logged for one action label, e.g. ``d.ev("call")``."""
        return dict(self.evs)[action_label]

    def __str__(self) -> str:
        odds = "-" if self.pot_odds is None else f"{self.pot_odds:.1%}"
        lines = [
            f"{self.street} {self.hole_cards} board [{self.board}] "
            f"pot {self.pot} to call {self.to_call}",
            f"  equity {self.equity:.1%} +/- {self.std_error:.1%} vs top {self.villain_range:.0%}"
            f", pot odds {odds}",
            *(f"  EV({name}) = {ev:+.1f}" for name, ev in self.evs),
            *(f"  {note}" for note in self.notes),
            f"  -> {self.action}",
        ]
        return "\n".join(lines)


def cards_text(cards: Sequence[Card]) -> str:
    """``"AsKs"`` style text for a sequence of cards."""
    return "".join(str(c) for c in cards)
