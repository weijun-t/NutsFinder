"""Action translation: mapping real bet sizes onto the few sizes the abstraction knows.

The CFR blueprint is solved in an *abstract* game where a player may only
fold, check/call, or bet/raise a handful of sizes (e.g. preflop: open to 3bb,
pot, all-in; postflop: pot, all-in). A real opponent can bet any whole number
of chips. Two problems follow, and this module solves both:

1. **Their bets (translation).** The opponent bets an amount the abstraction
   does not have. Which abstract action do we pretend they took, so that we
   can look up our strategy? -> :func:`translate_bet`.
2. **Our bets (sizing).** Our strategy says "pot". How many *real* chips is
   that? -> :func:`to_for_fraction` / :func:`real_action_for_fraction`.

Sizes are pot fractions
-----------------------
Every size is a fraction ``x`` of the pot *after calling*, exactly as in
:func:`nutsfinder.bots.common.raise_to_for_fraction`::

    to = current_bet + x * (P + c)          x = (to - current_bet) / (P + c)

with ``P`` the pot (including any bet faced), ``c`` the bettor's call amount
and ``current_bet`` the street total to match, all measured *in the state
the bettor faced before betting*. Fractions are scale-free: "pot" means the
same thing in a 200-chip pot and a 20,000-chip pot, which is what lets us
compare a real bet with an abstract size when the real pot and the abstract
pot differ.

All-in is the fraction of the effective all-in
(:func:`nutsfinder.bots.common.effective_max_to`). Pass :data:`ALL_IN`
(``math.inf``) for it: every size is clamped to ``[min_to, eff_max_to]`` at
the real spot, so ``inf`` becomes exactly the effective all-in.

Preflop sizes in big blinds. "Open to 3bb" is converted to a fraction at the
real spot with :func:`fraction_for_bb`. Worked example, blinds 50/100, the
button (small blind) to act: ``P = 150``, ``c = 50``, ``current_bet = 100``,
so open to 300 is ``x = (300 - 100) / 200 = 1.0``; and for the big blind
after a limp (``P = 200``, ``c = 0``) a raise to 300 is again
``200 / 200 = 1.0``. **With the standard blinds (SB = BB/2), "open to 3bb"
and "pot" are the same raise at an unopened node.** They are merged
(deduplicated, see below) rather than treated as two sizes.

The pseudo-harmonic mapping (Ganzfried & Sandholm 2013, IJCAI)
---------------------------------------------------------------
The abstraction has neighbouring sizes ``A < B`` and the opponent bets ``x``
with ``A <= x <= B``. Treat the bet as ``A`` with probability::

    f(x) = (B - x)(1 + A) / ((B - A)(1 + x)),      otherwise as B.

Where it comes from. In the simplest bluffing game (the bettor either has
the nuts or nothing; pot 1, bet ``x``), the defender calls a bet of ``x``
with probability ``1 / (1 + x)``: that makes a bluff break even (win the
pot 1 when they fold, lose ``x`` when they call: ``(1 - q) - q x = 0``).
If we answer ``x`` with our "A" strategy with probability ``f`` and our "B"
strategy otherwise, we call with probability ``f/(1+A) + (1-f)/(1+B)``.
Choosing ``f`` so that this equals the right answer ``1/(1+x)`` gives::

    f = (1/(1+x) - 1/(1+B)) / (1/(1+A) - 1/(1+B))

which simplifies to the formula above. So ``f`` is *linear in 1/(1+x)*
(the "price" of the bet), ``f(A) = 1``, ``f(B) = 0``, and it decreases
monotonically in between. Worked example: ``A = 1/2``, ``B = 1``,
``x = 3/4`` gives ``f = (0.25 * 1.5) / (0.5 * 1.75) = 0.4286``: a 3/4-pot bet
is read as half pot 43% of the time and as pot 57% of the time (it is
closer to pot in "price" terms than the midpoint suggests).

Why randomise rather than map to the nearest size. Any *deterministic* rule
has a threshold ``t`` between ``A`` and ``B`` where its answer jumps. An
opponent who knows it bets just above ``t`` (e.g. 0.76 pot when nearest-size
puts the cut at 0.75): we answer as if facing a pot bet and fold as often
as against pot, though their bluff risked only 0.76. Just below ``t`` they
value-bet and get called as often as against half pot. The randomised
mapping has no jump, so there is no single size that breaks it; Ganzfried &
Sandholm found it far less exploitable than the earlier mappings. (It is
still not exploit-proof: off-tree bets remain a weakness of any blueprint.)

Rules for the edge cases
------------------------
- **Below the smallest size** (``0 < x < a_0``): by default the passive
  action (check, or call) is treated as a "bet of size 0" and the
  pseudo-harmonic mapping is used between ``0`` and ``a_0``:
  ``P(passive) = (a_0 - x) / (a_0 (1 + x))``. This is G&S's own treatment.
  Mapping every tiny bet to the smallest size instead (``below="smallest"``)
  is exploitable: the opponent min-bets into a big pot and we fold as if
  facing a pot bet, while the bet offers great odds. If *passive* is chosen,
  the abstract history says the opponent checked / called; the real game
  may still make us respond to their small bet, and CFRBot should then
  **call** (that is what "they really put in nothing" means for us, and it
  is cheap because ``x < a_0``).
- **Between two sizes**: pseudo-harmonic.
- **Above the largest non-all-in size**: all-in is just the largest size,
  so this is pseudo-harmonic between that size and all-in.
- **At or above the (effective) all-in**: all-in. A bet bigger than we can
  call is capped at the effective all-in before computing ``x``. If the
  abstract sizes contain no all-in, anything above the largest size maps to
  the largest size.
- **Clamping and duplicates.** Each abstract size is first turned into real
  chips at the real spot, clamped to ``[min_to, eff_max_to]``. Sizes that
  clamp to the same amount (pot = all-in when stacks are short; open-3bb =
  pot at an unopened node; everything = all-in when only a short all-in is
  legal) become one point. The result reports all of them in
  ``equivalent``, and ``index`` is the largest of them (the all-in, if it is
  in the group, because the real bet is then an all-in).
- **Min-raises.** If the smallest abstract size clamps up to ``min_to`` and
  the opponent min-raises, then ``x == a_0`` and the translation is
  deterministic: the smallest size.
- **All-in for less** (the bettor cannot afford a full raise): ``min_to ==
  max_to``, every size clamps to that one amount, so the bet is read as
  all-in.
- **Zero pot.** ``P + c`` is at least the two blinds (``new_hand`` requires
  ``small_blind >= 1``), so ``x`` is always defined; a state with ``P + c ==
  0`` raises ``ValueError``.

Randomness: :func:`translate_bet` takes a ``numpy.random.Generator`` and
draws exactly one ``rng.random()`` when (and only when) it mixes between two
sizes, so a seeded generator gives a reproducible translation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from nutsfinder.bots.common import aggressive_action, effective_max_to, raise_to_for_fraction
from nutsfinder.abstraction.tree import SizeLabel
from nutsfinder.engine.actions import Action, Bet, Call, Check, Raise
from nutsfinder.engine.betting import PublicState, apply_action, deal_board, new_hand, public_view

__all__ = [
    "ALL_IN",
    "SizePoint",
    "Translation",
    "pseudo_harmonic",
    "bet_fraction",
    "fraction_for_bb",
    "to_for_fraction",
    "to_for_bb",
    "size_points",
    "translate_bet",
    "view_before_action",
    "translate_last_action",
    "real_action_for_fraction",
    "label_fraction",
]

ALL_IN = math.inf
"""Pot fraction meaning "all-in": clamps to the effective all-in at any spot."""

BelowRule = Literal["pseudo_harmonic", "smallest"]


def pseudo_harmonic(x: float, a: float, b: float) -> float:
    """Probability of reading a bet of ``x`` as the smaller size ``a`` (not ``b``).

    ``f(x) = (b - x)(1 + a) / ((b - a)(1 + x))`` for ``0 <= a <= x <= b``,
    ``a < b``, all finite pot fractions. ``f(a) = 1``, ``f(b) = 0``, and
    ``f`` decreases in between. Example: ``pseudo_harmonic(0.75, 0.5, 1.0)``
    is ``0.375 / 0.875 = 0.4286``. With ``a = 0`` (the passive action as a
    bet of size 0) it is ``(b - x) / (b (1 + x))``.
    """
    if not all(math.isfinite(v) for v in (x, a, b)):
        raise ValueError(f"pot fractions must be finite: x={x}, a={a}, b={b}")
    if not 0 <= a < b:
        raise ValueError(f"need 0 <= a < b, got a={a}, b={b}")
    if not a <= x <= b:
        raise ValueError(f"need a <= x <= b, got x={x} outside [{a}, {b}]")
    return (b - x) * (1 + a) / ((b - a) * (1 + x))


def _pot_after_call(state: PublicState) -> int:
    legal = state.legal
    if legal is None:
        raise ValueError("nobody is to act in this state")
    denom = state.pot + legal.call_amount
    if denom <= 0:
        raise ValueError("pot after calling is 0: pot fractions are undefined")
    return denom


def _can_aggress(state: PublicState) -> None:
    legal = state.legal
    if legal is None or not (legal.can_bet or legal.can_raise):
        raise ValueError("betting/raising is not legal in this state")


def bet_fraction(state: PublicState, to: int) -> float:
    """The pot fraction of a bet/raise to ``to``, made from ``state``.

    ``x = (min(to, eff_max_to) - current_bet) / (P + c)``, with ``state`` the
    view of the player to act *before* they bet. Chips above the effective
    all-in (more than the other player can call) are ignored: they would be
    refunded. Example: pot 1,000 on the flop, nobody has bet, bet 750:
    ``x = 750 / 1,000 = 0.75``.
    """
    _can_aggress(state)
    capped = min(to, effective_max_to(state))
    return (capped - state.current_bet) / _pot_after_call(state)


def fraction_for_bb(state: PublicState, n_bb: float) -> float:
    """A preflop size given as "to ``n_bb`` big blinds", as a pot fraction at ``state``.

    Example (blinds 50/100, button to act): 3bb = 300 is
    ``(300 - 100) / (150 + 50) = 1.0``, the same as a pot-sized raise.
    """
    return bet_fraction(state, round(n_bb * state.big_blind))


def to_for_fraction(state: PublicState, fraction: float) -> int:
    """The real legal to-amount for a bet of ``fraction`` of the *real* pot.

    This is :func:`~nutsfinder.bots.common.raise_to_for_fraction` (clamped
    to ``[min_to, effective_max_to]``, so always legal and never more than
    the opponent can call), plus :data:`ALL_IN` (or any fraction too big to
    round) giving the effective all-in.
    """
    _can_aggress(state)
    if not math.isfinite(fraction) or fraction * _pot_after_call(state) > 1e15:
        return effective_max_to(state)
    if fraction < 0:
        raise ValueError(f"pot fraction must be >= 0, got {fraction}")
    return raise_to_for_fraction(state, fraction)


def to_for_bb(state: PublicState, n_bb: float) -> int:
    """A "raise to ``n_bb`` big blinds" as a real legal to-amount, clamped like the others."""
    _can_aggress(state)
    to = round(n_bb * state.big_blind)
    assert state.legal is not None
    return min(max(to, state.legal.min_to), effective_max_to(state))


@dataclass(frozen=True, slots=True)
class SizePoint:
    """One distinct real bet size: abstract sizes that clamp to the same chips.

    Attributes:
        to: the real to-amount (clamped) at the spot.
        fraction: its real pot fraction, ``bet_fraction(state, to)``.
        indices: positions in the caller's ``sizes`` that land here,
            ordered by nominal size (smallest first).
    """

    to: int
    fraction: float
    indices: tuple[int, ...]


def size_points(state: PublicState, sizes: Sequence[float]) -> tuple[SizePoint, ...]:
    """Clamp each abstract size to real chips at ``state`` and merge duplicates.

    Returns the distinct points in increasing order. Example: flop, pot 200,
    stacks 150 behind each; sizes ``[1.0, ALL_IN]``: pot would be 200 but
    clamps to the all-in 150, so there is one point ``to=150`` with
    ``indices=(0, 1)``.
    """
    if not sizes:
        raise ValueError("need at least one abstract size")
    order = sorted(range(len(sizes)), key=lambda i: (sizes[i], i))
    points: list[SizePoint] = []
    for i in order:
        to = to_for_fraction(state, sizes[i])
        if points and points[-1].to == to:
            last = points[-1]
            points[-1] = SizePoint(last.to, last.fraction, last.indices + (i,))
        else:
            points.append(SizePoint(to, bet_fraction(state, to), (i,)))
    return tuple(points)


@dataclass(frozen=True, slots=True)
class Translation:
    """How one real bet was read in the abstraction (log it when ``debug=True``).

    Attributes:
        x: the real bet as a pot fraction.
        index: the chosen abstract size (position in the caller's
            ``sizes``), or ``None`` for the passive action (check/call).
        equivalent: every size index that clamps to the same real amount as
            the chosen one (empty for passive). Use any of them that exists
            at the abstract node.
        lower, upper: the two neighbouring points (pot fractions; ``0.0`` is
            the passive action) the bet fell between; equal if it was read
            deterministically.
        p_lower: probability the mapping gave to ``lower``.
    """

    x: float
    index: int | None
    equivalent: tuple[int, ...]
    lower: float
    upper: float
    p_lower: float

    @property
    def passive(self) -> bool:
        return self.index is None

    def __str__(self) -> str:
        chosen = "check/call" if self.index is None else f"size #{self.index}"
        if self.lower == self.upper:
            return f"bet {self.x:.3f} pot -> {chosen} (deterministic)"
        return (
            f"bet {self.x:.3f} pot between {self.lower:.3f} and {self.upper:.3f}: "
            f"P(lower) = {self.p_lower:.3f} -> {chosen}"
        )


def _point_result(x: float, point: SizePoint) -> Translation:
    return Translation(x, point.indices[-1], point.indices, point.fraction, point.fraction, 1.0)


def translate_bet(
    state: PublicState,
    to: int,
    sizes: Sequence[float],
    rng: np.random.Generator,
    *,
    below: BelowRule = "pseudo_harmonic",
) -> Translation:
    """Read a real bet/raise to ``to`` as one of the abstract ``sizes``.

    Args:
        state: the bettor's view *before* they bet (``state.legal`` must
            allow a bet or raise, and ``to`` must be legal there); see
            :func:`view_before_action` to rebuild it from history.
        to: the real bet amount / raise-to total.
        sizes: the abstract sizes available to the bettor at the matching
            abstract node, as pot fractions (:data:`ALL_IN` for all-in;
            preflop bb sizes via :func:`fraction_for_bb`).
        rng: the caller's seeded generator (one draw when mixing).
        below: what to do below the smallest size (see the module notes).

    Worked example: flop, pot 1,000, sizes ``[1.0, ALL_IN]`` with 9,000
    behind; a bet of 500 (``x = 0.5``) is below pot, so it is read as check
    with probability ``(1 - 0.5) / (1 * 1.5) = 1/3`` and as pot otherwise.
    """
    _can_aggress(state)
    assert state.legal is not None
    action: Action = Bet(to) if state.legal.can_bet else Raise(to)
    if not state.legal.is_legal(action):
        raise ValueError(f"{action!r} is not legal in this state: {state.legal}")
    x = bet_fraction(state, to)
    points = size_points(state, sizes)
    capped = min(to, effective_max_to(state))

    first, last = points[0], points[-1]
    if capped >= last.to:
        return _point_result(x, last)
    for p in points:
        if capped == p.to:
            return _point_result(x, p)

    if capped < first.to:
        if below == "smallest":
            return _point_result(x, first)
        if below != "pseudo_harmonic":
            raise ValueError(f"unknown rule for small bets: {below!r}")
        p_low = pseudo_harmonic(x, 0.0, first.fraction)
        if rng.random() < p_low:
            return Translation(x, None, (), 0.0, first.fraction, p_low)
        return Translation(x, first.indices[-1], first.indices, 0.0, first.fraction, p_low)

    for lo, hi in zip(points, points[1:]):
        if lo.to < capped < hi.to:
            # Clamp guards against float noise: x lies between the fractions in exact chips.
            xx = min(max(x, lo.fraction), hi.fraction)
            p_low = pseudo_harmonic(xx, lo.fraction, hi.fraction)
            pick = lo if rng.random() < p_low else hi
            return Translation(x, pick.indices[-1], pick.indices, lo.fraction, hi.fraction, p_low)
    raise AssertionError("unreachable: the bet lies between two points")  # pragma: no cover


def view_before_action(view: PublicState, index: int = -1) -> PublicState:
    """The actor's view just before ``view.history[index]`` was taken.

    Rebuilt by replaying the public record through :mod:`betting` itself
    (``new_hand`` with the same stacks and blinds, every action, and the
    public board cards when a street opens), so its pot, current bet,
    ``min_to`` and ``max_to`` are exactly what the bettor faced. Only public
    information is used.
    """
    history = view.history
    if not history:
        raise ValueError("no actions in the history")
    target = index % len(history)
    state = new_hand(
        button=view.button,
        stacks=view.starting_stacks,
        small_blind=view.small_blind,
        big_blind=view.big_blind,
    )
    for i, record in enumerate(history[: target + 1]):
        while state.cards_needed:
            n = len(state.board)
            state = deal_board(state, view.board[n : n + state.cards_needed])
        if i == target:
            return public_view(state, record.player)
        state = apply_action(state, record.player, record.action)
    raise AssertionError("unreachable")  # pragma: no cover


def translate_last_action(
    view: PublicState,
    sizes: Sequence[float],
    rng: np.random.Generator,
    *,
    below: BelowRule = "pseudo_harmonic",
) -> Translation:
    """:func:`translate_bet` for the last action in ``view.history`` (a Bet or Raise)."""
    if not view.history:
        raise ValueError("no actions in the history")
    action = view.history[-1].action
    match action:
        case Bet(amount=to) | Raise(to=to):
            pass
        case _:
            raise ValueError(f"last action {action!r} is not a bet or raise")
    return translate_bet(view_before_action(view), to, sizes, rng, below=below)


def real_action_for_fraction(state: PublicState, fraction: float) -> Action:
    """Our abstract "bet ``fraction`` of the pot" as a legal real action.

    Sized from the **real** pot (the abstract pot differs after any
    translation), clamped to ``[min_to, effective_max_to]``. If the real
    state allows no bet or raise at all (e.g. the opponent is already
    all-in, though the abstract node still had a raise), returns ``Call()``
    or ``Check()`` instead: the closest legal action that keeps the hand
    going.
    """
    legal = state.legal
    if legal is None:
        raise ValueError("nobody is to act in this state")
    if not (legal.can_bet or legal.can_raise):
        return Call() if legal.can_call else Check()
    return aggressive_action(legal, to_for_fraction(state, fraction))


def label_fraction(state: PublicState, label: SizeLabel) -> float:
    """A tree edge's :class:`~nutsfinder.abstraction.tree.SizeLabel` as a pot fraction at ``state``.

    ``pot:f`` is ``f``; ``all_in`` is :data:`ALL_IN`; ``open_bb:n`` is
    :func:`fraction_for_bb` at the real spot (floored at 0: an "open to 3bb"
    below the real current bet, possible only with odd real blinds, means
    the smallest raise). Use it to build :func:`translate_bet`'s ``sizes``
    from the edges of the abstract node actually reached.
    """
    if label.kind == "all_in":
        return ALL_IN
    if label.kind == "open_bb":
        return max(0.0, fraction_for_bb(state, label.value))
    return label.value
