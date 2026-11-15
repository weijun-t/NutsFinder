"""The five betting actions a player can take: fold, check, call, bet, raise.

Why ``Bet`` and ``Raise`` are separate types
--------------------------------------------
Mechanically a bet and a raise do the same thing: they set the player's total
on this street to some amount. We could have one ``BetTo`` action. We keep
them separate anyway, because the *rules* that make each one legal differ and
mixing them up is a common bot bug:

- ``Bet`` is only legal when nobody has put chips in on this street yet
  (the current bet is 0). That only happens on the flop, turn and river:
  preflop the big blind already counts as a bet, so the first aggressive
  action preflop is a *raise*.
- ``Raise`` is only legal when there is already a bet to raise.

So a bot that tries to "bet" into an existing bet gets a clear ``ValueError``
instead of the engine silently guessing what it meant. Logs also read like
real poker ("bets 300", "raises to 900").

Amounts are always a "to" total, never a "by"
---------------------------------------------
Both amounts mean "my total chips in on *this street* after the action":

- ``Bet(300)``: put 300 in. On a fresh street you had 0 in, so the bet amount
  and the "to" total are the same number.
- ``Raise(900)``: make my street total 900. If I already had 300 in (say I bet
  300 and got raised), only 600 more chips leave my stack.

``Call`` and ``Check`` carry no amount: the engine works out how much a call
costs (possibly less than the full bet, if the caller is all-in).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Fold", "Check", "Call", "Bet", "Raise", "Action"]


@dataclass(frozen=True, slots=True)
class Fold:
    """Give up the hand.

    Always legal for the player to act, even when checking is free. Folding
    then is never better than checking, but the rules allow it, so the engine
    does too (see ``betting.py``).
    """

    def __str__(self) -> str:
        return "fold"


@dataclass(frozen=True, slots=True)
class Check:
    """Pass without putting chips in. Only legal when not facing a bet."""

    def __str__(self) -> str:
        return "check"


@dataclass(frozen=True, slots=True)
class Call:
    """Match the current bet (or go all-in for less if the stack is short)."""

    def __str__(self) -> str:
        return "call"


def _check_positive_int(name: str, value: object) -> None:
    # bool is a subclass of int in Python (True == 1), so reject it explicitly:
    # Bet(True) is almost certainly a bug, not a 1-chip bet.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True, slots=True)
class Bet:
    """Open the betting on a street: put ``amount`` chips in.

    ``amount`` is the player's street total after betting (the same as the
    chips added, because nothing was in yet). Must be a positive int.
    """

    amount: int

    def __post_init__(self) -> None:
        _check_positive_int("bet amount", self.amount)

    def __str__(self) -> str:
        return f"bet {self.amount}"


@dataclass(frozen=True, slots=True)
class Raise:
    """Raise an existing bet so the player's street total becomes ``to``.

    ``to`` is the total, not the increase: facing a bet of 300, ``Raise(900)``
    means "make it 900", an increase of 600 over the bet. Must be a positive int.
    """

    to: int

    def __post_init__(self) -> None:
        _check_positive_int("raise-to amount", self.to)

    def __str__(self) -> str:
        return f"raise to {self.to}"


Action = Fold | Check | Call | Bet | Raise
"""Any one of the five betting actions."""
