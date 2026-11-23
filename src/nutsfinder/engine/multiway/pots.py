"""Uncalled bets, layered side pots and odd-chip splitting for N players.

Uncalled bet
------------
When a street closes, or everyone else folds, take the highest total
contribution ``c_max`` over **all** players (folded ones included) and the
second highest ``c_second``. ``c_max - c_second`` was never matched by
anybody, so it goes back to its owner. Folded players count, because a
player who called and later folded did match part of the bet. Ties mean
nothing is returned. Heads-up this is exactly the HU engine's rule.

Side pots
---------
Inputs: each seat's final ``total_bets`` (after the uncalled return) and
which seats are still live (not folded)::

    levels = sorted({c[i] for i live})
    prev = 0
    for L in levels:
        amount   = sum(min(c[j], L) - min(c[j], prev) for every seat j)
        eligible = [i live with c[i] >= L]
        prev = L

Folded chips fall into whichever layers they reach. After the uncalled
return nothing is left above the top live level; :func:`side_pots` still
puts any such chips into the top pot rather than lose them (mid-street
summaries can in principle see this) so chips are always conserved.

Odd chips (TDA Rules 19-20): each pot is split on its own; any remainder
goes one chip at a time to the winners in clockwise order starting from the
first seat left of the button.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nutsfinder.engine.multiway.positions import clockwise_from

__all__ = ["Pot", "uncalled_bet", "side_pots", "split_amount"]


@dataclass(frozen=True, slots=True)
class Pot:
    """One pot layer: its chips and the live seats that can win it (seat order)."""

    amount: int
    eligible: tuple[int, ...]


def uncalled_bet(total_bets: Sequence[int]) -> tuple[int | None, int]:
    """``(seat, chips)`` of the unmatched top contribution, or ``(None, 0)``."""
    if len(total_bets) < 2:
        raise ValueError("need at least two seats")
    order = sorted(range(len(total_bets)), key=lambda s: total_bets[s], reverse=True)
    top, second = total_bets[order[0]], total_bets[order[1]]
    if top > second:
        return order[0], top - second
    return None, 0


def side_pots(total_bets: Sequence[int], live: Sequence[bool]) -> tuple[Pot, ...]:
    """Layer the contributions into a main pot and side pots (see module doc).

    Returns pots from the main pot upward; every eligible set contains the
    next one. Zero-chip layers cannot occur (levels are distinct and > 0
    contributions sit below each), except when all live contributions are
    0, which gives no pots at all.
    """
    if len(total_bets) != len(live):
        raise ValueError("total_bets and live must have the same length")
    levels = sorted({c for c, alive in zip(total_bets, live, strict=True) if alive and c > 0})
    pots: list[Pot] = []
    prev = 0
    for level in levels:
        amount = sum(min(c, level) - min(c, prev) for c in total_bets)
        eligible = tuple(s for s, c in enumerate(total_bets) if live[s] and c >= level)
        pots.append(Pot(amount, eligible))
        prev = level
    overflow = sum(c - prev for c in total_bets if c > prev)
    if overflow:
        if pots:
            pots[-1] = Pot(pots[-1].amount + overflow, pots[-1].eligible)
        else:
            alive = tuple(s for s in range(len(live)) if live[s])
            pots.append(Pot(overflow, alive))
    return tuple(pots)


def split_amount(amount: int, winners: Sequence[int], n: int, button: int) -> dict[int, int]:
    """Split ``amount`` among ``winners``; odd chips go clockwise from the button.

    Contract: shares sum to ``amount``, differ by at most 1, and the larger
    shares go to the winners that come first clockwise starting at
    ``button + 1``.
    """
    if not winners:
        raise ValueError("a pot needs at least one winner")
    ordered = [s for s in clockwise_from(n, (button + 1) % n) if s in set(winners)]
    share, odd = divmod(amount, len(ordered))
    return {seat: share + (1 if k < odd else 0) for k, seat in enumerate(ordered)}
