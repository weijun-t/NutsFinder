"""Settling a finished N-player hand: each pot, its winners, everyone's net.

- **Everyone else folded:** the last live player takes the whole pot (after
  the uncalled return), with no cards compared. It is reported as one pot.
- **Showdown:** every live hand is evaluated with :func:`evaluator.evaluate`
  (two hole cards + five board cards). The pots are layered as in
  :mod:`.pots`; each pot goes to the best hand among its eligible seats, and
  a tie splits it with odd chips clockwise from the button (per pot).

Showdown order (TDA Rule 16): the last bettor/raiser on the river shows
first; if nobody bet the river (including all-in run-outs, where TDA Rule 15
tables every hand at once anyway) the first live seat left of the button
shows first. Every live hand is revealed; folded hands never are.

``net[seat] = final_stack - starting_stack``; the nets always sum to 0
(checked on every hand; a violation raises ``RuntimeError``).
"""

from __future__ import annotations

from dataclasses import dataclass

from nutsfinder.engine.betting import Street
from nutsfinder.engine.evaluator import HandRank, evaluate
from nutsfinder.engine.multiway.pots import Pot, side_pots, split_amount
from nutsfinder.engine.multiway.positions import clockwise_from
from nutsfinder.engine.multiway.state import MultiHandState

__all__ = ["PotAward", "MultiSettlement", "settle", "showdown_order"]


@dataclass(frozen=True, slots=True)
class PotAward:
    """One pot and how it was paid. ``shares`` is per seat in ``winners`` order."""

    amount: int
    eligible: tuple[int, ...]
    winners: tuple[int, ...]
    shares: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MultiSettlement:
    """How a finished hand paid out. Per-seat tuples are indexed by seat.

    Attributes:
        pots: each pot (main first) with its eligible seats and winners.
        winners: every seat that won chips, in seat order.
        ranks: each seat's hand rank at showdown, ``None`` if folded or no
            showdown happened.
        winnings: chips each seat took from the pots.
        final_stacks: stacks after collecting.
        net: ``final_stacks - starting_stacks``; sums to 0.
    """

    pots: tuple[PotAward, ...]
    winners: tuple[int, ...]
    ranks: tuple[HandRank | None, ...]
    winnings: tuple[int, ...]
    final_stacks: tuple[int, ...]
    net: tuple[int, ...]


def settle(state: MultiHandState) -> MultiSettlement:
    """Pay out a finished hand (``ValueError`` if it is not over, or a showdown
    lacks hole cards). Chip conservation is checked (``RuntimeError``)."""
    if not state.is_hand_over:
        raise ValueError("the hand is not over yet")
    n = state.n
    if state.is_showdown:
        ranks = _showdown_ranks(state)
        pots = side_pots(state.total_bets, state.live)
        awards = tuple(_award(p, ranks, n, state.button) for p in pots)
    else:
        ranks = (None,) * n
        winner = state.live.index(True)
        awards = (PotAward(state.pot, (winner,), (winner,), (state.pot,)),)
    winnings = [0] * n
    for award in awards:
        for seat, share in zip(award.winners, award.shares, strict=True):
            winnings[seat] += share
    final = tuple(s + w for s, w in zip(state.stacks, winnings, strict=True))
    net = tuple(f - s for f, s in zip(final, state.starting_stacks, strict=True))
    if sum(winnings) != state.pot:
        raise RuntimeError(f"pots paid {sum(winnings)} but the pot is {state.pot}")
    if sum(final) != sum(state.starting_stacks) or sum(net) != 0:
        raise RuntimeError(f"chips not conserved: final {final}, net {net}")
    winners = tuple(s for s in range(n) if winnings[s] > 0)
    return MultiSettlement(awards, winners, ranks, tuple(winnings), final, net)


def _showdown_ranks(state: MultiHandState) -> tuple[HandRank | None, ...]:
    if len(state.hole_cards) != state.n:
        raise ValueError("a showdown needs every seat's hole cards in the state")
    return tuple(
        None if state.folded[s] else evaluate([*state.hole_cards[s], *state.board])
        for s in range(state.n)
    )


def _award(pot: Pot, ranks: tuple[HandRank | None, ...], n: int, button: int) -> PotAward:
    """Give ``pot`` to the best hand(s) among its eligible seats."""
    best = max(ranks[s] for s in pot.eligible)  # type: ignore[type-var]
    winners = tuple(s for s in pot.eligible if ranks[s] == best)
    split = split_amount(pot.amount, winners, n, button)
    return PotAward(pot.amount, pot.eligible, winners, tuple(split[s] for s in winners))


def showdown_order(state: MultiHandState) -> tuple[int, ...]:
    """Live seats in the order they show (TDA Rule 16, see module doc)."""
    n = state.n
    if state.street == Street.RIVER and state.last_aggressor is not None:
        start = state.last_aggressor
    else:
        start = (state.button + 1) % n
    return tuple(s for s in clockwise_from(n, start) if not state.folded[s])
