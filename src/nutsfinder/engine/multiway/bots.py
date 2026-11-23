"""Baseline multiway bots, and an adapter that seats a heads-up bot at N=2.

The three baselines make exactly the same decisions (and consume their RNG
exactly the same way) as their heads-up counterparts in ``nutsfinder.bots``;
they only read ``view.legal``, ``view.pot`` and ``view.current_bet``. The
N=2 differential test relies on that.

:class:`HUBotAdapter` wraps any heads-up :class:`~nutsfinder.engine.bot.Bot`
so it can play in the multiway engine **at two players only**: it converts
the view and the end-of-hand summary to the heads-up types.
"""

from __future__ import annotations

import random

from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import PublicState
from nutsfinder.engine.bot import Bot, HandSummary
from nutsfinder.engine.cards import Card
from nutsfinder.engine.multiway.bot import MultiBot, MultiHandSummary
from nutsfinder.engine.multiway.state import MultiPublicState
from nutsfinder.engine.showdown import Settlement

__all__ = [
    "MultiRandomBot",
    "MultiCallStationBot",
    "MultiAlwaysRaiseBot",
    "HUBotAdapter",
    "to_hu_view",
]


def _require_turn(view: MultiPublicState) -> None:
    legal = view.legal
    if legal is None or legal.player != view.seat:
        raise ValueError(f"seat {view.seat} was asked to act out of turn")


class MultiRandomBot(MultiBot):
    """Uniform over legal action kinds (fold, check/call, bet/raise), then sizes."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:
        _require_turn(view)
        legal = view.legal
        assert legal is not None
        kinds = ["fold", "check" if legal.can_check else "call"]
        if legal.can_bet or legal.can_raise:
            kinds.append("bet" if legal.can_bet else "raise")
        kind = self._rng.choice(kinds)
        if kind in ("bet", "raise"):
            size = self._rng.randint(legal.min_to, legal.max_to)
            return Bet(size) if kind == "bet" else Raise(size)
        return {"fold": Fold(), "check": Check(), "call": Call()}[kind]

    def __repr__(self) -> str:
        return "MultiRandomBot()"


class MultiCallStationBot(MultiBot):
    """Check if checking is legal, otherwise call. Never folds or raises."""

    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:
        _require_turn(view)
        assert view.legal is not None
        return Check() if view.legal.can_check else Call()

    def __repr__(self) -> str:
        return "MultiCallStationBot()"


class MultiAlwaysRaiseBot(MultiBot):
    """Pot-sized bet/raise whenever legal (clamped to the range), else check/call.

    Pot-sized raise-to = ``current_bet + pot + call``, where ``pot`` counts
    every chip in, this street included (same formula as heads-up).
    """

    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:
        _require_turn(view)
        legal = view.legal
        assert legal is not None
        if legal.can_bet or legal.can_raise:
            target = view.current_bet + view.pot + legal.call_amount
            size = min(max(target, legal.min_to), legal.max_to)
            return Bet(size) if legal.can_bet else Raise(size)
        return Check() if legal.can_check else Call()

    def __repr__(self) -> str:
        return "MultiAlwaysRaiseBot()"


# --- Heads-up adapter -------------------------------------------------------


def _pair(values: tuple, name: str) -> tuple:  # type: ignore[type-arg]
    if len(values) != 2:
        raise ValueError(f"heads-up adapter needs 2 seats, {name} has {len(values)}")
    return (values[0], values[1])


def to_hu_view(view: MultiPublicState) -> PublicState:
    """The heads-up ``PublicState`` equivalent of a 2-player multiway view."""
    if view.n_players != 2:
        raise ValueError(f"heads-up adapter needs 2 players, got {view.n_players}")
    folded = [s for s in (0, 1) if view.folded[s]]
    return PublicState(
        seat=view.seat,
        button=view.button,
        small_blind=view.small_blind,
        big_blind=view.big_blind,
        starting_stacks=_pair(view.starting_stacks, "starting_stacks"),
        blinds=_pair(view.blinds, "blinds"),
        stacks=_pair(view.stacks, "stacks"),
        street_bets=_pair(view.street_bets, "street_bets"),
        total_bets=_pair(view.total_bets, "total_bets"),
        pot=view.pot,
        current_bet=view.current_bet,
        last_raise=view.last_raise,
        street=view.street,
        board=view.board,
        to_act=view.to_act,
        folded=folded[0] if folded else None,
        history=view.history,
        refunds=view.refunds,
        legal=view.legal,
    )


def to_hu_summary(summary: MultiHandSummary) -> HandSummary:
    """The heads-up ``HandSummary`` equivalent of a 2-player summary."""
    r = summary.result
    settlement = Settlement(
        winners=r.winners,
        ranks=_pair(r.ranks, "ranks"),
        winnings=_pair(r.winnings, "winnings"),
        final_stacks=_pair(r.final_stacks, "final_stacks"),
        net=_pair(r.net, "net"),
    )
    return HandSummary(
        seat=summary.seat,
        final_view=to_hu_view(summary.final_view),
        shown_cards=_pair(summary.shown_cards, "shown_cards"),
        result=settlement,
    )


class HUBotAdapter(MultiBot):
    """Seat a heads-up :class:`Bot` in the multiway engine (2 players only)."""

    def __init__(self, bot: Bot) -> None:
        self.bot = bot

    def act(self, view: MultiPublicState, hole_cards: tuple[Card, ...]) -> Action:
        return self.bot.act(to_hu_view(view), hole_cards)

    def end_hand(self, summary: MultiHandSummary) -> None:
        self.bot.end_hand(to_hu_summary(summary))

    def reseed(self, seed: int) -> None:
        self.bot.reseed(seed)

    def __repr__(self) -> str:
        return f"HUBotAdapter({self.bot!r})"
