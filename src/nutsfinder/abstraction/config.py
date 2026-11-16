"""The abstraction spec: which bet sizes and how many card buckets.

One frozen dataclass, :class:`AbstractionConfig`, holds every choice that
shapes the abstract game, so a trained blueprint can store the config it was
trained with and a bot can rebuild the exact same tree. The defaults are the
"recommended minimal abstraction" of docs/phase6_research.md section 3:

=========  ==================================  ===========================================  =====================
Street     Card buckets                        Actions                                      Raise cap
=========  ==================================  ===========================================  =====================
Preflop    169 classes (lossless)              fold, call, open to 3bb, pot raise, all-in   3 non-all-in raises
Postflop   50 E[HS] buckets each (imperfect    fold, check/call, pot bet/raise, all-in      2 (bet + raise)
           recall)
=========  ==================================  ===========================================  =====================

Bet sizes are fractions of **the pot after calling**, the convention of
:func:`nutsfinder.bots.common.raise_to_for_fraction`:

    raise_to = current_bet + fraction * (pot + to_call)

Worked example (preflop, SB to act, blinds 50/100): pot 150, to_call 50, so
a pot raise is to 100 + 1.0 * 200 = 300 = 3bb. That is also the 3bb open, so
the first preflop raise has one non-all-in size either way. After a 3bb open
the BB faces pot 400, to_call 200: pot 3-bet to 300 + 600 = 900. Then the
SB: pot 1,200, to_call 600, pot 4-bet to 900 + 1,800 = 2,700. That is the
third non-all-in raise (open, 3-bet, 4-bet), so the BB may only fold, call
or go all-in (a pot 5-bet would have been to 8,100).

Raise cap semantics: the cap counts the *non-all-in* bets and raises made on
the current street (the big blind is a forced bet, not a raise). Once the
cap is reached the player may still fold, call or go all-in. All-in is
offered whenever the rules allow any raise at all.

Fold is offered only when facing a bet: folding when a check is free is
legal in the engine but never better than checking, so the abstraction
leaves it out (as Leduc does).
"""

from __future__ import annotations

from dataclasses import dataclass

from nutsfinder.engine.betting import BIG_BLIND, SMALL_BLIND, STARTING_STACK

__all__ = ["AbstractionConfig", "DEFAULT_CONFIG", "PREFLOP_BUCKETS"]

PREFLOP_BUCKETS = 169
"""Preflop buckets are the 169 hand classes (lossless)."""


@dataclass(frozen=True, slots=True)
class AbstractionConfig:
    """Action and card abstraction of the heads-up game (see module doc).

    Attributes:
        stack: both players' starting stack in chips (the engine's default 100bb).
        small_blind, big_blind: blinds in chips (engine defaults 50 / 100).
        preflop_open_bb: size of the *first* preflop raise, as a raise-to total
            in big blinds (3.0 -> raise to 300). Used instead of the pot
            fractions for that raise (for the SB open, and for the BB's raise
            after a limp).
        preflop_fractions: pot fractions for later preflop raises (3-bet on).
        postflop_fractions: pot fractions for flop, turn and river bets/raises.
        preflop_cap: max non-all-in raises preflop.
        postflop_cap: max non-all-in bets + raises per postflop street.
        postflop_buckets: E[HS] buckets on the flop, turn and river.
        ehs_method: ``"exact"`` (exact E[HS] by enumeration, no noise) or
            ``"mc"`` (Monte Carlo with ``mc_samples`` samples, seeded from the
            suit-canonical hand and board).
        mc_samples: samples per E[HS] estimate when ``ehs_method == "mc"``.
    """

    stack: int = STARTING_STACK
    small_blind: int = SMALL_BLIND
    big_blind: int = BIG_BLIND
    preflop_open_bb: float = 3.0
    preflop_fractions: tuple[float, ...] = (1.0,)
    postflop_fractions: tuple[float, ...] = (1.0,)
    preflop_cap: int = 3
    postflop_cap: int = 2
    postflop_buckets: tuple[int, int, int] = (50, 50, 50)
    ehs_method: str = "exact"
    mc_samples: int = 1000

    def __post_init__(self) -> None:
        if not 1 <= self.small_blind <= self.big_blind < self.stack:
            raise ValueError("need 1 <= small_blind <= big_blind < stack")
        if self.preflop_open_bb * self.big_blind <= self.big_blind:
            raise ValueError("the preflop open must be a raise (more than 1bb)")
        if any(f <= 0 for f in self.preflop_fractions + self.postflop_fractions):
            raise ValueError("pot fractions must be positive")
        if self.preflop_cap < 1 or self.postflop_cap < 0:
            raise ValueError("raise caps: preflop >= 1, postflop >= 0")
        if len(self.postflop_buckets) != 3 or min(self.postflop_buckets) < 1:
            raise ValueError("postflop_buckets: three positive counts (flop, turn, river)")
        if self.ehs_method not in ("exact", "mc") or self.mc_samples < 1:
            raise ValueError("ehs_method is 'exact' or 'mc' (with mc_samples >= 1)")

    def buckets(self, street: int) -> int:
        """Card buckets on ``street`` (0 = preflop, 1-3 = flop, turn, river)."""
        return PREFLOP_BUCKETS if street == 0 else self.postflop_buckets[street - 1]

    def cap(self, street: int) -> int:
        """Non-all-in raise cap on ``street``."""
        return self.preflop_cap if street == 0 else self.postflop_cap

    def fractions(self, street: int, raises_so_far: int) -> tuple[float, ...]:
        """Pot fractions to offer (empty for the preflop open, which has a fixed size)."""
        if street == 0:
            return () if raises_so_far == 0 else self.preflop_fractions
        return self.postflop_fractions


DEFAULT_CONFIG = AbstractionConfig()
"""The recommended minimal abstraction (100bb, 50/100 blinds)."""
