"""Matches between two bots, A and B, measured from A's point of view.

Two kinds of match, both reproducible from one ``master_seed`` and both with
a length fixed *before* the first hand is dealt (there is deliberately no
"stop when it looks significant" option: peeking at the p-value and stopping
when it dips under 0.05 makes false positives far more common than 5%).

Plain match (:func:`plain_match`)
---------------------------------
``n_hands`` hands, A in seat 0, the button alternating (A has it on even
hands). Unit = one hand, result = A's net in big blinds.

Duplicate match (:func:`duplicate_match`)
-----------------------------------------
Most of the noise in poker is card luck: who got aces. A duplicate match
cancels it by playing every deal twice with the bots swapped. For pair i:

- game 1: ``play_hand([A, B], deal_seed_i, button=b_i)``
- game 2: ``play_hand([B, A], deal_seed_i, button=b_i)``

*Why this is "the same deal with the seats swapped".* The engine deals by
seat: seat 0 always gets the first two cards of the shuffled deck, seat 1
the next two, then the board, whatever the bots do. So with the same seed
the seat-0 cards, the seat-1 cards and the board are identical in both
games. The button is a *seat* too, and we keep it on the same seat ``b_i``
in both games. Swapping which bot sits where therefore moves A into B's
seat with B's cards *and* B's position (button or big blind), and vice
versa. Over one pair, A plays both hands of the deal from both positions.
(If we also flipped the button in game 2, A would get B's cards but keep
its own position, so position luck would not cancel.)

*Across pairs* the button seat alternates, ``b_i = i % 2``. Within a pair
both roles are covered anyway, so this is not needed for fairness; it just
means seat 0 and seat 1 hold the button equally often, so a hypothetical
bug that favoured a seat *index* could not hide in the results.

*Two "rooms".* Game 1 and game 2 are played by separate bot instances (one
copy of A and B in each room, made by the factories). That matters for bots
with memory (Phase 5): a single A that played game 1 would remember the
cards B showed and could use them in game 2. With rooms, each copy only
ever sees its own room's hands, exactly like a duplicate bridge or poker
tournament played at two tables.

*The unit is the pair.* Game 1 and game 2 share the cards, so their results
are strongly (negatively) correlated, and treating them as two independent
hands would get the SE wrong. Result of pair i = A's net in game 1 + A's
net in game 2, in bb. bb/100 = mean per pair / 2 hands * 100.

Why this reduces variance: Var(X1 + X2) = Var(X1) + Var(X2) + 2 Cov(X1, X2).
If A wins big with the aces in game 1, B usually wins big with the same
aces in game 2, so Cov is negative and the pair's variance is much smaller
than two independent hands'. For two deterministic copies of the same bot,
game 2 is game 1 with the labels swapped, so X2 = -X1 and every pair is
exactly 0: a perfect self-play check (tested).

Bot randomness
--------------
Bots with an RNG (``RandomBot``) are reseeded before every hand via
:meth:`Bot.reseed`, from seeds derived from the master seed: for unit i,
``(deal_seed_i, seed_a_i, seed_b_i)``. In a duplicate pair, A gets
``seed_a_i`` in *both* games and B gets ``seed_b_i`` in both ("synced by
bot, re-synced every hand"). Why:

- Reproducible: a hand's random choices depend only on (master seed, i),
  not on how many random numbers earlier hands happened to use.
- Variance reduction: the noise we want to cancel includes the bots' own
  dice. With the same stream in both games, a RandomBot tends to make
  similar choices in both, so its own luck partly cancels too. Measured on
  RandomBot vs AlwaysRaiseBot (4,000 pairs), the SD per pair was 58 bb with
  this syncing vs 72 bb with independent streams. Syncing *per hand*
  matters: one persistent stream drifts out of sync after the first hand
  where the two games use a different number of draws, and then it is no
  better than independent streams.
- It is not a free lunch. Against CallStationBot, RandomBot's synced choices
  are often *identical* in both games (it ignores its cards, and its legal
  options are often the same from either seat), so showdowns cancel exactly
  but its fold losses happen twice and add up. The two effects roughly
  offset (SD per pair 63 bb either way). The estimate is unbiased whichever
  way the streams are chosen; only its variance changes.
- Consequence for self-play: RandomBot vs RandomBot does *not* cancel
  exactly, because the two copies have different seeds (A's seed_a vs B's
  seed_b), so game 2 is not a mirror of game 1. It is still exactly fair by
  symmetry (A and B are interchangeable), so it must come out within the CI
  of 0, which is a genuine test of the CI rather than a trivial 0 = 0.

In a plain match the same per-hand reseeding is used (A gets seed_a_i, B
gets seed_b_i), so plain and duplicate runs with the same master seed share
deal seeds and bot seeds hand for hand.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from nutsfinder.engine.betting import BIG_BLIND
from nutsfinder.engine.bot import Bot
from nutsfinder.engine.hand_log import HandLog
from nutsfinder.engine.runner import play_hand
from nutsfinder.eval.stats import Summary, required_units, summarize

__all__ = ["BotFactory", "Game", "MatchResult", "describe_bot", "plain_match", "duplicate_match", "unit_seeds"]

BotFactory = Callable[[], Bot]
"""Anything that builds a fresh bot with no arguments, e.g. ``RandomBot``."""


@dataclass(frozen=True, slots=True)
class Game:
    """One played hand, and where bot A sat in it.

    Attributes:
        unit: index of the unit (hand or pair) this game belongs to.
        log: the full hand log.
        a_seat: the seat bot A occupied (0 or 1).
    """

    unit: int
    log: HandLog
    a_seat: int

    @property
    def a_net_bb(self) -> float:
        return self.log.result.net[self.a_seat] / BIG_BLIND

    @property
    def a_on_button(self) -> bool:
        return self.log.button == self.a_seat


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Everything a match produced, from bot A's point of view.

    Attributes:
        kind: ``"plain"`` or ``"duplicate"``.
        bot_a, bot_b: the bots' names (class names, as before).
        master_seed: the single seed the whole match is reproducible from.
        unit: ``"hand"`` or ``"pair"``.
        unit_results: A's result per unit, in bb.
        games: every hand played, in order (two per pair in a duplicate).
        stats: mean / SD / SE / CI / p-value of ``unit_results``.
        bot_a_desc, bot_b_desc: :func:`describe_bot` of each bot (its
            ``repr``: e.g. CFRBot's model file, iteration, commit and
            abstraction fingerprint). Default ``""`` for results built
            without it.
    """

    kind: str
    bot_a: str
    bot_b: str
    master_seed: int
    unit: str
    unit_results: tuple[float, ...]
    games: tuple[Game, ...]
    stats: Summary
    bot_a_desc: str = ""
    bot_b_desc: str = ""

    @property
    def n_units(self) -> int:
        return len(self.unit_results)

    @property
    def hands_played(self) -> int:
        return len(self.games)

    @property
    def mean_bb100(self) -> float:
        return self.stats.mean_bb100

    @property
    def sd_per_unit(self) -> float:
        """SD of one unit's result, in bb (per hand or per pair)."""
        return self.stats.sd

    @property
    def se_bb100(self) -> float:
        return self.stats.se_bb100

    @property
    def ci_bb100(self) -> tuple[float, float]:
        return self.stats.ci_bb100

    @property
    def p_value(self) -> float:
        return self.stats.p_value

    def hands_needed(self, power: float = 0.8) -> int | None:
        """Hands needed to detect the *observed* edge at this noise level.

        Uses :func:`required_units` with sigma = the measured SD per unit and
        e = the measured mean per unit, then converts units to hands. ``None``
        if the observed mean is exactly 0 (nothing to detect).
        """
        if self.stats.mean == 0:
            return None
        units = required_units(self.stats.sd, self.stats.mean, power=power)
        return units * self.stats.hands_per_unit

    def verdict(self) -> str:
        """Significant edge / no detectable edge (with what it would take)."""
        lo, hi = self.ci_bb100
        if self.stats.no_variation:
            return (
                f"no variation: every {self.unit} gave exactly {self.stats.mean:+g} bb, so "
                "there is no CI or p-value. This is not a statistical estimate; it is only "
                "meaningful if the setup forces it (e.g. deterministic bots whose duplicate "
                "pairs cancel exactly)"
            )
        if self.stats.significant:
            who = self.bot_a if self.stats.mean > 0 else self.bot_b
            p = "< 1e-300" if self.p_value == 0 else f"= {self.p_value:.2g}"
            return f"significant edge for {who} (p {p})"
        bound = max(abs(lo), abs(hi))
        text = f"no detectable edge: the 95% CI rules out edges beyond +/-{bound:.1f} bb/100"
        needed = self.hands_needed()
        if needed is None:
            return text + " (observed mean is exactly 0)"
        return (
            text + f"; underpowered for the observed {self.mean_bb100:+.2f} bb/100: "
            f"detecting it with 80% power would need about {needed:,} hands"
        )

    def summary(self) -> str:
        """A readable multi-line report."""
        lo, hi = self.ci_bb100
        s = self.stats
        return "\n".join(
            [
                f"{self.bot_a} vs {self.bot_b}: {self.kind} match, master seed {self.master_seed}",
                f"  {self.n_units:,} {self.unit}s, {self.hands_played:,} hands",
                f"  win rate for {self.bot_a}: {self.mean_bb100:+.2f} bb/100 "
                f"(95% CI {lo:+.2f} to {hi:+.2f}, normal approximation)",
                f"  SD {s.sd:.2f} bb per {self.unit}, SE {self.se_bb100:.2f} bb/100, "
                f"p = {self.p_value:.3g} (two-sided z-test vs 0)",
                f"  verdict: {self.verdict()}",
            ]
        )


def unit_seeds(master_seed: int, n_units: int) -> list[tuple[int, int, int]]:
    """Per-unit ``(deal_seed, seed_a, seed_b)``, all drawn from one master seed."""
    rng = random.Random(master_seed)
    return [(rng.getrandbits(64), rng.getrandbits(64), rng.getrandbits(64)) for _ in range(n_units)]


def _name(bot: Bot) -> str:
    return type(bot).__name__


def describe_bot(bot: Bot) -> str:
    """``repr(bot)``, which identifies its configuration (a trained vs untrained CFRBot differ here).

    Falls back to the class name if the bot has no ``__repr__`` of its own
    (the default one contains a memory address, which is not reproducible).
    """
    if type(bot).__repr__ is object.__repr__:
        return _name(bot)
    return repr(bot)


def plain_match(make_a: BotFactory, make_b: BotFactory, n_hands: int, master_seed: int) -> MatchResult:
    """``n_hands`` independent hands, A in seat 0, button alternating.

    Unit = one hand. The length is fixed up front; there is no early stop.
    """
    if n_hands < 2:
        raise ValueError("need at least 2 hands to estimate a standard deviation")
    a, b = make_a(), make_b()
    desc_a, desc_b = describe_bot(a), describe_bot(b)  # before play: configuration, not state
    games: list[Game] = []
    for i, (deal_seed, seed_a, seed_b) in enumerate(unit_seeds(master_seed, n_hands)):
        a.reseed(seed_a)
        b.reseed(seed_b)
        log = play_hand([a, b], deal_seed, button=i % 2)
        games.append(Game(unit=i, log=log, a_seat=0))
    results = tuple(g.a_net_bb for g in games)
    return MatchResult(
        kind="plain",
        bot_a=_name(a),
        bot_b=_name(b),
        master_seed=master_seed,
        unit="hand",
        unit_results=results,
        games=tuple(games),
        stats=summarize(results, hands_per_unit=1),
        bot_a_desc=desc_a,
        bot_b_desc=desc_b,
    )


def duplicate_match(
    make_a: BotFactory, make_b: BotFactory, n_pairs: int, master_seed: int
) -> MatchResult:
    """``n_pairs`` duplicate pairs (2 * n_pairs hands). See the module docstring.

    Unit = one pair; its result = A's net in game 1 + A's net in game 2.
    """
    if n_pairs < 2:
        raise ValueError("need at least 2 pairs to estimate a standard deviation")
    # Room 1: A in seat 0. Room 2: B in seat 0. Separate instances per room.
    a1, b1 = make_a(), make_b()
    a2, b2 = make_a(), make_b()
    desc_a, desc_b = describe_bot(a1), describe_bot(b1)  # before play: configuration, not state
    games: list[Game] = []
    results: list[float] = []
    for i, (deal_seed, seed_a, seed_b) in enumerate(unit_seeds(master_seed, n_pairs)):
        button = i % 2  # same button *seat* in both games of the pair
        for bot, seed in ((a1, seed_a), (b1, seed_b), (a2, seed_a), (b2, seed_b)):
            bot.reseed(seed)
        game1 = Game(unit=i, log=play_hand([a1, b1], deal_seed, button=button), a_seat=0)
        game2 = Game(unit=i, log=play_hand([b2, a2], deal_seed, button=button), a_seat=1)
        games += [game1, game2]
        results.append(game1.a_net_bb + game2.a_net_bb)
    return MatchResult(
        kind="duplicate",
        bot_a=_name(a1),
        bot_b=_name(b1),
        master_seed=master_seed,
        unit="pair",
        unit_results=tuple(results),
        games=tuple(games),
        stats=summarize(results, hands_per_unit=2),
        bot_a_desc=desc_a,
        bot_b_desc=desc_b,
    )
