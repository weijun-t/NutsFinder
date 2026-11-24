"""MultiBayesBot: BayesBot (tier 3) for 2-8 players, one pooled Beta-Binomial model per opponent seat.

It is :class:`~nutsfinder.bots.multiway.equity_bot.MultiEquityBot` with the
fixed range and fold numbers replaced by learned frequencies, exactly as
:class:`~nutsfinder.bots.bayes_bot.BayesBot` does heads-up (read its module
docstring first: the statistics, the priors and the formulas (b)-(d) are
reused unchanged, as are :class:`~nutsfinder.bots.bayes_bot.BetaStat`,
:data:`~nutsfinder.bots.bayes_bot.DEFAULT_PRIORS`,
:func:`~nutsfinder.bots.bayes_bot.fold_stat_name` and
:func:`~nutsfinder.bots.bayes_bot.bet_fraction`). What is new:

1. One model per opponent seat
==============================
Seats are fixed within a session, so seat identity is opponent identity.
Every opponent seat has its own counts for every statistic.

2. Heads-up and multiway pots are different events
==================================================
Folding to a bet when you are the only one left to beat it is not the same
event as folding to a bet with two other players still in. So the
betting statistics are kept twice, by the number of live players at the
moment of the decision (2 -> ``@hu``, 3+ -> ``@mw``)::

    agg_pre, agg_post, fold_pre, fold_pre_big, fold_post, fold_post_big   x  {@hu, @mw}
    vpip, bluff                                                           (one each: per hand / per showdown)

``vpip`` is a per-hand statistic (every hand starts multiway at a full
table) and ``bluff`` comes from showdowns, which are rare enough already.
Decisions read the context they are in: a bet of ours into one opponent
uses its ``fold_*@hu``, into several its ``fold_*@mw``; a range-narrowing
step for a bet the opponent made with ``n`` players live uses
``agg_*@hu`` or ``agg_*@mw`` accordingly.

3. Partial pooling: a population prior fitted across opponents
==============================================================
Per opponent and per context, data arrive about (N - 1) times more slowly
than heads-up, and some cells (``fold_post_big@hu``) may see a handful of
events in a whole session. So each opponent's prior is not the fixed
``DEFAULT_PRIORS`` entry but a **population prior** estimated from all the
*other* opponents (empirical Bayes, Beta-Binomial hierarchical model)::

    p_j ~ Beta(s μ, s (1 - μ))           for every opponent j (same μ, s)
    k_j | p_j ~ Binomial(n_j, p_j)

- ``μ`` (population mean) is the base prior mean updated with the pooled data
  of the other seats: ``μ_-j = (a0 + Σ_{i≠j} k_i) / (a0 + b0 + Σ_{i≠j} n_i)``.
  Leaving seat ``j`` out keeps its own data from being counted twice.
- ``s`` (the strength: how alike opponents are) by the method of moments,
  over **all** seats, ``j`` included: whether the population is
  heterogeneous is a property of everybody, and leaving ``j`` out would
  make an outlier look typical (two identical calling stations would
  "prove" the population is homogeneous and drown a folder's 23/23 folds
  under a strength-100 prior: the first version did exactly that, caught
  by ``test_bayes_learns_a_station_and_a_folder_apart``). Over the seats
  with at least ``min_n`` chances, the n-weighted variance
  of the raw rates ``r_i = k_i / n_i`` is ``V = Σ n_i (r_i - μ)^2 / Σ n_i``,
  whose expectation is ``τ² + S μ (1 - μ) / Σ n_i`` (between-seat spread
  ``τ²`` plus binomial noise, S seats). So ``τ² = V - S μ(1 - μ)/Σ n_i`` and
  a Beta with that variance has strength ``s = μ(1 - μ)/τ² - 1``, clamped
  to ``[min_strength, max_strength]`` (default [2, 100]); ``τ² <= 0``
  (no visible spread) gives ``max_strength``; ``μ`` here is the pooled mean
  of all seats. With fewer than
  ``min_seats`` (2) such seats ``s`` stays at the base prior's strength (10).

Opponent ``j``'s posterior is then ``Beta(s μ + k_j, s (1 - μ) + n_j - k_j)``.
Effect: if every opponent folds about as often, a new or rarely seen seat
starts at the field's rate with a strong prior (it is probably like the
others); if they differ a lot (a calling station next to a folder), ``s``
falls towards 2 and each seat's own data take over within a few hands.
With no data anywhere this is exactly ``DEFAULT_PRIORS``, so (as HU) the
untrained bot plays exactly like MultiEquityBot (``learn=False`` is tested
to be identical).

The gating constants of BayesBot stay at the *base* prior means
(``p0 = 0.1`` for folds, ``a0 = 0.5`` aggression, ``v0 = 0.7`` vpip,
``b0 = 0.15`` bluff): they define what "folds more than assumed" means and
are not re-estimated.

4. What it learns from, and when
================================
:meth:`MultiBayesBot.end_hand` replays the finished hand from its public
record with the multiway engine's own rules (so every opponent decision
comes with its exact legal actions and the number of live players) and adds
each opponent's events to that seat's counts. Hole cards are used only for
seats whose cards were shown at a showdown (``summary.shown_cards[j]`` not
``None``); nobody else's. Models persist across hands (and across
:meth:`reseed`, which only restarts the RNG); :meth:`reset_model` forgets,
:meth:`fresh_copy` gives an independent copy with empty models.

Known weaknesses
================
- Everything in BayesBot's list, plus: opponents are modelled independently
  (no "the table is tight" correlation beyond the population prior).
- The method-of-moments strength is noisy with few seats (2-3 opponents
  with data); the clamps keep it sane, not optimal.
- Contexts split the data further; ``fold_*@hu`` against a given seat may
  stay near the population for a long time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from nutsfinder.bots.bayes_bot import (
    DEFAULT_PRIORS,
    RANDOM_BLUFF_RATE,
    BetaStat,
    bet_fraction,
    fold_stat_name,
)
from nutsfinder.bots.multiway.common import live_opponents
from nutsfinder.bots.multiway.equity import multiway_equity
from nutsfinder.bots.multiway.equity_bot import MultiEquityBot, aggression_count, calling_share
from nutsfinder.engine.actions import Bet, Call, Fold, Raise
from nutsfinder.engine.betting import Street
from nutsfinder.engine.cards import Card
from nutsfinder.engine.multiway.bot import MultiHandSummary
from nutsfinder.engine.multiway.state import (
    MultiHandState,
    MultiPublicState,
    apply_action,
    deal_board,
    legal_actions,
    new_hand,
)
from nutsfinder.engine.preflop import PREFLOP_EQUITY, hand_class, snap_to_grid

__all__ = [
    "MultiBayesBot",
    "PooledModels",
    "CONTEXT_STATS",
    "SHARED_STATS",
    "stat_key",
    "context_of",
    "multiway_public_replay",
    "multiway_opponent_events",
]

CONTEXT_STATS = ("agg_pre", "agg_post", "fold_pre", "fold_pre_big", "fold_post", "fold_post_big")
SHARED_STATS = ("vpip", "bluff")


def context_of(n_live: int) -> str:
    """``"hu"`` with two live players, else ``"mw"``."""
    return "hu" if n_live <= 2 else "mw"


def stat_key(name: str, context: str | None = None) -> str:
    """``"fold_post@mw"`` for context statistics, the bare name for ``vpip`` / ``bluff``."""
    if name in SHARED_STATS:
        return name
    if name not in CONTEXT_STATS or context not in ("hu", "mw"):
        raise ValueError(f"unknown statistic {name!r} / context {context!r}")
    return f"{name}@{context}"


ALL_KEYS = SHARED_STATS + tuple(stat_key(n, c) for n in CONTEXT_STATS for c in ("hu", "mw"))


def _base(key: str) -> str:
    return key.split("@")[0]


@dataclass
class PooledModels:
    """Per-seat counts with an empirical-Bayes population prior (module doc, section 3).

    Attributes:
        priors: the base priors (``DEFAULT_PRIORS`` by default).
        counts: ``counts[seat][key]`` -> (yes, n) as a :class:`BetaStat` with the base prior.
        hands: hands observed per seat.
    """

    priors: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRIORS))
    min_strength: float = 2.0
    max_strength: float = 100.0
    min_seats: int = 2
    min_n: int = 5
    counts: dict[int, dict[str, BetaStat]] = field(default_factory=dict)
    hands: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.priors) != set(DEFAULT_PRIORS):
            raise ValueError(f"priors must have exactly the keys {sorted(DEFAULT_PRIORS)}")

    def _seat(self, seat: int) -> dict[str, BetaStat]:
        if seat not in self.counts:
            self.counts[seat] = {k: BetaStat(*self.priors[_base(k)]) for k in ALL_KEYS}
            self.hands[seat] = 0
        return self.counts[seat]

    def observe(self, seat: int, events: Sequence[tuple[str, bool]]) -> None:
        """Add one hand's ``(key, yes)`` events for ``seat``."""
        stats = self._seat(seat)
        for key, yes in events:
            stats[key].record(yes)
        self.hands[seat] += 1

    def reset(self) -> None:
        self.counts.clear()
        self.hands.clear()

    def base_mean(self, key: str) -> float:
        a, b = self.priors[_base(key)]
        return a / (a + b)

    def population(self, key: str, exclude: int | None = None) -> tuple[float, float]:
        """``(μ, s)`` of the population prior for ``key``: μ leaves seat ``exclude`` out, s uses every seat."""
        a0, b0 = self.priors[_base(key)]
        data = [(st[key].yes, st[key].n) for seat, st in self.counts.items() if seat != exclude]
        mu = (a0 + sum(k for k, _ in data)) / (a0 + b0 + sum(n for _, n in data))
        everyone = [(st[key].yes, st[key].n) for st in self.counts.values()]
        mu_all = (a0 + sum(k for k, _ in everyone)) / (a0 + b0 + sum(n for _, n in everyone))
        return mu, self._strength(everyone, mu_all, a0 + b0)

    def _strength(self, data: list[tuple[int, int]], mu: float, default: float) -> float:
        """Method-of-moments Beta strength from the seats with at least ``min_n`` chances."""
        rich = [(k, n) for k, n in data if n >= self.min_n]
        if len(rich) < self.min_seats:
            return default
        n_rich = sum(n for _, n in rich)
        v = sum(n * (k / n - mu) ** 2 for k, n in rich) / n_rich
        tau2 = v - len(rich) * mu * (1 - mu) / n_rich
        if tau2 <= 0:
            return self.max_strength
        s = mu * (1 - mu) / tau2 - 1
        return min(self.max_strength, max(self.min_strength, s))

    def stat(self, seat: int, key: str) -> BetaStat:
        """Seat ``seat``'s posterior for ``key``: its counts on the population prior."""
        own = self._seat(seat)[key]
        mu, s = self.population(key, exclude=seat)
        return BetaStat(s * mu, s * (1 - mu), own.yes, own.n)

    def randomness(self, seat: int) -> float:
        """BayesBot's ``m = clamp((b̂ - b0) / (0.5 - b0), 0, 1)`` for this seat."""
        b0 = self.base_mean("bluff")
        m = (self.stat(seat, "bluff").mean - b0) / (RANDOM_BLUFF_RATE - b0)
        return min(1.0, max(0.0, m))

    def lines(self, seat: int) -> list[str]:
        stats = self._seat(seat)
        out = [f"seat {seat} model ({self.hands.get(seat, 0)} hands) m={self.randomness(seat):.2f}"]
        for key in ALL_KEYS:
            if stats[key].n or key in ("vpip", "bluff", "fold_post@mw", "fold_post@hu"):
                mu, s = self.population(key, exclude=seat)
                out.append(f"  {key}: {self.stat(seat, key)} (population {mu:.3f}, strength {s:.1f})")
        return out


# --- reading a finished hand's public record --------------------------------------------


def multiway_public_replay(view: MultiPublicState) -> list[tuple[MultiHandState, Any]]:
    """``(state before, action record)`` for every action, rebuilt from public information only."""
    state = new_hand(view.starting_stacks, view.button, view.small_blind, view.big_blind)
    out = []
    for rec in view.history:
        while state.cards_needed:
            n = len(state.board)
            state = deal_board(state, view.board[n : n + state.cards_needed])
        out.append((state, rec))
        state = apply_action(state, rec.player, rec.action)
    return out


def _group(street: Street) -> str:
    return "pre" if street == Street.PREFLOP else "post"


def multiway_opponent_events(view: MultiPublicState) -> dict[int, list[tuple[str, bool]]]:
    """Per opponent seat, the ``(key, yes)`` betting events of one finished hand.

    The same definitions as :func:`nutsfinder.bots.bayes_bot.opponent_events`
    (``vpip`` chance = staying in cost chips preflop; ``agg_*`` chance = a
    bet/raise was legal; ``fold_*`` chance = facing a bet/raise whose last
    aggressor on this street is *us*). Context: ``agg_*`` from the number of
    live players at the opponent's decision (what :meth:`MultiBayesBot.opponent_width`
    reads for each of its bets); ``fold_*`` from the number live **when we
    made the bet** they are answering, because that is all the bot knows
    when it later reads the statistic to size a bet (players acting in
    between may fold, but the bot cannot know that in advance).
    """
    me = view.seat
    events: dict[int, list[tuple[str, bool]]] = {s: [] for s in range(view.n_players) if s != me}
    faced: dict[int, bool] = dict.fromkeys(events, False)
    voluntary: dict[int, bool] = dict.fromkeys(events, False)
    last_aggressor: dict[Street, int] = {}
    our_size = 0.0
    our_ctx = "mw"  # context of our last bet/raise: the context the bot reads when it bets
    for state, rec in multiway_public_replay(view):
        legal = legal_actions(state)
        ctx = context_of(state.n_live)
        if rec.player == me:
            if isinstance(rec.action, (Bet, Raise)):
                to = rec.action.amount if isinstance(rec.action, Bet) else rec.action.to
                our_size = bet_fraction(state.pot, legal.call_amount, state.current_bet, to)
                our_ctx = ctx
        else:
            j, g = rec.player, _group(rec.street)
            if rec.street == Street.PREFLOP:
                faced[j] |= legal.can_call
                voluntary[j] |= isinstance(rec.action, (Call, Raise, Bet))
            if legal.can_bet or legal.can_raise:
                events[j].append((stat_key(f"agg_{g}", ctx), isinstance(rec.action, (Bet, Raise))))
            if legal.can_call and last_aggressor.get(rec.street) == me:
                events[j].append((stat_key(fold_stat_name(rec.street, our_size), our_ctx),
                                  isinstance(rec.action, Fold)))
        if isinstance(rec.action, (Bet, Raise)):
            last_aggressor[rec.street] = rec.player
    for j in events:
        if faced[j]:
            events[j].insert(0, ("vpip", voluntary[j]))
    return events


# --- the bot ------------------------------------------------------------------------------


class MultiBayesBot(MultiEquityBot):
    """MultiEquityBot with per-seat, context-split, partially pooled opponent models.

    Args:
        seed: seeds the numpy generator (Monte Carlo, showdown bluff checks).
        priors: base Beta priors (default :data:`~nutsfinder.bots.bayes_bot.DEFAULT_PRIORS`).
        learn: update after each hand; ``False`` plays exactly like MultiEquityBot.
        bluff_samples: deals per postflop showdown bluff check.
        min_strength, max_strength, min_seats, min_n: the pooling fit (module doc, section 3).
        range_widths, bet_fractions, consider_all_in, all_in_max_spr, samples, time_budget, debug:
            as in :class:`MultiEquityBot`.
    """

    mode = "bayes"

    def __init__(self, seed: int | None = None, priors: dict[str, tuple[float, float]] | None = None,
                 learn: bool = True, bluff_samples: int = 200, min_strength: float = 2.0,
                 max_strength: float = 100.0, min_seats: int = 2, min_n: int = 5,
                 range_widths: tuple[float, ...] = (1.0, 0.5, 0.3, 0.15),
                 bet_fractions: tuple[float, ...] = (0.5, 1.0), consider_all_in: bool = True,
                 all_in_max_spr: float | None = 3.0, samples: int = 600, time_budget: float | None = None, debug: bool = False) -> None:
        self._init_kwargs: dict[str, Any] = dict(
            seed=seed, priors=None if priors is None else dict(priors), learn=learn,
            bluff_samples=bluff_samples, min_strength=min_strength, max_strength=max_strength,
            min_seats=min_seats, min_n=min_n, range_widths=range_widths, bet_fractions=bet_fractions,
            consider_all_in=consider_all_in, all_in_max_spr=all_in_max_spr, samples=samples,
            time_budget=time_budget, debug=debug,
        )  # fmt: skip
        self.models = PooledModels(dict(DEFAULT_PRIORS) if priors is None else dict(priors),
                                   min_strength, max_strength, min_seats, min_n)
        super().__init__(seed=seed, range_widths=range_widths, bet_fractions=bet_fractions,
                         consider_all_in=consider_all_in, all_in_max_spr=all_in_max_spr, max_fold=self.models.base_mean("fold_post"),
                         samples=samples, time_budget=time_budget, debug=debug)
        self.learn = learn
        self.bluff_samples = bluff_samples

    # --- memory ----------------------------------------------------------------------

    def reseed(self, seed: int) -> None:
        """Restart the RNG only; the opponent models survive (they are the point)."""
        super().reseed(seed)

    def reset_model(self) -> None:
        self.models.reset()

    def fresh_copy(self) -> MultiBayesBot:
        return type(self)(**self._init_kwargs)

    def end_hand(self, summary: MultiHandSummary) -> None:
        """Add the hand's events per opponent seat (public record; shown cards of showdown players only)."""
        if not self.learn:
            return
        view = summary.final_view
        for seat, events in multiway_opponent_events(view).items():
            shown = summary.shown_cards[seat]
            if shown is not None:
                events = events + [("bluff", e < 0.5) for e in self._bet_equities(view, seat, shown)]
            self.models.observe(seat, events)

    def _bet_equities(self, view: MultiPublicState, seat: int, cards: tuple[Card, ...]) -> list[float]:
        """The shown hand's equity vs one random hand on each street ``seat`` bet or raised (HU proxy)."""
        streets = sorted({r.street for r in view.history if r.player == seat and isinstance(r.action, (Bet, Raise))})
        out = []
        for street in streets:
            if street == Street.PREFLOP:
                out.append(PREFLOP_EQUITY[hand_class(cards)])
            else:
                board = view.board[: street.board_size]
                out.append(multiway_equity(cards, board, 1, self.bluff_samples, rng=self._rng).equity)
        return out

    # --- the model's effect on decisions -------------------------------------------------

    def _keep(self, k: float, m: float) -> float:
        return k + m * (1.0 - k)

    def opponent_width(self, view: MultiPublicState, seat: int) -> float:
        """BayesBot's ``w = base × Π_i [k_i + m (1 - k_i)]`` for ``seat``, each step in its own context."""
        md = self.models
        m = md.randomness(seat)
        w = 1.0
        if any(r.player == seat and r.street == Street.PREFLOP and isinstance(r.action, (Call, Raise, Bet))
               for r in view.history):
            vpip = md.stat(seat, "vpip")
            w *= self._keep(min(1.0, vpip.mean / md.base_mean("vpip")), m)
        W = self.range_widths
        k = 0
        n_live = view.n_players  # everyone is dealt in; only folds remove players
        for rec in view.history:
            if rec.player == seat and isinstance(rec.action, (Bet, Raise)):
                k += 1
                step = W[min(k, len(W) - 1)] / W[min(k - 1, len(W) - 1)]
                key = stat_key(f"agg_{_group(rec.street)}", context_of(n_live))
                agg = md.stat(seat, key)
                w *= self._keep(step ** (math.log(agg.mean) / math.log(md.base_mean(key))), m)
            if isinstance(rec.action, Fold):
                n_live -= 1
        return snap_to_grid(w * (1 + 1e-9))

    def _trusted(self, stat: BetaStat, p0: float, m: float) -> float:
        return stat.mean if stat.mean <= p0 else p0 + m * (stat.mean - p0)

    def _fold_key(self, view: MultiPublicState, to: int) -> str:
        legal = view.legal
        assert legal is not None
        frac = bet_fraction(view.pot, legal.call_amount, view.current_bet, to)
        return stat_key(fold_stat_name(view.street, frac), context_of(1 + len(live_opponents(view))))

    def fold_and_calling(self, view: MultiPublicState, seat: int, width: float, to: int,
                         added: int) -> tuple[float, float]:
        """BayesBot's (c) and (d) for ``seat``, with its own model and the current context."""
        md = self.models
        m = md.randomness(seat)
        key = self._fold_key(view, to)
        p0 = md.base_mean(key)
        p = self._trusted(md.stat(seat, key), p0, m)
        rho = p / p0
        mdf = calling_share(view.pot, added)
        f = min(p, (1 - mdf) * rho)
        u = 1.0 - (1.0 - max(0.0, 1.0 - rho)) * (1.0 - m)
        ctx = context_of(1 + len(live_opponents(view)))
        raises = md.stat(seat, stat_key(f"agg_{_group(view.street)}", ctx)).mean
        q = mdf + (1 - mdf) * u * (1.0 - raises)
        return f, snap_to_grid(width * q)

    def opponent_note(self, view: MultiPublicState, seat: int, width: float) -> str:
        return (f"seat {seat} ({view.positions[seat]}): top {width:.1%} after {aggression_count(view, seat)} "
                f"bet(s), m={self.models.randomness(seat):.2f}, {self.models.hands.get(seat, 0)} hands seen")

    def model_notes(self, view: MultiPublicState) -> list[str]:
        out: list[str] = []
        for s in live_opponents(view):
            out += self.models.lines(s)
        return out

    def __repr__(self) -> str:
        return f"MultiBayesBot(learn={self.learn}, seats_seen={len(self.models.counts)}, samples={self.samples})"
