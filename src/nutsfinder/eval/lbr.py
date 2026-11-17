"""Local Best Response (LBR): a lower bound on how exploitable a heads-up NLHE bot is.

Reference: Lisý & Bowling (2017), *Equilibrium Approximation Quality of Current
No-Limit Poker Bots* (arXiv 1612.07547); see ``docs/phase6_research.md`` section 4.

What LBR does
=============
LBR is a player that knows the opponent's *strategy* (what it would do with any
holding in any public spot) but not its cards or its dice. It

1. keeps a **belief** ``w(h)`` over the opponent's 1,326 holdings (minus those
   that touch LBR's cards or the board), starting uniform, and after every
   opponent action ``a`` applies Bayes' rule::

       w(h) <- w(h) * P(a | h, public history)        (then renormalise)

   using the opponent's own action probabilities (the *policy query*, below);
2. at each of its own decisions evaluates every candidate action with the
   standard LBR assumption that **after this action the hand is checked /
   called down** (no more betting), and plays the one with the highest
   expected value (ties: the more passive action):

       EV(fold)  = 0
       EV(check) = e * P
       EV(call)  = e * (P_eff + c) - c
       EV(bet x) = f * P + (1 - f) * (e_call * (P + x + r) - x)

   with (notation of :mod:`nutsfinder.bots.common`) ``e`` the equity of LBR's
   hand against the belief (exact, all run-outs, on the flop, turn and river:
   :func:`~nutsfinder.eval.exact_equity.hero_vs_each`; preflop from a
   precomputed table, below), ``P_eff`` the effective pot (an all-in for less
   only wins the matched part), ``f = sum_h w(h) P(fold | h, after our bet)``
   the belief-weighted fold probability of the opponent's *real* response to
   that exact bet, and ``e_call`` the equity against the belief re-weighted by
   ``1 - P(fold | h)`` (a raise in response is treated as a call: LBR does not
   look further ahead). Chips already in the pot are sunk, so folding is
   worth 0.
3. **Preflop** (``preflop=``, one of :data:`PREFLOP_POLICIES`):

   - ``"ev"`` (default): exactly the rule of point 2 (fold, check/call, and
     the same bet/raise sizes), with the belief so far (uniform, or updated by
     the opponent's preflop actions) and the opponent's real response to each
     raise from the same policy queries. A re-raise is treated as a call, as
     postflop.
   - ``"call"``: always check / call, never fold, never raise, as in Lisý &
     Bowling's LBR.
   - ``"fold_call"``: facing a bet, call iff ``EV(call) > 0`` (check-down
     equity against the belief); never raise.

   The *preflop check-down caveat*: the one-step EV preflop assumes the hand
   is checked down over all five board cards, ignoring that both players
   (LBR included) will act again on the flop, turn and river. So the rule can
   pick a preflop action that is not the best one given LBR's later play
   (e.g. fold a hand that would have been profitable to play on), and it
   ignores the postflop value of position. This can only make LBR a *weaker*
   strategy, i.e. loosen the bound; it can never invalidate it (below).

   Preflop equities: :func:`~nutsfinder.eval.exact_equity.preflop_vs_each`, a
   precomputed (169 hero classes x 1,326 opponent combos) table, 100,000 random
   boards per class, symmetrised over suit permutations, mapped to any hero by
   an exact suit relabelling (``models/preflop_vs_combo.npz``, built on first
   use). Its per-entry SE is about 0.16 points; tested within that against
   :func:`~nutsfinder.engine.equity.equity_vs_range` with 400,000 samples. The
   alternative, :func:`hero_vs_each` with 2,000 boards per hand, was off by up
   to 1.7 points against a range (the error is correlated across opponent
   combos because every combo shares the boards).

Why it is a lower bound. LBR is *some* strategy for the real game, using only
public information and its own cards. Played in both seats on the same deals
(duplicate), the game is symmetric and its value is 0, so any strategy's
average winnings against the opponent are at most the opponent's
exploitability (what a full best response would win). LBR's belief and its
greedy one-step EVs only affect how *tight* the bound is, never its validity,
as long as LBR does not cheat: it never sees the opponent's cards, and the
policy queries never use the opponent's private random state. This holds for
*every* choice of LBR's rules, the preflop ones included: a poor preflop rule
(always call, or a check-down EV that ignores later streets) makes LBR win
less and the bound looser, never larger than the true exploitability.

Candidate actions (``bet_fractions``, ``all_in``)
-------------------------------------------------
Fold (only when facing a bet), check/call, and bets/raises of 1/2 pot and pot
(pot after calling, :func:`~nutsfinder.bots.common.raise_to_for_fraction`) plus
all-in (the effective all-in). Lisý & Bowling found fold/call/pot/all-in the
strongest small set; we add half pot because CFRBot's postflop abstraction has
only pot and all-in, so half pot is an *off-tree* size that probes the
pseudo-harmonic translation (the "translation leak" this measurement is meant
to catch). Sizes are configurable.

The policy-query protocol
=========================
``oracle.range_probs(view, combos) -> (actions, probs)``: for the opponent's
view ``view`` (it is the opponent's turn) and holdings ``combos`` (indices into
:data:`~nutsfinder.abstraction.strength.COMBO_CARDS`), the distinct real actions
and a ``(len(combos), len(actions))`` matrix of probabilities. Contract:

- **No side effects on the real game.** A query must not change the real bot's
  state (memory, walks, caches) or consume its random number generator, so the
  hand plays out exactly as it would without LBR asking.
- **No leak.** A query uses only the public view, the hypothetical holding and
  the opponent's strategy, never its real cards or private random state.

``only=action`` lets an oracle return just the column the caller needs (LBR
asks for one action at a time). ``action_probs(bot, view, hole_cards)`` is the
one-holding convenience form.
:func:`make_oracle` picks the implementation per bot type:

- :class:`CFROracle` (CFRBot): **exact**. ``CFRBot.range_policy`` replays the
  public history on copies of the abstract walk, enumerating the hidden
  branches (pseudo-harmonic translations of LBR's bets, marginalised with their
  probabilities; CFRBot's own edge choices, weighted per holding by the
  blueprint) and applying the deterministic equity-rule overrides exactly as
  ``act`` does. Verified against sampled play in ``tests/test_lbr.py``.
- :class:`EquityFamilyOracle` (EquityBot, BayesBot, PotOddsBot): their ``act``
  is deterministic given their equity estimates, which are Monte Carlo with
  the bot's RNG. The query uses a **snapshot** (a deep copy taken at the start
  of each hand, so BayesBot's learned model is the one it plays this hand
  with; the model only changes in ``end_hand``, which the real instance still
  receives) whose equity method is replaced by **enumerated** equities
  (:func:`~nutsfinder.eval.exact_equity.holdings_vs_ranges`: exact on the turn
  and river; on the flop ``flop_runouts`` evenly spaced run-outs of the 1,176,
  default 200, RMS error ~1.5-2 points, chosen for speed; ``None`` = exact) and
  whose RNG is replaced by an object that raises if used. So the query is the
  bot's policy with (almost) noise-free equities: a point mass per holding.
  Postflop, EquityBot/BayesBot's decision is evaluated for all holdings at
  once (``_vectorised``: the same EV formulas on arrays, the snapshot's own
  model methods; tested equal to ``act`` per holding); preflop and for
  PotOddsBot, ``act`` runs once per distinct equity (or hand class). The real
  bot (400 samples, SE up to ~2.5 points) can disagree near a decision
  boundary; so that one disagreement cannot eliminate the true holding, these
  oracles declare ``likelihood_floor`` (default 0.02): Bayes uses
  ``max(P, floor)``.
- :class:`RandomBotOracle` (RandomBot): exact and card-independent.
- :class:`SnapshotOracle` (anything else, e.g. CallStationBot, AlwaysRaiseBot):
  ``act`` on a deep copy of a start-of-hand snapshot, reseeded from a hash of
  the public view and the holding (never the real RNG); a point mass per
  holding (a *sample* of the bot's mixed strategy for random bots).

Caching. Results are cached per (opponent view, holding) for the hand: the
opponent view after LBR's actual action is usually one LBR already queried as a
candidate, so the Bayes update after the opponent answers is free. The
equity-family oracle also caches exact equity arrays per (board, range) and
the ``act`` result per (view, equity tuple): the bots' decisions depend on the
holding only through those equities (preflop: through the hand class).

Duplicate deals and sessions
============================
:func:`lbr_session` plays ``n_pairs`` duplicate pairs: for pair ``i`` the same
deal (and button seat) is played twice, LBR in seat 0 then seat 1, each room
with its own LBR and its own opponent instance (built fresh per session, so a
learning opponent learns against LBR for the whole session, as in
:mod:`nutsfinder.eval.sessions`). Seeds are hashed from (master seed, session),
so the deals depend only on those and are **identical for every opponent**:
results against different opponents are paired deal by deal.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from nutsfinder.abstraction.strength import COMBO_CARDS, N_COMBOS, card_ids, combo_index
from nutsfinder.bots.common import (
    Decision,
    aggressive_action,
    cards_text,
    check_turn,
    chips_added,
    chips_to_call_raise,
    effective_max_to,
    effective_pot,
    ev_bet,
    ev_call,
    ev_check,
    label,
    pot_odds,
    raise_to_for_fraction,
)
from nutsfinder.bots.equity_bot import EquityBot
from nutsfinder.bots.pot_odds import PotOddsBot
from nutsfinder.bots.random_bot import RandomBot
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import (
    BIG_BLIND,
    HandState,
    PublicState,
    Street,
    apply_action,
    deal_board,
    new_hand,
    public_view,
)
from nutsfinder.engine.bot import Bot, HandSummary
from nutsfinder.engine.cards import Card
from nutsfinder.engine.preflop import hand_class, top_range
from nutsfinder.engine.runner import play_hand
from nutsfinder.eval.exact_equity import (CARD_COMBOS, combo_weights, hero_vs_each, holdings_vs_ranges,
                                          preflop_vs_each)
from nutsfinder.eval.match import describe_bot, unit_seeds

__all__ = [
    "PolicyOracle",
    "CFROracle",
    "EquityFamilyOracle",
    "SnapshotOracle",
    "RandomBotOracle",
    "make_oracle",
    "action_probs",
    "rebuild_state",
    "LBRBot",
    "PREFLOP_POLICIES",
    "LBRHand",
    "LBRSession",
    "lbr_session",
    "lbr_session_seed",
]

log = logging.getLogger(__name__)

PREFLOP_POLICIES = ("ev", "call", "fold_call")
"""LBR's preflop rules (module doc, point 3); the first is the default."""

_COMBO_CARDS_OBJ: list[tuple[Card, Card]] = [(Card.from_id(int(a)), Card.from_id(int(b)))
                                            for a, b in COMBO_CARDS.tolist()]


def _view_key(view: PublicState) -> tuple:
    """Everything public that a decision can depend on (the cache key of a query)."""
    return (view.seat, view.button, view.starting_stacks, view.small_blind, view.big_blind,
            view.history, view.board)


# --- Public state reconstruction ------------------------------------------------------------


def rebuild_state(view: PublicState, n_records: int | None = None) -> HandState:
    """The public :class:`HandState` (no hole cards) after the first ``n_records`` history records.

    Replays the public record through the engine (like
    :func:`~nutsfinder.abstraction.translation.view_before_action`), dealing the
    public board cards whenever a street opens. With ``n_records = None`` it is the
    current state, from which LBR applies hypothetical actions.
    """
    history = view.history if n_records is None else view.history[:n_records]
    state = new_hand(button=view.button, stacks=view.starting_stacks,
                     small_blind=view.small_blind, big_blind=view.big_blind)

    def deal(s: HandState) -> HandState:
        while s.cards_needed and len(s.board) + s.cards_needed <= len(view.board):
            n = len(s.board)
            s = deal_board(s, view.board[n:n + s.cards_needed])
        return s

    for rec in history:
        state = apply_action(deal(state), rec.player, rec.action)
    return deal(state)


# --- Policy oracles ---------------------------------------------------------------------------


class PolicyOracle(ABC):
    """Side-effect-free "what would you do with these cards here?" for one opponent instance.

    Attributes:
        likelihood_floor: Bayes updates use ``max(P(a | h), likelihood_floor)``
            (0 for exact oracles; positive for approximate models, so one
            disagreement with the real bot cannot eliminate the true holding).
        exact: True if the probabilities are the bot's real behaviour exactly.
    """

    likelihood_floor: float = 0.0
    exact: bool = False
    cacheable: bool = True

    def begin_hand(self) -> None:
        """Called once per hand before any query (snapshot state that may change between hands)."""

    @abstractmethod
    def range_probs(self, view: PublicState, combos: np.ndarray, only: Action | None = None
                    ) -> tuple[list[Action], np.ndarray]:
        """``(actions, probs)`` for each holding in ``combos`` at the opponent's ``view``.

        ``only``: the caller needs just this action's column; an oracle may then return
        only it (useful when the action space is huge, e.g. RandomBot's sizes).
        """

    def describe(self) -> str:
        return type(self).__name__


class CFROracle(PolicyOracle):
    """Exact: :meth:`nutsfinder.bots.cfr_bot.CFRBot.range_policy` (reads only blueprint, tree, card tables)."""

    exact = True

    def __init__(self, bot: Bot) -> None:
        from nutsfinder.bots.cfr_bot import CFRBot
        if not isinstance(bot, CFRBot):
            raise TypeError(f"{bot!r} is not a CFRBot")
        self._bot = bot

    def range_probs(self, view: PublicState, combos: np.ndarray, only: Action | None = None
                    ) -> tuple[list[Action], np.ndarray]:
        return self._bot.range_policy(view, combos)

    def describe(self) -> str:
        return "CFROracle(exact: blueprint + enumerated translations / own-edge branches)"


class _NoRandom:
    """Replaces a snapshot's RNG: any use is a bug in the oracle (it would make queries noisy)."""

    def __getattr__(self, name: str):
        raise RuntimeError(f"policy query tried to use the bot's RNG ({name})")

    def __deepcopy__(self, memo):
        return self


_SHARED_EQUITIES: OrderedDict[tuple, np.ndarray] = OrderedDict()
"""Enumerated equity arrays per (board, range width, flop run-outs), shared by every oracle of the
process (pure functions of public data; LRU-bounded). The two rooms of a duplicate pair have the
same board, so the second room's queries are mostly free."""
_SHARED_EQUITIES_MAX = 512


class EquityFamilyOracle(PolicyOracle):
    """EquityBot / BayesBot / PotOddsBot with exact equities on a start-of-hand snapshot (module doc)."""

    def __init__(self, bot: Bot, likelihood_floor: float = 0.02, flop_runouts: int | None = 200,
                 vectorised: bool = True) -> None:
        if not isinstance(bot, (EquityBot, PotOddsBot)):
            raise TypeError(f"{bot!r} is not an EquityBot/BayesBot/PotOddsBot")
        self._bot = bot
        self.vectorised = vectorised
        self.likelihood_floor = likelihood_floor
        self.flop_runouts = flop_runouts
        self._snap: Bot | None = None
        self._eq_cache = _SHARED_EQUITIES  # per process: both rooms of a duplicate pair share the board
        self._widths: tuple = ()
        self.equity_seconds = 0.0

    def describe(self) -> str:
        runouts = "all" if self.flop_runouts is None else self.flop_runouts
        return (f"EquityFamilyOracle(exact equities, flop run-outs {runouts}, "
                f"likelihood floor {self.likelihood_floor})")

    def begin_hand(self) -> None:
        snap = copy.deepcopy(self._bot)
        snap.debug = False
        snap.last_decision = None
        snap.time_budget = None
        snap._rng = _NoRandom()
        if isinstance(snap, EquityBot):
            snap.range_equities = self._range_equities  # instance attribute shadows the method
        else:
            snap._cache = {}
            snap.estimate_equity = self._estimate_equity
        self._snap = snap

    # The replacement equity methods (called by the snapshot's act).
    def _key(self, ids: tuple[int, ...], w: float | None) -> tuple:
        return (ids, w, self.flop_runouts if len(ids) == 3 else None)

    def _equity_arrays(self, board: tuple[Card, ...], widths: Sequence[float]) -> list[np.ndarray]:
        ids = tuple(card_ids(board))
        missing = [w for w in widths if self._key(ids, w) not in self._eq_cache]
        if missing:
            t0 = time.perf_counter()
            weights = np.stack([np.ones(N_COMBOS) if w is None else combo_weights(top_range(w)) for w in missing])
            runouts = self.flop_runouts if len(ids) == 3 else None
            eq = holdings_vs_ranges(ids, weights, max_runouts=runouts)
            for w, row in zip(missing, eq):
                row.setflags(write=False)
                self._eq_cache[self._key(ids, w)] = row
            self.equity_seconds += time.perf_counter() - t0
        # Collect (and mark as recently used) before evicting, so an older entry that is part of
        # this request cannot be dropped between the check above and the return.
        out = []
        for w in widths:
            k = self._key(ids, w)
            self._eq_cache.move_to_end(k)
            out.append(self._eq_cache[k])
        while len(self._eq_cache) > _SHARED_EQUITIES_MAX:
            self._eq_cache.popitem(last=False)
        return out

    def _range_equities(self, state: PublicState, hole_cards: tuple[Card, ...], widths: list[float]
                        ) -> dict[float, tuple[float, float]]:
        if state.street == Street.PREFLOP:
            return EquityBot.range_equities(self._snap, state, hole_cards, widths)  # table lookup, no RNG
        self._widths = tuple(widths)
        ids = card_ids(hole_cards)
        h = combo_index(ids[0], ids[1])
        return {w: (float(a[h]), 0.0) for w, a in zip(widths, self._equity_arrays(state.board, widths))}

    def _estimate_equity(self, state: PublicState, hole_cards: tuple[Card, ...]) -> tuple[float, float]:
        if state.street == Street.PREFLOP:
            return PotOddsBot.estimate_equity(self._snap, state, hole_cards)  # table lookup, no RNG
        self._widths = (None,)
        ids = card_ids(hole_cards)
        return float(self._equity_arrays(state.board, [None])[0][combo_index(ids[0], ids[1])]), 0.0

    def range_probs(self, view: PublicState, combos: np.ndarray, only: Action | None = None
                    ) -> tuple[list[Action], np.ndarray]:
        if self._snap is None:
            self.begin_hand()
        if self.vectorised and isinstance(self._snap, EquityBot) and view.street != Street.PREFLOP:
            return self._vectorised(view, np.asarray(combos, dtype=np.int64))
        return self.per_holding(view, combos)

    def _vectorised(self, view: PublicState, combos: np.ndarray) -> tuple[list[Action], np.ndarray]:
        """:meth:`EquityBot.act` for all holdings at once (postflop).

        Same steps and the same float operations as ``act`` (the EV helpers of
        :mod:`nutsfinder.bots.common` applied to arrays), with the snapshot's own
        state-only model methods (``villain_range_width``, ``candidate_sizes``,
        ``fold_and_calling``: BayesBot's learned versions for a BayesBot), so the
        choice per holding is identical to ``act`` with the exact equities
        (``np.argmax`` keeps the first maximum, as ``max`` does there). Checked
        against :meth:`per_holding` in ``tests/test_lbr.py``.
        """
        snap = self._snap
        assert isinstance(snap, EquityBot)
        legal = check_turn(view)
        pot, call = view.pot, legal.call_amount
        pot_eff = effective_pot(view)
        width = snap.villain_range_width(view)
        bets = []
        for to in snap.candidate_sizes(view):
            x, r = chips_added(view, to), chips_to_call_raise(view, to)
            f, calling = snap.fold_and_calling(view, width, to, x)
            bets.append((aggressive_action(legal, to), x, r, f, calling))
        needed = [width] + sorted({b[4] for b in bets} - {width}, reverse=True)
        arrays = dict(zip(needed, self._equity_arrays(view.board, needed)))
        equity = arrays[width][combos]
        actions: list[Action] = []
        evs: list[np.ndarray] = []
        if legal.can_call:
            actions += [Fold(), Call()]
            evs += [np.zeros(len(combos)), ev_call(equity, pot_eff, call)]
        else:
            actions.append(Check())
            evs.append(ev_check(equity, pot))
        for action, x, r, f, calling in bets:
            actions.append(action)
            evs.append(ev_bet(pot, x, r, f, arrays[calling][combos]))
        choice = np.argmax(np.stack(evs, axis=1), axis=1)
        probs = np.zeros((len(combos), len(actions)))
        probs[np.arange(len(combos)), choice] = 1.0
        used = sorted(set(choice.tolist()))
        return [actions[j] for j in used], probs[:, used]

    def per_holding(self, view: PublicState, combos: np.ndarray) -> tuple[list[Action], np.ndarray]:
        """One ``act`` call on the snapshot per distinct equity tuple (preflop: per hand class)."""
        if self._snap is None:
            self.begin_hand()
        snap = self._snap
        assert snap is not None
        chosen: list[Action] = []
        memo: dict[tuple, Action] = {}
        preflop = view.street == Street.PREFLOP
        for i, c in enumerate(np.asarray(combos).tolist()):
            cards = _COMBO_CARDS_OBJ[c]
            if preflop:
                key: tuple = (hand_class(cards),)
            elif self._widths:
                arrays = [self._eq_cache.get(self._key(tuple(card_ids(view.board)), w)) for w in self._widths]
                key = tuple(float(a[c]) for a in arrays) if all(a is not None for a in arrays) else ("_", c)
            else:
                key = ("_", c)
            a = memo.get(key)
            if a is None:
                self._widths = ()
                a = snap.act(view, cards)
                if not preflop and self._widths:
                    arrays = [self._eq_cache[self._key(tuple(card_ids(view.board)), w)] for w in self._widths]
                    memo[tuple(float(x[c]) for x in arrays)] = a
                elif preflop:
                    memo[key] = a
            chosen.append(a)
        return _point_masses(chosen)


class SnapshotOracle(PolicyOracle):
    """Any bot: ``act`` on a fresh deep copy of a start-of-hand snapshot, reseeded from public data.

    A point mass per holding. For a bot with randomness that is one *sample* of
    its mixed strategy (seeded by a hash of the view and the holding, never by the
    real bot's generator), hence the default ``likelihood_floor``. Bots in
    ``STATELESS`` are asked directly on the snapshot (they keep no state).
    """

    STATELESS: tuple[str, ...] = ("CallStationBot", "AlwaysRaiseBot")

    def __init__(self, bot: Bot, likelihood_floor: float = 0.02) -> None:
        self._bot = bot
        self.likelihood_floor = likelihood_floor
        self._snap: Bot | None = None

    def begin_hand(self) -> None:
        snap = copy.deepcopy(self._bot)
        if hasattr(snap, "debug"):
            snap.debug = False
        if hasattr(snap, "last_decision"):
            snap.last_decision = None
        self._snap = snap

    def range_probs(self, view: PublicState, combos: np.ndarray, only: Action | None = None
                    ) -> tuple[list[Action], np.ndarray]:
        if self._snap is None:
            self.begin_hand()
        stateless = type(self._bot).__name__ in self.STATELESS
        chosen: list[Action] = []
        base = hashlib.sha256(repr(_view_key(view)).encode()).digest()
        for i, c in enumerate(np.asarray(combos).tolist()):
            if stateless:
                bot = self._snap
            else:
                bot = copy.deepcopy(self._snap)
                bot.reseed(int.from_bytes(hashlib.sha256(base + c.to_bytes(2, "big")).digest()[:8], "big"))
            chosen.append(bot.act(view, _COMBO_CARDS_OBJ[c]))
        return _point_masses(chosen)


class RandomBotOracle(PolicyOracle):
    """Exact for :class:`~nutsfinder.bots.random_bot.RandomBot`: uniform over the legal action kinds,
    then uniform over the legal sizes; the same for every holding (so its actions carry no information)."""

    exact = True
    cacheable = False

    def range_probs(self, view: PublicState, combos: np.ndarray, only: Action | None = None
                    ) -> tuple[list[Action], np.ndarray]:
        legal = check_turn(view)
        n_kinds = 3 if (legal.can_bet or legal.can_raise) else 2
        n_sizes = legal.max_to - legal.min_to + 1 if n_kinds == 3 else 0
        if only is not None:
            p = 0.0
            if legal.is_legal(only):
                p = 1.0 / n_kinds / (n_sizes if isinstance(only, (Bet, Raise)) else 1)
            return [only], np.full((len(combos), 1), p)
        kinds: list[Action] = [Fold(), Check() if legal.can_check else Call()]
        probs = [1.0 / n_kinds] * 2
        if n_kinds == 3:
            kinds += [Bet(x) if legal.can_bet else Raise(x) for x in range(legal.min_to, legal.max_to + 1)]
            probs += [1.0 / n_kinds / n_sizes] * n_sizes
        return kinds, np.tile(np.array(probs), (len(combos), 1))


def _point_masses(chosen: list[Action]) -> tuple[list[Action], np.ndarray]:
    actions = list(dict.fromkeys(chosen))
    index = {a: j for j, a in enumerate(actions)}
    probs = np.zeros((len(chosen), len(actions)))
    probs[np.arange(len(chosen)), [index[a] for a in chosen]] = 1.0
    return actions, probs


def make_oracle(bot: Bot, **options) -> PolicyOracle:
    """The policy oracle for ``bot`` (see the module doc); ``options`` go to its constructor."""
    from nutsfinder.bots.cfr_bot import CFRBot
    if isinstance(bot, CFRBot):
        return CFROracle(bot)
    if isinstance(bot, (EquityBot, PotOddsBot)):
        return EquityFamilyOracle(bot, **options)
    if type(bot) is RandomBot:
        return RandomBotOracle()
    return SnapshotOracle(bot, **options)


def action_probs(bot: Bot, view: PublicState, hole_cards: tuple[Card, ...]) -> dict[Action, float]:
    """``{action: probability}`` of ``bot`` at ``view`` holding ``hole_cards``, without side effects.

    One-shot form of the protocol: builds an oracle (snapshotting the bot now).
    """
    oracle = make_oracle(bot)
    oracle.begin_hand()
    ids = card_ids(hole_cards)
    actions, probs = oracle.range_probs(view, np.array([combo_index(ids[0], ids[1])]))
    return {a: float(p) for a, p in zip(actions, probs[0]) if p > 0}


class _CachedOracle:
    """Per-hand cache of an oracle's answers, per (view, holding)."""

    def __init__(self, oracle: PolicyOracle) -> None:
        self.oracle = oracle
        self._cache: dict[tuple, dict[int, dict[Action, float]]] = {}
        self.queries = 0          # holdings actually computed
        self.hits = 0             # holdings answered from the cache
        self.seconds = 0.0

    def reset(self) -> None:
        self._cache.clear()

    def probs(self, view: PublicState, combos: np.ndarray, action: Action | None = None
              ) -> dict[Action, np.ndarray]:
        """``{action: P per holding}`` (aligned with ``combos``); only ``action`` if given."""
        if not self.oracle.cacheable:
            t0 = time.perf_counter()
            acts, mat = self.oracle.range_probs(view, np.asarray(combos, dtype=np.int64), only=action)
            self.seconds += time.perf_counter() - t0
            self.queries += len(combos)
            cols = {a: mat[:, j] for j, a in enumerate(acts)}
            wanted = [action] if action is not None else acts
            return {a: cols.get(a, np.zeros(len(combos))) for a in wanted}
        key = _view_key(view)
        known = self._cache.setdefault(key, {})
        clist = np.asarray(combos).tolist()
        missing = np.array([c for c in clist if c not in known], dtype=np.int64)
        self.hits += len(clist) - len(missing)
        if len(missing):
            t0 = time.perf_counter()
            acts, mat = self.oracle.range_probs(view, missing)
            self.seconds += time.perf_counter() - t0
            self.queries += len(missing)
            for c, row in zip(missing.tolist(), mat):
                known[c] = {a: float(p) for a, p in zip(acts, row) if p > 0}
        wanted = {action} if action is not None else {a for c in clist for a in known[c]}
        return {a: np.array([known[c].get(a, 0.0) for c in clist]) for a in wanted}


# --- The LBR player ---------------------------------------------------------------------------


@dataclass
class _HandMemory:
    key: tuple
    hole: tuple[Card, ...]
    belief: np.ndarray
    done: int = 0                                   # history records already used for Bayes
    records: tuple = ()                             # ... and the records themselves
    beliefs: list[np.ndarray] = field(default_factory=list)  # snapshots at our postflop decisions
    equity: dict[tuple, np.ndarray] = field(default_factory=dict)


class LBRBot(Bot):
    """Local Best Response against one opponent instance (see the module doc).

    Args:
        oracle: the opponent's :class:`PolicyOracle` (built with :func:`make_oracle`
            on the opponent instance this LBR plays against).
        bet_fractions: bet/raise sizes considered, as pot fractions (postflop, and
            preflop with ``preflop="ev"``).
        all_in: also consider the effective all-in.
        preflop: ``"ev"`` (default: the same one-step check-down EV rule as postflop),
            ``"call"`` (Lisý & Bowling: always check/call) or ``"fold_call"`` (call
            iff EV(call) > 0, never raise). See the module doc.
        likelihood_floor: overrides the oracle's floor if given.
        prune: holdings whose belief falls below ``prune`` times the largest are
            dropped (never queried again). 0 keeps everything.
        debug: record each decision in :attr:`last_decision` (equity, pot odds,
            EV of every action, fold probabilities, belief size) and log it.
        time_budget: seconds per decision; the computation is not cut short
            (that would change the bound), an overrun is only logged.
        seed: unused (LBR is deterministic); accepted for the bot conventions.
    """

    def __init__(self, oracle: PolicyOracle, bet_fractions: tuple[float, ...] = (0.5, 1.0), all_in: bool = True,
                 preflop: str = "ev", likelihood_floor: float | None = None, prune: float = 1e-9,
                 debug: bool = False, time_budget: float | None = None, seed: int | None = 0) -> None:
        if preflop not in PREFLOP_POLICIES:
            raise ValueError(f"preflop policy must be one of {PREFLOP_POLICIES}, got {preflop!r}")
        self.preflop = preflop
        self.oracle = oracle
        self.cached = _CachedOracle(oracle)
        self.bet_fractions = tuple(bet_fractions)
        self.all_in = all_in
        self.floor = oracle.likelihood_floor if likelihood_floor is None else likelihood_floor
        self.prune = prune
        self.debug = debug
        self.time_budget = time_budget
        self.seed = seed
        self.last_decision: Decision | None = None
        self._mem: _HandMemory | None = None
        self._last_beliefs: list[np.ndarray] = []
        self._need_begin = True
        self.stats = {"decisions": 0, "seconds": 0.0, "overruns": 0}

    def __repr__(self) -> str:
        return (f"LBRBot(bets={self.bet_fractions}+{'all-in' if self.all_in else ''}, preflop={self.preflop}, "
                f"floor={self.floor}, oracle={self.oracle.describe()})")

    def end_hand(self, summary: HandSummary) -> None:
        self._need_begin = True
        self._last_beliefs = self._mem.beliefs if self._mem is not None else []
        self._mem = None

    # --- belief ---------------------------------------------------------------------

    def _start(self, view: PublicState, hole: tuple[Card, ...]) -> _HandMemory:
        key = (view.button, view.starting_stacks, view.blinds)
        m = self._mem
        fresh = (m is None or self._need_begin or m.key != key or m.hole != tuple(hole)
                 or m.done > len(view.history) or view.history[:m.done] != m.records)
        if fresh:
            if self._need_begin or m is None:
                self.oracle.begin_hand()
                self.cached.reset()
                self._need_begin = False
            belief = np.ones(N_COMBOS)
            belief[CARD_COMBOS[card_ids(hole)].ravel()] = 0.0
            m = _HandMemory(key, tuple(hole), belief)
            self._mem = m
        assert m is not None
        return m

    def _live(self, m: _HandMemory, board: tuple[Card, ...]) -> np.ndarray:
        if board:
            m.belief[CARD_COMBOS[card_ids(board)].ravel()] = 0.0
        top = m.belief.max()
        if top <= 0:
            return np.zeros(0, dtype=np.int64)
        if self.prune > 0:
            m.belief[m.belief < self.prune * top] = 0.0
        return np.flatnonzero(m.belief > 0)

    def _update(self, m: _HandMemory, view: PublicState) -> None:
        """Bayes over the opponent's actions since the last update."""
        for i in range(m.done, len(view.history)):
            rec = view.history[i]
            if rec.player == view.seat:
                continue
            state = rebuild_state(view, i)
            opp_view = public_view(state, rec.player)
            live = self._live(m, opp_view.board)
            if len(live) == 0:
                continue
            p = self.cached.probs(opp_view, live, rec.action)[rec.action]
            new = m.belief[live] * np.maximum(p, self.floor)
            if new.sum() <= 0:
                # Every holding contradicts the model (only possible with floor 0 and an inexact
                # oracle): keep the old belief rather than lose all information.
                log.warning("LBR: opponent action %s has probability 0 for every holding", rec.action)
                continue
            m.belief[live] = new / new.sum()
        m.done = len(view.history)
        m.records = view.history

    # --- acting ---------------------------------------------------------------------

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        start = time.perf_counter()
        legal = check_turn(state)
        m = self._start(state, hole_cards)
        self._update(m, state)
        if state.street == Street.PREFLOP and self.preflop == "ev":
            action = self._check_down(state, hole_cards, m, legal)
        elif state.street == Street.PREFLOP and self.preflop == "fold_call" and legal.can_call:
            action = self._preflop_fold_call(state, hole_cards, m, legal)
        elif state.street == Street.PREFLOP:
            action = Call() if legal.can_call else Check()
            if self.debug:
                self.last_decision = Decision(state.street.name.lower(), cards_text(hole_cards), "", 1.0,
                                              math.nan, 0.0, state.pot, legal.call_amount,
                                              pot_odds(legal.call_amount, effective_pot(state)) if legal.can_call
                                              else None, (), action, ("preflop: LBR always checks/calls",))
        else:
            action = self._check_down(state, hole_cards, m, legal)
        elapsed = time.perf_counter() - start
        self.stats["decisions"] += 1
        self.stats["seconds"] += elapsed
        if self.time_budget is not None and elapsed > self.time_budget:
            self.stats["overruns"] += 1
            log.warning("LBR decision took %.2f s (budget %.2f s)", elapsed, self.time_budget)
        return action

    def _preflop_fold_call(self, state: PublicState, hole: tuple[Card, ...], m: _HandMemory, legal) -> Action:
        """``preflop="fold_call"``: facing a bet preflop, call iff EV(call) under check-down > 0 (never raise)."""
        live = self._live(m, state.board)
        w = m.belief[live] / m.belief[live].sum()
        e = float(w @ self._equity(m, state.board)[live])
        pot_eff = effective_pot(state)
        ev = ev_call(e, pot_eff, legal.call_amount)
        action = Call() if ev > 0 else Fold()
        if self.debug:
            self.last_decision = Decision(state.street.name.lower(), cards_text(hole), "", len(live) / N_COMBOS, e,
                                          0.0, state.pot, legal.call_amount, pot_odds(legal.call_amount, pot_eff),
                                          (("fold", 0.0), ("call", ev)), action, ("preflop: fold/call by EV",))
        return action

    def _equity(self, m: _HandMemory, board: tuple[Card, ...]) -> np.ndarray:
        key = tuple(card_ids(board))
        e = m.equity.get(key)
        if e is None:
            e = hero_vs_each(card_ids(m.hole), key) if key else preflop_vs_each(card_ids(m.hole))
            m.equity[key] = e
        return e

    def candidate_sizes(self, state: PublicState) -> list[int]:
        legal = state.legal
        assert legal is not None
        if not (legal.can_bet or legal.can_raise):
            return []
        sizes = {raise_to_for_fraction(state, f) for f in self.bet_fractions}
        if self.all_in:
            sizes.add(effective_max_to(state))
        return sorted(sizes)

    def _check_down(self, state: PublicState, hole: tuple[Card, ...], m: _HandMemory, legal) -> Action:
        """The one-step check-down EV rule (every postflop decision; preflop with ``preflop="ev"``)."""
        live = self._live(m, state.board)
        if len(live) == 0:  # cannot happen with a floor; be safe: play passively
            return Check() if legal.can_check else Call()
        w = m.belief[live] / m.belief[live].sum()
        if state.board:  # truth diagnostics are recorded for postflop decisions only
            m.beliefs.append(m.belief.copy())
        eq = self._equity(m, state.board)[live]
        e = float(w @ eq)
        pot, call = state.pot, legal.call_amount
        pot_eff = effective_pot(state)
        options: list[tuple[Action, float]] = []
        notes: list[str] = []
        if legal.can_call:
            options += [(Fold(), 0.0), (Call(), ev_call(e, pot_eff, call))]
        else:
            options.append((Check(), ev_check(e, pot)))
        if self.candidate_sizes(state):
            current = rebuild_state(state)
        for to in self.candidate_sizes(state):
            action = aggressive_action(legal, to)
            after = apply_action(current, state.seat, action)
            x, r = chips_added(state, to), chips_to_call_raise(state, to)
            if after.to_act is None:  # nobody can answer (cannot happen heads-up after a bet); count as called
                f_h = np.zeros(len(live))
            else:
                f_h = self.cached.probs(public_view(after, after.to_act), live, Fold())[Fold()]
            f = float(w @ f_h)
            wc = w * (1.0 - f_h)
            e_call = float(wc @ eq / wc.sum()) if wc.sum() > 0 else 0.0
            options.append((action, ev_bet(pot, x, r, f, e_call)))
            notes.append(f"{label(action)}: P(fold) {f:.1%}, equity when called {e_call:.1%}")
        best, _ = max(options, key=lambda o: o[1])  # first max: the more passive action wins ties
        if self.debug:
            ess = 1.0 / float(w @ w)
            notes.append(f"belief: {len(live)} live holdings, effective {ess:.0f}")
            self.last_decision = Decision(
                street=state.street.name.lower(), hole_cards=cards_text(hole), board=cards_text(state.board),
                villain_range=len(live) / N_COMBOS, equity=e, std_error=0.0, pot=pot, to_call=call,
                pot_odds=pot_odds(call, pot_eff) if legal.can_call else None,
                evs=tuple((label(a), v) for a, v in options), action=best, notes=tuple(notes))
            log.debug("%s", self.last_decision)
        return best

    def truth_diagnostics(self, opponent_cards: tuple[Card, ...]) -> list[float]:
        """After a hand (for evaluation only, never used for play): the belief's mass on the
        opponent's real holding at each of LBR's postflop decisions of the last hand."""
        ids = card_ids(opponent_cards)
        k = combo_index(ids[0], ids[1])
        return [float(b[k] / b.sum()) if b.sum() > 0 else 0.0 for b in self._last_beliefs]


# --- Duplicate sessions ----------------------------------------------------------------------


def lbr_session_seed(master_seed: int, session_index: int) -> int:
    """64-bit seed of LBR session ``session_index`` (SHA-256 of purpose, master seed, index)."""
    if session_index < 0:
        raise ValueError("session index must be >= 0")
    key = f"nutsfinder/lbr_session|{master_seed}|{session_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class LBRHand:
    """One hand: LBR's net (bb), its seat, the button seat, seconds, and diagnostics."""

    pair: int
    room: int
    lbr_seat: int
    button: int
    net_bb: float
    seconds: float
    deal_seed: int
    truth_mass: tuple[float, ...] = ()
    showdown: bool = False


@dataclass(frozen=True, slots=True)
class LBRSession:
    """One session of duplicate pairs: per-hand results, per-pair sums, costs."""

    index: int
    seed: int
    opponent: str
    opponent_desc: str
    lbr_desc: str
    hands: tuple[LBRHand, ...]
    seconds: float
    queries: int
    cache_hits: int
    query_seconds: float

    @property
    def pair_results(self) -> tuple[float, ...]:
        n = max(h.pair for h in self.hands) + 1
        out = [0.0] * n
        for h in self.hands:
            out[h.pair] += h.net_bb
        return tuple(out)

    @property
    def total_bb(self) -> float:
        return math.fsum(h.net_bb for h in self.hands)

    @property
    def mbb_per_game(self) -> float:
        return self.total_bb / len(self.hands) * 1000


def lbr_session(make_opponent: Callable[[], Bot], n_pairs: int, master_seed: int, session_index: int,
                lbr_options: dict | None = None, oracle_options: dict | None = None) -> LBRSession:
    """Play LBR session ``session_index``: ``n_pairs`` duplicate pairs, fresh instances per room.

    Pair i: deal ``deal_seed_i`` with the button on seat ``i % 2``, played as
    ``[LBR, opponent]`` in room 1 and ``[opponent, LBR]`` in room 2. The opponent is
    reseeded before every hand with ``seed_b_i`` (the same in both rooms, as in
    :func:`~nutsfinder.eval.match.duplicate_match`). Module-level and picklable
    (factories must be picklable) so sessions can run in a process pool.
    """
    if n_pairs < 1:
        raise ValueError("need at least one pair")
    lbr_options = dict(lbr_options or {})
    oracle_options = dict(oracle_options or {})
    seed = lbr_session_seed(master_seed, session_index)
    rooms = []
    for _ in range(2):
        opp = make_opponent()
        lbr = LBRBot(make_oracle(opp, **oracle_options), **lbr_options)
        rooms.append((lbr, opp))
    opp_desc = describe_bot(rooms[0][1])
    lbr_desc = repr(rooms[0][0])
    hands: list[LBRHand] = []
    t_session = time.perf_counter()
    for i, (deal_seed, _seed_a, seed_b) in enumerate(unit_seeds(seed, n_pairs)):
        button = i % 2
        for room, (lbr, opp) in enumerate(rooms):
            opp.reseed(seed_b)
            lbr_seat = room  # room 0: LBR in seat 0; room 1: LBR in seat 1
            bots = [lbr, opp] if lbr_seat == 0 else [opp, lbr]
            t0 = time.perf_counter()
            hand_log = play_hand(bots, deal_seed, button=button)
            dt = time.perf_counter() - t0
            opp_cards = hand_log.hole_cards[1 - lbr_seat]
            hands.append(LBRHand(
                pair=i, room=room, lbr_seat=lbr_seat, button=button,
                net_bb=hand_log.result.net[lbr_seat] / BIG_BLIND, seconds=dt, deal_seed=deal_seed,
                truth_mass=tuple(lbr.truth_diagnostics(opp_cards)), showdown=hand_log.folded is None))
    lbrs = [r[0] for r in rooms]
    return LBRSession(
        index=session_index, seed=seed, opponent=type(rooms[0][1]).__name__, opponent_desc=opp_desc,
        lbr_desc=lbr_desc, hands=tuple(hands), seconds=time.perf_counter() - t_session,
        queries=sum(b.cached.queries for b in lbrs), cache_hits=sum(b.cached.hits for b in lbrs),
        query_seconds=sum(b.cached.seconds for b in lbrs))
