"""CFRBot (tier 4): play the MCCFR blueprint of the abstract game in the real game.

Each decision has three steps.

1. **Where are we in the abstract tree?** Walk the real public history
   (``state.history``) through :class:`~nutsfinder.abstraction.tree.BettingTree`
   from the root. Seats are mapped by position: the button is tree seat 0.
   Fold, check and call follow the matching edge. A real bet or raise is
   *translated* (:func:`~nutsfinder.abstraction.translation.translate_bet`,
   pseudo-harmonic) onto the bet edges **of the node reached**: the
   ``sizes`` are the size labels of that node's own edges
   (:meth:`BettingTree.edge_labels`, converted to pot fractions at the
   real spot by :func:`~nutsfinder.abstraction.translation.label_fraction`),
   so a size the abstraction does not offer there (cap reached, merged
   with all-in) is never a candidate. The translation is randomised, so
   every choice is cached for the rest of the hand: the walk is
   incremental (only new history records are processed at each
   decision) and nothing is redrawn when the bot is asked again.
   Our own bets map to the edge we chose (also cached), not to a
   re-translation of our real chips.
2. **Which bucket?** :class:`~nutsfinder.abstraction.cards.CardAbstraction`
   on our own hole cards and the current board (the 169 classes preflop,
   the exact E[HS] bucket postflop): only cards the bot was handed.
3. **Which action?** Sample from the normalised average strategy at
   ``(node, bucket)`` with the bot's own seeded generator, then make it
   real: fold / check / call map directly, a bet edge is sized from the
   **real** pot (:func:`~nutsfinder.abstraction.translation.real_action_for_fraction`,
   or :func:`~nutsfinder.abstraction.translation.to_for_bb` for the
   preflop open, the effective all-in for ``all_in``), clamped to the
   legal range.

Off-tree situations and their fallbacks (it must never crash or act illegally)
-----------------------------------------------------------------------------
The real game and the abstract walk can disagree. Each case is resolved by
a fixed rule and reported in the debug log as the decision's ``mode``
(``blueprint`` when the blueprint decides, else ``passive-ahead`` or
``terminal``).

**Real bets read as passive.** Postflop the abstraction only has pot and
all-in, so the pseudo-harmonic mapping reads a bet of ``x`` pot (``x < 1``)
as a check/call with probability ``P = (1 - x) / (1 + x)`` (1/3 for half
pot, 1/2 for a third of pot). In that branch the abstract history says the
opponent put in nothing, while the real game makes us answer ``x``. Three
rules handle it.

- *Remap when passive would close the abstract street.* If the passive
  edge leads out of the street (a street transition or a terminal: e.g.
  we checked, they bet half pot, "check" would make it check-check), the
  bet is read instead as the **smallest non-all-in bet edge** of that node
  (deterministic, cached like every translation), and the node reached is
  marked *soft*. The blueprint then answers with fold / call / raise and
  the walk stays on the tree. At a soft node a sampled **fold is
  overridden by the equity rule** below (call if the hand is not
  dominated). Without that override the remap would be the
  ``below="smallest"`` rule, which a min-bettor exploits: a bet of 1% pot
  would be answered with the pot-bet defence (folding maybe half the
  range), winning the pot about half the time for a 1% risk. With it, a
  bet of ``x`` pot is folded only by hands whose equity is at most
  ``x / (1 + 2x)``, i.e. almost never for tiny bets.
- *Abstract check facing a real bet* (the passive edge did not close the
  street, e.g. the big blind leads small and we are at our own node after
  "check"): when the blueprint samples check but the real game has a bet
  to call, the equity rule decides between call and fold.
- *No bet edge to remap onto, abstract street already closed, or the
  abstract hand is over* (``passive-ahead`` / ``terminal``; e.g. a real bet
  of 95% of the stack read as all-in and called, real chips behind): check
  if checking is free, else the equity rule. We never bet in this mode.

**The equity rule.** Facing a real call of ``c`` into a pot ``P`` (the pot
already includes the bet), the call is worth it against an opponent
holding a random hand iff our equity ``e > c / (P + c)``. We use ``e`` =
E[HS] against a random hand (exact, the same number the buckets use;
preflop the 169-class equity vs a random hand), call iff ``e > c/(P + c)``
and fold otherwise. Why *vs random* and why it is the right size of change:

1. *It only removes dominated calls.* A betting range is at least as strong
   as a random hand (bets correlate with strength), so our equity against
   it is at most ``e``: if ``e <= c/(P+c)`` the call loses even against a
   random holding. (Implied odds of later streets are ignored; the hands
   concerned are near the bottom of the range, where they are small.)
2. *It cannot be farmed with small bets.* For a bet of ``x`` pot,
   ``c/(P+c) = x / (1 + 2x)``. On the river E[HS] *is* the percentile of
   our hand among random hands, so against a range spread like a random
   one we fold a fraction ``x/(1+2x)`` of it, below ``1 - MDF = x/(1+x)``
   (the most a defender may fold before pure bluffs of ``x`` profit). A 1%
   pot min-bet folds out ~1% of hands (the ones that beat nothing).
3. *Its cost.* The old rule (call 100% in the passive branch) gave exactly
   the bluffing-game defence ``1/(1+x)`` in aggregate (that is how the
   pseudo-harmonic ``P`` is derived, see translation.py), but called with
   the wrong hands. Folding dominated hands in that branch raises the
   aggregate fold rate by at most ``P * x/(1+2x)``, which is at most about
   0.10 of the range (near ``x = 0.3``) if our range is spread like a
   random one. That is a small concession to a perfectly polarised bluffer
   in exchange for not paying off value bets with air, which is what the
   bots in this repo mostly bet.

The rule draws no random numbers (so the generator stream is unchanged)
and depends only on our cards and public chips. A stronger reference range
than "random" (e.g. a value-weighted betting range) would fold more and
risk over-folding; it is left for a later tier.

**Other cases.**

- **Abstract street still open, real street closed** (e.g. our abstract
  raise became a real call because the real rules forbid raising after a
  short all-in): the walk closes the abstract street with passive
  (check / call) actions for whoever is to act there, then continues.
  The same "passive filler" is used if the abstract node belongs to the
  opponent when the real game asks us (a note in the debug log).
- **No bet edge to translate onto** (abstract raise cap or abstract
  all-in reached while the real game allows a raise): the real raise is
  read as a call.
- **A real fold where the abstraction has no fold edge** (folding when
  a check is free; RandomBot does it): the hand is over anyway.
- **Our sampled edge is impossible in the real spot**: a bet edge when
  the real rules allow no bet/raise becomes call / check; a fold edge
  when we could check becomes a check (folding for free is never
  better).
- **Unequal or non-100bb real stacks / other blinds**: the bot still
  plays the 100bb blueprint. Sizes are pot fractions and all-in is the
  real *effective* all-in, so every action stays legal and sensible in
  pot terms, but the strategy was not solved for that depth (it is
  increasingly wrong for short stacks). Documented, not fixed.
- **Untrained info sets** (zero strategy mass): played uniformly, the
  solver's convention; flagged ``reached=False`` in the debug log.

Seats and self-play: the walk is kept **per real seat** (a dict keyed by
seat), so one instance playing both seats (self-play with a single
object) keeps a stable walk and stable translations for each seat.

Randomness: one ``numpy`` PCG64 generator, seeded in the constructor
(default seed 0, so ``play_match([CFRBot(model), ...])`` is reproducible
without match-level reseeding; pass ``seed=None`` explicitly for OS
entropy) and by :meth:`reseed`, used for the translations and the action
draws (one uniform each), in the order the decisions happen. Same seed,
same model, same opponent -> identical hands.

Trained or not: ``CFRBot()`` (``model=None``) plays the untrained uniform
blueprint and warns (:class:`UntrainedBlueprintWarning`). An evaluation
script should check :attr:`CFRBot.is_trained` or call
:func:`require_trained` on the bot (or a bot built from its factory).

Debug (``debug=True``): every decision is stored in :attr:`last_decision`
(a :class:`CFRDecision`) and logged at DEBUG level: node and abstract path,
bucket and E[HS] (our equity against a random hand), pot odds, each
translation of the hand, the strategy, the average regret per action if
the model has regrets, the sampled edge and the real action. A blueprint
has no per-action EV: the closest thing it stores is the average
counterfactual regret, logged as such and not as an EV.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

import nutsfinder.abstraction.translation as tr
from nutsfinder.abstraction.blueprint import Blueprint
from nutsfinder.abstraction.cards import CardAbstraction
from nutsfinder.abstraction.config import AbstractionConfig
from nutsfinder.abstraction.strength import card_ids
from nutsfinder.abstraction.tree import CHANCE, TERMINAL, BettingTree, SizeLabel
from nutsfinder.bots.common import aggressive_action, cards_text, check_turn, effective_pot, label, pot_odds
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import ActionRecord, PublicState, Street
from nutsfinder.engine.bot import Bot, HandSummary
from nutsfinder.engine.cards import Card
from nutsfinder.engine.preflop import equity_vs_top, hand_class

__all__ = ["CFRBot", "CFRDecision", "UntrainedBlueprintWarning", "preload_turn_tables", "require_trained"]

log = logging.getLogger(__name__)


class UntrainedBlueprintWarning(UserWarning):
    """``CFRBot`` was built without a model and plays the untrained uniform blueprint."""


def require_trained(bot: Bot, min_iterations: int = 1) -> None:
    """Raise ``ValueError`` unless ``bot`` is a CFRBot with a trained blueprint (for eval scripts).

    Trained means: not ``Blueprint.uniform()`` and at least ``min_iterations``
    training iterations recorded in the checkpoint meta.
    """
    if not isinstance(bot, CFRBot):
        raise ValueError(f"{bot!r} is not a CFRBot")
    if not bot.is_trained or bot.blueprint.iteration < min_iterations:
        raise ValueError(f"{bot!r} is not trained (iteration {bot.blueprint.iteration:,}, "
                         f"need >= {min_iterations:,})")


@lru_cache(maxsize=4)
def _cards_for(config: AbstractionConfig) -> CardAbstraction:
    """One card abstraction per config, shared by all bots (its caches are pure functions of the board)."""
    return CardAbstraction(config)


def preload_turn_tables(path: str | Path, config: AbstractionConfig | None = None) -> None:
    """Load every exact turn E[HS] table into the card abstraction all CFRBots of ``config`` share.

    Optional: without it a new turn board costs one table (~10-50 ms) the first time it is seen.
    The file is written by ``examples/make_bucket_edges.py --save-turns`` (or the trainer).
    """
    from nutsfinder.abstraction.config import DEFAULT_CONFIG
    _cards_for(config or DEFAULT_CONFIG).tables.load_turns(Path(path))


def _edge_text(action: Action | None, labels: tuple[SizeLabel, ...]) -> str:
    if action is None:
        return "deal"
    name = type(action).__name__.lower()
    return f"{name}[{'/'.join(str(x) for x in labels)}]" if labels else name


@dataclass(frozen=True, slots=True)
class CFRDecision:
    """Why CFRBot did what it did, for one decision (recorded when ``debug=True``)."""

    street: str
    hole_cards: str
    board: str
    mode: str
    node: int
    path: str
    bucket: int | None
    ehs: float | None
    pot: int
    to_call: int
    pot_odds: float | None
    translations: tuple[str, ...]
    strategy: tuple[tuple[str, float], ...]
    regrets: tuple[tuple[str, float], ...] | None
    reached: bool
    chosen: str
    action: Action
    seconds: float
    notes: tuple[str, ...] = ()

    def __str__(self) -> str:
        odds = "-" if self.pot_odds is None else f"{self.pot_odds:.1%}"
        ehs = "-" if self.ehs is None else f"{self.ehs:.1%}"
        lines = [
            f"{self.street} {self.hole_cards} board [{self.board}] pot {self.pot} to call {self.to_call} "
            f"[{self.mode}]",
            f"  node {self.node}: {self.path or '(root)'}",
            f"  bucket {self.bucket}, E[HS] vs random hand {ehs}, pot odds {odds}",
            *(f"  translated: {t}" for t in self.translations),
            *(f"  P({name}) = {p:.3f}" for name, p in self.strategy),
            *(f"  avg regret({name}) = {r:+.2f} chips (cf. regret, not an EV)" for name, r in (self.regrets or ())),
            *(["  info set never reached in training: uniform"] if self.strategy and not self.reached else []),
            *(f"  {n}" for n in self.notes),
            f"  -> {self.chosen} = {self.action} ({self.seconds * 1000:.1f} ms)",
        ]
        return "\n".join(lines)


@dataclass
class _Walk:
    """The abstract walk of the current hand (see the module doc)."""

    key: tuple
    node: int = 0
    done: int = 0                       # history records processed
    records: tuple = ()                 # ... and the records themselves (new-hand detection)
    over: bool = False                  # a real fold the abstraction could not follow
    own: dict[int, tuple[int, int]] = field(default_factory=dict)  # record index -> (node, edge) we chose
    soft: set[int] = field(default_factory=set)  # nodes reached by a remapped passive read (module doc)
    translations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class CFRBot(Bot):
    """Plays an MCCFR blueprint (see the module doc).

    Args:
        model: a checkpoint path (``.npz`` written by ``examples/train_holdem.py``),
            a :class:`~nutsfinder.abstraction.blueprint.Blueprint`, or ``None``
            for the *untrained* (uniform) blueprint: what ``CFRBot()`` with no
            arguments plays, so match code can build it without a file (it
            warns with :class:`UntrainedBlueprintWarning`). Use
            ``functools.partial(CFRBot, model=path)`` as a match factory.
        seed: seeds the bot's generator (translations and action draws).
            Default 0 (reproducible); ``None`` means OS entropy.
        debug: record each decision in :attr:`last_decision` and log it.
        time_budget: seconds per decision. A decision is a table lookup
            plus at most one exact E[HS] table (~10-50 ms for a new turn
            board, cached afterwards), so there is nothing to cut short; the
            budget is only checked and an overrun is logged as a warning.
        below: translation rule below the smallest size (see translation.py).
    """

    def __init__(self, model: str | Path | Blueprint | None = None, seed: int | None = 0, *,
                 debug: bool = False, time_budget: float | None = None,
                 below: tr.BelowRule = "pseudo_harmonic") -> None:
        if isinstance(model, Blueprint):
            self.blueprint = model
        elif model is None:
            self.blueprint = _uniform()
            warnings.warn("CFRBot(model=None) plays the UNTRAINED uniform blueprint; pass a checkpoint "
                          "(e.g. models/holdem_official/latest.npz) for real play",
                          UntrainedBlueprintWarning, stacklevel=2)
        else:
            self.blueprint = Blueprint.load(model)
        self.tree: BettingTree = self.blueprint.tree
        self.cards = _cards_for(self.blueprint.config)
        self.debug = debug
        self.time_budget = time_budget
        self.below = below
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._walks: dict[int, _Walk] = {}  # one walk per real seat (self-play with one instance)
        self._walk: _Walk | None = None     # the walk of the seat of the current decision
        self.last_decision: CFRDecision | None = None

    def reseed(self, seed: int) -> None:
        """Restart the generator from ``seed`` (see :meth:`Bot.reseed`)."""
        self._rng = np.random.Generator(np.random.PCG64(seed))

    def end_hand(self, summary: HandSummary) -> None:
        """Forget the hand's walks and translations (nothing is learned across hands)."""
        self._walks.clear()
        self._walk = None

    @property
    def is_trained(self) -> bool:
        """False for the uniform blueprint and for tables with no recorded training iterations."""
        return self.blueprint.is_trained

    def __repr__(self) -> str:
        return f"CFRBot(model={self.blueprint.describe()})"

    # --- the walk ------------------------------------------------------------------

    def _tree_seat(self, state: PublicState, real_seat: int) -> int:
        return 0 if real_seat == state.button else 1

    def _passive_edge(self, node: int) -> int:
        """Index of the check/call edge (fold, if present, is first)."""
        acts = self.tree.actions(node)
        return 1 if isinstance(acts[0], Fold) else 0

    def _step(self, w: _Walk, j: int) -> None:
        w.node = self.tree.child_ids(w.node)[j]

    def _advance_to_street(self, w: _Walk, street: int) -> None:
        """Move the walk to ``street``: through deals, closing abstract streets passively if needed."""
        t = self.tree
        for _ in range(64):  # every step moves down the finite tree; the bound is a safety net
            n = w.node
            if t.player[n] == TERMINAL:
                return
            if t.street[n] >= street and t.player[n] != CHANCE:
                return
            if t.player[n] == CHANCE:
                if t.street[n] >= street:
                    return  # the abstract street closed, the real one has not
                self._step(w, 0)
                continue
            w.notes.append(f"abstract {Street(t.street[n]).name.lower()} still open: "
                           f"seat {t.player[n]} checks/calls to follow the real game")
            self._step(w, self._passive_edge(n))
        raise AssertionError("abstract walk did not terminate")  # pragma: no cover

    def _in_sync(self, w: _Walk, street: int) -> bool:
        t = self.tree
        n = w.node
        return not w.over and t.player[n] in (0, 1) and t.street[n] == street

    def _translate(self, w: _Walk, state: PublicState, i: int, to: int) -> int:
        """Edge index for a real bet/raise to ``to`` (history record i) at the walk's node."""
        n = w.node
        labels = self.tree.edge_labels(n)
        edge_of: list[int] = []
        sizes: list[float] = []
        pre = tr.view_before_action(state, i)
        for j, labs in enumerate(labels):
            for lab in labs:
                edge_of.append(j)
                sizes.append(tr.label_fraction(pre, lab))
        if not sizes:
            w.translations.append(f"#{i} real raise to {to}: no bet edge in the abstraction -> call")
            return self._passive_edge(n)
        res = tr.translate_bet(pre, to, sizes, self._rng, below=self.below)
        j = self._passive_edge(n) if res.passive else edge_of[res.index]
        text = f"#{i} to {to}: {res} -> {_edge_text(self.tree.actions(n)[j], labels[j])}"
        if res.passive and self._closes_street(n, j):
            # Passive would end the abstract street/hand while the real game makes the other seat
            # answer a bet: read it as the smallest non-all-in bet edge and mark the node soft
            # (a blueprint fold there is checked against the equity rule; module doc).
            bets = [k for k, labs in enumerate(labels) if labs and all(x.kind != "all_in" for x in labs)]
            if bets:
                j = bets[0]  # bet edges are in increasing size order
                w.soft.add(self.tree.child_ids(n)[j])
                text += f", which closes the abstract street: remapped to {_edge_text(self.tree.actions(n)[j], labels[j])}"
        w.translations.append(text)
        return j

    def _closes_street(self, n: int, j: int) -> bool:
        """Whether edge j of node n leaves the street (street transition or terminal)."""
        t = self.tree
        child = t.child_ids(n)[j]
        return not (t.player[child] in (0, 1) and t.street[child] == t.street[n])

    def _follow(self, w: _Walk, state: PublicState, i: int, rec: ActionRecord) -> None:
        """Advance the walk over real history record i."""
        t = self.tree
        self._advance_to_street(w, int(rec.street))
        for _ in range(8):
            if not self._in_sync(w, int(rec.street)):
                return  # abstract is ahead / terminal: a real-only action (e.g. our passive call)
            if t.player[w.node] == self._tree_seat(state, rec.player):
                break
            w.notes.append(f"#{i}: abstract node {w.node} belongs to the other seat: it checks/calls")
            self._step(w, self._passive_edge(w.node))
        else:  # pragma: no cover - the tree alternates seats, so two steps always suffice
            return
        n = w.node
        acts = t.actions(n)
        labels = t.edge_labels(n)
        match rec.action:
            case Fold():
                if isinstance(acts[0], Fold):
                    self._step(w, 0)
                else:
                    w.over = True
            case Check() | Call():
                self._step(w, self._passive_edge(n))
            case Bet(amount=to) | Raise(to=to):
                node, mine = w.own.get(i, (-1, -1))
                if node == n and labels[mine]:
                    self._step(w, mine)  # our own bet: the edge we chose, no re-translation
                else:
                    self._step(w, self._translate(w, state, i, to))

    def _sync(self, state: PublicState) -> _Walk:
        key = (state.button, state.starting_stacks, state.blinds)
        w = self._walks.get(state.seat)
        if w is None or w.key != key or state.history[:w.done] != w.records:
            w = _Walk(key)  # a new hand (end_hand normally resets; this also catches bots used without it)
        self._walks[state.seat] = w
        self._walk = w
        for i in range(w.done, len(state.history)):
            self._follow(w, state, i, state.history[i])
        w.done = len(state.history)
        w.records = state.history
        self._advance_to_street(w, int(state.street))
        return w

    # --- acting ----------------------------------------------------------------------

    def _real_action(self, state: PublicState, j: int) -> Action:
        legal = state.legal
        assert legal is not None
        edge = self.tree.actions(self._walk.node)[j]  # type: ignore[union-attr]
        labels = self.tree.edge_labels(self._walk.node)[j]  # type: ignore[union-attr]
        if isinstance(edge, Fold):
            return Fold() if legal.can_call else Check()
        if not labels:
            return Check() if legal.can_check else Call()
        if not (legal.can_bet or legal.can_raise):
            return Call() if legal.can_call else Check()
        kinds = {lab.kind: lab for lab in labels}
        if "all_in" in kinds:
            return tr.real_action_for_fraction(state, tr.ALL_IN)
        if "open_bb" in kinds:
            return aggressive_action(legal, tr.to_for_bb(state, kinds["open_bb"].value))
        return tr.real_action_for_fraction(state, labels[0].value)

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        start = time.perf_counter()
        legal = check_turn(state)
        w = self._sync(state)
        t = self.tree
        bucket = None
        strategy: tuple[tuple[str, float], ...] = ()
        regrets = None
        reached = True
        notes: list[str] = []
        # Our real seat must own the abstract node; if the opponent does, it checks/calls (module doc).
        for _ in range(4):
            if not self._in_sync(w, int(state.street)) or t.player[w.node] == self._tree_seat(state, state.seat):
                break
            w.notes.append(f"abstract node {w.node} belongs to the opponent: it checks/calls")
            self._step(w, self._passive_edge(w.node))
            self._advance_to_street(w, int(state.street))
        if self._in_sync(w, int(state.street)) and t.player[w.node] == self._tree_seat(state, state.seat):
            mode = "blueprint"
            n = w.node
            bucket = self.cards.bucket(card_ids(hole_cards), card_ids(state.board))
            probs = self.blueprint.strategy(n, bucket)
            u = self._rng.random()
            j = int(np.searchsorted(np.cumsum(probs), u, side="right"))
            j = min(j, len(probs) - 1)
            while probs[j] <= 0.0:  # rounding at the top end: take the last positive action
                j -= 1
            action = self._real_action(state, j)
            w.own[len(state.history)] = (n, j)
            edge_names = [_edge_text(a, labs) for a, labs in zip(t.actions(n), t.edge_labels(n))]
            chosen = edge_names[j]
            edge = t.actions(n)[j]
            if isinstance(edge, Fold) and n in w.soft and legal.can_call:
                action, why = self._equity_rule(state, hole_cards)
                notes.append(f"soft node (remapped passive read): blueprint fold -> equity rule: {why}")
            elif isinstance(edge, Check) and not legal.can_check:
                action, why = self._equity_rule(state, hole_cards)
                notes.append(f"abstract check but a real bet to call: equity rule: {why}")
            if self.debug:
                strategy = tuple(zip(edge_names, (float(p) for p in probs)))
                reached = self.blueprint.was_reached(n, bucket)
                r = self.blueprint.regret(n, bucket)
                regrets = None if r is None else tuple(zip(edge_names, (float(x) for x in r)))
        else:
            mode = "terminal" if (w.over or t.player[w.node] == TERMINAL) else "passive-ahead"
            if legal.can_check:
                action, chosen = Check(), "fallback check"
            else:
                action, why = self._equity_rule(state, hole_cards)
                chosen = f"fallback {type(action).__name__.lower()}"
                notes.append(f"equity rule: {why}")
        elapsed = time.perf_counter() - start
        if self.time_budget is not None and elapsed > self.time_budget:
            log.warning("CFRBot decision took %.3f s (budget %.3f s)", elapsed, self.time_budget)
        if self.debug:
            self.last_decision = self._decision(state, hole_cards, w, mode, bucket, strategy, regrets,
                                                reached, chosen, action, elapsed, notes)
            log.debug("%s", self.last_decision)
        return action

    def _equity(self, state: PublicState, hole_cards: tuple[Card, ...]) -> float:
        """Our equity against a random hand: E[HS] postflop, the class equity preflop (deterministic)."""
        if state.street == Street.PREFLOP:
            return equity_vs_top(hand_class(hole_cards), 1.0)[0]
        return self.cards.ehs(card_ids(hole_cards), card_ids(state.board))

    def _equity_rule(self, state: PublicState, hole_cards: tuple[Card, ...]) -> tuple[Action, str]:
        """Facing a real bet off the blueprint: call iff equity vs a random hand > c / (P_eff + c), else fold.

        ``P_eff`` is the effective pot (:func:`~nutsfinder.bots.common.effective_pot`): when our
        call is all-in for less than the bet, the opponent's unmatched excess is refunded, so it
        is not part of what we can win. Example: 500 behind facing a 3,000 bet into 200 gives
        ``P = 3,200`` but ``P_eff = 700``, so the call needs 500 / 1,200 = 41.7%, not 13.5%.
        """
        legal = state.legal
        assert legal is not None and legal.can_call
        c = legal.call_amount
        odds = pot_odds(c, effective_pot(state))
        e = self._equity(state, hole_cards)
        if e > odds:
            return Call(), f"E[HS] {e:.3f} > pot odds {odds:.3f}: call"
        return Fold(), f"E[HS] {e:.3f} <= pot odds {odds:.3f}: fold"

    def _decision(self, state: PublicState, hole_cards: tuple[Card, ...], w: _Walk, mode: str,
                  bucket: int | None, strategy, regrets, reached: bool, chosen: str, action: Action,
                  elapsed: float, notes: list[str]) -> CFRDecision:
        legal = state.legal
        assert legal is not None
        ehs = self._equity(state, hole_cards)
        call = legal.call_amount
        path = " ".join(_edge_text(a, ()) if not isinstance(a, (Bet, Raise)) else label(a)
                        for a in self.tree.path(w.node))
        return CFRDecision(
            street=state.street.name.lower(), hole_cards=cards_text(hole_cards), board=cards_text(state.board),
            mode=mode, node=w.node, path=path, bucket=bucket, ehs=ehs, pot=state.pot, to_call=call,
            pot_odds=pot_odds(call, effective_pot(state)) if legal.can_call else None,
            translations=tuple(w.translations), strategy=strategy, regrets=regrets, reached=reached,
            chosen=chosen, action=action, seconds=elapsed, notes=tuple(w.notes + notes))


@lru_cache(maxsize=1)
def _uniform() -> Blueprint:
    return Blueprint.uniform()
