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
a fixed rule and reported in the debug log as the decision's ``mode``:

- ``passive-ahead``: a real bet was read as *passive* (a check or call),
  so the abstract street (or hand) is over while the real game still asks
  us to respond to that bet. Rule: **call** (check if possible). That is
  what translation.py prescribes: in the abstract history the opponent
  put in nothing, and the bet is small (below the smallest abstract
  size) so the call is cheap. Our real call is not an abstract action.
  The same rule applies when the abstract node is a showdown / all-in
  terminal while the real hand goes on (e.g. a real bet of 95% of the
  stack read as all-in and called; the real stacks still have chips): in
  the abstraction we are committed to showdown, so we check / call down.
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

Randomness: one ``numpy`` PCG64 generator, seeded in the constructor and by
:meth:`reseed`, used for the translations and the action draws (one
uniform each), in the order the decisions happen. Same seed, same model,
same opponent -> identical hands.

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
from nutsfinder.bots.common import aggressive_action, cards_text, check_turn, label
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import ActionRecord, PublicState, Street
from nutsfinder.engine.bot import Bot, HandSummary
from nutsfinder.engine.cards import Card
from nutsfinder.engine.preflop import equity_vs_top, hand_class

__all__ = ["CFRBot", "CFRDecision", "preload_turn_tables"]

log = logging.getLogger(__name__)


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
    translations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class CFRBot(Bot):
    """Plays an MCCFR blueprint (see the module doc).

    Args:
        model: a checkpoint path (``.npz`` written by ``examples/train_holdem.py``),
            a :class:`~nutsfinder.abstraction.blueprint.Blueprint`, or ``None``
            for the *untrained* (uniform) blueprint: what ``CFRBot()`` with no
            arguments plays, so match code can build it without a file. Use
            ``functools.partial(CFRBot, model=path)`` as a match factory.
        seed: seeds the bot's generator (translations and action draws).
        debug: record each decision in :attr:`last_decision` and log it.
        time_budget: seconds per decision. A decision is a table lookup
            plus at most one exact E[HS] table (~10-50 ms for a new turn
            board, cached afterwards), so there is nothing to cut short; the
            budget is only checked and an overrun is logged as a warning.
        below: translation rule below the smallest size (see translation.py).
    """

    def __init__(self, model: str | Path | Blueprint | None = None, seed: int | None = None, *,
                 debug: bool = False, time_budget: float | None = None,
                 below: tr.BelowRule = "pseudo_harmonic") -> None:
        if isinstance(model, Blueprint):
            self.blueprint = model
        elif model is None:
            self.blueprint = _uniform()
        else:
            self.blueprint = Blueprint.load(model)
        self.tree: BettingTree = self.blueprint.tree
        self.cards = _cards_for(self.blueprint.config)
        self.debug = debug
        self.time_budget = time_budget
        self.below = below
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._walk: _Walk | None = None
        self.last_decision: CFRDecision | None = None

    def reseed(self, seed: int) -> None:
        """Restart the generator from ``seed`` (see :meth:`Bot.reseed`)."""
        self._rng = np.random.Generator(np.random.PCG64(seed))

    def end_hand(self, summary: HandSummary) -> None:
        """Forget the hand's walk and translations (nothing is learned across hands)."""
        self._walk = None

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
        w.translations.append(f"#{i} to {to}: {res} -> {_edge_text(self.tree.actions(n)[j], labels[j])}")
        return j

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
        key = (state.button, state.starting_stacks, state.blinds, state.seat)
        w = self._walk
        if w is None or w.key != key or state.history[:w.done] != w.records:
            w = _Walk(key)  # a new hand (end_hand normally resets; this also catches bots used without it)
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
            if self.debug:
                strategy = tuple(zip(edge_names, (float(p) for p in probs)))
                reached = self.blueprint.was_reached(n, bucket)
                r = self.blueprint.regret(n, bucket)
                regrets = None if r is None else tuple(zip(edge_names, (float(x) for x in r)))
        else:
            mode = "terminal" if (w.over or t.player[w.node] == TERMINAL) else "passive-ahead"
            action = Check() if legal.can_check else Call()
            chosen = "fallback check/call"
        elapsed = time.perf_counter() - start
        if self.time_budget is not None and elapsed > self.time_budget:
            log.warning("CFRBot decision took %.3f s (budget %.3f s)", elapsed, self.time_budget)
        if self.debug:
            self.last_decision = self._decision(state, hole_cards, w, mode, bucket, strategy, regrets,
                                                reached, chosen, action, elapsed, notes)
            log.debug("%s", self.last_decision)
        return action

    def _decision(self, state: PublicState, hole_cards: tuple[Card, ...], w: _Walk, mode: str,
                  bucket: int | None, strategy, regrets, reached: bool, chosen: str, action: Action,
                  elapsed: float, notes: list[str]) -> CFRDecision:
        legal = state.legal
        assert legal is not None
        if state.street == Street.PREFLOP:
            ehs = equity_vs_top(hand_class(hole_cards), 1.0)[0]
        else:
            ehs = self.cards.ehs(card_ids(hole_cards), card_ids(state.board))
        call = legal.call_amount
        path = " ".join(_edge_text(a, ()) if not isinstance(a, (Bet, Raise)) else label(a)
                        for a in self.tree.path(w.node))
        return CFRDecision(
            street=state.street.name.lower(), hole_cards=cards_text(hole_cards), board=cards_text(state.board),
            mode=mode, node=w.node, path=path, bucket=bucket, ehs=ehs, pot=state.pot, to_call=call,
            pot_odds=call / (state.pot + call) if legal.can_call else None,
            translations=tuple(w.translations), strategy=strategy, regrets=regrets, reached=reached,
            chosen=chosen, action=action, seconds=elapsed, notes=tuple(w.notes + notes))


@lru_cache(maxsize=1)
def _uniform() -> Blueprint:
    return Blueprint.uniform()
