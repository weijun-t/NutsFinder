"""The abstract betting tree, built once by driving the real engine.

Why build it with ``betting.py``
--------------------------------
Every abstract action must be legal under the real rules (minimum raises,
all-in clamping, who acts when, uncalled-bet refunds). Rather than
re-implement those rules, :func:`build_tree` starts from
:func:`~nutsfinder.engine.betting.new_hand`, offers the abstract actions of
:func:`abstract_actions` at each decision, and plays each with
:func:`~nutsfinder.engine.betting.apply_action`. Whatever the engine accepts
is in the tree, with the chips the engine computed. The tree is then frozen
into flat integer lists, so the MCCFR inner loop never builds a
``HandState``.

Seats: seat 0 is the **button** (small blind, acts first preflop, last
postflop) and seat 1 the big blind, for every node. A bot sitting in the
big blind plays player 1's strategy.

Node kinds
----------
- **decision** (``player`` 0 or 1): children are the abstract actions.
- **street transition** (``player == CHANCE``): betting closed with chips
  behind; the next street's cards come. One child: the first decision of
  the next street. (The cards themselves are dealt by the game adapter.)
- **terminal** (``player == TERMINAL``): ``FOLD`` (somebody folded) or
  ``SHOWDOWN`` (river betting closed, or all chips in before the river:
  the run-out has no decisions, so the all-in call is the leaf).

Per node we store the street, the chips each seat has put in
(``contrib``, after uncalled-bet refunds) and, for terminals, the payoff
basis: ``utility to seat 0 = fold_value`` for a fold, and
``= stake * (+1 win, 0 split, -1 loss)`` at a showdown, with ``stake`` the
contested chips per player. Both are zero-sum by construction: chips one
seat wins the other loses.

Worked example (default config). Root: SB to act, pot 150. Actions: fold,
call, raise to 300 (the 3bb open = pot raise), all-in (10,000). After
"fold" the SB loses its 50: fold_value = -50. After "raise to 300, call"
the flop comes with contributions (300, 300).

Info sets
---------
Each decision node n on street s gets ``buckets(s)`` info sets, numbered
``infoset_base[n] + bucket``: the betting history is the node, the card
knowledge is the bucket of the current street only (imperfect recall).
Nodes belong to one player, so the two players' info sets never collide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nutsfinder.abstraction.config import DEFAULT_CONFIG, AbstractionConfig
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import HandState, apply_action, deal_board, legal_actions, new_hand
from nutsfinder.engine.cards import FULL_DECK

__all__ = [
    "CHANCE", "TERMINAL", "NOT_TERMINAL", "FOLD", "SHOWDOWN",
    "BettingTree", "abstract_actions", "build_tree", "is_all_in_action",
]

CHANCE = -1     # street transition (matches nutsfinder.gametheory.game.CHANCE)
TERMINAL = -2
NOT_TERMINAL, FOLD, SHOWDOWN = 0, 1, 2

_DUMMY_DECK = FULL_DECK[:5]  # betting never looks at the cards


def _all_in_to(state: HandState) -> int:
    """The largest bet/raise-to worth making (what the opponent can match), as in bots/common.py."""
    p = state.to_act
    assert p is not None
    legal = legal_actions(state)
    eff = min(legal.max_to, state.current_bet + state.stacks[1 - p])
    return max(eff, legal.min_to)


def abstract_actions(state: HandState, raises: int, config: AbstractionConfig) -> tuple[Action, ...]:
    """The abstract actions at ``state``, in a fixed order: fold?, check/call, sizes up, all-in.

    ``raises`` is the number of non-all-in bets/raises already made on this
    street. Sizes are clamped to ``[min_to, all-in]``; a size that clamps to
    the all-in amount *is* the all-in (listed once, last), and equal sizes
    are listed once.
    """
    legal = legal_actions(state)
    acts: list[Action] = [Fold(), Call()] if legal.can_call else [Check()]
    if not (legal.can_bet or legal.can_raise):
        return tuple(acts)
    street = int(state.street)
    all_in = _all_in_to(state)
    sizes: set[int] = set()
    if raises < config.cap(street):
        if street == 0 and raises == 0:
            targets = [round(config.preflop_open_bb * config.big_blind)]
        else:
            base = state.pot + legal.call_amount
            targets = [round(state.current_bet + f * base) for f in config.fractions(street, raises)]
        sizes = {min(max(t, legal.min_to), all_in) for t in targets}
    sizes.discard(all_in)
    make = Bet if legal.can_bet else Raise
    acts += [make(to) for to in sorted(sizes)]
    acts.append(make(all_in))
    return tuple(acts)


def is_all_in_action(state: HandState, action: Action) -> bool:
    """Whether a bet/raise is the all-in of :func:`abstract_actions` (it does not count to the cap)."""
    to = action.amount if isinstance(action, Bet) else action.to if isinstance(action, Raise) else None
    return to is not None and to == _all_in_to(state)


@dataclass
class BettingTree:
    """The flattened tree (see module doc). Node 0 is the root; lists are indexed by node id.

    Edges: node n's children are ``children[first_child[n] : first_child[n] + n_children[n]]``,
    reached by ``edge_action`` at the same positions (engine actions; ``None`` on the
    single edge out of a street transition).
    """

    config: AbstractionConfig
    player: list[int] = field(default_factory=list)
    street: list[int] = field(default_factory=list)
    terminal: list[int] = field(default_factory=list)
    contrib: list[tuple[int, int]] = field(default_factory=list)
    raises: list[int] = field(default_factory=list)
    parent: list[int] = field(default_factory=list)
    fold_value: list[int] = field(default_factory=list)
    stake: list[int] = field(default_factory=list)
    first_child: list[int] = field(default_factory=list)
    n_children: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)
    edge_action: list[Action | None] = field(default_factory=list)
    infoset_base: list[int] = field(default_factory=list)
    n_infosets: int = 0

    def as_arrays(self) -> dict[str, np.ndarray]:
        """The integer fields as numpy int64 arrays (e.g. for ``np.savez`` next to a blueprint).

        ``contrib`` becomes an ``(n_nodes, 2)`` array; the engine actions are
        encoded as ``edge_kind`` (0 fold, 1 check, 2 call, 3 bet, 4 raise,
        -1 street transition) and ``edge_to`` (bet/raise-to amount, else 0).
        """
        kinds = {Fold: 0, Check: 1, Call: 2, Bet: 3, Raise: 4}
        out = {name: np.array(getattr(self, name), dtype=np.int64) for name in (
            "player", "street", "terminal", "raises", "parent", "fold_value", "stake",
            "first_child", "n_children", "children", "infoset_base")}
        out["contrib"] = np.array(self.contrib, dtype=np.int64).reshape(-1, 2)
        out["edge_kind"] = np.array([-1 if a is None else kinds[type(a)] for a in self.edge_action],
                                    dtype=np.int64)
        out["edge_to"] = np.array([a.amount if isinstance(a, Bet) else a.to if isinstance(a, Raise) else 0
                                   for a in self.edge_action], dtype=np.int64)
        return out

    @property
    def n_nodes(self) -> int:
        return len(self.player)

    def child_ids(self, n: int) -> list[int]:
        s = self.first_child[n]
        return self.children[s:s + self.n_children[n]]

    def actions(self, n: int) -> list[Action | None]:
        s = self.first_child[n]
        return self.edge_action[s:s + self.n_children[n]]

    def infoset(self, n: int, bucket: int) -> int:
        """Info-set index of decision node n with card bucket ``bucket``."""
        if not 0 <= bucket < self.config.buckets(self.street[n]):
            raise ValueError(f"bucket {bucket} out of range on street {self.street[n]}")
        return self.infoset_base[n] + bucket

    def decode_infoset(self, k: int) -> tuple[int, int]:
        """``(node, bucket)`` of info-set index k (inverse of :meth:`infoset`)."""
        import bisect
        bases = self._decision_bases()
        j = bisect.bisect_right([b for b, _ in bases], k) - 1
        base, node = bases[j]
        return node, k - base

    def _decision_bases(self) -> list[tuple[int, int]]:
        return [(b, n) for n, b in enumerate(self.infoset_base) if b >= 0]

    def path(self, n: int) -> list[Action]:
        """The actions from the root to node n (street transitions skipped)."""
        out = []
        while self.parent[n] >= 0:
            p = self.parent[n]
            a = self.actions(p)[self.child_ids(p).index(n)]
            if a is not None:
                out.append(a)
            n = p
        return out[::-1]

    def counts(self) -> dict[str, int]:
        """Node and info-set counts, overall and per street / player."""
        out = {"nodes": self.n_nodes, "infosets": self.n_infosets,
               "terminals": sum(1 for p in self.player if p == TERMINAL),
               "fold_terminals": sum(1 for t in self.terminal if t == FOLD),
               "showdown_terminals": sum(1 for t in self.terminal if t == SHOWDOWN),
               "street_transitions": sum(1 for p in self.player if p == CHANCE)}
        for s, name in enumerate(("preflop", "flop", "turn", "river")):
            dec = [n for n in range(self.n_nodes) if self.player[n] in (0, 1) and self.street[n] == s]
            out[f"{name}_decisions"] = len(dec)
            out[f"{name}_infosets"] = len(dec) * self.config.buckets(s)
        for p in (0, 1):
            out[f"player{p}_infosets"] = sum(self.config.buckets(self.street[n])
                                            for n in range(self.n_nodes) if self.player[n] == p)
        return out


def build_tree(config: AbstractionConfig = DEFAULT_CONFIG) -> BettingTree:
    """Build the abstract tree by playing every abstract action through the engine."""
    tree = BettingTree(config)
    root = new_hand(button=0, stacks=(config.stack, config.stack),
                    small_blind=config.small_blind, big_blind=config.big_blind)
    _expand(tree, root, raises=0, parent=-1)
    return tree


def _new_node(tree: BettingTree, state: HandState, raises: int, parent: int) -> int:
    n = tree.n_nodes
    tree.street.append(int(state.street))
    tree.contrib.append(state.total_bets)
    tree.raises.append(raises)
    tree.parent.append(parent)
    tree.first_child.append(0)
    tree.n_children.append(0)
    tree.infoset_base.append(-1)
    if state.is_hand_over or (state.to_act is None and min(state.stacks) == 0):
        # Hand over, or chips all in with streets still to come: no more decisions.
        folded = state.folded
        tree.player.append(TERMINAL)
        tree.terminal.append(FOLD if folded is not None else SHOWDOWN)
        a, b = state.total_bets
        tree.fold_value.append(0 if folded is None else (-a if folded == 0 else b))
        tree.stake.append(min(a, b))
    else:
        tree.player.append(CHANCE if state.to_act is None else state.to_act)
        tree.terminal.append(NOT_TERMINAL)
        tree.fold_value.append(0)
        tree.stake.append(0)
    return n


def _expand(tree: BettingTree, state: HandState, raises: int, parent: int) -> int:
    n = _new_node(tree, state, raises, parent)
    p = tree.player[n]
    if p == TERMINAL:
        return n
    if p == CHANCE:
        acts_children = [(None, deal_board(state, _DUMMY_DECK[len(state.board):len(state.board) + state.cards_needed]), 0)]
    else:
        tree.infoset_base[n] = tree.n_infosets
        tree.n_infosets += tree.config.buckets(int(state.street))
        acts_children = []
        for a in abstract_actions(state, raises, tree.config):
            counts = isinstance(a, (Bet, Raise)) and not is_all_in_action(state, a)
            acts_children.append((a, apply_action(state, p, a), raises + counts))
    # Children are expanded depth-first, then their ids are laid out contiguously.
    child_ids = [_expand(tree, child, r, n) for _, child, r in acts_children]
    tree.first_child[n] = len(tree.children)
    tree.n_children[n] = len(child_ids)
    tree.children += child_ids
    tree.edge_action += [a for a, _, _ in acts_children]
    return n
