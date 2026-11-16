"""Abstract heads-up hold'em as a :class:`~nutsfinder.gametheory.game.Game`.

The game is the real game, played on the abstract tree:

- **Chance** deals *real* cards once, at the root: both players' hole
  cards and the whole five-card board (9 distinct cards). Dealing the board
  up front instead of street by street is the same game (each card is
  still uniform given everything before it); the board is only *revealed*
  street by street, through the buckets. All 2.8e12 deals cannot be listed,
  so the game implements the optional ``sample_chance(h, rng)`` hook of
  :mod:`nutsfinder.gametheory.game` instead of ``chance_outcomes``; the
  MCCFR solver uses it to draw one deal per traversal.
- **Players** move on the nodes of :func:`~nutsfinder.abstraction.tree.build_tree`.
  Street transitions are skipped (the cards are already dealt).
- **Info sets**: ``infoset_key(h) = tree.infoset(node, bucket)``, an int,
  where the bucket is the acting player's bucket on the node's street,
  computed from *their own* hole cards and the board cards out on that
  street. The opponent's cards never enter it (``tests/test_abstraction_game.py``
  checks this with a spy deal). ``tree.decode_infoset`` gives back
  ``(node, bucket)``.
- **Utility** at a showdown compares the two real 7-card hands with the
  real :func:`~nutsfinder.engine.evaluator.evaluate`; at a fold it is the
  folder's contribution. Payoffs are chips to seat 0 (the button).

A history is the pair ``(node, deal)``: node -1 is the root chance node.
Deals are compared by identity (two separately sampled deals are different
histories even if their cards agree), which is all a sampling solver needs.

Worked example: deal As Ah to seat 0, 7c 2d to seat 1, board Kd 9s 4c 3h 2s.
Preflop seat 0 is in bucket 0 ("AA"); the root's info set is
``tree.infoset(0, 0) = 0``. If seat 0 raises to 300 and seat 1 calls, seat 1
acts first on the flop with the bucket of 7c 2d on Kd 9s 4c. If the hand
checks down, utility = +300 (aces win 300 from the big blind).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nutsfinder.abstraction.cards import CardAbstraction
from nutsfinder.abstraction.config import DEFAULT_CONFIG, AbstractionConfig
from nutsfinder.abstraction.tree import CHANCE as TREE_CHANCE
from nutsfinder.abstraction.tree import FOLD, BettingTree, build_tree
from nutsfinder.engine.cards import FULL_DECK
from nutsfinder.engine.evaluator import evaluate
from nutsfinder.gametheory.game import CHANCE

__all__ = ["Deal", "AbstractHoldem"]

_BOARD_SIZE = (0, 3, 4, 5)


@dataclass(frozen=True, eq=False, slots=True)
class Deal:
    """One real deal and what the abstraction makes of it.

    ``hole[p]``: seat p's two card ids; ``board``: five ids (flop, turn, river).
    ``buckets[p][s]``: seat p's bucket on street s, from ``hole[p]`` and the
    first ``(0, 3, 4, 5)[s]`` board cards only. ``result``: +1 if seat 0 wins
    the showdown, -1 if seat 1 does, 0 for a split.
    """

    hole: tuple[tuple[int, int], tuple[int, int]]
    board: tuple[int, int, int, int, int]
    buckets: tuple[tuple[int, int, int, int], tuple[int, int, int, int]]
    result: int


class AbstractHoldem:
    """Heads-up NLHE on the abstract tree with real cards (see module doc)."""

    def __init__(self, config: AbstractionConfig = DEFAULT_CONFIG,
                 cards: CardAbstraction | None = None, tree: BettingTree | None = None) -> None:
        self.config = config
        self.cards = cards if cards is not None else CardAbstraction(config)
        self.tree = tree if tree is not None else build_tree(config)
        t = self.tree
        # Hot-loop copies: plain lists and tuples, street transitions skipped.
        self._player = list(t.player)
        self._street = list(t.street)
        self._terminal = [x != 0 for x in t.terminal]
        self._fold = [x == FOLD for x in t.terminal]
        self._fold_value = list(t.fold_value)
        self._stake = list(t.stake)
        self._base = list(t.infoset_base)
        self._kids = [tuple(self._skip(c) for c in t.child_ids(n)) for n in range(t.n_nodes)]
        self._acts = [tuple(range(len(k))) for k in self._kids]
        self._root = self._skip(0)

    def _skip(self, n: int) -> int:
        while self.tree.player[n] == TREE_CHANCE:
            n = self.tree.child_ids(n)[0]
        return n

    # --- dealing
    def make_deal(self, hole0: tuple[int, int], hole1: tuple[int, int],
                  board: tuple[int, int, int, int, int]) -> Deal:
        """A :class:`Deal` from card ids (all 9 distinct): buckets and showdown result."""
        ids = [*hole0, *hole1, *board]
        if len(ids) != 9 or len(set(ids)) != 9 or not all(0 <= c < 52 for c in ids):
            raise ValueError(f"a deal needs 9 distinct card ids, got {ids}")
        cards = self.cards
        buckets = tuple(tuple(cards.bucket(h, board[:_BOARD_SIZE[s]]) for s in range(4))
                        for h in (hole0, hole1))
        r0 = evaluate([FULL_DECK[c] for c in (*hole0, *board)])
        r1 = evaluate([FULL_DECK[c] for c in (*hole1, *board)])
        result = (r0 > r1) - (r0 < r1)
        return Deal((tuple(hole0), tuple(hole1)), tuple(board), buckets, result)  # type: ignore[arg-type]

    def sample_chance(self, h: tuple[int, Deal | None], rng: np.random.Generator) -> Deal:
        """Draw a uniformly random deal (the optional sampling hook of the Game protocol)."""
        c = rng.permutation(52)[:9].tolist()
        return self.make_deal((c[0], c[1]), (c[2], c[3]), (c[4], c[5], c[6], c[7], c[8]))

    # --- the Game protocol
    def initial(self) -> tuple[int, Deal | None]:
        return (-1, None)

    def is_terminal(self, h: tuple[int, Deal | None]) -> bool:
        return h[0] >= 0 and self._terminal[h[0]]

    def player(self, h: tuple[int, Deal | None]) -> int:
        return CHANCE if h[0] < 0 else self._player[h[0]]

    def chance_outcomes(self, h: tuple[int, Deal | None]) -> list[tuple[Deal, float]]:
        raise NotImplementedError("2.8e12 deals cannot be enumerated: use sample_chance(h, rng)")

    def infoset_key(self, h: tuple[int, Deal | None]) -> int:
        n, deal = h
        assert deal is not None
        p = self._player[n]
        return self._base[n] + deal.buckets[p][self._street[n]]

    def actions(self, h: tuple[int, Deal | None]) -> tuple[int, ...]:
        """Action ids ``0..k-1``, aligned with ``tree.actions(node)``."""
        return self._acts[h[0]]

    def next(self, h: tuple[int, Deal | None], a: object) -> tuple[int, Deal | None]:
        n, deal = h
        if n < 0:
            assert isinstance(a, Deal)
            return (self._root, a)
        return (self._kids[n][a], deal)  # type: ignore[index]

    def utility(self, h: tuple[int, Deal | None]) -> float:
        n, deal = h
        if self._fold[n]:
            return float(self._fold_value[n])
        assert deal is not None
        return float(deal.result * self._stake[n])
