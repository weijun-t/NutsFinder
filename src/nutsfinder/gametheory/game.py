"""The ``Game`` protocol: two-player zero-sum extensive-form games with chance.

Why one interface
-----------------
Kuhn poker, Leduc hold'em and (later) the abstract hold'em game all implement
this protocol, so a CFR solver, a best-response routine and an
expected-value routine are written *once* and run on any of them. A solver
that reaches exploitability ~0 on Kuhn and Leduc (where exact best response
is cheap) has been checked before it ever sees hold'em.

The model
---------
An extensive-form game is a tree. A node is identified by its **history** h:
the sequence of everything that has happened so far (chance outcomes and
player actions). At every non-terminal h exactly one of these moves:

- **chance** (``player(h) == CHANCE``): picks action a with a known
  probability sigma_c(h, a). Card deals are chance moves.
- **player 0 or player 1**: picks an action from ``actions(h)``.

A player does not see the whole history: a poker player does not see the
opponent's card. Histories the acting player cannot tell apart form an
**information set** I, and ``infoset_key(h)`` names it. A strategy can only
depend on what the player knows, so it is a map I -> distribution over
actions, the same distribution for every h in I.

Reach probabilities. For a history h, pi(h) is the product of the
probabilities of every move on the path to h. It factorises by who moved:
pi(h) = pi_c(h) * pi_0(h) * pi_1(h). Worked example (Kuhn, below): P0 is dealt
K (1/3), P1 dealt Q (1/2), P0 bets with prob 0.75, P1 calls with prob 0.3:
pi_c = 1/6, pi_0 = 0.75, pi_1 = 0.3, pi = 0.0375.

Zero-sum. ``utility(h)`` is the payoff to player 0 at a terminal h; player 1
gets the negative. Payoffs are net chips won (antes included), so in Kuhn a
bet-call showdown win is +2.

The contract (what every implementation guarantees)
---------------------------------------------------
1. Histories are immutable and hashable (the games here use tuples of
   actions), so they can be dict keys and memoised. ``next`` returns a new
   history and never mutates its argument.
2. ``chance_outcomes(h)`` is only called when ``player(h) == CHANCE``; its
   probabilities are positive and sum to 1.
3. ``actions(h)`` and ``infoset_key(h)`` are only called when
   ``player(h) in (0, 1)``. ``actions(h)`` returns a tuple in a fixed order,
   and every history in the same information set returns the *same* tuple.
   So "action index k at info set I" is well defined: solvers may store
   regrets as arrays aligned with ``actions(h)``.
4. ``infoset_key(h)`` depends only on what the acting player knows (own
   private card, public cards, the public betting sequence) and *never* on
   the opponent's private card. Keys of player 0 and player 1 never collide,
   so one dict can hold both players' tables.
5. Perfect recall: the key encodes the player's whole observed past, so a
   player never forgets what it knew. (Exact best response below relies on
   this; an abstract hold'em game with imperfect recall will not, and must
   be evaluated differently.)
6. ``utility(h)`` is only called when ``is_terminal(h)``.

Strategy profiles
-----------------
A :data:`Profile` maps ``infoset_key -> action probabilities``. The
probabilities may be either a ``Mapping[action, float]`` or a sequence of
floats aligned with ``actions(h)`` (e.g. a list or numpy row). An info set
missing from the profile is played **uniformly at random**. Use
:func:`action_probs` to read a profile; it validates the distribution.

How to write a solver against this
----------------------------------
A full-traversal solver is one recursive function over histories::

    def walk(h, reach0, reach1, chance_reach):
        if game.is_terminal(h):
            return game.utility(h)                # payoff to player 0
        p = game.player(h)
        if p == CHANCE:
            return sum(prob * walk(game.next(h, a), reach0, reach1,
                                   chance_reach * prob)
                       for a, prob in game.chance_outcomes(h))
        key, acts = game.infoset_key(h), game.actions(h)
        sigma = current_strategy(key, len(acts))  # regret matching
        values = [walk(game.next(h, a), ...) for a in acts]
        ...  # regret update weighted by the *opponent's* reach x chance reach

Points that bite:

- ``utility`` and every value returned above are from **player 0's** point
  of view. For player 1's regrets use the negative (or carry a sign).
- Counterfactual weight = chance reach x opponent reach (not your own); the
  average-strategy accumulator is weighted by your *own* reach.
- Look up regrets by ``infoset_key(h)``, never by ``h``: that is what makes
  the strategy respect the player's information.
- :func:`infosets` enumerates every info set with its actions once, if you
  want to preallocate tables.
- Check the result with
  :func:`nutsfinder.gametheory.best_response.exploitability`.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterator, Mapping, Sequence
from typing import Protocol, TypeAlias

CHANCE: int = -1
"""Value of ``Game.player(h)`` at a chance node. Players are 0 and 1."""

History: TypeAlias = Hashable
Action: TypeAlias = Hashable
InfosetKey: TypeAlias = Hashable
ActionProbs: TypeAlias = Mapping[Action, float] | Sequence[float]
Profile: TypeAlias = Mapping[InfosetKey, ActionProbs]


class Game(Protocol):
    """A two-player zero-sum extensive-form game with chance. See module doc."""

    def initial(self) -> History:
        """The root history (usually a chance node dealing cards)."""
        ...

    def is_terminal(self, h: History) -> bool:
        """True when the game is over at h."""
        ...

    def utility(self, h: History) -> float:
        """Payoff to player 0 at terminal h (player 1 receives the negative)."""
        ...

    def player(self, h: History) -> int:
        """Who moves at non-terminal h: 0, 1 or :data:`CHANCE`."""
        ...

    def chance_outcomes(self, h: History) -> list[tuple[Action, float]]:
        """``[(action, probability), ...]`` at a chance node; probabilities sum to 1."""
        ...

    def infoset_key(self, h: History) -> InfosetKey:
        """Hashable name of the acting player's information set at h."""
        ...

    def actions(self, h: History) -> tuple[Action, ...]:
        """Legal actions at a player node, in a fixed order per info set."""
        ...

    def next(self, h: History, a: Action) -> History:
        """The history after action a (chance or player) is taken at h."""
        ...


def action_probs(profile: Profile, key: InfosetKey,
                 acts: Sequence[Action]) -> list[float]:
    """Probabilities of ``acts`` at info set ``key``, aligned with ``acts``.

    Missing info set -> uniform. A mapping entry may omit actions (they get
    probability 0). Raises ``ValueError`` on negative entries, a wrong
    length, or a total that is not 1 (tolerance 1e-6), because a silently
    unnormalised strategy makes every downstream number wrong.

    >>> action_probs({"K": {"b": 1.0}}, "K", ("p", "b"))
    [0.0, 1.0]
    >>> action_probs({}, "K", ("p", "b"))
    [0.5, 0.5]
    """
    entry = profile.get(key)
    if entry is None:
        return [1.0 / len(acts)] * len(acts)
    if isinstance(entry, Mapping):
        unknown = set(entry) - set(acts)
        if unknown:
            raise ValueError(f"info set {key!r}: unknown actions {unknown}")
        probs = [float(entry.get(a, 0.0)) for a in acts]
    else:
        probs = [float(x) for x in entry]
        if len(probs) != len(acts):
            raise ValueError(f"info set {key!r}: {len(probs)} probs for {len(acts)} actions")
    if any(p < 0.0 for p in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-6):
        raise ValueError(f"info set {key!r}: not a distribution: {probs}")
    return probs


def iter_histories(game: Game) -> Iterator[History]:
    """Every history of the game tree, depth-first, root first."""
    stack = [game.initial()]
    while stack:
        h = stack.pop()
        yield h
        if game.is_terminal(h):
            continue
        if game.player(h) == CHANCE:
            children = [a for a, _ in game.chance_outcomes(h)]
        else:
            children = list(game.actions(h))
        stack.extend(game.next(h, a) for a in reversed(children))


def infosets(game: Game) -> dict[InfosetKey, tuple[int, tuple[Action, ...]]]:
    """Every info set: ``key -> (player, actions)``, in first-visit order.

    Raises ``ValueError`` if two histories with the same key disagree on the
    player or the action tuple (a broken ``infoset_key``).
    """
    out: dict[InfosetKey, tuple[int, tuple[Action, ...]]] = {}
    for h in iter_histories(game):
        if game.is_terminal(h) or game.player(h) == CHANCE:
            continue
        key = game.infoset_key(h)
        entry = (game.player(h), tuple(game.actions(h)))
        if out.setdefault(key, entry) != entry:
            raise ValueError(f"info set {key!r} is inconsistent: {out[key]} vs {entry}")
    return out
