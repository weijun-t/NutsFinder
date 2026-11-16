"""Full-tree counterfactual regret minimisation (CFR) and its variants, for any :class:`Game`.

What CFR computes
-----------------
At every information set I of player i, keep a cumulative regret R(I, a)
per action and play the regret-matching strategy (see
:mod:`nutsfinder.gametheory.regret`). One iteration t:

1. sigma^t(I) = regret_matching(R(I, .)) at every info set.
2. Walk the tree once and compute, for every history h, the value v(h)
   (to player 0) of playing sigma^t from h on.
3. For every history h in an info set I of player i and every action a:

       r^t(I, a) += pi_c(h) * pi_{-i}(h) * s_i * (v(h.a) - v(h))

   where pi_c is chance's reach, pi_{-i} the *opponent's* reach (the
   counterfactual weight: "as if i had tried to reach I") and s_i = +1 for
   player 0, -1 for player 1 (values are stored from P0's side).
4. R += r^t; and the average-strategy accumulator
   S(I, a) += pi_i(h) * sigma^t(I, a), weighted by i's *own* reach.

The output is the normalised average S(I, .) / sum_a S(I, a). In a
two-player zero-sum game, if both players' average counterfactual regret is
at most eps, the average profile is a 2*eps-equilibrium; eps = O(1/sqrt(T))
(docs/phase6_research.md, section 5). The *current* strategy sigma^t need
not converge at all. An info set whose accumulator sums to 0 (never reached
by its owner with positive probability) is reported as uniform: any
strategy there is a best response, and uniform is the usual convention.

Worked example (Kuhn, iteration 1, all regrets 0 so sigma = uniform). P1
holds K facing a bet (info set "Kb"). Histories: (J, K, b) and (Q, K, b),
each with chance reach 1/6 and P0 reach 1/2 (P0 bet with prob 1/2). Payoffs
to P1: fold -1, call +2 (K wins), so v = 0.5 * (-1) + 0.5 * 2 = 0.5. Each
history adds 1/6 * 1/2 * (-1 - 0.5) = -1/8 to fold's regret and
1/6 * 1/2 * (2 - 0.5) = +1/8 to call's. Total R("Kb") = [-1/4, +1/4], so
iteration 2 always calls with K. The tests check this number.

Variants (all options of one solver)
------------------------------------
=========  =========  ==========  ======================  ======================
variant    updates    regret      iteration-t regret       average strategy
                      floor       weight / discount        weight
=========  =========  ==========  ======================  ======================
vanilla    simult.    no          1                        1 (uniform average)
cfr+       altern.    R=max(R,0)  1                        t (linear average)
linear     altern.    no          t                        t
dcfr       altern.    no          after iteration t:       after iteration t:
                                  R>0 x t^1.5/(t^1.5+1),   S x (t/(t+1))^2
                                  R<0 x t^0/(t^0+1) = 1/2  (so weight ~ t^2)
=========  =========  ==========  ======================  ======================

- **Simultaneous updates** (vanilla, Zinkevich et al. 2007): one tree walk
  per iteration computes both players' regrets against the same sigma^t.
- **Alternating updates** (CFR+, Tammelin 2014; also used for linear CFR
  and DCFR by Brown & Sandholm 2019): iteration t first updates player 0
  against sigma^t, then recomputes player 0's strategy and updates player 1
  against (sigma_0^{t+1}, sigma_1^t). Two tree walks per iteration; each
  player's average accumulates only on that player's own walk. Any variant
  can be switched with ``alternating=...``.
- **Linear CFR** (Brown & Sandholm 2019) weights iteration t by t. It is
  DCFR(1, 1, 1): discounting by t/(t+1) after each iteration leaves
  iteration t with weight prod_{s=t..T} s/(s+1) = t/(T+1), proportional to
  t; regret matching ignores the common factor. The tests check that the
  two give the same strategies.
- **DCFR(alpha, beta, gamma)** (Brown & Sandholm 2019; recommended 1.5, 0, 2):
  the discount is applied to the updated player's tables *after* adding
  iteration t's regret / strategy (our convention; applying it before
  only shifts t by one). beta = 0 halves negative regrets every iteration,
  so a bad action can recover quickly; gamma = 2 gives iteration t an
  average weight (t/(T+1))^2.

All variants are deterministic: there is no sampling, the tree order is
fixed, and the floating-point summation order is fixed, so the same game,
variant and iteration count give bit-identical tables.

Implementation: a flat, level-ordered tree
-------------------------------------------
The game tree is enumerated once (breadth first, so every depth is one
contiguous block of node ids) into numpy arrays: each node's parent, the
probability slot of the edge into it (info set offset + action index) and
chance probability, and each terminal's utility. Info sets are numbered
player 0 first, then player 1, so a player's slots are one contiguous range.
An iteration is then a handful of array operations per depth level:

- forward: reach[node] = reach[parent] * edge probability, kept separately
  for player 0, player 1 and chance;
- backward: v[parent] = sum over children of edge prob * v[child]
  (``np.bincount`` over parents), deepest level first;
- regrets and average contributions: ``np.bincount`` of per-edge weights
  into slots.

Leduc (1,939 histories, 288 info sets) costs about 0.12 ms per tree walk
in this form, so 10,000 CFR+ iterations take a few seconds; the arithmetic is exactly the recursive
pseudocode above (a recursive reference in ``tests/test_cfr.py`` checks the
two agree to 1e-12).
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from nutsfinder.gametheory.best_response import best_response_value, expected_value
from nutsfinder.gametheory.game import CHANCE, Action, Game, History, InfosetKey
from nutsfinder.gametheory.regret import regret_matching_flat

_TERMINAL = 2  # node kind besides 0, 1 and CHANCE


@dataclass(frozen=True)
class Variant:
    """Options that define a CFR variant (see the module table).

    ``regret_power`` / ``average_power``: iteration t's instantaneous regret
    / strategy contribution is multiplied by t**power (0 = unweighted).
    ``discount``: DCFR (alpha, beta, gamma), applied after each update.
    """

    name: str
    alternating: bool
    regret_floor: bool = False
    regret_power: float = 0.0
    average_power: float = 0.0
    discount: tuple[float, float, float] | None = None


VARIANTS: dict[str, Variant] = {
    "vanilla": Variant("vanilla", alternating=False),
    "cfr+": Variant("cfr+", alternating=True, regret_floor=True, average_power=1.0),
    "linear": Variant("linear", alternating=True, regret_power=1.0, average_power=1.0),
    "dcfr": Variant("dcfr", alternating=True, discount=(1.5, 0.0, 2.0)),
}


@dataclass(frozen=True)
class Checkpoint:
    """Exact evaluation of the average strategy after ``iteration`` iterations.

    ``br0`` / ``br1`` are the best-response values of each player against
    the other's average strategy, so the game value v* lies in
    [-br1, br0] (P0 can guarantee -br1 with its average strategy and P1
    holds P0 to br0). ``exploitability`` = (br0 + br1) / 2.
    ``seconds`` is cumulative solver time, evaluation excluded.
    """

    iteration: int
    exploitability: float
    br0: float
    br1: float
    value: float
    seconds: float

    @property
    def value_bounds(self) -> tuple[float, float]:
        """Certified interval for the game value (to player 0)."""
        return (-self.br1, self.br0)


class _FlatTree:
    """The game tree as level-ordered numpy arrays (see module doc)."""

    def __init__(self, game: Game) -> None:
        histories: list[History] = [game.initial()]
        parent, edge_action, chance_p, depth = [-1], [None], [1.0], [0]
        i = 0
        while i < len(histories):  # breadth first: depths are contiguous
            h = histories[i]
            if not game.is_terminal(h):
                if game.player(h) == CHANCE:
                    children = game.chance_outcomes(h)
                else:
                    children = [(a, 1.0) for a in game.actions(h)]
                for a, p in children:
                    histories.append(game.next(h, a))
                    parent.append(i)
                    edge_action.append(a)
                    chance_p.append(p)
                    depth.append(depth[i] + 1)
            i += 1
        n = len(histories)

        kind = np.empty(n, dtype=np.int64)
        util = np.zeros(n)
        keys: dict[InfosetKey, tuple[int, tuple[Action, ...]]] = {}
        node_key: list[InfosetKey | None] = [None] * n
        for k, h in enumerate(histories):
            if game.is_terminal(h):
                kind[k] = _TERMINAL
                util[k] = game.utility(h)
                continue
            p = game.player(h)
            kind[k] = p
            if p != CHANCE:
                key, acts = game.infoset_key(h), tuple(game.actions(h))
                if keys.setdefault(key, (p, acts)) != (p, acts):
                    raise ValueError(f"info set {key!r} is inconsistent")
                node_key[k] = key

        # Info sets: player 0 first, then player 1, first-visit order inside.
        order = [k for k, (p, _) in keys.items() if p == 0] + \
                [k for k, (p, _) in keys.items() if p == 1]
        self.keys: list[InfosetKey] = order
        self.actions: list[tuple[Action, ...]] = [keys[k][1] for k in order]
        self.owner = np.array([keys[k][0] for k in order], dtype=np.int64)
        offset, pos = {}, 0
        for k in order:
            offset[k] = pos
            pos += len(keys[k][1])
        self.offset = np.array([offset[k] for k in order], dtype=np.int64)
        self.n_slots = pos
        self.slot_infoset = np.repeat(np.arange(len(order)), [len(a) for a in self.actions])
        self.slot_size = np.repeat([len(a) for a in self.actions],
                                   [len(a) for a in self.actions]).astype(float)
        self.slot_owner = self.owner[self.slot_infoset]
        n0_slots = int(np.sum(self.slot_owner == 0))
        self.player_slots = (slice(0, n0_slots), slice(n0_slots, self.n_slots))

        par = np.array(parent, dtype=np.int64)
        slot = np.full(n, -1, dtype=np.int64)
        for k in range(1, n):
            key = node_key[parent[k]]
            if key is not None:
                slot[k] = offset[key] + keys[key][1].index(edge_action[k])
        self.n_nodes = n
        self.parent = par
        self.kind = kind
        self.util = util
        self.chance_p = np.array(chance_p)
        self.slot = slot
        parent_kind = np.full(n, CHANCE, dtype=np.int64)
        parent_kind[1:] = kind[par[1:]]
        # Edges out of each player's nodes: (child ids, parent ids, slots).
        self.edges = []
        for p in (0, 1):
            child = np.nonzero(parent_kind == p)[0]
            child = child[child > 0]
            self.edges.append((child, par[child], slot[child]))
        self.player_edge = np.nonzero(slot >= 0)[0]
        self.parent_is = [(parent_kind == p) & (np.arange(n) > 0) for p in (0, 1)]
        self.parent_is_chance = (parent_kind == CHANCE) & (np.arange(n) > 0)

        # Levels: node range [lo, hi) at depth d, with parents in level d-1.
        dep = np.array(depth)
        self.levels: list[tuple[int, int, np.ndarray, int, int]] = []
        bounds = [0] + [int(x) for x in np.nonzero(np.diff(dep))[0] + 1] + [n]
        for d in range(1, len(bounds) - 1):
            lo, hi = bounds[d], bounds[d + 1]
            plo, phi = bounds[d - 1], bounds[d]
            self.levels.append((lo, hi, par[lo:hi], plo, phi))

    # ------------------------------------------------------------- passes

    def edge_probs(self, sigma: np.ndarray) -> np.ndarray:
        """Probability of the edge into every node (root: 1)."""
        ep = self.chance_p.copy()
        ep[self.player_edge] = sigma[self.slot[self.player_edge]]
        return ep

    def reaches(self, ep: np.ndarray) -> np.ndarray:
        """Rows: pi_0, pi_1, pi_c for every node."""
        factors = np.ones((3, self.n_nodes))
        factors[0, self.parent_is[0]] = ep[self.parent_is[0]]
        factors[1, self.parent_is[1]] = ep[self.parent_is[1]]
        factors[2, self.parent_is_chance] = ep[self.parent_is_chance]
        reach = np.ones((3, self.n_nodes))
        for lo, hi, par, _, _ in self.levels:
            reach[:, lo:hi] = reach[:, par] * factors[:, lo:hi]
        return reach

    def values(self, ep: np.ndarray) -> np.ndarray:
        """v(h) to player 0 for every node under the edge probabilities ``ep``."""
        v = self.util.copy()
        for lo, hi, par, plo, phi in reversed(self.levels):
            v[plo:phi] += np.bincount(par - plo, weights=ep[lo:hi] * v[lo:hi],
                                      minlength=phi - plo)
        return v

    def profile(self, probs: np.ndarray) -> dict[InfosetKey, dict[Action, float]]:
        """Slot array -> ``{key: {action: prob}}``."""
        out = {}
        for key, acts, off in zip(self.keys, self.actions, self.offset):
            out[key] = {a: float(probs[off + j]) for j, a in enumerate(acts)}
        return out


class CFRSolver:
    """Full-tree CFR on any :class:`Game`.

    >>> from nutsfinder.gametheory import Kuhn, exploitability
    >>> s = CFRSolver(Kuhn(), "cfr+")
    >>> s.run(200)
    []
    >>> exploitability(Kuhn(), s.average_strategy()) < 0.01
    True

    ``variant`` is a key of :data:`VARIANTS` or a :class:`Variant`; keyword
    ``overrides`` replace its fields (e.g. ``alternating=False``). There is
    no randomness, so no seed.
    """

    def __init__(self, game: Game, variant: str | Variant = "vanilla", **overrides) -> None:
        base = VARIANTS[variant] if isinstance(variant, str) else variant
        self.variant: Variant = dataclasses.replace(base, **overrides) if overrides else base
        self.game = game
        self.tree = _FlatTree(game)
        self.regrets = np.zeros(self.tree.n_slots)
        self.strategy_sum = np.zeros(self.tree.n_slots)
        self.iteration = 0
        self.seconds = 0.0

    # --------------------------------------------------------------- core

    def _current(self) -> np.ndarray:
        return regret_matching_flat(self.regrets, self.tree.slot_infoset, self.tree.slot_size)

    def _update(self, players: tuple[int, ...], t: int) -> None:
        """Walk the tree under the current strategy and update ``players``' tables."""
        tree, var = self.tree, self.variant
        sigma = self._current()
        ep = tree.edge_probs(sigma)
        reach = tree.reaches(ep)
        v = tree.values(ep)
        for p in players:
            child, par, slot = tree.edges[p]
            sign = 1.0 if p == 0 else -1.0
            cf_weight = reach[2, par] * reach[1 - p, par]
            inst = np.bincount(slot, weights=cf_weight * sign * (v[child] - v[par]),
                               minlength=tree.n_slots)
            avg = np.bincount(slot, weights=reach[p, par] * ep[child], minlength=tree.n_slots)
            s = tree.player_slots[p]
            r = self.regrets[s] + (inst[s] * t ** var.regret_power if var.regret_power else inst[s])
            if var.regret_floor:
                np.maximum(r, 0.0, out=r)
            a = self.strategy_sum[s] + (avg[s] * t ** var.average_power
                                        if var.average_power else avg[s])
            if var.discount is not None:
                alpha, beta, gamma = var.discount
                pos, neg = t ** alpha / (t ** alpha + 1.0), t ** beta / (t ** beta + 1.0)
                r *= np.where(r > 0.0, pos, neg)
                a *= (t / (t + 1.0)) ** gamma
            self.regrets[s] = r
            self.strategy_sum[s] = a

    def step(self) -> None:
        """One CFR iteration (two tree walks if updates alternate)."""
        start = time.perf_counter()
        self.iteration += 1
        t = self.iteration
        if self.variant.alternating:
            self._update((0,), t)
            self._update((1,), t)
        else:
            self._update((0, 1), t)
        self.seconds += time.perf_counter() - start

    def run(self, iterations: int, *, eval_every: int | None = None,
            eval_at: Iterable[int] = (),
            callback: Callable[[Checkpoint], None] | None = None) -> list[Checkpoint]:
        """Run ``iterations`` more iterations; evaluate exactly at checkpoints.

        A checkpoint is taken whenever the (absolute) iteration count is a
        multiple of ``eval_every`` or is in ``eval_at`` (e.g. log-spaced
        points for a log-log plot). Returns the checkpoints in order;
        ``callback`` sees each as it is made.
        """
        at = set(eval_at)
        out: list[Checkpoint] = []
        for _ in range(iterations):
            self.step()
            t = self.iteration
            if t in at or (eval_every and t % eval_every == 0):
                cp = self.evaluate()
                out.append(cp)
                if callback is not None:
                    callback(cp)
        return out

    # ------------------------------------------------------------ outputs

    def _normalise(self, table: np.ndarray) -> np.ndarray:
        tree = self.tree
        totals = np.bincount(tree.slot_infoset, weights=table)[tree.slot_infoset]
        out = np.empty_like(table)
        np.divide(table, totals, out=out, where=totals > 0.0)
        flat = totals <= 0.0
        out[flat] = 1.0 / tree.slot_size[flat]
        return out

    def average_strategy(self) -> dict[InfosetKey, dict[Action, float]]:
        """The normalised average strategy (the CFR output) as a profile.

        Info sets never reached by their owner (zero accumulator) are uniform.
        """
        return self.tree.profile(self._normalise(self.strategy_sum))

    def current_strategy(self) -> dict[InfosetKey, dict[Action, float]]:
        """The regret-matching strategy for the next iteration (need not converge)."""
        return self.tree.profile(self._current())

    def evaluate(self) -> Checkpoint:
        """Exact exploitability, BR values and value of the average strategy."""
        prof = self.average_strategy()
        br0 = best_response_value(self.game, prof, 0)
        br1 = best_response_value(self.game, prof, 1)
        return Checkpoint(self.iteration, (br0 + br1) / 2.0, br0, br1,
                          expected_value(self.game, prof), self.seconds)


def log_spaced(last: int, per_decade: int = 10) -> list[int]:
    """Distinct integers 1..last spaced evenly on a log scale, ending at ``last``.

    >>> log_spaced(100, 2)
    [1, 3, 10, 32, 100]
    """
    n = int(round(np.log10(last) * per_decade))
    pts = sorted({int(round(x)) for x in np.logspace(0, np.log10(last), n + 1)} | {last})
    return pts
