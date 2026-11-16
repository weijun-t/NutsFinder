"""External-sampling Monte Carlo CFR (ES-MCCFR) for any :class:`Game`.

Why sample
----------
Full-tree CFR (:mod:`nutsfinder.gametheory.cfr`) walks every history every
iteration. Leduc has 1,939 histories, so that is cheap; an abstract hold'em
game has far too many deals to enumerate. Monte Carlo CFR (Lanctot et al.
2009) walks only a *sampled* part of the tree per iteration, and replaces the
exact counterfactual regrets by unbiased estimates of them. Regret matching
tolerates zero-mean noise in its inputs, so the average strategy still
converges (in probability), just more slowly per iteration and much more
cheaply per iteration.

External sampling (what this module does)
-----------------------------------------
One *traversal* for player i (the "traverser"):

- at a **chance** node, sample one outcome with its probability;
- at an **opponent** node, sample one action from the opponent's current
  regret-matching strategy sigma^t;
- at a **traverser** node, try *every* action.

So the traverser sees its whole own subtree, but only one "world" of
chance and opponent choices (one per history: different traverser branches
get independent samples, because they are different histories).

The sampled value. Write the recursion's return value at a history h as
v~(h) (from i's point of view). At a traverser node
v~(h) = sum_a sigma(I, a) v~(h.a); elsewhere v~(h) = v~(h.a_sampled). Then
for any history h, v~(h) is a sum over the terminals z below h that were
sampled of pi_i(h, z) u_i(z): only the traverser's probabilities are
multiplied in, because chance and opponent were *sampled*, not weighted.

Why it is unbiased. A terminal z below h is in the sample with
probability q(h, z) = pi_c(h, z) pi_{-i}(h, z) (every chance and opponent
choice between h and z must go z's way; the traverser's choices are all
taken). Hence

    E[v~(h)] = sum_z q(h, z) pi_i(h, z) u_i(z) = sum_z pi^sigma(h, z) u_i(z) = v_i(h),

the exact value of h under sigma^t. And h itself is reached by the
sampling with probability pi_c(h) pi_{-i}(h), exactly the counterfactual
weight. So the regret update done at every *visited* traverser node,

    R(I, a) += v~(h.a) - v~(h),

has expectation sum_{h in I} pi_c(h) pi_{-i}(h) (v_i(h.a) - v_i(h)): the
full-tree CFR regret increment, with no importance weights needed (they
cancel because the sampling distribution *is* the reach). This is the
pseudocode comment "sampling already weights by opponent and chance reach"
in docs/phase6_research.md section 2. ``tests/test_mccfr.py`` checks this
identity by Monte Carlo against :class:`~nutsfinder.gametheory.cfr.CFRSolver`.

Worked example (Kuhn, all regrets 0, so everyone plays 1/2-1/2). Traverser
P1 at info set "Kb" (holds K, faces a bet). Each of the histories
(J, K, b), (Q, K, b) is reached with probability pi_c * pi_0 = 1/6 * 1/2
= 1/12. At such a visit fold returns -1, call +2, v~ = 0.5, so the update is
[-1.5, +1.5]. Expected update: 2 * 1/12 * [-1.5, +1.5] = [-1/4, +1/4], the
number the full-tree solver's worked example gets in one exact iteration.

The average strategy ("simple" opponent-node averaging)
-------------------------------------------------------
CFR's output is sigma-bar(I) proportional to sum_t pi_p^t(I) sigma^t(I),
weighted by the owner p's *own* reach. In ES we add sigma^t(I) to S(I) at
every **opponent** node the traversal visits, i.e. player p's tables are
averaged during the *other* player's traversal. Why that has the right
expectation: an opponent history h in I is visited with probability
pi_c(h) pi_p(h) (chance sampled, p's own actions sampled from sigma^t, the
traverser expands everything), so

    E[S(I) increment] = sigma^t(I) * pi_p^t(I) * c_I,   c_I = sum_{h in I} pi_c(h).

(pi_p(h) is the same for every h in I under perfect recall.) c_I does not
depend on t, so it cancels when S(I) is normalised: the normalised S is the
reach-weighted average CFR wants, estimated without bias in its numerator
and denominator (the ratio is consistent, as any ratio estimator).
This is Lanctot's "simple averaging" for ES and OpenSpiel's default; it is
cheaper than "stochastically weighted averaging" (carry pi_p / q down the
path and weight by it, which outcome sampling needs because there p's own
actions are not sampled on-policy; for ES the weight would be 1 anyway), and
it never touches the traverser's own tables twice. The tests check the
expectation identity above by Monte Carlo.

Iterations, alternation, weighting
----------------------------------
One **iteration** t = one traversal for player 0, then one for player 1
(alternating updates, like the full-tree alternating variants; player 1's
traversal already sees player 0's updated regrets). ``traversals`` counts
both. Iteration t's regret and average increments are multiplied by a
weight:

- ``linear=False``: weight 1 (plain ES-MCCFR);
- ``linear=True``: weight t, i.e. **linear CFR** (Brown & Sandholm 2019),
  identical to Pluribus's "multiply R and S by (t/D)/(t/D + 1) every D
  iterations" up to the granularity D (see the identity in
  :mod:`~nutsfinder.gametheory.cfr`). Early iterations, played against
  near-random opponents, are forgotten in proportion. With
  ``linear_until=T0`` the weight is min(t, T0): Pluribus stopped discounting
  after its first 400 minutes.

Regret matching is scale-invariant, so only *relative* weights matter.

Regret-based pruning (optional, off by default)
-----------------------------------------------
Pluribus skipped traverser actions whose cumulative regret is below a large
negative threshold C in 95% of traversals (after a warm-up), because such
actions have probability 0 now and will for a long time. Here:
``prune_threshold=C`` (in units of the stored, *weighted* regrets, so with
``linear=True`` its meaning drifts with t^2; Pluribus used a fixed C too),
``prune_prob`` and ``prune_after``. An action is pruned only if its regret
is below C (C must be negative) *and* the current strategy gives it
probability 0: when every regret is <= 0, regret matching is uniform, and
skipping an action that is still being played would drop its share of the
node value. Actions leading straight to a terminal are never pruned (they
are free to evaluate). Pruning gives up exact unbiasedness in two ways: a
pruned action's regret is not updated in that traversal, and the opponent
nodes inside the pruned subtree are not visited, so the *opponent's average
strategy* misses those contributions too. It is a speed trick, not part of
the convergence argument above.

Data structures (built to scale to ~1e5 info sets)
--------------------------------------------------
Info sets are discovered lazily as traversals reach them (hold'em cannot be
enumerated). Each gets an index in first-visit order; its actions occupy a
contiguous slot range ``offset[k] : offset[k] + n_actions[k]`` of two flat
float64 arrays, ``regrets`` and ``strategy_sum``, which grow by doubling.
1e5 info sets x 3 actions x 2 tables x 8 bytes = 4.8 MB. The traversal only
calls :class:`Game` methods; nothing in it knows which game it is playing.

Reallocation. Growing replaces the arrays, and a traversal can discover a
new info set deep in the recursion, so the traversal never holds a
reference to ``_R`` / ``_S`` across a recursive call: every read and write
goes through ``self._R`` / ``self._S`` at the moment it happens, and regret
updates are added as increments rather than written back from a copy taken
before the recursion. (A cached reference wrote updates into the discarded
array; the regression test compares a lazily discovered run with a
pre-registered one.)

Determinism. All randomness comes from one ``numpy.random.Generator``
(PCG64) seeded in the constructor and drawn in fixed-size blocks of
uniforms, consumed in a fixed order. The same game, options, seed and
iteration count give bit-identical tables; no global RNG is used.

Sampled chance (games too big to list). If the game has the optional
``sample_chance(h, rng)`` method (:class:`~nutsfinder.gametheory.game.SampledChance`,
e.g. abstract hold'em with its 2.8e12 deals), chance nodes call it instead
of ``chance_outcomes`` and walk the one outcome it returns. That is exactly
"sample one outcome with its probability", so the argument above is
unchanged. Its ``rng`` is a second PCG64 generator seeded with
``SeedSequence([seed, 1])``, created only for such games: for every other
game the uniform stream, and hence every table, is bit-identical to a
solver without the hook.

Cost measure. ``nodes`` counts every history the traversals visit
(chance, player and terminal), the analogue of the full-tree solver's
``iterations x walks x n_nodes``. On this laptop wall times are unreliable
(thermal throttling), so convergence is reported against node touches too.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from nutsfinder.gametheory.best_response import best_response_value, expected_value
from nutsfinder.gametheory.cfr import Checkpoint
from nutsfinder.gametheory.game import CHANCE, Action, Game, History, InfosetKey

_BLOCK = 4096  # uniforms drawn per RNG call


@dataclass(frozen=True)
class MCCFRCheckpoint(Checkpoint):
    """A :class:`~nutsfinder.gametheory.cfr.Checkpoint` plus sampling cost.

    ``nodes``: histories visited so far (all traversals); ``traversals``:
    2 x iterations; ``infosets``: info sets discovered so far.
    """

    nodes: int = 0
    traversals: int = 0
    infosets: int = 0


class MCCFRSolver:
    """External-sampling MCCFR on any :class:`Game` (see module doc).

    >>> from nutsfinder.gametheory import Kuhn, exploitability
    >>> s = MCCFRSolver(Kuhn(), seed=1)
    >>> s.run(2000)
    []
    >>> exploitability(Kuhn(), s.average_strategy()) < 0.05
    True

    Parameters
    ----------
    game: any :class:`Game`.
    seed: seeds the solver's own PCG64 generator.
    linear: weight iteration t by t (linear CFR); ``linear_until`` caps the weight.
    prune_threshold, prune_prob, prune_after: regret-based pruning (off when
        ``prune_threshold`` is None).
    known_infosets: optional ``key -> (player, actions)`` (e.g.
        :func:`~nutsfinder.gametheory.game.infosets`) to register info sets
        up front in a fixed order; otherwise they are added on first visit.
    """

    def __init__(self, game: Game, *, seed: int = 0, linear: bool = False,
                 linear_until: int | None = None, prune_threshold: float | None = None,
                 prune_prob: float = 0.95, prune_after: int = 0,
                 known_infosets: Mapping[InfosetKey, tuple[int, tuple[Action, ...]]] | None = None
                 ) -> None:
        if linear_until is not None and linear_until < 1:
            raise ValueError("linear_until must be >= 1")
        if prune_threshold is not None and not prune_threshold < 0.0:
            raise ValueError("prune_threshold must be negative (prune only actions with "
                             "large negative regret)")
        if not 0.0 <= prune_prob <= 1.0:
            raise ValueError("prune_prob must be in [0, 1]")
        self.game = game
        self.seed = seed
        self.linear = linear
        self.linear_until = linear_until
        self.prune_threshold = prune_threshold
        self.prune_prob = prune_prob
        self.prune_after = prune_after
        # Optional hook for games whose chance outcomes cannot be listed (see
        # game.SampledChance). Its draws come from a second stream, so games
        # without the hook consume exactly the same uniforms as before.
        self._sample_chance = getattr(game, "sample_chance", None)
        self.reseed(seed)
        # Info-set table: key -> index; per index offset, size, owner, actions.
        self._index: dict[InfosetKey, int] = {}
        self.keys: list[InfosetKey] = []
        self.actions: list[tuple[Action, ...]] = []
        self.owner: list[int] = []
        self._offset: list[int] = []
        self._verified: list[bool] = []  # pre-registered entries are checked on first visit
        self._n_slots = 0
        self._R = np.zeros(64)
        self._S = np.zeros(64)
        self.iteration = 0
        self.traversals = 0
        self.nodes = 0
        self.seconds = 0.0
        if known_infosets is not None:
            for key, (p, acts) in known_infosets.items():
                self._register(key, p, tuple(acts), verified=False)

    def reseed(self, seed: int) -> None:
        """Restart every random stream from ``seed``, exactly as the constructor seeds them.

        The tables, counters and options are untouched. After ``reseed(s)``
        the solver draws the same numbers as a fresh ``MCCFRSolver(..., seed=s)``
        would, so a long-lived solver can run independent, reproducible
        batches (the parallel trainer in ``examples/train_holdem.py`` does
        this once per round per worker).
        """
        self.seed = seed
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._chance_rng = (np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1])))
                            if self._sample_chance is not None else None)
        self._buf: list[float] = []
        self._pos = 0

    # ------------------------------------------------------------ tables

    def _register(self, key: InfosetKey, player: int, acts: tuple[Action, ...],
                  verified: bool = True) -> int:
        if key in self._index:
            raise ValueError(f"info set {key!r} registered twice")
        k = len(self.keys)
        self._verified.append(verified)
        self._index[key] = k
        self.keys.append(key)
        self.actions.append(acts)
        self.owner.append(player)
        self._offset.append(self._n_slots)
        self._n_slots += len(acts)
        if self._n_slots > len(self._R):
            cap = max(2 * len(self._R), self._n_slots)
            self._R = np.concatenate([self._R, np.zeros(cap - len(self._R))])
            self._S = np.concatenate([self._S, np.zeros(cap - len(self._S))])
        return k

    @property
    def regrets(self) -> np.ndarray:
        """Cumulative (weighted) regrets, one slot per (info set, action). A view."""
        return self._R[:self._n_slots]

    @property
    def strategy_sum(self) -> np.ndarray:
        """Average-strategy accumulator, aligned with :attr:`regrets`. A view."""
        return self._S[:self._n_slots]

    @property
    def offsets(self) -> np.ndarray:
        """First slot of every info set (index order = :attr:`keys`)."""
        return np.array(self._offset, dtype=np.int64)

    def tables(self) -> tuple[dict[InfosetKey, list[float]], dict[InfosetKey, list[float]]]:
        """``(regrets, strategy_sum)`` as ``{key: [per action]}`` (order-free comparison)."""
        r, s = {}, {}
        for key, acts, off in zip(self.keys, self.actions, self._offset):
            r[key] = self._R[off:off + len(acts)].tolist()
            s[key] = self._S[off:off + len(acts)].tolist()
        return r, s

    # ----------------------------------------------------------- sampling

    def _uniform(self) -> float:
        if self._pos == len(self._buf):
            self._buf = self._rng.random(_BLOCK).tolist()
            self._pos = 0
        u = self._buf[self._pos]
        self._pos += 1
        return u

    def _sample(self, probs: list[float]) -> int:
        """Index j with probability probs[j] (inverse CDF on one uniform)."""
        u = self._uniform()
        acc, last = 0.0, 0
        for j, p in enumerate(probs):
            if p > 0.0:
                acc += p
                last = j
                if u < acc:
                    return j
        return last  # u >= sum(probs) by rounding: take the last possible outcome

    # ---------------------------------------------------------- traversal

    @staticmethod
    def _match(r: list[float]) -> list[float]:
        """Regret matching on a short list (same rule as :func:`regret.regret_matching`)."""
        pos = [x if x > 0.0 else 0.0 for x in r]
        tot = sum(pos)
        if tot <= 0.0:
            return [1.0 / len(r)] * len(r)
        return [x / tot for x in pos]

    def _walk(self, h: History, i: int, w: float, prune: bool) -> float:
        """Sampled value of h to traverser i; updates tables on the way back."""
        g = self.game
        self.nodes += 1
        if g.is_terminal(h):
            u = g.utility(h)
            return u if i == 0 else -u
        p = g.player(h)
        if p == CHANCE:
            if self._sample_chance is not None:
                return self._walk(g.next(h, self._sample_chance(h, self._chance_rng)), i, w, prune)
            outs = g.chance_outcomes(h)
            j = self._sample([q for _, q in outs])
            return self._walk(g.next(h, outs[j][0]), i, w, prune)
        key = g.infoset_key(h)
        k = self._index.get(key)
        if k is None:
            k = self._register(key, p, tuple(g.actions(h)))
        elif not self._verified[k]:
            acts = tuple(g.actions(h))
            if acts != self.actions[k] or p != self.owner[k]:
                raise ValueError(
                    f"known_infosets entry for {key!r} is (player {self.owner[k]}, "
                    f"{self.actions[k]!r}) but the game has (player {p}, {acts!r})")
            self._verified[k] = True
        acts = self.actions[k]
        off = self._offset[k]
        n = len(acts)
        regrets = self._R[off:off + n].tolist()
        sigma = self._match(regrets)
        if p != i:
            S = self._S  # no recursion between this read and the writes
            for j in range(n):
                S[off + j] += w * sigma[j]
            j = self._sample(sigma)
            return self._walk(g.next(h, acts[j]), i, w, prune)
        vals = [0.0] * n
        explored = [True] * n
        thr = self.prune_threshold
        for j in range(n):
            child = g.next(h, acts[j])
            if prune and sigma[j] == 0.0 and regrets[j] < thr and not g.is_terminal(child):
                explored[j] = False
                continue
            vals[j] = self._walk(child, i, w, prune)
        node = sum(s * v for s, v in zip(sigma, vals))
        R = self._R  # read *after* the recursion, which may have grown (replaced) _R
        for j in range(n):
            if explored[j]:
                R[off + j] += w * (vals[j] - node)  # increments, never stale overwrites
        return node

    def weight(self, t: int) -> float:
        """Iteration t's weight on regret and average increments."""
        if not self.linear:
            return 1.0
        return float(t if self.linear_until is None else min(t, self.linear_until))

    def traverse(self, traverser: int, weight: float = 1.0) -> float:
        """One external-sampling traversal for ``traverser`` from the root.

        Updates ``traverser``'s regrets and the opponent's average-strategy
        accumulator, each increment multiplied by ``weight``. Returns the
        sampled value v~(root) to the traverser (an unbiased estimate of its
        value under the current strategies). :meth:`step` calls this twice.
        """
        if traverser not in (0, 1):
            raise ValueError(f"traverser must be 0 or 1, got {traverser}")
        prune = (self.prune_threshold is not None and self.iteration > self.prune_after
                 and self._uniform() < self.prune_prob)
        self.traversals += 1
        return self._walk(self.game.initial(), traverser, weight, prune)

    def step(self) -> None:
        """One iteration: a traversal for player 0, then one for player 1."""
        start = time.perf_counter()
        self.iteration += 1
        w = self.weight(self.iteration)
        self.traverse(0, w)
        self.traverse(1, w)
        self.seconds += time.perf_counter() - start

    def run(self, iterations: int, *, eval_every: int | None = None,
            eval_at: Iterable[int] = (),
            callback: Callable[[MCCFRCheckpoint], None] | None = None,
            max_seconds: float | None = None) -> list[MCCFRCheckpoint]:
        """Run up to ``iterations`` more iterations; evaluate exactly at checkpoints.

        Same contract as :meth:`CFRSolver.run`: a checkpoint is taken when the
        absolute iteration count is a multiple of ``eval_every`` or is in
        ``eval_at``. ``max_seconds`` stops early once solver time (evaluation
        excluded) exceeds it. Exact evaluation enumerates the whole tree, so
        only use checkpoints on small games.
        """
        at = set(eval_at)
        out: list[MCCFRCheckpoint] = []
        for _ in range(iterations):
            if max_seconds is not None and self.seconds >= max_seconds:
                break
            self.step()
            t = self.iteration
            if t in at or (eval_every and t % eval_every == 0):
                cp = self.evaluate()
                out.append(cp)
                if callback is not None:
                    callback(cp)
        return out

    # ------------------------------------------------------------ outputs

    def _profile(self, table: np.ndarray) -> dict[InfosetKey, dict[Action, float]]:
        out = {}
        for key, acts, off in zip(self.keys, self.actions, self._offset):
            row = table[off:off + len(acts)].tolist()
            tot = sum(x for x in row if x > 0.0)
            if tot > 0.0:
                out[key] = {a: (x / tot if x > 0.0 else 0.0) for a, x in zip(acts, row)}
            else:
                out[key] = {a: 1.0 / len(acts) for a in acts}
        return out

    def average_strategy(self) -> dict[InfosetKey, dict[Action, float]]:
        """Normalised average strategy (the MCCFR output) as a profile.

        Info sets with a zero accumulator, and info sets never discovered
        (absent from the profile), are played uniformly, the same
        convention as :meth:`CFRSolver.average_strategy`.
        """
        return self._profile(self.strategy_sum)

    def current_strategy(self) -> dict[InfosetKey, dict[Action, float]]:
        """The regret-matching strategy of the next iteration."""
        return self._profile(self.regrets)

    def evaluate(self) -> MCCFRCheckpoint:
        """Exact exploitability, BR values and value of the average strategy."""
        prof = self.average_strategy()
        br0 = best_response_value(self.game, prof, 0)
        br1 = best_response_value(self.game, prof, 1)
        return MCCFRCheckpoint(self.iteration, (br0 + br1) / 2.0, br0, br1,
                               expected_value(self.game, prof), self.seconds,
                               nodes=self.nodes, traversals=self.traversals,
                               infosets=len(self.keys))

    def options(self) -> dict[str, object]:
        """Constructor options (for logging next to results)."""
        return {"seed": self.seed, "linear": self.linear, "linear_until": self.linear_until,
                "prune_threshold": self.prune_threshold, "prune_prob": self.prune_prob,
                "prune_after": self.prune_after}


__all__ = ["MCCFRCheckpoint", "MCCFRSolver"]
