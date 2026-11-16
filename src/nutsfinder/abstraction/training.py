"""Parallel external-sampling MCCFR for the abstract hold'em game.

Python has a GIL, so the solver is parallelised with *processes*. The
scheme (docs/phase6_research.md section 3, "Parallelism under the GIL"):

1. The master holds the tables ``R`` (cumulative regrets) and ``S``
   (strategy sums), in the fixed layout of
   :func:`~nutsfinder.abstraction.blueprint.slot_offsets` (every one of the
   57,304 default info sets is pre-registered, so slot indices are the same
   in every process).
2. A **round**: the master sends the same snapshot of ``R`` to ``W``
   workers. Worker ``w`` runs ``B`` ordinary ES-MCCFR iterations starting
   from that snapshot (its own regrets keep updating within its batch;
   ``S`` starts at 0) and returns the deltas ``dR = R_w - R``, ``dS = S_w``.
3. The master adds the deltas **in worker order** (so floating-point sums
   do not depend on which process finished first) and the next round
   starts from the new tables.

Seeds. Worker w in round r reseeds its solver with
:func:`round_seed` = SHA-256 of ``(master seed, r, w)`` (the hashing
recipe of :mod:`nutsfinder.engine.preflop`), so a run is a pure function of
``(seed, workers, batch, config, iterations)``: the OS scheduling and the
number of physical processes play no part, the in-process runner
(``pool=None``) gives bit-identical tables, and a run resumed from a
checkpoint equals the uninterrupted run (tested).

Linear weighting by *global* iteration. Round r covers the global
iterations ``t0 + 1 .. t0 + W*B`` (``t0 = r * W * B``); worker w gets
``t0 + w*B + 1 .. t0 + (w+1)*B`` and weights iteration t by t, exactly the
weights a serial run would use for the same iteration numbers. So the
weighted sums ``R`` and ``S`` are the same estimators as in a serial run;
only *which strategy each iteration samples against* differs (next).

The stale-snapshot approximation
--------------------------------
In serial MCCFR iteration t samples against sigma^t, the regret-matching
strategy after iterations 1..t-1. Here an iteration in worker w's batch
sees the snapshot plus **its own worker's** earlier updates of the round,
but not the other ``W - 1`` workers' updates from the same round: its
strategy is up to ``(W - 1) * B`` iterations stale (on top of the usual
within-batch lag). Every update is still a valid sampled regret for the
strategy it was computed against, so this is CFR with delayed regret
updates; I know of no formal convergence guarantee for it, but it is the
standard way to parallelise MCCFR (and harmless once ``W * B`` is small
next to the iteration count ``t``: regret matching changes the strategy
by roughly ``W * B / t`` per round late in training). Two consequences:

- The very first round plays ``W * B`` iterations against the uniform
  strategy. Linear weighting gives those iterations tiny weights
  (weight t vs t up to millions), so they wash out.
- A 1-worker run is *exactly* serial MCCFR reseeded every ``B``
  iterations (no staleness at all); a test checks the delta-summing
  against that serial reference.

What is logged (see :class:`RoundStats` and :func:`convergence_proxies`)
-------------------------------------------------------------------------
Iterations/s, nodes per iteration, info sets touched, and two cheap
convergence proxies chosen because they cost one pass over the tables:

- **avg-strategy change**: total-variation distance between the
  normalised average strategy at this checkpoint and the previous one,
  per info set, averaged with weights proportional to each info set's
  strategy mass (a proxy for how often it is reached), plus the
  unweighted mean over reached info sets. It goes to 0 as the average
  strategy settles (a necessary, not sufficient, sign of convergence).
- **regret proxy**: ``(Rbar_0 + Rbar_1) / 2`` in mbb/hand, with
  ``Rbar_p = sum over p's info sets of max_a R+(I, a) / sum of weights``.
  With perfect recall and exact regrets that bounds the exploitability
  of the average strategy (the CFR theorem). Here regrets are *sampled*
  (the max of noisy positives is biased upward) and recall is imperfect,
  so it is only a trend line, not a bound.

Not used: a head-to-head match against EquityBot at each checkpoint costs
minutes per few hundred pairs and its SE (tens of bb/100) is far larger
than the change between checkpoints; the in-abstraction best response
needs a full (cards x betting) traversal, which is step 8's job, not a
per-checkpoint cost.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nutsfinder.abstraction.blueprint import (
    abstraction_fingerprint, check_fingerprint, config_from_dict, config_to_dict, known_infosets,
    load_tables, save_tables, slot_offsets, tree_for,
)
from nutsfinder.abstraction.cards import CardAbstraction, EHSTables
from nutsfinder.abstraction.config import DEFAULT_CONFIG, AbstractionConfig
from nutsfinder.abstraction.holdem_game import AbstractHoldem
from nutsfinder.gametheory.mccfr import MCCFRSolver

__all__ = [
    "TrainOptions", "Trainer", "RoundStats", "convergence_proxies", "round_seed", "weight_sum",
    "init_worker", "run_batch",
]


def round_seed(master_seed: int, rnd: int, worker: int) -> int:
    """64-bit seed of worker ``worker`` in round ``rnd`` (hashed: streams never overlap)."""
    key = f"nutsfinder/train_holdem|{master_seed}|{rnd}|{worker}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def weight_sum(iterations: int, linear: bool, linear_until: int | None) -> float:
    """Sum of the iteration weights ``w(1) + ... + w(T)`` (see :meth:`MCCFRSolver.weight`)."""
    t = iterations
    if not linear:
        return float(t)
    if linear_until is None or t <= linear_until:
        return t * (t + 1) / 2.0
    cap = linear_until
    return cap * (cap + 1) / 2.0 + (t - cap) * float(cap)


@dataclass(frozen=True)
class TrainOptions:
    """Everything that determines the tables (plus the number of workers and the batch)."""

    seed: int = 0
    workers: int = 22
    batch: int = 500
    linear: bool = True
    linear_until: int | None = None
    prune_threshold: float | None = None
    prune_prob: float = 0.95
    prune_after: int = 0
    config: AbstractionConfig = DEFAULT_CONFIG

    def __post_init__(self) -> None:
        if self.workers < 1 or self.batch < 1:
            raise ValueError("workers and batch must be >= 1")

    @property
    def per_round(self) -> int:
        return self.workers * self.batch

    def as_meta(self) -> dict[str, Any]:
        return {"seed": self.seed, "workers": self.workers, "batch": self.batch, "linear": self.linear,
                "linear_until": self.linear_until, "prune_threshold": self.prune_threshold,
                "prune_prob": self.prune_prob, "prune_after": self.prune_after,
                "config": config_to_dict(self.config)}

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> TrainOptions:
        return cls(seed=meta["seed"], workers=meta["workers"], batch=meta["batch"], linear=meta["linear"],
                   linear_until=meta["linear_until"], prune_threshold=meta["prune_threshold"],
                   prune_prob=meta["prune_prob"], prune_after=meta["prune_after"],
                   config=config_from_dict(meta["config"]))


# --- worker side --------------------------------------------------------------------

_SOLVER: MCCFRSolver | None = None


def make_game(config: AbstractionConfig, turn_tables: str | Path | None) -> AbstractHoldem:
    """The game with (optionally) every turn table preloaded: warm MCCFR is ~25x faster than cold."""
    tables = EHSTables()
    if turn_tables is not None:
        tables.load_turns(Path(turn_tables))
    return AbstractHoldem(config, cards=CardAbstraction(config, tables=tables), tree=tree_for(config))


def init_worker(opts: TrainOptions, turn_tables: str | Path | None) -> None:
    """Pool initializer: build the game and a solver with every info set pre-registered."""
    global _SOLVER
    game = make_game(opts.config, turn_tables)
    _SOLVER = MCCFRSolver(game, seed=0, linear=opts.linear, linear_until=opts.linear_until,
                          prune_threshold=opts.prune_threshold, prune_prob=opts.prune_prob,
                          prune_after=opts.prune_after, known_infosets=known_infosets(game.tree))


def run_batch(task: tuple[int, int, int, int, int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, int, float]:
    """One worker's share of a round: ``(dR, dS, nodes visited, seconds)``.

    ``task = (master_seed, round, worker, first_iteration, n_iterations, R_snapshot)``;
    global iterations ``first_iteration + 1 .. first_iteration + n_iterations``.
    Every piece of solver state that affects the result (RNG streams,
    regrets, strategy sums, iteration counter) is reset here, so the result
    does not depend on what this process ran before.
    """
    master_seed, rnd, worker, first, n, snapshot = task
    s = _SOLVER
    if s is None:
        raise RuntimeError("init_worker was not called in this process")
    start = time.perf_counter()
    s.reseed(round_seed(master_seed, rnd, worker))
    s.regrets[:] = snapshot
    s.strategy_sum[:] = 0.0
    s.nodes = 0
    s.iteration = first
    for _ in range(n):
        s.step()
    d_r = s.regrets - snapshot
    d_s = s.strategy_sum.copy()
    return d_r, d_s, s.nodes, time.perf_counter() - start


# --- master side --------------------------------------------------------------------


@dataclass
class RoundStats:
    """Totals accumulated between two checkpoints."""

    iterations: int = 0
    nodes: int = 0
    wall: float = 0.0
    worker_seconds: float = 0.0
    touched: np.ndarray | None = None  # info sets whose S changed

    def as_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "wall_s": round(self.wall, 3),
            "it_per_s": round(self.iterations / self.wall, 1) if self.wall else None,
            "it_per_s_per_worker": round(self.iterations / self.worker_seconds, 1) if self.worker_seconds else None,
            "nodes_per_it": round(self.nodes / self.iterations, 2) if self.iterations else None,
            "infosets_touched": int(self.touched.sum()) if self.touched is not None else 0,
        }


def normalised(strategy_sum: np.ndarray, offset: np.ndarray, n_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(sigma per slot, strategy mass per info set)``: the average strategy, uniform where unreached."""
    idx = np.repeat(np.arange(len(n_actions)), n_actions)
    pos = np.maximum(strategy_sum, 0.0)
    tot = np.bincount(idx, weights=pos, minlength=len(n_actions))
    safe = np.where(tot > 0, tot, 1.0)
    sigma = np.where((tot > 0)[idx], pos / safe[idx], 1.0 / n_actions[idx])
    return sigma, tot


def convergence_proxies(regrets: np.ndarray, strategy_sum: np.ndarray, prev_sigma: np.ndarray | None,
                        offset: np.ndarray, n_actions: np.ndarray, owner: np.ndarray, wsum: float,
                        big_blind: int) -> tuple[dict[str, float], np.ndarray]:
    """The two per-checkpoint proxies of the module doc, and the current sigma (for the next call)."""
    sigma, mass = normalised(strategy_sum, offset, n_actions)
    out: dict[str, float] = {}
    pos_max = np.maximum(np.maximum.reduceat(regrets, offset), 0.0)
    if wsum > 0:
        rbar = [float(pos_max[owner == p].sum() / wsum) for p in (0, 1)]
        out["regret_proxy_mbb"] = (rbar[0] + rbar[1]) / 2.0 / big_blind * 1000.0
    reached = mass > 0
    out["infosets_reached"] = int(reached.sum())
    if prev_sigma is not None:
        idx = np.repeat(np.arange(len(n_actions)), n_actions)
        tv = 0.5 * np.bincount(idx, weights=np.abs(sigma - prev_sigma), minlength=len(n_actions))
        if reached.any():
            out["strategy_tv_weighted"] = float((tv * mass).sum() / mass.sum())
            out["strategy_tv_mean"] = float(tv[reached].mean())
            out["strategy_tv_max"] = float(tv[reached].max())
    return out, sigma


@dataclass
class Trainer:
    """The master: owns the tables, runs rounds, writes checkpoints. See the module doc.

    ``pool`` is a ``multiprocessing`` pool whose processes ran
    :func:`init_worker` with the same options; ``None`` runs the worker
    tasks one after another in this process (same results, bit for bit).
    """

    opts: TrainOptions
    turn_tables: str | Path | None = None
    pool: Any = None
    regrets: np.ndarray = field(init=False)
    strategy_sum: np.ndarray = field(init=False)
    iteration: int = 0
    round: int = 0
    nodes: int = 0
    wall: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        tree = tree_for(self.opts.config)
        self.offset, self.n_actions = slot_offsets(tree)
        owners = np.empty(tree.n_infosets, dtype=np.int64)
        for n in range(tree.n_nodes):
            if tree.infoset_base[n] >= 0:
                b = tree.infoset_base[n]
                owners[b:b + tree.config.buckets(tree.street[n])] = tree.player[n]
        self.owner = owners
        n_slots = int(self.n_actions.sum())
        self._idx = np.repeat(np.arange(len(self.n_actions)), self.n_actions)
        self.regrets = np.zeros(n_slots)
        self.strategy_sum = np.zeros(n_slots)
        self._prev_sigma: np.ndarray | None = None
        self._segment = RoundStats(touched=np.zeros(tree.n_infosets, dtype=bool))
        self._local_ready = False

    # --- running
    def _map(self, tasks: Sequence[tuple]) -> list[tuple]:
        if self.pool is not None:
            return self.pool.map(run_batch, tasks, chunksize=1)
        if not self._local_ready:
            init_worker(self.opts, self.turn_tables)
            self._local_ready = True
        return [run_batch(t) for t in tasks]

    def run_round(self) -> None:
        o = self.opts
        start = time.perf_counter()
        snapshot = self.regrets.copy()
        tasks = [(o.seed, self.round, w, self.iteration + w * o.batch, o.batch, snapshot)
                 for w in range(o.workers)]
        results = self._map(tasks)
        seg = self._segment
        for d_r, d_s, nodes, secs in results:  # fixed order: deterministic floating-point sums
            self.regrets += d_r
            self.strategy_sum += d_s
            self.nodes += nodes
            seg.nodes += nodes
            seg.worker_seconds += secs
            seg.touched |= np.bincount(self._idx, weights=np.abs(d_s), minlength=len(self.n_actions)) > 0
        self.iteration += o.per_round
        self.round += 1
        wall = time.perf_counter() - start
        self.wall += wall
        seg.iterations += o.per_round
        seg.wall += wall

    def weight_sum(self) -> float:
        return weight_sum(self.iteration, self.opts.linear, self.opts.linear_until)

    # --- checkpoints
    def meta(self, commit: str = "unknown") -> dict[str, Any]:
        return {**self.opts.as_meta(), "iteration": self.iteration, "round": self.round, "nodes": self.nodes,
                "wall_s": self.wall, "weight_sum": self.weight_sum(), "commit": commit,
                "n_infosets": len(self.n_actions), "n_slots": int(self.n_actions.sum()),
                "abstraction": abstraction_fingerprint(self.opts.config),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def checkpoint_stats(self) -> dict[str, Any]:
        """Close the current segment: speed, touched info sets and the convergence proxies."""
        proxies, self._prev_sigma = convergence_proxies(
            self.regrets, self.strategy_sum, self._prev_sigma, self.offset, self.n_actions, self.owner,
            self.weight_sum(), self.opts.config.big_blind)
        row = {"iteration": self.iteration, "round": self.round, **self._segment.as_dict(), **proxies,
               "total_wall_s": round(self.wall, 1)}
        self._segment = RoundStats(touched=np.zeros(len(self.n_actions), dtype=bool))
        self.history.append(row)
        return row

    def save(self, path: Path, commit: str = "unknown") -> None:
        save_tables(path, self.regrets, self.strategy_sum, self.meta(commit))

    @classmethod
    def resume(cls, path: Path, turn_tables: str | Path | None = None, pool: Any = None,
               opts: TrainOptions | None = None) -> Trainer:
        """Continue from a checkpoint. ``opts`` (if given) must equal the stored options."""
        regrets, strategy_sum, meta = load_tables(path)
        stored = TrainOptions.from_meta(meta)
        check_fingerprint(meta, stored.config, str(path))
        if opts is not None and opts != stored:
            raise ValueError(f"options differ from the checkpoint's: {opts} vs {stored} "
                             "(a resumed run equals the uninterrupted one only with the same options)")
        t = cls(stored, turn_tables, pool)
        if regrets.shape != t.regrets.shape:
            raise ValueError("checkpoint tables do not match this config's tree")
        t.regrets[:] = regrets
        t.strategy_sum[:] = strategy_sum
        t.iteration, t.round, t.nodes, t.wall = meta["iteration"], meta["round"], meta["nodes"], meta["wall_s"]
        t._prev_sigma = normalised(t.strategy_sum, t.offset, t.n_actions)[0]
        return t


def rounds_for(iterations: int, opts: TrainOptions) -> int:
    """Whole rounds needed to reach at least ``iterations`` iterations."""
    return math.ceil(iterations / opts.per_round)
