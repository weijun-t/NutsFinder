"""The trained hold'em strategy ("blueprint") on disk and in memory.

One file format serves both the trainer (``examples/train_holdem.py``
checkpoints, resumable) and the bot (:class:`~nutsfinder.bots.cfr_bot.CFRBot`):
an ``.npz`` with

- ``regrets``, ``strategy_sum``: float64, one slot per (info set, action),
  in the fixed layout of :func:`slot_offsets`;
- ``meta``: a JSON string (0-d unicode array, so loading never needs
  pickle) with the :class:`AbstractionConfig`, the iteration count, the
  seed, the commit and the training options, plus an ``abstraction``
  fingerprint (see :func:`abstraction_fingerprint`).

Fingerprint. The layout is a function of the config *and of the code* that
builds the tree and the buckets. If ``build_tree``/``abstract_actions`` or
the bucket edges in ``bucket_data.py`` change while the config stays the
same, the slot count can still match and an old strategy would silently be
paired with a new tree or new buckets. Every checkpoint therefore stores a
SHA-256 of the tree arrays, the edge size labels and the bucket edges, and
:class:`Blueprint` and the trainer's resume refuse a mismatch. Checkpoints
written before the fingerprint existed load with a warning.

Table layout. Info-set keys of :class:`~nutsfinder.abstraction.holdem_game.AbstractHoldem`
are the ints ``0 .. tree.n_infosets - 1`` (``tree.infoset(node, bucket)``).
The trainer pre-registers every one of them with
:class:`~nutsfinder.gametheory.mccfr.MCCFRSolver` *in key order*
(:func:`known_infosets`), so key k's actions occupy slots
``offset[k] : offset[k] + n_actions(k)`` with ``offset`` the running sum of
action counts. The layout is a pure function of the config, which is why a
checkpoint stores only the config and not the tree.

The average strategy. CFR's output is the normalised ``strategy_sum`` per
info set; an info set whose sum is 0 (never reached by the owner's
sampling) is played uniformly, the convention of
:meth:`MCCFRSolver.average_strategy`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from nutsfinder.abstraction.config import DEFAULT_CONFIG, AbstractionConfig
from nutsfinder.abstraction.tree import BettingTree, build_tree

__all__ = [
    "FORMAT", "Blueprint", "abstraction_fingerprint", "check_fingerprint", "config_from_dict", "config_to_dict", "known_infosets",
    "load_tables", "save_tables", "slot_offsets", "tree_for",
]

FORMAT = 1


def config_to_dict(config: AbstractionConfig) -> dict[str, Any]:
    """The config as plain JSON-able data (tuples become lists)."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(config).items()}


def config_from_dict(d: dict[str, Any]) -> AbstractionConfig:
    """Inverse of :func:`config_to_dict`."""
    fields = {f.name for f in dataclasses.fields(AbstractionConfig)}
    unknown = set(d) - fields
    if unknown:
        raise ValueError(f"unknown config fields {sorted(unknown)}")
    return AbstractionConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


@lru_cache(maxsize=4)
def tree_for(config: AbstractionConfig) -> BettingTree:
    """The betting tree of ``config``, built once per process (it is never mutated)."""
    return build_tree(config)


@lru_cache(maxsize=4)
def abstraction_fingerprint(config: AbstractionConfig) -> str:
    """SHA-256 (hex) of everything that fixes the meaning of a table slot.

    Covers the tree's integer arrays (shape and values), each edge's size
    labels, and the postflop bucket edges actually used by
    :class:`~nutsfinder.abstraction.cards.CardAbstraction`. Equal
    fingerprints mean slot k is the same (info set, action) in both.
    """
    from nutsfinder.abstraction.bucket_data import EDGES  # the edges cards.py uses

    tree = tree_for(config)
    h = hashlib.sha256()
    for name, arr in sorted(tree.as_arrays().items()):
        a = np.ascontiguousarray(arr, dtype=np.int64)
        h.update(f"{name}:{a.shape}".encode())
        h.update(a.tobytes())
    h.update(repr(tree.edge_sizes).encode())
    for street in sorted(EDGES):
        h.update(f"edges{street}:{len(EDGES[street])}".encode())
        h.update(np.asarray(EDGES[street], dtype=np.float64).tobytes())
    return h.hexdigest()


def check_fingerprint(meta: dict[str, Any], config: AbstractionConfig, where: str = "checkpoint") -> None:
    """Raise if ``meta`` was written for a different tree/buckets; warn if it predates fingerprints."""
    stored = meta.get("abstraction")
    if stored is None:
        if not meta.get("untrained"):
            warnings.warn(f"{where} has no abstraction fingerprint (written before it existed); "
                          "cannot verify that it matches this tree and these buckets", stacklevel=3)
        return
    current = abstraction_fingerprint(config)
    if stored != current:
        raise ValueError(f"{where} was trained on a different abstraction (fingerprint {stored[:12]}..., "
                         f"this code builds {current[:12]}...): the tree or bucket edges changed")


def known_infosets(tree: BettingTree) -> dict[int, tuple[int, tuple[int, ...]]]:
    """``key -> (player, action ids)`` for every info set, in key order (for ``MCCFRSolver``)."""
    out: dict[int, tuple[int, tuple[int, ...]]] = {}
    for n in sorted((n for n in range(tree.n_nodes) if tree.infoset_base[n] >= 0),
                    key=lambda n: tree.infoset_base[n]):
        acts = tuple(range(tree.n_children[n]))
        for b in range(tree.config.buckets(tree.street[n])):
            out[tree.infoset_base[n] + b] = (tree.player[n], acts)
    assert list(out) == list(range(tree.n_infosets))
    return out


def slot_offsets(tree: BettingTree) -> tuple[np.ndarray, np.ndarray]:
    """``(offset, n_actions)`` per info-set key (int64 arrays) in the pre-registered layout."""
    n_actions = np.array([len(a) for _, a in known_infosets(tree).values()], dtype=np.int64)
    offset = np.zeros(len(n_actions), dtype=np.int64)
    np.cumsum(n_actions[:-1], out=offset[1:])
    return offset, n_actions


def save_tables(path: Path, regrets: np.ndarray, strategy_sum: np.ndarray, meta: dict[str, Any]) -> None:
    """Write a checkpoint atomically (temp file, then rename: a crash never leaves half a file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"format": FORMAT, **meta}
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, regrets=np.asarray(regrets, dtype=np.float64),
                 strategy_sum=np.asarray(strategy_sum, dtype=np.float64),
                 meta=np.array(json.dumps(meta, sort_keys=True)))
    os.replace(tmp, path)


def load_tables(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """``(regrets, strategy_sum, meta)`` from a checkpoint written by :func:`save_tables`."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: checkpoint format {meta.get('format')!r}, expected {FORMAT}")
        return data["regrets"].copy(), data["strategy_sum"].copy(), meta


class Blueprint:
    """A normalised average strategy on the abstract tree, ready for lookups.

    ``strategy(node, bucket)`` is the probability vector over
    ``tree.actions(node)``. Build one with :meth:`load` (a checkpoint
    file), :meth:`from_tables` (arrays in memory) or :meth:`uniform` (no
    training: every info set uniform, the solver's iteration-0 output).
    The arrays are read-only, so one instance can be shared by many bots.
    """

    def __init__(self, config: AbstractionConfig, strategy_sum: np.ndarray | None,
                 regrets: np.ndarray | None = None, meta: dict[str, Any] | None = None) -> None:
        self.config = config
        self.tree = tree_for(config)
        self.offset, self.n_actions = slot_offsets(self.tree)
        n_slots = int(self.n_actions.sum())
        self.meta = dict(meta or {})
        if meta is not None:
            check_fingerprint(self.meta, config, "blueprint")
        idx = np.repeat(np.arange(len(self.n_actions)), self.n_actions)
        if strategy_sum is None:
            strategy_sum = np.zeros(n_slots)
        for name, arr in (("strategy_sum", strategy_sum), ("regrets", regrets)):
            if arr is not None and arr.shape != (n_slots,):
                raise ValueError(f"{name} has shape {arr.shape}, the tree of this config needs ({n_slots},)")
        pos = np.where(strategy_sum > 0.0, strategy_sum, 0.0)
        tot = np.bincount(idx, weights=pos, minlength=len(self.n_actions))
        reached = tot > 0.0
        sigma = np.where(reached[idx], pos / np.where(reached, tot, 1.0)[idx], 1.0 / self.n_actions[idx])
        sigma.setflags(write=False)
        self.sigma = sigma
        self.reached = reached
        self.regrets = None
        if regrets is not None:
            # Average (per unit of iteration weight) counterfactual regret, in chips.
            w = float(self.meta.get("weight_sum") or 0.0)
            r = regrets / w if w > 0 else regrets.copy()
            r.setflags(write=False)
            self.regrets = r

    @classmethod
    def from_tables(cls, config: AbstractionConfig, regrets: np.ndarray | None,
                    strategy_sum: np.ndarray, meta: dict[str, Any] | None = None) -> Blueprint:
        return cls(config, strategy_sum, regrets, meta)

    @classmethod
    def uniform(cls, config: AbstractionConfig = DEFAULT_CONFIG) -> Blueprint:
        """The untrained blueprint: every info set plays uniformly."""
        return cls(config, None, None, {"iteration": 0, "untrained": True})

    @classmethod
    def load(cls, path: Path | str) -> Blueprint:
        """Load a checkpoint (cached per path and modification time)."""
        p = Path(path).resolve()
        st = p.stat()
        return _load_cached(str(p), st.st_mtime_ns, st.st_size)

    @property
    def iteration(self) -> int:
        return int(self.meta.get("iteration", 0))

    def slots(self, node: int, bucket: int) -> slice:
        k = self.tree.infoset(node, bucket)
        off = int(self.offset[k])
        return slice(off, off + int(self.n_actions[k]))

    def strategy(self, node: int, bucket: int) -> np.ndarray:
        """Average-strategy probabilities over ``tree.actions(node)`` (sums to 1)."""
        return self.sigma[self.slots(node, bucket)]

    def was_reached(self, node: int, bucket: int) -> bool:
        """Whether training ever accumulated strategy mass here (else it is the uniform default)."""
        return bool(self.reached[self.tree.infoset(node, bucket)])

    def regret(self, node: int, bucket: int) -> np.ndarray | None:
        """Average cumulative regret per action (chips per unit weight), if the file had regrets."""
        return None if self.regrets is None else self.regrets[self.slots(node, bucket)]

    def describe(self) -> str:
        if self.meta.get("untrained"):
            return "untrained (uniform)"
        return f"{self.meta.get('path', 'tables')} @ {self.iteration:,} it"


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime_ns: int, size: int) -> Blueprint:
    regrets, strategy_sum, meta = load_tables(Path(path))
    config = config_from_dict(meta["config"])
    return Blueprint(config, strategy_sum, regrets, {**meta, "path": Path(path).name})
