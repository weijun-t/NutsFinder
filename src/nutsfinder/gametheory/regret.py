"""Regret matching and regret-matching+ (the per-info-set learner inside CFR).

The setting
-----------
Repeatedly pick a mixed strategy sigma^t over actions a_1..a_n; afterwards
you learn the payoff u^t(a) every action *would* have earned. The
instantaneous regret of a is how much better a was than what you played:

    r^t(a) = u^t(a) - sum_b sigma^t(b) u^t(b).

The cumulative regret is R^T(a) = sum_{t <= T} r^t(a).

Regret matching (Hart & Mas-Colell 2000)
----------------------------------------
Play each action in proportion to its *positive* cumulative regret:

    sigma^{T+1}(a) = max(R^T(a), 0) / sum_b max(R^T(b), 0),

and if no action has positive regret, play uniformly. Worked example:
R = [2, -1, 1] -> positive parts [2, 0, 1] -> sigma = [2/3, 0, 1/3]. With
R = [-3, 0] there is no positive regret -> [1/2, 1/2].

Guarantee: the *average* regret max_a R^T(a) / T is at most
Delta * sqrt(n) / sqrt(T) (Delta = payoff range), so it tends to 0. CFR runs
one regret matcher per information set.

Regret-matching+ (Tammelin 2014)
--------------------------------
Same strategy rule, but the cumulative regret is floored at zero *after
every update*:

    R^{T}(a) = max(R^{T-1}(a) + r^T(a), 0).

Why it helps: plain RM remembers every bad round as a debt that must be
repaid before an action is played again. RM+ forgets the debt. Worked
example, two actions, start R = [3, 0]:

    round 1, r = [-5, +2]   RM:  R = [-2, 2] -> sigma = [0, 1]
                            RM+: R = [ 0, 2] -> sigma = [0, 1]
    round 2, r = [+3, -1]   RM:  R = [ 1, 1] -> sigma = [1/2, 1/2]
                            RM+: R = [ 3, 1] -> sigma = [3/4, 1/4]

Round 2 favoured action 1. RM+ responds at once (3/4 on it), while RM is
still held back by the -2 debt it carried out of round 1. In CFR this makes
the current strategy track the opponent's changes faster; the price is that
RM+ is worse under sampling noise (a noisy negative sample is forgotten,
a noisy positive one is not).

Both scalar (list) versions, for reading and testing, and a vectorised
version over a flat array of many info sets (what the solver uses) are
provided. All are deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def regret_matching(regrets: Sequence[float]) -> list[float]:
    """Strategy proportional to the positive part of ``regrets``; uniform if none is positive.

    >>> regret_matching([2.0, -1.0, 1.0])
    [0.6666666666666666, 0.0, 0.3333333333333333]
    >>> regret_matching([-3.0, 0.0])
    [0.5, 0.5]
    """
    if not regrets:
        raise ValueError("regret_matching needs at least one action")
    pos = [r if r > 0.0 else 0.0 for r in regrets]
    total = sum(pos)
    if total <= 0.0:
        return [1.0 / len(pos)] * len(pos)
    return [p / total for p in pos]


def regret_matching_plus_update(cumulative: Sequence[float],
                                instant: Sequence[float]) -> list[float]:
    """RM+ cumulative-regret update: ``max(R + r, 0)`` elementwise.

    >>> regret_matching_plus_update([3.0, 0.0], [-5.0, 2.0])
    [0.0, 2.0]
    """
    if len(cumulative) != len(instant):
        raise ValueError("cumulative and instant regrets differ in length")
    return [max(c + r, 0.0) for c, r in zip(cumulative, instant)]


def regret_matching_flat(regrets: np.ndarray, segment: np.ndarray,
                         segment_size: np.ndarray) -> np.ndarray:
    """Regret matching for many info sets at once.

    ``regrets`` is a flat array of slots (one slot per (info set, action));
    ``segment[k]`` is the info-set index of slot k and ``segment_size[k]``
    the number of actions of that info set. Each info set's slots are
    normalised independently; an info set with no positive regret is
    uniform. Same result as :func:`regret_matching` applied per info set.
    """
    pos = np.maximum(regrets, 0.0)
    totals = np.bincount(segment, weights=pos)[segment]
    out = np.empty_like(pos)
    np.divide(pos, totals, out=out, where=totals > 0.0)
    flat = totals <= 0.0
    out[flat] = 1.0 / segment_size[flat]
    return out
