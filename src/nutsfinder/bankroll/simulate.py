"""Monte Carlo bankroll paths by block bootstrap of real results (numpy, seeded, deterministic).

Why blocks
----------
Hands inside a session are not independent (a learning bot's results drift
within the session; streaks), and single hands are fat-tailed (+/-100 bb
all-ins). Resampling whole *room sessions* (250 consecutive hands one bot
played in one seat against one fresh opponent, see
:mod:`nutsfinder.bankroll.estimate`) keeps both: the within-session
dependence and the real shape of the tails. A path is a concatenation of
blocks drawn uniformly with replacement.

What is simulated
-----------------
- :func:`running_min_fixed_stake`: fixed stake. The bankroll path is
  B + S_t (S_t = cumulative winnings), and ruin from B means min_t S_t <= -B.
  So ONE simulation of the running minimum M = min_t S_t gives the whole
  risk-of-ruin curve: RoR(B) = P(M <= -B) (:func:`ror_curve`). Ruin is
  checked after every hand: each block carries its own minimum prefix sum,
  so a path is only touched once per block.
- :func:`simulate_continuous_kelly`: the stake is a fixed fraction f of the
  current bankroll, resized after every hand: W <- W (1 + f X). Tracks
  log-wealth, its running minimum (to estimate P(ever halving)) and growth.
  With a fixed f the log-path inside a block does not depend on the wealth
  level, so it uses the same per-block summaries trick (total and lowest
  prefix of log(1 + f x)).
- :func:`simulate_stake_levels`: realistic resizing: stakes are powers of
  2 (1/64, ..., 1/2, 1, 2, 4, ... of the starting stake), chosen at the start
  of each block as the largest stake s with W >= N s (N = the bankroll
  requirement in big blinds of the stake played, e.g. sigma^2 / (c mu) for
  c x Kelly). Ruin = W <= 0 at any hand.

Monte Carlo error vs data uncertainty
-------------------------------------
Binomial CIs on simulated probabilities (:func:`wilson_ci`) measure only the
*simulation* noise (finite paths). They take the data as fixed: the
uncertainty in mu and sigma themselves is handled separately (bootstrap of
the estimates in :mod:`nutsfinder.bankroll.estimate`).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["block_summaries", "running_min_fixed_stake", "ror_curve", "wilson_ci",
           "simulate_continuous_kelly", "simulate_stake_levels"]


def block_summaries(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per block: total and minimum prefix sum (the lowest point reached inside the block, <= 0 not forced)."""
    pre = np.cumsum(blocks, axis=1)
    return pre[:, -1], pre.min(axis=1)


def running_min_fixed_stake(blocks: np.ndarray, n_paths: int, horizon_blocks: int, rng: np.random.Generator,
                            chunk: int = 64, checkpoints: dict[int, np.ndarray] | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """``(M, S_T)``: each path's minimum cumulative result (hand by hand) and final result, in bb.

    ``blocks``: (n_blocks, block_len) per-hand results; a path is ``horizon_blocks`` blocks drawn
    uniformly with replacement. ``M`` includes the start (0), so M <= 0.

    ``checkpoints``: optional dict whose keys are block counts; it is filled in place with the
    running minimum of the SAME paths after that many blocks (e.g. ``{100: None}`` in a 200-block
    run gives the half-horizon minimum, for a horizon check on identical paths).
    """
    totals, minpre = block_summaries(np.asarray(blocks, dtype=float))
    cum = np.zeros(n_paths)
    m = np.zeros(n_paths)
    stops = sorted(k for k in (checkpoints or {}) if 0 < k <= horizon_blocks)
    done = 0
    while done < horizon_blocks:
        nxt = min([x for x in stops if x > done] + [horizon_blocks])
        k = min(chunk, nxt - done)
        idx = rng.integers(0, len(totals), size=(n_paths, k))
        t = totals[idx]
        ends = cum[:, None] + np.cumsum(t, axis=1)
        starts = ends - t
        m = np.minimum(m, (starts + minpre[idx]).min(axis=1))
        cum = ends[:, -1]
        done += k
        if checkpoints is not None and done in stops:
            checkpoints[done] = m.copy()
    return m, cum


def wilson_ci(k: np.ndarray | int, n: int, z: float = 1.959964) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion k / n (behaves at 0 and 1, unlike p +/- z SE)."""
    k = np.asarray(k, dtype=float)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return np.maximum(centre - half, 0.0), np.minimum(centre + half, 1.0)


def ror_curve(m: np.ndarray, bankrolls: np.ndarray) -> dict[str, np.ndarray]:
    """P(ruin) for each bankroll B = P(M <= -B), with Wilson 95% CIs."""
    bankrolls = np.asarray(bankrolls, dtype=float)
    ms = np.sort(m)
    k = np.searchsorted(ms, -bankrolls, side="right")  # count of M <= -B
    lo, hi = wilson_ci(k, len(m))
    return {"bankroll": bankrolls, "p": k / len(m), "lo": lo, "hi": hi, "k": k, "n": len(m)}


def simulate_continuous_kelly(blocks: np.ndarray, f: float, n_paths: int, horizon_blocks: int,
                              rng: np.random.Generator, record_paths: int = 0, record_every: int = 10) -> dict:
    """Bet the fraction ``f`` of the bankroll per bb, resized every hand: W <- W (1 + f X).

    Returns final log-wealth (start W = 1), the running minimum of log-wealth,
    whether a path was wiped out (1 + f X <= 0, only possible if f >= 1 / max loss),
    and optionally ``record_paths`` sample paths of log W every ``record_every`` hands.
    """
    blocks = np.asarray(blocks, dtype=float)
    nb, L = blocks.shape
    # With a fixed fraction f, log W moves by log(1 + f x) per hand whatever the wealth level, so each
    # block's log-path is fixed: precompute its total and lowest prefix once (as for the fixed stake),
    # and a simulated path only needs one step per block.
    g_all = 1 + f * blocks
    block_dead = (g_all <= 0).any(axis=1)
    lg_all = np.log(np.where(g_all > 0, g_all, 1.0))
    pre = np.cumsum(lg_all, axis=1)
    tot, low = pre[:, -1], pre.min(axis=1)
    logw = np.zeros(n_paths)
    minlog = np.zeros(n_paths)
    dead = np.zeros(n_paths, dtype=bool)
    rec = []
    for _ in range(horizon_blocks):
        idx = rng.integers(0, nb, size=n_paths)
        dead |= block_dead[idx]
        minlog = np.minimum(minlog, logw + low[idx])
        if record_paths:
            k = min(record_paths, n_paths)
            rec.append(logw[:k, None] + pre[idx[:k]][:, record_every - 1::record_every])
        logw = logw + tot[idx]
    logw = np.where(dead, -np.inf, logw)
    minlog = np.where(dead, -np.inf, minlog)
    out = {"logw": logw, "minlog": minlog, "dead": dead, "hands": horizon_blocks * L}
    if record_paths:
        out["paths"] = np.concatenate([np.zeros((min(record_paths, n_paths), 1))] + rec, axis=1)
        out["path_hands"] = np.arange(out["paths"].shape[1]) * record_every
    return out


def simulate_stake_levels(blocks: np.ndarray, w0: float, req_bb: float, n_paths: int, horizon_blocks: int,
                          rng: np.random.Generator, min_level: int = -6, factor: float = 2.0) -> dict:
    """Discrete stakes ``factor**k`` (k >= min_level), resized between blocks; start bankroll ``w0`` (bb of stake 1).

    At each block start: s = largest ``factor**k`` with W >= req_bb * s, at least ``factor**min_level``
    (below that you keep playing the smallest stake). Ruin = W <= 0 at any hand (checked
    with each block's minimum prefix). Returns ruin flags, final W, P(ever below w0/2) flags
    (checked at block boundaries and at each block's low point), the number of stake changes, and
    ``ruin_by_block``: the share of paths ruined after each block (ruin grows with the horizon).
    """
    totals, minpre = block_summaries(np.asarray(blocks, dtype=float))
    W = np.full(n_paths, float(w0))
    ruined = np.zeros(n_paths, dtype=bool)
    halved = np.zeros(n_paths, dtype=bool)
    changes = np.zeros(n_paths, dtype=np.int64)
    prev_k = None
    log_f = math.log(factor)
    ruin_by_block = np.zeros(horizon_blocks)
    for b in range(horizon_blocks):
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.floor(np.log(np.maximum(W, 1e-300) / req_bb) / log_f + 1e-12)
        k = np.maximum(k, min_level)
        if prev_k is not None:
            changes += (k != prev_k) & ~ruined
        prev_k = k
        s = factor**k
        idx = rng.integers(0, len(totals), size=n_paths)
        low = W + s * minpre[idx]
        halved |= (low <= w0 / 2) & ~ruined
        ruined |= low <= 0
        W = np.where(ruined, 0.0, W + s * totals[idx])
        ruin_by_block[b] = ruined.mean()
    return {"ruined": ruined, "W": W, "halved": halved, "changes": changes,
            "hands": horizon_blocks * blocks.shape[1], "ruin_by_block": ruin_by_block}
