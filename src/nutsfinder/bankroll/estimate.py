"""Per-hand mu and sigma of a matchup from session data, with cluster-bootstrap CIs.

The data: the official session matches (``results/*_sessions.json``) store
``hand_results_bb``: for every session, A's net in bb for each of its
``2 * pairs`` hands in play order (game 1 and game 2 of pair 0, then pair 1, ...).

Three variances, and why they differ
------------------------------------
- ``var_hand``: the plain variance of single hands, pooled over everything.
  It is what you would plug into the formulas if hands were i.i.d.
- ``var_dup``: Var(duplicate session total) / hands per session. In a
  duplicate session, game 2 replays game 1's cards with the seats swapped,
  so the card luck of the two games cancels (strongly negative
  covariance). This is the right variance for *measuring* an edge, and it
  is far smaller than ``var_hand``. It is the WRONG variance for a bankroll:
  a real player sits in one seat and gets no mirror hand to cancel the luck.
- ``var_room``: Var(room-session total) / hands per room. Each session is
  played in two "rooms" (separate bot instances); room 1's 250 hands (the
  even positions) are a realistic continuous match of A against a fresh
  B, and so are room 2's. Rooms are independent across sessions (fresh bots,
  fresh seeds), so the room total is the i.i.d. block a bankroll model needs.
  ``var_room / var_hand`` measures dependence *within* a session (learning,
  streaks): 1 if the hands of a room were independent.

So the diffusion's sigma^2 for FIXED-STAKE ruin (and for anything resized only
between sessions, and for the fluctuation of log-wealth) is ``var_room`` (the
"session-level variance"), not ``var_hand``, and certainly not ``var_dup``.

The Kelly fraction with PER-HAND resizing is different: the drift of log W per
hand, E[log(1 + f X)] ~ f mu - f^2 E[X^2] / 2, is an expectation over single
hands, so it uses the per-hand second moment E[X^2] = var_hand + mu^2 whatever
the dependence: ``kelly_bankroll = E[X^2] / mu``. When the two variances differ
(BayesBot vs RandomBot: var_room / var_hand = 1.67) so do the two answers.

Uncertainty: every statistic gets a percentile bootstrap CI that resamples
whole SESSIONS (both rooms together: they share cards, so they are not
independent of each other), which keeps the dependence structure.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nutsfinder.bankroll.kelly import exact_kelly_fraction, lundberg_exponent

__all__ = ["SessionData", "load_session_data", "point_estimates", "bootstrap_estimates", "estimate_table_row"]


@dataclass(frozen=True)
class SessionData:
    """Per-hand results of one matchup: ``hands[s, i]`` = A's net (bb) in hand i of session s."""

    name: str
    hands: np.ndarray
    master_seed: int | None = None
    source: str = ""

    @property
    def n_sessions(self) -> int:
        return self.hands.shape[0]

    @property
    def hands_per_session(self) -> int:
        return self.hands.shape[1]

    @property
    def rooms(self) -> np.ndarray:
        """``(n_sessions, 2, hands/2)``: room 1 = game 1 of each pair (even positions), room 2 = odd."""
        return np.stack([self.hands[:, 0::2], self.hands[:, 1::2]], axis=1)

    @property
    def room_blocks(self) -> np.ndarray:
        """All room sequences as rows ``(2 * n_sessions, hands/2)``: the bankroll simulation's blocks."""
        r = self.rooms
        return r.reshape(-1, r.shape[2])

    def shifted(self, rake_bb_per_hand: float) -> SessionData:
        """The same data with a rake of ``rake_bb_per_hand`` taken from A on every hand."""
        return SessionData(self.name, self.hands - rake_bb_per_hand, self.master_seed, self.source)


def load_session_data(path: str | Path, name: str | None = None) -> SessionData:
    d = json.loads(Path(path).read_text())
    hands = np.array(d["hand_results_bb"], dtype=float)
    if hands.ndim != 2 or hands.shape[1] % 2:
        raise ValueError(f"{path}: expected sessions x (even number of) hands")
    return SessionData(name or f"{d['bot_a']} vs {d['bot_b']}", hands, d.get("master_seed"), str(path))


def _stats_from_parts(T: np.ndarray, R: np.ndarray, Q: np.ndarray, H: int) -> dict[str, np.ndarray]:
    """Vectorised statistics. ``T``: (..., n) session totals; ``R``: (..., n, 2) room totals;
    ``Q``: (..., n) per-session sum of squared hand results; ``H``: hands per session."""
    n = T.shape[-1]
    N = n * H
    mu = T.sum(-1) / N
    var_hand = (Q.sum(-1) - N * mu**2) / (N - 1)
    var_dup = T.var(-1, ddof=1) / H
    Rf = R.reshape(*R.shape[:-2], -1)
    var_room = Rf.var(-1, ddof=1) / (H // 2)
    return {"mu": mu, "var_hand": var_hand, "var_dup": var_dup, "var_room": var_room}


def _parts(data: SessionData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = data.hands
    return h.sum(1), data.rooms.sum(2), np.square(h).sum(1)


def _derived(s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mu, vr = s["mu"], s["var_room"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = mu > 0
        out = dict(s)
        out["bb100"] = 100 * mu
        out["sd_hand"] = np.sqrt(s["var_hand"])
        out["sd_dup"] = np.sqrt(s["var_dup"])
        out["sd_room"] = np.sqrt(vr)
        out["ratio_room"] = vr / s["var_hand"]
        out["ratio_dup"] = s["var_dup"] / s["var_hand"]
        ex2 = s["var_hand"] + mu**2   # per-hand second moment E[X^2]
        out["ex2_hand"] = ex2
        # Kelly with PER-HAND resizing: drift of log W = f mu - f^2 E[X^2] / 2 (a per-hand expectation,
        # unaffected by dependence between hands) -> f* = mu / E[X^2], bankroll E[X^2] / mu.
        out["kelly_bankroll"] = np.where(pos, ex2 / mu, np.inf)
        out["kelly_growth"] = np.where(pos, mu**2 / (2 * ex2), 0.0)
        # Gaussian Kelly when stakes change only BETWEEN sessions (the bet is a whole session):
        # mean and variance of a session total -> bankroll var_room / mu (the exact session-level
        # version is ``exact_kelly_bankroll_room``).
        out["kelly_bankroll_session_gauss"] = np.where(pos, vr / mu, np.inf)
        # Drawdowns under per-hand resizing: P(fall to x) = x^((2/c - 1) * E[X^2] / var_room).
        out["drawdown_exponent_scale"] = ex2 / vr
        out["bankroll_1pct"] = np.where(pos, vr * math.log(100) / (2 * mu), np.inf)
        out["bankroll_1pct_naive_hand"] = np.where(pos, s["var_hand"] * math.log(100) / (2 * mu), np.inf)
        out["bankroll_1pct_dup"] = np.where(pos, s["var_dup"] * math.log(100) / (2 * mu), np.inf)
    return out


def _inv(f: float) -> float:
    return math.inf if f == 0 else (0.0 if math.isinf(f) else 1 / f)


def point_estimates(data: SessionData) -> dict[str, float]:
    """mu (bb/hand), the three variances and everything derived from them, plus exact-Kelly / Lundberg."""
    T, R, Q = _parts(data)
    d = {k: float(v) for k, v in _derived(_stats_from_parts(T, R, Q, data.hands_per_session)).items()}
    hands = data.hands.ravel()
    blocks = data.room_blocks.sum(1)
    d["lundberg_hand"] = float(lundberg_exponent(hands))     # i.i.d. hands
    d["lundberg_room"] = float(lundberg_exponent(blocks))    # i.i.d. room sessions (per bb)
    d["diffusion_exponent"] = 2 * d["mu"] / d["var_room"] if d["mu"] > 0 else 0.0
    d["exact_kelly_bankroll_hand"] = _inv(exact_kelly_fraction(hands)) if d["mu"] > 0 else math.inf
    # Resizing only between room sessions: the bet is a whole 250-hand block.
    # NaN when no room session lost (the sample then has no finite Kelly optimum).
    d["exact_kelly_bankroll_room"] = _inv(exact_kelly_fraction(blocks)) if d["mu"] > 0 else math.inf
    if blocks.min() >= 0:
        d["exact_kelly_bankroll_room"] = math.nan
        d["lundberg_room"] = math.nan
    d["worst_room_bb"] = float(blocks.min())
    d["worst_room_drawdown_bb"] = float(np.cumsum(data.room_blocks, axis=1).min())
    d["n_sessions"] = data.n_sessions
    d["hands"] = int(data.hands.size)
    d["se_mu"] = float(np.sqrt(d["var_dup"] / data.hands.size))  # SE of mu: duplicate session totals
    return d


def bootstrap_estimates(data: SessionData, rng: np.random.Generator, n_boot: int = 4000,
                        keys: tuple[str, ...] | None = None, level: float = 0.95) -> dict[str, tuple[float, float]]:
    """Percentile CIs, resampling whole sessions (both rooms together) ``n_boot`` times."""
    T, R, Q = _parts(data)
    n = data.n_sessions
    out: dict[str, list[np.ndarray]] = {}
    chunk = 1000
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        d = _derived(_stats_from_parts(T[idx], R[idx], Q[idx], data.hands_per_session))
        for k, v in d.items():
            out.setdefault(k, []).append(v)
    a = (1 - level) / 2
    res = {}
    for k, vs in out.items():
        if keys is not None and k not in keys:
            continue
        v = np.concatenate(vs)
        res[k] = (float(np.quantile(v, a)), float(np.quantile(v, 1 - a)))
    return res


def bootstrap_draws(data: SessionData, rng: np.random.Generator, n_boot: int = 4000) -> dict[str, np.ndarray]:
    """The raw bootstrap replicates of mu and var_room (for propagating uncertainty into curves)."""
    T, R, Q = _parts(data)
    idx = rng.integers(0, data.n_sessions, size=(n_boot, data.n_sessions))
    return _stats_from_parts(T[idx], R[idx], Q[idx], data.hands_per_session)


def estimate_table_row(data: SessionData, rng: np.random.Generator, n_boot: int = 4000) -> dict:
    """Point estimates and CIs of one matchup, JSON-friendly."""
    p = point_estimates(data)
    ci = bootstrap_estimates(data, rng, n_boot)
    return {"matchup": data.name, "source": data.source, "master_seed": data.master_seed,
            "point": p, "ci95": {k: list(v) for k, v in ci.items()}}
