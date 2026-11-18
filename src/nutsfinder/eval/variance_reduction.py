"""Unbiased variance reduction for bot-vs-bot results: all-in luck adjustment and a chance control variate.

Duplicate matching (:mod:`nutsfinder.eval.match`) cancels the luck of the
*hole cards*. What is left is mostly *board* luck: the same all-in loses in
one room and wins in the other only if the bots took different lines, and
every river card still swings whatever pot is out there. This module removes
part of that board luck without changing the expected result.

The one idea: subtract a zero-mean term
---------------------------------------
Let X be A's result in a hand (bb). Before each board card is dealt (flop,
turn, river: "chance nodes" k = 1, 2, 3) define

    e_{k-1} = A's exact equity (win + tie/2) given BOTH hole cards and the board so far,
    e_k     = the same after the card(s) are dealt,
    P_k     = the pot at that moment (both players have put in P_k / 2: a
              street only ends with the bets equal, uncalled chips are returned),
    L_k     = P_k (e_k - e_{k-1})     "the luck of chance node k", in chips.

Given everything before the deal (both hands, the board so far, all actions),
the new cards are uniform over the unseen deck (the deck is shuffled
uniformly; NOTHING here depends on what the bots believe). Equity is a
martingale under that dealing: E[e_k | before the deal] = e_{k-1} (the
equity before the deal *is* the average over the possible deals). P_k is
fixed before the deal. Hence

    E[L_k | everything before chance node k] = P_k (E[e_k | ...] - e_{k-1}) = 0,

and so, for any SUBSET of chance nodes chosen by a rule that only looks at
information available before the deal, E[X - sum L_k] = E[X]: the adjusted
result is UNBIASED, whatever the bots do. (This is the chance-node half of
AIVAT, Burch et al. 2018, with the fixed value function
V = (e - 1/2) * pot: "A's share of the pot if it were checked down now,
minus what A put in". Unbiasedness needs only that V is a fixed function of
the state and that the expectation over the deal is exact; how good V is
only affects how much variance is removed.)

When is it NOT unbiased? If the rule for including L_k peeked at the future
(e.g. "only for hands that reached showdown": whether a hand reaches
showdown depends on the cards that come later), if e were computed against
a range or with a bot's private model instead of the exact run-out
distribution given both hands, or if the expectation over the deal were
approximated (a Monte Carlo e_{k-1} would make E[L_k] a small non-zero
number). Here everything is exact enumeration.

Two estimators
--------------
- **all-in adjustment** (``allin``): only the chance nodes after betting is
  over (both players all-in, or one all-in and called, before the river).
  For those hands the telescoping sum gives exactly
  X - sum L_k = (e_j - 1/2) P: the hand is paid out at its equity at the
  moment of the all-in, the conditional expectation of the result given
  everything up to the all-in. The rule "betting is over" is known when the
  cards are dealt, so it is a legal subset.
- **chance control variate** (``chance``): every board chance node that was
  reached. It also removes board luck from hands that are still being bet,
  but V ignores future betting, so for hands that later end in a fold the
  term can ADD variance; the measurement decides whether it helps.

Hole-card luck (the deal itself) cancels exactly within a duplicate pair, so
it is not adjusted.

Everything is from the point of view of seat 0 and flipped for seat 1
(L for seat 1 = -L for seat 0, the game is zero-sum).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nutsfinder.engine.betting import Street
from nutsfinder.engine.hand_log import HandLog
from nutsfinder.eval.hu_equity import card_id, equity_hu

__all__ = ["HandLuck", "hand_luck", "adjusted_net", "ESTIMATORS", "session_adjustments", "compare_estimators"]

ESTIMATORS = ("raw", "allin", "chance")


@dataclass(frozen=True, slots=True)
class HandLuck:
    """Board luck of one hand, seat 0's view, in chips.

    Attributes:
        net0: seat 0's realised net (chips).
        streets: board streets dealt (1 = flop, 2 = turn, 3 = river), in order.
        pots: pot P_k at each of those chance nodes.
        equities: e_0 (preflop), then e_k after each dealt street.
        luck: L_k = P_k (e_k - e_{k-1}) for each dealt street.
        allin_street: if betting was over before the river (showdown with the
            last action before the river), the street of that last action;
            else ``None``.
        big_blind: chips per bb (to convert).
    """

    net0: int
    streets: tuple[int, ...]
    pots: tuple[int, ...]
    equities: tuple[float, ...]
    luck: tuple[float, ...]
    allin_street: int | None
    big_blind: int

    def adjusted0(self, estimator: str) -> float:
        """Seat 0's adjusted net in chips."""
        if estimator == "raw":
            return float(self.net0)
        if estimator == "chance":
            return self.net0 - sum(self.luck)
        if estimator == "allin":
            if self.allin_street is None:
                return float(self.net0)
            return self.net0 - sum(l for k, l in zip(self.streets, self.luck) if k > self.allin_street)
        raise ValueError(f"unknown estimator {estimator!r}; use one of {ESTIMATORS}")


def _pot_before_street(log: HandLog, street: int) -> int:
    """Pot when the cards of ``street`` are dealt: blinds + all chips of earlier streets - refunds so far."""
    put = list(log.blinds)
    for r in log.actions:
        if r.street < street:
            put[r.player] += r.chips
    for f in log.refunds:
        if f.street < street:
            put[f.player] -= f.chips
    if put[0] != put[1]:
        raise AssertionError(f"unequal contributions {put} at a chance node (seed {log.seed})")
    return put[0] + put[1]


def hand_luck(log: HandLog) -> HandLuck:
    """Equities, pots and luck terms of every board chance node the hand reached (exact enumeration)."""
    h0 = [card_id(c) for c in log.hole_cards[0]]
    h1 = [card_id(c) for c in log.hole_cards[1]]
    board = [card_id(c) for c in log.board]
    n = len(board)
    streets = tuple(k for k, size in ((1, 3), (2, 4), (3, 5)) if n >= size)
    eqs = [equity_hu(h0, h1, [])] if streets else []
    pots, luck = [], []
    for k in streets:
        size = Street(k).board_size
        e = equity_hu(h0, h1, board[:size])
        p = _pot_before_street(log, k)
        luck.append(p * (e - eqs[-1]))
        pots.append(p)
        eqs.append(e)
    allin = None
    if log.folded is None and log.actions and log.actions[-1].street < Street.RIVER:
        allin = int(log.actions[-1].street)
    return HandLuck(log.result.net[0], streets, tuple(pots), tuple(eqs), tuple(luck), allin, log.big_blind)


def adjusted_net(log: HandLog, seat: int, estimator: str) -> float:
    """``seat``'s adjusted net in bb under ``estimator`` (``raw``, ``allin`` or ``chance``)."""
    hl = hand_luck(log)
    v = hl.adjusted0(estimator)
    return (v if seat == 0 else -v) / log.big_blind


def session_adjustments(games: Sequence) -> dict[str, np.ndarray]:
    """For a session's games (:class:`nutsfinder.eval.match.Game`), A's per-hand net (bb) under each
    estimator, plus per-hand diagnostics (all-in street, end street flags)."""
    out: dict[str, list] = {e: [] for e in ESTIMATORS}
    out["allin_street"] = []
    out["showdown"] = []
    out["a_on_button"] = []
    for g in games:
        hl = hand_luck(g.log)
        sign = 1 if g.a_seat == 0 else -1
        for e in ESTIMATORS:
            out[e].append(sign * hl.adjusted0(e) / hl.big_blind)
        out["allin_street"].append(-1 if hl.allin_street is None else hl.allin_street)
        out["showdown"].append(g.log.folded is None)
        out["a_on_button"].append(g.a_on_button)
    return {k: np.asarray(v) for k, v in out.items()}


def compare_estimators(raw_totals: np.ndarray, adj_totals: np.ndarray, hands_per_unit: int,
                       rng: np.random.Generator, n_boot: int = 4000) -> dict:
    """Plain vs adjusted, paired over the same units (sessions): means, the difference, SD ratio.

    - the paired difference D = raw - adjusted per unit, with a normal 95% CI.
      Unbiasedness predicts E[D] = 0; ``mde_bias_bb100`` is the bias this
      check could detect with 80% power (2.8 SE of D), i.e. its power.
    - ``sd_ratio`` = SD(adjusted) / SD(raw) with a percentile bootstrap CI
      over units; ``var_ratio`` = its square, and ``hands_saved`` = 1 - var_ratio
      is the fraction of hands the adjusted estimator needs less for the same SE.
    """
    from nutsfinder.eval.stats import summarize

    raw_totals = np.asarray(raw_totals, dtype=float)
    adj_totals = np.asarray(adj_totals, dtype=float)
    n = len(raw_totals)
    sr = summarize(raw_totals.tolist(), hands_per_unit)
    sa = summarize(adj_totals.tolist(), hands_per_unit)
    sd = summarize((raw_totals - adj_totals).tolist(), hands_per_unit)
    idx = rng.integers(0, n, size=(n_boot, n))
    ratio = adj_totals[idx].std(axis=1, ddof=1) / raw_totals[idx].std(axis=1, ddof=1)
    r = float(adj_totals.std(ddof=1) / raw_totals.std(ddof=1))
    lo, hi = np.quantile(ratio, [0.025, 0.975])
    return {
        "n_units": n,
        "raw": {"mean_bb100": sr.mean_bb100, "se_bb100": sr.se_bb100, "ci95_bb100": list(sr.ci_bb100),
                "sd_bb100_per_unit": sr.sd * 100 / hands_per_unit},
        "adjusted": {"mean_bb100": sa.mean_bb100, "se_bb100": sa.se_bb100, "ci95_bb100": list(sa.ci_bb100),
                     "sd_bb100_per_unit": sa.sd * 100 / hands_per_unit},
        "difference_raw_minus_adjusted": {"mean_bb100": sd.mean_bb100, "se_bb100": sd.se_bb100,
                                          "ci95_bb100": list(sd.ci_bb100), "p_value": sd.p_value,
                                          "mde_bias_bb100": 2.8 * sd.se_bb100},
        "sd_ratio": r, "sd_ratio_ci95": [float(lo), float(hi)],
        "var_ratio": r**2, "var_ratio_ci95": [float(lo) ** 2, float(hi) ** 2],
        "hands_saved": 1 - r**2, "hands_saved_ci95": [1 - float(hi) ** 2, 1 - float(lo) ** 2],
    }
