"""Where a win rate comes from: by position, by street, and fold "leaks".

These are diagnostics, not verdicts. The headline number and its CI come
from the match's units (:mod:`nutsfinder.eval.match`). The breakdowns here
are computed per *game* (single hand), so:

- In a duplicate match, A's button games and A's big-blind games are the
  two halves of the same pairs, i.e. the same deals. Each half on its own
  carries the full card luck again, so the per-position SEs are much wider
  than the headline SE. That is expected: duplicate cancels luck only when
  the two halves are added up.
- Splitting by street the hand *ended on* is a selection, not a random
  sample (the bots' actions decide which hands reach the river), so no CI
  is attached: read it as "where the chips changed hands".
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from nutsfinder.engine.actions import Fold
from nutsfinder.engine.betting import Street
from nutsfinder.engine.hand_log import HandLog
from nutsfinder.eval.match import Game, MatchResult
from nutsfinder.eval.stats import summarize

__all__ = ["by_position", "by_end_street", "fold_stats", "FoldStats", "breakdown"]

END_STREETS = ("preflop", "flop", "turn", "river", "showdown")


def by_position(result: MatchResult) -> dict[str, dict[str, float]]:
    """A's bb/100 with A on the button vs in the big blind, per game."""
    out: dict[str, dict[str, float]] = {}
    for label, on_button in (("button", True), ("big_blind", False)):
        nets = [g.a_net_bb for g in result.games if g.a_on_button == on_button]
        s = summarize(nets, hands_per_unit=1)
        out[label] = {
            "hands": s.n,
            "bb100": s.mean_bb100,
            "se_bb100": s.se_bb100,
            "ci_low": s.ci_bb100[0],
            "ci_high": s.ci_bb100[1],
        }
    return out


def end_street(log: HandLog) -> str:
    """The street a hand ended on (a fold there), or ``"showdown"``."""
    if log.folded is None:
        return "showdown"
    folds = [r for r in log.actions if isinstance(r.action, Fold)]
    return folds[-1].street.name.lower()


def by_end_street(result: MatchResult) -> dict[str, dict[str, float]]:
    """Share of hands ending on each street, and A's bb/100 contributed by them.

    The ``bb100_contribution`` values add up to A's overall bb/100.
    """
    total = result.hands_played
    out = {s: {"hands": 0, "share": 0.0, "bb100_contribution": 0.0} for s in END_STREETS}
    for g in result.games:
        row = out[end_street(g.log)]
        row["hands"] += 1
        row["bb100_contribution"] += 100 * g.a_net_bb / total
    for row in out.values():
        row["share"] = row["hands"] / total
    return out


@dataclass
class FoldStats:
    """Per street: how often a seat faced a bet, folded to it, or folded for free."""

    faced_bet: dict[str, int]
    folded_to_bet: dict[str, int]
    free_folds: dict[str, int]

    def fold_to_bet_rate(self, street: str) -> float:
        n = self.faced_bet[street]
        return self.folded_to_bet[street] / n if n else math.nan


def _decisions(log: HandLog) -> Iterable[tuple[int, str, bool, bool]]:
    """Yield (seat, street, facing_a_bet, folded) for every action in a hand.

    "Facing a bet" means the opponent has more chips in on this street than
    the actor, so there is something to call. It is rebuilt from the log by
    tracking each seat's chips per street (the blinds count preflop).
    """
    street_bets = [0, 0]
    street = None
    for r in log.actions:
        if r.street != street:
            street = r.street
            street_bets = list(log.blinds) if street == Street.PREFLOP else [0, 0]
        me = r.player
        facing = street_bets[1 - me] > street_bets[me]
        yield me, street.name.lower(), facing, isinstance(r.action, Fold)
        street_bets[me] += r.chips


def fold_stats(games: Iterable[Game], who: str) -> FoldStats:
    """Fold behaviour of bot ``who`` ("a" or "b") over the given games."""
    streets = [s.name.lower() for s in Street]
    stats = FoldStats({s: 0 for s in streets}, {s: 0 for s in streets}, {s: 0 for s in streets})
    for g in games:
        seat = g.a_seat if who == "a" else 1 - g.a_seat
        for actor, street, facing, folded in _decisions(g.log):
            if actor != seat:
                continue
            if facing:
                stats.faced_bet[street] += 1
                stats.folded_to_bet[street] += folded
            elif folded:
                stats.free_folds[street] += 1
    return stats


def _fold_dict(fs: FoldStats) -> dict[str, dict[str, float]]:
    return {
        s: {
            "faced_bet": fs.faced_bet[s],
            "fold_to_bet_rate": fs.fold_to_bet_rate(s),
            "free_folds": fs.free_folds[s],
        }
        for s in fs.faced_bet
    }


def leaks(result: MatchResult) -> list[str]:
    """Plain-language list of the most obvious leaks for both bots."""
    notes: list[str] = []
    for who, name in (("a", result.bot_a), ("b", result.bot_b)):
        fs = fold_stats(result.games, who)
        free = sum(fs.free_folds.values())
        if free:
            notes.append(
                f"{name} ({who.upper()}) folded {free:,} times when checking was free "
                f"(pure loss: it gives up its equity in the pot for nothing)"
            )
        for s in fs.faced_bet:
            rate = fs.fold_to_bet_rate(s)
            if fs.faced_bet[s] >= 100 and rate > 0.5:
                notes.append(
                    f"{name} ({who.upper()}) folds {rate:.0%} to {s} bets "
                    f"({fs.faced_bet[s]:,} times faced): very exploitable by betting"
                )
            if fs.faced_bet[s] >= 100 and rate == 0:
                notes.append(
                    f"{name} ({who.upper()}) never folds to {s} bets "
                    f"({fs.faced_bet[s]:,} times faced): bluffing it is pointless, value-bet it"
                )
    return notes


def breakdown(result: MatchResult) -> dict:
    """All diagnostics as one JSON-friendly dict."""
    return {
        "by_position": by_position(result),
        "by_end_street": by_end_street(result),
        "folds_a": _fold_dict(fold_stats(result.games, "a")),
        "folds_b": _fold_dict(fold_stats(result.games, "b")),
        "leaks": leaks(result),
    }
