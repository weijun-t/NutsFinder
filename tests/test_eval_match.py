"""Tests for plain and duplicate matches.

The key properties of a duplicate match:

1. Same deal, swapped seats: in both games of a pair the board and each
   seat's hole cards are identical, the button is on the same seat, and the
   bots have swapped seats. So bot A holds in game 2 exactly the cards (and
   the position) bot B held in game 1.
2. Deterministic self-play cancels exactly. If A and B are the same
   deterministic strategy, game 2 is game 1 with the names swapped: the same
   seat makes the same decisions with the same cards. So the seat-0 net is
   the same in both games. A sat in seat 0 in game 1 and seat 1 in game 2,
   so A's pair result is net0 + net1 of one hand = 0 (poker is zero-sum).
3. Everything is reproducible from the master seed.
"""

from __future__ import annotations

import math

import pytest

from nutsfinder.bots import AlwaysRaiseBot, CallStationBot, RandomBot
from nutsfinder.engine.actions import Action
from nutsfinder.engine.betting import PublicState
from nutsfinder.engine.bot import Bot
from nutsfinder.engine.cards import Card
from nutsfinder.eval.breakdown import by_end_street, by_position, end_street, fold_stats
from nutsfinder.eval.match import duplicate_match, plain_match, unit_seeds


def test_duplicate_deal_swaps_cards_and_seats() -> None:
    r = duplicate_match(RandomBot, CallStationBot, 200, master_seed=5)
    assert r.hands_played == 400 and r.n_units == 200
    for i in range(r.n_units):
        g1, g2 = r.games[2 * i], r.games[2 * i + 1]
        assert g1.unit == g2.unit == i
        assert g1.log.seed == g2.log.seed
        assert g1.log.button == g2.log.button == i % 2
        assert (g1.a_seat, g2.a_seat) == (0, 1)
        # A gets B's cards and B gets A's.
        assert g2.log.hole_cards[g2.a_seat] == g1.log.hole_cards[1 - g1.a_seat]
        assert g2.log.hole_cards[1 - g2.a_seat] == g1.log.hole_cards[g1.a_seat]
        # A gets B's position.
        assert g2.a_on_button == (not g1.a_on_button)
        # Same board as far as both were dealt.
        k = min(len(g1.log.board), len(g2.log.board))
        assert g1.log.board[:k] == g2.log.board[:k]
        # Pair result is A's net in both games.
        assert r.unit_results[i] == pytest.approx(g1.a_net_bb + g2.a_net_bb)


def test_cards_dealt_in_full_are_identical() -> None:
    # CallStation vs CallStation always reaches showdown: all 5 board cards.
    r = duplicate_match(CallStationBot, CallStationBot, 50, master_seed=9)
    for i in range(r.n_units):
        g1, g2 = r.games[2 * i], r.games[2 * i + 1]
        assert len(g1.log.board) == 5 and g1.log.board == g2.log.board
        assert g1.log.hole_cards == g2.log.hole_cards  # per seat


@pytest.mark.parametrize("bot", [CallStationBot, AlwaysRaiseBot])
def test_deterministic_self_play_is_exactly_zero(bot) -> None:
    r = duplicate_match(bot, bot, 500, master_seed=3)
    assert all(x == 0 for x in r.unit_results)
    assert r.mean_bb100 == 0 and r.se_bb100 == 0
    # Zero spread: no p-value, never "significant", labelled as no variation.
    assert r.stats.no_variation and not r.stats.significant
    assert math.isnan(r.p_value)
    assert r.verdict().startswith("no variation")


def test_always_raise_vs_call_station_is_exactly_zero() -> None:
    # Neither bot looks at its cards, and the betting comes out the same from
    # either position (raise to 300 preflop, then pot-sized bets that
    # CallStation calls), so every hand is a pot of the same size decided by
    # the cards alone. Swapping the cards then cancels each pair exactly.
    r = duplicate_match(AlwaysRaiseBot, CallStationBot, 500, master_seed=4)
    assert all(x == 0 for x in r.unit_results)


def test_random_self_play_within_ci_of_zero() -> None:
    # Not exact: the two RandomBots have different seeds. But they are the
    # same strategy, so the true edge is 0 and must lie inside the CI (for
    # this fixed seed; over many seeds it would fail about 5% of the time).
    r = duplicate_match(RandomBot, RandomBot, 2000, master_seed=11)
    lo, hi = r.ci_bb100
    assert r.se_bb100 > 0
    assert lo <= 0 <= hi


def test_determinism_from_master_seed() -> None:
    for match in (duplicate_match, plain_match):
        n = 300
        r1 = match(RandomBot, AlwaysRaiseBot, n, master_seed=21)
        r2 = match(RandomBot, AlwaysRaiseBot, n, master_seed=21)
        r3 = match(RandomBot, AlwaysRaiseBot, n, master_seed=22)
        assert r1.unit_results == r2.unit_results
        assert [g.log for g in r1.games] == [g.log for g in r2.games]
        assert r1.unit_results != r3.unit_results


class SeedSpy(Bot):
    """Records every seed it is given; plays like CallStationBot."""

    instances: list[SeedSpy] = []

    def __init__(self) -> None:
        self.seeds: list[int] = []
        self._inner = CallStationBot()
        SeedSpy.instances.append(self)

    def reseed(self, seed: int) -> None:
        self.seeds.append(seed)

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        return self._inner.act(state, hole_cards)


def test_bot_seeds_synced_by_bot_per_pair() -> None:
    SeedSpy.instances = []
    duplicate_match(SeedSpy, SeedSpy, 20, master_seed=8)
    a1, b1, a2, b2 = SeedSpy.instances  # creation order in duplicate_match
    seeds = unit_seeds(8, 20)
    # A gets the same seed in both games of a pair, and a new one every pair.
    assert a1.seeds == a2.seeds == [s[1] for s in seeds]
    assert b1.seeds == b2.seeds == [s[2] for s in seeds]
    assert a1.seeds != b1.seeds


def test_plain_match_alternates_button() -> None:
    r = plain_match(RandomBot, CallStationBot, 100, master_seed=2)
    assert r.hands_played == r.n_units == 100
    assert [g.log.button for g in r.games] == [i % 2 for i in range(100)]
    assert all(g.a_seat == 0 for g in r.games)
    assert r.stats.hands_per_unit == 1


def test_bb100_conversion_for_pairs() -> None:
    r = duplicate_match(RandomBot, CallStationBot, 100, master_seed=6)
    total_bb = sum(g.a_net_bb for g in r.games)
    assert r.mean_bb100 == pytest.approx(100 * total_bb / r.hands_played)


def test_breakdowns_add_up() -> None:
    r = duplicate_match(RandomBot, AlwaysRaiseBot, 300, master_seed=12)
    pos = by_position(r)
    assert pos["button"]["hands"] == pos["big_blind"]["hands"] == 300
    assert (pos["button"]["bb100"] + pos["big_blind"]["bb100"]) / 2 == pytest.approx(r.mean_bb100)
    streets = by_end_street(r)
    assert sum(v["hands"] for v in streets.values()) == r.hands_played
    assert sum(v["bb100_contribution"] for v in streets.values()) == pytest.approx(r.mean_bb100)
    # AlwaysRaiseBot never folds; CallStation games all end at showdown.
    assert sum(fold_stats(r.games, "b").folded_to_bet.values()) == 0
    cs = duplicate_match(CallStationBot, CallStationBot, 10, master_seed=1)
    assert all(end_street(g.log) == "showdown" for g in cs.games)


def test_summary_mentions_the_numbers() -> None:
    r = duplicate_match(RandomBot, RandomBot, 50, master_seed=1)
    text = r.summary()
    assert "50 pairs, 100 hands" in text and "master seed 1" in text and "verdict" in text


def test_too_short() -> None:
    with pytest.raises(ValueError):
        duplicate_match(RandomBot, RandomBot, 1, master_seed=0)
    with pytest.raises(ValueError):
        plain_match(RandomBot, RandomBot, 1, master_seed=0)


def test_se_is_computed_over_pairs() -> None:
    # The unit of analysis is the pair: n = number of pairs, not hands.
    from math import sqrt
    from statistics import stdev

    n_pairs = 300
    r = duplicate_match(RandomBot, AlwaysRaiseBot, n_pairs, master_seed=11)
    assert r.n_units == n_pairs
    assert r.hands_played == 2 * n_pairs
    assert r.stats.n == n_pairs
    assert r.stats.se == pytest.approx(stdev(r.unit_results) / sqrt(n_pairs))
    # Each pair result is A's net in game 1 plus A's net in game 2.
    for i, unit in enumerate(r.unit_results):
        g1, g2 = r.games[2 * i], r.games[2 * i + 1]
        assert unit == pytest.approx(g1.a_net_bb + g2.a_net_bb)


def test_bot_descriptions_are_taken_before_play():
    """``bot_*_desc`` records the configuration, not state learned during the match."""
    from nutsfinder.bots import BayesBot, RandomBot
    from nutsfinder.eval.match import duplicate_match, plain_match

    for m in (duplicate_match(BayesBot, RandomBot, 3, 1), plain_match(BayesBot, RandomBot, 4, 1)):
        assert "hands_seen=0" in m.bot_a_desc
