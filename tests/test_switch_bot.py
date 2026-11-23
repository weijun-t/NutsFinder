"""Tests for SwitchBot: switches at exactly K in both duplicate rooms, delegates exactly, reseeds both styles."""

from __future__ import annotations

import functools
import pickle

import pytest

from nutsfinder.bots import AlwaysRaiseBot, BayesBot, CallStationBot, RandomBot, SwitchBot
from nutsfinder.engine.actions import Bet, Raise
from nutsfinder.eval.match import duplicate_match
from nutsfinder.eval.sessions import play_session

K = 4


def _switch(first, second, k=K):
    return functools.partial(SwitchBot, first, second, k)


def _aggressive(game, seat) -> bool:
    return any(r.player == seat and isinstance(r.action, (Bet, Raise)) for r in game.log.actions)


def test_switches_at_exactly_k_in_both_rooms():
    # B = SwitchBot(CallStation -> AlwaysRaise). AlwaysRaise can always raise preflop (or bet), so a
    # hand without any bet/raise by B is a CallStation hand; CallStation never bets.
    m = duplicate_match(CallStationBot, _switch(CallStationBot, AlwaysRaiseBot), 10, master_seed=11)
    for g in m.games:
        b_seat = 1 - g.a_seat
        assert _aggressive(g, b_seat) == (g.unit >= K), (g.unit, g.a_seat)
    # both rooms: game 1 (a_seat 0) and game 2 (a_seat 1) of every pair
    assert {(g.unit, g.a_seat) for g in m.games} == {(i, s) for i in range(10) for s in (0, 1)}


def test_counts_finished_hands_and_reports_style():
    bot = SwitchBot(CallStationBot, AlwaysRaiseBot, 2)
    assert not bot.switched and isinstance(bot.active, CallStationBot)
    duplicate_match(CallStationBot, lambda: bot, 2, master_seed=3)  # the same instance in both rooms: 4 hands
    assert bot.hands_played == 4 and bot.switched and isinstance(bot.active, AlwaysRaiseBot)


def test_k_zero_is_style_b_from_the_start():
    a = duplicate_match(RandomBot, _switch(CallStationBot, AlwaysRaiseBot, 0), 6, master_seed=5)
    b = duplicate_match(RandomBot, AlwaysRaiseBot, 6, master_seed=5)
    assert a.unit_results == b.unit_results


def test_rejects_negative_k_and_has_a_config_repr():
    with pytest.raises(ValueError):
        SwitchBot(CallStationBot, AlwaysRaiseBot, -1)
    assert repr(SwitchBot(CallStationBot, AlwaysRaiseBot, 125)) == (
        "SwitchBot(CallStationBot() -> AlwaysRaiseBot() after 125 hands)")


def test_factory_is_picklable_for_worker_processes():
    f = pickle.loads(pickle.dumps(_switch(RandomBot, CallStationBot)))
    assert isinstance(f(), SwitchBot)


def test_reseed_is_forwarded_to_both_styles():
    bot = SwitchBot(RandomBot, RandomBot, 1)
    bot.reseed(42)
    ref = RandomBot(42)
    assert bot.first._rng.random() == ref._rng.random()
    ref = RandomBot(42)
    assert bot.second._rng.random() == ref._rng.random()


def test_pre_switch_hands_equal_style_a_alone_with_a_random_style():
    # Style A is random: identical only if A gets the same reseed stream as when it plays alone.
    sw = duplicate_match(RandomBot, _switch(RandomBot, CallStationBot), 10, master_seed=21)
    alone = duplicate_match(RandomBot, RandomBot, 10, master_seed=21)
    assert sw.unit_results[:K] == alone.unit_results[:K]
    assert sw.unit_results[K:] != alone.unit_results[K:]


def test_post_switch_hands_equal_style_b_alone_against_a_memoryless_target():
    # The target (RandomBot) has no memory, so after the switch it faces exactly "style B alone".
    sw = duplicate_match(RandomBot, _switch(RandomBot, AlwaysRaiseBot), 10, master_seed=22)
    alone = duplicate_match(RandomBot, AlwaysRaiseBot, 10, master_seed=22)
    assert sw.unit_results[K:] == alone.unit_results[K:]
    assert sw.unit_results[:K] != alone.unit_results[:K]


def test_learning_target_pre_window_identity_and_post_window_difference():
    bayes = functools.partial(BayesBot, samples=40, bluff_samples=40)
    k, pairs = 3, 6
    sw = play_session(bayes, _switch(CallStationBot, AlwaysRaiseBot, k), pairs, 9, 0)
    alone_a = play_session(bayes, CallStationBot, k, 9, 0)  # the first k pairs of the same session
    assert sw.hand_results[: 2 * k] == alone_a.hand_results
    assert sw.seed == alone_a.seed

