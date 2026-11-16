"""Tests for session-based duplicate matches (valid stats for learning bots).

What must hold:

1. Every session builds fresh bots: two copies of A and two of B (one per
   room), and no instance ever plays a hand of another session or room.
2. Session seeds are deterministic, distinct, and do not depend on how many
   sessions are played.
3. A session's result is the sum of its hands; bb/100 = total / hands * 100;
   the headline stats are summarize() over sessions.
4. Session s is exactly duplicate_match(..., session_seed(master, s)).
5. The number of worker processes cannot change any number.
6. A learning bot's change in behaviour shows up in the learning curve, with
   CIs across sessions.
7. For bots without memory, the session estimate agrees with one long
   duplicate match within CI.
"""

from __future__ import annotations

import math

import pytest

from nutsfinder.bots import AlwaysRaiseBot, CallStationBot, EquityBot, RandomBot
from nutsfinder.engine.actions import Action, Call, Check, Fold
from nutsfinder.engine.betting import PublicState
from nutsfinder.engine.bot import Bot, HandSummary
from nutsfinder.engine.cards import Card
from nutsfinder.eval.match import duplicate_match, unit_seeds
from nutsfinder.eval.report import save_session_json, save_session_plot, session_result_dict
from nutsfinder.eval.sessions import (
    learning_curve,
    learning_gain,
    play_session,
    session_by_position,
    session_duplicate_match,
    session_seed,
)
from nutsfinder.eval.stats import summarize


# --- test-only bots -------------------------------------------------------------


class SpyBot(Bot):
    """Check/call bot that records every construction, reseed and finished hand."""

    instances: list["SpyBot"] = []

    def __init__(self) -> None:
        self.reseeds: list[int] = []
        self.hands_seen = 0
        SpyBot.instances.append(self)

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        return Check() if state.legal.can_check else Call()

    def reseed(self, seed: int) -> None:
        self.reseeds.append(seed)

    def end_hand(self, summary: HandSummary) -> None:
        self.hands_seen += 1


class FoldThenCallBot(Bot):
    """A toy learner: folds its first LEARN_AFTER hands, then plays like a call station.

    Its behaviour depends on how many hands *this instance* has seen, which
    is exactly the kind of memory that breaks per-pair i.i.d. assumptions.
    """

    LEARN_AFTER = 10

    def __init__(self) -> None:
        self.hands_seen = 0

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        if self.hands_seen < self.LEARN_AFTER:
            return Fold()
        return Check() if state.legal.can_check else Call()

    def end_hand(self, summary: HandSummary) -> None:
        self.hands_seen += 1


# --- 1. fresh instances -------------------------------------------------------------


def test_fresh_instances_per_session_and_room() -> None:
    SpyBot.instances.clear()
    n_sessions, pairs, master = 3, 4, 123
    r = session_duplicate_match(SpyBot, CallStationBot, n_sessions, pairs, master)
    spies = SpyBot.instances
    # Two copies of A per session (room 1 and room 2), nothing else constructed.
    assert len(spies) == 2 * n_sessions
    assert len({id(s) for s in spies}) == len(spies)
    for s_idx in range(n_sessions):
        room1, room2 = spies[2 * s_idx], spies[2 * s_idx + 1]
        expected = [seed_a for _, seed_a, _ in unit_seeds(session_seed(master, s_idx), pairs)]
        for spy in (room1, room2):
            # Each instance plays exactly its own session's pairs, one hand per pair...
            assert spy.hands_seen == pairs
            # ... with that session's bot seeds and no other.
            assert spy.reseeds == expected
    assert r.n_sessions == n_sessions and r.hands_played == n_sessions * 2 * pairs


def test_factory_construction_count_for_both_bots() -> None:
    counts = {"a": 0, "b": 0}

    def make_a() -> Bot:
        counts["a"] += 1
        return CallStationBot()

    def make_b() -> Bot:
        counts["b"] += 1
        return AlwaysRaiseBot()

    session_duplicate_match(make_a, make_b, 5, 3, master_seed=1)
    assert counts == {"a": 10, "b": 10}


# --- 2. seeds ------------------------------------------------------------------------


def test_session_seeds_deterministic_and_distinct() -> None:
    seeds = [session_seed(7, s) for s in range(2000)]
    assert seeds == [session_seed(7, s) for s in range(2000)]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= x < 2**64 for x in seeds)
    # Other master seeds do not overlap (in particular not shifted copies).
    assert not set(seeds) & {session_seed(8, s) for s in range(2000)}
    with pytest.raises(ValueError):
        session_seed(7, -1)


def test_sessions_do_not_depend_on_session_count() -> None:
    short = session_duplicate_match(RandomBot, AlwaysRaiseBot, 3, 6, master_seed=42)
    long = session_duplicate_match(RandomBot, AlwaysRaiseBot, 5, 6, master_seed=42)
    assert long.sessions[:3] == short.sessions


def test_sessions_get_different_deals() -> None:
    r = session_duplicate_match(RandomBot, CallStationBot, 6, 5, master_seed=3)
    deals = [tuple(g.log.seed for g in s.games) for s in r.sessions]
    assert len(set(deals)) == len(deals)
    all_seeds = [x for d in deals for x in d[::2]]
    assert len(set(all_seeds)) == len(all_seeds)


def test_reproducible() -> None:
    a = session_duplicate_match(RandomBot, CallStationBot, 4, 5, master_seed=9)
    b = session_duplicate_match(RandomBot, CallStationBot, 4, 5, master_seed=9)
    assert a == b


# --- 3. arithmetic ---------------------------------------------------------------------


def test_session_result_is_sum_of_hands_and_bb100() -> None:
    r = session_duplicate_match(RandomBot, AlwaysRaiseBot, 6, 7, master_seed=5)
    for s in r.sessions:
        assert len(s.hand_results) == 14 and s.hands == 14
        assert s.hand_results == tuple(g.a_net_bb for g in s.games)
        for i, p in enumerate(s.pair_results):
            assert p == pytest.approx(s.hand_results[2 * i] + s.hand_results[2 * i + 1])
        assert s.total_bb == pytest.approx(sum(s.hand_results))
        assert s.bb100 == pytest.approx(s.total_bb / 14 * 100)
    st = summarize([s.total_bb for s in r.sessions], hands_per_unit=14)
    assert r.stats == st
    assert r.mean_bb100 == pytest.approx(sum(r.session_bb100) / 6)
    sd = math.sqrt(sum((x - r.mean_bb100) ** 2 for x in r.session_bb100) / 5)
    assert r.sd_session_bb100 == pytest.approx(sd)
    assert r.se_bb100 == pytest.approx(sd / math.sqrt(6))
    lo, hi = r.ci_bb100
    assert (lo + hi) / 2 == pytest.approx(r.mean_bb100)
    assert (hi - lo) / 2 == pytest.approx(1.959964 * r.se_bb100, rel=1e-5)


def test_learning_curve_buckets() -> None:
    r = session_duplicate_match(RandomBot, CallStationBot, 5, 12, master_seed=8)
    curve = learning_curve(r, bucket_pairs=5)
    assert [(b.start, b.stop) for b in curve] == [(0, 5), (5, 10), (10, 12)]
    for b in curve:
        vals = [sum(s.pair_results[b.start : b.stop]) for s in r.sessions]
        assert b.stats == summarize(vals, hands_per_unit=2 * (b.stop - b.start))
        assert b.stats.n == r.n_sessions  # one value per session per bucket
    # Hand-weighted average of the buckets is the overall mean.
    weighted = sum(b.stats.mean_bb100 * (b.stop - b.start) for b in curve) / 12
    assert weighted == pytest.approx(r.mean_bb100)


def test_position_breakdown_is_per_session() -> None:
    r = session_duplicate_match(RandomBot, CallStationBot, 4, 6, master_seed=2)
    pos = session_by_position(r)
    total = (pos["button"]["bb100"] + pos["big_blind"]["bb100"]) / 2
    assert total == pytest.approx(r.mean_bb100)


def test_rejects_too_few_units() -> None:
    with pytest.raises(ValueError):
        session_duplicate_match(RandomBot, CallStationBot, 1, 10, master_seed=1)
    with pytest.raises(ValueError):
        session_duplicate_match(RandomBot, CallStationBot, 10, 1, master_seed=1)


# --- 4. exact agreement with duplicate_match -------------------------------------------


@pytest.mark.parametrize("make_a,make_b", [(RandomBot, AlwaysRaiseBot), (FoldThenCallBot, RandomBot)])
def test_session_is_exactly_a_fresh_duplicate_match(make_a, make_b) -> None:
    master = 77
    r = session_duplicate_match(make_a, make_b, 3, 15, master)
    for s in r.sessions:
        d = duplicate_match(make_a, make_b, 15, session_seed(master, s.index))
        assert s.pair_results == d.unit_results
        assert s.games == d.games


def test_deterministic_matchups_exact() -> None:
    # Deterministic self-play cancels every pair, so every session is exactly 0.
    r = session_duplicate_match(CallStationBot, CallStationBot, 4, 10, master_seed=1)
    assert all(s.total_bb == 0 for s in r.sessions)
    assert r.stats.no_variation and math.isnan(r.p_value)
    assert "no variation" in r.verdict()
    # Deterministic non-learning bots: pooling the sessions' pairs is the same
    # thing as a duplicate match on those deals, and the session mean equals the
    # pooled per-pair mean exactly (equal-sized sessions).
    r = session_duplicate_match(AlwaysRaiseBot, CallStationBot, 4, 10, master_seed=1)
    assert r.mean_bb100 == pytest.approx(r.naive_pair_stats().mean_bb100)


# --- 5. processes ------------------------------------------------------------------------


def test_process_count_does_not_change_results() -> None:
    args = (RandomBot, AlwaysRaiseBot, 6, 8, 2024)
    serial = session_duplicate_match(*args, processes=1)
    for procs in (2, 3):
        par = session_duplicate_match(*args, processes=procs)
        assert par == serial  # sessions incl. every hand log, seeds and stats


# --- 6. learning bot ------------------------------------------------------------------


def test_learning_curve_detects_change_in_behaviour() -> None:
    # Against CallStationBot everything is deterministic: while the toy folds it
    # loses 0.5 bb on the button + 1 bb in the big blind per pair = -75 bb/100;
    # once it calls, it is CallStation vs CallStation and every pair is 0.
    learn = FoldThenCallBot.LEARN_AFTER
    r = session_duplicate_match(FoldThenCallBot, CallStationBot, 3, 2 * learn, master_seed=4)
    curve = learning_curve(r, bucket_pairs=learn // 2)
    assert [b.stats.mean_bb100 for b in curve] == pytest.approx([-75, -75, 0, 0])
    assert r.mean_bb100 == pytest.approx(-37.5)


def test_learning_gain_is_significant_with_noise() -> None:
    # Against RandomBot there is real noise; the change is still detected,
    # with a CI computed across independent sessions.
    learn = FoldThenCallBot.LEARN_AFTER
    r = session_duplicate_match(FoldThenCallBot, RandomBot, 30, 2 * learn, master_seed=6)
    gain = learning_gain(r, early=(0, learn), late=(learn, 2 * learn))
    assert gain.significant and gain.mean_bb100 > 0
    curve = learning_curve(r, bucket_pairs=learn)
    assert curve[0].stats.ci_bb100[1] < curve[1].stats.ci_bb100[0]  # CIs do not even overlap


def test_learning_bot_long_match_estimate_depends_on_length() -> None:
    # Why sessions are needed: with one persistent bot, "the" bb/100 of a
    # learning bot changes with match length (here -75 -> -37.5 -> -15).
    m = [duplicate_match(FoldThenCallBot, CallStationBot, n, 1).mean_bb100 for n in (10, 20, 50)]
    assert m == pytest.approx([-75, -37.5, -15])


# --- 7. agreement in expectation ---------------------------------------------------


def _agree(a, b) -> bool:
    """Independent estimates agree within the 95% CI of their difference."""
    return abs(a.mean_bb100 - b.mean_bb100) <= 1.96 * math.hypot(a.se_bb100, b.se_bb100)


def test_agrees_with_duplicate_for_non_learning_bot() -> None:
    sess = session_duplicate_match(RandomBot, AlwaysRaiseBot, 40, 25, master_seed=11)
    dup = duplicate_match(RandomBot, AlwaysRaiseBot, 1000, master_seed=12)
    assert _agree(sess, dup)
    # No memory, so the design effect should be near 1 (no within-session correlation).
    assert 0.6 < sess.design_effect() < 1.6


@pytest.mark.slow
def test_equity_bot_session_agrees_with_duplicate() -> None:
    sess = session_duplicate_match(EquityBot, RandomBot, 40, 50, master_seed=13, processes=4)
    dup = duplicate_match(EquityBot, RandomBot, 2000, master_seed=14)
    assert sess.stats.significant and dup.stats.significant
    assert _agree(sess, dup)


# --- report -------------------------------------------------------------------------


def test_json_and_plot(tmp_path) -> None:
    r = session_duplicate_match(RandomBot, CallStationBot, 4, 10, master_seed=3)
    d = session_result_dict(r, bucket_pairs=5)
    assert d["n_sessions"] == 4 and d["hands_per_session"] == 20
    assert len(d["hand_results_bb"]) == 4 and len(d["hand_results_bb"][0]) == 20
    assert len(d["learning_curve"]["buckets"]) == 2
    save_session_json(r, tmp_path / "s.json", bucket_pairs=5)
    save_session_plot(r, tmp_path / "s.png", bucket_pairs=5)
    assert (tmp_path / "s.json").stat().st_size > 0 and (tmp_path / "s.png").stat().st_size > 0


def test_play_session_without_games() -> None:
    s = play_session(RandomBot, CallStationBot, 5, master_seed=1, session_index=2, keep_games=False)
    full = play_session(RandomBot, CallStationBot, 5, master_seed=1, session_index=2)
    assert s.games == () and s.hand_results == full.hand_results
