"""Tests for Local Best Response (``nutsfinder.eval.lbr``), its policy queries and exact equities.

What is checked:

- exact equities (``eval/exact_equity.py``) against the engine's equity code;
- ``CFRBot.range_policy``: a probability distribution, equal to the blueprint where
  nothing is hidden, and equal in law to what ``act`` really plays (sampled
  action sequences vs predicted sequence probabilities, with mixed translations);
- the equity-family oracle: the vectorised decision equals ``act`` per holding;
- **no side effects**: a match where LBR queries the opponent and a replay of
  the same actions without any query give identical logs and opponent state;
- **no leak**: LBR's actions do not change when only the opponent's cards change;
- sanity values: vs always-fold exactly +750 mbb/g; vs CallStation and
  AlwaysRaise clearly positive; legality (incl. unequal stacks); determinism.
"""

from __future__ import annotations

import collections
import copy
import functools

import numpy as np
import pytest
from test_cfr_bot import BP, hole

from nutsfinder.abstraction.cards import ehs_all
from nutsfinder.abstraction.config import DEFAULT_CONFIG
from nutsfinder.abstraction.strength import COMBO_CARDS, card_ids, combo_index
from nutsfinder.bots import AlwaysRaiseBot, BayesBot, CallStationBot, CFRBot, EquityBot, PotOddsBot, RandomBot
from nutsfinder.bots.cfr_bot import _cards_for
from nutsfinder.bots.common import aggressive_action, raise_to_for_fraction
from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import Street, apply_action, deal_board, new_hand, public_view
from nutsfinder.engine.bot import Bot
from nutsfinder.engine.cards import Card, parse_cards
from nutsfinder.engine.equity import RANDOM_RANGE, equity_vs_range
from nutsfinder.engine.preflop import top_range
from nutsfinder.engine.runner import play_hand
from nutsfinder.eval.exact_equity import combo_weights, hero_vs_each, holdings_vs_ranges, preflop_vs_each
from nutsfinder.eval.lbr import (
    CFROracle,
    RandomBotOracle,
    EquityFamilyOracle,
    LBRBot,
    SnapshotOracle,
    action_probs,
    lbr_session,
    make_oracle,
    rebuild_state,
)
from nutsfinder.eval.match import unit_seeds


# --- helpers ------------------------------------------------------------------------------------


class AlwaysFoldBot(Bot):
    """Folds whenever asked, even when checking is free (the classic 'always fold')."""

    def act(self, state, hole_cards):
        return Fold()


class FoldOrCheckBot(Bot):
    """Checks when free, folds to any bet."""

    def act(self, state, hole_cards):
        return Check() if state.legal.can_check else Fold()


class ScriptBot(Bot):
    """Deterministic: the first bet/raise of each street is off-tree (0.35 pot preflop, 0.5 pot after), then calls."""

    def act(self, v, h):
        legal = v.legal
        mine = [r for r in v.history if r.player == v.seat and r.street == v.street and isinstance(r.action, (Bet, Raise))]
        if (legal.can_bet or legal.can_raise) and not mine:
            return aggressive_action(legal, raise_to_for_fraction(v, 0.5 if v.street else 0.35))
        return Call() if legal.can_call else Check()


def play_fixed(bots, holes, board, button, stacks=(10_000, 10_000)):
    """Play one hand with chosen cards; yields (seat, view, action) and returns nothing."""
    s = new_hand(button=button, stacks=stacks, hole_cards=holes)
    while not s.is_hand_over:
        if s.cards_needed:
            n = len(s.board)
            s = deal_board(s, board[n:n + s.cards_needed])
            continue
        p = s.to_act
        v = public_view(s, p)
        a = bots[p].act(v, holes[p])
        yield p, v, a
        s = apply_action(s, p, a)


def combo(cards: str) -> int:
    ids = card_ids(parse_cards(cards))
    return combo_index(ids[0], ids[1])


# --- exact equities -----------------------------------------------------------------------------


def test_hero_vs_each_is_exact_on_the_river_and_matches_monte_carlo_on_the_turn():
    hero = parse_cards("AsKd")
    river = parse_cards("2c7d9h3sKh")
    e = hero_vs_each(card_ids(hero), card_ids(river))
    for opp in ("QcQd", "Kc2d", "8s6s", "KsAh"):
        exact = equity_vs_range(hero, [tuple(parse_cards(opp))], river, samples=10).equity
        assert e[combo(opp)] == pytest.approx(exact)
    assert np.isnan(e[combo("AsQc")]) and np.isnan(e[combo("2c3c")])
    turn = parse_cards("2c7d9h3s")
    e = hero_vs_each(card_ids(hero), card_ids(turn))
    r = equity_vs_range(hero, [tuple(parse_cards("9c9s"))], turn, samples=4000, seed=3)
    assert abs(e[combo("9c9s")] - r.equity) < 4 * r.std_error + 1e-9


def test_holdings_vs_ranges_matches_the_engine():
    river = parse_cards("2c7d9h3sKh")
    w = np.stack([combo_weights(top_range(0.3)), np.ones(1326)])
    eq = holdings_vs_ranges(card_ids(river), w)
    for h in ("AsKd", "QcQd", "8s6s", "4d5d"):
        cards = parse_cards(h)
        assert eq[0, combo(h)] == pytest.approx(equity_vs_range(cards, top_range(0.3), river, samples=2000).equity)
        assert eq[1, combo(h)] == pytest.approx(equity_vs_range(cards, RANDOM_RANGE, river, samples=2000).equity)
    turn = parse_cards("2c7d9h3s")
    eq = holdings_vs_ranges(card_ids(turn), w[:1])
    r = equity_vs_range(parse_cards("AsKd"), top_range(0.3), turn, samples=6000, seed=1)
    assert abs(eq[0, combo("AsKd")] - r.equity) < 4 * r.std_error


def test_hero_vs_each_and_holdings_vs_ranges_agree():
    turn = card_ids(parse_cards("2c7d9h3s"))
    hero = combo("AsKd")
    onehot = np.zeros((1, 1326))
    onehot[0, hero] = 1.0
    theirs = holdings_vs_ranges(turn, onehot)[0]
    ours = hero_vs_each(card_ids(parse_cards("AsKd")), turn)
    ok = ~np.isnan(ours)
    assert np.allclose(1.0 - theirs[ok], ours[ok], atol=1e-12)


def test_ehs_all_matches_the_scalar_lookup():
    cards = _cards_for(DEFAULT_CONFIG)
    rng = np.random.default_rng(5)
    for n in (3, 4, 5):
        board = [int(x) for x in rng.choice(52, n, replace=False)]
        e = ehs_all(cards.tables, board)
        for h in rng.choice(1326, 60, replace=False).tolist():
            a, b = (int(x) for x in COMBO_CARDS[h])
            if a in board or b in board:
                assert np.isnan(e[h])
            else:
                assert e[h] == cards.ehs([a, b], board)


# --- CFRBot policy queries ---------------------------------------------------------------------


HOLES = (hole("Kd9d"), hole("Ts8s"))
BOARD = parse_cards("9h7d2c5s3d")


def _spot_views():
    """A few CFRBot (seat 0) views along a hand against ScriptBot (off-tree bets)."""
    cfr = CFRBot(BP, 11)
    return [v for p, v, a in play_fixed([cfr, ScriptBot()], HOLES, BOARD, button=1) if p == 0]


def test_range_policy_is_a_distribution_and_matches_single_holding_queries():
    bot = CFRBot(BP, 0)
    for v in _spot_views():
        live = np.array([c for c in range(1326) if not set(COMBO_CARDS[c].tolist()) & set(card_ids(v.board))])
        actions, probs = bot.range_policy(v, live)
        assert probs.shape == (len(live), len(actions))
        assert (probs >= 0).all() and np.allclose(probs.sum(axis=1), 1.0)
        for k in (0, 17, len(live) - 1):
            cards = tuple(Card.from_id(int(x)) for x in COMBO_CARDS[live[k]])
            single = bot.policy(v, cards)
            assert set(single) <= set(actions)
            for a, p in single.items():
                assert p == pytest.approx(probs[k, actions.index(a)])


def test_range_policy_at_the_root_is_the_blueprint_strategy():
    bot = CFRBot(BP, 0)
    s = new_hand(button=0, hole_cards=HOLES)
    v = public_view(s, 0)
    probs = bot.policy(v, HOLES[0])
    bucket = bot.cards.bucket_cards(HOLES[0], ())
    sigma = BP.strategy(0, bucket)
    expected = collections.defaultdict(float)
    for j, p in enumerate(sigma):
        expected[bot._real_action(v, j, node=0)] += float(p)
    assert probs == pytest.approx(dict(expected))


def test_policy_queries_have_no_side_effects_on_the_bot():
    bot = CFRBot(BP, 7)
    views = _spot_views()
    bot.act(views[0], HOLES[0])
    rng_state = copy.deepcopy(bot._rng.bit_generator.state)
    walks = copy.deepcopy(bot._walks)
    live = np.array([c for c in range(1326) if not set(COMBO_CARDS[c].tolist()) & set(card_ids(BOARD))])
    for v in views:
        bot.range_policy(v, live[::50])
    assert bot._rng.bit_generator.state == rng_state
    assert repr(bot._walks) == repr(walks)


def _sequence_check(n_hands: int, button: int, tolerance_sd: float = 4.5) -> None:
    """Empirical frequency of each full CFRBot action sequence vs the product of policy() probabilities."""
    seen = collections.Counter()
    pred: dict[tuple, float] = {}
    for seed in range(n_hands):
        cfr = CFRBot(BP, seed)
        seq, prob = [], 1.0
        for p, v, a in play_fixed([cfr, ScriptBot()], HOLES, BOARD, button):
            seq.append((p, str(a)))
            if p == 0:
                prob *= cfr.policy(v, HOLES[0]).get(a, 0.0)
        key = tuple(seq)
        seen[key] += 1
        assert pred.setdefault(key, prob) == pytest.approx(prob)
        assert prob > 0, f"played a sequence the policy calls impossible: {key}"
    for key, count in seen.most_common(6):
        p = pred[key]
        sd = (n_hands * p * (1 - p)) ** 0.5
        assert abs(count - n_hands * p) <= tolerance_sd * sd + 2, (key, count, n_hands * p)


def test_policy_matches_sampled_play_with_mixed_translations():
    _sequence_check(250, button=1)


@pytest.mark.slow
def test_policy_matches_sampled_play_long():
    for button in (0, 1):
        _sequence_check(3000, button)


# --- equity-family oracle ----------------------------------------------------------------------


def _random_views(opponent: Bot, n_hands: int, seed: int):
    """Opponent views (with its seat) from hands against RandomBot."""
    out = []
    for i, deal in enumerate(unit_seeds(seed, n_hands)):
        rnd = RandomBot(deal[1])
        opponent.reseed(deal[2])
        holder = []
        orig = opponent.act

        def spy(v, h, orig=orig, holder=holder):
            holder.append(v)
            return orig(v, h)

        opponent.act = spy
        try:
            play_hand([rnd, opponent], deal[0], button=i % 2)
        finally:
            del opponent.act
        out += holder
    return out


@pytest.mark.parametrize("make", [EquityBot, BayesBot])
def test_vectorised_equity_oracle_equals_act_per_holding(make):
    bot = make(seed=1)
    views = [v for v in _random_views(bot, 25, 31) if v.street != Street.PREFLOP][:8]
    assert views
    oracle = EquityFamilyOracle(bot, flop_runouts=60)
    oracle.begin_hand()
    for v in views:
        live = np.array([c for c in range(1326) if not set(COMBO_CARDS[c].tolist()) & set(card_ids(v.board))])[::7]
        a1, p1 = oracle.range_probs(v, live)
        a2, p2 = oracle.per_holding(v, live)
        m1 = [a1[j] for j in p1.argmax(axis=1)]
        m2 = [a2[j] for j in p2.argmax(axis=1)]
        assert m1 == m2


def test_equity_oracle_never_touches_the_rng_and_snapshot_is_independent():
    bot = BayesBot(seed=3)
    oracle = EquityFamilyOracle(bot, flop_runouts=40)
    oracle.begin_hand()
    state = bot._rng.getstate()
    views = [v for v in _random_views(BayesBot(seed=3), 6, 8) if v.street != Street.PREFLOP][:3]
    for v in views:
        oracle.range_probs(v, np.arange(50) + 600)
        oracle.per_holding(v, np.arange(5) + 600)
    assert bot._rng.getstate() == state
    assert oracle._snap is not bot and oracle._snap.model is not bot.model


def test_action_probs_one_shot_forms():
    v = _spot_views()[0]
    assert sum(action_probs(CFRBot(BP, 0), v, HOLES[0]).values()) == pytest.approx(1.0)
    assert action_probs(CallStationBot(), v, HOLES[0]) == {Call() if v.legal.can_call else Check(): 1.0}
    assert isinstance(make_oracle(PotOddsBot(seed=1)), EquityFamilyOracle)
    assert isinstance(make_oracle(CFRBot(BP, 0)), CFROracle)
    assert isinstance(make_oracle(RandomBot(1)), RandomBotOracle)
    assert isinstance(make_oracle(AlwaysFoldBot()), SnapshotOracle)


def test_rebuild_state_reproduces_the_public_view():
    for v in _spot_views():
        assert public_view(rebuild_state(v), v.seat) == v


# --- no side effects: a match with queries equals a replay without them -------------------------


class Replayer(Bot):
    """Plays a recorded list of actions (no queries, no thinking)."""

    def __init__(self, actions):
        self.actions = list(actions)

    def act(self, state, hole_cards):
        return self.actions.pop(0)


def _logs_with_and_without_queries(make_opponent, n_pairs=6, seed=77, preflop="ev"):
    """Play LBR vs opponent; then replay LBR's actions with a Replayer vs a fresh opponent."""
    def run(lbr_factory):
        opp = make_opponent()
        lbr = lbr_factory(opp)
        logs = []
        for i, (deal, _a, seed_b) in enumerate(unit_seeds(seed, n_pairs)):
            opp.reseed(seed_b)
            logs.append(play_hand([lbr(i), opp] if i % 2 == 0 else [opp, lbr(i)], deal, button=i % 3 % 2))
        return logs, opp

    holder = {}

    def with_lbr(opp):
        bot = LBRBot(make_oracle(opp), preflop=preflop)
        return lambda i: bot

    logs1, opp1 = run(with_lbr)
    lbr_actions = [[r.action for r in log.actions if r.player == (0 if i % 2 == 0 else 1)]
                   for i, log in enumerate(logs1)]

    def without(opp):
        return lambda i: Replayer(lbr_actions[i])

    logs2, opp2 = run(without)
    holder["opps"] = (opp1, opp2)
    return logs1, logs2, holder


@pytest.mark.parametrize("make_opponent", [functools.partial(CFRBot, BP, 5), functools.partial(EquityBot, seed=2),
                                           functools.partial(BayesBot, seed=2), functools.partial(PotOddsBot, seed=2)],
                         ids=["CFRBot", "EquityBot", "BayesBot", "PotOddsBot"])
@pytest.mark.parametrize("preflop", ["ev", "call"])
def test_queries_do_not_change_the_game(make_opponent, preflop):
    logs1, logs2, holder = _logs_with_and_without_queries(make_opponent, preflop=preflop)
    assert logs1 == logs2
    o1, o2 = holder["opps"]
    if isinstance(o1, BayesBot):  # it learned the same things from both runs
        assert o1.model.lines() == o2.model.lines()


# --- no leak ------------------------------------------------------------------------------------


@pytest.mark.parametrize("preflop", ["ev", "call"])
def test_lbr_actions_do_not_depend_on_the_opponents_cards(preflop):
    # The opponent must act the same whatever it holds (a call station), so any change in
    # LBR's actions could only come from seeing the opponent's cards.
    opp_cls = CallStationBot
    lbr_cards = hole("AhJc")
    board = parse_cards("Jd8c4s2h9s")
    for button in (0, 1):
        sequences = set()
        for opp in ("7c7d", "KsQs", "3h2d", "AdJs"):
            opponent = opp_cls()
            lbr = LBRBot(make_oracle(opp_cls()), preflop=preflop)
            moves = [(p, str(a)) for p, v, a in play_fixed([lbr, opponent], (lbr_cards, hole(opp)), board, button)
                     if p == 0]
            sequences.add(tuple(moves))
        assert len(sequences) == 1


def test_lbr_only_ever_holds_public_information_and_its_cards():
    from test_info_leak import cards_in
    opp = CallStationBot()
    lbr = LBRBot(make_oracle(opp))
    for i, (deal, _a, _b) in enumerate(unit_seeds(5, 6)):
        seen = []
        orig = lbr._start

        def spy(view, hole, orig=orig, seen=seen):
            m = orig(view, hole)
            seen.append(m)
            return m

        lbr._start = spy
        log = play_hand([lbr, opp], deal, button=i % 2)
        del lbr._start
        mine = set(log.hole_cards[0]) | set(log.board)
        for m in seen:
            assert cards_in(m.hole) | cards_in(m.records) <= mine


# --- sanity values ------------------------------------------------------------------------------


@pytest.mark.parametrize("folder", [AlwaysFoldBot, FoldOrCheckBot])
def test_lbr_vs_always_fold_is_exactly_750_mbb_per_game(folder):
    # Blinds 50/100. LBR in the big blind: the button folds, +50. LBR on the button: it limps
    # (preflop it always calls), the big blind folds (or checks and folds to LBR's flop bet), +100.
    # Mean over both seats: 75 chips = 0.75 bb = 750 mbb/g, with no variance at all.
    s = lbr_session(folder, 10, 1, 0)
    assert all(h.net_bb == (0.5 if h.lbr_seat != h.button else 1.0) for h in s.hands)
    assert s.mbb_per_game == pytest.approx(750.0)


def _mean_ci(values):
    m = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return m, m - 1.96 * se


def test_lbr_beats_call_station_by_a_lot():
    s = lbr_session(CallStationBot, 25, 2, 0)
    m, lo = _mean_ci(s.pair_results)
    assert s.mbb_per_game > 3000 and lo > 0


def test_lbr_beats_always_raise():
    s = lbr_session(AlwaysRaiseBot, 100, 3, 0)
    m, lo = _mean_ci(s.pair_results)
    assert s.mbb_per_game > 0 and lo > 0


@pytest.mark.parametrize("make_opponent", [functools.partial(RandomBot, 4), AlwaysRaiseBot,
                                           functools.partial(CFRBot, BP, 4)], ids=["Random", "AlwaysRaise", "CFR"])
def test_lbr_acts_legally_including_unequal_stacks(make_opponent):
    opp = make_opponent()
    lbr = LBRBot(make_oracle(opp))
    for i, (deal, _a, seed_b) in enumerate(unit_seeds(9, 16)):
        opp.reseed(seed_b)
        stacks = [(10_000, 10_000), (10_000, 1_250), (730, 10_000), (180, 2_000)][i % 4]
        play_hand([lbr, opp] if i % 2 else [opp, lbr], deal, button=i % 3 % 2, stacks=stacks)  # raises if illegal


def test_lbr_session_is_deterministic_and_deals_are_shared_across_opponents():
    a = lbr_session(functools.partial(CFRBot, BP, 0), 4, 123, 1)
    b = lbr_session(functools.partial(CFRBot, BP, 0), 4, 123, 1)
    c = lbr_session(CallStationBot, 4, 123, 1)
    assert [h.net_bb for h in a.hands] == [h.net_bb for h in b.hands]
    assert [h.deal_seed for h in a.hands] == [h.deal_seed for h in c.hands]
    assert [(h.lbr_seat, h.button) for h in a.hands] == [(h.lbr_seat, h.button) for h in c.hands]


def test_debug_logs_equity_pot_odds_and_evs():
    lbr = LBRBot(make_oracle(CallStationBot()), preflop="call", debug=True)  # so the hand reaches the flop
    seen = []
    for p, v, a in play_fixed([lbr, CallStationBot()], (hole("AhAd"), hole("7c2d")), parse_cards("Ac8c4s2h9s"), 0):
        if p == 0 and v.street != Street.PREFLOP:
            d = lbr.last_decision
            seen.append(d)
            assert d.action == a and dict(d.evs)
            assert 0.9 < d.equity <= 1.0
    assert seen and any(name.startswith("bet") for name, _ in seen[0].evs)


def test_hero_vs_each_preflop_is_a_close_deterministic_estimate():
    e1 = hero_vs_each(card_ids(parse_cards("AsAd")), [])
    e2 = hero_vs_each(card_ids(parse_cards("AsAd")), [])
    assert np.array_equal(e1, e2, equal_nan=True)
    assert e1[combo("KcKh")] == pytest.approx(0.82, abs=0.03)  # AA vs KK is about 82%
    assert e1[combo("7c2h")] == pytest.approx(0.88, abs=0.03)


class ShoveBot(Bot):
    """All-in whenever it may bet or raise, else call/check."""

    def act(self, state, hole_cards):
        legal = state.legal
        if legal.can_bet or legal.can_raise:
            return aggressive_action(legal, legal.max_to)
        return Call() if legal.can_call else Check()


def test_fold_call_preflop_option_folds_junk_to_shoves_and_stays_legal():
    # Facing a 100bb shove from a random range, a call needs ~49.5% equity: about half the hands fold.
    opp = ShoveBot()
    lbr = LBRBot(make_oracle(opp), preflop="fold_call")
    folds = calls = 0
    for i, (deal, _a, _b) in enumerate(unit_seeds(12, 30)):
        log = play_hand([lbr, opp], deal, button=i % 2)
        mine = [r.action for r in log.actions if r.player == 0 and r.street == Street.PREFLOP]
        folds += Fold() in mine
        calls += Call() in mine
    assert folds > 3 and calls > 3
    always = LBRBot(make_oracle(opp), preflop="call")  # the paper's rule: calls everything
    for i, (deal, _a, _b) in enumerate(unit_seeds(12, 10)):
        log = play_hand([always, opp], deal, button=i % 2)
        assert Fold() not in [r.action for r in log.actions if r.player == 0]
    with pytest.raises(ValueError):
        LBRBot(make_oracle(opp), preflop="raise")


def test_shared_equity_cache_eviction_keeps_requested_rows(monkeypatch):
    """Regression: a full LRU cache must not evict an already-cached row that is part of the request."""
    import nutsfinder.eval.lbr as lbr_mod

    cache = lbr_mod.OrderedDict()
    monkeypatch.setattr(lbr_mod, "_SHARED_EQUITIES_MAX", 1)
    oracle = lbr_mod.EquityFamilyOracle(EquityBot(seed=1))
    oracle._eq_cache = cache
    board = tuple(parse_cards("As Kd 7c 2h"))
    first = oracle._equity_arrays(board, [0.5])
    both = oracle._equity_arrays(board, [0.5, 0.2])  # 0.5 is the oldest entry; 0.2 is new
    assert both[0] is first[0] and len(both) == 2



# --- preflop="ev" and the preflop equity table ---------------------------------------------


@pytest.mark.parametrize("hero, frac", [("AsAh", 1.0), ("7c2d", 1.0), ("KsQs", 0.1), ("JhTd", 0.3),
                                        ("5d5c", 0.05), ("9h8h", 0.15)])
def test_preflop_table_matches_equity_vs_range(hero, frac):
    cards = parse_cards(hero)
    rng = RANDOM_RANGE if frac >= 1 else top_range(frac)
    e = preflop_vs_each(card_ids(cards))
    w = combo_weights(rng)
    ok = ~np.isnan(e) & (w > 0)
    est = float(w[ok] @ e[ok] / w[ok].sum())
    ref = equity_vs_range(cards, rng, samples=200_000, seed=5)
    # table SE <= 0.16 points, reference SE ~0.1 points
    assert abs(est - ref.equity) < 0.004


def test_preflop_table_is_an_exact_suit_relabelling():
    a = preflop_vs_each(card_ids(parse_cards("AsKs")))
    b = preflop_vs_each(card_ids(parse_cards("AhKh")))
    assert a[combo("QsJs")] == b[combo("QhJh")] and a[combo("QhJh")] == b[combo("QsJs")]
    assert np.isnan(a[combo("AsQd")]) and not np.isnan(a[combo("AhQd")])
    assert a[combo("KcKd")] == pytest.approx(0.34, abs=0.02)  # AKs vs KK (blocked K) is about 34%


def _first_preflop(lbr, opp, hero, villain, button):
    for p, v, a in play_fixed([lbr, opp], (hole(hero), hole(villain)), parse_cards("2c3d8h9sTd"), button):
        if p == 0 and v.street == Street.PREFLOP:
            return a, lbr.last_decision
    raise AssertionError("LBR never acted preflop")


def test_ev_preflop_raises_strong_hands_against_a_wide_range():
    for button in (0, 1):
        lbr = LBRBot(make_oracle(CallStationBot()), debug=True)  # "ev" is the default
        a, d = _first_preflop(lbr, CallStationBot(), "AhAd", "7c2d", button)
        assert isinstance(a, Raise), a
        evs = dict(d.evs)
        assert max(v for k, v in evs.items() if k.startswith("raise")) > max(evs.get("call", -1e9), evs.get("check", -1e9))


def test_ev_preflop_folds_junk_to_a_big_raise_and_calls_strong_hands():
    opp = ShoveBot()  # button 1: the opponent is the small blind and shoves 100bb first
    a, d = _first_preflop(LBRBot(make_oracle(opp), debug=True), opp, "7c2d", "AsKd", 1)
    assert a == Fold() and dict(d.evs)["call"] < 0
    a, _ = _first_preflop(LBRBot(make_oracle(opp)), opp, "AhAd", "7s2h", 1)
    assert a in (Call(),) or isinstance(a, Raise)


def test_ev_preflop_is_deterministic_and_legal_with_odd_stacks():
    for opp_cls in (RandomBot, AlwaysRaiseBot, CallStationBot):
        for i, (deal, _a, _b) in enumerate(unit_seeds(31, 12)):
            stacks = [(10_000, 10_000), (10_000, 1_250), (730, 10_000), (180, 2_000)][i % 4]
            logs = []
            for _ in range(2):
                opp = opp_cls(seed=7) if opp_cls is RandomBot else opp_cls()
                lbr = LBRBot(make_oracle(opp), preflop="ev")
                logs.append(play_hand([lbr, opp] if i % 2 else [opp, lbr], deal, button=i % 3 % 2, stacks=stacks))
            assert [(r.player, r.action) for r in logs[0].actions] == [(r.player, r.action) for r in logs[1].actions]
