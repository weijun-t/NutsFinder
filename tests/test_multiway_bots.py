"""Multiway bots: equity, EV accounting, legality, determinism, information, learning, speed."""

from __future__ import annotations

import dataclasses
import random
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from nutsfinder.bots.equity_bot import EquityBot
from nutsfinder.bots.multiway import (
    MultiBayesBot,
    MultiCFRBot,
    MultiDecision,
    MultiEquityBot,
    MultiPotOddsBot,
    PooledModels,
    exact_multiway_equity,
    multiway_equity,
    sample_deals,
)
from nutsfinder.bots.multiway.bayes_bot import multiway_opponent_events, stat_key
from nutsfinder.bots.multiway.cfr_bot import FALLBACK_MODE, HU_MODE, heads_up_subgame
from nutsfinder.bots.multiway.common import ev_aggressive, ev_call, expected_winnings
from nutsfinder.bots.multiway.equity import preflop_vs_random, range_weights, top_weights
from nutsfinder.bots.pot_odds import PotOddsBot
from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.cards import FULL_DECK, Card, Deck, parse_cards
from nutsfinder.engine.preflop import class_combos
from nutsfinder.engine.evaluator import evaluate
from nutsfinder.engine.multiway import (
    MultiAlwaysRaiseBot,
    MultiBot,
    MultiCallStationBot,
    MultiHandState,
    MultiRandomBot,
    apply_action,
    new_hand,
    play_hand_multi,
    play_match_multi,
    public_view,
)
from nutsfinder.engine.multiway.bots import to_hu_view

MODEL = Path(__file__).resolve().parents[1] / "models" / "holdem_official" / "latest.npz"
needs_model = pytest.mark.skipif(not MODEL.is_file(), reason="no trained blueprint at models/holdem_official/latest.npz")


def c(s: str) -> tuple[Card, ...]:
    return tuple(parse_cards(s))


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


# --- 1. Multiway equity ------------------------------------------------------------------


def test_three_way_all_in_on_the_turn_matches_exact_enumeration() -> None:
    hero, a, b, board = c("AsJs"), c("QhQd"), c("7c8c"), c("Ks9c6dQs")  # draws vs a set vs draws
    exact = exact_multiway_equity(hero, [a, b], board)
    assert 0 < exact < 1
    r = multiway_equity(hero, board, [range_weights([a]), range_weights([b])], 20_000, seed=3)
    assert abs(r.equity - exact) < 4 * r.std_error + 1e-9


def test_exact_turn_enumeration_by_hand() -> None:
    # Hero AhAd (top set) vs KcKd (second set) vs 2s3s on Ac Kh 7h 9d: 42 rivers. Only the Ks beats
    # the aces (quad kings); a paired board gives aces full over kings full; 2s3s can make nothing.
    hero, a, b, board = c("AhAd"), c("KcKd"), c("2s3s"), c("AcKh7h9d")
    assert exact_multiway_equity(hero, [a, b], board) == pytest.approx(41 / 42)


def test_four_way_flop_matches_exact() -> None:
    hero, opps, board = c("JhTh"), [c("AcKd"), c("9s9d"), c("2h3h")], c("Qh9h4c")
    exact = exact_multiway_equity(hero, opps, board)
    r = multiway_equity(hero, board, [range_weights([o]) for o in opps], 20_000, seed=5)
    assert abs(r.equity - exact) < 4 * r.std_error


def test_river_strengths_agree_with_scalar_evaluator() -> None:
    board = c("2c7d9hJsKc")
    deals = sample_deals(c("AhAd"), board, [None, None, None], 300, rng(1))
    for row in range(0, 300, 37):
        hands = [c("AhAd")] + [tuple(FULL_DECK[x] for x in _combo_cards(int(k))) for k in deals.combos[row]]
        ranks = [evaluate(list(h) + list(board)) for h in hands]
        s = deals.strengths[row]
        for i in range(4):
            for j in range(4):
                assert (ranks[i] > ranks[j]) == (s[i] > s[j])
                assert (ranks[i] == ranks[j]) == (s[i] == s[j])


def _combo_cards(k: int) -> tuple[int, int]:
    from nutsfinder.abstraction.strength import COMBO_CARDS

    return int(COMBO_CARDS[k, 0]), int(COMBO_CARDS[k, 1])


def test_board_plays_splits_pot_equally() -> None:
    board = c("AsKsQsJsTs")  # royal flush on board: everybody ties
    for k in (1, 2, 4, 7):
        r = multiway_equity(c("2c3d"), board, k, 200, seed=1)
        assert r.equity == pytest.approx(1 / (k + 1))
        assert r.std_error == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("k, expected", [(1, 0.852), (2, 0.735), (4, 0.559)])
def test_aces_vs_random_hands_known_values(k: int, expected: float) -> None:
    r = multiway_equity(c("AsAh"), (), k, 40_000, seed=11)
    assert abs(r.equity - expected) < 0.01
    eq, se = preflop_vs_random("AA", k)
    assert abs(eq - expected) < 0.01 and se < 0.01


def test_preflop_table_is_monotone_in_k() -> None:
    for cls in ("AA", "KQs", "72o", "55"):
        values = [preflop_vs_random(cls, k)[0] for k in range(1, 8)]
        assert all(a > b for a, b in zip(values, values[1:]))


def test_sampling_is_seeded_and_respects_card_removal_and_ranges() -> None:
    hero, board = c("AsAh"), c("Kd7c2h")
    narrow = top_weights(0.05)
    a = sample_deals(hero, board, [narrow, None, narrow], 1000, rng(7))
    b = sample_deals(hero, board, [narrow, None, narrow], 1000, rng(7))
    assert np.array_equal(a.strengths, b.strengths) and np.array_equal(a.combos, b.combos)
    d = sample_deals(hero, board, [narrow, None, narrow], 1000, rng(8))
    assert not np.array_equal(a.combos, d.combos)
    known = {x.id for x in hero + board}
    for row in a.combos:
        cards = [x for k in row for x in _combo_cards(int(k))]
        assert len(set(cards)) == 6 and not (set(cards) & known)
    assert all(narrow[k] > 0 for k in a.combos[:, 0]) and all(narrow[k] > 0 for k in a.combos[:, 2])


def test_joint_rejection_is_exact_for_overlapping_ranges() -> None:
    # opp1 in {AhKh, QcQd}, opp2 in {AhJh, 2c2d}: the legal joint deals are equally likely (1/3 each).
    # Sequential dealing would give (AhKh, 2c2d) probability 1/2 instead.
    r1 = range_weights([c("AhKh"), c("QcQd")])
    r2 = range_weights([c("AhJh"), c("2c2d")])
    deals = sample_deals(c("5s6s"), c("9d8d7c"), [r1, r2], 6000, rng(2))
    pairs = Counter(tuple(int(x) for x in row) for row in deals.combos)
    assert len(pairs) == 3
    for count in pairs.values():
        assert abs(count / deals.n - 1 / 3) < 0.03


def test_impossible_ranges_still_give_a_deal() -> None:
    aa = range_weights([c("AcAd")])
    deals = sample_deals(c("AsKs"), (), [aa, aa], 50, rng(0))  # both "hold" the only AA combo left
    assert deals.n == 50
    for row in deals.combos:
        assert len({x for k in row for x in _combo_cards(int(k))}) == 4


def test_subset_shares() -> None:
    deals = sample_deals(c("AsAh"), (), [None] * 3, 2000, rng(4))
    full, one = deals.share_mean(), deals.share_mean([0])
    assert one > full  # fewer opponents, bigger share
    assert deals.share_mean([]) == 1.0


# --- 2. EV accounting -----------------------------------------------------------------------


def _state(stacks, button=0, actions=()) -> MultiHandState:  # type: ignore[no-untyped-def]
    s = new_hand(stacks, button)
    for a in actions:
        s = apply_action(s, s.to_act, a)
    return s


def test_ev_call_heads_up_is_the_hu_formula() -> None:
    s = _state((10_000, 10_000), 0, [Raise(300)])  # SB raises to 300, BB to act facing 200
    v = public_view(s, s.to_act)
    e = 0.4
    ev, winnable = ev_call(v, lambda opps: e if opps else 1.0)
    assert winnable == 600 and ev == pytest.approx(e * (v.pot + 200) - 200)


def test_ev_bet_heads_up_is_the_hu_formula() -> None:
    s = _state((10_000, 10_000), 0, [Call()])  # limp; BB to act may raise
    v = public_view(s, s.to_act)
    f, e = 0.3, 0.55
    ev, p_all = ev_aggressive(v, 400, {s.to_act ^ 1: f}, lambda opps: e if opps else 1.0)
    x, r, pot = 300, 300, v.pot
    assert p_all == pytest.approx(f)
    assert ev == pytest.approx(f * pot + (1 - f) * (e * (pot + x + r) - x))


def test_bet_wins_outright_only_if_everyone_folds() -> None:
    s = _state((10_000,) * 4, 0, [Call(), Call(), Call()])  # 4-way limped, BB to act
    v = public_view(s, s.to_act)
    folds = {j: 0.5 for j in range(4) if j != s.to_act}
    ev, p_all = ev_aggressive(v, 300, folds, lambda opps: 0.0)  # we never win a showdown
    assert p_all == pytest.approx(0.125)
    assert ev == pytest.approx(0.125 * v.pot - 0.875 * 200)


def test_side_pot_winnings_by_hand() -> None:
    # 3 players, seat 1 short (500). Everyone all-in-ish: contributions 500 / 2000 / 2000.
    live = [True, True, True]
    # Main pot 1,500 (all three), side pot 3,000 (seats 0 and 2). Seat 0 is "me" here.
    share = {(): 1.0, (1, 2): 0.30, (2,): 0.60, (1,): 0.7}
    w = expected_winnings(lambda o: share[o], 0, [2000, 500, 2000], live)
    assert w == pytest.approx(1500 * 0.30 + 3000 * 0.60)
    # The short stack's view: only the main pot.
    w1 = expected_winnings(lambda o: {(0, 2): 0.5}[o], 1, [2000, 500, 2000], live)
    assert w1 == pytest.approx(1500 * 0.5)


def test_call_all_in_for_less_prices_only_the_reachable_pot() -> None:
    # BTN raises to 3,000, SB calls, the BB (600 behind after posting 100... i.e. 500) calls all-in.
    s = _state((10_000, 10_000, 600), 0, [Raise(3000), Call()])
    v = public_view(s, 2)
    assert v.to_act == 2 and v.legal.call_amount == 500
    ev, winnable = ev_call(v, lambda o: {(0, 1): 0.25}[o])
    # The BB can only win the main pot: 600 from each of the three players.
    assert winnable == 1800
    assert ev == pytest.approx(0.25 * 1800 - 500)


# --- 3. Legality fuzz: 2..8 players, uneven stacks -----------------------------------------


class FolderBot(MultiBot):
    """Checks when free, folds to any bet."""

    def act(self, view, hole_cards):  # type: ignore[no-untyped-def]
        return Check() if view.legal.can_check else Fold()


def _new_bots(seed: int):  # type: ignore[no-untyped-def]
    return [
        lambda i: MultiPotOddsBot(seed=seed + i, samples=120),
        lambda i: MultiEquityBot(seed=seed + i, samples=120),
        lambda i: MultiBayesBot(seed=seed + i, samples=120, bluff_samples=50),
        lambda i: MultiRandomBot(seed + i),
        lambda i: MultiAlwaysRaiseBot(),
        lambda i: MultiCallStationBot(),
    ]


def test_fuzz_legal_actions_uneven_stacks() -> None:
    r = random.Random(20261124)
    hands = 0
    for t in range(36):
        n = 2 + t % 7
        makers = _new_bots(t * 100)
        bots = [makers[r.randrange(len(makers))](i) for i in range(n)]
        bots[0] = makers[t % 3](0)  # always at least one new bot
        for h in range(5):
            stacks = [r.choice([r.randint(1, 150), r.randint(150, 3000), r.randint(3000, 30_000)]) for _ in range(n)]
            log = play_hand_multi(bots, seed=t * 1000 + h, button=h % n, stacks=stacks)
            assert sum(log.result.net) == 0
            hands += 1
    assert hands == 180


# --- 4. Determinism and reseeding -----------------------------------------------------------


def _field(seed: int, n: int = 5):  # type: ignore[no-untyped-def]
    return [MultiPotOddsBot(seed, samples=150), MultiEquityBot(seed + 1, samples=150),
            MultiBayesBot(seed + 2, samples=150, bluff_samples=50), MultiRandomBot(seed + 3),
            MultiCallStationBot()][:n]


def _key(logs):  # type: ignore[no-untyped-def]
    return [(l.actions, l.result.net) for l in logs]


def test_same_seed_same_match() -> None:
    a = play_match_multi(_field(1), 30, 99)
    b = play_match_multi(_field(1), 30, 99)
    assert _key(a) == _key(b)
    d = play_match_multi(_field(2), 30, 99)
    assert _key(a) != _key(d)


@pytest.mark.parametrize("make", [lambda: MultiPotOddsBot(0, samples=100), lambda: MultiEquityBot(0, samples=100),
                                  lambda: MultiBayesBot(0, samples=100)])
def test_reseed_restarts_the_rng(make) -> None:  # type: ignore[no-untyped-def]
    s = _state((10_000,) * 4, 0, [Call()])
    s = apply_action(s, s.to_act, Call())
    s = apply_action(s, s.to_act, Call())
    s = apply_action(s, s.to_act, Check())
    from nutsfinder.engine.multiway import deal_board

    s = deal_board(s, c("Kd7c2h"))
    v = public_view(s, s.to_act)
    bot = make()
    bot.debug = True
    bot.reseed(5)
    first = bot.act(v, c("QhJh"))
    d1 = bot.last_decision
    bot.reseed(5)
    assert bot.act(v, c("QhJh")) == first
    assert bot.last_decision.evs == d1.evs


# --- 5. No information leaks ----------------------------------------------------------------


def _reachable(obj: object) -> Iterator[object]:
    stack, seen = [obj], set()
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        yield x
        if isinstance(x, Card | str | int | float | bool | type(None) | np.ndarray):
            continue
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            stack.extend(getattr(x, f.name) for f in dataclasses.fields(x))
        elif isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())
        elif isinstance(x, tuple | list | set | frozenset):
            stack.extend(x)
        elif hasattr(x, "__dict__"):
            stack.extend(vars(x).values())


class Recorder(MultiBot):
    """Wraps a bot; records its inputs and checks its memory only holds cards it was shown."""

    def __init__(self, inner: MultiBot) -> None:
        self.inner = inner
        self.inputs: list[tuple[str, object]] = []
        self.allowed: set[Card] = set()

    def act(self, view, hole_cards):  # type: ignore[no-untyped-def]
        for x in _reachable(view):
            assert not isinstance(x, MultiHandState | Deck)
        self.allowed |= set(hole_cards) | set(view.board)
        self.inputs.append(("act", (view, hole_cards)))
        action = self.inner.act(view, hole_cards)
        self._check_memory()
        return action

    def end_hand(self, summary):  # type: ignore[no-untyped-def]
        self.allowed |= {x for h in summary.shown_cards if h for x in h} | set(summary.final_view.board)
        self.inputs.append(("end", summary))
        self.inner.end_hand(summary)
        self._check_memory()

    def _check_memory(self) -> None:
        cards = {x for x in _reachable(vars(self.inner)) if isinstance(x, Card)}
        assert cards <= self.allowed, f"{self.inner!r} holds cards it was never shown: {cards - self.allowed}"


@pytest.mark.parametrize("make", [lambda s: MultiPotOddsBot(s, samples=100, debug=True),
                                  lambda s: MultiEquityBot(s, samples=100, debug=True),
                                  lambda s: MultiBayesBot(s, samples=100, bluff_samples=40, debug=True)])
def test_bots_use_only_the_view_and_their_own_cards(make) -> None:  # type: ignore[no-untyped-def]
    spy = Recorder(make(3))
    bots = [spy, MultiRandomBot(1), MultiCallStationBot(), MultiAlwaysRaiseBot(), MultiRandomBot(2)]
    logs = play_match_multi(bots, 25, 4)
    # Replay: a fresh bot fed exactly the recorded inputs (no engine, no deck) makes the same choices.
    twin = make(3)
    actions = iter(rec.action for log in logs for rec in log.actions if rec.player == 0)
    for kind, arg in spy.inputs:
        if kind == "act":
            assert twin.act(*arg) == next(actions)  # type: ignore[misc]
        else:
            twin.end_hand(arg)  # type: ignore[arg-type]


# --- 6. Two players: sensible, close to the HU bots -----------------------------------------


def _agreement(multi: MultiBot, hu_bot, n_hands: int, seed: int) -> float:  # type: ignore[no-untyped-def]
    """Share of the multi bot's decisions on which the HU bot (asked on the HU view) picks the same kind."""
    same = total = 0

    class Probe(MultiBot):
        def act(self, view, hole_cards):  # type: ignore[no-untyped-def]
            nonlocal same, total
            a = multi.act(view, hole_cards)
            b = hu_bot.act(to_hu_view(view), hole_cards)
            same += type(a) is type(b)
            total += 1
            return a

    play_match_multi([Probe(), MultiRandomBot(seed)], n_hands, seed)
    return same / total


def test_two_player_pot_odds_matches_hu_pot_odds() -> None:
    assert _agreement(MultiPotOddsBot(1, samples=600), PotOddsBot(1, samples=600), 150, 7) > 0.9


def test_two_player_equity_bot_mostly_matches_hu_equity_bot() -> None:
    assert _agreement(MultiEquityBot(1, samples=800), EquityBot(1, samples=800), 150, 8) > 0.8


def test_two_player_sanity_checks() -> None:
    s = _state((10_000, 10_000), 0, [Raise(10_000)])  # SB shoves, BB decides
    v = public_view(s, 1)
    for bot in (MultiPotOddsBot(0), MultiEquityBot(0), MultiBayesBot(0)):
        assert isinstance(bot.act(v, c("7c2d")), Fold)
        assert isinstance(bot.act(v, c("AhAd")), Call)
    s = _state((10_000, 10_000), 0, [Call()])  # limp: BB may check
    v = public_view(s, 1)
    for bot in (MultiPotOddsBot(0), MultiEquityBot(0), MultiBayesBot(0)):
        assert not isinstance(bot.act(v, c("7c2d")), Fold)


def test_multiway_needs_more_than_heads_up_to_continue() -> None:
    # K2o has 51% against one random hand: heads-up it calls a random shove. 6-handed, facing UTG's
    # shove with four players still to act (who the model says call too), it needs 1/6 of a six-way
    # pot and has 14% against five random hands: fold.
    s2 = _state((10_000, 10_000), 0, [Raise(10_000)])
    assert isinstance(MultiPotOddsBot(0).act(public_view(s2, 1), c("Kc2d")), Call)
    s6 = _state((10_000,) * 6, 0, [Raise(10_000)])
    v = public_view(s6, s6.to_act)
    assert isinstance(MultiPotOddsBot(0).act(v, c("Kc2d")), Fold)


def test_debug_record_explains_decision() -> None:
    bot = MultiEquityBot(0, debug=True)
    s = _state((10_000,) * 6, 0, [Raise(300)])
    v = public_view(s, s.to_act)
    action = bot.act(v, c("AhKh"))
    d = bot.last_decision
    assert isinstance(d, MultiDecision) and d.action == action and d.n_opponents == 5
    assert {"fold", "call"} <= {name for name, _ in d.evs}
    assert d.pot_odds is not None and "P(all 5 fold)" in str(d)


# --- 7. MultiBayesBot learns per opponent ------------------------------------------------------


def test_pooled_models_start_at_the_priors_and_shrink_sparse_seats() -> None:
    m = PooledModels()
    st = m.stat(3, "fold_post@mw")
    assert (st.alpha, st.beta) == (1.0, 9.0)
    # Five seats that fold half the time, plenty of data: a new seat starts near 50%, strongly.
    for seat in range(5):
        m.observe(seat, [("fold_post@mw", i % 2 == 0) for i in range(200)])
    new = m.stat(9, "fold_post@mw")
    assert abs(new.mean - 0.5) < 0.02 and new.alpha + new.beta >= 50
    # Seats that differ a lot: weak pooling, own data dominate fast.
    d = PooledModels()
    for seat in range(4):
        d.observe(seat, [("fold_post@mw", seat % 2 == 0) for _ in range(100)])
    assert d.population("fold_post@mw")[1] == pytest.approx(2.0)
    assert d.stat(0, "fold_post@mw").mean > 0.9 and d.stat(1, "fold_post@mw").mean < 0.1


def test_bayes_learns_a_station_and_a_folder_apart() -> None:
    bayes = MultiBayesBot(0, samples=150, bluff_samples=50)
    play_match_multi([bayes, MultiCallStationBot(), FolderBot(), MultiCallStationBot()], 200, 3)
    m = bayes.models
    station = [m.stat(1, k) for k in ("fold_pre@mw", "fold_post@mw", "fold_post@hu", "fold_pre@hu")]
    folder = [m.stat(2, k) for k in ("fold_pre@mw", "fold_post@mw", "fold_post@hu", "fold_pre@hu")]
    assert sum(s.n for s in station) > 20 and sum(s.n for s in folder) > 20
    assert all(s.mean < 0.1 for s in station if s.n >= 5)
    assert any(s.mean > 0.5 for s in folder if s.n >= 5)
    assert m.stat(2, "vpip").mean < m.stat(1, "vpip").mean


def test_bayes_without_learning_plays_like_equity_bot() -> None:
    a = play_match_multi([MultiBayesBot(4, learn=False, samples=150), MultiRandomBot(1), MultiCallStationBot(),
                          MultiAlwaysRaiseBot()], 30, 6)
    b = play_match_multi([MultiEquityBot(4, samples=150), MultiRandomBot(1), MultiCallStationBot(),
                          MultiAlwaysRaiseBot()], 30, 6)
    assert _key(a) == _key(b)


def test_bayes_uses_shown_cards_only_of_players_who_showed() -> None:
    bayes = MultiBayesBot(0, samples=100, bluff_samples=30)
    logs = play_match_multi([bayes, MultiCallStationBot(), MultiAlwaysRaiseBot(), FolderBot()], 40, 1)
    m = bayes.models
    assert m.stat(3, "bluff").n == 0  # the folder never reaches a showdown
    showdowns_raiser = sum(1 for l in logs if l.show_order and not l.folded[2])
    assert m.stat(2, "bluff").n >= min(1, showdowns_raiser)


def test_opponent_events_split_heads_up_and_multiway() -> None:
    s = _state((10_000,) * 3, 0, [Raise(300), Fold()])  # BTN raises, SB folds, BB to act (now HU)
    s = apply_action(s, s.to_act, Fold())
    v = public_view(s, 0)
    ev = multiway_opponent_events(v)
    assert (stat_key("agg_pre", "mw"), False) in ev[1]  # SB decided with 3 live
    # The BB folded with 2 live, but our raise was made 3-handed, and that is the context the bot
    # reads when it sizes a bet: fold events take the context of our bet.
    assert (stat_key("fold_pre", "mw"), True) in ev[2]
    assert ("vpip", False) in ev[1] and ("vpip", False) in ev[2]
    # Heads-up from our bet on: a flop bet into one opponent is recorded as @hu.
    s = _state((10_000,) * 3, 0, [Call(), Fold(), Check()])  # BTN limps, SB folds, BB checks
    from nutsfinder.engine.multiway import deal_board

    s = deal_board(s, c("Kd7c2h"))
    s = apply_action(s, s.to_act, Check())  # BB checks
    s = apply_action(s, s.to_act, Bet(100))  # we bet
    s = apply_action(s, s.to_act, Fold())
    assert (stat_key("fold_post", "hu"), True) in multiway_opponent_events(public_view(s, 0))[2]


# --- 8. MultiCFRBot -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def blueprint():  # type: ignore[no-untyped-def]
    if not MODEL.is_file():
        pytest.skip("no trained blueprint")
    from nutsfinder.abstraction.blueprint import Blueprint

    return Blueprint.load(MODEL)


def test_heads_up_subgame_rules() -> None:
    six = (10_000,) * 6
    # button 0: SB 1, BB 2, UTG 3, MP 4, CO 5. Folds to CO, CO opens, BTN and SB fold: CO vs BB.
    s = _state(six, 0, [Fold(), Fold(), Raise(250), Fold(), Fold()])
    hu_view, why = heads_up_subgame(public_view(s, 2))
    assert hu_view is not None and hu_view.seat == 1 and hu_view.pot == 350, why
    # Same, but the SB called before folding later? Instead: SB limps -> 3 live -> no.
    s = _state(six, 0, [Fold(), Fold(), Raise(250), Fold(), Call()])
    assert heads_up_subgame(public_view(s, 2))[0] is None
    # Blind vs blind: folds to the SB, SB raises -> wrong postflop order -> fallback.
    s = _state(six, 0, [Fold(), Fold(), Fold(), Fold(), Raise(300)])
    assert heads_up_subgame(public_view(s, 2))[0] is None
    # UTG raises, CO calls, the rest fold: neither is the BB -> fallback.
    s = _state(six, 0, [Raise(250), Fold(), Call(), Fold(), Fold(), Fold()])
    assert heads_up_subgame(public_view(s, s.to_act or 3))[0] is None


@needs_model
def test_cfr_two_players_is_the_hu_cfr_bot(blueprint) -> None:  # type: ignore[no-untyped-def]
    from nutsfinder.bots.cfr_bot import CFRBot
    from nutsfinder.engine.multiway import HUBotAdapter

    a = play_match_multi([MultiCFRBot(blueprint, seed=3), MultiRandomBot(1)], 40, 5)
    b = play_match_multi([HUBotAdapter(CFRBot(blueprint, seed=3)), MultiRandomBot(1)], 40, 5)
    assert _key(a) == _key(b)


@needs_model
def test_cfr_multiway_records_modes_and_stays_legal(blueprint) -> None:  # type: ignore[no-untyped-def]
    bots = [MultiCFRBot(blueprint, seed=i, debug=True, fallback_samples=120) for i in range(2)]
    field = [MultiRandomBot(5), FolderBot(), FolderBot(), FolderBot()]  # folders: heads-up spots happen
    play_match_multi([bots[0], *field, bots[1]], 60, 9)
    modes = bots[0].mode_counts + bots[1].mode_counts
    assert modes[HU_MODE] > 0 and modes[FALLBACK_MODE] > 0
    assert all(m == FALLBACK_MODE or m.startswith(HU_MODE) for m in modes)


@needs_model
def test_cfr_heads_up_spot_uses_blueprint(blueprint) -> None:  # type: ignore[no-untyped-def]
    six = (10_000,) * 6
    s = _state(six, 0, [Fold(), Fold(), Raise(250), Fold(), Fold()])
    bot = MultiCFRBot(blueprint, seed=0, debug=True)
    action = bot.act(public_view(s, 2), c("AhKd"))
    assert bot.last_mode.startswith(HU_MODE) and bot.last_decision.mode == bot.last_mode
    assert s.to_act == 2 and public_view(s, 2).legal.is_legal(action)


# --- 9. Speed ------------------------------------------------------------------------------


class Timed(MultiBot):
    def __init__(self, inner: MultiBot) -> None:
        self.inner, self.times = inner, []

    def act(self, view, hole_cards):  # type: ignore[no-untyped-def]
        t = time.perf_counter()
        a = self.inner.act(view, hole_cards)
        self.times.append(time.perf_counter() - t)
        return a

    def end_hand(self, summary):  # type: ignore[no-untyped-def]
        self.inner.end_hand(summary)


def _six_max_ms() -> float:
    from nutsfinder.bots.multiway.equity import warm_up

    warm_up()
    timed = Timed(MultiEquityBot(0))
    play_match_multi([timed, MultiCallStationBot(), MultiCallStationBot(), MultiRandomBot(1),
                      MultiCallStationBot(), MultiRandomBot(2)], 40, 3)
    return 1000 * sum(timed.times) / len(timed.times)


def test_equity_bot_speed_gross_regression_guard() -> None:
    # About 10 ms per decision on an idle P-core of the development laptop, but 3-6x more when Windows
    # schedules the process on an efficiency core or other jobs load the machine. This only catches
    # gross regressions; the ~50 ms target is checked by the slow test below, on a quiet machine.
    ms = _six_max_ms()
    assert ms < 300, f"{ms:.1f} ms per decision"


@pytest.mark.slow
def test_equity_bot_speed_target() -> None:
    ms = _six_max_ms()
    assert ms < 50, f"{ms:.1f} ms per decision (target ~50 ms; run on a quiet machine)"


# --- 10. Never crash on a legal state ------------------------------------------------------


def test_equity_bot_survives_ranges_that_cannot_be_dealt_and_logs_it() -> None:
    # Every opponent assumed to hold the top 1% (AA, KK, QQ: 18 combos) while we hold an ace and a king,
    # five opponents: no joint deal exists. The bot must still act legally and say what it did.
    s = _state((10_000,) * 6, 0, [Raise(300), Raise(900), Raise(2700), Call()])
    v = public_view(s, s.to_act)
    bot = MultiEquityBot(0, range_widths=(0.01,), debug=True)
    action = bot.act(v, c("AsKd"))
    assert v.legal.is_legal(action)
    assert any("could not be dealt jointly" in n for n in bot.last_decision.notes)


def test_range_emptied_by_card_removal_becomes_random() -> None:
    aa_kk = range_weights([*class_combos("AA"), *class_combos("KK")])
    deals = sample_deals(c("AsKs"), c("AhAdKhKd2c"), [aa_kk, None], 100, rng(0))
    assert deals.widened == (0,) and deals.n == 100


_BLUEPRINT: list = []


def _shared_blueprint():  # type: ignore[no-untyped-def]
    if not _BLUEPRINT:
        from nutsfinder.abstraction.blueprint import Blueprint

        _BLUEPRINT.append(Blueprint.load(MODEL))
    return _BLUEPRINT[0]


def _fuzz_field(n: int, seed: int):  # type: ignore[no-untyped-def]
    kinds = [lambda i: MultiPotOddsBot(seed + i, samples=30), lambda i: MultiEquityBot(seed + i, samples=30),
             lambda i: MultiBayesBot(seed + i, samples=30, bluff_samples=20),
             lambda i: MultiEquityBot(seed + i, samples=30, range_widths=(0.3, 0.05, 0.01)),  # very narrow
             lambda i: MultiRandomBot(seed + i), lambda i: MultiAlwaysRaiseBot(), lambda i: MultiCallStationBot(),
             lambda i: FolderBot()]
    if MODEL.is_file():  # MultiCFRBot too, when the blueprint is there (model-gated)
        kinds.append(lambda i: MultiCFRBot(_shared_blueprint(), seed=seed + i, fallback_samples=30))
    r = random.Random(seed)
    bots = [kinds[i % 4](i) for i in range(3)] + [kinds[r.randrange(len(kinds))](i) for i in range(3, n)]
    r.shuffle(bots)
    return bots


def _fuzz_many(n_hands: int, seed: int) -> int:
    r = random.Random(seed)
    played = 0
    for table in range(n_hands // 50):
        n = (6, 8)[table % 2]
        bots = _fuzz_field(n, seed * 7919 + table)
        for h in range(50):
            stacks = [r.choice([10_000, 10_000, r.randint(1, 400), r.randint(400, 25_000)]) for _ in range(n)]
            log = play_hand_multi(bots, seed=seed * 100_003 + table * 50 + h, button=h % n, stacks=stacks)
            assert sum(log.result.net) == 0
            played += 1
    return played


def test_fuzz_quick_six_and_eight_seats() -> None:
    assert _fuzz_many(200, 1) == 200


@pytest.mark.slow
def test_fuzz_5000_hands_six_and_eight_seats() -> None:
    assert _fuzz_many(5000, 2) == 5000


# --- 11. Review fixes: all-in gate, fallback accounting, table reproducibility, CFR modes ---------


def _station_reader() -> MultiBayesBot:
    """A MultiBayesBot that has learned five seats never fold and never raise."""
    bot = MultiBayesBot(0)
    for seat in range(1, 6):
        bot.models.observe(seat, [("vpip", True)] * 200 + [(stat_key("fold_pre", "mw"), False)] * 200
                           + [(stat_key("agg_pre", "mw"), False)] * 200)
    return bot


def test_no_deep_preflop_shove_with_a_marginal_hand_against_stations() -> None:
    # Button 5: UTG (seat 2) is first to act; 10,000 behind vs a 150 pot (SPR about 66).
    s = _state((10_000,) * 6, 5)
    v = public_view(s, s.to_act)
    assert abs(preflop_vs_random(c("As5s"), 5)[0] - 0.22) < 0.005  # 22% vs five random hands
    hands = [c(h) for h in ("As5s", "7c7d", "Jh9h", "KhTd")]
    # Without the gate the one-street model shoves hands just above their fair share 100bb deep:
    # once equity beats 1/(k+1), EV(bet) is linear in the size, so the biggest size wins.
    ungated = _station_reader()
    ungated.all_in_max_spr = None
    assert sum(ungated.act(v, h) == Raise(10_000) for h in hands) >= 1
    gated = _station_reader()
    for h in hands:
        assert gated.act(v, h) != Raise(v.legal.max_to)
    assert gated.stats["all_in_gated"] == len(hands)


def test_no_shove_below_fair_share() -> None:
    # Review bug: UTG 8d2d (13.9% vs five random hands, fair share 16.7%) shoved 100bb deep, because
    # "five callers with AA-QQ" gave it 18.7%. Even with the all-in gate off it must not shove now.
    s = _state((10_000,) * 6, 5)
    v = public_view(s, s.to_act)
    for bot in (_station_reader(), MultiEquityBot(0)):
        bot.all_in_max_spr = None
        bot.debug = True
        action = bot.act(v, c("8d2d"))
        assert action != Raise(10_000), str(bot.last_decision)
        assert bot.last_decision.ev("raise to 10000") < 0


def test_all_in_gate_still_allows_low_spr_jams() -> None:
    # Stacks of 600 (6bb): the jam is a small multiple of the pot, so it stays a candidate.
    s = _state((600,) * 6, 5)
    v = public_view(s, s.to_act)
    bot = MultiEquityBot(0)
    assert v.legal.max_to in bot.candidate_sizes(v)


def test_fallback_rows_are_counted_outside_debug() -> None:
    s = _state((10_000,) * 6, 0, [Raise(300), Raise(900), Raise(2700), Call()])
    v = public_view(s, s.to_act)
    bot = MultiEquityBot(0, range_widths=(0.01,))  # impossible ranges, debug off
    bot.act(v, c("AsKd"))
    assert bot.stats["decisions"] == 1 and bot.stats["deals"] >= 1
    assert bot.stats["fallback_rows"] > 0 or bot.stats["widened"] > 0


def test_eight_random_hands_need_no_fallback() -> None:
    deals = sample_deals(c("AsKd"), (), [None] * 7, 3000, rng(1))
    assert deals.fallback_rows == 0


def test_preflop_table_entries_rebuild_exactly() -> None:
    from nutsfinder.bots.multiway.equity import PREFLOP_TABLE_FILE, build_preflop_table

    with np.load(PREFLOP_TABLE_FILE) as z:
        classes = [str(x) for x in z["classes"]]
        eq, se, samples, seed = z["equity"], z["se"], int(z["samples"]), int(z["seed"])
    for cls in ("AA", "72o"):
        i = classes.index(cls)
        e2, s2 = build_preflop_table(samples, seed, classes=[cls])
        assert np.array_equal(e2[0], eq[i]) and np.array_equal(s2[0], se[i])


@needs_model
def test_cfr_uses_only_the_view_and_its_own_cards(blueprint) -> None:  # type: ignore[no-untyped-def]
    spy = Recorder(MultiCFRBot(blueprint, seed=3, debug=True, fallback_samples=100))
    bots = [spy, FolderBot(), MultiRandomBot(1), FolderBot(), MultiCallStationBot()]
    logs = play_match_multi(bots, 40, 4)
    twin = MultiCFRBot(blueprint, seed=3, debug=True, fallback_samples=100)
    actions = iter(rec.action for log in logs for rec in log.actions if rec.player == 0)
    for kind, arg in spy.inputs:
        if kind == "act":
            assert twin.act(*arg) == next(actions)  # type: ignore[misc]
        else:
            twin.end_hand(arg)  # type: ignore[arg-type]


@needs_model
def test_cfr_fuzz_six_and_eight_seats(blueprint) -> None:  # type: ignore[no-untyped-def]
    r = random.Random(77)
    for table in range(6):
        n = (6, 8)[table % 2]
        bots = _fuzz_field(n, 500 + table)
        bots[table % n] = MultiCFRBot(blueprint, seed=table, fallback_samples=30)
        bots[(table + 3) % n] = MultiCFRBot(blueprint, seed=table + 100, fallback_samples=30)
        for h in range(10):
            stacks = [r.choice([10_000, r.randint(1, 400), r.randint(400, 25_000)]) for _ in range(n)]
            log = play_hand_multi(bots, seed=table * 100 + h, button=h % n, stacks=stacks)
            assert sum(log.result.net) == 0
