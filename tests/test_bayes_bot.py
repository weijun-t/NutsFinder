"""Tests for BayesBot (tier 3): the Beta maths, the stat counting, the decisions, memory.

Part 1 checks the Beta-Binomial arithmetic against numbers worked out by
hand. Part 2 builds hands action by action and checks exactly which events
are counted. Part 3 checks how the model changes ranges, fold estimates and
decisions (equity stubbed, as in ``test_ev_bots.py``). Part 4 plays real
hands: convergence, memory across reseeds, equivalence with EquityBot when
not learning, legality, chip conservation and the spy check for leaks.
"""

from __future__ import annotations

import math

import pytest
from test_effective_stack import play_unequal
from test_info_leak import SpyBot, check_spy

from nutsfinder.bots import AlwaysRaiseBot, BayesBot, CallStationBot, EquityBot, PotOddsBot, RandomBot
from nutsfinder.bots.bayes_bot import (
    DEFAULT_PRIORS,
    BetaStat,
    OpponentModel,
    beta_cdf,
    beta_quantile,
    opponent_events,
    public_replay,
)
from nutsfinder.engine.actions import Action, Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import HandState, PublicState, apply_action, deal_board, new_hand, public_view
from nutsfinder.engine.bot import HandSummary
from nutsfinder.engine.cards import Card, parse_cards
from nutsfinder.engine.runner import hand_seeds, play_hand, play_match
from nutsfinder.eval.match import duplicate_match

FLOP, TURN, RIVER = parse_cards("2c7d9h"), parse_cards("Js"), parse_cards("3d")


def hole(s: str) -> tuple[Card, ...]:
    return tuple(parse_cards(s))


def view(state: HandState) -> PublicState:
    assert state.to_act is not None
    return public_view(state, state.to_act)


def play(state: HandState, *moves: tuple[int, Action]) -> HandState:
    """Apply moves, dealing the fixed board whenever the engine asks for cards."""
    for seat, action in moves:
        while state.cards_needed:
            state = deal_board(state, {3: FLOP, 1: TURN if len(state.board) == 3 else RIVER}[state.cards_needed])
        state = apply_action(state, seat, action)
    return state


def events(state: HandState, hero: int) -> list[tuple[str, bool]]:
    return opponent_events(public_view(state, hero))


# --- Part 1: Beta maths -------------------------------------------------------------


def test_posterior_arithmetic_by_hand():
    s = BetaStat(1, 9)
    assert s.mean == 0.1
    for yes in (True, True, False, True, True):  # 4 folds in 5
        s.record(yes)
    assert (s.alpha, s.beta) == (5, 10)  # Beta(1 + 4, 9 + 5 - 4)
    assert s.mean == pytest.approx(1 / 3)  # 5 / 15
    for i in range(45):  # 36 more folds in 45: 40 of 50 in total
        s.record(i < 36)
    assert (s.yes, s.n) == (40, 50)
    assert s.mean == pytest.approx(41 / 60)
    # The posterior mean is a weighted average of prior mean and raw rate.
    w = 10 / 60
    assert s.mean == pytest.approx(w * 0.1 + (1 - w) * 40 / 50)


def test_beta_cdf_known_values():
    for x in (0.1, 0.37, 0.9):
        assert beta_cdf(x, 1, 1) == pytest.approx(x)  # uniform
        assert beta_cdf(x, 1, 9) == pytest.approx(1 - (1 - x) ** 9)
        assert beta_cdf(x, 3, 1) == pytest.approx(x**3)
    # Integer a, b: I_x(a, b) = P(Binomial(a + b - 1, x) >= a). Beta(2, 3) at 0.3:
    # 1 - 0.7^4 - 4 * 0.3 * 0.7^3 = 0.3483.
    assert beta_cdf(0.3, 2, 3) == pytest.approx(0.3483)
    # Large, unequal parameters (both branches of the continued fraction).
    assert beta_cdf(0.5, 300, 300) == pytest.approx(0.5)
    assert beta_cdf(0.2, 41, 19) < 1e-12 < beta_cdf(0.9, 41, 19) < 1


def test_beta_quantile_inverts_the_cdf():
    assert beta_quantile(0.5, 1, 4) == pytest.approx(1 - 0.5**0.25)
    for a, b in ((1, 9), (5, 10), (41, 19), (0.5, 0.5), (200, 3)):
        for q in (0.025, 0.5, 0.975):
            assert beta_cdf(beta_quantile(q, a, b), a, b) == pytest.approx(q, abs=1e-9)


def test_credible_interval_exact_vs_normal():
    s = BetaStat(1, 9)
    lo, hi = s.credible_interval()
    # Beta(1, 9) has a closed-form CDF, so the quantiles too: 1 - (1 - q)^(1/9).
    assert lo == pytest.approx(1 - 0.975 ** (1 / 9))
    assert hi == pytest.approx(1 - 0.025 ** (1 / 9))
    assert 0 < lo < s.mean < hi < 1
    n_lo, _ = s.normal_interval()
    assert n_lo < 0  # the normal approximation leaves [0, 1] near the edge


def test_credible_interval_shrinks_like_one_over_root_n():
    widths = []
    for n in (10, 100, 1000):
        s = BetaStat(5, 5)
        for i in range(n):
            s.record(i % 2 == 0)
        lo, hi = s.credible_interval()
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]
    # Ten times the data: about sqrt(10) times narrower.
    assert widths[1] / widths[2] == pytest.approx(math.sqrt(10), rel=0.1)


def test_model_reset_and_priors():
    m = OpponentModel()
    m.observe([("vpip", True), ("fold_post", False)])
    assert m.hands == 1 and m["vpip"].n == 1
    m.reset()
    assert m.hands == 0 and all(s.n == 0 for s in m.stats.values())
    assert {k: (s.prior_a, s.prior_b) for k, s in m.stats.items()} == DEFAULT_PRIORS
    assert all(a + b == 10 for a, b in DEFAULT_PRIORS.values())
    with pytest.raises(ValueError):
        OpponentModel({"vpip": (1, 1)})


def test_randomness_from_bluff_rate():
    m = OpponentModel()
    assert m.randomness() == 0.0  # at the prior
    for i in range(400):
        m["bluff"].record(i % 2 == 0)  # half the shown bets were weak
    # Beta(201.5, 208.5): mean 0.491, m = (0.491 - 0.15) / 0.35 = 0.976
    assert m.randomness() == pytest.approx((201.5 / 410 - 0.15) / 0.35)
    assert m.randomness() > 0.95
    m.reset()
    for _ in range(100):
        m["bluff"].record(False)  # more honest than the prior: clamp at 0
    assert m.randomness() == 0.0


# --- Part 2: counting from hand-built hands (hero = seat 1 unless noted) -----------------


def test_vpip_counts_a_limp_but_not_a_big_blind_check():
    # Opponent (seat 0, button) limps, hero checks the option.
    s = play(new_hand(button=0), (0, Call()), (1, Check()))
    assert events(s, hero=1)[:2] == [("vpip", True), ("agg_pre", False)]
    # Hero (seat 0, button) limps, opponent (big blind) checks its option: free, so no vpip chance.
    assert events(s, hero=0) == [("agg_pre", False)]


def test_big_blind_raising_after_a_limp_is_voluntary():
    s = play(new_hand(button=0), (0, Call()), (1, Raise(400)), (0, Fold()))
    # Seen from seat 0: the big blind never faced a bet, so no vpip chance; it chose to raise.
    assert events(s, hero=0) == [("agg_pre", True)]


def test_fold_to_our_raise_preflop():
    # Hero (button) raises, opponent (big blind) folds.
    s = play(new_hand(button=1), (1, Raise(300)), (0, Fold()))
    assert events(s, hero=1) == [("vpip", False), ("agg_pre", False), ("fold_pre", True)]


def test_folding_to_the_big_blind_is_not_fold_to_bet():
    # Opponent (button / small blind) folds at once: it faced only the forced blind.
    s = play(new_hand(button=0), (0, Fold()))
    assert events(s, hero=1) == [("vpip", False), ("agg_pre", False)]


def test_raise_then_fold_to_our_three_bet():
    s = play(new_hand(button=0), (0, Raise(300)), (1, Raise(900)), (0, Fold()))
    assert events(s, hero=1) == [
        ("vpip", True),
        ("agg_pre", True),  # the open: it could raise and did
        ("agg_pre", False),  # facing the 3-bet it could raise again and did not
        ("fold_pre", True),
    ]


def test_postflop_aggression_and_fold_buckets():
    # Limp/check. Flop: opponent (BB, seat 1) bets, hero calls. Turn: opponent checks,
    # hero bets pot (400), opponent calls. River: opponent checks, hero shoves 9,400, opponent folds.
    s = play(
        new_hand(button=0),
        (0, Call()), (1, Check()),
        (1, Bet(100)), (0, Call()),
        (1, Check()), (0, Bet(400)), (1, Call()),
        (1, Check()), (0, Bet(9400)), (1, Fold()),
    )  # fmt: skip
    assert events(s, hero=0) == [
        ("agg_pre", False),  # checked its option
        ("agg_post", True),  # flop bet
        ("agg_post", False),  # turn check
        ("agg_post", False),  # facing our bet it could raise, and only called
        ("fold_post", False),  # called our pot-sized bet (fraction 1.0)
        ("agg_post", False),  # river check
        ("fold_post_big", True),  # folded to our all-in (9,400 into 1,200)
    ]


def test_facing_an_all_in_is_no_aggression_chance():
    s = play(new_hand(button=1), (1, Raise(10_000)), (0, Call()))
    assert events(s, hero=1) == [("vpip", True), ("fold_pre_big", False)]


def test_their_own_bet_called_by_us_gives_no_fold_event():
    s = play(new_hand(button=0), (0, Raise(300)), (1, Call()), (1, Check()), (0, Bet(300)), (1, Call()))
    assert [name for name, _ in events(s, hero=1)] == ["vpip", "agg_pre", "agg_post"]


def test_public_replay_matches_the_real_hand():
    s = play(new_hand(button=0, stacks=(10_000, 700)), (0, Raise(300)), (1, Call()), (1, Bet(400)), (0, Call()))
    steps = public_replay(public_view(s, 0))
    assert [rec for _, rec in steps] == list(s.history)
    assert all(state.hole_cards == () for state, _ in steps)  # it never had any hidden cards


def summary_for(state: HandState, seat: int, shown) -> HandSummary:
    from nutsfinder.engine.showdown import settle

    return HandSummary(seat, public_view(state, seat), shown, settle(state))


def test_bluff_events_only_at_showdown():
    # Opponent (seat 0) raises preflop with 72o, bets the flop, checks down: shown at showdown.
    cards = (hole("7c2d"), hole("AsKs"))
    s = new_hand(button=0, hole_cards=cards)
    s = play(s, (0, Raise(300)), (1, Call()), (1, Check()), (0, Bet(300)), (1, Call()),
             (1, Check()), (0, Check()), (1, Check()), (0, Check()))  # fmt: skip
    assert s.is_showdown
    bot = BayesBot(seed=1, bluff_samples=300)
    bot.end_hand(summary_for(s, 1, cards))
    # Preflop: 72o is far below 50% vs a random hand (weak). On the 2c7d9h flop it is two
    # pair, sevens and twos: far above 50% (not weak). So 1 weak bet out of 2 streets.
    assert (bot.model["bluff"].yes, bot.model["bluff"].n) == (1, 2)
    # The same betting without a showdown reveals nothing.
    bot2 = BayesBot(seed=1)
    bot2.end_hand(summary_for(s, 1, (None, None)))
    assert bot2.model["bluff"].n == 0
    assert bot2.model["vpip"].n == bot.model["vpip"].n == 1


# --- Part 3: how the model changes decisions ---------------------------------------------


def set_rate(bot: BayesBot, name: str, yes: int, n: int) -> None:
    stat = bot.model[name]
    stat.yes, stat.n = yes, n


def test_prior_reproduces_equity_bot_ranges():
    s = play(new_hand(button=0), (0, Raise(300)), (1, Raise(900)), (0, Raise(2700)), (1, Raise(8100)))
    s = apply_action(s, 0, Call())
    bayes, eq = BayesBot(), EquityBot()
    # Walk through every prefix of the action and compare both seats' views.
    state = new_hand(button=0)
    for rec in s.history:
        for seat in (0, 1):
            v = public_view(state, seat)
            if v.legal is not None:
                assert bayes.villain_range_width(v) == eq.villain_range_width(v)
        state = apply_action(state, rec.player, rec.action)


def test_aggression_worked_example():
    bot = BayesBot()
    set_rate(bot, "agg_post", 35, 40)  # Beta(40, 10): mean 0.8
    assert bot.model["agg_post"].mean == pytest.approx(0.8)
    s = play(new_hand(button=0), (0, Call()), (1, Check()), (1, Bet(100)), (0, Call()))
    v = public_view(play(s, (1, Check())), 0)  # turn, opponent checked: one bet so far
    assert bot.villain_range_width(v) == 0.7  # 0.8 snapped down
    s = play(s, (1, Bet(200)), (0, Raise(600)), (1, Raise(1800)))
    # Bets: flop bet, turn bet, turn raise = 3 filters: 0.8 * 0.6^0.322 * 0.5^0.322
    expected = 0.8 * 0.6 ** (math.log(0.8) / math.log(0.5)) * 0.5 ** (math.log(0.8) / math.log(0.5))
    assert expected == pytest.approx(0.543, abs=1e-3)
    assert bot.villain_range_width(view(s)) == 0.5  # 0.543 snapped down
    # A maniac's raises mean almost nothing.
    set_rate(bot, "agg_post", 990, 1000)
    assert bot.villain_range_width(view(s)) == 0.85


def test_random_actions_widen_the_range():
    bot = BayesBot()
    s = play(new_hand(button=0), (0, Raise(300)), (1, Call()), (1, Check()), (0, Bet(300)))
    assert bot.villain_range_width(view(s)) == 0.3  # EquityBot: two bets -> top 30%
    set_rate(bot, "bluff", 210, 400)  # Beta(211.5, 198.5): mean 0.52 > 0.5, so m = 1
    assert bot.villain_range_width(view(s)) == 1.0


def test_tight_vpip_narrows_a_call():
    bot = BayesBot()
    s = play(new_hand(button=0), (0, Call()), (1, Check()))  # opponent (seat 0) limped
    assert bot.villain_range_width(public_view(s, 1)) == 1.0
    set_rate(bot, "vpip", 28, 90)  # Beta(35, 65): 0.35, half as loose as the prior's 0.7
    assert bot.villain_range_width(public_view(s, 1)) == 0.5


def checked_to_us() -> PublicState:
    """Flop, pot 1,000 (SB raised to 500, BB called), BB checks: SB (seat 0) to act."""
    s = play(new_hand(button=0), (0, Raise(500)), (1, Call()), (1, Check()))
    return view(s)


def test_fold_model_at_prior_is_equity_bot():
    v = checked_to_us()
    bayes, eq = BayesBot(), EquityBot()
    for to, x in ((500, 500), (1000, 1000), (9500, 9500)):
        assert bayes.fold_and_calling(v, 1.0, to, x) == eq.fold_and_calling(v, 1.0, to, x)


def test_fold_model_against_a_folder_and_a_station():
    v = checked_to_us()
    folder = BayesBot()
    set_rate(folder, "fold_post", 19, 40)  # Beta(20, 30): 0.4
    # An honest folder (m = 0): folds above the prior are not trusted, so this is EquityBot.
    assert folder.fold_and_calling(v, 1.0, 500, 500) == EquityBot().fold_and_calling(v, 1.0, 500, 500)
    set_rate(folder, "bluff", 210, 400)  # its shown bets look random: m = 1
    f, calling = folder.fold_and_calling(v, 1.0, 500, 500)
    assert f == pytest.approx(0.4)  # p* = 0.4, rho = 4: min(0.4, 4 * 1/3)
    # m = 1: its calls are random too. Aggression at the prior 0.5, so only half of the
    # widening applies: q = 2/3 + 1/3 * 1 * 0.5 = 0.833 -> 0.70.
    assert calling == 0.7
    station = BayesBot()
    set_rate(station, "fold_post", 0, 90)  # Beta(1, 99): 0.01, rho = 0.1
    f, calling = station.fold_and_calling(v, 1.0, 500, 500)
    assert f == pytest.approx(0.01)
    # Aggression still at the prior 0.5: q = 2/3 + 1/3 * 0.9 * 0.5 = 0.817 -> 0.70
    assert calling == 0.7
    set_rate(station, "agg_post", 0, 90)  # Beta(5, 95): it (almost) never raises
    # q = 2/3 + 1/3 * 0.9 * 0.95 = 0.952 -> 0.85: nearly everything calls
    assert station.fold_and_calling(v, 1.0, 500, 500)[1] == 0.85
    raiser = BayesBot()
    set_rate(raiser, "fold_post", 0, 90)
    set_rate(raiser, "agg_post", 90, 90)  # never folds, but raises instead of calling
    assert raiser.fold_and_calling(v, 1.0, 500, 500)[1] == 0.6  # about MDF, as EquityBot
    # All-ins use their own bucket, untouched here: still the prior.
    assert folder.fold_and_calling(v, 1.0, 9500, 9500)[0] == pytest.approx(0.1)


def stub(bot, equity: float):
    bot.debug = True
    bot.range_equities = lambda state, hole_cards, widths: {w: (equity, 0.0) for w in widths}
    return bot


def test_bluffs_a_folder_but_not_a_station():
    v = checked_to_us()
    # 10% equity. EquityBot: EV(check) = 100 > EV(bet 500) = 0.1*1000 + 0.9*(200 - 500) = -170.
    assert stub(EquityBot(), 0.1).act(v, hole("4s3s")) == Check()
    folder = stub(BayesBot(), 0.1)
    set_rate(folder, "fold_post", 19, 40)
    # Folds 40%, but honestly (m = 0): not trusted, so it plays like EquityBot and checks.
    assert folder.act(v, hole("4s3s")) == Check()
    set_rate(folder, "bluff", 210, 400)  # a random folder (m = 1)
    # EV(bet 500) = 0.4 * 1000 + 0.6 * (0.1 * 2000 - 500) = 220 > 100.
    assert folder.act(v, hole("4s3s")) == Bet(500)
    assert folder.last_decision.ev("bet 500") == pytest.approx(220)
    station = stub(BayesBot(), 0.1)
    set_rate(station, "fold_post", 0, 90)
    assert station.act(v, hole("4s3s")) == Check()


def test_debug_decision_shows_the_model():
    bot = BayesBot(seed=3, debug=True)
    bot.act(checked_to_us(), hole("Ah9c"))
    d = bot.last_decision
    text = str(d)
    assert "equity" in text and "pot odds" in text and "EV(check)" in text
    assert any(n.startswith("model (0 hands)") for n in d.notes)
    assert any("fold_post_big:" in n and "(0/0)" in n for n in d.notes)


# --- Part 4: real hands ------------------------------------------------------------------


def run(bot: BayesBot, opponent, n_hands: int, seed: int) -> BayesBot:
    play_match([bot, opponent], n_hands=n_hands, master_seed=seed)
    return bot


def test_converges_against_always_raise():
    bot = run(BayesBot(1, samples=30), AlwaysRaiseBot(), 200, seed=5)
    assert bot.model["agg_pre"].mean > 0.95 and bot.model["agg_post"].mean > 0.9
    assert bot.model["vpip"].mean > 0.95
    assert bot.model.randomness() > 0.7  # its shown bets look random
    assert bot.villain_range_width(
        view(play(new_hand(button=0), (0, Raise(300)), (1, Raise(900)), (0, Raise(2700))))
    ) >= 0.85


def test_converges_against_call_station():
    bot = run(BayesBot(1, samples=30), CallStationBot(), 200, seed=6)
    m = bot.model
    assert m["vpip"].mean > 0.95  # it pays to see every flop
    assert m["fold_post"].n + m["fold_post_big"].n > 50
    assert all(m[k].yes == 0 for k in ("fold_pre", "fold_pre_big", "fold_post", "fold_post_big"))
    pooled = (m["fold_post"].alpha + m["fold_post_big"].alpha) / (
        m["fold_post"].alpha + m["fold_post"].beta + m["fold_post_big"].alpha + m["fold_post_big"].beta
    )
    assert pooled < 0.05


def test_intervals_shrink_with_data():
    bot = BayesBot(1, samples=30)
    widths = []
    for chunk in range(3):
        play_match([bot, AlwaysRaiseBot()], n_hands=(20, 80, 200)[chunk], master_seed=chunk)
        lo, hi = bot.model["agg_pre"].credible_interval()
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]


def test_reseed_keeps_the_model_reset_forgets():
    bot = run(BayesBot(1, samples=30), RandomBot(2), 30, seed=7)
    counts = {k: (s.yes, s.n) for k, s in bot.model.stats.items()}
    assert bot.model.hands == 30
    bot.reseed(123)
    assert {k: (s.yes, s.n) for k, s in bot.model.stats.items()} == counts
    copy = bot.fresh_copy()
    assert copy.model.hands == 0 and copy.model is not bot.model and copy.samples == 30
    bot.reset_model()
    assert bot.model.hands == 0 and all(s.n == 0 for s in bot.model.stats.values())


def test_match_reseeding_does_not_wipe_memory():
    # duplicate_match reseeds every hand; the bot must still learn across hands.
    made = []

    def make():
        made.append(BayesBot(samples=30))
        return made[-1]

    duplicate_match(make, AlwaysRaiseBot, 20, master_seed=3)
    assert [b.model.hands for b in made] == [20, 20]  # one bot per room, 20 hands each
    # Counts accumulate across the reseeds (not every hand gives an agg_pre chance:
    # facing our preflop all-in, it cannot raise).
    assert made[0].model["agg_pre"].n > 10


@pytest.mark.parametrize("opponent", [RandomBot, AlwaysRaiseBot, PotOddsBot], ids=lambda b: b.__name__)
def test_not_learning_is_exactly_equity_bot(opponent):
    eq = duplicate_match(lambda: EquityBot(samples=60), opponent, 40, master_seed=11)
    bayes = duplicate_match(lambda: BayesBot(samples=60, learn=False), opponent, 40, master_seed=11)
    assert [g.log for g in bayes.games] == [g.log for g in eq.games]


def test_first_hand_is_equity_bot():
    # Before it has seen anything, a learning BayesBot plays EquityBot's hand.
    for i, seed in enumerate(hand_seeds(4, 20)):
        a = play_hand([BayesBot(i, samples=60), RandomBot(i)], seed, button=i % 2)
        b = play_hand([EquityBot(i, samples=60), RandomBot(i)], seed, button=i % 2)
        assert a == b


def test_learning_is_deterministic():
    a = run(BayesBot(1, samples=30), RandomBot(2), 60, seed=9)
    b = run(BayesBot(1, samples=30), RandomBot(2), 60, seed=9)
    assert {k: (s.yes, s.n) for k, s in a.model.stats.items()} == {
        k: (s.yes, s.n) for k, s in b.model.stats.items()
    }


OPPONENTS = {
    "RandomBot": lambda: RandomBot(1),
    "CallStationBot": CallStationBot,
    "AlwaysRaiseBot": AlwaysRaiseBot,
    "PotOddsBot": lambda: PotOddsBot(2, samples=30),
    "EquityBot": lambda: EquityBot(5, samples=30),
    "BayesBot": lambda: BayesBot(6, samples=30),
}


@pytest.mark.parametrize("opponent", OPPONENTS, ids=str)
def test_unequal_stacks_legal_and_conserved(opponent):
    assert play_unequal(lambda: BayesBot(4, samples=30), OPPONENTS[opponent], n_hands=150, seed=31) > 0


@pytest.mark.slow
@pytest.mark.parametrize("opponent", OPPONENTS, ids=str)
def test_unequal_stacks_legal_and_conserved_default_settings(opponent):
    assert play_unequal(lambda: BayesBot(6), OPPONENTS[opponent], n_hands=3_000, seed=32) > 0


@pytest.mark.slow
@pytest.mark.parametrize("opponent", OPPONENTS, ids=str)
def test_equal_stacks_legal_and_conserved_default_settings(opponent):
    logs = play_match([BayesBot(5), OPPONENTS[opponent]()], n_hands=3_000, master_seed=22)
    for log in logs:
        assert sum(log.result.net) == 0 and sum(log.result.final_stacks) == sum(log.starting_stacks)


class SpyBayes(BayesBot):
    """A BayesBot that records everything it is handed (see ``test_info_leak.py``)."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed, samples=30)
        self.seen_acts: list = []
        self.seen_ends: list = []

    def act(self, state, hole_cards):
        self.seen_acts.append((state, hole_cards))
        return super().act(state, hole_cards)

    def end_hand(self, summary):
        self.seen_ends.append(summary)
        super().end_hand(summary)


def test_bayes_bot_never_sees_hidden_cards():
    spies = [SpyBayes(1), SpyBot(2)]
    for i, seed in enumerate(hand_seeds(12, 150)):
        log = play_hand(spies if i % 2 == 0 else spies[::-1], seed, button=i % 2)
        seat = 0 if i % 2 == 0 else 1
        spy = spies[0]
        assert len(spy.seen_ends) == 1
        check_spy(spy, seat, log, spy.seen_acts, spy.seen_ends[0])
        for s in spies:
            s.seen_acts.clear()
            s.seen_ends.clear()
    assert spies[0].model.hands == 150


def test_model_depends_only_on_public_record_and_shown_cards():
    # Two hands with the same public record (opponent folds, so nothing is shown) but
    # different hole cards must update the model identically.
    def folded_hand(cards):
        s = new_hand(button=0, hole_cards=cards)
        return play(s, (0, Raise(300)), (1, Call()), (1, Bet(400)), (0, Fold()))

    a, b = BayesBot(1), BayesBot(1)
    a.end_hand(summary_for(folded_hand((hole("AsAh"), hole("7c2d"))), 1, (None, None)))
    b.end_hand(summary_for(folded_hand((hole("3s2h"), hole("KcKd"))), 1, (None, None)))
    assert {k: (s.yes, s.n) for k, s in a.model.stats.items()} == {
        k: (s.yes, s.n) for k, s in b.model.stats.items()
    }

