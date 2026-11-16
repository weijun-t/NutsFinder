"""Tests for the EV bots: PotOddsBot (tier 1) and EquityBot (tier 2).

Part 1 builds single situations by hand. To test the *decision rules*
separately from the Monte Carlo, most tests replace the bot's equity
estimate with a fixed number (``estimate_equity`` / ``range_equities``),
then check the action and the logged EVs against numbers worked out by
hand. Part 2 plays real hands against every baseline and checks legality,
chip conservation and determinism.
"""

from __future__ import annotations

import pytest

from nutsfinder.bots import AlwaysRaiseBot, CallStationBot, EquityBot, PotOddsBot, RandomBot
from nutsfinder.bots.common import ev_bet, ev_call, ev_check, pot_odds, raise_to_for_fraction
from nutsfinder.bots.equity_bot import calling_share, fold_probability
from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import HandState, PublicState, apply_action, deal_board, new_hand, public_view
from nutsfinder.engine.cards import Card, parse_cards
from nutsfinder.engine.preflop import PREFLOP_EQUITY
from nutsfinder.engine.runner import play_match

FLOP = parse_cards("2c7d9h")


def hole(s: str) -> tuple[Card, ...]:
    return tuple(parse_cards(s))


def view(state: HandState) -> PublicState:
    assert state.to_act is not None
    return public_view(state, state.to_act)


def flop_pot_1000() -> HandState:
    """SB (seat 0, button) raises to 500, BB calls: pot 1,000 on the flop, BB to act."""
    s = apply_action(new_hand(button=0), 0, Raise(500))
    s = apply_action(s, 1, Call())
    return deal_board(s, FLOP)


def facing_flop_bet() -> HandState:
    """Flop pot 1,000, BB bets 500: SB to act with P = 1,500 and c = 500 (pot odds 25%)."""
    return apply_action(flop_pot_1000(), 1, Bet(500))


def checked_to_raiser() -> HandState:
    """Flop pot 1,000, BB checks: SB (the preflop raiser) to act, nothing to call."""
    return apply_action(flop_pot_1000(), 1, Check())


def with_equity(bot, equity: float):
    bot.estimate_equity = lambda state, hole_cards: (equity, 0.0)
    return bot


# --- The formulas --------------------------------------------------------------


def test_pot_odds_worked_example():
    # Pot was 1,000, opponent bets 500: P = 1,500 (includes the bet), c = 500.
    assert pot_odds(500, 1500) == 0.25
    assert ev_call(0.30, 1500, 500) == pytest.approx(100)
    assert ev_call(0.20, 1500, 500) == pytest.approx(-100)
    assert ev_call(0.25, 1500, 500) == pytest.approx(0)
    assert ev_check(0.4, 1000) == pytest.approx(400)


def test_bet_ev_formula():
    # f * P + (1 - f) * (e * (P + x + r) - x)
    assert ev_bet(1000, 500, 500, 0.3, 0.6) == pytest.approx(0.3 * 1000 + 0.7 * (0.6 * 2000 - 500))
    # A pure bluff with the balanced fold frequency x / (P + x) breaks even.
    f = 500 / 1500
    assert ev_bet(1000, 500, 500, f, 0.0) == pytest.approx(0.0)


def test_fold_model():
    assert calling_share(1000, 1000) == 0.5  # pot-sized bet: MDF 1/2
    assert calling_share(1000, 500) == pytest.approx(2 / 3)
    assert fold_probability(1000, 500, max_fold=0.5) == pytest.approx(1 / 3)
    assert fold_probability(1000, 9000, max_fold=0.3) == 0.3  # capped


def test_fraction_sizing():
    assert raise_to_for_fraction(view(checked_to_raiser()), 2 / 3) == 667
    # Facing 500 into 1,500: pot-sized raise to 500 + 1,500 + 500 = 2,500.
    assert raise_to_for_fraction(view(facing_flop_bet()), 1.0) == 2500


# --- PotOddsBot decisions ------------------------------------------------------------


def test_pot_odds_bot_clear_call():
    assert with_equity(PotOddsBot(), 0.40).act(view(facing_flop_bet()), hole("AsKs")) == Call()


def test_pot_odds_bot_clear_fold():
    assert with_equity(PotOddsBot(), 0.10).act(view(facing_flop_bet()), hole("AsKs")) == Fold()


def test_pot_odds_bot_boundary():
    # Exactly the pot odds (25%): EV(call) = 0, so it folds; just above, it calls.
    spot = view(facing_flop_bet())
    assert with_equity(PotOddsBot(), 0.25).act(spot, hole("AsKs")) == Fold()
    assert with_equity(PotOddsBot(), 0.2501).act(spot, hole("AsKs")) == Call()


def test_pot_odds_bot_raises_for_value():
    assert with_equity(PotOddsBot(), 0.80).act(view(facing_flop_bet()), hole("AsKs")) == Raise(2500)


def test_pot_odds_bot_value_bets_and_checks_back():
    spot = view(checked_to_raiser())
    assert with_equity(PotOddsBot(), 0.65).act(spot, hole("AsKs")) == Bet(667)
    assert with_equity(PotOddsBot(), 0.55).act(spot, hole("AsKs")) == Check()


def test_pot_odds_bot_never_folds_when_it_can_check():
    assert with_equity(PotOddsBot(), 0.0).act(view(checked_to_raiser()), hole("AsKs")) == Check()


def test_pot_odds_bot_real_equity_on_the_river():
    # Board Ah Kh Qh Jh 2c: Th9c makes a royal flush (equity exactly 1); 4c3d has only
    # ace-high, far below the 1/3 pot odds of calling a pot-sized bet.
    s = new_hand(button=0)
    s = apply_action(s, 0, Call())
    s = apply_action(s, 1, Check())
    s = deal_board(s, parse_cards("AhKhQh"))
    s = apply_action(s, 1, Check())
    s = apply_action(s, 0, Check())
    s = deal_board(s, parse_cards("Jh"))
    s = apply_action(s, 1, Check())
    s = apply_action(s, 0, Check())
    s = deal_board(s, parse_cards("2c"))
    bot = PotOddsBot(debug=True)
    assert bot.act(view(s), hole("Th9c")) == Bet(133)  # 2/3 of a 200 pot
    assert bot.last_decision.equity == 1.0 and bot.last_decision.std_error == 0.0
    facing = apply_action(s, 1, Bet(200))
    assert bot.act(view(facing), hole("4c3d")) == Fold()


def test_pot_odds_bot_preflop_uses_the_table():
    bot = PotOddsBot(debug=True)
    bot.act(view(new_hand(button=0)), hole("7h2c"))
    assert bot.last_decision.equity == PREFLOP_EQUITY["72o"]
    # SB facing the big blind: call 50 into 150, pot odds 50 / 200 = 25% < 34%: it calls.
    assert bot.last_decision.pot_odds == 0.25 and bot.last_decision.action == Call()


def test_pot_odds_bot_debug_output():
    bot = with_equity(PotOddsBot(debug=True), 0.40)
    action = bot.act(view(facing_flop_bet()), hole("AsKs"))
    d = bot.last_decision
    assert d.action == action
    assert (d.equity, d.pot, d.to_call, d.pot_odds) == (0.40, 1500, 500, 0.25)
    assert d.ev("fold") == 0.0
    assert d.ev("call") == pytest.approx(0.40 * 2000 - 500)
    assert "raise to 2500" in dict(d.evs)
    text = str(d)
    assert "equity 40.0%" in text and "pot odds 25.0%" in text and "EV(call)" in text


def test_pot_odds_bot_reuses_its_equity_within_a_street():
    bot = PotOddsBot(seed=1, debug=True)
    s = flop_pot_1000()
    bot.act(view(s), hole("Ah9c"))  # BB checks or bets on the flop
    first = bot.last_decision.equity
    s = apply_action(s, 1, Check())
    s = apply_action(s, 0, Bet(500))
    bot.act(view(s), hole("Ah9c"))  # facing a bet on the same flop: no new simulation
    assert bot.last_decision.equity == first and len(bot._cache) == 1
    bot.reseed(2)
    assert not bot._cache


def test_debug_off_records_nothing():
    bot = PotOddsBot()
    bot.act(view(facing_flop_bet()), hole("AsKs"))
    assert bot.last_decision is None


# --- EquityBot decisions ---------------------------------------------------------------


def with_range_equities(bot, table: dict[float, float]):
    """Replace the simulation with fixed equities per range width; record what was asked."""
    bot.asked = []

    def fake(state, hole_cards, widths):
        bot.asked.append(list(widths))
        return {w: (table[w], 0.0) for w in widths}

    bot.range_equities = fake
    return bot


def test_equity_bot_ev_arithmetic_when_checked_to():
    # Pot 1,000, nothing to call, opponent has not bet or raised: R = any hand.
    # Sizes: 1/2 pot = 500, pot = 1,000, all-in = 9,500.
    # Calling ranges (w * MDF, rounded down to the grid):
    #   500:   2/3          -> 60%
    #   1,000: 1/2          -> 50%
    #   9,500: 1,000/10,500 -> 9.5% -> 7.5%
    # Fold probability: x / (P + x) = 1/3, 1/2, 0.905, all capped at max_fold = 0.1.
    eq = {1.0: 0.70, 0.6: 0.60, 0.5: 0.55, 0.075: 0.40}
    bot = with_range_equities(EquityBot(debug=True), eq)
    action = bot.act(view(checked_to_raiser()), hole("AsKs"))
    d = bot.last_decision
    assert bot.asked == [[1.0, 0.6, 0.5, 0.075]]
    assert d.ev("check") == pytest.approx(700)  # 0.70 * 1,000
    assert d.ev("bet 500") == pytest.approx(0.1 * 1000 + 0.9 * (0.60 * 2000 - 500))  # 730
    assert d.ev("bet 1000") == pytest.approx(0.1 * 1000 + 0.9 * (0.55 * 3000 - 1000))  # 685
    assert d.ev("bet 9500") == pytest.approx(0.1 * 1000 + 0.9 * (0.40 * 20000 - 9500))  # -1,250
    assert action == Bet(500) == d.action


def test_equity_bot_ev_arithmetic_facing_a_bet():
    # Facing 500 into P = 1,500. The bet narrows R to the top 50%.
    # Raise sizes: 1/2 pot = 500 + 0.5 * 2,000 = 1,500; pot = 2,500; all-in = 9,500.
    # x = 1,500 / 2,500 / 9,500 (we have nothing in yet), r = 1,000 / 2,000 / 9,000.
    # Calling share P / (P + x) = 0.5 / 0.375 / 0.136 -> 0.5 * that = 25% / 18.75% / 6.8%
    #   -> grid 25% / 15% / 5%.
    eq = {0.5: 0.30, 0.25: 0.20, 0.15: 0.15, 0.05: 0.10}
    bot = with_range_equities(EquityBot(debug=True), eq)
    action = bot.act(view(facing_flop_bet()), hole("AsKs"))
    d = bot.last_decision
    assert d.villain_range == 0.5
    assert bot.asked == [[0.5, 0.25, 0.15, 0.05]]
    assert d.ev("fold") == 0.0
    assert d.ev("call") == pytest.approx(0.30 * 2000 - 500)  # 100
    assert d.ev("raise to 1500") == pytest.approx(0.1 * 1500 + 0.9 * (0.20 * 4000 - 1500))  # -480
    assert action == Call()


def test_equity_bot_folds_without_odds():
    bot = with_range_equities(EquityBot(), {0.5: 0.10, 0.25: 0.05, 0.15: 0.05, 0.05: 0.05})
    assert bot.act(view(facing_flop_bet()), hole("AsKs")) == Fold()


def test_range_narrows_after_raises():
    bot = EquityBot(debug=True)
    s = new_hand(button=0)
    assert bot.villain_range_width(public_view(s, 1)) == 1.0  # blinds are not raises
    s = apply_action(s, 0, Raise(300))  # SB raises: BB assumes top 50%
    assert bot.villain_range_width(view(s)) == 0.5
    bot.act(view(s), hole("9d8c"))
    d = bot.last_decision
    assert d.villain_range == 0.5
    assert d.equity < PREFLOP_EQUITY["98o"]  # a stronger range: 98o does worse
    s = apply_action(s, 1, Raise(900))
    s = apply_action(s, 0, Raise(2700))  # second raise by the SB: top 30%
    assert bot.villain_range_width(view(s)) == 0.3
    assert bot.villain_range_width(public_view(s, 0)) == 0.5  # SB's view of the BB: one raise


def test_equity_bot_real_spot_is_reproducible_and_debuggable():
    spot = view(checked_to_raiser())
    a, b = EquityBot(seed=1, debug=True), EquityBot(seed=1, debug=True)
    assert a.act(spot, hole("Ah9c")) == b.act(spot, hole("Ah9c"))
    assert a.last_decision == b.last_decision
    d = a.last_decision
    assert 0 < d.std_error < 0.03  # 400 samples: SE at most 2.5 points
    assert {name for name, _ in d.evs} == {"check", "bet 500", "bet 1000", "bet 9500"}
    assert len(d.notes) == 3 and "equity when called" in d.notes[0]
    a.reseed(5)
    first = a.act(spot, hole("Ah9c")), a.last_decision
    a.reseed(5)
    assert (a.act(spot, hole("Ah9c")), a.last_decision) == first


# --- Real hands ------------------------------------------------------------------------

OPPONENTS = {
    "RandomBot": lambda: RandomBot(1),
    "CallStationBot": CallStationBot,
    "AlwaysRaiseBot": AlwaysRaiseBot,
    "PotOddsBot": lambda: PotOddsBot(2, samples=30),
}
FAST = {
    "PotOddsBot": lambda: PotOddsBot(3, samples=30),
    "EquityBot": lambda: EquityBot(4, samples=30),
}


def play_and_check(make_bot, make_opponent, n_hands: int, seed: int):
    logs = play_match([make_bot(), make_opponent()], n_hands=n_hands, master_seed=seed)
    for log in logs:  # an illegal action would have raised inside play_match
        assert sum(log.result.net) == 0
        assert sum(log.result.final_stacks) == sum(log.starting_stacks)
        assert min(log.result.final_stacks) >= 0
    return logs


@pytest.mark.parametrize("bot", FAST, ids=str)
@pytest.mark.parametrize("opponent", OPPONENTS, ids=str)
def test_legal_and_chips_conserved_vs_every_opponent(bot, opponent):
    # 30 Monte Carlo samples and 200 hands keep this fast (the code paths are the same);
    # the slow test below plays 3,000 hands per pairing with the default settings.
    play_and_check(FAST[bot], OPPONENTS[opponent], n_hands=200, seed=21)


@pytest.mark.parametrize("bot", FAST, ids=str)
def test_real_hands_are_deterministic(bot):
    first = play_match([FAST[bot](), RandomBot(9)], n_hands=200, master_seed=8)
    second = play_match([FAST[bot](), RandomBot(9)], n_hands=200, master_seed=8)
    assert first == second


@pytest.mark.slow
@pytest.mark.parametrize("bot", [PotOddsBot, EquityBot], ids=lambda b: b.__name__)
@pytest.mark.parametrize("opponent", OPPONENTS, ids=str)
def test_legal_and_chips_conserved_default_settings(bot, opponent):
    play_and_check(lambda: bot(5), OPPONENTS[opponent], n_hands=3_000, seed=22)
