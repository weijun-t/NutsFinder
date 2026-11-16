"""The effective stack: with unequal stacks, chips only change hands up to the shorter stack.

See "The effective stack" in :mod:`nutsfinder.bots.common` for the maths:

- a bet/raise is never sized above ``eff_to = min(max_to, current_bet +
  opponent's stack)``, and its EV counts only the chips really at risk;
- a call that is all-in for less than the bet uses the effective pot
  ``P_eff = P - excess`` (the opponent's unmatched excess is refunded).

Part 1: the two worked examples from the docstrings, built through the real
engine with the equity stubbed. Part 2: real hands with random unequal
stacks: every action legal, chips conserved, and no bet ever bigger than
the opponent can call.
"""

from __future__ import annotations

import random

import pytest

from nutsfinder.bots import AlwaysRaiseBot, CallStationBot, EquityBot, PotOddsBot, RandomBot
from nutsfinder.bots.common import chips_added, effective_max_to, effective_pot, pot_odds
from nutsfinder.engine.actions import Bet, Call, Check, Fold, Raise
from nutsfinder.engine.betting import HandState, PublicState, apply_action, deal_board, new_hand, public_view
from nutsfinder.engine.bot import Bot
from nutsfinder.engine.cards import Card, parse_cards
from nutsfinder.engine.runner import play_hand

FLOP = parse_cards("Kh7d2c")


def hole(s: str) -> tuple[Card, ...]:
    return tuple(parse_cards(s))


def view(state: HandState) -> PublicState:
    assert state.to_act is not None
    return public_view(state, state.to_act)


def limp_check_flop(stacks: tuple[int, int]) -> HandState:
    """Seat 0 (button) limps, seat 1 checks: pot 200 on the flop, seat 1 to act."""
    s = new_hand(button=0, stacks=stacks)
    s = apply_action(s, 0, Call())
    s = apply_action(s, 1, Check())
    return deal_board(s, FLOP)


def example_a() -> HandState:
    """Stacks (10,000, 600): limp/check, BB checks the flop, SB bets 3,000. BB to act, 500 behind."""
    s = apply_action(limp_check_flop((10_000, 600)), 1, Check())
    return apply_action(s, 0, Bet(3000))


def example_b() -> HandState:
    """Stacks (10,000, 1,000): limp/check, BB checks the flop. SB (deep) to act, pot 200."""
    return apply_action(limp_check_flop((10_000, 1_000)), 1, Check())


def short_stack_facing_shove() -> HandState:
    """Preflop, stacks (10,000, 600): the SB shoves to 10,000, BB (100 posted, 500 behind) to act."""
    return apply_action(new_hand(button=0, stacks=(10_000, 600)), 0, Raise(10_000))


def stub_pot_odds(equity: float) -> PotOddsBot:
    bot = PotOddsBot(debug=True)
    bot.estimate_equity = lambda state, hole_cards: (equity, 0.0)
    return bot


def stub_equity_bot(equity: float) -> EquityBot:
    bot = EquityBot(debug=True)
    bot.range_equities = lambda state, hole_cards, widths: {w: (equity, 0.0) for w in widths}
    return bot


# --- The helpers -------------------------------------------------------------------


def test_effective_pot_example_a():
    v = view(example_a())
    assert (v.pot, v.legal.call_amount) == (3200, 500)
    # excess = 3,000 - (0 + 500) = 2,500 is refunded whatever happens.
    assert effective_pot(v) == 700
    assert pot_odds(500, effective_pot(v)) == pytest.approx(500 / 1200)  # 41.7%, not 13.5%


def test_effective_max_to_example_b():
    v = view(example_b())
    assert v.legal.max_to == 9900  # the engine does not cap by the opponent's stack
    assert effective_max_to(v) == 900  # 0 + the short stack's 900
    assert chips_added(v, 9900) == 900  # only 900 of a 9,900 shove is at risk


def test_min_raise_the_opponent_cannot_cover():
    # Stacks (10,000, 250): SB raises to 200, BB calls (50 behind), flop, BB checks.
    # A minimum bet is one big blind (100), more than the 50 the BB can call.
    s = new_hand(button=0, stacks=(10_000, 250))
    s = apply_action(s, 0, Raise(200))
    s = apply_action(s, 1, Call())
    s = deal_board(s, FLOP)
    s = apply_action(s, 1, Check())
    v = view(s)
    assert v.stacks[1] == 50 and v.legal.min_to == 100  # a minimum bet is one big blind
    # eff_to = 0 + 50 < min_to: the only sensible size is the minimum bet; 50 of it is at risk.
    assert effective_max_to(v) == 100
    assert chips_added(v, 100) == 50


@pytest.mark.parametrize("stacks", [(10_000, 10_000), (5_000, 5_000)])
def test_equal_stacks_change_nothing(stacks):
    # With equal stacks the caps never bind: this is why Phase 4 results are unchanged.
    s = apply_action(new_hand(button=0, stacks=stacks), 0, Raise(300))
    for state in (s, apply_action(s, 1, Raise(stacks[0]))):
        v = view(state)
        assert effective_pot(v) == v.pot
        if v.legal.can_raise:
            assert effective_max_to(v) == v.legal.max_to


# --- Worked example A: calling all-in for less -----------------------------------------


def test_example_a_pot_odds_bot_folds():
    bot = stub_pot_odds(0.25)
    assert bot.act(view(example_a()), hole("Qs9s")) == Fold()
    d = bot.last_decision
    assert d.pot_odds == pytest.approx(500 / 1200)
    assert d.ev("call") == pytest.approx(0.25 * 1200 - 500)  # -200
    assert any("effective pot 700" in note for note in d.notes)


def test_example_a_equity_bot_folds():
    bot = stub_equity_bot(0.25)
    assert bot.act(view(example_a()), hole("Qs9s")) == Fold()
    assert bot.last_decision.ev("call") == pytest.approx(-200)


def test_example_a_still_calls_with_enough_equity():
    # Break-even is 500 / 1,200 = 41.7%.
    assert stub_pot_odds(0.40).act(view(example_a()), hole("Qs9s")) == Fold()
    assert stub_pot_odds(0.45).act(view(example_a()), hole("Qs9s")) == Call()


# --- Short stack facing a shove ------------------------------------------------------


def test_short_stack_facing_shove():
    v = view(short_stack_facing_shove())
    # P = 10,000 + 100 = 10,100, c = 500, excess = 10,000 - 600 = 9,400, P_eff = 700:
    # the pot that can really be won is 600 + 600 = 1,200 = P_eff + c.
    assert (v.pot, v.legal.call_amount, effective_pot(v)) == (10_100, 500, 700)
    assert stub_pot_odds(0.40).act(v, hole("7c2d")) == Fold()  # old code: 500 / 10,600 = 4.7%, call
    assert stub_pot_odds(0.45).act(v, hole("7c2d")) == Call()
    bot = stub_equity_bot(0.40)
    assert bot.act(v, hole("7c2d")) == Fold()
    assert bot.last_decision.pot_odds == pytest.approx(500 / 1200)
    # Real preflop equity of 72o vs a random hand is about 35%: fold.
    assert PotOddsBot().act(v, hole("7c2d")) == Fold()


# --- Worked example B: betting into a short stack --------------------------------------


def test_example_b_equity_bot_shoves_for_the_effective_stack():
    bot = stub_equity_bot(0.8)
    action = bot.act(view(example_b()), hole("AsKs"))
    d = bot.last_decision
    # Sizes 1/2 pot, pot, all-in capped at 900. f = min(0.1, x / (200 + x)) = 0.1 for all.
    assert {name for name, _ in d.evs} == {"check", "bet 100", "bet 200", "bet 900"}
    assert d.ev("check") == pytest.approx(160)
    assert d.ev("bet 100") == pytest.approx(0.1 * 200 + 0.9 * (0.8 * 400 - 100))  # 218
    assert d.ev("bet 200") == pytest.approx(0.1 * 200 + 0.9 * (0.8 * 600 - 200))  # 272
    assert d.ev("bet 900") == pytest.approx(0.1 * 200 + 0.9 * (0.8 * 2000 - 900))  # +650 (old: -970)
    assert action == Bet(900) == d.action
    assert max(d.evs, key=lambda kv: kv[1])[0] == "bet 900"


def test_example_b_pot_odds_bot_sizes_are_capped():
    v = view(example_b())
    bot = stub_pot_odds(0.9)
    bot.raise_fraction = bot.bet_fraction = 20.0  # a huge "pot fraction" must still stop at 900
    assert bot.act(v, hole("AsKs")) == Bet(900)


# --- Real hands with random unequal stacks ---------------------------------------------


class Checked(Bot):
    """Wraps a bot and checks every bet/raise against the effective stack."""

    def __init__(self, inner: Bot) -> None:
        self.inner = inner
        self.aggressive = 0

    def act(self, state, hole_cards):
        action = self.inner.act(state, hole_cards)
        legal = state.legal
        if isinstance(action, (Bet, Raise)):
            self.aggressive += 1
            to = action.amount if isinstance(action, Bet) else action.to
            eff_to = min(legal.max_to, state.current_bet + state.stacks[1 - state.seat])
            assert to <= max(eff_to, legal.min_to), (state, action)
        if state.starting_stacks[0] == state.starting_stacks[1]:
            assert effective_pot(state) == state.pot
        return action

    def reseed(self, seed: int) -> None:
        self.inner.reseed(seed)

    def end_hand(self, summary) -> None:
        self.inner.end_hand(summary)

    def __repr__(self) -> str:
        return f"Checked({self.inner!r})"


def random_stacks(rng: random.Random) -> tuple[int, int]:
    """Mostly deep vs short, sometimes tiny (below the blinds), sometimes equal."""
    kind = rng.random()
    if kind < 0.1:
        n = rng.randint(1, 20_000)
        return n, n
    short = rng.randint(1, 150) if kind < 0.25 else rng.randint(100, 5_000)
    deep = rng.randint(short, 20_000)
    return (deep, short) if rng.random() < 0.5 else (short, deep)


def play_unequal(make_bot, make_opponent, n_hands: int, seed: int) -> int:
    rng = random.Random(seed)
    bot = Checked(make_bot())
    bots = [bot, make_opponent()]
    for i in range(n_hands):
        stacks = random_stacks(rng)
        order = bots if i % 2 == 0 else bots[::-1]
        log = play_hand(order, rng.getrandbits(64), button=rng.randrange(2), stacks=stacks)
        # play_hand raises ValueError on an illegal action.
        assert sum(log.result.net) == 0
        assert sum(log.result.final_stacks) == sum(stacks)
        assert min(log.result.final_stacks) >= 0
    return bot.aggressive


BOTS = {
    "PotOddsBot": lambda: PotOddsBot(3, samples=30),
    "EquityBot": lambda: EquityBot(4, samples=30),
}
OPPONENTS = {
    "RandomBot": lambda: RandomBot(1),
    "CallStationBot": CallStationBot,
    "AlwaysRaiseBot": AlwaysRaiseBot,
    "PotOddsBot": lambda: PotOddsBot(2, samples=30),
    "EquityBot": lambda: EquityBot(5, samples=30),
}


@pytest.mark.parametrize("bot", BOTS, ids=str)
@pytest.mark.parametrize("opponent", OPPONENTS, ids=str)
def test_unequal_stacks_legal_and_conserved(bot, opponent):
    assert play_unequal(BOTS[bot], OPPONENTS[opponent], n_hands=150, seed=31) > 0


@pytest.mark.slow
@pytest.mark.parametrize("bot", [PotOddsBot, EquityBot], ids=lambda b: b.__name__)
@pytest.mark.parametrize("opponent", OPPONENTS, ids=str)
def test_unequal_stacks_legal_and_conserved_default_settings(bot, opponent):
    assert play_unequal(lambda: bot(6), OPPONENTS[opponent], n_hands=3_000, seed=32) > 0
