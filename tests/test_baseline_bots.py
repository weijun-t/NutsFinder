"""Tests for the deterministic baselines: CallStationBot and AlwaysRaiseBot.

Part 1 checks single decisions in hand-built situations (including the
pot-sized arithmetic from the AlwaysRaiseBot docstring). Part 2 plays a few
thousand hands of the baselines against RandomBot and each other, and checks
that every action was legal, that no chips appeared or vanished, and that a
rerun gives identical hands.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from nutsfinder.bots import AlwaysRaiseBot, CallStationBot, RandomBot
from nutsfinder.bots.always_raise import pot_sized_to
from nutsfinder.engine.actions import Action, Bet, Call, Check, Raise
from nutsfinder.engine.betting import (
    STARTING_STACK,
    HandState,
    PublicState,
    apply_action,
    deal_board,
    new_hand,
    public_view,
)
from nutsfinder.engine.bot import Bot, HandSummary
from nutsfinder.engine.cards import Card, parse_cards
from nutsfinder.engine.runner import play_match

HOLE: tuple[Card, ...] = tuple(parse_cards("AsKs"))
FLOP = parse_cards("2c7d9h")


# --- Hand-built situations ---------------------------------------------------


def view(state: HandState) -> PublicState:
    """The public view for whoever is to act."""
    assert state.to_act is not None
    return public_view(state, state.to_act)


def preflop_sb(stacks: tuple[int, int] = (STARTING_STACK, STARTING_STACK)) -> HandState:
    """Fresh hand, seat 0 on the button (small blind) to act."""
    return new_hand(button=0, stacks=stacks)


def bb_after_limp() -> HandState:
    """SB limps (calls 50); the big blind (seat 1) has the option."""
    return apply_action(preflop_sb(), 0, Call())


def flop_pot_1000() -> HandState:
    """SB raises to 500, BB calls: pot 1,000 on the flop, BB (seat 1) first."""
    s = apply_action(preflop_sb(), 0, Raise(500))
    s = apply_action(s, 1, Call())
    return deal_board(s, FLOP)


def facing_500_into_1000() -> HandState:
    """On the 1,000 flop, BB bets 500; the SB (seat 0) faces it."""
    return apply_action(flop_pot_1000(), 1, Bet(500))


def facing_all_in() -> HandState:
    """SB shoves preflop; the BB can call or fold but not raise."""
    return apply_action(preflop_sb(), 0, Raise(STARTING_STACK))


def facing_short_all_in_blind() -> HandState:
    """The BB has only 75 chips and posts them all; the SB faces 25 more."""
    return preflop_sb(stacks=(STARTING_STACK, 75))


# --- Pot-sized arithmetic (the docstring examples) ---------------------------


def test_pot_sized_preflop_sb_raise_is_300():
    v = view(preflop_sb())
    assert (v.pot, v.current_bet, v.legal.call_amount) == (150, 100, 50)
    assert pot_sized_to(v) == 100 + (150 + 50) == 300


def test_pot_sized_bet_into_1000_is_1000():
    v = view(flop_pot_1000())
    assert (v.pot, v.current_bet, v.legal.call_amount) == (1000, 0, 0)
    assert pot_sized_to(v) == 1000


def test_pot_sized_raise_facing_500_into_1000_is_2500():
    v = view(facing_500_into_1000())
    # The pot already contains the 500 bet.
    assert (v.pot, v.current_bet, v.legal.call_amount) == (1500, 500, 500)
    assert pot_sized_to(v) == 500 + (1500 + 500) == 2500


def test_pot_sized_bb_option_after_limp_is_300():
    v = view(bb_after_limp())
    # Nothing to call: raise by the 200 pot, on top of the 100 bet.
    assert (v.pot, v.current_bet, v.legal.call_amount) == (200, 100, 0)
    assert pot_sized_to(v) == 300


# --- AlwaysRaiseBot decisions ------------------------------------------------


@pytest.mark.parametrize(
    ("make_state", "expected"),
    [
        (preflop_sb, Raise(300)),
        (bb_after_limp, Raise(300)),
        (flop_pot_1000, Bet(1000)),
        (facing_500_into_1000, Raise(2500)),
        (facing_all_in, Call()),
        (facing_short_all_in_blind, Call()),
    ],
)
def test_always_raise_decisions(make_state, expected):
    v = view(make_state())
    action = AlwaysRaiseBot().act(v, HOLE)
    assert action == expected
    assert v.legal.is_legal(action)


def test_always_raise_postflop_first_to_act_after_limped_pot():
    s = apply_action(bb_after_limp(), 1, Check())
    v = view(deal_board(s, FLOP))
    assert v.seat == 1  # the big blind acts first after the flop
    assert AlwaysRaiseBot().act(v, HOLE) == Bet(200)


def test_always_raise_short_stack_preflop_goes_all_in():
    # SB has only 250 chips: the pot-sized raise to 300 is capped at all-in.
    v = view(preflop_sb(stacks=(250, STARTING_STACK)))
    assert pot_sized_to(v) == 300
    assert v.legal.max_to == 250
    assert AlwaysRaiseBot().act(v, HOLE) == Raise(250)


def test_always_raise_short_stack_postflop_bet_capped():
    # Both put in 500 preflop from 800-chip stacks: 300 left, pot 1,000.
    s = apply_action(preflop_sb(stacks=(800, 800)), 0, Raise(500))
    s = apply_action(s, 1, Call())
    v = view(deal_board(s, FLOP))
    assert pot_sized_to(v) == 1000
    assert AlwaysRaiseBot().act(v, HOLE) == Bet(300)


def test_always_raise_floors_at_min_to():
    # Not reachable in real play (the pot is always at least the last raise
    # increment), but the clamp must hold if min_to were above pot size.
    v = view(flop_pot_1000())
    fake = replace(v, legal=replace(v.legal, min_to=5000))
    assert AlwaysRaiseBot().act(fake, HOLE) == Bet(5000)


def test_always_raise_checks_when_betting_is_not_allowed():
    # Hand-edited legal actions: nothing to call and no bet allowed.
    v = view(flop_pot_1000())
    fake = replace(v, legal=replace(v.legal, can_bet=False, min_to=0, max_to=0))
    assert AlwaysRaiseBot().act(fake, HOLE) == Check()


# --- CallStationBot decisions ------------------------------------------------


@pytest.mark.parametrize(
    ("make_state", "expected"),
    [
        (preflop_sb, Call()),
        (bb_after_limp, Check()),
        (flop_pot_1000, Check()),
        (facing_500_into_1000, Call()),
        (facing_all_in, Call()),
        (facing_short_all_in_blind, Call()),
    ],
)
def test_call_station_decisions(make_state, expected):
    v = view(make_state())
    action = CallStationBot().act(v, HOLE)
    assert action == expected
    assert v.legal.is_legal(action)


@pytest.mark.parametrize("bot", [CallStationBot(), AlwaysRaiseBot()])
def test_refuses_to_act_out_of_turn(bot):
    with pytest.raises(ValueError):
        bot.act(public_view(preflop_sb(), 1), HOLE)  # seat 0 is to act


# --- Matches: legality, chip conservation, determinism -----------------------


class LegalityChecker(Bot):
    """Wraps a bot and asserts every action it returns is legal."""

    def __init__(self, bot: Bot) -> None:
        self.bot = bot
        self.actions_seen = 0

    def act(self, state: PublicState, hole_cards: tuple[Card, ...]) -> Action:
        action = self.bot.act(state, hole_cards)
        assert state.legal is not None and state.legal.is_legal(action), (
            f"{self.bot!r} chose illegal {action} with {state.legal}"
        )
        self.actions_seen += 1
        return action

    def end_hand(self, summary: HandSummary) -> None:
        self.bot.end_hand(summary)


def make_bot(name: str, seed: int) -> Bot:
    return {
        "random": lambda: RandomBot(seed),
        "call": CallStationBot,
        "raise": AlwaysRaiseBot,
    }[name]()


# (bot in seat 0, bot in seat 1, master seed). The button alternates, so
# each bot plays both positions.
PAIRINGS = [
    ("call", "random", 101),
    ("raise", "random", 102),
    ("call", "raise", 103),
    ("call", "call", 104),
    ("raise", "raise", 105),
]


@pytest.mark.parametrize(("a", "b", "seed"), PAIRINGS)
def test_matches_are_legal_and_conserve_chips(a, b, seed):
    n_hands = 1000
    bots = [LegalityChecker(make_bot(a, 1)), LegalityChecker(make_bot(b, 2))]
    logs = play_match(bots, n_hands=n_hands, master_seed=seed)
    assert len(logs) == n_hands
    for log in logs:
        assert sum(log.result.net) == 0
        assert sum(log.result.final_stacks) == sum(log.starting_stacks)
        assert min(log.result.final_stacks) >= 0
    assert bots[0].actions_seen > 0 and bots[1].actions_seen > 0


def test_call_station_only_checks_or_calls_in_play():
    logs = play_match([CallStationBot(), RandomBot(7)], n_hands=500, master_seed=3)
    for log in logs:
        for rec in log.actions:
            if rec.player == 0:
                assert isinstance(rec.action, (Check, Call))


def test_always_raise_never_folds_in_play():
    logs = play_match([AlwaysRaiseBot(), RandomBot(7)], n_hands=500, master_seed=4)
    assert all(log.folded != 0 for log in logs)


@pytest.mark.parametrize(("a", "b", "seed"), PAIRINGS)
def test_matches_are_deterministic(a, b, seed):
    first = play_match([make_bot(a, 5), make_bot(b, 6)], n_hands=200, master_seed=seed)
    second = play_match([make_bot(a, 5), make_bot(b, 6)], n_hands=200, master_seed=seed)
    assert first == second
